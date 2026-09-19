#!/usr/bin/env python3
"""Phase 6 Round 3 -- EXP-9B: fine-tune a TwinLiteNetPlus lane arm (B0/B1/B2).

Thin adapter around models/round3/tlp_variants.Round3TLP so the SECOND
architecture trains under ONE protocol for all three arms.  The ONLY thing that
differs between arms is the lane-path intervention:

    B0 baseline  stock head
    B1 spatial   + learned 1/4 tap into the lane head's 1/4 input  (zero-init)
    B2 channel   + 1x1 bottleneck widening at the 1/4 stage        (zero-init, FLOPs-matched)

Protocol (IDENTICAL for every arm -- changing any line below is a pre-registration
amendment, not an edit):
    dataset    datasets.bdd100k split=tri_train, letterbox 640, hflip aug
    init       weights/TwinLiteNetPlus_<preset>.pth  (SAME for all arms)
    loss       losses.multitask_loss.seg_ce_loss on DA and lane, fg_weight=10
               -- the same CE ruler used by R2 and by EXP-9A, so the two
               second-architecture testbeds are scored by the same loss form
    optimizer  AdamW lr=1e-4 wd=5e-4, grad-clip 10.0
    schedule   CosineAnnealingLR over epochs*steps
    seed       fixed per arm
    det head   absent (TLP has none); MultiTaskLoss is called with det_preds=None

WHY lr IS 1e-4 AND NOT 1e-3 (registered choice, fixed before any arm ran):
The arms start from the RELEASED checkpoint.  A high LR on a 2-epoch budget would
move the pretrained backbone far enough that all three arms converge to "equally
damaged", which would destroy the contrast we are trying to measure.  1e-4 lets
the new zero-init paths learn while the released feature extractor stays near its
released optimum.  This is a *design* choice made before measurement, recorded in
phase6_round3_statistics.csv alongside the budget calibration.

Usage
    python scripts/phase6_round3_tlp_train.py --arm spatial --preset small \
        --outdir experiments/phase6/round3_tlp_B1 --epochs 2 --seed 0
    # micro-probe (verify the chain + measure ms/step, no real training):
    ... --arm baseline --outdir experiments/phase6/round3_tlp_probe --max-steps 60
"""
import argparse
import json
import os
import random as _rnd
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "baselines", "TwinLiteNetPlus"))

# NB: the minimal `collate`, not `collate_train` -- collate_train indexes
# out["det_targets"] unconditionally, and TLP arms run with_det=False.
from datasets.bdd100k import BDD100KDataset, collate            # noqa: E402
from losses.multitask_loss import MultiTaskLoss                 # noqa: E402
from phase2_load_baseline_weights import _strip_module, WEIGHTS  # noqa: E402
from models.round3.tlp_variants import Round3TLP, ARMS, channel_plan, solve_dc  # noqa: E402
from evaluation.evaluate_baseline import normalize_batch        # noqa: E402


def load_released(model, preset):
    """Load the released TLP checkpoint into the wrapped net (new paths excluded)."""
    sd = torch.load(os.path.join(WEIGHTS, "TwinLiteNetPlus_%s.pth" % preset),
                    map_location="cpu", weights_only=False)
    sd = _strip_module(sd.get("state_dict", sd))
    n, missing = model.load_pretrained(sd)
    return n, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--preset", default="small")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--num-images", type=int, default=0, help="0 = full tri_train")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epoch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=100,
                    help="rolling checkpoint interval in steps (0 disables)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <outdir>/checkpoint.pt + progress.json if present")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "training_log.txt")

    # Resume guard: a finished arm must not be re-run -- it would loop zero epochs
    # and clobber a good metrics.json with empty loss values.
    if args.resume:
        _p = os.path.join(args.outdir, "progress.json")
        _m = os.path.join(args.outdir, "metrics.json")
        if os.path.exists(_p) and os.path.exists(_m):
            try:
                if json.load(open(_p)).get("done"):
                    print("[r3-tlp] %s already complete - nothing to do" % args.outdir, flush=True)
                    return
            except (ValueError, KeyError):
                pass
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _rnd.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[r3-tlp] torch %s cuda=%s device=%s" % (torch.__version__, torch.cuda.is_available(), device), flush=True)
    print("[r3-tlp] GPU %s" % (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"), flush=True)

    C14, C0 = channel_plan(args.preset)
    model = Round3TLP(args.preset, args.arm).to(device)
    n, missing = load_released(model, args.preset)
    assert not missing, "unexpected missing keys vs released ckpt: %s" % missing[:8]
    print("[r3-tlp] arm=%s preset=%s C14=%d C0=%d dc=%d loaded=%d extra_params=%d" % (
        args.arm, args.preset, C14, C0, model.dc, n, model.extra_params()), flush=True)

    # --- step-0 equivalence, MEASURED (not asserted in prose) -------------------
    # All three arms must be bit-identical to the released baseline before the
    # first optimiser step, otherwise the "single variable" claim is false.
    #
    # NB: the reference MUST live on the same device as the arm.  Comparing a CPU
    # reference against a CUDA arm yields a ~5e-2 delta from cuDNN/CPU float32
    # ordering alone -- identical for every arm, which is exactly the fingerprint
    # of a device artefact rather than an intervention.  (First version of this
    # block made that mistake; the gate caught it because the *same* nonzero value
    # appeared for two different interventions.)
    step0 = None
    if args.arm != "baseline":
        torch.manual_seed(0)
        ref = Round3TLP(args.preset, "baseline").to(device)
        load_released(ref, args.preset)
        ref.eval()
        model.eval()
        x = torch.randn(1, 3, 640, 640).to(device)   # ONE input, one device
        with torch.no_grad():
            a = [t.float().cpu() for t in ref(x)]
            b = [t.float().cpu() for t in model(x)]
        step0 = max((b[i] - a[i]).abs().max().item() for i in (0, 1))
        model.train()
        del ref
        print("[r3-tlp] step-0 max|delta| vs released baseline = %.3e" % step0, flush=True)
        assert step0 <= 1e-5, "arm is NOT a no-op at step 0 (%g)" % step0

    loss_fn = MultiTaskLoss(torch.zeros(3, 3, 2), nc=1, lambda_det=0.0,
                            lambda_da=1.0, lambda_lane=1.0, img_size=640).to(device)

    ds = BDD100KDataset(os.path.join(ROOT, "data", "bdd100k"), split="tri_train",
                        train=True, with_det=False, with_da=True, with_lane=True)
    if args.num_images:
        ds.names = ds.names[:args.num_images]
    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate,
                        drop_last=True, generator=g, pin_memory=(device.type == "cuda"),
                        prefetch_factor=(4 if args.num_workers > 0 else None))
    print("[r3-tlp] train images=%d batch=%d steps/epoch=%d" % (len(ds), args.batch_size, len(loader)), flush=True)

    def make_loader(ep):
        """Deterministic per-epoch shuffle (see the same helper in the YOLOP
        adapter): an epoch's order is reproducible from (seed, ep) alone, so a
        resume replays exactly the order the interrupted run used."""
        gg = torch.Generator(); gg.manual_seed(args.seed * 1000 + ep)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate,
                          drop_last=True, generator=gg, pin_memory=(device.type == "cuda"),
                          prefetch_factor=(4 if args.num_workers > 0 else None))

    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * len(loader), 1))

    # ------------- RESUME (atomic writes; see YOLOP adapter for rationale) ------
    ckpt_path = os.path.join(args.outdir, "checkpoint.pt")
    prog_path = os.path.join(args.outdir, "progress.json")
    start_ep, start_it, resumed = 1, 0, False
    if args.resume and os.path.exists(ckpt_path) and os.path.exists(prog_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        try:
            opt.load_state_dict(ck["opt_state"])
            sched.load_state_dict(ck["sched_state"])
        except (ValueError, KeyError) as e:
            print("[r3-tlp] WARN could not restore opt/sched (%s)" % e, flush=True)
        pr = json.load(open(prog_path))
        start_ep, start_it = int(pr["epoch"]), int(pr["step"])
        resumed = True
        print("[r3-tlp] RESUME from %s: epoch=%d step=%d" % (ckpt_path, start_ep, start_it), flush=True)

    def save_ckpt(ep, it, done):
        ck = {"epoch": ep, "model_state": model.state_dict(), "opt_state": opt.state_dict(),
              "sched_state": sched.state_dict(), "seed": args.seed, "arm": args.arm,
              "preset": args.preset, "dc": model.dc,
              "params": sum(p.numel() for p in model.parameters())}
        tmp = ckpt_path + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, ckpt_path)                  # atomic
        tmp_p = prog_path + ".tmp"
        with open(tmp_p, "w") as f:
            json.dump({"epoch": ep, "step": it, "done": done}, f)
        os.replace(tmp_p, prog_path)

    cfg_rec = {"arm": args.arm, "preset": args.preset, "epochs": args.epochs,
               "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
               "seed": args.seed, "num_images": len(ds), "max_steps": args.max_steps,
               "dc": model.dc, "extra_params": model.extra_params(),
               "C14": C14, "C0": C0, "step0_max_delta": step0,
               "loss": "losses.multitask_loss.seg_ce_loss (fg_weight=10) on DA+lane",
               "init": "weights/TwinLiteNetPlus_%s.pth" % args.preset,
               "params": sum(p.numel() for p in model.parameters())}
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(cfg_rec, f, indent=2)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    vram_warned = False
    t_all = time.time()
    last = {}
    for ep in range(1, args.epochs + 1):
        if ep < start_ep:
            print("[r3-tlp] ep %d already complete (resume) - skip" % ep, flush=True)
            continue
        loader = make_loader(ep)
        skip_to = start_it if (resumed and ep == start_ep) else 0
        model.train()
        t0 = time.time()
        run, steps = {}, 0
        for it, batch in enumerate(loader):
            if it < skip_to:                     # replay the deterministic order
                continue
            img = batch["image"].to(device, non_blocking=True)
            # Symmetry with the evaluator (norm="unit" for TwinLiteNetPlus).  Routed
            # through the shared helper so neither family can drift from its own
            # evaluator -- see evaluation/evaluate_baseline.py:normalize_batch.
            img = normalize_batch(img, "unit")
            da_m = batch["da_mask"].to(device)
            lane_m = batch["lane_mask"].to(device)
            opt.zero_grad()
            da, lane = model(img)
            total, losses = loss_fn(None, da, lane, None, da_m, lane_m, None, img_size=640)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            # --- VRAM-cliff guard (pre-registration 11.3) -----------------------
            # Same guard as the YOLOP trainer.  Added 2026-09-11: the flag below
            # was declared on the first pass but the check itself was never
            # applied, so this trainer silently lacked the guard while the run
            # notes claimed both had it.  On Windows/WDDM an over-subscribed
            # allocation pages VRAM to host RAM (~18x slower) instead of raising
            # OOM, so fail loudly.
            if device.type == "cuda":
                frac = (torch.cuda.max_memory_reserved()
                        / torch.cuda.get_device_properties(0).total_memory)
                if frac > 0.96:
                    raise RuntimeError(
                        "[r3-tlp] VRAM-CLIFF ABORT: %.1f%% of device memory reserved. "
                        "WDDM will silently page to host RAM (~18x slower). "
                        "Lower --batch-size (pre-registration 11.3)." % (100.0 * frac))
                if frac > 0.90 and not vram_warned:
                    vram_warned = True
                    print("[r3-tlp] WARN VRAM guard: %.1f%% of device memory reserved - "
                          "inside the measured cliff band" % (100.0 * frac), flush=True)
            for k, v in losses.items():
                run[k] = run.get(k, 0.0) + float(v)
            steps += 1
            if args.ckpt_every and (it + 1) % args.ckpt_every == 0:
                save_ckpt(ep, it + 1, False)
            if (it + 1) % max(args.log_every, 1) == 0 or it == 0:
                avg = {k: v / steps for k, v in run.items()}
                el = time.time() - t0
                done = it + 1 - skip_to
                line = ("[%s] r3_tlp arm=%s ep%d step %d/%d | %.1fm in, ETA %.1fm "
                        "| %.0fms/step | mem %dMiB peak %dMiB "
                        "| loss %.4f da %.4f lane %.4f" % (
                            time.strftime("%H:%M:%S"), args.arm, ep, it + 1, len(loader),
                            el / 60, (len(loader) - it - 1) / max(done / max(el, 1e-9), 1e-9) / 60,
                            el / max(done, 1) * 1e3,
                            torch.cuda.memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0,
                            torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0,
                            avg.get("total", 0), avg.get("da", 0), avg.get("lane", 0)))
                print(line, flush=True)
                with open(log_path, "a") as f:
                    f.write(line + "\n")
            last = {k: v / max(steps, 1) for k, v in run.items()}
            if args.max_steps and steps >= args.max_steps:
                break
        sched.step()
        final = (ep == args.epochs)
        save_ckpt(ep + 1, 0, final)
        line = "[r3-tlp] ep %d DONE avg_loss=%.4f elapsed %.1fm saved checkpoint.pt%s" % (
            ep, last.get("total", 0), (time.time() - t_all) / 60, " [FINAL]" if final else "")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0
    wall_min = (time.time() - t_all) / 60
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"stage": "tlp_finetune", "arm": args.arm, "preset": args.preset,
                   "params": sum(p.numel() for p in model.parameters()),
                   "extra_params": model.extra_params(), "epochs": args.epochs,
                   "num_images": len(ds), "batch_size": args.batch_size, "seed": args.seed,
                   "lr": args.lr, "dc": model.dc,
                   "final_avg_loss": last.get("total", 0), "final_da_loss": last.get("da", 0),
                   "final_lane_loss": last.get("lane", 0), "wall_min": round(wall_min, 2),
                   "peak_gpu_mem_mib": round(peak, 1), "max_steps": args.max_steps,
                   "step0_max_delta": step0}, f, indent=2)
    print("[r3-tlp] done -> %s (wall %.1f min, peak %.0f MiB)" % (args.outdir, wall_min, peak), flush=True)


if __name__ == "__main__":
    main()

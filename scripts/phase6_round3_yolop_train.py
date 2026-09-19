#!/usr/bin/env python3
"""Phase 6 Round 3 -- EXP-9A: fine-tune YOLOP with a switched anchor set.

Thin adapter so the SECOND architecture trains under the SAME protocol as R2
(trac's dataset, trac's loss form, trac's optimizer schedule).  The only thing
that differs between arms is the anchor set -- which is the intervention.

Protocol (identical for every arm, R2 values):
    dataset        datasets.bdd100k tri_train (letterbox 640, hflip aug)
    det loss       losses/yolo_loss.YOLOLoss   (rule: 0.5 < box/anchor < 2.0)
    seg loss       losses.multitask_loss.seg_ce_loss (CE, fg_weight=10)
    optimizer      AdamW lr=1e-3  wd=5e-4  grad-clip 10.0
    schedule       CosineAnnealingLR over epochs*steps
    init           weights/YOLOP_End-to-end.pth   (SAME for all arms)
    seed           fixed per arm

ANCHOR SETS come from experiments/phase6/phase6_round3_anchor_summary.json --
the STEP-1 zero-training artefact -- so the set that was ANALYSED is bit-identical
to the set that is TRAINED.  `shipped` is re-read live from the built model.
A mismatch aborts instead of silently training a different set.

WHY TRAC'S LOSS AND NOT YOLOP'S: YOLOP's own matcher uses anchor_t=4.0 plus
5-neighbour offset expansion; trac's uses the strict project rule with no
expansion.  Two different rulers cannot be pooled.  Holding the RULER fixed and
varying only the anchors keeps the cross-architecture comparison single-variable
(registered in docs/PHASE6_ROUND3_PREREGISTRATION.md section 3).  The native-rule
readings are reported separately as a measured confound.

Usage
    python scripts/phase6_round3_yolop_train.py --anchors-preset shipped \
        --outdir experiments/phase6/round3_yolop_D --epochs 2 --seed 0
    # micro-probe (verify the chain + measure ms/step, no real training):
    ... --anchors-preset shipped --max-steps 60
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from datasets.bdd100k import BDD100KDataset, collate_train  # noqa: E402
from losses.multitask_loss import MultiTaskLoss, seg_ce_loss  # noqa: E402
from evaluation.evaluate_baseline import (  # noqa: E402
    build_yolop, normalize_batch)

SUMMARY = os.path.join(ROOT, "experiments", "phase6", "phase6_round3_anchor_summary.json")
STRIDES = [8, 16, 32]


def arm_anchors(preset):
    """Return (pixels (nl,na,2) float array, provenance str).  Live-read shipped;
    file-read flip/kmeans from the STEP-1 artefact."""
    model = build_yolop("cpu")
    det = model.model[model.detector_index]
    shipped = det.anchor_grid.reshape(det.nl, det.na, 2).detach().cpu().numpy().astype(float)
    if preset == "shipped":
        return shipped, "live from built model anchor_grid"
    with open(SUMMARY) as f:
        s = json.load(f)
    key = {"flip": "flipped", "kmeans": "kmeans"}[preset]
    A = np.asarray(s[key], dtype=float).reshape(shipped.shape)
    # guard against a stale artefact: the file must agree with what STEP-1 analysed
    assert A.shape == shipped.shape, "anchor shape mismatch"
    return A, "%s <- %s" % (key, os.path.relpath(SUMMARY, ROOT))


def apply_anchors(model, px):
    """Overwrite BOTH detector buffers so the head decodes with the arm's set."""
    det = model.model[model.detector_index]
    nl, na, _ = px.shape
    assert (nl, na) == (det.nl, det.na), "anchor count mismatch"
    t = torch.tensor(px, dtype=torch.float32)
    with torch.no_grad():
        det.anchor_grid.copy_(t.view(nl, 1, na, 1, 1, 2))
        det.anchors.copy_(t / torch.tensor(STRIDES, dtype=torch.float32).view(-1, 1, 1))


def seg_logits(p):
    """YOLOP's seg heads are wrapped in nn.Sigmoid() inside its own forward.
    Recover logits exactly (logit(sigmoid(z)) = z) so trac's CE loss applies."""
    return torch.logit(p.clamp(1e-6, 1.0 - 1e-6))


def gpu_note(device):
    if device.type != "cuda":
        return "cpu"
    return "mem %d/%dMiB peak %dMiB" % (
        torch.cuda.memory_allocated() / 1024 ** 2,
        torch.cuda.max_memory_allocated() / 1024 ** 2,
        torch.cuda.max_memory_allocated() / 1024 ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors-preset", required=True, choices=["shipped", "flip", "kmeans"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--num-images", type=int, default=0, help="0 = full tri_train")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epoch (micro-probe uses ~60)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=100,
                    help="rolling checkpoint interval in steps (0 disables); see RESUME NOTE")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <outdir>/checkpoint.pt + progress.json if present")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "training_log.txt")

    # Resume guard: if this arm already finished, do nothing at all.  Without it a
    # relaunch after a crash that happened between "training done" and "eval/append
    # done" would loop zero epochs and CLOBBER a good metrics.json with empty loss
    # values -- a silently corrupted result row.
    if args.resume:
        _p = os.path.join(args.outdir, "progress.json")
        _m = os.path.join(args.outdir, "metrics.json")
        if os.path.exists(_p) and os.path.exists(_m):
            try:
                if json.load(open(_p)).get("done"):
                    print("[r3-yolop] %s already complete - nothing to do" % args.outdir, flush=True)
                    return
            except (ValueError, KeyError):
                pass
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random as _rnd
    _rnd.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[r3-yolop] torch %s cuda=%s device=%s" % (torch.__version__, torch.cuda.is_available(), device), flush=True)
    print("[r3-yolop] GPU %s" % (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"), flush=True)

    px, prov = arm_anchors(args.anchors_preset)
    print("[r3-yolop] arm=%s anchors from %s" % (args.anchors_preset, prov), flush=True)
    for lv in range(3):
        print("    L%d: %s" % (lv + 1, " ".join("(%g,%g)" % (w, h) for w, h in px[lv])), flush=True)

    model = build_yolop(device)
    apply_anchors(model, px)
    det = model.model[model.detector_index]
    n_params = sum(p.numel() for p in model.parameters())
    print("[r3-yolop] params=%.3fM nl=%d na=%d stride=%s" % (
        n_params / 1e6, det.nl, det.na, det.stride.tolist()), flush=True)

    loss_fn = MultiTaskLoss(torch.tensor(px, dtype=torch.float32), nc=1, img_size=640,
                            lambda_det=1.0, lambda_da=1.0, lambda_lane=1.0).to(device)

    ds = BDD100KDataset(os.path.join(ROOT, "data", "bdd100k"), split="tri_train", train=True)
    if args.num_images:
        ds.names = ds.names[:args.num_images]
        if getattr(ds, "_det_by_name", None):
            keep = set(ds.names)
            ds._det_by_name = {k: v for k, v in ds._det_by_name.items() if k in keep}
    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_train,
                        drop_last=True, generator=g, pin_memory=(device.type == "cuda"),
                        prefetch_factor=(4 if args.num_workers > 0 else None))
    print("[r3-yolop] train images=%d batch=%d steps/epoch=%d" % (len(ds), args.batch_size, len(loader)), flush=True)

    def make_loader(ep):
        """Per-epoch loader with a DETERMINISTIC shuffle: seeding off (seed, ep)
        rather than drawing sequentially from one generator means an epoch's
        batch order is reproducible from the epoch number alone.  Without this,
        resuming mid-run would silently replay a DIFFERENT order than the
        interrupted run used, which is exactly the kind of quiet protocol drift
        this project forbids.  All arms share this scheme, so the comparison is
        unaffected."""
        gg = torch.Generator(); gg.manual_seed(args.seed * 1000 + ep)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_train,
                          drop_last=True, generator=gg, pin_memory=(device.type == "cuda"),
                          prefetch_factor=(4 if args.num_workers > 0 else None))

    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs * len(loader), 1))

    # ---------------- RESUME ----------------
    # An abrupt power-off is a real event on this machine (two unclean shutdowns in
    # three days), so progress must survive one.  Checkpoints are written to a temp
    # path and then os.replace()d -> the swap is atomic, so power loss during the
    # write can never leave a half-written checkpoint behind.
    ckpt_path = os.path.join(args.outdir, "checkpoint.pt")
    prog_path = os.path.join(args.outdir, "progress.json")
    start_ep, start_it, resumed = 1, 0, False
    if args.resume and os.path.exists(ckpt_path) and os.path.exists(prog_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        try:
            opt.load_state_dict(ck["opt_state"])
            sched.load_state_dict(ck["sched_state"])
        except (ValueError, KeyError) as e:          # opt/sched state is optional
            print("[r3-yolop] WARN could not restore opt/sched (%s)" % e, flush=True)
        pr = json.load(open(prog_path))
        start_ep, start_it = int(pr["epoch"]), int(pr["step"])
        resumed = True
        print("[r3-yolop] RESUME from %s: epoch=%d step=%d" % (ckpt_path, start_ep, start_it), flush=True)

    def save_ckpt(ep, it, done):
        ck = {"epoch": ep, "model_state": model.state_dict(), "opt_state": opt.state_dict(),
              "sched_state": sched.state_dict(), "seed": args.seed,
              "arm": args.anchors_preset, "anchors_px": px.tolist(), "params": n_params}
        tmp = ckpt_path + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, ckpt_path)                   # atomic
        tmp_p = prog_path + ".tmp"
        with open(tmp_p, "w") as f:
            json.dump({"epoch": ep, "step": it, "done": done}, f)
        os.replace(tmp_p, prog_path)

    cfg_rec = {"arm": args.anchors_preset, "anchors_px": px.tolist(), "anchors_provenance": prov,
               "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
               "weight_decay": args.weight_decay, "seed": args.seed, "num_images": len(ds),
               "loss": "losses.multitask_loss.MultiTaskLoss (project rule 0.5<r<2.0)",
               "init": "weights/YOLOP_End-to-end.pth", "params": n_params,
               "max_steps": args.max_steps}
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(cfg_rec, f, indent=2)
    with open(os.path.join(args.outdir, "anchors.json"), "w") as f:
        json.dump(px.tolist(), f)          # read back by the eval side

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    last = {}
    for ep in range(1, args.epochs + 1):
        if ep < start_ep:
            print("[r3-yolop] ep %d already complete (resume) - skip" % ep, flush=True)
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
            # INPUT-DISTRIBUTION PARITY WITH EVAL (added 2026-09-11).
            # The dataset yields [0,1]; the evaluator normalises with ImageNet
            # mean/std for YOLOP.  The first STEP-2 run trained on un-normalised
            # [0,1] inputs, so every arm learned an input convention the evaluator
            # never uses and lost ~90% of the released mAP (0.7712 -> 0.07 at 200
            # val images).  Both sides now go through the one shared helper, so
            # this cannot drift again.
            img = normalize_batch(img, "imagenet")
            det_t = batch["det_targets"].to(device)
            da_m = batch["da_mask"].to(device)
            lane_m = batch["lane_mask"].to(device)
            opt.zero_grad()
            d, da, lane = model(img)                    # d = list of 3 raw level tensors
            total, losses = loss_fn(d, seg_logits(da), seg_logits(lane), det_t, da_m, lane_m,
                                    None, img_size=640)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            # --- VRAM-cliff guard (pre-registration 11.3) -----------------------
            # On Windows/WDDM an over-subscribed allocation does NOT raise OOM: the
            # driver silently pages VRAM to host RAM over PCIe, which pins the SM
            # clock at maximum while drawing almost no power, and costs ~18x per
            # step.  Measured: batch 8 = 99.85% of the card = 5532 ms/step; batch 6
            # = 74.7% = 306 ms/step.  Abort loudly rather than burn hours silently.
            if device.type == "cuda":
                frac = (torch.cuda.max_memory_reserved()
                        / torch.cuda.get_device_properties(0).total_memory)
                if frac > 0.96:
                    raise RuntimeError(
                        "[r3-yolop] VRAM-CLIFF ABORT: %.1f%% of device memory reserved. "
                        "WDDM will silently page to host RAM (~18x slower). "
                        "Lower --batch-size (pre-registration 11.3)." % (100.0 * frac))
                if frac > 0.90 and not vram_warned:
                    vram_warned = True
                    print("[r3-yolop] WARN VRAM guard: %.1f%% of device memory reserved - "
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
                line = ("[%s] r3_yolop arm=%s ep%d step %d/%d | %.1fm in, ETA %.1fm "
                        "| %.0fms/step | %s | loss %.4f det %.4f da %.4f lane %.4f" % (
                            time.strftime("%H:%M:%S"), args.anchors_preset, ep, it + 1, len(loader),
                            el / 60, (len(loader) - it - 1) / max(done / max(el, 1e-9), 1e-9) / 60,
                            el / max(done, 1) * 1e3, gpu_note(device),
                            avg.get("total", 0), avg.get("det", 0), avg.get("da", 0), avg.get("lane", 0)))
                print(line, flush=True)
                with open(log_path, "a") as f:
                    f.write(line + "\n")
            last = {k: v / max(steps, 1) for k, v in run.items()}
            if args.max_steps and steps >= args.max_steps:
                break
        sched.step()
        final = (ep == args.epochs)
        save_ckpt(ep + 1, 0, final)              # next epoch starts at step 0
        line = "[r3-yolop] ep %d DONE avg_loss=%.4f elapsed %.1fm saved checkpoint.pt%s" % (
            ep, last.get("total", 0), (time.time() - t_all) / 60, " [FINAL]" if final else "")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0
    wall_min = (time.time() - t_all) / 60
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"stage": "yolop_finetune", "arm": args.anchors_preset, "params": n_params,
                   "epochs": args.epochs, "num_images": len(ds), "batch_size": args.batch_size,
                   "seed": args.seed, "lr": args.lr,
                   "final_avg_loss": last.get("total", 0),
                   "final_det_loss": last.get("det", 0), "wall_min": round(wall_min, 2),
                   "peak_gpu_mem_mib": round(peak, 1), "max_steps": args.max_steps}, f, indent=2)
    print("[r3-yolop] done -> %s (wall %.1f min, peak %.0f MiB)" % (args.outdir, wall_min, peak), flush=True)


if __name__ == "__main__":
    main()

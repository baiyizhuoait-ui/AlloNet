"""Deployment profile for Model B -- what an edge runtime actually has to do.

Answers three questions with measurements, not assertions:
  1. COST INVENTORY -- params, MACs/FLOPs at both canvases, the per-component
     split, the op-count mix (which is what an NPU scheduler pays for), the
     weight bytes at FP32/FP16/INT8, and the largest single activation.
  2. CPU LATENCY -- real measured wall-clock on this machine's CPU, 1 thread and
     all threads, at 640x640 and 384x640.  A modern x86 laptop CPU is a
     defensible *upper anchor* for the strong-CPU end of the device ladder.
  3. INT8 FIDELITY -- a weight+activation fake-quantisation simulation with a
     calibration pass, then mask-level agreement against the FP32 model.  This
     is a SIMULATION of PTQ, not a real quantised deployment; it bounds the
     damage, it does not certify a vendor toolchain.

CPU only, no checkpoint training, no GPU.  Reads weights if a checkpoint is
given (so the fidelity numbers come from the real trained model).
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.static_model import StaticMultiTaskModel          # noqa: E402
from profiling.flops_real import count_flops                  # noqa: E402


def load_model(cfg_path, ckpt):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = StaticMultiTaskModel(cfg["model"])
    info = {"cfg": cfg_path, "params": sum(p.numel() for p in model.parameters())}
    if ckpt and os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        for key in ("model", "state_dict", "model_state", "net"):
            if isinstance(ck.get(key), dict):
                sd = ck[key]
                break
        else:
            sd = ck if isinstance(ck, dict) and all(
                isinstance(v, torch.Tensor) for v in ck.values()) else {}
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        info["ckpt"] = ckpt
        info["epoch"] = ck.get("epoch")
        info["missing"] = len(missing)
        info["unexpected"] = len(unexpected)
    model.eval()
    return model, cfg, info


# --------------------------------------------------------------- 1. inventory
def inventory(model, x):
    """Per-component MACs (attributed via top-level child), op mix, big allocs."""
    cur = {"name": "?"}
    hooks = []
    for name, mod in model.named_children():
        def mk(n):
            def pre(mod_, inp):
                cur["name"] = n
            return pre
        hooks.append(mod.register_forward_pre_hook(mk(name)))

    orig_conv, orig_interp = F.conv2d, F.interpolate
    acc, ops, peak = {}, Counter(), {"bytes": 0, "what": ""}

    def note(t, what):
        b = t.numel() * t.element_size()
        if b > peak["bytes"]:
            peak.update(bytes=b, what=what)

    def conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        out = orig_conv(input, weight, bias, stride, padding, dilation, groups)
        n, oh, ow = input.shape[0], out.shape[2], out.shape[3]
        m = n * oh * ow * weight.shape[0] * (input.shape[1] // groups) \
            * weight.shape[2] * weight.shape[3]
        k = cur["name"]
        acc[k] = acc.get(k, 0) + m
        ops["conv2d"] += 1
        note(out, f"conv out {tuple(out.shape)} @{k}")
        return out

    def interp(t, *a, **kw):
        out = orig_interp(t, *a, **kw)
        ops["interpolate"] += 1
        note(out, f"upsample out {tuple(out.shape)} @{cur['name']}")
        return out

    F.conv2d, F.interpolate = conv, interp
    try:
        with torch.no_grad():
            model(x)
    finally:
        F.conv2d, F.interpolate = orig_conv, orig_interp
        for h in hooks:
            h.remove()

    mods = Counter(type(m).__name__ for m in model.modules())
    return acc, ops, mods, peak


# ---------------------------------------------------------------- 2. latency
def latency(model, hw, threads, warmup=8, reps=30):
    x = torch.randn(1, 3, *hw)
    torch.set_num_threads(threads)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 90))


# ------------------------------------------------------- 3. INT8 fidelity sim
def qsim_outputs(model, images, nbits=8, observer="absmax", calib=30):
    """Fake quantisation placed where a real runtime puts it.

    Weights are per-output-channel INT8.  Activations are per-tensor and are
    quantised at the output of BatchNorm / ReLU -- NOT at the raw conv output.
    That distinction matters: in every shipping toolchain (RKNN, nncase,
    TensorRT) conv+BN is fused into one op and the quantiser sees the post-BN
    tensor.  Quantising the pre-BN conv output was measured first and rejected;
    the pre-BN ranges are much wider and unbounded, which made an
    already-outlier-sensitive observer look worse than it is.
    """
    qmax = 2 ** (nbits - 1) - 1
    st = {"mode": "off", "w": {}, "a": {}, "wi": 0}
    orig = F.conv2d

    def obs(t):
        a = t.detach().abs()
        if observer == "absmax":
            return float(a.max())
        f = a.flatten()
        if f.numel() > 200000:
            f = f[:: max(1, f.numel() // 200000)]
        return float(torch.quantile(f, 0.9999))

    def qdq(t, s):
        return torch.clamp(torch.round(t / s), -qmax, qmax) * s

    def conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        w = weight
        if st["mode"] == "off":
            return orig(input, w, bias, stride, padding, dilation, groups)
        i = st["wi"]
        st["wi"] += 1
        if i == 0:                      # the input image is quantised too
            v = obs(input)
            if st["mode"] == "calib":
                st["a"]["in"] = max(st["a"].get("in", 0.0), v)
            elif st["a"].get("in", 0.0) > 0:
                input = qdq(input, st["a"]["in"] / qmax)
        wam = w.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
        if st["mode"] == "calib":
            st["w"][i] = torch.maximum(st["w"].get(i, torch.zeros_like(wam)), wam)
        else:
            w = qdq(w, (st["w"][i] / qmax).clamp_min(1e-8))
        return orig(input, w, bias, stride, padding, dilation, groups)

    hooks = []

    def act_hook(key):
        def h(mod, inp, out):
            if st["mode"] == "off":
                return out
            v = obs(out)
            if st["mode"] == "calib":
                st["a"][key] = max(st["a"].get(key, 0.0), v)
                return out
            s = st["a"].get(key, 0.0) / qmax
            return qdq(out, s) if s > 0 else out
        return h

    for i, m in enumerate(model.modules()):
        if isinstance(m, (nn.BatchNorm2d, nn.ReLU)):
            hooks.append(m.register_forward_hook(act_hook(f"m{i}")))

    def run(mode, imgs):
        F.conv2d = conv
        outs = []
        try:
            with torch.no_grad():
                for im in imgs:
                    st["wi"], st["mode"] = 0, mode
                    outs.append(model(im.unsqueeze(0)))
        finally:
            F.conv2d = orig
        return outs

    F.conv2d = conv
    try:
        with torch.no_grad():
            for im in images[:calib]:
                st["wi"], st["mode"] = 0, "calib"
                model(im.unsqueeze(0))
    finally:
        F.conv2d = orig

    # DEFECT FIXED 2026-09-14.  These two lines used to sit HERE, i.e. the
    # activation hooks were removed before the evaluation pass ran.  The effect
    # was silent and large: the "W8A8" numbers below were in fact
    # per-output-channel WEIGHT + input-image quantisation only, with every
    # BatchNorm/ReLU activation left in FP32 -- i.e. the dominant error source
    # in CNN post-training quantisation was switched off during measurement
    # (activations were still *observed* during calibration, so nothing looked
    # wrong).  Reproduced exactly with
    #   scripts/phase6_quant_head_audit.py --act-quant off --only B_inner_only
    # which returns the old numbers digit-for-digit (DA 0.99906 / DA fgIoU
    # 0.98833 / lane 0.99929 / lane fgIoU 0.94846).  With the hooks live, the
    # honest W8A8 numbers are DA 0.99426 / 0.94269 and lane 0.99644 / 0.73946.
    # The activation quantiser is ALSO incomplete in a second way: it is only
    # attached to BN/ReLU modules, so the detection head's concatenated output
    # -- where box coordinates in pixels share one scale with probabilities in
    # [0,1] -- was never quantised at all.  See phase6_quant_head_audit.py for
    # what happens when it is.
    fp = run("off", images[calib:])
    q = run("apply", images[calib:])
    for h in hooks:
        h.remove()
    return fp, q, st


def mask_agree(a, b):
    pa, pb = a.argmax(1), b.argmax(1)
    agree = float((pa == pb).float().mean())
    fa, fb = (pa > 0), (pb > 0)
    inter = float((fa & fb).sum())
    union = float((fa | fb).sum())
    return agree, (inter / union if union else float("nan")), float(fa.float().mean()), \
        float(fb.float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/phase6_r4_R2_thin14_z16.yaml")
    ap.add_argument("--ckpt", default="experiments/phase6/final/B100/checkpoint.pt")
    ap.add_argument("--data", default="data/bdd100k")
    ap.add_argument("--n-qsim", type=int, default=90)
    ap.add_argument("--outdir", default="experiments/phase6/deploy_profile")
    args = ap.parse_args()

    out = {}
    model, cfg, info = load_model(args.cfg, args.ckpt)
    out["model"] = info
    nthreads = os.cpu_count() or 4
    out["machine"] = {"cpu_count": nthreads,
                      "threads_name": torch.get_num_threads()}

    for tag, hw in (("640x640", (640, 640)), ("384x640", (384, 640))):
        x = torch.randn(1, 3, *hw)
        flops = count_flops(model, x) / 1e9
        acc, ops, mods, peak = inventory(model, x)
        out.setdefault("cost", {})[tag] = {
            "GFLOPs": round(flops, 4), "GMACs": round(flops / 2, 4),
            "macs_by_component": {k: round(v / 1e9, 4) for k, v in sorted(acc.items())},
            "ops": dict(ops),
            "peak_activation_MB": round(peak["bytes"] / 2 ** 20, 2),
            "peak_activation_what": peak["what"],
        }
        if tag == "640x640":
            out["modules"] = dict(mods)

    p = info["params"]
    out["weight_bytes"] = {"FP32": p * 4, "FP16": p * 2, "INT8": p,
                           "FP32_MB": round(p * 4 / 2 ** 20, 3),
                           "FP16_MB": round(p * 2 / 2 ** 20, 3),
                           "INT8_MB": round(p / 2 ** 20, 3)}

    out["cpu_latency_ms"] = {}
    th_list = sorted({1, min(4, nthreads)})
    for tag, hw in (("640x640", (640, 640)), ("384x640", (384, 640))):
        for th in th_list:
            p50, p90 = latency(model, hw, th)
            out["cpu_latency_ms"][f"{tag}_t{th}"] = {
                "p50": round(p50, 1), "p90": round(p90, 1),
                "FPS": round(1000.0 / p50, 1)}

    # ---- INT8 fidelity on real val images
    if os.path.exists(os.path.join(args.data, "splits", "tri_val.txt")):
        from datasets.bdd100k import BDD100KDataset
        # images only: loading the detection-label JSON for the whole val split
        # costs hundreds of MB and we do not need it here.
        ds = BDD100KDataset(args.data, split="tri_val", img_size=640,
                            with_det=False, with_da=False, with_lane=False)
        # CONTIGUOUS sample, not a strided one: the first 30 are consumed by the
        # calibration pass, so the list must be strictly longer than that.
        n = min(len(ds), max(args.n_qsim, 40))
        imgs = [ds[i]["image"] for i in range(n)]
        variants = [("W8A8_absmax", 8, "absmax"),
                    ("W8A8_p9999", 8, "p9999"),
                    ("W6A6_absmax", 6, "absmax")]
        res = {"n_images_eval": None, "variants": {}}
        for name, bits, obsrv in variants:
            fp, q8, _ = qsim_outputs(model, imgs, nbits=bits, observer=obsrv)
            if res["n_images_eval"] is None:
                res["n_images_eval"] = len(fp)
            v = {"bits": bits, "observer": obsrv}
            for j, task in enumerate(("det", "da", "lane")):
                if task == "det":
                    a = torch.cat([o[j].flatten() for o in fp])
                    b = torch.cat([o[j].flatten() for o in q8])
                    v[task] = {
                        "logit_cosine": round(float(F.cosine_similarity(
                            a.unsqueeze(0), b.unsqueeze(0))), 6),
                        "relative_L2": round(float((a - b).norm() / a.norm()), 5)}
                else:
                    ag, iou, fpa, fpb = [], [], [], []
                    for o, q in zip(fp, q8):
                        g, u, fa, fb = mask_agree(o[j], q[j])
                        ag.append(g); iou.append(u); fpa.append(fa); fpb.append(fb)
                    mfa, mfb = float(np.mean(fpa)), float(np.mean(fpb))
                    v[task] = {
                        "mask_pixel_agreement": round(float(np.mean(ag)), 5),
                        "fgIoU_FP32_vs_int8": round(float(np.nanmean(iou)), 5),
                        "fg_rate_FP32": round(mfa, 5),
                        "fg_rate_int8": round(mfb, 5),
                        "fg_area_inflation": round(mfb / mfa, 3) if mfa else None}
            res["variants"][name] = v
        out["int8_sim"] = res
        out["int8_sim"]["_note"] = (
            "FAKE-QUANT simulation, not a vendor toolchain result. Per-output-"
            "channel weights; per-tensor activations quantised at BatchNorm/ReLU "
            "outputs, i.e. where a fused conv+BN graph puts them, calibrated on "
            "30 images. W8A8_absmax is the naive observer; W8A8_p9999 is the "
            "usual PTQ setting; W6A6_absmax is a margin probe, NOT a deployment "
            "candidate. No bias correction, no AdaRound, no QAT, so these are a "
            "LOWER bound on what a real toolchain achieves.")
        out["int8_sim"]["_defect_fixed_2026_09_14"] = (
            "Runs before this date removed the activation hooks before the eval "
            "pass, so their 'W8A8' figures were weight+input quantisation only "
            "(activations stayed FP32).  Those inflated numbers are "
            "DA 0.99906/0.98833 and lane 0.99929/0.94846; with the hooks live "
            "the same protocol gives DA 0.99426/0.94269 and lane 0.99644/"
            "0.73946.  Reproduce the old behaviour with "
            "scripts/phase6_quant_head_audit.py --act-quant off.  NOTE this "
            "simulation still does NOT quantise the detection head's output "
            "tensor -- see phase6_quant_head_audit.py, where quantising it "
            "per-tensor drives 100% of the objectness channel to zero.")
    else:
        out["int8_sim"] = {"error": "split tri_val.txt not found"}

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "profile.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

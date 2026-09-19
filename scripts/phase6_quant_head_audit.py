"""Does the detection head survive a REALISTIC INT8 output quantiser?

WHY THIS EXISTS.  Two published results bracket our own fake-quant claim, and
they disagree:

  * Q-YOLOP (arXiv:2307.04537) -- panoptic driving perception on BDD100K.
    INT8 PTQ drove drivable-area mIoU 0.842 -> 0.285 and lane 0.402 -> 0.248
    ("catastrophic loss of accuracy for segmentation tasks"); they needed QAT.
  * YOLOP @ ENSICAEN x Valeo (INT8 PTQ, entropy calibration) -- DA mIoU
    91.24 -> 90.99 (negligible) but detection mAP@50 71.94 -> 58.20 (-13.7),
    and the FIRST attempt gave 0.0 mAP: no boxes at all.
    Root cause, in their words: the final Concat merges probability scores
    (in [0,1]) with bounding-box coordinates (in [0,640]); one quantisation
    scale is dictated by the coordinates, so the probabilities are "drowned
    in the first quantisation bin and rounded to 0".

Our own head has EXACTLY that topology (models/representation/det_from_z.py:227-231,
models/heads/det_head.py:58-66):

    y = x.sigmoid()
    y[..., 0:2] = (y[...,0:2]*2 - 0.5 + grid) * stride       # xy in PIXELS
    y[..., 2:4] = (y[...,2:4]*2)**2 * anchor_grid            # wh in PIXELS
    return torch.cat(outs, 1)                                # xy|wh|obj|cls

and our previous INT8 simulation could not see it: quantisers were registered on
BatchNorm2d / ReLU modules only, and there is no BN or ReLU after that Cat.  The
tensor that carries the objectness and class scores straight into NMS was never
quantised, so "detection logit cosine = 1.000000" was measured on a tensor whose
own output quantiser did not exist.

WHAT THIS SCRIPT MEASURES -- four variants, all on the real trained checkpoint,
all with the same BN/ReLU inner quantisers:

  A fp32          no quantisation at all (reference)
  B inner_only    previous behaviour: inner quantisers, head output untouched
  C inner+head_pt inner quantisers AND a single per-TENSOR quantiser on the
                  concatenated head output -- the naive-toolchain behaviour that
                  the Valeo report blames
  D inner+head_pc inner quantisers AND a per-CHANNEL quantiser on the same
                  tensor -- the cheapest implementable fix (per-axis activation
                  quantisation; the Valeo graph-normalisation trick reaches the
                  same end by rescaling the coordinate branches before the Cat)

Metrics are GT-FREE so nothing depends on the label files: objectness survival
(fraction of obj values that round to exactly 0, and the step size that does it),
boxes surviving NMS, and -- the operative one -- how many of the FP32 detections
are preserved in the quantised model at IoU>=0.5.  DA / lane mask agreement is
carried along as a control: variants C and D must not move it, since the head
hook touches the detection tensor only.

This bounds the damage; it is still a SIMULATION, not a vendor toolchain.
No bias correction, no AdaRound, no QAT -> a LOWER bound on what a real
toolchain gets, for every variant.

CPU only, bounded memory (<= ~1200 MB), does not touch the GPU.
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# Pin our own package into sys.modules BEFORE anything can shadow it: the
# baselines tree ships its own `models/` and has shadowed ours twice already.
import models.static_model                                            # noqa: E402,F401
from models.static_model import StaticMultiTaskModel                  # noqa: E402
from evaluation.nms import non_max_suppression                        # noqa: E402


MEM_FLOOR_MB = 5000


def mem_available_mb():
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return int(ln.split()[1]) // 1024
    except OSError:
        pass
    return None


def load_model(cfg_path, ckpt):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = StaticMultiTaskModel(cfg["model"])
    sd = {}
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
    info = {"cfg": cfg_path, "ckpt": ckpt,
            "params": sum(p.numel() for p in model.parameters())}
    if sd:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        info["missing"] = len(missing)
        info["unexpected"] = len(unexpected)
        info["epoch"] = ck.get("epoch")
    model.eval()
    return model, info


# ---------------------------------------------------------------- quantisers
def qdq(t, s, qmax):
    return torch.clamp(torch.round(t / s), -qmax, qmax) * s


class Sim:
    """Weight/activation fake-quant with an OPTIONAL detection-head hook."""

    def __init__(self, model, nbits=8, observer="p9999", head_mode="none",
                 head_bits=None, act_quant=True):
        self.model = model
        self.qmax = 2 ** (nbits - 1) - 1
        self.head_qmax = 2 ** ((head_bits or nbits) - 1) - 1
        self.observer = observer
        self.head_mode = head_mode            # none | per_tensor | per_channel
        self.act_quant = act_quant
        self.mode = "off"
        self.a, self.w = {}, {}
        self.head_scale = None
        self.head_branch_scale = None
        self._conv_i = 0
        self._hooks = []
        self._orig_conv = F.conv2d

    # --- observer ---------------------------------------------------------
    def obs(self, t):
        a = t.detach().abs().flatten()
        if self.observer == "absmax":
            return float(a.max())
        if a.numel() > 200000:
            a = a[:: max(1, a.numel() // 200000)]
        return float(torch.quantile(a, 0.9999))

    def obs_branch(self, t):
        """per-channel (last-axis) scale: one scale per channel of the Cat."""
        a = t.detach().abs().flatten(end_dim=-2)
        if self.observer == "absmax":
            return a.amax(dim=0)
        idx = max(1, a.shape[0] // 4000)
        return torch.quantile(a[::idx], 0.9999, dim=0)

    # --- install ----------------------------------------------------------
    def install(self):
        def act_hook(key):
            def h(mod, inp, out):
                if self.mode == "off":
                    return out
                v = self.obs(out)
                if self.mode == "calib":
                    self.a[key] = max(self.a.get(key, 0.0), v)
                    return out
                s = self.a.get(key, 0.0) / self.qmax
                return qdq(out, s, self.qmax) if s > 0 else out
            return h

        for i, m in enumerate(self.model.modules()):
            if self.act_quant and isinstance(m, (nn.BatchNorm2d, nn.ReLU)):
                self._hooks.append(m.register_forward_hook(act_hook(f"m{i}")))

        if self.head_mode != "none":
            def head_hook(mod, inp, out):
                if self.mode == "off":
                    return out
                if self.head_mode == "per_tensor":
                    v = self.obs(out)
                    if self.mode == "calib":
                        self.head_scale = max(self.head_scale or 0.0, v)
                        return out
                    if not self.head_scale:
                        return out
                    s = self.head_scale / self.head_qmax
                    self.last_head_step = float(s)
                    return qdq(out, s, self.head_qmax)
                v = self.obs_branch(out)
                if self.mode == "calib":
                    self.head_branch_scale = (v if self.head_branch_scale is None
                                              else torch.maximum(self.head_branch_scale, v))
                    return out
                if self.head_branch_scale is None:
                    return out
                s = (self.head_branch_scale / self.head_qmax).clamp_min(1e-8)
                self.last_head_step = float(s.min())
                return qdq(out, s.view(1, 1, -1), self.head_qmax)
            self._hooks.append(
                self.model.det_head.register_forward_hook(head_hook))

        orig = self._orig_conv

        def conv(input, weight, bias=None, stride=1, padding=0, dilation=1,
                 groups=1):
            if self.mode == "off":
                return orig(input, weight, bias, stride, padding, dilation, groups)
            w = weight
            i = self._conv_i
            self._conv_i += 1
            if i == 0:
                v = self.obs(input)
                if self.mode == "calib":
                    self.a["in"] = max(self.a.get("in", 0.0), v)
                elif self.a.get("in", 0.0) > 0:
                    input = qdq(input, self.a["in"] / self.qmax, self.qmax)
            wam = weight.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
            if self.mode == "calib":
                self.w[i] = torch.maximum(self.w.get(i, torch.zeros_like(wam)), wam)
            else:
                w = qdq(weight, (self.w[i] / self.qmax).clamp_min(1e-8), self.qmax)
            return orig(input, w, bias, stride, padding, dilation, groups)

        F.conv2d = conv
        return self

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        F.conv2d = self._orig_conv

    def run(self, imgs, mode):
        self._conv_i = 0
        self.mode = mode
        outs = []
        with torch.no_grad():
            for im in imgs:
                self._conv_i = 0
                outs.append(self.model(im.unsqueeze(0)))
        return outs


# ------------------------------------------------------------------ metrics
def mask_stats(ref_ag, ag):
    """argmax maps, uint8.  pixel agreement + foreground IoU."""
    a = (ref_ag > 0)
    b = (ag > 0)
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return float((ref_ag == ag).mean()), (inter / union if union else float("nan"))


def det_stats(det, ref_det):
    """GT-free detection survival.

    `det` / `ref_det`: (1, 25200, no) decoded tensors, raw (pre-NMS).
    Returns objectness survival + NMS box survival + box-preservation rate.
    """
    o = det[0, :, 4]
    ro = ref_det[0, :, 4]
    out = {
        "obj_frac_rounded_to_zero": float((o == 0).float().mean()),
        "obj_ref_frac_zero": float((ro == 0).float().mean()),
        "obj_max": float(o.max()), "obj_ref_max": float(ro.max()),
        "obj_p999": float(torch.quantile(o, 0.999)),
        "obj_ref_p999": float(torch.quantile(ro, 0.999)),
        "head_step_obj_ratio": None,   # filled by caller
    }
    # NMS at the operating threshold (0.25) and at the harness threshold (0.001)
    for tag, ct in (("c25", 0.25), ("c001", 0.001)):
        b = non_max_suppression(det.clone(), conf_thres=ct, iou_thres=0.6)[0]
        rb = non_max_suppression(ref_det.clone(), conf_thres=ct, iou_thres=0.6)[0]
        out[f"nms_boxes_{tag}"] = int(b.shape[0])
        out[f"nms_boxes_ref_{tag}"] = int(rb.shape[0])
    # box preservation at IoU>=0.5 against the FP32 reference detections
    rb = non_max_suppression(ref_det.clone(), conf_thres=0.25, iou_thres=0.6)[0]
    b = non_max_suppression(det.clone(), conf_thres=0.25, iou_thres=0.6)[0]
    if rb.shape[0] and b.shape[0]:
        iou = _iou_xyxy(rb[:, :4], b[:, :4])
        best = iou.max(dim=1).values
        out["boxes_preserved_frac"] = float((best >= 0.5).float().mean())
        out["matched_box_iou_mean"] = float(best.mean())
    else:
        out["boxes_preserved_frac"] = 0.0 if rb.shape[0] else None
        out["matched_box_iou_mean"] = None
    return out


def _iou_xyxy(a, b):
    a = a.clone(); b = b.clone()
    a[:, 2:] -= a[:, :2]; b[:, 2:] -= b[:, :2]
    a[:, :2] -= a[:, 2:] / 2; a[:, 2:] = a[:, :2] + 2 * a[:, 2:]
    b[:, :2] -= b[:, 2:] / 2; b[:, 2:] = b[:, :2] + 2 * b[:, 2:]
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    ua = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None] \
        + ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :] - inter
    return inter / ua.clamp_min(1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/phase6_r4_R2_thin14_z16.yaml")
    ap.add_argument("--ckpt", default="experiments/phase6/final/B100/checkpoint.pt")
    ap.add_argument("--data", default="data/bdd100k")
    ap.add_argument("--n", type=int, default=90, help="calib 30 + eval rest")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--act-quant", choices=("on", "off"), default="on",
                    help="'off' reproduces the DEFECT in phase6_deploy_profile.py, "
                         "which removed its activation hooks before the eval pass "
                         "and therefore measured weight-only quantisation while "
                         "calling it W8A8")
    ap.add_argument("--only", default="", help="comma list of variant names to run")
    ap.add_argument("--outdir", default="experiments/phase6/deploy_profile")
    ap.add_argument("--tag", default="quant_head_audit")
    args = ap.parse_args()

    avail = mem_available_mb()
    if avail is not None and avail < MEM_FLOOR_MB:
        print(f"[guard] MemAvailable {avail} MB < {MEM_FLOOR_MB} MB -> refusing "
              f"to start (this box OOM'd once already).", file=sys.stderr)
        sys.exit(2)
    torch.set_num_threads(4)

    model, info = load_model(args.cfg, args.ckpt)
    print(f"[load] params={info['params']} missing={info.get('missing')} "
          f"unexpected={info.get('unexpected')} epoch={info.get('epoch')} "
          f"mem_avail={avail} MB")

    from datasets.bdd100k import BDD100KDataset
    ds = BDD100KDataset(args.data, split="tri_val", img_size=640,
                        with_det=False, with_da=False, with_lane=False)
    n = min(len(ds), max(args.n, 40))
    imgs = [ds[i]["image"] for i in range(n)]
    calib, ev = imgs[:30], imgs[30:]
    print(f"[data] {n} images ({len(calib)} calib / {len(ev)} eval)")

    def argmax_maps(outs):
        return [(o[1].argmax(1)[0].to(torch.uint8).numpy(),
                 o[2].argmax(1)[0].to(torch.uint8).numpy()) for o in outs]

    # ---- reference: calibrate inner quantisers on the calib slice
    # ---- reference: true FP32 (no quantisation at all) for the box reference
    s_fp = Sim(model, nbits=args.bits).install()
    fp = s_fp.run(ev, "off")
    s_fp.close()
    gc.collect()

    ref_det = torch.cat([o[0] for o in fp], 0)
    ref_ag = argmax_maps(fp)
    del fp
    gc.collect()

    variants = {
        "B_inner_only": dict(head_mode="none"),
        "C_inner_head_pertensor": dict(head_mode="per_tensor"),
        "D_inner_head_perchannel": dict(head_mode="per_channel"),
    }
    res = {"model": info, "bits": args.bits, "act_quant": args.act_quant,
           "n_calib": len(calib), "n_eval": len(ev), "mem_avail_mb": avail,
           "variants": {}, "_note": __doc__.strip().split("\n\n")[0]}

    only = [x for x in args.only.split(",") if x]
    for name, kw in variants.items():
        if only and name not in only:
            continue
        s = Sim(model, nbits=args.bits, observer="p9999",
                act_quant=(args.act_quant == "on"), **kw).install()
        s.run(calib, "calib")
        outs = s.run(ev, "apply")
        step = getattr(s, "last_head_step", None)
        ag = argmax_maps(outs)
        v = {"head_mode": kw["head_mode"]}
        # detection: aggregate the GT-free survival statistics over eval images
        acc = {}
        for i, o in enumerate(outs):
            st = det_stats(o[0][None] if o[0].dim() == 2 else o[0],
                           ref_det[i * 1: i * 1 + 1])
            for k, val in st.items():
                if val is None:
                    continue
                acc.setdefault(k, []).append(val)
        for k, vals in acc.items():
            v[k] = round(float(np.mean(vals)), 5)
        # segmentation control
        d_ag = [mask_stats(ref_ag[i][0], ag[i][0]) for i in range(len(ev))]
        l_ag = [mask_stats(ref_ag[i][1], ag[i][1]) for i in range(len(ev))]
        v["da_pixel_agreement"] = round(float(np.mean([x[0] for x in d_ag])), 5)
        v["da_fg_iou"] = round(float(np.nanmean([x[1] for x in d_ag])), 5)
        v["lane_pixel_agreement"] = round(float(np.mean([x[0] for x in l_ag])), 5)
        v["lane_fg_iou"] = round(float(np.nanmean([x[1] for x in l_ag])), 5)
        if step is not None:
            v["head_output_quant_step"] = round(step, 6)
            v["head_step_vs_obj_unit_interval"] = round(step, 6)
        res["variants"][name] = v
        s.close()
        del outs, ag
        gc.collect()
        print(f"[{name}] " + json.dumps(v, ensure_ascii=False))

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, args.tag + ".json")
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, out)
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()

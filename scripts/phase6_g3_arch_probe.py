#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6 / G3 -- second-architecture pre-audit (ZERO GPU, CPU only).

Implements the read-only checks required by the 跨架构验证预审 skill BEFORE any
training is scheduled:

  1. equivalence  -- the factory's default path must build the ORIGINAL encoder,
                     bit-for-bit.  Proven by state_dict keys + parameter count +
                     a seeded forward pass, not by an isinstance assertion.
  2. contract     -- the new topology must satisfy the interface
                     StaticMultiTaskModel actually indexes: exported channel
                     widths == cfg["stages"], resolutions 1/8-1/16-1/32, and the
                     highres branch must hand back f1 with stages[0] channels.
  3. budget match -- solve for the new-encoder width multiplier that lands the
                     TOTAL model parameter count on a target budget, using the
                     frozen ruler (profiling/flops_real.count_flops, 2*MACs at
                     640x640).  Targets are the arch-1 tiers so the two
                     architectures' spend amounts are comparable.
  4. opinventory  -- operator/call histogram of the new topology.  Counted by
                     wrapping F.conv2d / F.interpolate during a forward, NOT by
                     jit.trace (trace leaves CallMethod nodes un-inlined and
                     reports nothing useful).

IMPLEMENTATION NOTE (cost me a silent wrong table once): every construction of
StaticMultiTaskModel in this file must happen while sm.LightEncoder is swapped
for build_encoder, otherwise the model silently builds with the ORIGINAL
encoder and the printed "total params / FLOPs" describe the wrong architecture.
Hence the single `patched()` context manager below -- nothing builds outside it.

Nothing here trains, touches a checkpoint, or writes into experiments/*.csv.

Usage:
    python scripts/phase6_g3_arch_probe.py
Output: experiments/phase6/g3/g3_arch_probe.json  (+ _sweep.csv, + stdout)
"""
import argparse
import csv
import contextlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.encoder.light_encoder import LightEncoder  # noqa: E402
from models.encoder.factory import build_encoder  # noqa: E402
import models.static_model as sm  # noqa: E402

H = W = 640

# The exact reference encoder of the frozen Phase-6 R2 config
# (configs/phase6_r4_R2_thin14_z16.yaml).  Do not "tidy" these numbers.
REF_ENC = {"stem": 16, "stages": [32, 64, 96, 128], "blocks": [2, 2, 2]}

REF_MODEL = {
    "encoder": dict(REF_ENC),
    "representation": {"z_channels": 16},
    "detection": {"nc": 1, "from_z": True, "z_proj": True, "det_ch": 32},
    "segmentation": {"hidden": 32},
}

# arch-1 numbers these are anchored to, from experiments/phase3a/exp3A_encoder.csv
# and experiments/phase2d/expD_budget.csv (20ep, seed 0):
#
#   cell            params M   FLOPs G   mAP50    da_mIoU  lane_mIoU
#   E-base + z16      0.1889    1.0597   0.3204    0.8564    0.5867   <- anchor
#   E-base + z128     0.2860    2.0230   0.3224    0.8553    0.5879   <- spend on Z
#   E-large + z16     0.2915    1.4279   0.3585    0.8589    0.5878   <- spend on encoder
#
# The ledger's decisive test (phase3a_allocation_ledger.py section D) is a
# LIKE-FOR-LIKE comparison of the last two rows: nearly equal TOTAL params
# (+1.9%), 29.4% fewer FLOPs for the encoder-heavy cell, and it is >= on
# 6/6 metrics.  arch-2 must reproduce that *shape*, so the tiers are solved as a
# CHAIN rather than against absolute targets:
#
#   (1) base tier   : total(z=16) matched to the arch-1 anchor 0.189M
#   (2) Z-spend cell: SAME encoder, z=128   -> its total is measured, not assumed
#   (3) enc-spend   : z=16, encoder widened so its total matches (2)
#
# That is what makes the arch-2 pair "near-equal params" by construction instead
# of by luck.  A previous version of this file used a *spend amount* (0.379M) as
# the target for the encoder arm, which is wrong: it double-counts, because the
# encoder arm starts from the same base as the Z arm.
ANCHOR_TOTAL_M = 0.189
Z_SPEND = 128


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@contextlib.contextmanager
def patched():
    """Make StaticMultiTaskModel build the arch selected by cfg['arch'].

    This is exactly what the real one-line patch to models/static_model.py does
    (LightEncoder(...) -> build_encoder(...)); using it here means the probe
    measures the patched behaviour without editing the training path.
    """
    keep = sm.LightEncoder
    sm.LightEncoder = build_encoder
    try:
        yield
    finally:
        sm.LightEncoder = keep


def build_cfg(enc, z=16):
    return {
        "encoder": dict(enc),
        "representation": {"z_channels": z},
        "detection": {"nc": 1, "from_z": True, "z_proj": True, "det_ch": 32},
        "segmentation": {"hidden": 32},
    }


# --------------------------------------------------------------------------- 1
def check_equivalence():
    """Factory default path == LightEncoder, structurally and numerically."""
    out = {}
    torch.manual_seed(0)

    e_old = LightEncoder(REF_ENC).eval()
    e_new = build_encoder(REF_ENC).eval()
    out["encoder_class"] = type(e_new).__name__
    out["encoder_is_light"] = isinstance(e_new, LightEncoder)
    out["encoder_keys_equal"] = (list(e_old.state_dict()) ==
                                 list(e_new.state_dict()))
    out["encoder_params"] = [n_params(e_old), n_params(e_new)]
    out["encoder_params_equal"] = n_params(e_old) == n_params(e_new)

    e_new.load_state_dict(e_old.state_dict())
    x = torch.randn(1, 3, H, W)
    with torch.no_grad():
        a = e_old(x)
        b = e_new(x)
    out["encoder_forward_maxdiff"] = float(max(
        (u - v).abs().max().item() for u, v in zip(a, b)))

    with patched():
        torch.manual_seed(0)
        m_new = sm.StaticMultiTaskModel(REF_MODEL).eval()
    torch.manual_seed(0)
    m_old = sm.StaticMultiTaskModel(REF_MODEL).eval()
    out["model_keys_equal"] = (list(m_old.state_dict()) ==
                               list(m_new.state_dict()))
    out["model_params"] = [n_params(m_old), n_params(m_new)]
    out["model_params_equal"] = n_params(m_old) == n_params(m_new)
    m_new.load_state_dict(m_old.state_dict())
    with torch.no_grad():
        d0, a0, l0 = m_old(x)
        d1, a1, l1 = m_new(x)
    out["model_forward_maxdiff"] = float(max(
        (u - v).abs().max().item()
        for u, v in ((d0, d1), (a0, a1), (l0, l1))))
    out["PASS"] = bool(out["encoder_is_light"] and out["encoder_keys_equal"]
                       and out["encoder_params_equal"]
                       and out["encoder_forward_maxdiff"] == 0.0
                       and out["model_keys_equal"]
                       and out["model_params_equal"]
                       and out["model_forward_maxdiff"] == 0.0)
    return out


# --------------------------------------------------------------------------- 2
def check_contract(enc_cfg, z=16):
    """Exported widths/resolutions must match what StaticMultiTaskModel indexes."""
    res = {"encoder_cfg": enc_cfg, "z_channels": z}
    enc = build_encoder(enc_cfg).eval()
    stages = list(enc_cfg["stages"])
    x = torch.randn(1, 3, H, W)
    with torch.no_grad():
        f2, f3, f4 = enc(x)
        (g2, g3, g4), extra = enc(x, highres=True)
    res["shapes"] = {"F2": list(f2.shape), "F3": list(f3.shape),
                     "F4": list(f4.shape)}
    res["resolutions_ok"] = (f2.shape[-2:] == (H // 8, W // 8)
                             and f3.shape[-2:] == (H // 16, W // 16)
                             and f4.shape[-2:] == (H // 32, W // 32))
    res["channels_ok"] = ([f2.shape[1], f3.shape[1], f4.shape[1]] == stages[1:])
    res["highres_f1_channels"] = int(extra["f1"].shape[1])
    res["highres_f1_ok"] = (extra["f1"].shape[1] == stages[0]
                            and extra["f1"].shape[-2:] == (H // 4, W // 4))
    res["highres_f0_shape"] = list(extra["f0"].shape)
    res["fw_return_tuple_ok"] = (len((f2, f3, f4)) == 3)

    with patched():
        m = sm.StaticMultiTaskModel(build_cfg(enc_cfg, z)).eval()
    res["actual_encoder_class"] = type(m.encoder).__name__
    with torch.no_grad():
        det, da, lane = m(x)
    res["model_out"] = {"det": list(det.shape), "da": list(da.shape),
                        "lane": list(lane.shape)}
    res["model_build_ok"] = (res["actual_encoder_class"]
                             == type(enc).__name__)
    res["PASS"] = bool(res["resolutions_ok"] and res["channels_ok"]
                       and res["highres_f1_ok"] and res["model_build_ok"])
    return res


# --------------------------------------------------------------------------- 3
def scale_stages(stages, k):
    return [max(8, int(round(c * k / 8.0)) * 8) for c in stages]


def solve_chain(anchor_M=ANCHOR_TOTAL_M, t_grid=(2, 3, 4), k_lo=0.50, k_hi=1.60,
                k_step=0.01, min_growth=1.20):
    """Solve the arch-2 three-cell chain.  Returns (picks, grid).

    min_growth: the encoder-spend tier must be at least this multiple of the
    base tier's total params, otherwise "spend" would be a no-op.
    """
    from profiling.flops_real import count_flops
    base = [32, 64, 96, 128]
    grid = []
    with patched():
        for t in t_grid:
            for i in range(int(round((k_hi - k_lo) / k_step)) + 1):
                k = round(k_lo + i * k_step, 4)
                cfg = dict(arch="ir", stem=16, stages=scale_stages(base, k),
                           blocks=[2, 2, 2], expand=t)
                try:
                    m = sm.StaticMultiTaskModel(build_cfg(cfg, 16))
                    grid.append(dict(t=t, k=k, stages=cfg["stages"],
                                     total_params=n_params(m)))
                except Exception as e:  # noqa: BLE001
                    grid.append(dict(t=t, k=k, stages=cfg["stages"],
                                     total_params=None,
                                     error=f"{type(e).__name__}: {e}"))
    live = [r for r in grid if r["total_params"]]

    # (1) base tier -- closest total to the arch-1 anchor, at z = 16
    b = min(live, key=lambda r: abs(r["total_params"] / 1e6 - anchor_M))

    def cfg_of(r, z, blocks=(2, 2, 2)):
        return dict(arch="ir", stem=16, stages=r["stages"], blocks=list(blocks),
                    expand=r["t"])

    # (2) Z-spend cell -- identical encoder, z = 128.  MEASURED.
    with patched():
        m_z = sm.StaticMultiTaskModel(build_cfg(cfg_of(b, Z_SPEND), Z_SPEND))
    z_total = n_params(m_z)

    # (3) encoder-spend tier -- wider encoder, z = 16, total matched to (2)
    cand = [r for r in live
            if (r["t"], r["k"]) != (b["t"], b["k"])
            and r["total_params"] >= b["total_params"] * min_growth]
    e = min(cand, key=lambda r: abs(r["total_params"] - z_total))

    x = torch.randn(1, 3, H, W)
    picks = []
    for name, r, z, note in (
            ("arch2_base_z16", b, 16, "anchor: matched to arch-1 E-base+z16"),
            ("arch2_zspend_z128", b, Z_SPEND,
             "same encoder as base, larger Z (MEASURED)"),
            ("arch2_encspend_z16", e, 16,
             "wider encoder, z16, total matched to zspend")):
        cfg = cfg_of(r, z)
        torch.manual_seed(0)
        enc = build_encoder(cfg).eval()
        with patched():
            m = sm.StaticMultiTaskModel(build_cfg(cfg, z)).eval()
        with torch.no_grad():
            f = count_flops(m, x)
        picks.append(dict(tier=name, note=note, cfg=cfg, stages=r["stages"],
                          expand=r["t"], k=r["k"], z=z,
                          enc_params=n_params(enc), total_params=n_params(m),
                          total_params_M=round(n_params(m) / 1e6, 4),
                          flops_G=round(f / 1e9, 4)))
    return picks, live


# --------------------------------------------------------------------------- 4
def opinventory(enc_cfg):
    """Function-level op counts: robust, unlike jit.trace on a module tree."""
    enc = build_encoder(enc_cfg).eval()
    x = torch.randn(1, 3, H, W)
    c = {"conv2d_total": 0, "conv2d_pointwise_1x1": 0,
         "conv2d_depthwise": 0, "interpolate": 0, "relu": 0, "macs": 0}
    o_conv, o_interp, o_relu = F.conv2d, F.interpolate, F.relu

    def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1,
               groups=1):
        out = o_conv(input, weight, bias, stride, padding, dilation, groups)
        k = weight.shape[2] * weight.shape[3]
        c["conv2d_total"] += 1
        if groups > 1 and groups == weight.shape[0] and groups == input.shape[1]:
            c["conv2d_depthwise"] += 1
        elif k == 1:
            c["conv2d_pointwise_1x1"] += 1
        c["macs"] += (input.shape[0] * out.shape[2] * out.shape[3]
                      * weight.shape[0] * (input.shape[1] // groups) * k)
        return out

    def interpolate(inp, *a, **kw):
        c["interpolate"] += 1
        return o_interp(inp, *a, **kw)

    def relu(inp, *a, **kw):
        c["relu"] += 1
        return o_relu(inp, *a, **kw)

    F.conv2d, F.interpolate, F.relu = conv2d, interpolate, relu
    try:
        with torch.no_grad():
            enc(x)
    finally:
        F.conv2d, F.interpolate, F.relu = o_conv, o_interp, o_relu

    mods = {}
    for m in enc.modules():
        mods[type(m).__name__] = mods.get(type(m).__name__, 0) + 1
    c["module_types"] = dict(sorted(mods.items(), key=lambda kv: -kv[1]))
    c["mn_macs"] = round(c["macs"] / 1e6, 3)
    c["gflops_2xmacs"] = round(2 * c["macs"] / 1e9, 4)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="experiments/phase6/g3")
    args = ap.parse_args()

    torch.set_num_threads(2)
    out = {"reference_encoder_cfg": REF_ENC, "reference_model_cfg": REF_MODEL,
           "anchor_total_M": ANCHOR_TOTAL_M, "z_spend": Z_SPEND}

    print("=" * 80)
    print("1) EQUIVALENCE -- factory default path vs the original encoder")
    print("=" * 80)
    eq = check_equivalence()
    for k, v in eq.items():
        print(f"  {k:26s} {v}")
    out["equivalence"] = eq
    assert eq["PASS"], "equivalence FAILED -- do not proceed"

    print()
    print("=" * 80)
    print("2) CONTRACT -- interface expected by StaticMultiTaskModel")
    print("=" * 80)
    cand = dict(arch="ir", stem=16, stages=REF_ENC["stages"],
                blocks=REF_ENC["blocks"], expand=2)
    ct = check_contract(cand)
    for k, v in ct.items():
        print(f"  {k:24s} {v}")
    out["contract"] = ct
    assert ct["PASS"], "contract FAILED -- do not proceed"

    print()
    print("=" * 80)
    print("3) TIER CHAIN -- solve the arch-2 cells so the pair is near-equal")
    print("=" * 80)
    picks, grid = solve_chain()
    print(f"  searched {len(grid)} (t,k) points, "
          f"{len([r for r in grid if r['total_params']])} built OK")
    for p in picks:
        print(f"  {p['tier']:20s} t={p['expand']} k={p['k']} "
              f"stages={str(p['stages']):24s} z={p['z']:3d} "
              f"encoder={p['enc_params']:7d} total={p['total_params_M']:.4f}M "
              f"FLOPs={p['flops_G']:.4f}G")
        print(f"  {'':20s} {p['note']}")
    if len(picks) == 3:
        pz, pe = picks[1], picks[2]
        dp = (pe["total_params_M"] - pz["total_params_M"]) / pz["total_params_M"]
        df = (pe["flops_G"] - pz["flops_G"]) / pz["flops_G"]
        print(f"\n  LIKE-FOR-LIKE pair (arch-2): z-spend vs enc-spend")
        print(f"    params : {pz['total_params_M']:.4f}M vs "
              f"{pe['total_params_M']:.4f}M  ({dp:+.1%})")
        print(f"    FLOPs  : {pz['flops_G']:.4f}G vs {pe['flops_G']:.4f}G  "
              f"({df:+.1%})")
        ok = abs(dp) <= 0.05
        print(f"    within the +/-5% budget-equivalence tolerance: {ok}")
        out["like_for_like_arch2"] = {"params_rel": dp, "flops_rel": df,
                                      "within_5pct": ok}
    out["budget_picks"] = picks
    out["budget_grid"] = [r for r in grid if r["total_params"]]

    # reference on the same ruler, measured now
    from profiling.flops_real import count_flops
    x = torch.randn(1, 3, H, W)
    torch.manual_seed(0)
    m_ref = sm.StaticMultiTaskModel(REF_MODEL).eval()
    with torch.no_grad():
        f_ref = count_flops(m_ref, x)
    ref = dict(tag="REFERENCE_dws_R2thin14", stages=REF_ENC["stages"],
               enc_params=n_params(LightEncoder(REF_ENC)),
               total_params=n_params(m_ref),
               total_params_M=round(n_params(m_ref) / 1e6, 4),
               flops_G=round(f_ref / 1e9, 4))
    print(f"\n  {'REFERENCE(dws)':20s} encoder={ref['enc_params']} "
          f"total={ref['total_params_M']:.4f}M FLOPs={ref['flops_G']:.4f}G")
    out["reference_measured"] = ref

    print()
    print("=" * 80)
    print("4) OP INVENTORY -- new encoder vs original (function-level counts)")
    print("=" * 80)
    inv_new = opinventory(cand)
    inv_old = opinventory(REF_ENC)
    for tag, inv in (("ir", inv_new), ("dws", inv_old)):
        print(f"  {tag:4s} conv={inv['conv2d_total']:3d} "
              f"(1x1 {inv['conv2d_pointwise_1x1']:3d}, "
              f"dw {inv['conv2d_depthwise']:3d})  "
              f"resize={inv['interpolate']} relu={inv['relu']}  "
              f"encoder FLOPs={inv['gflops_2xmacs']:.4f}G")
    print(f"       ir  modules: {inv_new['module_types']}")
    print(f"       dws modules: {inv_old['module_types']}")
    out["op_inventory"] = {"ir": inv_new, "dws": inv_old}

    os.makedirs(os.path.join(ROOT, args.outdir), exist_ok=True)
    p = os.path.join(ROOT, args.outdir, "g3_arch_probe.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)

    csvp = os.path.join(ROOT, args.outdir, "g3_arch_probe_sweep.csv")
    with open(csvp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "expand", "k", "stages", "z", "enc_params",
                    "total_params", "total_params_M", "flops_G", "note"])
        for p2 in picks:
            w.writerow([p2["tier"], p2["expand"], p2["k"], p2["stages"],
                        p2["z"], p2["enc_params"], p2["total_params"],
                        p2["total_params_M"], p2["flops_G"], p2["note"]])
        w.writerow(["REFERENCE_dws_R2thin14", "", "", REF_ENC["stages"], 16,
                    ref["enc_params"], ref["total_params"],
                    ref["total_params_M"], ref["flops_G"], "arch-1 base"])
    print(f"\n[wrote] {p}\n[wrote] {csvp}")


if __name__ == "__main__":
    main()

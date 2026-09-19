#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6 / G3 -- solve BOTH architectures' three-tier chains and emit configs.

WHY THIS EXISTS (read before trusting any cross-architecture number)
-------------------------------------------------------------------
Two confounds were found while auditing the arch-1 allocation ledger
(experiments/phase3a/exp3A_encoder.csv against experiments/phase2d/expD_budget.csv):

  CONFOUND 1 -- supervision regime.  DEFAULT_ANCHORS_3S became the IoU-k-means
  set on 2026-09-08 23:16 (commit c8aca35).  Configs that omit an `anchors:` key
  silently inherit whatever the default was ON THE DAY THEY RAN.  Reading the
  run-start time off each run's config.yaml mtime (train.py writes it at start):

      expD_z16_e20      2026-09-04 23:34   pre-change  -> OLD anchors
      expD_z128_e20     2026-09-05 06:42   pre-change  -> OLD anchors
      exp3A_esmall_z16  2026-09-05 11:12   pre-change  -> OLD anchors
      exp3A_elarge_z16  2026-09-05 13:10   pre-change  -> OLD anchors
      exp4A_r2_z*       2026-09-07         pre-change  -> OLD anchors
      B100 / B100_s1/s2 2026-09-13+        post-change -> k-means

  So the ledger is internally anchor-consistent, but it lives entirely in the
  regime the code itself documents as supervision-handicapped, while the paper's
  headline table lives in the k-means regime.  The measured size of that regime
  shift, on an identical model at identical 20ep (R4-A0lean_old vs
  R4-A0lean_km, both 0.2014M / 1.0796G), is +0.148 mAP50 -- an order of
  magnitude larger than the +0.036 the ledger's encoder arm claims.

  CONFOUND 2 -- detection topology.  The ledger's cells (phase2b/2d/3a configs)
  declare `detection: {nc: 1}` only, i.e. detection reads the encoder's
  multi-scale F2/F3/F4 and BYPASSES Z (the "R0" head).  The current model uses
  DetFromZ, where all three tasks flow through the bottleneck (the "R2" head).
  The question "spend on the encoder or on Z?" is only well posed when the tasks
  actually compete for Z, so an R0-based ledger cannot answer it for R2.

Consequence: G3 must not compare a fresh arch-2 number against the historical
arch-1 ledger.  Both architectures are re-measured here under ONE configuration
(R2 detection from Z, lean segmentation, explicit k-means anchors, 20ep, seed 0),
so the encoder topology is the only variable left.

The tier chain (per architecture), solved with the frozen FLOPs ruler
(profiling/flops_real.count_flops, 2*MACs @ 640x640):

  (1) base      : z=16, smallest encoder tier
  (2) z-spend   : SAME encoder, z=128            <- total MEASURED, not assumed
  (3) enc-spend : z=16, larger encoder, total matched to (2)

(3) is matched to (2) because the ledger's decisive test is a LIKE-FOR-LIKE
comparison at near-equal total params: if the two cells differ in budget the
"allocation" question is not identified.

Usage:
    python scripts/phase6_g3_tier_solve.py --emit
Output: configs/phase6_g3_*.yaml, experiments/phase6/g3/g3_tier_solve.json
"""
import argparse
import contextlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from models.encoder.factory import build_encoder  # noqa: E402
import models.static_model as sm  # noqa: E402


@contextlib.contextmanager
def patched():
    """Make StaticMultiTaskModel honour cfg['encoder']['arch'].

    TRAP (hit twice): StaticMultiTaskModel calls `LightEncoder(enc_cfg)` on a
    module-level name.  Constructing it WITHOUT this context silently builds the
    ORIGINAL encoder, so every arch-2 row reports arch-1's params and FLOPs --
    and the width search appears to work while varying a knob that does nothing
    (the totals come out identical for every value of `expand`).  Every model
    construction in this file goes through here.
    """
    keep = sm.LightEncoder
    sm.LightEncoder = build_encoder
    try:
        yield
    finally:
        sm.LightEncoder = keep

H = W = 640

# Frozen k-means anchor set == DEFAULT_ANCHORS_3S (models/representation/det_from_z.py).
# Written EXPLICITLY into every G3 config so the cell cannot be re-pointed at a
# different default by a later code change -- that is exactly how confound 1 arose.
KM_ANCHORS = [[[9, 8], [18, 15], [32, 24]],
              [[49, 38], [80, 52], [65, 102]],
              [[124, 82], [166, 136], [237, 214]]]

CFG_TEMPLATE = """\
# Phase 6 / G3 -- second-architecture transfer test.  GENERATED FILE.
#
# Emitted by scripts/phase6_g3_tier_solve.py; edit that script, not this file.
# Cell: {tier}   architecture: {arch_desc}
# Solved geometry (measured on the frozen FLOPs ruler at 640x640, batch 1):
#   encoder params {enc_params}   total params {total_params}   FLOPs {flops_G} G
#
# Held FIXED across every G3 cell so the encoder topology is the only variable:
#   - detection reads Compact Z (from_z), z_proj on
#   - segmentation is lean (hidden: 32)
#   - anchors are written EXPLICITLY (k-means).  A config that omits this key
#     inherits whatever the code default is on the day it runs -- that silent
#     dependency is what put the old allocation ledger in a different
#     supervision regime from the paper's main table.
model:
  encoder:
{encoder_yaml}
  representation:
    z_channels: {z}
  detection:
    nc: 1
    from_z: true
    z_proj: true
    det_ch: 32
    anchors:
{anchors_yaml}
  segmentation:
    hidden: 32
input_size: [640, 640]
train:
  stage: A
  train_split: tri_train
  batch_size: 16
  num_workers: 2
  epochs: {epochs}
  lr: 1e-3
  weight_decay: 5e-4
  lambda_da: 1.0
  lambda_lane: 1.0
  lambda_budget: 0.0
  grad_clip: 10.0
data:
  root: data/bdd100k
"""


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def build_cfg(enc, z):
    return {"encoder": dict(enc), "representation": {"z_channels": z},
            "detection": {"nc": 1, "from_z": True, "z_proj": True,
                          "det_ch": 32, "anchors": KM_ANCHORS},
            "segmentation": {"hidden": 32}}


def build_model(enc, z):
    """Build + ASSERT the requested encoder topology is the one that got built.

    The assertion is the point: params/FLOPs for the wrong architecture are
    perfectly plausible numbers, so nothing downstream would notice.
    """
    with patched():
        m = sm.StaticMultiTaskModel(build_cfg(enc, z))
    want = "InvertedResidualEncoder" if enc.get("arch") == "ir" else "LightEncoder"
    got = type(m.encoder).__name__
    if got != want:
        raise AssertionError(
            f"encoder dispatch failed: wanted {want}, built {got} "
            f"(cfg arch={enc.get('arch')!r})")
    return m


def params_of(enc, z):
    return n_params(build_model(enc, z))


def dws_enc(stem, stages, blocks=(2, 2, 2)):
    return {"stem": stem, "stages": list(stages), "blocks": list(blocks)}


def ir_enc(stem, stages, expand, blocks=(2, 2, 2)):
    return {"arch": "ir", "stem": stem, "stages": list(stages),
            "blocks": list(blocks), "expand": expand}


# arch-1 encoder tiers, exactly as used in Phase 3A (exp3A_encoder.csv)
A1_ESMALL = dws_enc(12, [24, 40, 64, 80])
A1_EBASE = dws_enc(16, [32, 64, 96, 128])
A1_ELARGE = dws_enc(20, [40, 80, 128, 168])


def solve_arch1():
    """Chain on the original dws encoder."""
    base = A1_EBASE
    p_base = params_of(base, 16)
    p_zspend = params_of(base, 128)          # MEASURED
    # encoder tier whose z=16 total matches the z-spend total
    cands = []
    for stem in (16, 20, 24, 28):
        for k in [x / 100 for x in range(80, 181, 2)]:
            st = [max(8, int(round(c * k / 8.0)) * 8) for c in [32, 64, 96, 128]]
            e = dws_enc(stem, st)
            cands.append((e, params_of(e, 16)))
    # keep only real capacity increases over base
    cands = [(e, p) for e, p in cands if p >= p_base * 1.15]
    enc, p_enc = min(cands, key=lambda ep: abs(ep[1] - p_zspend))
    return dict(arch="dws", base=base, zspend=dict(base), encspend=enc,
                p_base=p_base, p_zspend=p_zspend, p_encspend=p_enc)


def solve_arch2():
    """Chain on the new inverted-residual encoder.  The base tier is matched to
    arch-1's base tier total params so the two architectures start from the same
    budget (the ledger normalises per 0.01M, but a matched base keeps the
    like-for-like pair honest)."""
    anchor = params_of(A1_EBASE, 16)
    cands = []
    for t in (2, 3, 4):
        for k in [x / 100 for x in range(55, 161, 1)]:
            st = [max(8, int(round(c * k / 8.0)) * 8) for c in [32, 64, 96, 128]]
            e = ir_enc(16, st, t)
            cands.append((e, params_of(e, 16)))
    base, p_base = min(cands, key=lambda ep: abs(ep[1] - anchor))
    p_zspend = params_of(base, 128)          # MEASURED
    cands2 = [(e, p) for e, p in cands
              if p >= p_base * 1.15 and e != base]
    enc, p_enc = min(cands2, key=lambda ep: abs(ep[1] - p_zspend))
    return dict(arch="ir", base=base, zspend=dict(base), encspend=enc,
                p_base=p_base, p_zspend=p_zspend, p_encspend=p_enc)


def encoder_yaml(enc):
    lines = []
    order = ["arch", "stem", "stages", "blocks", "expand"]
    for k in order:
        if k not in enc:
            continue
        v = enc[k]
        if isinstance(v, list):
            lines.append(f"    {k}: [{', '.join(str(x) for x in v)}]")
        else:
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def anchors_yaml():
    return "\n".join(
        "      - [" + ", ".join(f"[{a}, {b}]" for a, b in scale) + "]"
        for scale in KM_ANCHORS)


def verify_emitted(cells, epochs):
    """Read the emitted configs back and confirm they build the intended model.

    Same readback discipline as the Round-3/Round-4 CSV appenders: never trust
    a string you just wrote.  Also asserts the encoder CLASS, because a config
    whose `arch` did not take effect would produce plausible numbers.
    """
    import yaml
    from profiling.flops_real import count_flops
    x = torch.randn(1, 3, H, W)
    bad = []
    for c in cells:
        p = os.path.join(ROOT, "configs", c["name"] + ".yaml")
        with open(p) as fh:
            cfg = yaml.safe_load(fh)
        mcfg = cfg["model"]
        want_arch = c["enc"].get("arch", "dws")
        want_cls = ("InvertedResidualEncoder" if want_arch == "ir"
                    else "LightEncoder")
        got_cls = type(build_encoder(mcfg["encoder"])).__name__
        m = sm.StaticMultiTaskModel(mcfg).eval()
        got_cls_model = type(m.encoder).__name__
        with torch.no_grad():
            f = count_flops(m, x)
        n = n_params(m)
        checks = {
            "encoder_class": got_cls == want_cls,
            "model_encoder_class": got_cls_model == want_cls,
            "params": n == c["total_params"],
            "flops": abs(f / 1e9 - c["flops_G"]) < 1e-3,
            "z": mcfg["representation"]["z_channels"] == c["z"],
            "epochs": cfg["train"]["epochs"] == epochs,
            "anchors": mcfg["detection"]["anchors"] == KM_ANCHORS,
        }
        ok = all(checks.values())
        print(f"  {c['name']:28s} {'OK ' if ok else 'FAIL'} "
              f"cls={got_cls_model} params={n} FLOPs={f/1e9:.4f}G")
        if not ok:
            print(f"      failing checks: "
                  f"{[k for k, v in checks.items() if not v]}")
            bad.append((c["name"], checks))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true", help="write the config files")
    ap.add_argument("--verify", action="store_true",
                    help="read the emitted configs back and check them")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    torch.set_num_threads(2)
    from profiling.flops_real import count_flops
    x = torch.randn(1, 3, H, W)

    a1 = solve_arch1()
    a2 = solve_arch2()

    cells = []
    for tag, sol in (("a1", a1), ("a2", a2)):
        for tier, enc, z, desc in (
                ("base", sol["base"], 16, "smallest encoder tier"),
                ("zspend", sol["zspend"], 128, "same encoder, larger Z"),
                ("encspend", sol["encspend"], 16,
                 "larger encoder, matched to the z-spend total")):
            m = build_model(enc, z).eval()
            with torch.no_grad():
                f = count_flops(m, x)
            cells.append(dict(name=f"phase6_g3_{tag}_{tier}_z{z}", arch=tag,
                              tier=tier,
                              enc=enc, z=z, desc=desc,
                              enc_params=n_params(build_encoder(enc)),
                              total_params=n_params(m),
                              total_params_M=round(n_params(m) / 1e6, 4),
                              flops_G=round(f / 1e9, 4)))

    print("=" * 96)
    print("G3 TIER CHAIN -- one configuration (R2 det + lean seg + explicit k-means "
          "anchors), two encoder topologies")
    print("=" * 96)
    print(f"{'cell':24s} {'arch':5s} {'tier':9s} {'z':>4s} "
          f"{'enc params':>11s} {'total':>9s} {'FLOPs':>8s}   encoder cfg")
    for c in cells:
        print(f"{c['name']:24s} {c['arch']:5s} {c['tier']:9s} {c['z']:>4d} "
              f"{c['enc_params']:>11d} {c['total_params_M']:>8.4f}M "
              f"{c['flops_G']:>7.4f}G   {c['enc']}")

    print()
    for tag in ("a1", "a2"):
        cs = [c for c in cells if c["arch"] == tag]
        b, z, e = cs[0], cs[1], cs[2]
        dp = (e["total_params_M"] - z["total_params_M"]) / z["total_params_M"]
        df = (e["flops_G"] - z["flops_G"]) / z["flops_G"]
        print(f"[{tag}] LIKE-FOR-LIKE  z-spend {z['total_params_M']:.4f}M/"
              f"{z['flops_G']:.4f}G  vs  enc-spend {e['total_params_M']:.4f}M/"
              f"{e['flops_G']:.4f}G")
        print(f"      params {dp:+.1%}   FLOPs {df:+.1%}   "
              f"within +/-5%: {abs(dp) <= 0.05}")
        print()

    os.makedirs(os.path.join(ROOT, "experiments", "phase6", "g3"), exist_ok=True)
    with open(os.path.join(ROOT, "experiments", "phase6", "g3",
                           "g3_tier_solve.json"), "w") as fh:
        json.dump({"anchors": KM_ANCHORS, "cells": cells,
                   "arch1_solution": {k: v for k, v in a1.items()},
                   "arch2_solution": {k: v for k, v in a2.items()}},
                  fh, indent=2)

    if args.emit:
        written = []
        for c in cells:
            path = os.path.join(ROOT, "configs", c["name"] + ".yaml")
            with open(path, "w") as fh:
                fh.write(CFG_TEMPLATE.format(
                    tier=c["tier"], arch_desc=c["arch"] + " encoder",
                    enc_params=c["enc_params"],
                    total_params=c["total_params"],
                    flops_G=c["flops_G"],
                    encoder_yaml=encoder_yaml(c["enc"]),
                    z=c["z"], anchors_yaml=anchors_yaml(), epochs=args.epochs))
            written.append(path)
        print(f"[emit] wrote {len(written)} configs:")
        for w in written:
            print("   ", os.path.relpath(w, ROOT))

    if args.verify:
        print()
        print("[verify] reading the emitted configs back")
        bad = verify_emitted(cells, args.epochs)
        if bad:
            raise SystemExit(f"CONFIG VERIFY FAILED: {bad}")
        print("[verify] all configs build the intended architecture")


if __name__ == "__main__":
    main()

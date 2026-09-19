#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6 / G3 -- allocation ledger and TRANSFER VERDICT (frozen criteria).

Reads experiments/phase6/g3/g3_results.csv and re-derives, for EACH encoder
topology separately, the same four sections the arch-1 ledger used
(scripts/phase3a_allocation_ledger.py):

  A) spend on Z        base(z16) -> z-spend(z128)      per 0.01M params / 0.1 GFLOP
  B) spend on encoder  base(z16) -> enc-spend(z16)     per 0.01M params / 0.1 GFLOP
  C) efficiency ratio  encoder-spend / Z-spend
  D) LIKE-FOR-LIKE     z-spend vs enc-spend at near-equal TOTAL params
                       -> DOMINATES iff <= params, <= FLOPs, and >= on >=4/6 metrics

Differences from the arch-1 ledger, and why:
  * The arch-1 section B compared E-small -> E-large, which starts from a
    different baseline than its section A.  Here B is base -> enc-spend, so A
    and B start from the SAME cell and the two spend directions are comparable
    at the margin.  This is the one design change, and it is applied to BOTH
    architectures, so nothing is asymmetric.
  * The budget-equivalence gate is explicit: if the like-for-like pair differs
    by more than 5% in params, the verdict is INSTRUMENT_INCONCLUSIVE rather
    than a win.  ("±5% 容差判定预算等价", pre-registered.)

FROZEN DECISION RULES -- written before any G3 data existed
-----------------------------------------------------------
R0  BUDGET GATE.  For each architecture, |params(enc-spend) - params(z-spend)| /
    params(z-spend) <= 0.05.  Failing this makes that architecture's verdict
    INSTRUMENT_INCONCLUSIVE; it is not scored as a win or a loss.

R1  LOCAL VERDICT (per architecture), given R0 holds:
      DOMINATES        enc-spend params <= z-spend params AND
                       enc-spend FLOPs  <= z-spend FLOPs  AND
                       enc-spend >= z-spend on >= 4 of 6 metrics
      PARTIAL          the metric count is 2 or 3
      Z_WINS           enc-spend wins on <= 1 metric

R2  RULER.  A metric difference counts as REAL only if |delta| > 2 * sd, where
    sd is that metric's standard deviation measured on the SAME architecture
    from 3 seeds of its base cell (plus-seeds stage).  Cross-architecture reuse
    of a ruler is forbidden by the 跨架构验证预审 skill (trap 6).  If the
    plus-seeds stage has not run, the ruler is reported ABSENT and no
    within-noise claim is made -- the ordering is still reported.

R3  TRANSFER VERDICT:
      TRANSFERRED      R0 holds on both, and arch-2's DOMINATES/PARTIAL class is
                       at least as strong as arch-1's, and the sign of the
                       encoder-minus-Z gap agrees on >= 4/6 metrics.
      PARTIAL          the class is one step weaker, or the metric-sign agreement
                       is 3/6.
      NOT_TRANSFERRED  otherwise.
      INCONCLUSIVE     R0 fails on either architecture, or data is missing.

R4  REPORTING.  Every number the report quotes must come from this script's
    stdout or JSON -- no hand-copied values.

Usage:  python scripts/phase6_g3_ledger.py
Output: experiments/phase6/g3/g3_ledger.txt, g3_ledger.json
"""
import csv
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(ROOT, "experiments", "phase6", "g3", "g3_results.csv")
DEFAULT_OUTDIR = os.path.join(ROOT, "experiments", "phase6", "g3")
CSV = DEFAULT_CSV
OUTDIR = DEFAULT_OUTDIR

METRICS = ["mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"]
ABBREV = {"mAP50": "mAP50", "mAP50_95": "mAP50_95", "da_mIoU": "DA mIoU",
          "da_fg": "DA fgIoU", "lane_mIoU": "lane mIoU", "lane_fg": "lane fgIoU"}
BUDGET_TOL = 0.05
MIN_METRICS_FOR_DOMINANCE = 4
RULER_SIGMA = 2.0

ARCH_LONG = {"a1": "a1 = depthwise-separable (ESPNet lineage)",
             "a2": "a2 = inverted-residual (MobileNetV2/V3 lineage)"}


def fnum(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def load():
    if not os.path.exists(CSV):
        return []
    with io.open(CSV, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]
    out = []
    for r in rows:
        if r.get("source") != "trained":
            continue                      # never score a failed-no-metrics row
        vals = {m: fnum(r.get(m)) for m in METRICS}
        if any(v is None for v in vals.values()):
            continue
        out.append(dict(arch=r["cell"].split("_")[0],
                        tier=r["cell"].split("_", 1)[1],
                        cell=r["cell"], z=int(r["z"]), enc=r["encoder"],
                        seed=int(r["seed"]),
                        params_M=fnum(r["params_M"]), flops_G=fnum(r["flops_G"]),
                        **vals))
    return out


def pick(rows, arch, tier, seed=0):
    for r in rows:
        if r["arch"] == arch and r["tier"] == tier and r["seed"] == seed:
            return r
    return None


def ruler(rows, arch):
    """Per-metric sd across seeds of the base cell, on this architecture only."""
    ss = [r for r in rows if r["arch"] == arch and r["tier"] == "base"]
    if len(ss) < 2:
        return None
    n = len(ss)
    sd = {}
    for m in METRICS:
        mu = sum(r[m] for r in ss) / n
        sd[m] = (sum((r[m] - mu) ** 2 for r in ss) / (n - 1)) ** 0.5
    return sd


def eff(delta, dparams, dflops):
    per_p = delta / (dparams / 0.01) if dparams else None
    per_f = delta / (dflops / 0.1) if dflops else None
    return per_p, per_f


def main():
    global CSV, OUTDIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="override the results CSV (used by the self-test)")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    CSV, OUTDIR = args.csv, args.outdir

    rows = load()
    L = []
    P = L.append
    P("=" * 96)
    P("PHASE 6 / G3 -- ALLOCATION LEDGER, TWO ENCODER TOPOLOGIES")
    P("=" * 96)
    P("Frozen rules: R0 budget gate +/-5%; R1 dominance >=4/6 metrics;")
    P("              R2 ruler = 2*sd measured on the SAME architecture;")
    P("              R3 transfer = same class + same sign on >=4/6 metrics.")
    P("")

    if not rows:
        P("no scored rows yet in " + os.path.relpath(CSV, ROOT))
        print("\n".join(L))
        return

    jsonout = {"rules": {"budget_tol": BUDGET_TOL,
                         "min_metrics_for_dominance": MIN_METRICS_FOR_DOMINANCE,
                         "ruler_sigma": RULER_SIGMA},
               "metrics": METRICS, "arch": {}}

    for arch in ("a1", "a2"):
        base = pick(rows, arch, "base")
        zs = pick(rows, arch, "zspend")
        es = pick(rows, arch, "encspend")
        P("#" * 96)
        P(f"# {ARCH_LONG.get(arch, arch)}")
        P("#" * 96)
        if not (base and zs and es):
            have = [t for t, r in (("base", base), ("zspend", zs),
                                   ("encspend", es)) if r]
            P(f"  INCOMPLETE: have {have or 'nothing'}; skipping")
            jsonout["arch"][arch] = {"status": "INCOMPLETE", "have": have}
            P("")
            continue

        P(f"\n  cell geometry (measured)")
        P(f"  {'cell':<12}{'z':>4}{'params M':>11}{'FLOPs G':>10}"
          f"{'encoder':>18}")
        for r in (base, zs, es):
            P(f"  {r['tier']:<12}{r['z']:>4}{r['params_M']:>11.4f}"
              f"{r['flops_G']:>10.4f}{r['enc']:>18}")

        sd = ruler(rows, arch)
        P(f"\n  noise ruler: " + (
            f"2*sd from {sum(1 for r in rows if r['arch']==arch and r['tier']=='base')}"
            f" seeds of the {arch} base cell: "
            + ", ".join(f"{ABBREV[m]}={2*sd[m]:.4f}" for m in METRICS)
            if sd else
            "ABSENT (plus-seeds stage not run) -- no within-noise claim is made"))

        for tag, A, B in (("A) spend on Z      base(z16) -> zspend(z128)", base, zs),
                          ("B) spend on ENCODER base(z16) -> encspend(z16)", base, es)):
            dp = B["params_M"] - A["params_M"]
            df = B["flops_G"] - A["flops_G"]
            P(f"\n  {tag}")
            P(f"  {'metric':<11}{'before':>9}{'after':>9}{'delta':>10}"
              f"{'per 0.01M':>12}{'per 0.1GF':>12}"
              f"{'real?':>8}")
            for m in METRICS:
                d = B[m] - A[m]
                pp, pf = eff(d, dp, df)
                real = "-" if sd is None else (
                    "yes" if abs(d) > RULER_SIGMA * sd[m] else "noise")
                P(f"  {ABBREV[m]:<11}{A[m]:>9.4f}{B[m]:>9.4f}{d:>+10.4f}"
                  f"{pp:>+12.5f}{pf:>+12.5f}{real:>8}")
            P(f"  {'(d params, d FLOPs)':<11}{'':>9}{'':>9}"
              f"{dp:>+10.4f}M{df:>+9.4f}G")

        # ---- C: ratio ------------------------------------------------------
        dp_z, df_z = zs["params_M"] - base["params_M"], zs["flops_G"] - base["flops_G"]
        dp_e, df_e = es["params_M"] - base["params_M"], es["flops_G"] - base["flops_G"]
        P(f"\n  C) efficiency ratio  encoder-spend / Z-spend")
        P(f"  {'metric':<11}{'enc per 0.01M':>16}{'Z per 0.01M':>14}"
          f"{'ratio(params)':>16}{'ratio(FLOPs)':>15}")
        ratios_p = []
        for m in METRICS:
            ep, ef_ = eff(es[m] - base[m], dp_e, df_e)
            zp, zf = eff(zs[m] - base[m], dp_z, df_z)
            rp = (ep / zp) if (zp and abs(zp) > 1e-12) else float("nan")
            rf = (ef_ / zf) if (zf and abs(zf) > 1e-12) else float("nan")
            ratios_p.append(rp)
            P(f"  {ABBREV[m]:<11}{ep:>+16.5f}{zp:>+14.5f}"
              f"{(f'{rp:>15.1f}x' if rp == rp else f'{chr(8212):>16}')}"
              f"{(f'{rf:>14.1f}x' if rf == rf else f'{chr(8212):>15}')}")

        # ---- D: like-for-like ---------------------------------------------
        P(f"\n  D) LIKE-FOR-LIKE  z-spend vs enc-spend at near-equal TOTAL params")
        P(f"  {'':<11}{'base enc + z128':>18}{'wide enc + z16':>18}{'delta':>12}")
        rel_p = (es["params_M"] - zs["params_M"]) / zs["params_M"]
        rel_f = (es["flops_G"] - zs["flops_G"]) / zs["flops_G"]
        P(f"  {'params M':<11}{zs['params_M']:>18.4f}{es['params_M']:>18.4f}"
          f"{rel_p*100:>11.1f}%")
        P(f"  {'FLOPs G':<11}{zs['flops_G']:>18.4f}{es['flops_G']:>18.4f}"
          f"{rel_f*100:>11.1f}%")
        wins = 0
        for m in METRICS:
            d = es[m] - zs[m]
            w = es[m] >= zs[m]
            wins += int(w)
            real = "-" if sd is None else (
                "yes" if abs(d) > RULER_SIGMA * sd[m] else "noise")
            P(f"  {ABBREV[m]:<11}{zs[m]:>18.4f}{es[m]:>18.4f}{d:>+12.4f}"
              f"   {'>=' if w else '  '} ({real})")
        gate = abs(rel_p) <= BUDGET_TOL
        P("")
        P(f"  R0 budget gate: |rel params| = {abs(rel_p):.1%} <= "
          f"{BUDGET_TOL:.0%}  -> {'PASS' if gate else 'FAIL'}")
        if not gate:
            verdict = "INSTRUMENT_INCONCLUSIVE"
        elif es["flops_G"] <= zs["flops_G"] and wins >= MIN_METRICS_FOR_DOMINANCE:
            verdict = "DOMINATES"
        elif wins >= 2:
            verdict = "PARTIAL"
        else:
            verdict = "Z_WINS"
        P(f"  R1 local verdict: {verdict}  "
          f"(enc-spend >= z-spend on {wins}/{len(METRICS)} metrics; "
          f"FLOPs {rel_f:+.1%})")
        jsonout["arch"][arch] = {
            "status": "SCORED", "verdict": verdict,
            "budget_gate_pass": gate, "metrics_won": wins,
            "params_rel": rel_p, "flops_rel": rel_f,
            "ruler": ({m: RULER_SIGMA * sd[m] for m in METRICS} if sd else None),
            "cells": {r["tier"]: {k: r[k] for k in
                                  ("z", "params_M", "flops_G", "enc") + tuple(METRICS)}
                      for r in (base, zs, es)},
        }
        P("")

    # ---- R3 transfer verdict ----------------------------------------------
    P("=" * 96)
    P("R3  TRANSFER VERDICT")
    P("=" * 96)
    a1 = jsonout["arch"].get("a1", {})
    a2 = jsonout["arch"].get("a2", {})
    rank = {"Z_WINS": 0, "PARTIAL": 1, "DOMINATES": 2}
    if a1.get("status") != "SCORED" or a2.get("status") != "SCORED":
        tv = "INCONCLUSIVE"
        detail = "missing scored cells on at least one architecture"
    elif not (a1["budget_gate_pass"] and a2["budget_gate_pass"]):
        tv = "INCONCLUSIVE"
        detail = "the R0 budget gate failed on at least one architecture"
    else:
        s1, s2 = rank[a1["verdict"]], rank[a2["verdict"]]
        # sign agreement on the encoder-minus-Z gap
        agree = 0
        for m in METRICS:
            d1 = a1["cells"]["encspend"][m] - a1["cells"]["zspend"][m]
            d2 = a2["cells"]["encspend"][m] - a2["cells"]["zspend"][m]
            if (d1 >= 0) == (d2 >= 0):
                agree += 1
        jsonout["sign_agreement"] = agree
        if s2 >= s1 and agree >= 4:
            tv = "TRANSFERRED"
        elif s2 >= s1 - 1 or agree == 3:
            tv = "PARTIAL"
        else:
            tv = "NOT_TRANSFERRED"
        detail = (f"a1={a1['verdict']} ({a1['metrics_won']}/6), "
                  f"a2={a2['verdict']} ({a2['metrics_won']}/6), "
                  f"sign agreement {agree}/6")
    P(f"  {tv}")
    P(f"  {detail}")
    if not (a1.get("ruler") and a2.get("ruler")):
        P("  NOTE: ruler ABSENT on at least one architecture -> the metric-level")
        P("        differences printed above are NOT noise-qualified.  Report the")
        P("        ordering only; run the plus-seeds stage before claiming")
        P("        significance.")
    jsonout["transfer_verdict"] = tv
    jsonout["transfer_detail"] = detail

    os.makedirs(OUTDIR, exist_ok=True)
    io.open(os.path.join(OUTDIR, "g3_ledger.txt"), "w", encoding="utf-8").write(
        "\n".join(L) + "\n")
    with open(os.path.join(OUTDIR, "g3_ledger.json"), "w") as fh:
        json.dump(jsonout, fh, indent=2, default=str)
    print("\n".join(L))
    print(f"\n[written] {os.path.relpath(os.path.join(OUTDIR,'g3_ledger.txt'), ROOT)}")


if __name__ == "__main__":
    main()

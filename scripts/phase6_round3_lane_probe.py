#!/usr/bin/env python3
"""Phase 6 Round 3 -- EXP-9B arm construction + FLOPs-parity verification.

Builds the three lane arms, loads the SAME released checkpoint into all of them,
measurers params / FLOPs with the real counter, and writes the design block of
experiments/phase6/phase6_round3_lane.csv.

Parity rule (pre-registration section 4.2): FLOPs(channel) must equal
FLOPs(baseline) + [FLOPs(spatial) - FLOPs(baseline)] within +/-5 %.  If it does
not, the arm pair is reported as NOT-RUN-able with the measured gap -- a
non-parity pair would be uninterpretable, and silently reporting it anyway is the
failure mode this check exists to prevent.
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "baselines", "TwinLiteNetPlus"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from profiling.flops_real import count_flops  # noqa: E402
from phase2_load_baseline_weights import _strip_module, WEIGHTS  # noqa: E402
from models.round3.tlp_variants import Round3TLP, ARMS, channel_plan, solve_dc  # noqa: E402

CSV = os.path.join(ROOT, "experiments", "phase6", "phase6_round3_lane.csv")
TOL = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="small")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    rows = []
    for preset in args.presets.split(","):
        sd = torch.load(os.path.join(WEIGHTS, f"TwinLiteNetPlus_{preset}.pth"),
                        map_location="cpu", weights_only=False)
        sd = _strip_module(sd.get("state_dict", sd))
        x = torch.randn(1, 3, 640, 640).to(dev)
        res = {}
        for arm in ARMS:
            m = Round3TLP(preset, arm).to(dev)
            n, miss = m.load_pretrained(sd)
            assert not miss, miss[:5]
            m.eval()
            with torch.no_grad():
                fl = count_flops(m, x, reps=1)
            p = sum(q.numel() for q in m.parameters())
            res[arm] = {"params": p, "flops": fl, "extra": m.extra_params(),
                        "dc": m.dc, "loaded": n}
            print("  %-9s params=%9d (%+.0f)  FLOPs=%.6fG  dc=%d" % (
                arm, p, p - res["baseline"]["params"], fl / 1e9, m.dc))

        base, sp, ch = (res["baseline"], res["spatial"], res["channel"])
        d_sp = sp["flops"] - base["flops"]
        d_ch = ch["flops"] - base["flops"]
        gap = abs(d_ch - d_sp) / max(abs(d_sp), 1.0)
        verdict = "PARITY-OK" if gap <= TOL else "NOT-RUN-able"
        C14, C0 = channel_plan(preset)
        print("  preset=%s C14=%d C0=%d dc=%d | dFLOPs spatial=%.6fG channel=%.6fG "
              "gap=%.2f%% -> %s" % (preset, C14, C0, solve_dc(preset),
                                    d_sp / 1e9, d_ch / 1e9, 100 * gap, verdict))
        rows.append((preset, res, d_sp, d_ch, gap, verdict, C14, C0))

    head = ("stage,preset,arm,params,extra_params,dc,flops_G,d_flops_vs_baseline_G,"
            "parity_gap_pct,parity_verdict,lane_mIoU,lane_fg_iou,da_mIoU,note")
    lines = [head]
    for preset, res, d_sp, d_ch, gap, verdict, C14, C0 in rows:
        for arm in ARMS:
            r = res[arm]
            d = r["flops"] - res["baseline"]["flops"]
            lines.append("design,%s,%s,%d,%d,%d,%.6f,%.6f,%.4f,%s,NA,NA,NA,%s" % (
                preset, arm, r["params"], r["extra"], r["dc"], r["flops"] / 1e9,
                d / 1e9, gap, verdict,
                "primary preset; C14=%d C0=%d; step0 identical to baseline" % (C14, C0)))
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    with open(CSV, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("WROTE %s" % os.path.relpath(CSV, ROOT))
    bad = [r for r in rows if r[5] != "PARITY-OK"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

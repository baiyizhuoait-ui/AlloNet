#!/usr/bin/env python3
"""Phase 6 Round 3 -- append one TRAINED-arm result row to phase6_round3_trained.csv.

Keeps the trained rows in their own file so the STEP-1 zero-training schema
(phase6_round3_detection.csv: assignment diagnostics) is never overloaded with
training columns -- mixing the two would make the "which numbers are measured
without training" question unanswerable from the CSV alone.

Fails LOUDLY on a missing required key instead of writing NA: a silent NA in a
results table is indistinguishable from an arm that genuinely produced nothing,
which is the failure class this repo has already been bitten by.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys

HEADER = ("experiment,arm,arch,intervention,budget_epochs,budget_images,batch,seed,"
          "params_M,flops_G,mAP50,mAP50_95,da_mIoU,lane_mIoU,lane_fg_iou,"
          "det_loss_first,det_loss_last,lane_loss_first,lane_loss_last,"
          "wall_min,peak_gpu_mem_mib,git_commit")

# keys that MUST exist in the eval metrics.json for each kind
REQUIRED = {
    "yolop": ("mAP50", "mAP50_95", "lane_mIoU"),
    "tlp": ("lane_mIoU", "lane_fg_iou", "da_mIoU"),
}


def _tok(text, key):
    """Return the list of floats following `key` in a training-log line."""
    out = []
    for m in re.finditer(r"\b%s ([0-9.]+)" % re.escape(key), text):
        try:
            out.append(float(m.group(1)))
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["yolop", "tlp"])
    ap.add_argument("--experiment", required=True, help="EXP-9A | EXP-9B")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--intervention", required=True)
    ap.add_argument("--train", required=True, help="train metrics.json")
    ap.add_argument("--eval", required=True, help="eval metrics.json")
    ap.add_argument("--train-log", default=None, help="training_log.txt (loss trajectory)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--flops-g", type=float, required=True)
    args = ap.parse_args()

    tr = json.load(open(args.train))
    ev = json.load(open(args.eval)) if os.path.exists(args.eval) else {}
    miss = [k for k in REQUIRED[args.kind] if k not in ev]
    if miss:
        raise SystemExit("FATAL: eval metrics.json missing %s (eval=%s)" % (miss, args.eval))

    log = open(args.train_log).read() if args.train_log and os.path.exists(args.train_log) else ""
    det = _tok(log, "det")
    lane = _tok(log, "lane")

    def f(v, fmt="%.4f"):
        return "NA" if v is None else fmt % v

    row = [
        args.experiment, args.arm, args.arch, args.intervention,
        str(tr.get("epochs", "NA")), str(tr.get("num_images", "NA")),
        str(tr.get("batch_size", "NA")), str(tr.get("seed", 0)),
        "%.4f" % (tr.get("params", 0) / 1e6),
        "%.4f" % args.flops_g,
        f(ev.get("mAP50")), f(ev.get("mAP50_95")),
        f(ev.get("da_mIoU")), f(ev.get("lane_mIoU")), f(ev.get("lane_fg_iou")),
        f(det[0] if det else tr.get("final_det_loss")),
        f(det[-1] if det else tr.get("final_det_loss")),
        f(lane[0] if lane else tr.get("final_lane_loss")),
        f(lane[-1] if lane else tr.get("final_lane_loss")),
        f(tr.get("wall_min"), "%.1f"), f(tr.get("peak_gpu_mem_mib"), "%.0f"),
        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()[:40] or "unknown",
    ]
    cols = HEADER.split(",")
    if not os.path.exists(args.csv):
        with open(args.csv, "w", newline="") as fh:
            csv.writer(fh).writerow(cols)
    with open(args.csv, "a", newline="") as fh:
        csv.writer(fh).writerow(row)

    # SELF-VERIFYING APPEND (added 2026-09-11, after the B1/B2 corruption).
    # The first STEP-2 run joined fields with "," and never quoted them, so the
    # comma inside "learned 1/4 tap into lane head (zero-init, +204 params)"
    # shifted every later cell by one and lane_mIoU silently read da_mIoU's
    # value. A table that is off by one column is the same failure class as a
    # silent NA -- it still looks like a number. So read the file back and prove
    # the row we just wrote parses to the values the source metrics.json holds.
    with open(args.csv, newline="") as fh:
        back = [r for r in csv.DictReader(fh)
                if r.get("experiment") == args.experiment and r.get("arm") == args.arm]
    if not back:
        raise SystemExit("FATAL: appended row %s/%s did not read back" % (args.experiment, args.arm))
    got = back[-1]
    for key, wrote in (("mAP50", row[10]), ("lane_mIoU", row[13]),
                       ("intervention", args.intervention)):
        if got.get(key) != wrote:
            raise SystemExit("FATAL: readback mismatch on %s: wrote %r, read %r "
                             "(field quoting/alignment broken)" % (key, wrote, got.get(key)))
    print("APPENDED %s %s: mAP50=%s lane_mIoU=%s wall=%smin [readback OK]" % (
        args.experiment, args.arm, row[10], row[13], row[19]))


if __name__ == "__main__":
    main()

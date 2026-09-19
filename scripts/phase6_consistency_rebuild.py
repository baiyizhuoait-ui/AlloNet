#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild a consistency-sweep aggregate from its run log.

Why this exists
  phase6_consistency_official.py wrote `within_1.0` as a numpy.bool_, which
  json.dump cannot serialise.  The TypeError fired *after* the output file had
  been opened, so stageB_in384/consistency.json was left truncated mid-value
  (`"within_1.0": ` and then EOF) -- unreadable, and it looked like a missing
  artifact to the watcher.  The sweep itself was fine: all nine models reported
  their `[done] {...}` line to the log.

  The fix (float()/bool() casting + atomic tmp-then-rename write) is in the
  official script, so this can never happen again.  Re-running the sweep to get
  the JSON back would cost ~48 min of GPU that is needed for the seed arms, so
  the aggregate is rebuilt from the log instead -- with `rebuilt_from_log` set
  so nobody can mistake it for a first-hand artifact.

  Every number here comes from the `[done]` lines verbatim; nothing is
  recomputed, rescaled or dropped.  If a model is missing from the log it is
  reported as missing, never silently skipped.

Usage:
    gpu_env/bin/python scripts/phase6_consistency_rebuild.py \
        experiments/phase6/consistency/stageB_in384.log \
        experiments/phase6/consistency/stageB_in384
"""
import datetime as _dt
import importlib.util
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DONE_RE = re.compile(r"^\[done\]\s+(?P<key>.+?):\s+(?P<payload>\{.*\})\s*$")
PROTO_CANVAS_RE = re.compile(r"\[proto\]\s+canvas=\((\d+),\s*(\d+)\)")
PROTO_GT_RE = re.compile(r"\[proto\]\s+gt=\[(?P<gt>[^\]]*)\]\s+images=(?P<n>\d+)\s+det=(?P<det>\S+)")
HDR_RE = re.compile(r"input=(?P<input>\S+)\s+num_images=(?P<num>\d+)")


def load_published():
    """Reuse the official table so the rebuilt checks cannot drift from the tool."""
    path = os.path.join(ROOT, "scripts", "phase6_consistency_official.py")
    spec = importlib.util.spec_from_file_location("_pco", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PUBLISHED


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    log_path, outdir = sys.argv[1], sys.argv[2]
    if not os.path.exists(log_path):
        print(f"no such log: {log_path}")
        return 2

    models = {}
    errors = {}
    order = []
    proto = {}
    for line in open(log_path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("####") and "input=" in line:
            m = HDR_RE.search(line)
            if m:
                proto["input"] = m.group("input")
        m = PROTO_CANVAS_RE.search(line)
        if m:
            proto["canvas"] = [int(m.group(1)), int(m.group(2))]
        m = PROTO_GT_RE.search(line)
        if m:
            proto["gt_sources"] = [s.strip().strip("'\"") for s in m.group("gt").split(",") if s.strip()]
            proto["n_images"] = int(m.group("n"))
            proto["det_axis"] = m.group("det") == "on"
        m = DONE_RE.match(line)
        if m:
            key = m.group("key")
            row = json.loads(m.group("payload"))  # verbatim from the run
            if key not in models:
                order.append(key)
            models[key] = row
            continue
        if line.startswith("[FAIL]"):
            errors[line.split(":", 1)[1].split(":", 1)[0].strip() or "?"] = line

    if not models:
        print("no [done] lines found -- refusing to write an empty aggregate")
        return 1

    published = load_published()
    checks = []
    for key in order:
        row = models[key]
        mname, _, mpre = key.partition(":")
        pub = published.get((mname, mpre))
        if not pub:
            continue
        for mk, pk in (("da_mIoU_official", "da_mIoU"), ("lane_fg_iou_official", "lane_fg_iou"),
                       ("lane_line_acc_official", "lane_line_acc"), ("mAP50", "mAP50")):
            if pk in pub and mk in row:
                d = float(row[mk]) * 100 - float(pub[pk])
                checks.append({"model": key, "metric": pk, "measured": round(float(row[mk]) * 100, 2),
                               "published": float(pub[pk]), "delta": round(d, 2),
                               "within_1.0": bool(abs(d) <= 1.0)})

    metrics = {"input": proto.get("input", "?"),
               "canvas": proto.get("canvas"),
               "gt_sources": proto.get("gt_sources", []),
               "split": "tri_val",
               "n_images": proto.get("n_images"),
               "det_axis": proto.get("det_axis", False),
               "models": {k: models[k] for k in order},
               "errors": errors,
               "published_checks": checks,
               # provenance: this file was NOT written by the sweep process itself
               "rebuilt_from_log": os.path.relpath(log_path, ROOT),
               "rebuilt_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "rebuilt_note": ("aggregate reconstructed from the run log after "
                                "json.dump aborted on a numpy.bool_ (within_1.0) and left "
                                "the original consistency.json truncated; numbers are "
                                "verbatim [done] lines, not re-measured")}

    os.makedirs(outdir, exist_ok=True)

    def _atomic_write(path, text):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)

    def _cell(v):
        # numpy scalars are not all Python floats; a bare isinstance(v, float) check
        # silently degrades to "-" (that is how the lane columns went blank once).
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v if v is not None else "-")

    _atomic_write(os.path.join(outdir, "consistency.json"),
                  json.dumps(metrics, indent=2, default=str))

    gts = metrics["gt_sources"] or ["ours", "official"]
    md = [f"# Consistency sweep -- input={metrics['input']}, gt={'+'.join(gts)}",
          f"canvas {metrics['canvas']}, content 640x360, {metrics['n_images']} images",
          "",
          f"> Rebuilt from `{metrics['rebuilt_from_log']}` at {metrics['rebuilt_at']} -- the "
          f"original aggregate write crashed on a numpy.bool_; values are verbatim log lines.",
          ""]
    hdr = ["model"] + (["mAP50"] if metrics["det_axis"] else []) + \
          [f"da_mIoU_{s}" for s in gts] + [f"lane_fg_iou_{s}" for s in gts] + \
          [f"lane_mIoU_{s}" for s in gts] + [f"lane_line_acc_{s}" for s in gts]
    md.append("| " + " | ".join(hdr) + " |")
    md.append("|" + "---|" * len(hdr))
    for k in order:
        row = models[k]
        cells = [k] + [_cell(row.get(h)) for h in hdr[1:]]
        md.append("| " + " | ".join(cells) + " |")

    if checks:
        md += ["", "## vs published (official GT / published protocol)", "",
               "| model | metric | measured | published | delta | within +-1.0 |",
               "|---|---|---:|---:|---:|---|"]
        for c in checks:
            md.append(f"| {c['model']} | {c['metric']} | {c['measured']:.2f} | "
                      f"{c['published']:.1f} | {c['delta']:+.2f} | "
                      f"{'YES' if c['within_1.0'] else '**NO**'} |")
        ok = sum(c["within_1.0"] for c in checks)
        md += ["", f"**{ok}/{len(checks)} published references reproduced within +-1.0.**"]
    _atomic_write(os.path.join(outdir, "consistency.md"), "\n".join(md) + "\n")

    print(f"models rebuilt: {len(order)}")
    for k in order:
        print(f"  {k}")
    if errors:
        print(f"errors: {list(errors)}")
    print(f"wrote {outdir}/consistency.json and consistency.md")
    if checks:
        ok = sum(c["within_1.0"] for c in checks)
        print(f"[rule] {ok}/{len(checks)} published references reproduced within +-1.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6 -- protocol consistency: do the PUBLISHED numbers reproduce?

Two protocol axes are swept independently:

  --input 640   our canvas   : letterbox(1280x720 -> 640x640), pad cropped away
  --input 384   official     : letterbox(1280x720 -> 640x384), pad cropped away

  --gt ours      data/bdd100k/{lanes,segments}/masks/val        (production GT)
  --gt official  data/bdd100k/official_eval/{lane_line_,drivable_are_}annotations/val
  --gt both      one forward pass, both GT sources accumulated   (default)
                 -> a PURE GT swap on IDENTICAL predictions

WHY THIS IS A CLEAN 2x2
  scale is 0.5 in both canvases, so both leave a 640x360 content region and the
  metric is always computed on the SAME 640x360 pixel plane.  The official
  pipeline does exactly letterbox(image,(384,640)) + cv2.resize(label,(640,360))
  + threshold(>1)  [baselines/TwinLiteNetPlus/BDD100K.py:13,213,222,229], i.e.
  the same label plane we crop out of our 640x640 letterbox.  The official
  letterbox (BDD100K.py:20) is the same aspect-preserving min() resize our
  datasets/bdd100k.py:31 uses.

  So the only two things that can move a number are
    (a) the canvas the model sees          640x640 vs 640x384
    (b) which annotation file the GT comes from
  and this script isolates them.

PRE-REGISTERED DECISION RULE  (fixed before the run)
  The (gt=official, input=384) cell is the published protocol.  It must
  reproduce the official rows:
      TriLiteNet tiny : DA mIoU 88.5 | Lane IoU 24.2 | mAP50 49.6
      TLP nano        : DA mIoU 87.3 | Lane IoU 23.3 | Lane Acc 70.2
  within +-1.0 absolute.  If it does, the harness is protocol-compliant and any
  remaining gap in a same-protocol table is a MODEL difference, not a
  measurement artefact.  If it does not, the residual must be reported as an
  unresolved protocol unknown rather than absorbed into a model story.

NOT RE-MEASURED HERE (already certified, do not redo):
  detection mAP at 640x640 reproduces the published TriLiteNet triplet
  49.53/63.26/72.35 vs 49.6/63.2/72.3 -- so the det axis is a CONTROL, and it is
  only run when the canvas is 640 (the YOLOP-family decode is tied to the
  training grid; feeding a 384-tall canvas would change the anchor grid and make
  the det number incomparable rather than informative).

  gpu_env/bin/python scripts/phase6_consistency_official.py \
      --models TriLiteNet:tiny,TwinLiteNetPlus:nano,OursStatic:<ckpt> \
      --input 384 --gt both --num-images 2500 --outdir <dir>
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "baselines"))

# Pre-import OUR `models` package before anything builds a baseline.  The
# baseline loaders push their own repo roots onto sys.path
# (scripts/phase2_load_baseline_weights.py:59-60 puts baselines/TriLiteNet/lib
# on it), and baselines/TriLiteNet/lib/models/ then shadows our models/ for the
# REST OF THE PROCESS.  Observed 2026-09-14: build_ours died with
# "No module named 'models.static_model'" only after two baselines had been
# evaluated, and because the crash was inside the model loop the whole
# consistency.json was lost.  Binding the real package into sys.modules first
# makes the shadowing impossible.  Do not remove.
import models.static_model    # noqa: E402,F401
import models.adaptive_model  # noqa: E402,F401

from datasets.bdd100k import letterbox  # noqa: E402   production resize, not a copy
from evaluation.metrics import SegmentationMetric, evaluate_detection  # noqa: E402
from evaluation.nms import non_max_suppression  # noqa: E402
from evaluation.evaluate_baseline import build, forward_outs, normalize_batch  # noqa: E402

CANVAS = {"640": (640, 640), "384": (384, 640)}   # (H, W) fed to the model
NATIVE = (720, 1280)                              # BDD100K source resolution
DET_CATEGORIES = ("car", "bus", "truck", "train")  # single-class convention

# published reference rows (repo-local READMEs, see PHASE6_ALGORITHM_LEDGER.md)
PUBLISHED = {
    ("TriLiteNet", "tiny"): {"da_mIoU": 88.5, "lane_fg_iou": 24.2, "mAP50": 49.6},
    ("TriLiteNet", "small"): {"da_mIoU": 90.5, "lane_fg_iou": 27.6, "mAP50": 63.2},
    ("TriLiteNet", "base"): {"da_mIoU": 92.0, "lane_fg_iou": 29.8, "mAP50": 72.3},
    ("TwinLiteNetPlus", "nano"): {"da_mIoU": 87.3, "lane_fg_iou": 23.3, "lane_line_acc": 70.2},
    ("TwinLiteNetPlus", "small"): {"da_mIoU": 90.6, "lane_fg_iou": 29.3, "lane_line_acc": 75.8},
    ("TwinLiteNetPlus", "medium"): {"da_mIoU": 92.0, "lane_fg_iou": 32.3, "lane_line_acc": 79.1},
    ("TwinLiteNetPlus", "large"): {"da_mIoU": 92.9, "lane_fg_iou": 34.2, "lane_line_acc": 81.9},
}


def content_geometry(canvas):
    """(scale, pad_h, pad_w, nh, nw) for a native 1280x720 image in `canvas`."""
    ch, cw = canvas
    h, w = NATIVE
    r = min(ch / h, cw / w)
    nh, nw = round(h * r), round(w * r)
    return r, (ch - nh) // 2, (cw - nw) // 2, nh, nw


def load_gt(name, kind, src, data_root, nh, nw):
    """Binarised GT in the 640x360 content plane. kind: 'lanes' | 'segments'."""
    if src == "ours":
        p = os.path.join(data_root, kind, "masks", "val", f"{name}.png")
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(p)
        if kind == "segments":
            m = (m > 0)                       # 0=bg, 1/2=drivable
        else:
            m = (m > 0) & (m < 255)           # 255=bg, 2..48=lane
        m = m.astype(np.uint8)
        lb, _, _ = letterbox(m, (640, 640), color=0)   # production path, then crop
        return lb[140:140 + 360, 0:640]
    if src == "official":
        sub = "lane_line_annotations" if kind == "lanes" else "drivable_are_annotations"
        p = os.path.join(data_root, "official_eval", sub, "val", f"{name}.png")
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise FileNotFoundError(p)
        # official: cv2.resize(label, (W_, 360)) then threshold(>1)
        g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return (g > 1).astype(np.uint8)
    raise ValueError(src)


def det_gt_for(name, det_by_name, r, pad_h, pad_w):
    labs = det_by_name.get(name, {"labels": []})["labels"]
    out = []
    for lab in labs:
        if lab.get("category") not in DET_CATEGORIES or lab.get("box2d") is None:
            continue
        b = lab["box2d"]
        out.append([float(b["x1"]) * r + pad_w, float(b["y1"]) * r + pad_h,
                    float(b["x2"]) * r + pad_w, float(b["y2"]) * r + pad_h])
    return torch.tensor(out, dtype=torch.float32) if out else torch.zeros((0, 4))


def guard_exclusive(force):
    """This is a measurement: refuse to share the GPU with a training run."""
    try:
        n = subprocess.run(["pgrep", "-c", "-f", "training/train.py"],
                           capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        n = "0"
    n = int(n or 0)
    if n and not force:
        print(f"[guard] {n} training process(es) alive -- REFUSING to measure."
              "  Stop them first (scripts/phase6_safe_stop.sh --yes), or pass --force.")
        sys.exit(3)
    return n


def evaluate_model(spec, args, device, names, det_by_name, r, pad_h, pad_w, nh, nw,
                   gt_sources, run_det):
    mname, _, mpre = spec.partition(":")
    mpre = mpre or None
    key = f"{mname}:{mpre}" if mpre else mname
    print(f"\n===== {key} =====", flush=True)
    model, norm = build(mname, mpre, device)
    model.eval()

    det_preds, det_gts = [], []
    da_m = {s: SegmentationMetric(2) for s in gt_sources}
    lane_m = {s: SegmentationMetric(2) for s in gt_sources}
    t0 = time.time()
    with torch.no_grad():
        for it, name in enumerate(names):
            img = cv2.imread(os.path.join(args.data_root, "images", "100k", "val",
                                          f"{name}.jpg"))
            if img is None:
                raise FileNotFoundError(name)
            lb, (scale, pw, ph), (oh, ow) = letterbox(img, args.canvas)
            assert (scale, pw, ph, oh, ow) == (r, pad_w, pad_h, nh, nw)
            x = torch.from_numpy(np.ascontiguousarray(
                lb[:, :, ::-1].transpose(2, 0, 1))).float().unsqueeze(0).to(device) / 255.0
            x = normalize_batch(x, norm)
            det_logits, da_logits, lane_logits = forward_outs(model, mname, x)

            if run_det and det_logits is not None:
                d = non_max_suppression(det_logits, conf_thres=args.conf,
                                        iou_thres=args.iou)[0]
                det_preds.append((d[:, :4].cpu(), d[:, 4].cpu(), d[:, 5].long().cpu())
                                 if len(d) else None)
                det_gts.append(det_gt_for(name, det_by_name, r, pad_h, pad_w))

            for s in gt_sources:
                if da_logits is not None:
                    da_pred = da_logits[0].argmax(0)[ph:ph + oh, pw:pw + ow].cpu().numpy()
                    da_m[s].add_batch(da_pred, load_gt(name, "segments", s, args.data_root, nh, nw))
                if lane_logits is not None:
                    lane_pred = lane_logits[0].argmax(0)[ph:ph + oh, pw:pw + ow].cpu().numpy()
                    lane_m[s].add_batch(lane_pred, load_gt(name, "lanes", s, args.data_root, nh, nw))
            if (it + 1) % 1000 == 0:
                print(f"  [{it+1}/{len(names)}] {time.time()-t0:.0f}s", flush=True)

    row = {"n": len(names), "secs": round(time.time() - t0, 1)}
    if run_det and det_preds:
        row["mAP50"] = round(evaluate_detection(det_preds, det_gts)["mAP50"], 4)
    for s in gt_sources:
        if da_m[s].conf.sum() > 0:
            row[f"da_mIoU_{s}"] = round(da_m[s].m_iou(), 4)
        if lane_m[s].conf.sum() > 0:
            row[f"lane_fg_iou_{s}"] = round(lane_m[s].fg_iou(), 4)
            row[f"lane_mIoU_{s}"] = round(lane_m[s].m_iou(), 4)
            row[f"lane_line_acc_{s}"] = round(lane_m[s].line_accuracy(), 4)
    del model
    torch.cuda.empty_cache()
    print(f"[done] {key}: {json.dumps(row)}", flush=True)
    return key, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True,
                    help="comma list, e.g. TriLiteNet:tiny,TwinLiteNetPlus:nano,"
                         "OursStatic:/abs/path/checkpoint.pt")
    ap.add_argument("--input", default="384", choices=["640", "384"])
    ap.add_argument("--gt", default="both", choices=["ours", "official", "both"])
    ap.add_argument("--split", default="tri_val")
    ap.add_argument("--num-images", type=int, default=0)
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "bdd100k"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    guard_exclusive(args.force)
    device = torch.device(args.device)
    args.canvas = CANVAS[args.input]
    r, pad_h, pad_w, nh, nw = content_geometry(args.canvas)
    assert (nh, nw) == (360, 640), f"content region {nh}x{nw} != 640x360"
    gt_sources = ["ours", "official"] if args.gt == "both" else [args.gt]
    run_det = (args.input == "640")

    names = [ln.strip() for ln in
             open(os.path.join(args.data_root, "splits", f"{args.split}.txt")) if ln.strip()]
    if args.num_images:
        names = names[:args.num_images]
    det_by_name = {}
    if run_det:
        jf = os.path.join(args.data_root, "labels", "bdd100k_labels_images_val.json")
        det_by_name = {x["name"].rsplit(".", 1)[0]: x for x in json.load(open(jf))}

    print(f"[proto] canvas={args.canvas} content={nh}x{nw} scale={r:.4f} pad=({pad_w},{pad_h})")
    print(f"[proto] gt={gt_sources} images={len(names)} det={'on' if run_det else 'skipped'}")

    metrics = {"input": args.input, "canvas": list(args.canvas), "gt_sources": gt_sources,
               "split": args.split, "n_images": len(names), "det_axis": run_det,
               "models": {}, "errors": {}}
    for spec in args.models.split(","):
        spec = spec.strip()
        if not spec:
            continue
        # one model failing must not lose the whole sweep (the Stage-A lesson:
        # a shadowed import aborted the run before anything was written)
        try:
            key, row = evaluate_model(spec, args, device, names, det_by_name,
                                      r, pad_h, pad_w, nh, nw, gt_sources, run_det)
        except Exception as e:  # noqa: BLE001
            key = spec.partition(":")[0] + (":" + spec.partition(":")[2] if ":" in spec else "")
            metrics["errors"][key] = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {key}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        metrics["models"][key] = row

    # ---- published reference + decision rule ------------------------------
    checks = []
    for key, row in metrics["models"].items():
        mname, _, mpre = key.partition(":")
        pub = PUBLISHED.get((mname, mpre))
        if not pub:
            continue
        for mk, pk in (("da_mIoU_official", "da_mIoU"), ("lane_fg_iou_official", "lane_fg_iou"),
                       ("lane_line_acc_official", "lane_line_acc"), ("mAP50", "mAP50")):
            if pk in pub and mk in row:
                # float()/bool() are not cosmetic: row[mk] and PUBLISHED values can be
                # numpy scalars, and numpy.bool_ is NOT JSON-serialisable.  On 2026-09-14
                # that TypeError fired inside json.dump AFTER the file had been opened for
                # writing, so consistency.json was left truncated mid-value and silently
                # unreadable.  Cast every leaf, and write atomically below.
                d = float(row[mk]) * 100 - float(pub[pk])
                checks.append({"model": key, "metric": pk, "measured": round(float(row[mk]) * 100, 2),
                               "published": float(pub[pk]), "delta": round(d, 2),
                               "within_1.0": bool(abs(d) <= 1.0)})
    metrics["published_checks"] = checks

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

        def _atomic_write(path, text):
            # write to a sibling .tmp then rename: an exception mid-write must never leave a
            # half-serialised artifact behind (the 2026-09-14 truncation).
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                fh.write(text)
            os.replace(tmp, path)

        def _cell(v):
            # Row values are plain floats, but they arrive from a dict that may hold
            # numpy scalars (json.dumps above passes default=str for the same reason).
            # None means the axis was not accumulated for this run -- print "-", never 0.
            if v is None:
                return "-"
            try:
                return "{:.4f}".format(float(v))
            except (TypeError, ValueError):
                return str(v)

        _atomic_write(os.path.join(args.outdir, "consistency.json"),
                      json.dumps(metrics, indent=2, default=str))
        md = [f"# Consistency sweep -- input={args.input}, gt={'+'.join(gt_sources)}",
              f"canvas {args.canvas}, content 640x360, {len(names)} images", ""]
        # NOTE: the column names must match the row keys verbatim.  evaluate_model emits
        # da_mIoU_<s> / lane_fg_iou_<s> / lane_mIoU_<s> / lane_line_acc_<s>; an earlier
        # revision spelled two of these lane_fgIoU_/lane_acc_, so those two columns silently
        # printed "-" for every model.  Keep this list in lockstep with evaluate_model.
        hdr = ["model"] + (["mAP50"] if run_det else []) + \
              [f"da_mIoU_{s}" for s in gt_sources] + \
              [f"lane_fg_iou_{s}" for s in gt_sources] + \
              [f"lane_mIoU_{s}" for s in gt_sources] + \
              [f"lane_line_acc_{s}" for s in gt_sources]
        md.append("| " + " | ".join(hdr) + " |")
        md.append("|" + "---|" * len(hdr))
        for k, row in metrics["models"].items():
            cells = [k] + [_cell(row.get(h)) for h in hdr[1:]]
            md.append("| " + " | ".join(cells) + " |")
        for k, e in metrics["errors"].items():
            md.append(f"\n- **{k}**: FAILED -- {e}")
        if checks:
            md += ["", "## vs published (official GT / published protocol)", "",
                   "| model | metric | measured | published | delta | within +-1.0 |",
                   "|---|---|---:|---:|---:|---|"]
            for c in checks:
                md.append(f"| {c['model']} | {c['metric']} | {c['measured']:.2f} | "
                          f"{c['published']:.1f} | {c['delta']:+.2f} | "
                          f"{'YES' if c['within_1.0'] else '**NO**'} |")
        _atomic_write(os.path.join(args.outdir, "consistency.md"), "\n".join(md) + "\n")
        print(f"\nsaved -> {args.outdir}/consistency.json , consistency.md")

    if checks:
        ok = sum(c["within_1.0"] for c in checks)
        print(f"\n[rule] {ok}/{len(checks)} published references reproduced within +-1.0")
        for c in checks:
            print(f"  {'OK ' if c['within_1.0'] else 'OFF'} {c['model']:22s} {c['metric']:14s} "
                  f"{c['measured']:7.2f} vs {c['published']:5.1f}  ({c['delta']:+.2f})")
    else:
        print("\n[rule] no published reference row for these models -- nothing to check")


if __name__ == "__main__":
    main()

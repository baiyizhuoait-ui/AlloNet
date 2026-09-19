#!/usr/bin/env python3
"""Phase 6 Round 3 -- EXP-9A zero-training assignment analysis (0 GPU).

Cross-architecture test of H1 (detection's binding resource is supervision /
assignment, not raw feature capacity).  Instrument: anchor-set quality on the
second architecture (YOLOP), measured the same way as in Round 1-2 (EXP-03 /
phase5_anchor_kmeans) so the two architectures are read against identical
statistics.

ARMS (all 3 levels x 3 anchors = 9, supplied from disk, never from memory)
  D    shipped   YOLOP's own anchors, read LIVE from the built model
                 (model.model[detector_index].anchor_grid -> pixels).  No
                 hard-coded copy, so a silently-changed upstream cannot drift.
  D-   degraded  aspect-flipped D (swap w,h).  Replicates the *pathology class*
                 of R2's pre-2026-09-08 default set (which was aspect-flipped
                 vs the data, h/w ~2.2-3.3 against a near-square 0.85 median).
  D+   repaired  IoU-k-means refit on BDD100K tri_train, same procedure and
                 same n as scripts/phase5_anchor_kmeans.py.
CONTEXT (not arms -- R2's own sets, for cross-architecture comparison)
  R2old / R2km   the two sets that produced the +0.1392 mAP50 effect.

RULES -- reported side by side, because "a hole" is a fact about the RULE as
much as about the anchors (phase5_anchor_kmeans.py already made this point):
  project : 0.5 < box/anchor < 2.0   (losses/yolo_loss.py:106  -- our protocol)
  yolov5  : 0.25 < box/anchor < 4.0  (YOLOP TRAIN.ANCHOR_THRESHOLD=4.0,
                                      lib/config/default.py:100 -- theirs)

The stride cancels in box_px/anchor_px, so assignment depends only on anchor
PIXEL size, not on the level's stride (phase5_anchor_coverage.py:5-7).

Writes experiments/phase6/phase6_round3_detection.csv
       experiments/phase6/phase6_round3_anchor_summary.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from datasets.bdd100k import BDD100KDataset  # noqa: E402

N_IMAGES = 3000
SEED = 0
OUT_DIR = "experiments/phase6"

R2_OLD = np.array([[4, 12], [7, 19], [11, 28],
                   [17, 40], [25, 58], [38, 89],
                   [62, 136], [88, 206], [124, 412]], dtype=float)
R2_KM = np.array([[9, 8], [18, 15], [32, 24],
                  [49, 38], [80, 52], [65, 102],
                  [124, 82], [166, 136], [237, 214]], dtype=float)


def shipped_anchors():
    """Live-read YOLOP's pixel anchors from the built model.  Returns (9,2)."""
    from evaluation.evaluate_baseline import build_yolop
    m = build_yolop("cpu")
    det = m.model[m.detector_index]
    grid = det.anchor_grid.reshape(det.nl, det.na, 2).detach().cpu().numpy()
    return grid.reshape(-1, 2).astype(float), grid


def iou_wh(a, b):
    inter = np.minimum(a[:, None, 0], b[None, :, 0]) * np.minimum(a[:, None, 1], b[None, :, 1])
    union = a[:, None, 0] * a[:, None, 1] + b[None, :, 0] * b[None, :, 1] - inter
    return inter / np.maximum(union, 1e-9)


def kmeans(boxes, k, iters=100, seed=SEED):
    rng = np.random.default_rng(seed)
    cent = boxes[rng.choice(boxes.shape[0], k, replace=False)]
    for _ in range(iters):
        assign = (1.0 - iou_wh(boxes, cent)).argmin(1)
        new = cent.copy()
        for j in range(k):
            m = assign == j
            if m.sum():
                new[j] = boxes[m].mean(0)
        if np.allclose(new, cent):
            break
        cent = new
    return cent


def zero_pos(boxes, anchors, lo, hi):
    """(n,) bool -> no (anchor) satisfies lo < box/anchor < hi on both axes."""
    r = boxes[:, None, :] / anchors[None, :, :]
    ok = (r > lo).all(-1) & (r < hi).all(-1)
    return ok.sum(1) == 0, ok.sum(1)


def stats(boxes, anchors, lo, hi):
    zp, nm = zero_pos(boxes, anchors, lo, hi)
    iou = iou_wh(boxes, anchors).max(1)
    return dict(n=len(boxes), mean_best_iou=float(iou.mean()),
                zero_pos=float(zp.mean()), mean_matches=float(nm.mean()),
                pos_targets=int(nm.sum()))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    shp, shp_grid = shipped_anchors()
    print("=== YOLOP shipped anchors, read LIVE from the built model (pixels, 640 canvas) ===")
    for lv in range(3):
        print("  L%d (stride %d): %s" % (lv + 1, [8, 16, 32][lv],
              " ".join("(%g,%g)" % (w, h) for w, h in shp_grid[lv])))
    print("  aspect h/w : %s" % " ".join("%.2f" % (h / w) for w, h in shp))

    # ---- GT boxes (det-only load: no mask I/O) ----
    ds = BDD100KDataset("data/bdd100k", split="tri_train",
                        with_det=True, with_da=False, with_lane=False)
    ws, hs = [], []
    for i in range(min(len(ds), N_IMAGES)):
        t = ds[i]["det_targets"]
        if t is None:
            continue
        t = np.asarray(t)
        if t.ndim != 2 or t.shape[0] == 0:
            continue
        ws.extend((t[:, 3] * 640.0).tolist())
        hs.extend((t[:, 4] * 640.0).tolist())
    boxes = np.stack([np.asarray(ws), np.asarray(hs)], 1)
    side = np.sqrt(boxes[:, 0] * boxes[:, 1])
    print("\n=== GT population === %d boxes from %d tri_train images" % (len(boxes), N_IMAGES))
    print("  side px: p10 %.1f  p50 %.1f  p90 %.1f | aspect h/w median %.2f"
          % (np.percentile(side, 10), np.percentile(side, 50), np.percentile(side, 90),
             float(np.median(boxes[:, 1] / boxes[:, 0]))))

    # ---- D+ : refit k-means on the same data ----
    km = kmeans(boxes, 9)
    km = km[np.argsort(km[:, 0] * km[:, 1])]
    flip = np.stack([shp[:, 1], shp[:, 0]], 1)

    arms = [
        ("D", "shipped (YOLOP own, live)", shp),
        ("D-", "aspect-flip of shipped", flip),
        ("D+", "IoU-k-means refit (n=%d)" % N_IMAGES, km),
        ("ctx_R2old", "R2 old aspect-flipped default", R2_OLD),
        ("ctx_R2km", "R2 k-means (won +0.1392)", R2_KM),
    ]
    buckets = [("ALL", np.ones_like(side, bool)),
               ("small(<32)", side < 32),
               ("medium(32-96)", (side >= 32) & (side < 96)),
               ("large(>=96)", side >= 96)]
    rules = [("project", 0.5, 2.0), ("yolov5", 0.25, 4.0)]

    rows = ["stage,arm,anchor_set,rule,bucket,n_gt,mean_best_iou,zero_pos_rate,mean_matches,pos_targets"]
    summary = {"n_images": N_IMAGES, "n_gt": int(len(boxes)),
               "gt_side_p10": float(np.percentile(side, 10)),
               "gt_side_p50": float(np.percentile(side, 50)),
               "gt_side_p90": float(np.percentile(side, 90)),
               "gt_aspect_median": float(np.median(boxes[:, 1] / boxes[:, 0])),
               "shipped": shp.tolist(), "flipped": flip.tolist(), "kmeans": km.tolist(),
               "arms": {}}

    for key, desc, A in arms:
        summary["arms"][key] = {"desc": desc, "anchors": A.tolist(), "by_rule": {}}
        for rname, lo, hi in rules:
            per_bucket = {}
            for bname, m in buckets:
                if m.sum() == 0:
                    continue
                s = stats(boxes[m], A, lo, hi)
                per_bucket[bname] = s
                rows.append("%s,%s,\"%s\",%s,%s,%d,%.4f,%.4f,%.3f,%d" % (
                    "zero_training", key, desc, rname, bname, s["n"],
                    s["mean_best_iou"], s["zero_pos"], s["mean_matches"], s["pos_targets"]))
            summary["arms"][key]["by_rule"][rname] = per_bucket

    # ---- per-level best-match share (which level wins each GT) ----
    lvl = {"D": {}, "D+": {}}
    for key, A in (("D", shp), ("D+", km)):
        L = A.reshape(3, 3, 2)
        ious = np.stack([iou_wh(boxes, L[i]).max(1) for i in range(3)], 1)  # (n,3)
        win = ious.argmax(1)
        lvl[key] = {int(i): float((win == i).mean()) for i in range(3)}
    summary["level_win_share"] = lvl

    with open(os.path.join(OUT_DIR, "phase6_round3_detection.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(os.path.join(OUT_DIR, "phase6_round3_anchor_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- console report ----
    for rname, lo, hi in rules:
        print("\n=== rule %-8s (%.2f < box/anchor < %.2f) ===" % (rname, lo, hi))
        print("  %-11s %8s %8s %10s %9s" % ("arm", "meanIoU", "zeroPos", "matches/gt", "posTargets"))
        for key, _d, A in arms:
            s = summary["arms"][key]["by_rule"][rname]["ALL"]
            print("  %-11s %8.3f %7.1f%% %10.2f %9d" % (
                key, s["mean_best_iou"], 100 * s["zero_pos"], s["mean_matches"], s["pos_targets"]))
    print("\n=== level win share (argmax best-IoU level) ===")
    for k, v in lvl.items():
        print("  %-4s L1 %.1f%%  L2 %.1f%%  L3 %.1f%%" % (k, 100 * v[0], 100 * v[1], 100 * v[2]))
    print("\nWROTE %s/phase6_round3_detection.csv + phase6_round3_anchor_summary.json" % OUT_DIR)


if __name__ == "__main__":
    main()

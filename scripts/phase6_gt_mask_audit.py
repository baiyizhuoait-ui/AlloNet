#!/usr/bin/env python3
"""Phase 6 -- ground-truth mask integrity audit (zero training, CPU only).

Question raised in review: our lane column is systematically 4-10 points below the
published numbers for the SAME models, and the gap GROWS with model capacity.
Hypothesis: the ground-truth mask is corrupted on the way to the metric.

Traced path (production code, unmodified):
  datasets/bdd100k.py:_load_mask   -> binary HxW uint8 at native 1280x720
  datasets/bdd100k.py:142/157      -> letterbox(mask, (640,640), color=0)
  datasets/bdd100k.py:33           -> cv2.resize(..., interpolation=INTER_LINEAR)   <-- bilinear on a BINARY mask
  evaluation/evaluate_baseline.py:403/407
                                   -> .numpy().astype(np.uint8)                     <-- truncates 0<v<1 to 0

So any interpolated pixel (0 < v < 1) becomes background. Thick DA regions keep a solid
1.0 core; a ~2 px lane line has almost NO solid core, so it is eroded or dotted.

This script measures fg-pixel retention for both masks under three protocols:
  RAW       native-resolution binary label, direct from disk   (what the paper counts)
  SHIPPED   through the production loader + the eval binarisation
  NEAREST   same pipeline but with INTER_NEAREST resize (the standard fix)
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.bdd100k import BDD100KDataset, letterbox  # noqa: E402


def raw_mask(path, kind):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if kind == "segments":
        return (m > 0).astype(np.uint8)
    return ((m > 0) & (m < 255)).astype(np.uint8)


def nearest_letterbox(img, new_shape=(640, 640)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2
    out = cv2.copyMakeBorder(resized, top, new_shape[0] - nh - top,
                             left, new_shape[1] - nw - left,
                             cv2.BORDER_CONSTANT, value=0)
    return out, (r, left, top), (nh, nw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--split", default="tri_val")
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "bdd100k"))
    args = ap.parse_args()

    ds = BDD100KDataset(args.data_root, split=args.split, img_size=640,
                        with_det=False, with_da=True, with_lane=True)
    names = ds.names[:args.n]
    print(f"[audit] {len(names)} images from {args.split}")

    accum = {k: {p: 0.0 for p in ("RAW", "SHIPPED", "NEAREST")} for k in ("da", "lane")}
    raw_tot = {"da": 0.0, "lane": 0.0}

    for i, name in enumerate(names):
        for kind, key in (("segments", "da"), ("lanes", "lane")):
            p = os.path.join(args.data_root, kind, "masks", args.split.split("_")[-1], f"{name}.png")
            if not os.path.exists(p):
                continue
            raw = raw_mask(p, kind)
            h, w = raw.shape
            raw_fg = float(raw.sum())
            raw_tot[key] += raw_fg

            sh, (s, l, t), (nh, nw) = letterbox(raw, (640, 640), color=0)
            sh_fg = float(sh[t:t + nh, l:l + nw].astype(np.uint8).sum())

            nr, (s2, l2, t2), (nh2, nw2) = nearest_letterbox(raw, (640, 640))
            nr_fg = float(nr[t2:t2 + nh2, l2:l2 + nw2].astype(np.uint8).sum())

            accum[key]["RAW"] += raw_fg / (h * w)
            accum[key]["SHIPPED"] += sh_fg / (nh * nw)
            accum[key]["NEAREST"] += nr_fg / (nh2 * nw2)
        if (i + 1) % 100 == 0:
            print(f"  .. {i + 1}/{len(names)}")

    n = len(names)
    print("\n[audit] foreground pixel fraction (mean over images)\n")
    print("| mask | RAW (native label) | SHIPPED (loader+eval) | NEAREST (fix) | shipped retention |")
    print("|---|---:|---:|---:|---:|")
    for key, label in (("da", "Drivable area"), ("lane", "Lane")):
        r, s, nr = (accum[key][p] / n for p in ("RAW", "SHIPPED", "NEAREST"))
        print(f"| {label} | {r * 100:.3f}% | {s * 100:.3f}% | {nr * 100:.3f}% | "
              f"**{s / r * 100:.1f}%** |")

    print("\n[audit] interpretation")
    print("  shipped retention << 100% on Lane  => the GT is eroded before it reaches the metric.")
    print("  NEAREST retention ~100%            => the loss is caused by INTER_LINEAR, not by resizing per se.")
    print("  A halved-width GT deflates IoU for every model, and deflates thin-line")
    print("  (i.e. better) predictions hardest -- which is the observed gap pattern.")


if __name__ == "__main__":
    main()

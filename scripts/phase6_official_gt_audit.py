#!/usr/bin/env python3
"""Phase 6 -- official-vs-ours ground-truth audit (zero training, CPU only).

Question this answers: our lane/DA numbers are systematically below the published
ones. Is that the models, or is it the ground truth? The published pipelines read a
SEPARATE annotation package (Google-Drive `*_annotations.zip`), not the BDD100K
released `lanes/masks` and `segments/masks` that we evaluate against.

This script compares the two label sets on the SAME images, at native resolution,
under both candidate binarisations, and reports:
  1. id-set overlap between our split list and the package
  2. value histograms of each encoding
  3. positive (foreground) fraction of each
  4. pixel-wise agreement (Jaccard) between the two label sets
  5. what the official `>1` threshold does to OUR file (the decisive check)

No model, no GPU, no training.
"""
import argparse, glob, os, sys
import numpy as np
import cv2


def hist(path, kind):
    """kind = 'lane' | 'da'. Returns (unique->count) dict for one mask."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    u, c = np.unique(g, return_counts=True)
    return dict(zip(u.tolist(), c.tolist())), g.shape


def binarise_ours(g, kind):
    """What our data pipeline does: lane = 0<v<255 ; da = v>0 (per datasets/bdd100k.py)."""
    if kind == "lane":
        return (g > 0) & (g < 255)
    return g > 0


def binarise_official(g):
    """What the third-party pipelines do to their own package: threshold(.,1,255,BINARY)."""
    return g > 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/bdd100k")
    ap.add_argument("--official-root", default="data/bdd100k/official_eval")
    ap.add_argument("--split", default="tri_val")
    ap.add_argument("--n", type=int, default=400, help="images sampled")
    args = ap.parse_args()

    R, O = args.data_root, args.official_root

    # ---- 1. id sets ----------------------------------------------------------
    sp = os.path.join(R, "splits", f"{args.split}.txt")
    with open(sp) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    off_lane = {os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(O, "lane_line_annotations", "val", "*.png"))}
    off_da = {os.path.splitext(os.path.basename(p))[0]
              for p in glob.glob(os.path.join(O, "drivable_are_annotations", "val", "*.png"))}
    ours_lane = {os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(R, "lanes", "masks", "val", "*.png"))}
    ours_da = {os.path.splitext(os.path.basename(p))[0]
               for p in glob.glob(os.path.join(R, "segments", "masks", "val", "*.png"))}
    print("=" * 78)
    print("1. ID SETS")
    print(f"   our split {args.split:<10}: {len(names)} names")
    print(f"   ours  lane masks : {len(ours_lane)}   da masks: {len(ours_da)}")
    print(f"   pkg   lane masks : {len(off_lane)}   da masks: {len(off_da)}")
    print(f"   pkg-lane ∩ split : {len(off_lane & set(names))} / {len(names)}")
    print(f"   pkg-da   ∩ split : {len(off_da & set(names))} / {len(names)}")

    # ---- 2/3. encodings on a sample -----------------------------------------
    step = max(1, len(names) // args.n)
    sample = names[::step][: args.n]
    print("=" * 78)
    print(f"2. ENCODINGS  (sample n={len(sample)})")
    for kind, our_sub, off_sub in (("lane", "lanes/masks", "lane_line_annotations"),
                                   ("da", "segments/masks", "drivable_are_annotations")):
        ours_h, off_h = {}, {}
        shapes = set()
        for nm in sample:
            a = os.path.join(R, our_sub, "val", f"{nm}.png")
            b = os.path.join(O, off_sub, "val", f"{nm}.png")
            if os.path.exists(a):
                h, sh = hist(a, kind)
                if h:
                    shapes.add(("ours", sh))
                    for k, v in h.items():
                        ours_h[k] = ours_h.get(k, 0) + v
            if os.path.exists(b):
                h, sh = hist(b, kind)
                if h:
                    shapes.add(("pkg", sh))
                    for k, v in h.items():
                        off_h[k] = off_h.get(k, 0) + v
        print(f"\n   --- {kind.upper()} ---")
        print(f"   our encoding keys   : {sorted(ours_h)[:12]}"
              f"{' ...' if len(ours_h) > 12 else ''}  ({len(ours_h)} distinct)")
        print(f"   pkg encoding keys   : {sorted(off_h)[:12]}"
              f"{' ...' if len(off_h) > 12 else ''}  ({len(off_h)} distinct)")
        for tag, h in (("ours", ours_h), ("pkg ", off_h)):
            tot = sum(h.values()) or 1
            fg = sum(v for k, v in h.items() if 0 < k < 255) if kind == "lane" \
                else sum(v for k, v in h.items() if k > 0)
            print(f"   {tag} foreground      : {fg/tot*100:.3f}%")
        print(f"   shapes seen         : {sorted(shapes)}")

    # ---- 4/5. agreement + the decisive threshold test ------------------------
    print("=" * 78)
    print("4. AGREEMENT between the two label sets, and")
    print("5. what the OFFICIAL threshold (>1) does to OUR file")
    for kind, our_sub, off_sub in (("lane", "lanes/masks", "lane_line_annotations"),
                                   ("da", "segments/masks", "drivable_are_annotations")):
        jac, pos_o, pos_p, pos_thr_on_ours, n = [], [], [], [], 0
        for nm in sample:
            a = os.path.join(R, our_sub, "val", f"{nm}.png")
            b = os.path.join(O, off_sub, "val", f"{nm}.png")
            if not (os.path.exists(a) and os.path.exists(b)):
                continue
            ga = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
            gb = cv2.imread(b, cv2.IMREAD_GRAYSCALE)
            if ga is None or gb is None:
                continue
            if ga.shape != gb.shape:
                gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            A = binarise_ours(ga, kind)
            B = binarise_official(gb)
            inter = np.logical_and(A, B).sum()
            union = np.logical_or(A, B).sum()
            if union:
                jac.append(inter / union)
            pos_o.append(A.mean())
            pos_p.append(B.mean())
            pos_thr_on_ours.append(binarise_official(ga).mean())
            n += 1
        print(f"\n   --- {kind.upper()} ---  (n={n} paired)")
        print(f"   foreground ours        : {np.mean(pos_o)*100:.3f}%")
        print(f"   foreground pkg         : {np.mean(pos_p)*100:.3f}%")
        print(f"   Jaccard(ours, pkg)     : {np.mean(jac):.4f}"
              f"   (median {np.median(jac):.4f})")
        print(f"   official >1 on OUR file: {np.mean(pos_thr_on_ours)*100:.3f}%"
              f"   <-- if ~100%, the two files CANNOT be the same encoding")
    print("=" * 78)
    print("AUDIT DONE")


if __name__ == "__main__":
    sys.exit(main())

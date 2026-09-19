"""Regression tripwire for the 2026-09-11 EXP-9A input-distribution drift.

WHAT WENT WRONG (D2): the Round-3 YOLOP trainer fed the model the dataset's raw
[0,1] tensors, while the evaluator fed ImageNet-normalised ones.  Every "trained"
arm was therefore scored on an input distribution it had never been optimised
for; the released mAP collapsed 0.7712 -> 0.07 and the whole EXP-9A table was
meaningless.  Code review did not catch it, because each file looked correct in
isolation -- the defect lived in the RELATION between the two.

This check pins that relation.  It fails loudly if it ever drifts again.

THREE LAYERS
  L1 (data): the tensor the trainer builds is element-wise identical to the one
             the evaluator builds, for each family's own normalisation.
  L2 (non-vacuity): the two normalisations are actually different, so L1 cannot
             pass trivially.
  L3 (structure): neither trainer implements normalisation itself -- both must
             route through evaluation.evaluate_baseline.normalize_batch.  This is
             the layer that stops a future edit from re-introducing a private
             copy of the arithmetic.

Exit code 0 = PARITY-OK, non-zero = drift detected.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.bdd100k import BDD100KDataset                      # noqa: E402
from evaluation.evaluate_baseline import normalize_batch, preprocess  # noqa: E402

FAMILIES = {"YOLOP": "imagenet", "TwinLiteNetPlus-small": "unit"}
TRAINERS = ["scripts/phase6_round3_yolop_train.py",
            "scripts/phase6_round3_tlp_train.py"]


def fail(msg):
    print("[parity] FAIL: " + msg)
    sys.exit(1)


def main():
    ds = BDD100KDataset(os.path.join(ROOT, "data", "bdd100k"), split="tri_val",
                        with_det=True, with_da=True, with_lane=True)

    # --- L0: the eval-mode dataset must be deterministic, else L1 is meaningless
    a, b = ds[0]["image"], ds[0]["image"]
    if not torch.equal(a, b):
        fail("dataset is non-deterministic in eval mode (hflip leaking in?)")
    print("[parity] L0 eval-mode dataset deterministic: OK")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- L1: evaluator path == trainer path, per family
    for fam, norm in FAMILIES.items():
        item_img = ds[0]["image"]
        ev = preprocess(item_img, norm).cpu()                                   # evaluator
        tr = normalize_batch(item_img.unsqueeze(0).to(dev), norm).cpu()         # trainer
        d = (ev - tr).abs().max().item()
        print("[parity] L1 %-20s norm=%-8s max|train-eval| = %.3e" % (fam, norm, d))
        if d != 0.0:
            fail("train/eval input tensors differ for %s (max|delta|=%.3e)" % (fam, d))

    # --- L2: non-vacuity -- the normalisations must not be no-ops relative to each other
    x = ds[0]["image"].unsqueeze(0).to(dev)
    if torch.equal(normalize_batch(x, "imagenet"), normalize_batch(x, "unit")):
        fail("imagenet and unit normalisation produced identical tensors "
             "(L1 would pass vacuously)")

    # --- L3: structural -- no trainer may carry its own normalisation arithmetic
    for rel in TRAINERS:
        src = open(os.path.join(ROOT, rel)).read()
        if "normalize_batch(" not in src:
            fail("%s does not call normalize_batch()" % rel)
        if "IMAGENET_STD" in src or "IMAGENET_MEAN" in src:
            fail("%s contains a private copy of the ImageNet normalisation "
                 "constants - route it through normalize_batch()" % rel)
    print("[parity] L2 non-vacuity: OK")
    print("[parity] L3 trainers use the shared helper only: OK")
    print("[parity] PARITY-OK")


if __name__ == "__main__":
    main()

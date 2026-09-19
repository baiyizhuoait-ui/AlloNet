#!/usr/bin/env python3
"""phase6 / R4B -- D9 SCOPE REPAIR (device-wide samples taken from a co-tenant probe).

D9 = the `mem a/b MiB` field is DEVICE-WIDE (nvidia-smi memory.used). The code fix
(commit 131c763) switched the VRAM guard to the per-process `self alloc/reserved`
field. This script repairs the DATA that was written before that fix. Two edits only:

1) R6-R2thinL4  peak_gpu_mem_mib  7692 -> 2468
   The lr-probe arm ran 09-12 21:34-23:57. 19 of its 4380 memory samples read >=2500MiB
   and all 19 fall inside 23:36:57-23:39:5x -- exactly when this session ran two GPU
   probes on the same card (probe mtimes 23:36:48 / 23:37:13). The other 4361 samples
   read <2500MiB; the arm own peak is 2468MiB, equal to its sibling R4R2thin40
   (2468MiB, 8760/8760 samples <2500MiB) and to R4R2s1 (2461MiB).

2) R4-R2thin14 seed 1  git_commit  131c763... -> ba7b23d...
   R2s1 trained 00:02-02:23, i.e. BEFORE 131c763 was created (00:18). Its log holds 0
   per-process `self` samples, proving it ran the pre-D9 train.py. Its row was registered
   retroactively (D10), so the auto-recorded HEAD is not the training commit.

No other field, arm or row is modified. Both tables are backed up first.
"""
import csv, shutil, subprocess, sys, pathlib

CSV  = pathlib.Path("experiments/phase6/phase6_round4_results.csv")
TRAIN_COMMIT = "ba7b23dbe10dd84a489bfb6fd0cd7af3e1b199e3"
edits = []

shutil.copy2(CSV, str(CSV) + ".pred9.bak")
rows = list(csv.DictReader(CSV.open(newline="")))
hdr = list(rows[0].keys())

for r in rows:
    if r["cell"] == "R6-R2thinL4" and r["peak_gpu_mem_mib"] == "7692":
        edits.append(("R6-R2thinL4", "peak_gpu_mem_mib", r["peak_gpu_mem_mib"], "2468"))
        r["peak_gpu_mem_mib"] = "2468"
    if r["cell"] == "R4-R2thin14" and str(r["seed"]) == "1" and r["git_commit"].startswith("131c763"):
        edits.append(("R4-R2thin14/seed1", "git_commit", r["git_commit"][:8], TRAIN_COMMIT[:8]))
        r["git_commit"] = TRAIN_COMMIT

with CSV.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=hdr)
    w.writeheader()
    w.writerows(rows)

# readback: re-open and re-assert
back = list(csv.DictReader(CSV.open(newline="")))
for r in back:
    if r["cell"] == "R6-R2thinL4":
        assert r["peak_gpu_mem_mib"] == "2468", r
    if r["cell"] == "R4-R2thin14" and str(r["seed"]) == "1":
        assert r["git_commit"] == TRAIN_COMMIT, r
print("[fix_d9_scope] rows=%d  edits=%d" % (len(back), len(edits)))
for e in edits:
    print("   %-22s %-20s %s -> %s" % e)
print("[fix_d9_scope] readback OK  (backup: %s.pred9.bak)" % CSV)

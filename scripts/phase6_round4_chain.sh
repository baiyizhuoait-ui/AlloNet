#!/bin/bash
# Phase 6 Round 4 chain -- BOTTLENECK-AWARE ASYMMETRIC ARCHITECTURE (final round).
#
# PREREGISTRATION: experiments/phase6/phase6_round4_preregistration.md  (READ FIRST)
# COST MODEL (measured, no training): experiments/phase6/phase6_round4_cost.csv
# ISOLATION PROOF (no training):      scripts/phase6_round4_isolation_check.py
#
# Halt politely at ANY arm boundary:   touch experiments/phase6/round4/STOP_CHAIN
# Resume after a power cut is automatic: train.py checkpoints every epoch and the
# runner passes --resume, so the worst case is one lost epoch (~7 min), not an arm.
#
# 8 arms x 20ep @bs16, ~134-162 min each => ~19-20 h total.  Staged so that the
# decisive science (stage A) lands first: stages B-D only matter once A has been
# read, and the STOP flag can be dropped in between.
set -u
ROOT=~/ai_study/trac
cd "$ROOT" || exit 1
OUT=experiments/phase6/round4
FLAG=$OUT/STOP_CHAIN
mkdir -p "$OUT"
rm -f "$FLAG"

run () { # tag config epochs seed cell supervision
  if [ -f "$FLAG" ]; then echo "[r4] STOP_CHAIN set - halt before $1"; exit 0; fi
  echo "[r4] $(date '+%F %T') -> $1 ($5 seed$4)"
  bash scripts/phase6_round4_run.sh "$1" "$2" "$3" "$4" "$5" "$6" || {
    rc=$?
    if [ "$rc" = "2" ]; then echo "[r4] VRAM guard aborted $1 - stopping chain"; exit 2; fi
    echo "[r4] WARN $1 exited $rc - continuing to next arm"; }
}

echo "[r4] ===== ROUND 4 start $(date '+%F %T') ====="

# ---------------- Stage A: EXP-10 lane-repair cost ladder (decisive) ----------
# R1_uponly = 1/4 resolution with NO encoder lateral -> is resolution enough, or
#             is the new compute unit (the s1 1x1 lateral) actually required?
#             Architecture Design Rule 1: a compute unit must be justified or removed.
run R4R1up   configs/phase6_r4_R4_uponly14_z16.yaml 20 0 R4-R1up14     km    # 1.6129G
run R4R2thin configs/phase6_r4_R2_thin14_z16.yaml   20 0 R4-R2thin14   km    # 1.1656G  (= Model B)
run R4R3min  configs/phase6_r4_R3_min14_z16.yaml    20 0 R4-R3min14    km    # 1.0174G

# ---------------- Stage B: EXP-13 ablation, baseline cell to 3 seeds ----------
# C1 (lean + old anchors) had seed0 only; the +det/+lane/+both cells already have
# 3 seeds (round2/phase5 for +lane, phase6 combo20 for +both). This closes the 2x2.
run R4A0s1   configs/phase4a_r2_z16.yaml            20 1 R4-A0lean_old old
run R4A0s2   configs/phase4a_r2_z16.yaml            20 2 R4-A0lean_old old

# ---------------- Stage C: does the uniform model get the repair for free? ----
# U-minus = A-uniform encoder + thin 1/4 lane: 0.3224M / 1.6158G, i.e. cheaper on
# BOTH axes than Model U (0.3339M / 1.6650G) while carrying a 1/4 lane branch.
run R4Umin   configs/phase6_r4_U_min_km.yaml        20 0 R4-Umin_km    km

# ---------------- Stage D: Model U to 3 seeds --------------------------------
# seed0 already exists (experiments/phase6c/exp9_aunif_km, mAP50 0.5332).
run R4Us1    configs/phase6c_e9_aunif_km.yaml       20 1 R4-Ufloor_km  km
run R4Us2    configs/phase6c_e9_aunif_km.yaml       20 2 R4-Ufloor_km  km

echo "[r4] ===== ROUND 4 DONE $(date '+%F %T') ====="
echo "[r4] NEXT: scripts/phase6_round4_verdict.py (mechanical) -- not launched here."

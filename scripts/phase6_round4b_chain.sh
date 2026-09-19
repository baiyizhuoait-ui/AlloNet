#!/bin/bash
# Phase 6 Round 4B -- honestness repair + the rent/no-rent decider.
#
# PREREGISTRATION: experiments/phase6/phase6_round4_preregistration.md  §13 (READ FIRST)
# ISOLATION PROOF: scripts/phase6_round4b_isolation_check.py   (no training)
#
# Arm order is by DECISION VALUE, not by convenience:
#   Stage 1 decides whether renting a big GPU for a 100-epoch run is worth ANY money.
#   Stages 2-4 are the honesty/seed repairs that any verdict needs regardless.
#
# Halt politely at ANY arm boundary:   touch experiments/phase6/round4/STOP_CHAIN
# Resume after a power cut is automatic (train.py checkpoints every epoch).
set -u
ROOT=~/ai_study/trac
cd "$ROOT" || exit 1
OUT=experiments/phase6/round4
FLAG=$OUT/STOP_CHAIN
mkdir -p "$OUT"

# --- D10 FIX (2026-09-13 02:23) ------------------------------------------------
# bash reads a script file by BYTE OFFSET. The D9 fix edited scripts/phase6_round4_run.sh
# while this chain was still executing it, so at the next arm boundary bash resumed at a
# stale offset and died with `syntax error near unexpected token )`. R4R2s1 own 2.3h
# train+eval were unharmed; only its CSV registration was lost (repaired retroactively).
# The chain now runs a FROZEN COPY, so the normal engineering response to a defect -
# editing the working script - can no longer corrupt a run in flight.
SNAP=$OUT/run4_frozen.sh
cp -f scripts/phase6_round4_run.sh "$SNAP"
echo "[r4b] runner frozen -> $SNAP sha1=$(sha1sum "$SNAP" | cut -c1-8)"
# ------------------------------------------------------------------------------

# --- D11 FIX (2026-09-13 02:34): single-instance lock --------------------------
# Two chains were started ~90s apart because a launch whose wrapper was SIGTERM-ed had in
# fact survived (setsid detaches it). Both then drove the SAME outdir and the SAME
# train_<tag>.log, and tee() truncates on open: R4R2s2 lost its arm and R4A0o1 started
# twice. A chain that cannot prove it is the only one must not run (fail-closed).
exec 9>"$OUT/.round4b_chain.lock"
if ! flock -n 9; then
  echo "[r4b] FATAL: another round4b chain holds $OUT/.round4b_chain.lock - refusing to double-run"
  exit 3
fi
echo "[r4b] chain lock acquired (pid $$)"
# ------------------------------------------------------------------------------

run () { # tag config epochs seed cell supervision
  if [ -f "$FLAG" ]; then echo "[r4b] STOP_CHAIN set - halt before $1"; exit 0; fi
  echo "[r4b] $(date '+%F %T') -> $1 ($5 seed$4)"
  bash "$SNAP" "$1" "$2" "$3" "$4" "$5" "$6" || {
    rc=$?
    if [ "$rc" = "2" ]; then echo "[r4b] VRAM guard aborted $1 - stopping chain"; exit 2; fi
    echo "[r4b] WARN $1 exited $rc - continuing"; }
}

echo "[r4b] ===== ROUND 4B start $(date '+%F %T') ====="

# ---- Stage 1 (DECIDES THE RENTAL): epoch-scaling probe -----------------------
# Fresh 40-epoch run of Model B, seed 0, identical config, compared against the
# existing 20-epoch Model B. NOT a --resume extension on purpose: train.py builds
# CosineAnnealingLR with T_max = epochs * steps and restores sched_state on resume,
# so resuming 20->40 would run 20 further epochs at the LR FLOOR and fabricate a
# "no headroom" answer. A fresh run has a correctly-scoped cosine over 40 epochs;
# the 20 vs 40 endpoints are both converged, which is what the question needs.
# Cost ~4.7 h.
run R4R2thin40 configs/phase6_r4_R2_thin14_z16.yaml  40 0 R5-R2thin40   km

# ---- Stage 2: is the 20-epoch number OPTIMISATION-limited? -------------------
# lr 1e-3 -> 1e-4, everything else identical. ~2.4 h.
run R4R2thinL4 configs/phase6_r4_R2thin_lr1e4_z16.yaml 20 0 R6-R2thinL4  km

# ---- Stage 3: Model B to 3 seeds (Level-3 seed requirement) ------------------
run R4R2s1   configs/phase6_r4_R2_thin14_z16.yaml 20 1 R4-R2thin14 km
run R4R2s2   configs/phase6_r4_R2_thin14_z16.yaml 20 2 R4-R2thin14 km

# ---- Stage 4: EXP-13 C1 with the HISTORICAL anchors actually pinned ----------
# The earlier R4-A0lean_old rows are retracted to R4-A0lean_km in the CSV by
# scripts/phase6_round4b_fix_d7.py (config archaeology + metric signature).
run R4A0o1   configs/phase6_r4_lean_old.yaml 20 1 R4-A0lean_old old
run R4A0o2   configs/phase6_r4_lean_old.yaml 20 2 R4-A0lean_old old

echo "[r4b] ===== ROUND 4B DONE $(date '+%F %T') ====="

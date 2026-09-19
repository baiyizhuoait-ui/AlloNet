#!/usr/bin/env bash
# Phase 6 -- G1 budget arm: a 200-epoch FRESH run, sequenced AFTER the seed chain.
#
#   Usage:  bash scripts/phase6_g1_budget200.sh            # wait, then run
#           bash scripts/phase6_g1_budget200.sh --dry-run  # show the plan, no wait
#           bash scripts/phase6_g1_budget200.sh --now      # run NOW; refuse if the chain holds the lock
#
#   --now exists because a human is now the scheduler. The default mode polls the chain's
#   lock for up to 36 h and then starts a 24 h job by itself -- fine when the intent is
#   "queue behind the seeds", but a footgun if you meant "start it now and only now":
#   you would get no signal that it had quietly begun six hours later. --now refuses
#   instead of waiting, so the start is always something you actually saw happen.
#
# WHY A WAITER AND NOT A CHAIN EDIT
#   scripts/phase6_seeds_resume.sh is RUNNING right now (seeds 1 -> 2). Round-4 D10:
#   editing a script that bash is currently reading shifts its read offset and yields
#   a phantom failure. So the 200ep arm is NOT appended to the live chain; it waits on
#   the chain's own lock ($OUT/.d8_seeds.lock) and starts the moment the chain exits.
#
# WHY FRESH AND NOT --resume  (preregistration section 13.5, FROZEN)
#   CosineAnnealingLR(T_max = epochs * len(loader)) restores T_max AND last_epoch on
#   resume. Continuing 100 -> 200 would therefore run 100 more epochs on the LR floor
#   and FAKE convergence -- it would fabricate exactly the answer we are looking for.
#   A mid-run resume of the SAME 200ep arm is fine and is handled by the runner itself;
#   what is forbidden is stitching 100 onto 200.
#
# FROZEN DECISION RULE (G123 plan section 2.5, written before any 200ep data exists)
#   SATURATED      |d gap| over 100 -> 200 <  2 sigma_da  -> deficit is architectural
#   STILL CLOSING  |d gap| over 100 -> 200 >= 2 sigma_da  -> deficit is mostly budget
#   sigma_da is reported with BOTH rulers (frozen 0.0011 and the family-measured 0.0030);
#   the weaker claim wins. Both outcomes are publishable -- this is a low-risk arm.
set -u
cd ~/ai_study/trac || exit 1

PY=~/ai_study/gpu_env/bin/python
OUT=experiments/phase6/final
CFG=configs/phase6_r4_R2_thin14_z16.yaml
EPOCHS=200
SEED=0                 # the SAME seed as B100, so 100/200 ep is a like-for-like pair
CELL=B200
TAG=B200
L=$OUT/g1_budget200.log
DRY=0
NOW=0
for _a in "$@"; do
  case "$_a" in
    --dry-run) DRY=1 ;;
    --now)     NOW=1 ;;
    *) echo "usage: $0 [--dry-run|--now]" >&2; exit 64 ;;
  esac
done

say() { echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
mkdir -p "$OUT"

# --- idempotency: a finished 200ep arm is never re-run ------------------------
if [ -f "$OUT/$TAG/checkpoint.pt" ] && [ -f "$OUT/${TAG}_eval/metrics.json" ] && [ "$DRY" -eq 0 ]; then
  say "G1: $TAG already has a finished checkpoint + metrics -> nothing to do"
  exit 0
fi

say "==============================================================="
say "  G1 BUDGET ARM -- ${EPOCHS}ep FRESH -- seed=${SEED} (dry_run=${DRY})"
if [ "$NOW" -eq 1 ]; then
  say "  mode: --now (start here and now; refuse if $OUT/.d8_seeds.lock is held)"
else
  say "  mode: waiter (queue behind the seed chain; polls its lock)"
fi
say "==============================================================="

# --- take the SAME lock the seed chain holds: the two can never overlap --------
# (single 8 GB card; running two arms at once is what caused the 2026-09-14 10:17 OOM)
exec 9>"$OUT/.d8_seeds.lock"

if [ "$DRY" -eq 1 ]; then
  if flock -n 9; then
    say "dry-run: lock is FREE (no seed chain running at this instant)"
  else
    say "dry-run: lock is HELD by a live chain (expected while seeds 1/2 run)"
  fi
  say "dry-run: would then call  bash scripts/phase6_final100_run.sh $TAG $CFG $EPOCHS $SEED $CELL km"
  exit 0
fi

ACQ=0
if [ "$NOW" -eq 1 ]; then
  # manual mode: never wait. See the header -- a waiter that silently starts a 24 h job
  # hours later is a footgun once a human is the scheduler.
  if flock -n 9; then
    ACQ=1
    say "G1: --now given and the lock is FREE -> starting immediately."
  else
    say "G1: --now given but the seed chain still holds $OUT/.d8_seeds.lock."
    say "    NOT starting: two arms on one 8 GB card is the 2026-09-14 10:17 OOM."
    say "    Re-run once the chain has exited, or drop --now to queue behind it."
    exit 2
  fi
else
  for i in $(seq 1 2160); do                   # up to 36 h of 60 s polls
    if flock -n 9; then ACQ=1; break; fi
    if [ $((i % 30)) -eq 0 ]; then
      say "waiting (${i} min) -- live trainers: $(ps -eo cmd | grep -c '[t]rain\.py --config')"
    fi
    sleep 60
  done
  if [ "$ACQ" -ne 1 ]; then
    say "G1: gave up after 36 h -- the chain still holds the lock. NOT starting."
    exit 1
  fi
  say "G1: lock acquired -> the seed chain has exited."
fi

# --- record the seed-chain end state (the 200ep point is only interpretable ----
# --- alongside a complete 100ep point, so log what actually happened) ----------
LAST2=$(grep -oE "ep [0-9]+ DONE" "$OUT/B100_s2/training_log.txt" 2>/dev/null \
        | grep -oE "[0-9]+" | sort -n | tail -1)
LAST2=${LAST2:-0}
LAST1=$(grep -oE "ep [0-9]+ DONE" "$OUT/B100_s1/training_log.txt" 2>/dev/null \
        | grep -oE "[0-9]+" | sort -n | tail -1)
LAST1=${LAST1:-0}
say "G1: seed chain end state -- B100_s1 last ep DONE = ${LAST1}, B100_s2 last ep DONE = ${LAST2}"
if [ "$LAST2" -lt "$EPOCHS" ] && [ "$LAST2" -lt 100 ]; then
  say "G1: NOTE seed2 did not reach 100ep. The 200ep arm is still valid on its own"
  say "    (it is compared against B100 seed0), but the 100ep error bar will be thin."
  say "    Audit the chain log: $OUT/seeds_resume.log"
fi

# --- wait for real memory headroom before a 24 h run --------------------------
AVAIL=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
for i in $(seq 1 24); do                        # up to 2 h
  if [ "$AVAIL" -ge 6000 ]; then break; fi
  say "G1: MemAvailable=${AVAIL} MiB (< 6000) -- waiting 5 min (attempt ${i}/24)"
  sleep 300
  AVAIL=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
done
if [ "$AVAIL" -lt 6000 ]; then
  say "G1: only ${AVAIL} MiB available after 2 h -- refusing to start."
  say "    (the 2026-09-14 10:17 OOM killed two arms this way; fail closed, do not risk it)"
  exit 1
fi
say "G1: MemAvailable = ${AVAIL} MiB -- headroom OK"

# --- launch -------------------------------------------------------------------
say "G1: launching  bash scripts/phase6_final100_run.sh $TAG $CFG $EPOCHS $SEED $CELL km"
bash scripts/phase6_final100_run.sh "$TAG" "$CFG" "$EPOCHS" "$SEED" "$CELL" km 2>&1 | tee -a "$L"
RC=$?
say "G1: runner rc=$RC"

# --- post-run report (does NOT judge -- the G123 plan owns the verdict) --------
CK="$OUT/$TAG/checkpoint.pt"
EPOK=$("$PY" - "$CK" <<'PYEOF' 2>/dev/null || echo -1
import sys, os, torch
p = sys.argv[1]
print(torch.load(p, map_location="cpu", weights_only=False)["epoch"] if os.path.exists(p) else -1)
PYEOF
)
EPOK=${EPOK:--1}
say "G1: final checkpoint epoch = ${EPOK} / ${EPOCHS}"
if [ -f "$OUT/${TAG}_eval/metrics.json" ]; then
  "$PY" - "$OUT/${TAG}_eval/metrics.json" <<'PYEOF' 2>&1 | tee -a "$L"
import json, sys
m = json.load(open(sys.argv[1]))
for k in ("parameters", "flops", "mAP50", "da_mIoU", "da_fg_iou", "lane_mIoU", "lane_fg_iou"):
    v = m.get(k)
    print(f"    {k:14s} = {v}")
PYEOF
  say "G1: next -> ~/ai_study/gpu_env/bin/python scripts/phase6_g1_budget_curve.py"
else
  say "G1: WARNING no metrics.json -- the arm did not complete; nothing to add to the curve"
fi
say "G1: BUDGET ARM DONE"

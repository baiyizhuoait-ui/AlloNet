#!/usr/bin/env bash
# Phase 6 -- RESTART / RESUME path for the FINAL-100 seed arms.
#
#   Usage:  bash scripts/phase6_seeds_resume.sh [--dry-run] [seeds...]
#           bash scripts/phase6_seeds_resume.sh            # both seeds (1 2)
#           bash scripts/phase6_seeds_resume.sh 1          # seed 1 only
#
# WHY A SEPARATE SCRIPT
#   phase6_d8_then_seeds.sh is the ONE-SHOT chain: probe -> seed1 -> seed2. It is
#   the wrong entry point for a restart, for three reasons, all of them fixed here.
#   (Round 4 D10: the running chain's file must never be edited -- a modified
#   script shifts bash's read offset and yields a phantom failure. So this is a
#   NEW file, and the old chain keeps running untouched.)
#
#   FIX 1  the probe stage is NOT re-run. The D8 measurement is already certified
#          (experiments/phase6/final/d8_latency/{raw.json,summary.md}); it also
#          REQUIRES an idle, cool card, so re-running it after a training session
#          would replace a valid measurement with an invalid one.
#   FIX 2  a row written as `...,failed-no-metrics,` for this (cell,seed) is
#          QUARANTINED before the runner starts. Otherwise the runner's own
#          idempotency guard ("skip a cell already appended") would treat the
#          failure record as a completed run and silently SKIP the seed forever,
#          leaving a bogus row in the results CSV.
#   FIX 3  the "trained to the last epoch but eval never produced metrics.json"
#          livelock is broken. In that state --resume makes start_epoch > epochs,
#          train.py's epoch loop never runs, `avg` is never bound, and
#          training/train.py:486 raises NameError BEFORE metrics.json is written;
#          the runner's VRAM guard then aborts (no mem samples in a log with zero
#          epochs) and eval never runs. Every subsequent restart repeats it. Here
#          the finished checkpoint is simply evaluated directly with the SAME
#          evaluator the runner uses, then the runner is invoked, sees the
#          metrics, and appends the row.
#
#   Nothing else in the training path changes: same runner, same config, same
#   epochs, same checkpoint format. The resume itself is the runner's own
#   `--resume "$CKPT"` (scripts/phase6_final100_run.sh), which restores model,
#   optimizer, cosine schedule and all five RNG streams (training/train.py:424-455).
#   Worst case loss on an interruption = ONE epoch (~7 min), never the whole arm.
set -u
cd ~/ai_study/trac || exit 1
PY=~/ai_study/gpu_env/bin/python
CFG=configs/phase6_r4_R2_thin14_z16.yaml
OUT=experiments/phase6/final
CSV=experiments/phase6/final_results.csv
QF=experiments/phase6/final_results.csv.quarantined_failed_rows
EPOCHS=100
CELL=B100
L=$OUT/seeds_resume.log

DRY=0
SEEDS=()
for a in "$@"; do
  case "$a" in
    --dry-run|-n) DRY=1 ;;
    *) SEEDS+=("$a") ;;
  esac
done
[ ${#SEEDS[@]} -gt 0 ] || SEEDS=(1 2)

say() { echo "[$(date '+%F %T')] $*" | tee -a "$L"; }

mkdir -p "$OUT"

# --- same lock file as the one-shot chain: a restart can never collide with a
# --- still-live chain (Round 4 D11). flock is released when the holder dies.
if [ "$DRY" -eq 0 ]; then
  exec 9>"$OUT/.d8_seeds.lock"
  if ! flock -n 9; then
    echo "[resume] another chain holds $OUT/.d8_seeds.lock -- refusing to start."
    echo "         Check: ps -eo pid,etime,cmd | grep -E 'd8_then_seeds|phase6_final100_run|train.py'"
    exit 1
  fi
fi

say "==============================================================="
say "  SEEDS RESUME  seeds=[${SEEDS[*]}]  dry_run=$DRY"
say "==============================================================="

for S in "${SEEDS[@]}"; do
  TAG="B100_s$S"
  TR="$OUT/$TAG"
  EVL="${TR}_eval"
  CKPT="$TR/checkpoint.pt"
  MP="$EVL/metrics.json"
  TLOG="$OUT/train_${TAG}.log"

  say "--- seed $S ($TAG) ---"

  # --- already completed? (a real row carries source=trained) ---------------
  if [ -f "$CSV" ] && awk -F, -v c="$CELL" -v s="$S" \
       'NR>1 && $2==c && $18==s && $19=="trained"{f=1} END{exit !f}' "$CSV"; then
    say "seed $S: COMPLETE row already in $CSV -> nothing to do"
    continue
  fi

  # --- FIX 2: quarantine a failure row so the idempotency guard can't skip ---
  if [ "$DRY" -eq 0 ] && [ -f "$CSV" ]; then
    "$PY" - "$CSV" "$QF" "$CELL" "$S" <<'PYEOF' 2>&1 | tee -a "$L"
import csv, sys
csvp, qf, cell, seed = sys.argv[1:5]
rows = list(csv.reader(open(csvp, newline="")))
hdr, body = rows[0], rows[1:]
keep, quar = [], []
for r in body:
    if len(r) >= 19 and r[1] == cell and r[17] == str(seed) and "failed" in r[18]:
        quar.append(r)
    else:
        keep.append(r)
if quar:
    with open(qf, "a", newline="") as fh:
        csv.writer(fh).writerows(quar)
    with open(csvp, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr); w.writerows(keep)
    print(f"[resume] QUARANTINED {len(quar)} failed row(s) for {cell}/seed{seed} -> {qf}"
          " (they would have made the runner skip this seed forever)")
else:
    print(f"[resume] no failed row for {cell}/seed{seed}; idempotency guard clean")
PYEOF
  fi

  # --- what state is this arm in? ------------------------------------------
  LAST_DONE=$(grep -oE "ep [0-9]+ DONE" "$TR/training_log.txt" 2>/dev/null \
              | grep -oE "[0-9]+" | sort -n | tail -1)
  LAST_DONE=${LAST_DONE:-0}
  if [ -f "$MP" ]; then
    say "seed $S: metrics.json present -> runner will reuse the eval and append the row"
    STATE="eval-reuse"
  elif [ ! -f "$CKPT" ]; then
    say "seed $S: no checkpoint -> fresh ${EPOCHS}-ep run (~12 h)"
    STATE="fresh"
  elif [ "$LAST_DONE" -ge "$EPOCHS" ]; then
    say "seed $S: FIX 3 -- checkpoint reached ep${LAST_DONE} but metrics.json is missing."
    say "seed $S: evaluating the finished checkpoint directly (same evaluator as the runner)"
    if [ "$DRY" -eq 0 ]; then
      "$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" \
        --outdir "$EVL" 2>&1 | tee -a "${EVL}.log" > "$EVL/eval_stdout.log"
      [ -f "$MP" ] && say "seed $S: metrics.json rebuilt by direct eval" \
                   || say "seed $S: WARNING direct eval produced no metrics.json -- runner will report it"
    fi
    STATE="eval-repair"
  else
    say "seed $S: RESUMING from ep${LAST_DONE} -> ${EPOCHS} (lost <= 1 epoch of work)"
    STATE="resume"
  fi

  # --- checkpoint integrity (torch.save is not atomic and there is no rotation,
  # --- so a kill landing inside the write would destroy the only resume point) ---
  if [ -f "$CKPT" ] && [ "$DRY" -eq 0 ]; then
    if "$PY" - "$CKPT" <<'PYEOF' >/dev/null 2>&1
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert isinstance(c, dict) and "epoch" in c and "model_state" in c
PYEOF
    then
      say "seed $S: checkpoint.pt loads OK; refreshed the .safe snapshot"
      cp -f "$CKPT" "$CKPT.safe"
    elif [ -f "$CKPT.safe" ]; then
      say "seed $S: checkpoint.pt is UNREADABLE (torn write) -> restored from $CKPT.safe"
      cp -f "$CKPT.safe" "$CKPT"
    else
      say "seed $S: checkpoint.pt is UNREADABLE and there is no .safe copy -> only a fresh run is possible"
    fi
  fi

  # --- WORKAROUND: the runner passes a RELATIVE --resume, and train.py joins it
  # --- onto the (also relative) outdir, so the path arrives DOUBLED and the
  # --- resume silently degrades to "training fresh from epoch 1" (see the round-1
  # --- evidence in scripts/phase6_resume_selftest.sh, and the analysis in
  # --- phase6_safe_stop.sh). Staging a copy at the path train.py will actually
  # --- open makes the resume work WITHOUT editing the runner. Delete this block
  # --- if the runner is ever changed to pass an absolute path ($ROOT/$CKPT).
  if [ "$STATE" = "resume" ] && [ "$DRY" -eq 0 ]; then
    DOUBLED="$TR/$CKPT"
    mkdir -p "$(dirname "$DOUBLED")"
    cp -f "$CKPT" "$DOUBLED"
    say "seed $S: staged $DOUBLED so the runner's relative --resume resolves"
  fi

  if [ "$DRY" -eq 1 ]; then
    say "seed $S: [dry-run] would call: bash scripts/phase6_final100_run.sh $TAG $CFG $EPOCHS $S $CELL km   (state=$STATE)"
    continue
  fi

  say "seed $S: launching  bash scripts/phase6_final100_run.sh $TAG $CFG $EPOCHS $S $CELL km"
  bash scripts/phase6_final100_run.sh "$TAG" "$CFG" "$EPOCHS" "$S" "$CELL" km 2>&1 | tee -a "$L"
  RC=$?
  say "seed $S: runner rc=$RC"
done

# --- post-run audit ----------------------------------------------------------
# A `trained` row is only legitimate if the arm actually finished. The runner has
# no completion check, so a run that dies mid-way (OOM, kill) can still be
# evaluated on its PARTIAL checkpoint and appended as `epochs=100, source=trained`.
# The authoritative test is the final checkpoint's own epoch counter: if it is
# below the epoch budget, the row describes a model that does not exist.
for S in "${SEEDS[@]}"; do
  TAG="B100_s$S"
  TR="$OUT/$TAG"
  CKPT="$TR/checkpoint.pt"
  MP="${TR}_eval/metrics.json"
  ROW=$(awk -F, -v t="$TAG" -v c="$CELL" -v e="$EPOCHS" -v s="$S" \
        'NR>1 && $1==t && $2==c && $5==e && $18==s && $19=="trained"' "$CSV" 2>/dev/null | head -1)
  [ -n "$ROW" ] || { say "audit seed $S: no trained row (arm did not complete)"; continue; }
  EPOK=$("$PY" - "$CKPT" <<'PYEOF' 2>/dev/null
import sys, os, torch
p = sys.argv[1]
print(torch.load(p, map_location="cpu", weights_only=False)["epoch"] if os.path.exists(p) else -1)
PYEOF
)
  EPOK=${EPOK:--1}
  if [ "$EPOK" -lt "$EPOCHS" ] || [ ! -f "$MP" ]; then
    say "audit seed $S: *** FABRICATED ROW *** checkpoint epoch=$EPOK (< $EPOCHS) or metrics missing."
    say "audit seed $S: quarantining the row -> $QF"
    echo "$ROW" >> "$QF"
    awk -F, -v t="$TAG" -v c="$CELL" -v s="$S" \
      'NR==1 || !($1==t && $2==c && $18==s)' "$CSV" > "$CSV.tmp" && mv "$CSV.tmp" "$CSV"
  else
    say "audit seed $S: row is backed by a finished ${EPOK}-epoch checkpoint + metrics.json -- OK"
  fi
done

say "--- CSV rows for $CELL ---"
grep -E "^${TAG}|^${CELL}," "$CSV" 2>/dev/null | tail -5 | tee -a "$L"
say "SEEDS RESUME DONE"

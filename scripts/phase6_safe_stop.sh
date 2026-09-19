#!/usr/bin/env bash
# Phase 6 -- SAFE STOP for a running FINAL-100 chain.
#
#   bash scripts/phase6_safe_stop.sh           # DRY: show what would be stopped, change nothing
#   bash scripts/phase6_safe_stop.sh --yes     # actually stop
#
# WHY NOT JUST `touch STOP_CHAIN` ?
#   STOP_CHAIN is only tested at the TOP of the runner, i.e. BEFORE an arm starts.
#   Setting it while seed 1 trains does NOT interrupt seed 1 -- it only prevents
#   seed 2 from starting. To stop mid-arm you must kill processes, and the naive
#   way (`pkill train.py`) is DANGEROUS. Here is what happens if you do that:
#
#     the runner is blocked on `python train.py | tee`, NOT on a completion check.
#     Kill python and the runner simply carries on: the VRAM guard reads whatever
#     mem samples the truncated log has, the eval runs on the PARTIAL checkpoint
#     (say epoch 31 of 100), metrics.json gets written, and the CSV receives a row
#     labelled `epochs=100, source=trained`. A 31-epoch model would be published as
#     a 100-epoch result. (Seen in miniature on 2026-09-14 10:17:57, when an OOM
#     kill of seed 1 produced exactly that kind of row -- with `failed-no-metrics`
#     instead of `trained` only because no checkpoint existed yet to evaluate.)
#
#   So the order below is deliberate: the RUNNER dies first, the chain second, and
#   the trainer last, so nothing is left alive that can turn a dead run into a row.
#
# WHAT IS PRESERVED
#   train.py writes outdir/checkpoint.pt at the end of every epoch (model weights,
#   optimizer, cosine schedule, all five RNG streams). Stopping loses at most the
#   epoch in flight (~7 min for FINAL-100), never the arm. This script additionally
#   verifies the checkpoint LOADS and keeps a `.safe` copy, because torch.save is
#   not atomic -- a kill landing inside the write would otherwise destroy the only
#   resumable state (there is no checkpoint rotation).
set -u
cd ~/ai_study/trac || exit 1
PY=~/ai_study/gpu_env/bin/python
OUT=experiments/phase6/final
CSV=experiments/phase6/final_results.csv
L=$OUT/safe_stop.log

DOIT=0
for a in "$@"; do case "$a" in --yes|-y) DOIT=1 ;; esac; done

say() { echo "[$(date '+%F %T')] $*" | tee -a "$L"; }

# Anchored patterns: the local shell that runs this script has a cmdline of the
# form "bash -lc cd /home/...", which contains these strings but does not START
# with them, so ^ keeps pkill from killing its own caller.
# The trainer pattern is deliberately narrow: it matches ONLY the FINAL-100 arms
# (outdir experiments/phase6/final/B100*), so a CPU self-test running out of
# experiments/phase6/resume_selftest is reported but never killed by this script.
TRAINER_PAT='^~/ai_study/gpu_env/bin/python training/train\.py --config configs/phase6_r4_R2_thin14_z16\.yaml --outdir experiments/phase6/final/B100'
PATS=(
  '^bash scripts/phase6_final100_run\.sh'
  '^bash scripts/phase6_d8_then_seeds\.sh'
  '^bash scripts/phase6_seeds_resume\.sh'
  "$TRAINER_PAT"
  '^~/ai_study/gpu_env/bin/python scripts/phase6_d8_latency_probe\.py'
)

list_pids() {  # print all pids matching any pattern
  local p
  for p in "${PATS[@]}"; do pgrep -f "$p" 2>/dev/null; done | sort -u
}

echo "=== what is running that this script would stop ==="
FOUND=$(list_pids)
if [ -z "$FOUND" ]; then
  echo "  (nothing -- no chain / runner / trainer alive)"
else
  for p in $FOUND; do
    printf "  pid %-7s etime %-9s %s\n" "$p" "$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')" \
      "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-110)"
  done
fi
echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader

if [ "$DOIT" -eq 0 ]; then
  echo
  echo "DRY RUN -- nothing was touched.  Add --yes to actually stop."
  echo "Cost of stopping: at most the epoch in flight (~7 min); the arm resumes later."
  exit 0
fi

say "==== SAFE STOP requested ===="
[ -f "$CSV" ] && CSV_BEFORE=$(wc -l < "$CSV") || CSV_BEFORE=0
PY_ROWS_BEFORE=$(pgrep -fc "$TRAINER_PAT" 2>/dev/null || echo 0)
say "pre-kill: CSV lines=$CSV_BEFORE  trainer procs=$PY_ROWS_BEFORE"

# --- 1. protect the resume point BEFORE anything is killed -------------------
# torch.save is not atomic, so grab (and verify) the current checkpoint while the
# trainer is still healthy: this becomes the fallback if the kill tears the file.
for CK in $(ls -d experiments/phase6/final/B100_s*/checkpoint.pt \
                experiments/phase6/final/B100/checkpoint.pt 2>/dev/null); do
  ok=0
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if "$PY" - "$CK" <<'PYEOF' >/dev/null 2>&1
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert isinstance(c, dict) and "epoch" in c and "model_state" in c
PYEOF
    then ok=1; break; fi
    sleep 0.5
  done
  if [ "$ok" = 1 ]; then
    cp -f "$CK" "$CK.safe"
    say "verified+snapshotted $CK (epoch=$(PYTHONPATH= "$PY" -c "
import torch;print(torch.load('$CK',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null)) -> $CK.safe"
  else
    say "WARNING $CK does not load even before the kill (checkpoint may already be damaged)"
  fi
done

# --- 2. kill in the order that cannot fabricate a result row ----------------
kill_pat() {  # $1 = pattern, $2 = label
  local pids
  pids=$(pgrep -f "$1" 2>/dev/null || true)
  [ -z "$pids" ] && { say "  $2: none alive"; return 0; }
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null
  say "  $2: SIGKILL -> $pids"
}
say "R1 kill the RUNNER first (so it can never eval a partial checkpoint and append a row)"
kill_pat '^bash scripts/phase6_final100_run\.sh' "runner"
sleep 1
say "R2 kill the CHAIN second (releases the flock so a restart can take it)"
kill_pat '^bash scripts/phase6_d8_then_seeds\.sh' "chain"
kill_pat '^bash scripts/phase6_seeds_resume\.sh' "resume-chain"
sleep 1
say "R3 kill the TRAINER last"
kill_pat "$TRAINER_PAT" "trainer"
kill_pat '^~/ai_study/gpu_env/bin/python scripts/phase6_d8_latency_probe\.py' "probe"

# --- 3. verify, and repair a torn checkpoint --------------------------------
sleep 2
LEFT=$(list_pids)
if [ -z "$LEFT" ]; then say "verified: no chain/runner/trainer processes remain"; else
  say "WARNING still alive: $LEFT"; fi

for CK in $(ls -d experiments/phase6/final/B100_s*/checkpoint.pt 2>/dev/null); do
  if "$PY" - "$CK" <<'PYEOF' >/dev/null 2>&1
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert isinstance(c, dict) and "epoch" in c and "model_state" in c
PYEOF
  then say "post-kill: $CK loads OK (epoch $(PYTHONPATH= "$PY" -c "
import torch;print(torch.load('$CK',map_location='cpu',weights_only=False)['epoch'])" 2>/dev/null))"
  elif [ -f "$CK.safe" ]; then
    cp -f "$CK.safe" "$CK"; say "post-kill: $CK was TORN by the kill -> restored from $CK.safe"
  else say "post-kill: $CK is unreadable and no .safe copy exists"; fi
done

# --- 4. did anything get written into the results CSV after all? -------------
CSV_AFTER=$(wc -l < "$CSV" 2>/dev/null || echo 0)
if [ "$CSV_AFTER" != "$CSV_BEFORE" ]; then
  say "*** ANOMALY: CSV grew $CSV_BEFORE -> $CSV_AFTER during the stop. Inspect the last line: ***"
  tail -n 2 "$CSV" | tee -a "$L"
  say "*** a row added now would describe a run that did not finish -- remove it before publishing ***"
else
  say "CSV unchanged at $CSV_AFTER lines (no fabricated row)"
fi

say "=== STOPPED. To continue later:  bash scripts/phase6_seeds_resume.sh 1 2 ==="
say "=== (that restarts from the last completed epoch; the probe is NOT re-run) ==="

#!/usr/bin/env bash
# Phase 6 -- reconnect the seed arms the moment the consistency sweep ends.
#
#   Why: the user's sequence is "pause seed1 -> run consistency -> reconnect seed1".
#   Holding the GPU open between the two stages wastes wall clock for no gain, so
#   this watcher waits for the sweep's python to exit and then starts the normal
#   resume chain (scripts/phase6_seeds_resume.sh), which itself is idempotent:
#   it resumes seed 1 from its last completed epoch and then runs seed 2.
#
#   It is a NEW file on purpose: scripts/phase6_consistency_stageB.sh is still
#   being read by a live bash, and editing a running script shifts bash's read
#   offset (Round-4 D10 -- that is exactly how a phantom failure was produced
#   once already).  Never edit this file while it is running either.
#
#   It deliberately does NOT abort if the sweep failed: the user's instruction is
#   "consistency first, then seed1", and a failed sweep is still a finished
#   sweep.  Whatever it produced is listed in the log for inspection.
set -u
cd ~/ai_study/trac || exit 1
D=experiments/phase6/consistency
L=$D/auto_resume.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$L"; }

mkdir -p "$D"
say "watcher up -- waiting for the consistency sweep to exit"
while pgrep -f 'scripts/phase6_consistency_official\.py' >/dev/null 2>&1; do sleep 15; done
say "consistency sweep has exited"

for f in "$D/stageB_in384/consistency.json" "$D/stageA_640/consistency.json"; do
  if [ -f "$f" ]; then say "  artifact present: $f"; else say "  artifact MISSING: $f"; fi
done
if [ -f "$D/stageB_in384/consistency.md" ]; then
  say "--- stageB_in384/consistency.md ---"
  cat "$D/stageB_in384/consistency.md" >> "$L"
fi

say "GPU before relaunch: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
say "relaunching the seed arms: bash scripts/phase6_seeds_resume.sh 1 2"
bash scripts/phase6_seeds_resume.sh 1 2 2>&1 | tee -a "$L"
say "seeds_resume exited rc=$?"

#!/usr/bin/env bash
# Phase 6 -- FINAL-100 follow-ups, ordered shortest-first (PI instruction, 2026-09-14).
#
#   STAGE 1  D8 latency probe          zero training, ~15-20 min, fail-closed on a shared GPU
#   STAGE 2  B100 seed 1               100 ep, ~12 h
#   STAGE 3  B100 seed 2               100 ep, ~12 h
#
# Order rationale: the latency probe MUST run on an idle, cool card, and it is the
# shortest job, so it goes first -- after 24 h of training the GPU would be hot and
# the measurement conditions would no longer be the ones we want to certify.
# It also must never overlap a training chain (Round 4 D9/D11: both incidents were
# same-card contention, and the two worst latency readings in that round are
# exactly the arms that were running during the D11 double-chain collision).
#
# Seeds 1 and 2 are separate stages with separate exit codes and separate log
# banners, so their durations and outcomes are reported independently.
#
# The runner is NOT modified here -- Round 4 D10: never edit a script a chain is
# executing; a modified file shifts bash's read offset and produces a phantom failure.
set -u
cd ~/ai_study/trac || exit 1
PY=~/ai_study/gpu_env/bin/python
OUT=experiments/phase6/final
L=$OUT/d8_then_seeds.log

mkdir -p "$OUT"

# single-instance lock (Round 4 D11)
exec 9>"$OUT/.d8_seeds.lock"
if ! flock -n 9; then
  echo "[chain] another instance holds $OUT/.d8_seeds.lock -- refusing to start"
  exit 1
fi

rm -f "$OUT/STOP_CHAIN" "$OUT/ABORTS.txt"

{
  echo "================================================================"
  echo "  [$(date '+%F %T')] CHAIN START   D8 probe -> seed 1 -> seed 2"
  echo "================================================================"
} >> "$L"

echo "===== [$(date '+%F %T')] STAGE 1/3  D8 latency probe =====" >> "$L"
"$PY" scripts/phase6_d8_latency_probe.py >> "$L" 2>&1
RC1=$?
echo "----- [$(date '+%F %T')] STAGE 1/3 D8 probe rc=$RC1 -----" >> "$L"

for S in 1 2; do
  echo "===== [$(date '+%F %T')] STAGE seed $S  (100 ep, fresh) =====" >> "$L"
  bash scripts/phase6_final100_run.sh "B100_s$S" \
      configs/phase6_r4_R2_thin14_z16.yaml 100 "$S" B100 km >> "$L" 2>&1
  RC=$?
  echo "----- [$(date '+%F %T')] STAGE seed $S rc=$RC -----" >> "$L"
done

echo "===== [$(date '+%F %T')] CHAIN DONE =====" >> "$L"

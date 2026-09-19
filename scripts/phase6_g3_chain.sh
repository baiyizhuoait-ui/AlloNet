#!/bin/bash
# Phase 6 / G3 chain: run the cross-architecture allocation cells in order.
#
#   ab         : stage A (2 cells x 4ep pipeline probe, own output dir) then the
#                core 6 cells x 20ep.  Recommended entry point.
#   core       : 6 cells -- 2 architectures x {base z16, z-spend z128, enc-spend z16}
#   plus-seeds : core + 4 cells (seeds 1,2 on each architecture's base cell), which
#                is what re-calibrates the noise ruler on the second architecture.
#                The 跨架构验证预审 skill forbids reusing arch-1's ruler here.
#
# Guard: never run while the seed chain owns the GPU.  Two processes training on
# one 8 GB card is exactly the 2026-09-14 10:17 OOM that killed both seeds.
#
# Usage:
#   bash scripts/phase6_g3_chain.sh ab --wait
#   bash scripts/phase6_g3_chain.sh plus-seeds --now
set -u
ROOT=~/ai_study/trac
cd "$ROOT" || exit 1

MODE="${1:-core}"
WHEN="${2:---wait}"
OUT=experiments/phase6/g3
LOCK=$OUT/.g3.lock
D8_LOCK=experiments/phase6/final/.d8_seeds.lock
mkdir -p "$OUT"

case "$MODE" in
  core|plus-seeds|ab) ;;
  *) echo "[g3] unknown mode '$MODE' (core|plus-seeds|ab)"; exit 64 ;;
esac
case "$WHEN" in
  --wait|--now) ;;
  *) echo "[g3] unknown flag '$WHEN' (--wait|--now)"; exit 64 ;;
esac

# --- GPU availability gate --------------------------------------------------
gpu_busy() {
  pgrep -f "training/train\.py" >/dev/null 2>&1 && return 0
  pgrep -f "evaluation/evaluate_baseline\.py" >/dev/null 2>&1 && return 0
  # The seed chain holds its lock with flock; the FILE it locks stays on disk
  # after the holder exits, so testing for existence would block G3 forever.
  # Probe the lock itself instead.
  if [ -f "$D8_LOCK" ]; then
    flock -n "$D8_LOCK" -c true >/dev/null 2>&1 || return 0
  fi
  return 1
}

if gpu_busy; then
  if [ "$WHEN" = "--now" ]; then
    echo "[g3] REFUSED: the GPU is busy (seed chain training or $D8_LOCK present)."
    echo "     Re-run with --wait, or wait for the seed chain to finish."
    echo "     Two concurrent trainers on an 8 GB card is what OOMed on 2026-09-14."
    exit 2
  fi
  echo "[g3] waiting for the GPU to free up (poll every 60 s, max 8 h)..."
  WAITED=0
  while gpu_busy; do
    sleep 60
    WAITED=$((WAITED + 1))
    if [ "$WAITED" -ge 480 ]; then
      echo "[g3] gave up after 8 h of waiting"; exit 2
    fi
    [ $((WAITED % 10)) -eq 0 ] && echo "[g3]   still waiting (${WAITED} min)"
  done
  echo "[g3] GPU free after ${WAITED} min"
fi

# --- single-instance lock ---------------------------------------------------
if [ -f "$LOCK" ]; then
  HOLDER=$(cat "$LOCK" 2>/dev/null || echo "?")
  if kill -0 "$HOLDER" 2>/dev/null; then
    echo "[g3] REFUSED: another G3 chain (pid ${HOLDER}) holds $LOCK"; exit 2
  fi
  echo "[g3] stale lock from pid ${HOLDER} - taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

CELLS=(
  "g3_a1_base_z16|configs/phase6_g3_a1_base_z16.yaml|20|0|a1_base|dws|base|16"
  "g3_a2_base_z16|configs/phase6_g3_a2_base_z16.yaml|20|0|a2_base|ir|base|16"
  "g3_a1_zspend_z128|configs/phase6_g3_a1_zspend_z128.yaml|20|0|a1_zspend|dws|zspend|128"
  "g3_a1_encspend_z16|configs/phase6_g3_a1_encspend_z16.yaml|20|0|a1_encspend|dws|encspend|16"
  "g3_a2_zspend_z128|configs/phase6_g3_a2_zspend_z128.yaml|20|0|a2_zspend|ir|zspend|128"
  "g3_a2_encspend_z16|configs/phase6_g3_a2_encspend_z16.yaml|20|0|a2_encspend|ir|encspend|16"
)
if [ "$MODE" = "plus-seeds" ]; then
  CELLS+=(
    "g3_a1_base_z16_s1|configs/phase6_g3_a1_base_z16.yaml|20|1|a1_base|dws|base|16"
    "g3_a1_base_z16_s2|configs/phase6_g3_a1_base_z16.yaml|20|2|a1_base|dws|base|16"
    "g3_a2_base_z16_s1|configs/phase6_g3_a2_base_z16.yaml|20|1|a2_base|ir|base|16"
    "g3_a2_base_z16_s2|configs/phase6_g3_a2_base_z16.yaml|20|2|a2_base|ir|base|16"
  )
fi

# --- stage A: pipeline probe ------------------------------------------------
# 4 epochs on one cell per architecture.  Purpose: prove the new topology trains
# on the GPU at all (everything so far was CPU-only forward/contract checks),
# and MEASURE the real min/epoch instead of extrapolating it from seed0's
# 100-epoch wall time.  Written to its own directory and CSV on purpose.
if [ "$MODE" = "ab" ]; then
  PB=experiments/phase6/g3/probe4
  mkdir -p "$PB"
  echo "[g3] stage A: 4ep pipeline probe -> $PB" | tee -a "$OUT/g3_chain.log"
  for spec in \
    "g3_p4_a1_base|configs/phase6_g3_a1_base_z16.yaml|4|0|a1_base|dws|base|16" \
    "g3_p4_a2_base|configs/phase6_g3_a2_base_z16.yaml|4|0|a2_base|ir|base|16"
  do
    IFS='|' read -r TAG CFG EP SEED CELL ENC TIER Z <<< "$spec"
    G3_OUT="$PB" G3_CSV="$PB/g3_probe_results.csv" \
      bash scripts/phase6_g3_run.sh "$TAG" "$CFG" "$EP" "$SEED" "$CELL" \
           "${ENC}|${TIER}" "$Z"
    RC=$?
    if [ "$RC" -ne 0 ]; then
      echo "[g3] stage A ${TAG} FAILED rc=${RC} - stopping before stage B" \
          | tee -a "$OUT/g3_chain.log"
      exit "$RC"
    fi
  done
  echo "[g3] stage A complete $(date '+%F %T')" | tee -a "$OUT/g3_chain.log"
fi

echo "[g3] mode=$MODE cells=${#CELLS[@]} started $(date '+%F %T')" | tee -a "$OUT/g3_chain.log"
for spec in "${CELLS[@]}"; do
  IFS='|' read -r TAG CFG EP SEED CELL ENC TIER Z <<< "$spec"
  bash scripts/phase6_g3_run.sh "$TAG" "$CFG" "$EP" "$SEED" "$CELL" "${ENC}|${TIER}" "$Z"
  RC=$?
  if [ "$RC" -ne 0 ]; then
    echo "[g3] ${TAG} returned rc=${RC}; stopping the chain (see $OUT/ABORTS.txt)" \
        | tee -a "$OUT/g3_chain.log"
    exit "$RC"
  fi
done
echo "[g3] chain complete $(date '+%F %T')" | tee -a "$OUT/g3_chain.log"

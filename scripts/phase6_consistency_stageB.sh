#!/usr/bin/env bash
# Phase 6 -- STAGE B of the official-GT consistency sweep.
#
#   bash scripts/phase6_consistency_stageB.sh [input] [num_images]
#       input      640 | 384          (default: 640)
#       num_images 0 = full tri_val   (default: 0)
#
# Runs every released open baseline that the repo can actually build, plus our
# own FINAL-100 checkpoint, through
#     scripts/phase6_consistency_official.py --gt both
# so ONE forward pass yields both GT columns and the table is a pure GT swap.
#
# NOT included, and why (do not "fix" by adding them):
#   YOLOP    -- build_yolop hardcodes cfg.MODEL.IMAGE_SIZE=[640,640] and decodes
#               with its own anchor grid; feeding a 384-tall canvas would change
#               the grid and produce an incomparable number, not a better one.
#               At input=640 it is run by the Stage-A/`:640` path separately.
#   A-YOLOM / GDMNet / YOLOPv2 / v3 / YOLOPX / SCAM-P / RMT-PPAD
#               -- no official weights are staged in weights/, so they cannot be
#               measured at all. They stay literature-only rows (PHASE6_ALGORITHM_LEDGER.md).
set -u
cd ~/ai_study/trac || exit 1
PY=~/ai_study/gpu_env/bin/python
IN=${1:-640}
NIMG=${2:-0}
D=experiments/phase6/consistency
CK="experiments/phase6/final/B100/checkpoint.pt"
LOG=$D/stageB_in${IN}.log

if [ ! -f "$CK" ]; then echo "missing $CK"; exit 1; fi

MODELS="OursStatic:$PWD/$CK"
MODELS="$MODELS,TriLiteNet:tiny,TriLiteNet:small,TriLiteNet:base"
MODELS="$MODELS,TwinLiteNetPlus:nano,TwinLiteNetPlus:small,TwinLiteNetPlus:medium,TwinLiteNetPlus:large"
MODELS="$MODELS,TwinLiteNet"

mkdir -p "$D"
: > "$LOG"
echo "############ STAGE B  input=$IN  num_images=$NIMG  $(date '+%F %T') ############" | tee -a "$LOG"
echo "models: $MODELS" | tee -a "$LOG"
"$PY" scripts/phase6_consistency_official.py \
    --models "$MODELS" --input "$IN" --gt both --num-images "$NIMG" \
    --outdir "$D/stageB_in${IN}" 2>&1 \
  | tee -a "$LOG" | grep -vE "UserWarning|meshgrid|_VF"
# tee BEFORE grep on purpose. The old order (`python | grep | tee log`) makes grep
# block-buffer its stdout, so the log only appeared in 4 KB bursts and a live run
# looked stalled -- and if the run was killed, every progress line still sitting in
# grep's buffer was lost. tee-first keeps the log complete and raw (warnings
# included: they are evidence), while grep still filters what lands on the console.
echo "STAGE B DONE $(date '+%F %T')" | tee -a "$LOG"

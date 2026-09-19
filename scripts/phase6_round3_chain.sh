#!/bin/bash
# Phase 6 -- Round 3 STEP-2 chain: the minimal training the user approved (choice 4.B).
#
#   EXP-9A  YOLOP anchor ladder   A0 shipped / A1 aspect-flip / A2 k-means
#   EXP-9B  TLP lane arms         B0 baseline / B1 spatial    / B2 channel
#
# BUDGET FROZEN FROM MEASURED MICRO-PROBE RATES (pre-registration sections 5 and 11.3):
#   YOLOP  batch 8  -> 5532 ms/step, peak 8139/8151 MiB = 99.85% -> VRAM-CLIFF STALL.
#          SM clock pinned at MAX (2970 MHz) yet only ~30 W drawn while nvidia-smi
#          reports 100%: the GPU is idle-WAITING, not computing.  18x slower than
#          batch 6 for only 33% more work => a threshold effect, not a scaling law.
#          Windows/WDDM silently pages the overflow to host RAM instead of raising OOM.
#          REJECTED (section 11.3).
#          batch 16 -> 55073 ms/step, peak 16017 MiB = OVERSUSBCRIBED (thrash). REJECTED.
#          batch 6  -> 306 ms/step, peak 6085/8151 MiB = 74.7% ( ~2.0 GiB spare). FROZEN
#   TLP    batch 16 -> 328-399 ms/step, peak ~4340 MiB = 53% (below the cliff). FROZEN
#          (batch 24 slower per image: 23.9 vs 20.5 ms)
#   Chosen budget (RE-FROZEN 2026-09-11, pre-registration section 11.4(d)):
#     YOLOP  full 69863 imgs x 1 ep @ bs6 = 11644 steps x 0.234s ~=  45 min/arm
#     TLP    full 69863     x 3 ep @bs16  = 13098 steps x 0.36s  ~=  79 min/arm
#   Why the YOLOP budget changed twice: section 5 was frozen on a 4887 ms/step
#   reading that was itself a VRAM-cliff artifact.  At the true batch-6 rate
#   (234 ms/step) the same 60-90 min window buys 11.6x more steps AND the whole
#   training split, which is the direct remedy for the power shortfall recorded
#   in section 11.1(d).  LR likewise: 1e-3 was borrowed from R2's FROM-SCRATCH
#   recipe; for a PRETRAINED detector the standard is 1e-4 (section 11.4(d)).
#
# Halt politely any time:  touch experiments/phase6/STOP_CHAIN
# Idempotent: an arm already present in phase6_round3_trained.csv is skipped.
# A failed arm writes <outdir>/TRAIN_FAILED and is NOT appended, so it is
# visible in the log and will be retried on the next launch -- a silent NA row
# would be indistinguishable from an arm that legitimately produced nothing.
set -u
ROOT=~/ai_study/trac
cd "$ROOT" || exit 1
PY=~/ai_study/gpu_env/bin/python
FLAG=experiments/phase6/STOP_CHAIN
CSV=experiments/phase6/phase6_round3_trained.csv
LOG=experiments/phase6/round3_chain.log
rm -f "$FLAG"

# --- FAIL-CLOSED PRE-FLIGHT -------------------------------------------------
# The 2026-09-11 EXP-9A failure was an input-distribution drift between the
# trainer and the evaluator, which no single-file review could catch.  Refuse to
# start any arm unless the train/eval parity gate passes.
if ! "$PY" scripts/round3_input_parity_check.py; then
  echo "[chain] !! ABORT: train/eval input-parity gate FAILED - refusing to train" | tee -a "$LOG"
  exit 1
fi

# YOLOP: num-images 0 = the FULL tri_train split (11644 steps/epoch at bs6).
YOLOP_IMAGES=0; YOLOP_EP=1; YOLOP_BS=6; YOLOP_LR=1e-4
TLP_EP=3; TLP_BS=16
SEED=0

halt () { if [ -f "$FLAG" ]; then echo "[chain] STOP_CHAIN set - halt before $1" | tee -a "$LOG"; exit 0; fi; }

done_already () { # experiment arm
  [ -f "$CSV" ] && awk -F, -v e="$1" -v a="$2" 'NR>1 && $1==e && $2==a {f=1} END{exit !f}' "$CSV"
}

# ---------------------------------------------------------------------------
run_yolop () { # tag preset flops_G
  local TAG="$1" PRESET="$2" FLOPS="$3"
  halt "$TAG"
  if done_already EXP-9A "$TAG"; then echo "[chain] EXP-9A $TAG already in CSV - SKIP" | tee -a "$LOG"; return 0; fi
  local TR=experiments/phase6/round3_yolop_${TAG} EV=experiments/phase6/round3_yolop_${TAG}_eval
  mkdir -p "$TR" "$EV"
  echo "===== [$(date '+%F %T')] EXP-9A $TAG TRAIN (anchors=$PRESET ep=$YOLOP_EP bs=$YOLOP_BS img=$YOLOP_IMAGES(full=0) lr=$YOLOP_LR) =====" | tee -a "$LOG"
  if ! "$PY" scripts/phase6_round3_yolop_train.py --anchors-preset "$PRESET" --outdir "$TR" \
       --epochs $YOLOP_EP --batch-size $YOLOP_BS --num-images $YOLOP_IMAGES --seed $SEED \
       --lr "$YOLOP_LR" \
       --resume --ckpt-every 100 \
       > "$TR/train_stdout.log" 2>&1; then
    touch "$TR/TRAIN_FAILED"; echo "[chain] !! EXP-9A $TAG TRAIN FAILED (see $TR/train_stdout.log)" | tee -a "$LOG"; return 1
  fi
  echo "===== [$(date '+%F %T')] EXP-9A $TAG EVAL (weights=$TR/checkpoint.pt, anchors from $TR/anchors.json) =====" | tee -a "$LOG"
  # BOTH vars are mandatory: the 2026-09-11 run passed only the anchors, so all
  # three arms were the released model and their lane/DA metrics came out
  # identical.  YOLOP_WEIGHTS is the arm; YOLOP_ANCHORS_JSON is its own ruler.
  if ! YOLOP_WEIGHTS="$TR/checkpoint.pt" YOLOP_ANCHORS_JSON="$TR/anchors.json" \
       "$PY" evaluation/evaluate_baseline.py \
       --baseline YOLOP --outdir "$EV" --no-profile > "$EV/eval_stdout.log" 2>&1; then
    touch "$TR/TRAIN_FAILED"; echo "[chain] !! EXP-9A $TAG EVAL FAILED" | tee -a "$LOG"; return 1
  fi
  if ! grep -q "WEIGHTS OVERRIDE" "$EV/eval_stdout.log"; then
    touch "$TR/TRAIN_FAILED"; echo "[chain] !! EXP-9A $TAG eval did NOT load trained weights (silent-substitution guard)" | tee -a "$LOG"; return 1
  fi
  "$PY" scripts/phase6_round3_append.py --kind yolop --experiment EXP-9A --arm "$TAG" \
    --arch YOLOP --intervention "anchor set = $PRESET (FLOPs-neutral)" \
    --train "$TR/metrics.json" --eval "$EV/metrics.json" --train-log "$TR/training_log.txt" \
    --csv "$CSV" --flops-g "$FLOPS" 2>&1 | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
run_tlp () { # tag arm flops_G intervention
  local TAG="$1" ARM="$2" FLOPS="$3" INTER="$4"
  halt "$TAG"
  if done_already EXP-9B "$TAG"; then echo "[chain] EXP-9B $TAG already in CSV - SKIP" | tee -a "$LOG"; return 0; fi
  local TR=experiments/phase6/round3_tlp_${TAG} EV=experiments/phase6/round3_tlp_${TAG}_eval
  mkdir -p "$TR" "$EV"
  echo "===== [$(date '+%F %T')] EXP-9B $TAG TRAIN (arm=$ARM preset=small ep=$TLP_EP bs=$TLP_BS full tri_train) =====" | tee -a "$LOG"
  if ! "$PY" scripts/phase6_round3_tlp_train.py --arm "$ARM" --preset small --outdir "$TR" \
       --epochs $TLP_EP --batch-size $TLP_BS --seed $SEED \
       --resume --ckpt-every 200 > "$TR/train_stdout.log" 2>&1; then
    touch "$TR/TRAIN_FAILED"; echo "[chain] !! EXP-9B $TAG TRAIN FAILED (see $TR/train_stdout.log)" | tee -a "$LOG"; return 1
  fi
  echo "===== [$(date '+%F %T')] EXP-9B $TAG EVAL =====" | tee -a "$LOG"
  if ! "$PY" evaluation/evaluate_baseline.py --baseline Round3TLP --preset small \
       --round3-arm "$ARM" --round3-weights "$TR/checkpoint.pt" --outdir "$EV" --no-profile \
       > "$EV/eval_stdout.log" 2>&1; then
    touch "$TR/TRAIN_FAILED"; echo "[chain] !! EXP-9B $TAG EVAL FAILED" | tee -a "$LOG"; return 1
  fi
  "$PY" scripts/phase6_round3_append.py --kind tlp --experiment EXP-9B --arm "$TAG" \
    --arch TwinLiteNetPlus-small --intervention "$INTER" \
    --train "$TR/metrics.json" --eval "$EV/metrics.json" --train-log "$TR/training_log.txt" \
    --csv "$CSV" --flops-g "$FLOPS" 2>&1 | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
echo "[chain] ROUND3 STEP-2 start $(date '+%F %T')" | tee -a "$LOG"
echo "[chain] budget: YOLOP ${YOLOP_IMAGES}img x ${YOLOP_EP}ep @bs${YOLOP_BS} | TLP full x ${TLP_EP}ep @bs${TLP_BS}" | tee -a "$LOG"

# EXP-9A -- anchor-quality ladder (A0 is the degraded end, A2 the refit end)
run_yolop A0_shipped shipped 30.8208
run_yolop A1_flip    flip    30.8208
run_yolop A2_kmeans  kmeans  30.8208

# EXP-9B -- lane spatial vs channel under FLOPs parity
run_tlp B0_baseline baseline 4.213447 "stock lane head (no added path)"
run_tlp B1_spatial  spatial  4.223738 "learned 1/4 tap into lane head (zero-init, +204 params)"
run_tlp B2_channel  channel  4.224096 "1x1 bottleneck widening at 1/4 stage (zero-init, dc=13, +263 params)"

echo "[chain] ROUND3 STEP-2 DONE $(date '+%F %T')" | tee -a "$LOG"
[ -f "$CSV" ] && column -s, -t "$CSV" | tee -a "$LOG"

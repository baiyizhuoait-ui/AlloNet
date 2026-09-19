#!/bin/bash
# Phase 6 / G3 single-cell runner: train + eval + append, resume-safe.
#
# Same 20-column schema as experiments/phase6/final_results.csv on purpose, so a
# G3 row can be concatenated with the arch-1 tables and fed to the ledger
# analysis without a translation layer.
#
#   variant = g3
#   cell    = <arch>_<tier>       e.g. a2_encspend
#   encoder = <arch>|<tier>       e.g. ir|encspend
#   z       = 16 or 128
#
# Usage: bash scripts/phase6_g3_run.sh <tag> <config> <epochs> <seed> <cell> <enc_label> <z>
set -u
PY=~/ai_study/gpu_env/bin/python
ROOT=~/ai_study/trac
cd "$ROOT" || exit 1

TAG="$1"; CFG="$2"; EP="$3"; SEED="$4"; CELL="$5"; ENC="${6:-ir}"; Z="${7:-16}"
# Output location is overridable so a short 4-epoch pipeline probe cannot land in
# the same CSV as the 20-epoch cells: the ledger picks one row per (tier, seed),
# and a stray 4ep row would be indistinguishable from the real one.
OUT="${G3_OUT:-experiments/phase6/g3}"
CSV="${G3_CSV:-$OUT/g3_results.csv}"
FLAG=$OUT/STOP_CHAIN
TR="$OUT/${TAG}"
EVL="${TR}_eval"
CKPT="$(realpath -m "$TR")/checkpoint.pt"
MP="$EVL/metrics.json"
TLOG="$OUT/train_${TAG}.log"
ELOG="${EVL}.log"
mkdir -p "$OUT"

HDR="variant,cell,z,encoder,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_mIoU,lane_fg,peak_gpu_mem_mib,final_train_loss,train_wall_min,seed,source,git_commit"
[ -f "$CSV" ] || echo "$HDR" > "$CSV"

if [ -f "$FLAG" ]; then
  echo "[g3] STOP_CHAIN set - halt before ${TAG}"; exit 0
fi

# idempotent relaunch: skip a cell already appended
if awk -F, -v c="$CELL" -v s="$SEED" -v z="$Z" \
     'NR>1 && $2==c && $18==s && $3==z {f=1} END{exit !f}' "$CSV"; then
  echo "[g3] ${CELL} z${Z} seed${SEED} already in CSV - SKIP"; exit 0
fi

if [ -f "$MP" ]; then
  echo "[g3] ${TAG} metrics present - eval reuse" | tee -a "$OUT/g3_chain.log"
else
  mkdir -p "$TR" "$EVL"
  T0=$(date +%s)
  RESUME=""
  if [ -f "$CKPT" ]; then RESUME="--resume $CKPT"; fi
  echo "===== [$(date '+%F %T')] ${TAG} TRAIN start (ep=${EP} seed=${SEED} ${RESUME:-fresh}) =====" \
      | tee -a "$OUT/g3_chain.log"
  # shellcheck disable=SC2086
  "$PY" training/train.py --config "$CFG" --outdir "$TR" --epochs "$EP" --seed "$SEED" $RESUME \
    2>&1 | tee "$TLOG" > "$TR/train_stdout.log"
  T1=$(date +%s)
  WALL_MIN=$(( (T1 - T0) / 60 ))
  echo "===== [$(date '+%F %T')] ${TAG} TRAIN done in ${WALL_MIN} min =====" | tee -a "$OUT/g3_chain.log"

  # --- VRAM cliff fail-closed guard (per-process `self a/b MiB` only) --------
  # WDDM silently pages instead of raising OOM, so an over-capacity run yields a
  # plausible-looking but degraded number. The per-process field is used because
  # the device-wide one is inflated by any co-tenant process (that caused a
  # false abort on 2026-09-13, 132MiB short of the threshold, on a good run).
  CAP=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  CAP=${CAP:-8151}
  SELFN=$(grep -acE "self +[0-9]+/[0-9]+MiB" "$TLOG" 2>/dev/null || true)
  SELFN=${SELFN:-0}
  if [ "$SELFN" -ge 1 ]; then
    PEAK_USED=$(grep -aoE "self +[0-9]+/[0-9]+MiB" "$TLOG" | sed 's/self *//; s/MiB//' \
        | cut -d/ -f2 | sort -n | tail -1)
    PCT=$(( PEAK_USED * 100 / CAP ))
    echo "[g3] ${TAG} peak SELF ${PEAK_USED}MiB / ${CAP}MiB = ${PCT}% (samples=${SELFN})" \
        | tee -a "$OUT/g3_chain.log"
  else
    echo "[g3] ABORT: ${TAG} no per-process mem samples - VRAM guard UNVERIFIED (fail-closed)" \
        | tee -a "$OUT/g3_chain.log"
    echo "${TAG}_ABORTED_unverified_vram" >> "$OUT/ABORTS.txt"
    exit 2
  fi
  if [ "${PCT:-0}" -gt 96 ]; then
    echo "[g3] ABORT: ${TAG} exceeded the 96% VRAM cliff (${PCT}%)" | tee -a "$OUT/g3_chain.log"
    echo "${TAG}_ABORTED_${PCT}pct" >> "$OUT/ABORTS.txt"
    exit 2
  fi

  if [ -f "$CKPT" ]; then
    "$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" \
        --outdir "$EVL" 2>&1 | tee -a "$ELOG" > "$EVL/eval_stdout.log"
  fi
fi

# --- architecture-load assertion (NEW for G3, fail-closed) -------------------
# evaluate_baseline.py builds the model from the config.yaml sitting next to the
# checkpoint and then loads with strict=False, keeping only shape-matching keys.
# If the encoder class were wrong, MOST keys would silently be skipped and the
# metrics would just look bad rather than raising. So require missing=0.
if [ -f "$ELOG" ]; then
  LOADLINE=$(grep -a "\[ours\] loaded" "$ELOG" | tail -1)
  if [ -n "$LOADLINE" ]; then
    MISSING=$(printf '%s' "$LOADLINE" | grep -oE "missing=[0-9]+" | grep -oE "[0-9]+")
    MISSING=${MISSING:-NA}
    echo "[g3] eval load check: ${LOADLINE}" | tee -a "$OUT/g3_chain.log"
    if [ "$MISSING" != "0" ]; then
      echo "[g3] ABORT: ${TAG} checkpoint did not load cleanly (missing=${MISSING}) -" \
           "the config beside the checkpoint does not match the trained weights" \
           | tee -a "$OUT/g3_chain.log"
      echo "${TAG}_ABORTED_partial_load_missing${MISSING}" >> "$OUT/ABORTS.txt"
      exit 3
    fi
  else
    echo "[g3] ABORT: ${TAG} eval log has no 'loaded' line - cannot verify the" \
         "checkpoint matched the architecture (fail-closed)" | tee -a "$OUT/g3_chain.log"
    echo "${TAG}_ABORTED_no_load_line" >> "$OUT/ABORTS.txt"
    exit 3
  fi
fi

if [ ! -f "$MP" ]; then
  echo "${TAG},${CELL},${Z},${ENC},${EP},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,${SEED},failed-no-metrics,unknown" >> "$CSV"
  echo "[g3] FAIL ${TAG} no metrics.json" | tee -a "$OUT/g3_chain.log"
  exit 1
fi

WALL_MIN=${WALL_MIN:-NA}
PEAK_MEM=${PEAK_USED:-NA}
FINAL_LOSS=$(grep -aoE "loss[^0-9]*[0-9]+\.[0-9]+" "$TLOG" 2>/dev/null | tail -n1 \
             | grep -oE "[0-9]+\.[0-9]+" | tail -n1)
FINAL_LOSS=${FINAL_LOSS:-NA}

"$PY" - "$TAG" "$CELL" "$EP" "$SEED" "$ENC" "$Z" "$CSV" "$MP" "$WALL_MIN" "$PEAK_MEM" "$FINAL_LOSS" <<'PYEOF'
import csv, json, subprocess, sys

(tag, cell, ep, seed, enc, z, csvp, mp, wall, mem, floss) = sys.argv[1:12]
m = json.load(open(mp))
g = lambda k, d=None: m.get(k, d)


def n4(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if f != f else round(f, 4)


commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                        text=True).stdout.strip()[:40] or "unknown"
row = ["g3", cell, z, enc, ep,
       n4((g("parameters") or 0) / 1e6), n4((g("flops") or 0) / 1e9),
       n4(g("fps")), n4(g("mAP50")), n4(g("mAP50_95")),
       n4(g("da_mIoU")), n4(g("da_fg_iou")), n4(g("lane_mIoU")), n4(g("lane_fg_iou")),
       str(mem), str(floss), str(wall), seed, "trained", commit]

with open(csvp, "a", newline="") as fh:
    csv.writer(fh).writerow(row)


# readback self-verification (Round-3 D5 discipline)
with open(csvp, newline="") as fh:
    back = list(csv.DictReader(fh))
last = back[-1]
assert last["mAP50"] == str(row[8]), f"readback mAP50 {last['mAP50']} != {row[8]}"
assert last["lane_mIoU"] == str(row[12]), f"readback lane_mIoU {last['lane_mIoU']} != {row[12]}"
assert last["cell"] == cell and last["seed"] == str(seed), "readback cell/seed mismatch"
assert last["z"] == str(z), "readback z mismatch"
assert last["params_M"] != "NA", "readback params missing"
print("[readback OK]", ",".join(str(x) for x in row))
PYEOF
echo "[g3] ${TAG} DONE" | tee -a "$OUT/g3_chain.log"

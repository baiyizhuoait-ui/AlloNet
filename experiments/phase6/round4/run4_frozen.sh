#!/bin/bash
# Phase 6 Round 4 generic runner (train + eval + append, resume-safe).
#
# Differences vs scripts/phase6_run.sh (which stays FROZEN for prior phases):
#   1. --resume: an arm interrupted by a power cut continues from its last
#      completed epoch instead of restarting (train.py saves checkpoint.pt every
#      epoch). Worst-case loss = one epoch (~7 min), not one arm (~162 min).
#   2. VRAM-cliff fail-closed guard: if the max `mem a/b MiB` seen in the log
#      exceeds 96% of card capacity, the arm is recorded as INVALID and the
#      chain aborts. Round 3 established that WDDM silently pages instead of
#      raising OOM, turning 10x slowdown into a plausible-looking number.
#   3. Append through csv.writer + readback self-verification (Round-3 D5 fix).
#   4. A second, wide table carries every metric the Round 4 brief requires
#      (per-size AP50/AP5095/recall, latency p50/p95, peak memory).
#
# Usage: bash scripts/phase6_round4_run.sh <tag> <config> <epochs> <seed> <cell> <supervision>
set -u
PY=~/ai_study/gpu_env/bin/python
ROOT=~/ai_study/trac
cd "$ROOT" || exit 1

TAG="$1"; CFG="$2"; EP="$3"; SEED="$4"; CELL="$5"; SUP="${6:-km}"
OUT=experiments/phase6/round4
CSV=experiments/phase6/phase6_round4_results.csv
WIDE=experiments/phase6/phase6_round4_metrics.csv
FLAG=$OUT/STOP_CHAIN
TR="$OUT/r4_${TAG}"
EVL="${TR}_eval"
CKPT="$TR/checkpoint.pt"
MP="$EVL/metrics.json"
TLOG="$OUT/train_${TAG}.log"
mkdir -p "$OUT"

HDR="variant,cell,z,encoder,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_mIoU,lane_fg,peak_gpu_mem_mib,final_train_loss,train_wall_min,seed,source,git_commit"
[ -f "$CSV" ] || echo "$HDR" > "$CSV"

if [ -f "$FLAG" ]; then echo "[r4] STOP_CHAIN set - halt before ${TAG}"; exit 0; fi

# idempotent relaunch: skip a cell already appended
if awk -F, -v c="$CELL" -v s="$SEED" 'NR>1 && $2==c && $18==s {f=1} END{exit !f}' "$CSV"; then
  echo "[r4] ${CELL} seed${SEED} already in CSV - SKIP"; exit 0
fi

if [ -f "$MP" ]; then
  echo "[r4] ${TAG} metrics present - eval reuse" | tee -a "$OUT/round4_chain.log"
else
  mkdir -p "$TR" "$EVL"
  T0=$(date +%s)
  RESUME=""
  if [ -f "$CKPT" ]; then RESUME="--resume $CKPT"; fi
  echo "===== [$(date '+%F %T')] ${TAG} TRAIN start (ep=${EP} seed=${SEED} ${RESUME:-fresh}) =====" | tee -a "$OUT/round4_chain.log"
  # shellcheck disable=SC2086
  "$PY" training/train.py --config "$CFG" --outdir "$TR" --epochs "$EP" --seed "$SEED" $RESUME \
    2>&1 | tee "$TLOG" > "$TR/train_stdout.log"
  T1=$(date +%s)
  WALL_MIN=$(( (T1 - T0) / 60 ))
  echo "===== [$(date '+%F %T')] ${TAG} TRAIN done in ${WALL_MIN} min =====" | tee -a "$OUT/round4_chain.log"

  # --- VRAM cliff fail-closed guard ---
  # Round 3 showed WDDM silently PAGES instead of raising OOM, so an over-capacity run
  # yields a plausible-looking but 10x-degraded number -- hence fail-closed.
  #
  # D6 FIX (2026-09-12): the pattern `mem +[0-9]+/[0-9]+MiB` kept the literal token
  # `mem`. After `tr '/' ' '` awk saw $1="mem", so `$1+0 = 0 > m` was never true and
  # the peak stayed 0 -> "0MiB / 0MiB = 0%" on all 8 Round-4 arms. Structurally dead.
  #
  # D9 FIX (2026-09-13): the `mem` field is DEVICE-WIDE (`nvidia-smi memory.used`), so a
  # co-tenant process inflates it. On 2026-09-13 the L4 arm recorded 7692MiB/8151MiB =
  # 94% -- 132MiB short of a FALSE ABORT of a good 2.4h run -- because two diagnostic
  # probes from this session shared the card (probe mtimes 23:36:48 / 23:37:13; all 17
  # high samples fall in 23:36:57-23:38:32; the same log reads 2468MiB just before and
  # after). The cliff decision now uses the PER-PROCESS `self <alloc>/<reserved>MiB`
  # field only; device-wide is kept for continuity and reported as a fallback.
  CAP=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  CAP=${CAP:-8151}
  SELFN=$(grep -acE "self +[0-9]+/[0-9]+MiB" "$TLOG" 2>/dev/null || true)
  SELFN=${SELFN:-0}
  if [ "$SELFN" -ge 1 ]; then
    PEAK_USED=$(grep -aoE "self +[0-9]+/[0-9]+MiB" "$TLOG" | sed 's/self *//; s/MiB//' \
        | cut -d/ -f2 | sort -n | tail -1)
    PEAK_DEN=$CAP
    PCT=$(( PEAK_USED * 100 / PEAK_DEN ))
    echo "[r4] ${TAG} peak SELF ${PEAK_USED}MiB / ${PEAK_DEN}MiB = ${PCT}% (samples=${SELFN})" | tee -a "$OUT/round4_chain.log"
  else
    MEMN=$(grep -aoE "[0-9]+/[0-9]+ ?MiB" "$TLOG" 2>/dev/null | wc -l)
    read -r PEAK_USED PEAK_DEN PCT <<<"$(grep -aoE "[0-9]+/[0-9]+ ?MiB" "$TLOG" 2>/dev/null \
        | tr -d 'MiB' | tr '/' ' ' \
        | awk '{ if ($1+0 > m) { m=$1+0; d=$2+0 } } END { if (d>0) printf "%d %d %d", m, d, m*100/d; else print "0 0 0" }')"
    echo "[r4] ${TAG} peak DEVICE-WIDE ${PEAK_USED}MiB / ${PEAK_DEN}MiB = ${PCT}% (samples=${MEMN}; FALLBACK - no per-process field in this log)" | tee -a "$OUT/round4_chain.log"
    if [ "${MEMN:-0}" -lt 1 ]; then
      echo "[r4] ABORT: ${TAG} produced no parseable mem samples - VRAM guard UNVERIFIED (fail-closed)" | tee -a "$OUT/round4_chain.log"
      echo "${TAG}_ABORTED_unverified_vram" >> "$OUT/ABORTS.txt"
      exit 2
    fi
  fi
  if [ "${PCT:-0}" -gt 96 ]; then
    echo "[r4] ABORT: ${TAG} exceeded the 96% VRAM cliff threshold (${PCT}%) - result would be silently paged, not valid" | tee -a "$OUT/round4_chain.log"
    echo "${TAG}_ABORTED_${PCT}pct" >> "$OUT/ABORTS.txt"
    exit 2
  fi
  PEAK_USED=${PEAK_USED:-NA}

  if [ -f "$CKPT" ]; then
    "$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" --outdir "$EVL" \
      2>&1 | tee -a "${EVL}.log" > "$EVL/eval_stdout.log"
  fi
fi

if [ ! -f "$MP" ]; then
  echo "${TAG},${CELL},16,ebase,${EP},NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,${SEED},failed-no-metrics,unknown" >> "$CSV"
  echo "[FAIL] ${TAG} no metrics.json" | tee -a "$OUT/round4_chain.log"
  exit 1
fi

WALL_MIN=${WALL_MIN:-NA}
PEAK_MEM=${PEAK_USED:-NA}
FINAL_LOSS=$(grep -aoE "loss[^0-9]*[0-9]+\.[0-9]+" "$TLOG" 2>/dev/null | tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1)
FINAL_LOSS=${FINAL_LOSS:-NA}

"$PY" - "$TAG" "$CELL" "$EP" "$SEED" "$SUP" "$CFG" "$CSV" "$WIDE" "$MP" "$WALL_MIN" "$PEAK_MEM" "$FINAL_LOSS" <<'PYEOF'
import csv, json, os, subprocess, sys

(tag, cell, ep, seed, sup, cfg, csvp, widep, mp, wall, mem, floss) = sys.argv[1:13]
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
row = ["r2", cell, 16, "ebase", ep,
       n4((g("parameters") or 0) / 1e6), n4((g("flops") or 0) / 1e9),
       n4(g("fps")), n4(g("mAP50")), n4(g("mAP50_95")),
       n4(g("da_mIoU")), n4(g("da_fg_iou")), n4(g("lane_mIoU")), n4(g("lane_fg_iou")),
       str(mem), str(floss), str(wall), seed, "trained", commit]

with open(csvp, "a", newline="") as fh:
    csv.writer(fh).writerow(row)

# readback self-verification (Round-3 D5 discipline: never trust a joined string)
with open(csvp, newline="") as fh:
    back = list(csv.DictReader(fh))
last = back[-1]
assert last["mAP50"] == str(row[8]), f"readback mAP50 {last['mAP50']} != {row[8]}"
assert last["lane_mIoU"] == str(row[12]), f"readback lane_mIoU {last['lane_mIoU']} != {row[12]}"
assert last["cell"] == cell and last["seed"] == str(seed), "readback cell/seed mismatch"
print("[readback OK]", ",".join(str(x) for x in row))

# --- wide table: everything the Round 4 brief asks for -----------------------
WK = ["model", "cell", "rung", "supervision", "epochs", "seed", "params", "flops",
      "mAP50", "mAP50_95", "det_AP50_small", "det_AP50_medium", "det_AP50_large",
      "det_AP5095_small", "det_AP5095_medium", "det_AP5095_large",
      "det_recall50_small", "det_recall50_medium", "det_recall50_large",
      "det_n_pred", "det_n_gt", "da_mIoU", "da_fg_iou", "da_pixel_acc",
      "lane_mIoU", "lane_fg_iou", "lane_pixel_acc",
      "parameters", "model_size_mb", "flops_G",
      "p50_latency_ms", "p95_latency_ms", "avg_latency_ms", "eval_fwd_ms",
      "eval_fps", "gpu_memory_mib", "wall_min", "seed", "git_commit"]
wrow = dict(rung=tag, model="r2", cell=cell, supervision=sup, epochs=ep, seed=seed,
            flops_G=n4((g("flops") or 0) / 1e9), wall_min=wall, git_commit=commit)
for k in WK:
    wrow.setdefault(k, g(k))
new = not os.path.exists(widep)
with open(widep, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=WK, extrasaction="ignore")
    if new:
        w.writeheader()
    w.writerow(wrow)
print("[wide OK]", cell, "seed", seed)
PYEOF
echo "[r4] ${TAG} DONE" | tee -a "$OUT/round4_chain.log"

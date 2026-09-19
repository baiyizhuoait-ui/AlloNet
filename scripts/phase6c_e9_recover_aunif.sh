#!/bin/bash
# EXP-09 cell-1 (aunif_km) eval recovery -- one-off, self-terminating.
#
# Why: the 2026-09-10 naming-v3 commit b0f8d4c renamed
#   scripts/load_baseline_weights.py -> scripts/phase2_load_baseline_weights.py
# without resyncing the module-name imports. Module names are not paths, so
# refs_pass2.py and check_naming.sh CHK11 (which both match PATH references)
# could not see them: every eval after 16:52 died with ModuleNotFoundError.
# Cell 1's 19:49 eval therefore appended a `failed-no-metrics` row.
# Fixed in the working tree at 21:2x (4 import sites, see NAMING_CONVENTION
# section 12.6). This script waits for the running chain to exit, re-runs ONLY
# the eval (checkpoint reuse, NO re-training) and replaces that failed row.
set -u
CHAIN_PID="${1:-536}"
PY=~/ai_study/gpu_env/bin/python
cd ~/ai_study/trac || exit 1

OUT=experiments/phase6c
TR=$OUT/exp9_aunif_km
EVL=${TR}_eval
CKPT=$TR/checkpoint.pt
CSV=$OUT/phase6c_e9_factorial.csv
TLOG=$TR/train_stdout.log
LOG=$OUT/recover_aunif.log

exec >>"$LOG" 2>&1
# Idempotent: if a background watcher and a scheduled job both fire, the second
# must not burn another GPU eval and re-measure fps under different load.
if grep -q "recovery complete" "$LOG" 2>/dev/null; then
  echo "=== already recovered -- no-op ==="
  exit 0
fi
echo "=== recovery armed $(date '+%F %T'); waiting for chain pid $CHAIN_PID ==="
while kill -0 "$CHAIN_PID" 2>/dev/null; do sleep 30; done
echo "=== chain exited $(date '+%F %T'); eval-only pass starts ==="

[ -f "$CKPT" ] || { echo "ABORT: no checkpoint at $CKPT"; exit 2; }

"$PY" evaluation/evaluate_baseline.py --baseline OursStatic --preset "$CKPT" \
    --outdir "$EVL" > "$EVL/eval_stdout.log" 2>&1
[ -f "$EVL/metrics.json" ] || { echo "ABORT: eval produced no metrics.json"; tail -5 "$EVL/eval_stdout.log"; exit 3; }
echo "eval OK $(date '+%F %T')"

WALL_MIN=$(grep -oE "aunif_km TRAIN done in [0-9]+ min" "$OUT/phase6_runner.log" | head -1 | grep -oE "[0-9]+")
WALL_MIN=${WALL_MIN:-NA}
PEAK_MEM=$(grep -aoE "peak_mem=[0-9]+" "$TLOG" 2>/dev/null | tail -n1 | cut -d= -f2)
PEAK_MEM=${PEAK_MEM:-NA}
FINAL_LOSS=$(grep -aoE "loss[^0-9]*[0-9]+\.[0-9]+" "$TLOG" 2>/dev/null | tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1)
FINAL_LOSS=${FINAL_LOSS:-NA}
echo "train-side fields recovered: wall=${WALL_MIN}min peak_mem=${PEAK_MEM} final_loss=${FINAL_LOSS}"

"$PY" - "$CSV" "$EVL/metrics.json" "$WALL_MIN" "$PEAK_MEM" "$FINAL_LOSS" <<'PYEOF'
import json, sys
csv, mp, wall, mem, floss = sys.argv[1:6]
m = json.load(open(mp))
g = lambda k, d=0.0: m.get(k, d)
def num(v):
    try: f = float(v)
    except (TypeError, ValueError): return "NA"
    return "NA" if f != f else "%.4f" % f
lines = open(csv).read().splitlines()
head, body = lines[0], lines[1:]
src_commit = "unknown"
kept = []
for l in body:
    f = l.split(",")
    if len(f) > 18 and f[1] == "aunif_km" and f[17] == "0":
        src_commit = f[19] if len(f) > 19 else "unknown"   # keep the SHA cell 1 was TRAINED under
        continue
    kept.append(l)
row = ["r2", "aunif_km", "16", "ebase", "20",
       num(g("parameters") / 1e6 if g("parameters") else "NA"),
       num(g("flops") / 1e9 if g("flops") else "NA"),
       "%.2f" % g("fps"), num(g("mAP50")), num(g("mAP50_95")),
       num(g("da_mIoU")), num(g("da_fg_iou")),
       num(g("lane_mIoU")), num(g("lane_fg_iou")),
       str(mem), str(floss), str(wall), "0", "trained", src_commit]
kept.append(",".join(row))
open(csv, "w").write("\n".join([head] + kept) + "\n")
print("REPLACED row cell=aunif_km seed=0 (git_commit preserved: %s)" % src_commit)
print("NEW ROW: " + ",".join(row))
PYEOF
echo "=== recovery complete $(date '+%F %T') ==="

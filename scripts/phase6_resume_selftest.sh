#!/usr/bin/env bash
# Phase 6 -- FINAL-100 follow-up: does an interrupted 100-ep arm really resume?
#
# ============================================================================
# !! THIS SCRIPT'S VERDICT WAS MISLEADING -- READ scripts/phase6_resume_selftest2.sh
# !!
# !! It reported "BIT-FOR-BIT EQUIVALENT", but it passed a RELATIVE --resume, and
# !! training/train.py:426 joins a relative --resume onto the (relative) outdir,
# !! so the path arrived doubled and EVERY resume silently fell back to
# !! "training fresh from epoch 1". The two runs it compared were therefore both
# !! FRESH runs: the test measured seed determinism, not resume.
# !!
# !! The real proof is round 2 (absolute path): two genuine "RESUME from ...
# !! start_epoch=2/3" lines, and 257/257 tensors bitwise identical.
# !! Keep this file as the record of HOW the bug was found -- its B.log contains
# !! the doubled-path line that exposed it. Do not cite its verdict.
# ============================================================================
#
# WHY THIS EXISTS
#   The FINAL-100 arm takes ~12 h. The PI needs to be able to stop seed 1 (say,
#   to use the machine during the day) and continue later. train.py *claims*
#   (training/train.py:418-455) that a resumed run is "BIT-FOR-BIT equivalent to
#   a run that was never interrupted", and the runner auto-adds --resume when a
#   checkpoint exists (scripts/phase6_final100_run.sh). But a grep of the whole
#   phase-6 tree finds exactly ONE historical "RESUME from" line, and it belongs
#   to a *different* script (round3_tlp_B0_baseline, whose checkpoint format is
#   `epoch=/step=`, not the current epoch-level format). So the current --resume
#   path is DESIGNED but NEVER EXERCISED. This script exercises it.
#
# WHAT IT DOES  (CPU ONLY -- never touches the GPU, so it cannot disturb the
#                live seed-1 run; the GPU job keeps its full card and its
#                per-process `self` VRAM accounting stays clean)
#   RUN A  uninterrupted          3 epochs, seed 7, 192 images, CPU
#   RUN B  interrupted twice      3 epochs, same seed/images, CPU
#            ep1 checkpoint written -> SIGKILL mid-ep2
#            resume                    -> ep2 checkpoint written -> SIGKILL mid-ep3
#            resume                    -> finishes ep3
#   COMPARE every tensor of A/B final weights bitwise, plus the "ep N DONE
#   avg_loss=" lines. If the claim holds, both must be IDENTICAL.
#
# A kill is always issued a few seconds AFTER an "ep N DONE" line appears, i.e.
# guaranteed mid-next-epoch -- the same timing a real power cut has, so the test
# measures the worst case the runner has to survive (loss <= 1 epoch).
set -u
cd ~/ai_study/trac || exit 1
PY=~/ai_study/gpu_env/bin/python
CFG=configs/phase6_r4_R2_thin14_z16.yaml
ST=experiments/phase6/resume_selftest
EP=3
SEED=7
NIMG=192
LOG=$ST/selftest.log

# be a good citizen while seed 1 trains: lowest priority, 2 threads only
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

rm -rf "$ST"
mkdir -p "$ST/A" "$ST/B"
: > "$LOG"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

done_epochs() {  # how many epochs COMPLETED in a run dir (from the append-only log)
  grep -c "DONE avg_loss" "$1/training_log.txt" 2>/dev/null || echo 0
}

# ---- RUN A : uninterrupted ------------------------------------------------
say "RUN A (uninterrupted, ${EP} ep, seed ${SEED}, ${NIMG} imgs, CPU)"
nice -n 19 "$PY" training/train.py --config "$CFG" --outdir "$ST/A" \
  --epochs "$EP" --seed "$SEED" --device cpu --allow-cpu --num-images "$NIMG" \
  --num-workers 0 --log-every 4 > "$ST/A.log" 2>&1
say "RUN A rc=$?"

# ---- RUN B : interrupted twice, resumed twice -----------------------------
say "RUN B (interrupted twice, resumed twice)"
B_ARGS=(training/train.py --config "$CFG" --outdir "$ST/B" --epochs "$EP"
        --seed "$SEED" --device cpu --allow-cpu --num-images "$NIMG"
        --num-workers 0 --log-every 4)

nice -n 19 "$PY" "${B_ARGS[@]}" > "$ST/B.log" 2>&1 &
BPID=$!
say "RUN B attempt 1 pid=$BPID"

wait_epoch_then_kill() {   # $1 = epoch that must already be DONE
  local want=$1 deadline=$((SECONDS + 1800))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if [ "$(done_epochs "$ST/B")" -ge "$want" ]; then
      sleep 4                      # now provably INSIDE epoch want+1
      kill -9 "$BPID" 2>/dev/null
      wait "$BPID" 2>/dev/null
      say "SIGKILL mid-epoch-$((want + 1)) (after ep${want} checkpoint)"
      return 0
    fi
    sleep 0.5
  done
  say "TIMEOUT waiting for ep${want} to complete"
  return 1
}

wait_epoch_then_kill 1
nice -n 19 "$PY" "${B_ARGS[@]}" --resume "$ST/B/checkpoint.pt" >> "$ST/B.log" 2>&1 &
BPID=$!
say "RUN B attempt 2 (resume) pid=$BPID"
wait_epoch_then_kill 2
nice -n 19 "$PY" "${B_ARGS[@]}" --resume "$ST/B/checkpoint.pt" >> "$ST/B.log" 2>&1 &
BPID=$!
say "RUN B attempt 3 (resume) pid=$BPID"
wait "$BPID" 2>/dev/null
say "RUN B finished rc=$?"

# ---- evidence: what each attempt actually did ------------------------------
say "--- RESUME lines in B.log ---"
grep -n "RESUME from\|training fresh" "$ST/B.log" | tee -a "$LOG"
say "--- A epochs ---"; grep -h "DONE avg_loss" "$ST/A/training_log.txt" | tee -a "$LOG"
say "--- B epochs ---"; grep -h "DONE avg_loss" "$ST/B/training_log.txt" | tee -a "$LOG"

# ---- decision: bit-for-bit? ----------------------------------------------
"$PY" - "$ST" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, torch
st = sys.argv[1]
a = torch.load(f"{st}/A/checkpoint.pt", map_location="cpu", weights_only=False)
b = torch.load(f"{st}/B/checkpoint.pt", map_location="cpu", weights_only=False)
print(f"[cmp] A epoch={a['epoch']}  B epoch={b['epoch']}"
      f"  cfg_sha A={a.get('cfg_sha','?')[:8]} B={b.get('cfg_sha','?')[:8]}")
print(f"[cmp] B rng keys present: {sorted((b.get('rng') or {}).keys())}")
print(f"[cmp] B opt/sched state present: {'opt_state' in b} / {'sched_state' in b}")
ka, kb = a["model_state"], b["model_state"]
assert set(ka) == set(kb), "tensor key sets differ"
bad, maxd = [], 0.0
for k in ka:
    if not torch.equal(ka[k], kb[k]):
        bad.append(k)
        maxd = max(maxd, (ka[k].float() - kb[k].float()).abs().max().item())
print(f"[cmp] tensors={len(ka)}  bitwise-identical={len(ka)-len(bad)}  differing={len(bad)}"
      f"  max|diff|={maxd:.3e}")
print("[cmp] VERDICT:", "BIT-FOR-BIT EQUIVALENT (the resume claim HOLDS)"
      if not bad else "NOT IDENTICAL -> " + ",".join(bad[:10]))
PYEOF

say "SELFTEST DONE"

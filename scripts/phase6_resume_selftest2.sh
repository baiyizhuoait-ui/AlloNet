#!/usr/bin/env bash
# Phase 6 -- resume self-test, ROUND 2: the ABSOLUTE-path variant.
#
# WHY ROUND 2 EXISTS (read this before trusting any "resume works" claim)
#   Round 1 (scripts/phase6_resume_selftest.sh, 2026-09-14 10:17-10:22) reported
#   "BIT-FOR-BIT EQUIVALENT", but its own log disproves the interpretation:
#
#       [train] --resume experiments/phase6/resume_selftest/B/experiments/
#               phase6/resume_selftest/B/checkpoint.pt not found; training fresh
#               from epoch 1
#
#   The path was doubled and the run restarted from scratch, three times
#   (B's log holds three "ep 1 DONE" lines). The comparison passed because a
#   fresh 3-epoch run at a fixed seed is deterministic -- i.e. it compared two
#   FRESH runs, not an interrupted one against an uninterrupted one. The verdict
#   was right about determinism and silent about resume.
#
#   Root cause (training/train.py:426):
#       rp = args.resume if os.path.isabs(args.resume) else os.path.join(outdir, args.resume)
#   With a RELATIVE --resume and a RELATIVE outdir the two paths are concatenated.
#   scripts/phase6_final100_run.sh passes exactly that pair (CKPT and TR are both
#   relative), so the production chain has the same defect: an interrupted arm
#   silently retrains from epoch 1 and the "loss <= 1 epoch" promise in its own
#   header comment is false. Cost of an interruption is the WHOLE arm (~12 h),
#   not one epoch. (Not a correctness bug: the retrained arm is still a valid
#   full-length run. It is a wall-clock bug, and a silent one.)
#
#   This script therefore passes an ABSOLUTE path, which is the one form the join
#   cannot corrupt, and re-runs only the interrupted arm. RUN A (the uninterrupted
#   reference) is reused from round 1 if it is still present.
set -u
cd ~/ai_study/trac || exit 1
PY=~/ai_study/gpu_env/bin/python
CFG=configs/phase6_r4_R2_thin14_z16.yaml
ST=experiments/phase6/resume_selftest
EP=3
SEED=7
NIMG=192
LOG=$ST/selftest_round2.log

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p "$ST"
: > "$LOG"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -f "$ST/A/checkpoint.pt" ] || { say "RUN A reference missing -- run scripts/phase6_resume_selftest.sh first"; exit 1; }

# start RUN B clean, so nothing from round 1 is reused
rm -rf "$ST/B"; mkdir -p "$ST/B"

B_ARGS=(training/train.py --config "$CFG" --outdir "$ST/B" --epochs "$EP"
        --seed "$SEED" --device cpu --allow-cpu --num-images "$NIMG"
        --num-workers 0 --log-every 4)

done_epochs() { grep -c "DONE avg_loss" "$ST/B/training_log.txt" 2>/dev/null || echo 0; }

wait_epoch_then_kill() {
  local want=$1 deadline=$((SECONDS + 1800))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if [ "$(done_epochs)" -ge "$want" ]; then
      sleep 4
      kill -9 "$BPID" 2>/dev/null
      wait "$BPID" 2>/dev/null
      say "SIGKILL mid-epoch-$((want + 1)) (after ep${want} checkpoint)"
      return 0
    fi
    sleep 0.5
  done
  say "TIMEOUT waiting for ep${want}"; return 1
}

say "RUN B round 2 (absolute --resume, interrupted twice)"
nice -n 19 "$PY" "${B_ARGS[@]}" > "$ST/B.log" 2>&1 &
BPID=$!
say "attempt 1 pid=$BPID (fresh)"
wait_epoch_then_kill 1

# THE FIX UNDER TEST: absolute path. /home/.../trac/<ST>/B/checkpoint.pt
ABS1="$PWD/$ST/B/checkpoint.pt"
say "attempt 2 pid=?  --resume $ABS1   (absolute)"
nice -n 19 "$PY" "${B_ARGS[@]}" --resume "$ABS1" >> "$ST/B.log" 2>&1 &
BPID=$!
wait_epoch_then_kill 2

say "attempt 3 pid=?  --resume $ABS1   (absolute)"
nice -n 19 "$PY" "${B_ARGS[@]}" --resume "$ABS1" >> "$ST/B.log" 2>&1 &
BPID=$!
wait "$BPID" 2>/dev/null
say "RUN B finished rc=$?"

# --- CRITICAL: did it actually resume this time? ----------------------------
say "--- resume evidence (this is the line round 1 did not have) ---"
grep -n "RESUME from\|training fresh" "$ST/B.log" | tee -a "$LOG"
RESUMED=$(grep -c "RESUME from" "$ST/B.log" || true)
FRESH=$(grep -c "training fresh" "$ST/B.log" || true)
say "attempts that actually resumed: $RESUMED ; that fell back to fresh: $FRESH"

say "--- B epochs (must be exactly 3: ep1 once, then ep2, ep3) ---"
grep -h "DONE avg_loss" "$ST/B/training_log.txt" | tee -a "$LOG"

"$PY" - "$ST" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, torch
st = sys.argv[1]
a = torch.load(f"{st}/A/checkpoint.pt", map_location="cpu", weights_only=False)
b = torch.load(f"{st}/B/checkpoint.pt", map_location="cpu", weights_only=False)
print(f"[cmp] A epoch={a['epoch']}  B epoch={b['epoch']}")
ka, kb = a["model_state"], b["model_state"]
assert set(ka) == set(kb)
bad, maxd = [], 0.0
for k in ka:
    if not torch.equal(ka[k], kb[k]):
        bad.append(k)
        maxd = max(maxd, (ka[k].float() - kb[k].float()).abs().max().item())
print(f"[cmp] tensors={len(ka)} identical={len(ka)-len(bad)} differing={len(bad)} max|diff|={maxd:.3e}")
print("[cmp] VERDICT:", "INTERRUPTED+RESUMED == UNINTERRUPTED, bit for bit"
      if not bad else "NOT IDENTICAL -> " + ",".join(bad[:10]))
PYEOF
say "ROUND 2 DONE"

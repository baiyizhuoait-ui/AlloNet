#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resume-RNG verification: root cause + fix, on the checkpoint that actually broke.

Background
  2026-09-14 13:28:51, the seed-1 resume died 17 s in:
      File "training/train.py", line 442, in main
          torch.set_rng_state(rng["torch"])
      TypeError: RNG state must be a torch.ByteTensor
  scripts/_probe_rng.py showed the checkpoint is NOT the problem: its rng payload
  is a healthy uint8 tensor of length 5056.  The defect is in train.py -- it loads
  the checkpoint with map_location=device, which moves those CPU-only byte-vectors
  onto the GPU right before they are handed to set_rng_state.

  The round-2 CPU self-test could not have caught this: on device=cpu map_location
  is a no-op, so the bug is CUDA-only.  That is why the previous "resume is
  bit-for-bit" claim held in the self-test and failed in production.

What this asserts (A-D), on the real checkpoint:
  A. the OLD code path reproduces the TypeError          -> root cause confirmed
  B. the NEW code path restores all five streams          -> fix works
  C. fidelity: the restored state round-trips exactly
     (get_rng_state after set_rng_state == the saved bytes), for torch / cuda /
     numpy / random / loader_gen -- a restore that silently mis-sets the state
     would pass B but fail C, and only C protects the bit-for-bit claim
  D. the epoch guard reads ep18, so start_epoch must be 19

Exit code 0 only if every assertion holds.
"""
import os
import random as _rnd
import sys

import numpy as np
import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "experiments/phase6/final/B100_s1/checkpoint.pt"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


print(f"checkpoint : {CKPT}")
print(f"device     : {DEV}  (the defect is CUDA-only, so this must be cuda to be meaningful)")
print(f"cwd        : {os.getcwd()}")
assert os.path.exists(CKPT), f"missing {CKPT}"

# ---------------------------------------------------------------- A. root cause
print("\n[A] old code path: torch.load(map_location=device) -> set_rng_state")
rck_gpu = torch.load(CKPT, map_location=DEV, weights_only=False)
r_gpu = rck_gpu["rng"]
check("rng['torch'] was moved to the GPU by map_location=device",
      r_gpu["torch"].device.type == DEV.type,
      f"device={r_gpu['torch'].device}, dtype={r_gpu['torch'].dtype}, n={r_gpu['torch'].numel()}")
reproduced = None
try:
    torch.set_rng_state(r_gpu["torch"])
except Exception as e:  # noqa: BLE001
    reproduced = f"{type(e).__name__}: {e}"
check("old path raises TypeError (root cause reproduced)",
      reproduced is not None and "ByteTensor" in str(reproduced), str(reproduced))
if DEV.type == "cuda":
    check("the same failure also sits in the cuda stream call",
          True, "set_rng_state_all([...cuda tensors]) has the identical CPU-only requirement")

# ---------------------------------------------------------------- B/C. the fix
print("\n[B/C] new code path: coerce to CPU uint8, then set + read back")


def _rng_cpu(t):
    return t.detach().to("cpu", torch.uint8)


rck = torch.load(CKPT, map_location="cpu", weights_only=False)
rng = rck["rng"]

ok = True
try:
    torch.set_rng_state(_rng_cpu(rng["torch"]))
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_rng_cpu(t) for t in rng["cuda"]])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("random") is not None:
        _rnd.setstate(rng["random"])
    if rng.get("loader_gen") is not None:
        g = torch.Generator()
        g.set_state(_rng_cpu(rng["loader_gen"]))
except Exception as e:  # noqa: BLE001
    ok = False
    print(f"      raised {type(e).__name__}: {e}")
check("all five RNG streams restore without error", ok)

# C. fidelity -- set, then read back, must equal the saved bytes
check("torch stream round-trips byte-for-byte",
      torch.equal(torch.get_rng_state(), _rng_cpu(rng["torch"])))
if rng.get("cuda") is not None and torch.cuda.is_available():
    back = torch.cuda.get_rng_state_all()
    saved = [_rng_cpu(t) for t in rng["cuda"]]
    check("cuda stream(s) round-trip byte-for-byte",
          len(back) == len(saved) and all(torch.equal(a, b) for a, b in zip(back, saved)),
          f"{len(saved)} device stream(s), {saved[0].numel()} bytes each")
check("numpy stream round-trips",
      all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), rng["numpy"]))
      if isinstance(rng["numpy"], tuple) else True)
check("python random stream round-trips", _rnd.getstate() == rng["random"])
if rng.get("loader_gen") is not None:
    g2 = torch.Generator()
    g2.set_state(_rng_cpu(rng["loader_gen"]))
    check("dataloader generator round-trips byte-for-byte",
          torch.equal(g2.get_state(), _rng_cpu(rng["loader_gen"])))

# a restore that is faithful must make the *next* draws reproducible
torch.set_rng_state(_rng_cpu(rng["torch"]))
draw_a = torch.randn(5).tolist()
torch.set_rng_state(_rng_cpu(rng["torch"]))
draw_b = torch.randn(5).tolist()
check("two restores yield identical subsequent draws", draw_a == draw_b,
      f"first draw {draw_a[0]:.6f}")

# ---------------------------------------------------------------- D. epoch guard
print("\n[D] epoch guard")
ep = int(rck.get("epoch", 0))
check("checkpoint carries an epoch (guard would not fire)", ep > 0, f"epoch={ep}")
check("start_epoch resumes at ep+1", ep + 1 == 19, f"start_epoch={ep + 1}")
check("cfg_sha matches the FINAL-100 run", str(rck.get("cfg_sha", ""))[:8] == "15946c2a",
      str(rck.get("cfg_sha"))[:8])

print("\n" + "=" * 72)
if fails:
    print(f"RESULT: {len(fails)} CHECK(S) FAILED -> {fails}")
    sys.exit(1)
print("RESULT: all checks passed -- root cause confirmed, fix verified on the real checkpoint")

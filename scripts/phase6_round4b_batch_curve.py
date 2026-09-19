#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6 / Round 4B closure -- BATCH SCALING CURVE on the local card (RTX 5060 8GB).

WHY THIS EXISTS
---------------
The GPU rental selection memo (`PHASE6_GPU_RENTAL_SELECTION.md` section 7) left one input
open: **24 GB or 48 GB?** The answer depends on how ms/img and peak memory move with batch
size, measured on the actual Model B. It was deferred to "after the 4B chain ends, when the
local GPU is free". The chain ended 2026-09-13 09:40:39 and the card is idle (0 % / 0 MiB).

WHAT IS MEASURED
----------------
For ascending batch sizes, on **synthetic tensors** (no DataLoader, so the number is pure
GPU-side fwd+bwd+opt -- the data pipeline is provably hidden by prefetch, rent memo sec. 1):
  * ms/step, ms/img
  * peak reserved / allocated memory

Synthetic input is deliberate: it isolates GPU-side cost from the CPU pipeline, which is
what the batch decision is about.

VRAM CLIFF GUARD (added after a real incident, see below)
---------------------------------------------------------
First run of this probe (17:45) hung: at batch=128 the card sat at 7874/8151 MiB = 96.6 %
with util=100 % for >4 min and emitted **zero** steps. That is the WDDM silent-paging cliff
this project already guards against in training -- and `torch.cuda.OutOfMemoryError` does NOT
fire, because Windows lets the allocation succeed and then pages. A guard on the *previous*
batch's peak cannot prevent the first overcommit, so the fix is a hard process cap:

    torch.cuda.set_per_process_memory_fraction(0.85)

With that, an over-large batch raises a clean OOM (recorded as a data point) instead of
paging for minutes. Run with `python -u` so progress streams through `tee`.

DISCIPLINE
----------
No training chain is running (verified 0 % / 0 MiB before launch). Per the D9 lesson,
diagnostic probes must never co-tenant a card with a live training arm. Writes nothing to
the repo except its own stdout / log.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402
from models.static_model import StaticMultiTaskModel  # noqa: E402

BATCHES = [8, 16, 32, 64, 96, 128]
WARMUP = 3
TIMED = 10
IMG = 640
VRAM_FRACTION = 0.85
STOP_ABOVE = 0.85


def tensors_of(obj, out):
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            tensors_of(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            tensors_of(v, out)
    return out


def build_model_b():
    km = yaml.safe_load(open(os.path.join(
        ROOT, "configs", "phase6_combo_danc_l14f1.yaml")))["model"]["detection"]["anchors"]
    cfg = {
        "encoder": {"stem": 16, "stages": [32, 64, 96, 128], "blocks": [2, 2, 2]},
        "representation": {"z_channels": 16},
        "detection": {"nc": 1, "from_z": True, "z_proj": True, "det_ch": 32, "anchors": km},
        "segmentation": {"hidden": 32, "lane_res": 4, "lane_use_f1": True, "lane_hidden": 16},
    }
    return StaticMultiTaskModel(cfg)


def main():
    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    total_mib = props.total_memory / 2**20
    cap = torch.cuda.get_device_capability(0)
    print("device: %s   sm_%d%d   VRAM %.0f MiB"
          % (torch.cuda.get_device_name(0), cap[0], cap[1], total_mib), flush=True)
    print("process VRAM hard cap: %.0f%% = %.0f MiB (OOM instead of paging)"
          % (VRAM_FRACTION * 100, VRAM_FRACTION * total_mib), flush=True)
    print("input: synthetic 3x%dx%d, batches %s   (no DataLoader -> pure GPU-side step)"
          % (IMG, IMG, BATCHES), flush=True)
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)

    model = build_model_b().to(dev).train()
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    print("params: %d" % n_par, flush=True)
    print("=" * 74, flush=True)
    print("%6s %10s %9s %13s %13s %6s"
          % ("batch", "ms/step", "ms/img", "peak_resv", "peak_alloc", "note"), flush=True)
    print("=" * 74, flush=True)

    rows = []
    stop = False
    for bs in BATCHES:
        if stop:
            print("%6d %10s %9s %13s %13s %6s" % (bs, "-", "-", "-", "-", "skip"), flush=True)
            rows.append((bs, None, None, None, None, "skip"))
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.randn(bs, 3, IMG, IMG, device=dev)
            for _ in range(WARMUP):
                opt.zero_grad(set_to_none=True)
                loss = sum(t.mean() for t in tensors_of(model(x), []))
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(TIMED):
                opt.zero_grad(set_to_none=True)
                loss = sum(t.mean() for t in tensors_of(model(x), []))
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0 / TIMED
            peak_r = torch.cuda.max_memory_reserved() / 2**20
            peak_a = torch.cuda.max_memory_allocated() / 2**20
            note = ""
            if peak_r > STOP_ABOVE * total_mib:
                note = "CLIFF"
                stop = True
            rows.append((bs, ms, ms / bs, peak_r, peak_a, note))
            print("%6d %10.2f %9.3f %10.0f MiB %10.0f MiB %6s"
                  % (bs, ms, ms / bs, peak_r, peak_a, note), flush=True)
            del x, loss
        except torch.cuda.OutOfMemoryError:
            print("%6d %10s %9s %13s %13s %6s" % (bs, "-", "-", "-", "-", "OOM"), flush=True)
            rows.append((bs, None, None, None, None, "OOM"))
            torch.cuda.empty_cache()
            stop = True

    print("=" * 74, flush=True)
    ok = [r for r in rows if r[1] is not None]
    if ok:
        ref = [r for r in ok if r[0] == 16]
        if ref:
            r0 = ref[0][2]
            print("\n--- relative to batch=16 (clean, no contention) ---", flush=True)
            for bs, ms, mpi, pr, pa, nt in ok:
                print("  b=%-4d ms/img %7.3f   x%.2f   peak_resv %5.0f MiB  %s"
                      % (bs, mpi, mpi / r0, pr, nt), flush=True)
        best = min(ok, key=lambda r: r[2])
        print("\nbest ms/img: batch=%d -> %.3f ms/img (peak_resv %.0f MiB)"
              % (best[0], best[2], best[3]), flush=True)
        top = max(ok, key=lambda r: r[3])
        print("largest non-OOM batch: %d @ peak_resv %.0f MiB  (%.0f%% of %.0f MiB)"
              % (top[0], top[3], 100 * top[3] / total_mib, total_mib), flush=True)
    print("\n[probe done]", flush=True)


if __name__ == "__main__":
    main()

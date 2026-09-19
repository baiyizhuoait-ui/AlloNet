#!/usr/bin/env python3
"""Phase 6 / D8 -- re-measure the latency column.

D8 in the defect log: "same model, p50 = 2.663 ms vs 7.670 ms (2.9x dispersion),
root cause not located" -> the latency / FPS column was declared untrustworthy
and excluded from every judgement.  This probe locates the cause and produces a
replacement column.

HYPOTHESIS (formed by reading the code, BEFORE measuring):

  evaluation/evaluate_baseline.py profiles OUR models with reps=3
  (line 327: `_profile_ours(model, (1,3,640,640), device, 3)`) while the baseline
  path uses the default reps=100 (line 329: `profile_model(model, device=...)`).
  profiling/benchmark.latency_ms then reports
  `p50 = sorted(times)[len(times)//2]`, i.e. the SECOND OF THREE SAMPLES, after
  only 10 warmup iterations.  This laptop GPU idles at 322 MHz against a
  3090 MHz maximum, so 10 iterations of a 1.17 GFLOP model do not reach the
  steady clock.  A 3-sample median of a ramping-clock process has enormous
  variance -- which is what a 2.9x spread looks like.

Two consequences, both testable:

  (1) The dispersion is an ESTIMATOR problem (n=3), not hardware instability.
  (2) The instrument is ASYMMETRIC: our rows and the baseline rows were never
      measured the same way.  That is worse than noise -- it biases the table.

THE DECISIVE TEST is the null replicate: five checkpoints of the SAME
architecture (B20 s0/s1/s2, B40 s0, B100 s0) -- 192,566 params / 1.1656 GFLOPs in
all five.  Under a good instrument they must agree to within repeatability.  Under
the as-shipped instrument they should scatter by ~3x.

Protocols, compared per (model, batch):
  AS_SHIPPED_OURS   warmup=10,  reps=3,   perf_counter + sync   x10  (what ours got)
  AS_SHIPPED_BASE   warmup=10,  reps=100, perf_counter + sync   x3   (what baselines got)
  FIXED             warmup=200, reps=300, CUDA events           x3   (proposed)

FIXED additionally records an SM-clock ramp trace, so "the clock ramps slowly" is
measured rather than asserted.

FAIL-CLOSED: exits non-zero if any other process holds the GPU.  Round 4's D9 and
D11 were both same-card contention incidents, and the two worst latency readings
in that round (both 7.670 ms) belong to the two arms affected by the D11
double-chain collision.  So this probe refuses to run on a shared card, and it
must NOT be run while a training chain is active.

ZERO TRAINING.  Read-only w.r.t. every metric table.

Usage:
    gpu_env/bin/python scripts/phase6_d8_latency_probe.py
    gpu_env/bin/python scripts/phase6_d8_latency_probe.py --batch 1 --repeats-fixed 2
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import evaluate_baseline as eb  # noqa: E402  (same builders as the real eval)

OUTDIR = os.path.join(ROOT, "experiments", "phase6", "final", "d8_latency")

# The rows of the paper's main table.  Five of these are the SAME architecture.
MODELS = [
    ("Ours B100 (100ep s0)", "OursStatic", "experiments/phase6/final/B100/checkpoint.pt"),
    ("Ours B40 (40ep s0)", "OursStatic", "experiments/phase6/round4/r4_R4R2thin40/checkpoint.pt"),
    ("Ours B20 (20ep s0)", "OursStatic", "experiments/phase6/round4/r4_R4R2thin/checkpoint.pt"),
    ("Ours B20 (20ep s1)", "OursStatic", "experiments/phase6/round4/r4_R4R2s1/checkpoint.pt"),
    ("Ours B20 (20ep s2)", "OursStatic", "experiments/phase6/round4/r4_R4R2s2/checkpoint.pt"),
    ("TriLiteNet tiny", "TriLiteNet", "tiny"),
    ("TriLiteNet small", "TriLiteNet", "small"),
    ("TwinLiteNetPlus nano", "TwinLiteNetPlus", "nano"),
]
OURS = [m[0] for m in MODELS if m[1] == "OursStatic"]


# ---------------------------------------------------------------------------
# GPU state / exclusivity guard
# ---------------------------------------------------------------------------
def _nvsmi(fields):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip()
    return [v.strip() for v in out.split(",")]


def _compute_apps():
    return subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()


def gpu_state():
    util, mem, temp, clk, pw = _nvsmi(
        "utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw")
    return dict(util_pct=int(util), mem_mib=int(mem), temp_c=int(temp),
                sm_mhz=int(clk), power_w=float(pw), compute_apps=_compute_apps())


def assert_exclusive():
    s = gpu_state()
    if s["compute_apps"]:
        raise SystemExit("FAIL-CLOSED: another process holds the GPU:\n" + s["compute_apps"])
    if s["mem_mib"] > 300:
        raise SystemExit(
            "FAIL-CLOSED: %d MiB already allocated with no compute app -- a foreign "
            "context is present." % s["mem_mib"])
    return s


# ---------------------------------------------------------------------------
# Latency protocols
# ---------------------------------------------------------------------------
def as_shipped_ours(model, x, warmup=10, reps=3):
    """Verbatim reproduction of what evaluate_baseline.py did for OUR rows."""
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            t.append((time.perf_counter() - t0) * 1e3)
    return t


def as_shipped_base(model, x, warmup=10, reps=100):
    """Verbatim reproduction of what the baseline path did (default reps=100)."""
    return as_shipped_ours(model, x, warmup=warmup, reps=reps)


def fixed(model, x, warmup=200, reps=300):
    """Long warmup + CUDA events.  No host-side sync inside the timed loop."""
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            model(x)
            ends[i].record()
        torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def ramp_trace(model, x, iters=600, every=50):
    """Separate session: measure how long the SM clock actually takes to ramp."""
    trace = []
    with torch.no_grad():
        for i in range(iters):
            model(x)
            if i % every == 0:
                trace.append([i, int(_nvsmi("clocks.sm")[0])])
        torch.cuda.synchronize()
    trace.append([iters, int(_nvsmi("clocks.sm")[0])])
    return trace


def summ(t):
    t = sorted(t)
    n = len(t)
    return dict(n=n, mean=round(st.mean(t), 4), p50=round(t[n // 2], 4),
                p95=round(t[min(n - 1, int(n * 0.95))], 4),
                p99=round(t[min(n - 1, int(n * 0.99))], 4),
                min=round(t[0], 4), max=round(t[-1], 4),
                sd=round(st.pstdev(t) if n > 1 else 0.0, 4))


# ---------------------------------------------------------------------------
def load_model(name, preset, device):
    if name == "OursStatic":
        m, _ = eb.build(name, os.path.join(ROOT, preset), device)
    else:
        m, _ = eb.build(name, preset, device)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--repeats-shipped", type=int, default=10)
    ap.add_argument("--repeats-base", type=int, default=3)
    ap.add_argument("--repeats-fixed", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip-ramp", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")

    state0 = assert_exclusive()
    print("[d8] exclusive GPU confirmed: %s" % state0, flush=True)
    print("[d8] torch %s / cuda %s / %s" % (
        torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)), flush=True)

    # One input tensor per batch size, shared by every model -> no input-side variance.
    torch.manual_seed(0)
    X = {b: torch.randn(b, 3, 640, 640, device=device) for b in args.batch}

    results = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "cuda": torch.version.cuda, "gpu_state_start": state0,
               "batch": args.batch, "sessions": [], "ramp": {}, "t_start": time.strftime("%F %T")}
    t0 = time.time()

    for name, kind, preset in MODELS:
        if args.only and name not in args.only:
            continue
        print("\n[d8] === %s ===" % name, flush=True)
        model = load_model(kind, preset, device)
        n_par = sum(p.numel() for p in model.parameters())
        print("[d8] params=%d" % n_par, flush=True)

        for b in args.batch:
            x = X[b]
            n_shipped = args.repeats_shipped if b == 1 else max(2, args.repeats_shipped // 5)
            n_base = args.repeats_base if b == 1 else max(1, args.repeats_base - 1)
            n_fixed = args.repeats_fixed if b == 1 else max(1, args.repeats_fixed - 1)

            fw, fr = (200, 300) if b == 1 else (120, 120)
            for proto, fn, nrep in (
                ("AS_SHIPPED_OURS", lambda: as_shipped_ours(model, x), n_shipped),
                ("AS_SHIPPED_BASE", lambda: as_shipped_base(model, x), n_base),
                ("FIXED", lambda: fixed(model, x, warmup=fw, reps=fr), n_fixed),
            ):
                tb = time.time()
                sessions = []
                for r in range(nrep):
                    before = gpu_state()
                    t = fn()
                    after = gpu_state()
                    sessions.append(dict(rep=r, stats=summ(t),
                                         clock_before=before["sm_mhz"],
                                         clock_after=after["sm_mhz"],
                                         temp_c=after["temp_c"],
                                         power_w=after["power_w"]))
                p50s = [s["stats"]["p50"] for s in sessions]
                rec = dict(model=name, params=n_par, batch=b, protocol=proto,
                           n_sessions=nrep, sessions=sessions,
                           p50_min=min(p50s), p50_max=max(p50s),
                           spread_ratio=round(max(p50s) / min(p50s), 3),
                           wall_s=round(time.time() - tb, 1))
                results["sessions"].append(rec)
                print("[d8] %-16s b=%-2d n=%-3d  p50 %s  spread=%.2fx  (%.0fs)" % (
                    proto, b, nrep,
                    " ".join("%.3f" % v for v in p50s), rec["spread_ratio"],
                    rec["wall_s"]), flush=True)

        if not args.skip_ramp:
            tr = ramp_trace(model, X[1])
            results["ramp"][name] = tr
            print("[d8] ramp (b=1, iters->MHz): %s" % tr, flush=True)

        del model
        torch.cuda.empty_cache()

    results["wall_total_s"] = round(time.time() - t0, 1)
    results["gpu_state_end"] = gpu_state()
    results["t_end"] = time.strftime("%F %T")

    with open(os.path.join(OUTDIR, "raw.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    write_summary(results)
    print("\n[d8] total wall %.1f min -> %s" % (results["wall_total_s"] / 60.0, OUTDIR))


def write_summary(R):
    L = []
    A = L.append
    A("# D8 -- latency instrument: root cause and replacement column\n")
    A("Probe: `scripts/phase6_d8_latency_probe.py`.  Zero training.  GPU was held "
      "exclusively (fail-closed guard); SM clock traced, not assumed.\n")
    A("Device: `%s`, torch %s / cuda %s.  Run %s -> %s (%.1f min wall).\n" % (
        R["device"], R["torch"], R["cuda"], R["t_start"], R["t_end"],
        R["wall_total_s"] / 60.0))

    A("\n## 1. The null replicate -- five checkpoints of the SAME architecture\n")
    A("192,566 params / 1.1656 GFLOPs in all five.  A good instrument must put them "
      "on top of each other; the as-shipped instrument scatters them.\n")
    def find(model, batch, proto):
        rs = [r for r in R["sessions"]
              if r["model"] == model and r["batch"] == batch and r["protocol"] == proto]
        return rs[0] if rs else None

    A("\n| model | AS_SHIPPED_OURS p50 (10 sessions) | spread | FIXED p50 (3 sessions) | spread |")
    A("|---|---:|---:|---:|---:|")
    for name in OURS:
        sh, fx = find(name, 1, "AS_SHIPPED_OURS"), find(name, 1, "FIXED")
        if not sh or not fx:
            continue
        A("| %s | %s | **%.2fx** | %s | %.2fx |" % (
            name,
            " ".join("%.2f" % s["stats"]["p50"] for s in sh["sessions"]), sh["spread_ratio"],
            " ".join("%.2f" % s["stats"]["p50"] for s in fx["sessions"]), fx["spread_ratio"]))

    sh_means, fx_means = [], []
    for name in OURS:
        sh, fx = find(name, 1, "AS_SHIPPED_OURS"), find(name, 1, "FIXED")
        if not sh or not fx:
            continue
        sh_means.append(sum(s["stats"]["p50"] for s in sh["sessions"]) / len(sh["sessions"]))
        fx_means.append(sum(s["stats"]["p50"] for s in fx["sessions"]) / len(fx["sessions"]))
    if sh_means and fx_means:
        A("\nNull-replicate spread across the identical checkpoints: "
          "**AS_SHIPPED_OURS %.2fx**, **FIXED %.2fx**.\n" % (
              max(sh_means) / min(sh_means), max(fx_means) / min(fx_means)))

    A("\n## 2. As-shipped asymmetry -- our rows vs the baseline rows\n")
    A("\n| model | AS_SHIPPED_OURS p50 | AS_SHIPPED_BASE p50 | FIXED p50 | FIXED p95 |")
    A("|---|---:|---:|---:|---:|")
    for name, _, _ in MODELS:
        def p50(proto):
            rs = [r for r in R["sessions"] if r["model"] == name and r["batch"] == 1
                  and r["protocol"] == proto]
            return rs[0]["sessions"][0]["stats"]["p50"] if rs else None
        def p95(proto):
            rs = [r for r in R["sessions"] if r["model"] == name and r["batch"] == 1
                  and r["protocol"] == proto]
            return rs[0]["sessions"][0]["stats"]["p95"] if rs else None
        A("| %s | %s | %s | **%s** | %s |" % (
            name, p50("AS_SHIPPED_OURS"), p50("AS_SHIPPED_BASE"),
            p50("FIXED"), p95("FIXED")))
    A("\n`AS_SHIPPED_OURS` used reps=3, `AS_SHIPPED_BASE` used reps=100 -- so the two "
      "blocks of the published table were never measured by the same instrument.\n")

    A("\n## 3. SM-clock ramp trace (batch 1, sampled during a 600-iteration session)\n")
    A("\n| model | iteration -> SM clock (MHz) |")
    A("|---|---|")
    for name, tr in R["ramp"].items():
        A("| %s | %s |" % (name, " ".join("%d@%d" % (i, c) for i, c in tr)))

    A("\n## 4. Replacement latency column (FIXED, batch 1, warmup 200 / reps 300)\n")
    A("\n| model | mean | p50 | p95 | p99 | sd | sd/p50 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for name, _, _ in MODELS:
        rs = [r for r in R["sessions"] if r["model"] == name and r["batch"] == 1
              and r["protocol"] == "FIXED"]
        if not rs:
            continue
        s = rs[0]["sessions"][0]["stats"]
        A("| %s | %.3f | **%.3f** | %.3f | %.3f | %.3f | %.2f%% |" % (
            name, s["mean"], s["p50"], s["p95"], s["p99"], s["sd"],
            100.0 * s["sd"] / s["p50"]))

    with open(os.path.join(OUTDIR, "summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

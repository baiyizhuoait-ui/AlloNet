"""Edge FPS estimate for Model B -- first order, and honest about its method.

WHY A MODEL AND NOT A MEASUREMENT: there is no RK3588 / K230 / old phone on this
desk.  What we do have is a trained model whose exact cost is known
(0.1926 M params, 1.1656 GFLOPs @640x640, 0.6994 GFLOPs @384x640) and a set of
PUBLISHED throughputs for YOLO-family models on the target devices.

METHOD.  For every device we fit a two-parameter latency model

        latency(model) = a + k * GFLOPs

from two published YOLO points measured on that same device (same precision,
same ballpark harness).  `a` is the per-frame fixed cost -- preprocessing, layer
dispatch, post-processing -- and `k` is the marginal cost of compute.  The fit
is then evaluated at OUR GFLOPs.  Two consequences fall out immediately and are
the actual finding: `a` is 8-12 ms on every device, while our whole compute term
is 0.3-3.7 ms.  A model this small is not compute-bound anywhere; it is
overhead-bound.  That is why the estimate clusters instead of spreading.

For devices with only ONE published anchor we say so and give a bounded range
instead of pretending to a fit.  For devices with NO anchor (old phones) the row
is scaled from the CPU measurement taken on this machine and is flagged as the
weakest evidence in the table.

Outputs experiments/phase6/deploy_profile/edge_fps.json + a printed table.
"""
import json
import os

OUT = "experiments/phase6/deploy_profile/edge_fps.json"

OURS = {"640x640": 1.1656, "384x640": 0.6994}

# published anchors: device -> list of (GFLOPs, latency_ms, label, source)
ANCHORS = {
    "RK3588 (1 NPU core)": [
        (8.7, 27.0, "YOLOv8n INT8 37 FPS", "Orange Pi 5 Max, RKNN INT8, end-to-end"),
        (28.6, 65.0, "YOLOv8s INT8 15 FPS", "same board / same harness"),
    ],
    "Jetson Orin Nano (15W)": [
        (8.7, 1000 / 70.0, "YOLOv8n INT8 65-75 FPS", "TensorRT INT8, MAXN 15W"),
        (28.6, 1000 / 53.0, "YOLOv8s INT8 48-58 FPS", "same harness"),
    ],
    "Jetson Nano (2019)": [
        (8.7, 1000 / 27.5, "YOLOv8n INT8 25-30 FPS", "JetPack 4.x, INT8"),
        (28.6, 1000 / 10.0, "YOLOv8s INT8 8-12 FPS", "same harness"),
    ],
}
# devices with a single published anchor: report a bound, not a fit
SINGLE = {
    "K230 CanMV (KPU)": (16.5, 1000 / 38.0, "YOLOv5s >=38 FPS (nncase INT8)",
                         "vendor spec, single anchor"),
    "RK3568 (1 TOPS NPU)": (16.5, 1000 / 15.0, "YOLOv5s 14-16 FPS",
                            "RKNN INT8, single anchor"),
    "RK3588 (3-core async)": (8.7, 1000 / 111.0, "YOLOv8n INT8 111 FPS",
                              "3 NPU cores asynchronous pipeline, single anchor"),
}
# no anchor at all: scaled from the CPU measurement on this machine
SCALED = {
    "Raspberry Pi 5 (4x A76)": (0.30, 0.55, 420, "CPU only, no NPU in use"),
    "Old mid-range phone CPU (SD660 / Helio G85 class)": (0.20, 0.35, 350,
                                                          "CPU only, NNAPI unreliable on this class"),
    "Old flagship phone (SD845 class, 2018)": (0.35, 0.60, 700,
                                               "CPU path measured proxy; Adreno 630 GPU would be faster"),
}

# the cheapest camera-SoC class: no published anchor for a 3-task model at all,
# so this row is a RANGE built from its two published constraints (0.5 TOPS
# peak NPU, and a CPU roughly an order of magnitude below a Pi 5) and is the
# weakest row in the table by construction.
CAMERA_SOC = {
    "Cheap IP-camera SoC (RV1106 / Ingenic T41 class, 0.5 TOPS, 64 MB DDR)":
        dict(GFLOPs=OURS["640x640"], npu_TFLOPS_assumed=0.15,
             price="¥60-100 (board)", note="no anchor; needs a 1-hour on-device test"),
}

PRICES = {
    "RK3588 (1 NPU core)": "¥300-800 board (6 TOPS)",
    "Jetson Orin Nano (15W)": "¥2000-2800 devkit",
    "Jetson Nano (2019)": "¥400-600 used",
    "K230 CanMV (KPU)": "¥239-299 board (JD, 2026-09)",
    "RK3568 (1 TOPS NPU)": "¥150-300 board",
    "RK3588 (3-core async)": "¥300-800 board (6 TOPS)",
    "Raspberry Pi 5 (4x A76)": "¥400-600 (4 GB)",
    "Old mid-range phone CPU (SD660 / Helio G85 class)": "¥300-500 used phone",
    "Old flagship phone (SD845 class, 2018)": "¥600-900 used phone",
    "Cheap IP-camera SoC (RV1106 / Ingenic T41 class, 0.5 TOPS, 64 MB DDR)":
        "¥60-100 board",
}

# 3-task host cost the YOLO anchors do NOT include: two 640x640 argmax, one
# 25200-box NMS for nc=1, and writing two full-res masks.
HOST_EXTRA_MS = (2.0, 4.0)


def fit(pts):
    (g1, m1), (g2, m2) = pts[0], pts[1]
    k = (m2 - m1) / (g2 - g1)
    return k, m1 - k * g1


def main():
    cpu = {}
    p = os.path.join("experiments/phase6/deploy_profile", "profile.json")
    if os.path.exists(p):
        with open(p) as f:
            prof = json.load(f)
        cpu = prof.get("cpu_latency_ms", {})

    rows = []
    for dev, pts in ANCHORS.items():
        k, a = fit([(p[0], p[1]) for p in pts])
        ms = a + k * OURS["640x640"]
        ceiling = 1000.0 / pts[0][1] * (pts[0][0] / OURS["640x640"])
        rows.append(dict(device=dev, price=PRICES.get(dev), method="two-anchor fit",
                         k=round(k, 3), a=round(a, 2),
                         latency_ms=round(ms + sum(HOST_EXTRA_MS) / 2, 1),
                         FPS=round(1000.0 / (ms + sum(HOST_EXTRA_MS) / 2), 1),
                         ceiling_FPS=round(ceiling),
                         anchors=[p[2] for p in pts],
                         source=pts[0][3], confidence="medium"))
    for dev, (g, ms, label, src) in SINGLE.items():
        # single anchor -> we can only bound.  Treat the anchor's own latency as
        # an UPPER bound on ours (our model is strictly cheaper in FLOPs by
        # 14-25x), and the family fixed cost ~12 ms as the lower bound.
        # single anchor -> we can only bound.  The anchor's own latency and the
        # family fixed cost (~12 ms) are two ends; which is the floor depends on
        # the device (the 3-core async pipeline beats the family floor).
        lo = min(12.0, ms) + sum(HOST_EXTRA_MS) / 2
        hi = max(12.0, ms) + sum(HOST_EXTRA_MS) / 2
        rows.append(dict(device=dev, price=PRICES.get(dev), method="single anchor -> bound",
                         latency_ms=round((lo + hi) / 2, 1),
                         latency_ms_lo=round(lo, 1), latency_ms_hi=round(hi, 1),
                         FPS=round(1000.0 / ((lo + hi) / 2), 1),
                         FPS_lo=round(1000.0 / hi, 1), FPS_hi=round(1000.0 / lo, 1),
                         anchors=[label], source=src, confidence="low"))
    if cpu.get("640x640_t1"):
        base1 = cpu["640x640_t1"]["p50"]
        base4 = cpu["640x640_t4"]["p50"]
        for dev, (f1, f4, _p, note) in SCALED.items():
            ms = base4 / max(f4, 1e-6)
            rows.append(dict(device=dev, price=PRICES.get(dev),
                             method="scaled from this machine's CPU measurement",
                             base_note=f"measured here: {base1} ms (1 thread) / "
                                       f"{base4} ms (4 threads)",
                             latency_ms=round(ms, 1),
                             FPS=round(1000.0 / ms, 1),
                             anchors=[note], confidence="low"))

    for dev, c in CAMERA_SOC.items():
        # GFLOPs / TFLOPS is numerically the latency in ms
        # (1.1656e9 / 0.15e12 s = 7.77e-3 s = 7.77 ms)
        t_npu = c["GFLOPs"] / c["npu_TFLOPS_assumed"]
        t_cpu = 45.0                                            # ms, CPU-bound guess
        rows.append(dict(device=dev, price=c["price"],
                         method="RANGE from published constraints, no anchor",
                         latency_ms=round(t_npu, 1), latency_ms_lo=round(t_npu, 1),
                         latency_ms_hi=round(t_cpu, 1),
                         FPS=round(1000.0 / t_cpu, 1), FPS_lo=round(1000.0 / t_cpu, 1),
                         FPS_hi=round(1000.0 / t_npu, 1),
                         anchors=[c["note"]], confidence="very low"))

    for r in rows:
        r.setdefault("price", None)

    out = {"ours_GFLOPs": OURS, "host_extra_ms": list(HOST_EXTRA_MS),
           "cpu_anchor_measured": cpu, "rows": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    for r in rows:
        print(f"{r['device']:<46} {r['method']:<40} {r['latency_ms']:>6} ms "
              f"{r['FPS']:>6} FPS  ({r['confidence']})")
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()

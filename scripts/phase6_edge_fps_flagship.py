"""Flagship-SoC decade ladder for Model B (2016 -> 2025, one flagship per year).

WHAT THIS IS, AND WHAT IT IS NOT.

It is NOT a measurement.  No phone, no development board and no vendor toolchain
has run Model B.  It is a SCALING EXTRAPOLATION from two things:

  (1) the exact cost of our model -- 0.1926 M params, 1.1656 GFLOPs @640x640
      (= 0.583 GMACs), fp32 weights 0.74 MB, an architecture whose only ops are
      conv / BN / ReLU / bilinear upsample;
  (2) PUBLISHED numbers for other models on the target silicon, used to put a
      number on each chip's *accelerator speed* and on its *per-frame host
      cost*.

THE ONLY MEASURED ANCHORS IN THIS FILE are the Qualcomm AI Hub runs below.  They
are the most useful public anchors that exist, because they hold the model, the
input resolution and the runtime fixed and vary only the chip -- which is
exactly the axis this ladder needs.

WHY FRAME RATE DOES NOT TRACK SILICON.  Every row is decomposed as

        latency = t_accel + t_host

with t_accel the accelerator (NPU/DSP/GPU) time and t_host the per-frame cost
that the accelerator does NOT pay: camera capture and format conversion,
resize/normalise, decode of three detection heads, NMS over 25200 anchors, and
two 640x640 mask argmax passes.  Our whole compute term is 1.17 GFLOPs, i.e.
2.4% of YOLOv5-M.  Even on 2016 silicon that is a single-digit-millisecond
quantity, while t_host sits at 12-38 ms on every chip in the table.  So the
decade ladder is dominated by host cost, and the headline result is that ten
years of accelerator progress buys roughly 4-6x end-to-end frame rate, not the
~100x the peak-TOPS numbers suggest.

CONFIDENCE.  'medium' only where a Qualcomm AI Hub anchor exists for the same
chip family.  Everything before 2022 is 'low' or 'very low' -- it rests on
ncnn / TNN / PaddleLite MobileNet-class measurements taken on 2019-2020 phones,
scaled to our GFLOPs, plus the vendors' own AI-engine generation claims.

Outputs experiments/phase6/deploy_profile/edge_fps_flagship.json + prints a table.
"""
import json
import os

OUT = "experiments/phase6/deploy_profile/edge_fps_flagship.json"

OURS = dict(params_M=0.1926, GFLOPs_640=1.1656, GMACs_640=0.583, fp32_MB=0.74)

# ---------------------------------------------------------------------------
# MEASURED: Qualcomm AI Hub, Yolo-v5 M (21.2 M params, 640x640), QNN_DLC w8a16.
# Same model / same resolution / same runtime, chip varied -> a clean chip axis.
# ---------------------------------------------------------------------------
YOLOV5M_GFLOPs = 48.0
AIHUB_YOLOV5M_W8A16_MS = {
    "Snapdragon 8 Gen 2": 12.485,
    "Snapdragon 8 Gen 3": 8.295,
    "Snapdragon 8 Elite": 7.295,
    "Snapdragon 8 Elite Gen 5": 3.291,
    "QCS8550 (8 Gen 2 class)": 12.515,
    "QCS8275 (entry IoT)": 25.413,
}
# MEASURED: Qualcomm AI Hub, YOLOv8-Detection-Quantized (YOLOv8n, 3.18 M, INT8)
AIHUB_YOLOV8N_INT8_MS = {
    "Snapdragon 8 Gen 2 (Galaxy S23 Ultra)": 2.111,
    "RB5 / QCS8250 (2019-class IoT)": 47.75,
}

# ---------------------------------------------------------------------------
# Non-Qualcomm accelerators used for calibration of the t_accel ranges.
#   TNN  v0.1 (2020-05-29): MobileNet-v1/v2, single-thread CPU and GPU, ms.
#   PaddleLite: MobileNet-v1/v2, armv8, 4 threads, ms.
#   ncnn: MobileNetV2-YOLOv3 on Adreno 640 (SD855+), Vulkan, ms.
# MobileNet-v2 is 0.30 GMACs @224; our model is 0.583 GMACs @640, i.e. 1.94x.
# ---------------------------------------------------------------------------
CALIBRATION = {
    "TNN_single_thread_ms": {
        "Snapdragon 835": {"mobilenet_v1_cpu": 94, "mobilenet_v1_gpu": 16,
                           "mobilenet_v2_cpu": 61, "mobilenet_v2_gpu": 14},
        "Snapdragon 845": {"mobilenet_v1_cpu": 60, "mobilenet_v1_gpu": 10,
                           "mobilenet_v2_cpu": 39, "mobilenet_v2_gpu": 8},
        "Kirin 970": {"mobilenet_v1_cpu": 88, "mobilenet_v1_gpu": 12,
                      "mobilenet_v2_cpu": 58, "mobilenet_v2_gpu": 11},
    },
    "PaddleLite_armv8_4thread_ms": {
        "Snapdragon 835": {"mobilenet_v1": 49.02, "mobilenet_v2": 42.66},
        "Snapdragon 845": {"mobilenet_v1": 19.18, "mobilenet_v2": 17.36},
        "Snapdragon 855": {"mobilenet_v1": 9.60, "mobilenet_v2": 9.29},
    },
    "ncnn_vulkan_ms": {"Snapdragon 855+ (Adreno 640)": {"mobilenetv2_yolov3": 30.98}},
}

# ---------------------------------------------------------------------------
# The ladder.  One flagship per vendor per release year.
#   t_accel_ms / t_host_ms are RANGES, not points.  Their width IS the finding:
#   the accelerator term shrinks by ~15x over the decade, the host term by ~2.5x.
# ---------------------------------------------------------------------------
LADDER = [
    # year, vendor, soc, accelerator, ai_engine_gen, t_accel_ms, t_host_ms, conf, basis
    (2016, "Qualcomm", "Snapdragon 835", "Hexagon 682 DSP (HVX), no dedicated NPU",
     "2nd-gen AI Engine", (25, 45), (25, 35), "very low",
     "TNN: MobileNetV2 61 ms CPU / 14 ms GPU; A75+1.8 GHz A55"),
    (2016, "MediaTek", "Helio X25 (MT6797T)", "no NPU; CPU/DSP only",
     "-", (45, 80), (28, 38), "very low",
     "10-core tri-cluster, 20 nm; no published INT8 accelerator"),
    (2017, "Qualcomm", "Snapdragon 845", "Hexagon 685 DSP + Adreno 630",
     "3rd-gen AI Engine (3x SD835)", (14, 26), (20, 28), "very low",
     "TNN: MobileNetV2 39 ms CPU / 8 ms GPU; PaddleLite mobilenet_v2 17.36 ms"),
    (2017, "MediaTek", "Helio X30 (MT6799)", "no NPU; CPU/GPU only",
     "-", (35, 65), (26, 34), "very low",
     "10 nm, PowerVR GT7400; no INT8 accelerator"),
    (2018, "Qualcomm", "Snapdragon 855", "Hexagon 690 DSP + Tensor Accelerator",
     "4th-gen AI Engine, >7 TOPS", (6, 14), (18, 25), "low",
     "PaddleLite: mobilenet_v1 9.60 ms (4 threads); ncnn Vulkan yolov3 31 ms"),
    (2018, "MediaTek", "Helio P90 (MT6779)", "APU 2.0, 1165 GMACs",
     "APU 2.0", (22, 40), (22, 30), "very low",
     "MediaTek's first flagship-class APU; 4x Helio P70 claim"),
    (2019, "Qualcomm", "Snapdragon 865", "Hexagon 698",
     "5th-gen AI Engine, 15 TOPS", (4, 10), (17, 23), "low",
     "generation claim 2x SD855; offset from the SD855 measured row"),
    (2019, "MediaTek", "Dimensity 1000 (MT6889)", "APU 3.0 (6 cores)",
     "APU 3.0", (10, 20), (19, 26), "low",
     "first Dimensity flagship; offset below the SD855/865 rows"),
    (2020, "Qualcomm", "Snapdragon 888", "Hexagon 780",
     "6th-gen AI Engine, 26 TOPS", (3, 8), (16, 22), "low",
     "26 TOPS published; 73% over SD865 per Qualcomm"),
    (2020, "MediaTek", "Dimensity 1000+", "APU 3.0",
     "APU 3.0", (8, 17), (18, 24), "low",
     "refresh of Dimensity 1000, 144 Hz display support"),
    (2021, "Qualcomm", "Snapdragon 8 Gen 1", "Hexagon (7th-gen AI Engine)",
     "7th-gen AI Engine", (3, 7), (15, 21), "low",
     "double performance-and-efficiency claim; no absolute TOPS published"),
    (2021, "MediaTek", "Dimensity 9000 (MT6983)", "APU 590",
     "APU 590", (4, 9), (16, 22), "low",
     "first Dimensity to match Snapdragon flagship AI tier"),
    (2022, "Qualcomm", "Snapdragon 8 Gen 2", "Hexagon + INT4 support",
     "8th-gen AI Engine", (2, 6), (14, 20), "medium",
     "AI Hub MEASURED: Yolo-v5 M w8a16 = 12.485 ms; YOLOv8n int8 = 2.111 ms"),
    (2022, "MediaTek", "Dimensity 9200 (MT6985)", "NPU 690",
     "NPU 690", (3.5, 8), (15, 21), "low",
     "positioned one tier below the SD8Gen2 row"),
    (2023, "Qualcomm", "Snapdragon 8 Gen 3", "Hexagon NPU",
     "9th-gen AI Engine, 45 TOPS", (2, 5), (13, 19), "medium",
     "AI Hub MEASURED: Yolo-v5 M w8a16 = 8.295 ms (1.51x over 8 Gen 2)"),
    (2023, "MediaTek", "Dimensity 9300 (MT6989)", "NPU 790",
     "NPU 790", (2.5, 6), (14, 20), "low",
     "ETHZ AI Benchmark v6: Dimensity 9400 6773 vs SD8Gen3 5374 (+26%)"),
    (2024, "Qualcomm", "Snapdragon 8 Elite", "Hexagon NPU (2nd gen in this line)",
     "Hexagon NPU, +45% AI", (1.5, 5), (13, 18), "medium",
     "AI Hub MEASURED: Yolo-v5 M w8a16 = 7.295 ms (only 1.14x over 8 Gen 3)"),
    (2024, "MediaTek", "Dimensity 9400 (MT6991)", "NPU 890, 48 TOPS",
     "NPU 890", (2, 6), (13, 19), "low",
     "48 TOPS (INT8); ETHZ 6773"),
    (2025, "Qualcomm", "Snapdragon 8 Elite Gen 5", "Hexagon NPU (3rd gen)",
     "Hexagon NPU", (1.5, 4), (12, 17), "medium",
     "AI Hub MEASURED: Yolo-v5 M w8a16 = 3.291 ms (2.52x over 8 Gen 3)"),
    (2025, "MediaTek", "Dimensity 9500", "NPU 990, ~100 TOPS",
     "NPU 990", (1.5, 5), (12, 18), "low",
     "NPU 990 (9th-gen), 4.21 GHz C1-Ultra; TSMC N3P"),
]

# ---------------------------------------------------------------------------
# The existing edge-device rows, carried over unchanged so the ladder can be
# read as one picture with the flagship axis (values from edge_fps.json).
# ---------------------------------------------------------------------------
EDGE_CARRIED = [
    ("RK3588 (3 NPU cores, async)", 74.0, "single-board NPU", "low"),
    ("Jetson Nano (2019, used)", 65.5, "Jetson", "medium"),
    ("Jetson Orin Nano 15W", 64.3, "Jetson", "medium"),
    ("RK3588 (1 NPU core)", 64.1, "single-board NPU", "medium"),
    ("K230 CanMV", 45.1, "single-board NPU", "low"),
    ("Old flagship phone SD845-class, CPU path", 35.5, "old phone CPU", "low"),
    ("Raspberry Pi 5 (4x A76, CPU)", 32.5, "CPU only", "low"),
    ("RK3568 (1 TOPS NPU)", 23.6, "single-board NPU", "low"),
    ("IP-camera SoC RV1106 / T41", 22.2, "camera SoC", "very low"),
    ("Old mid-range phone SD660 / G85, CPU", 20.7, "old phone CPU", "low"),
]


def fps_from_range(t_accel, t_host):
    lo_ms = t_accel[0] + t_host[0]
    hi_ms = t_accel[1] + t_host[1]
    mid_ms = 0.5 * (lo_ms + hi_ms)
    return dict(
        latency_ms_mid=round(mid_ms, 1),
        FPS_mid=round(1000.0 / mid_ms, 1),
        FPS_hi=round(1000.0 / lo_ms, 1),      # faster end
        FPS_lo=round(1000.0 / hi_ms, 1),      # slower end
        t_accel_ms=list(t_accel),
        t_host_ms=list(t_host),
    )


def main():
    rows = []
    for (year, vendor, soc, accel, gen, ta, th, conf, basis) in LADDER:
        r = dict(year=year, vendor=vendor, soc=soc, accelerator=accel,
                 ai_engine=gen, confidence=conf, basis=basis)
        r.update(fps_from_range(ta, th))
        rows.append(r)

    # measured accelerator throughput, derived from the AI Hub anchors
    tput = []
    for chip, ms in AIHUB_YOLOV5M_W8A16_MS.items():
        tput.append(dict(chip=chip, yolov5m_ms=ms,
                         GFLOPs_per_s=round(YOLOV5M_GFLOPs / (ms / 1000.0)),
                         vs_8gen2=round(AIHUB_YOLOV5M_W8A16_MS["Snapdragon 8 Gen 2"] / ms, 2)))

    out = dict(
        _what_this_is="scaling extrapolation, NOT a measurement; see module docstring",
        ours=OURS,
        measured_anchors=dict(
            yolov5m_w8a16_640_ms=AIHUB_YOLOV5M_W8A16_MS,
            yolov8n_int8_640_ms=AIHUB_YOLOV8N_INT8_MS,
            source="Qualcomm AI Hub model cards (huggingface.co/qualcomm)",
        ),
        measured_accelerator_throughput=tput,
        calibration=CALIBRATION,
        ladder=rows,
        edge_carried=EDGE_CARRIED,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"{'year':>4} {'vendor':<9} {'soc':<26} {'accel ms':>10} {'host ms':>9}"
          f" {'lat ms':>7} {'FPS':>6} {'range':>12} conf")
    for r in sorted(rows, key=lambda x: (x["year"], x["vendor"])):
        print(f"{r['year']:>4} {r['vendor']:<9} {r['soc']:<26}"
              f" {str(r['t_accel_ms']):>10} {str(r['t_host_ms']):>9}"
              f" {r['latency_ms_mid']:>7} {r['FPS_mid']:>6.1f}"
              f" {str(r['FPS_lo']) + '-' + str(r['FPS_hi']):>12} {r['confidence']}")
    print()
    print("measured accelerator throughput (44.8 GFLOPs YOLOv5-M per frame):")
    for t in tput:
        print(f"  {t['chip']:<26} {t['yolov5m_ms']:>7} ms   "
              f"{t['GFLOPs_per_s']:>7} GFLOPs/s   x{t['vs_8gen2']} vs 8 Gen 2")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

"""Which low-cost devices can actually RUN Model B, and at what precision?

The FPS table (phase6_edge_fps_estimate.py) answers "how fast", and it does so
honestly only for a handful of devices that have published anchors.  This table
answers the question that comes FIRST and that the FPS table cannot answer at
all: does the device have a path for our graph, and at what precision.

Why it matters: our previous framing said "FP32 is not what edge NPUs execute",
which is true but too coarse.  The real structure of the 2026 low-cost market is
a PRECISION LADDER, and the rungs are not the same on every part:

  INT8-only, no floating point at all : Coral Edge TPU, Hailo-8 / 8L, Arm
      Ethos-U (Vela), Rockchip RKNN family, K230 KPU, Sophgo CV1800B
  FP16 as well                        : Jetson Nano / Xavier / Orin family
  INT4 / FP4 as a native path         : Hailo-10H (INT4), Qualcomm Hexagon HMX
                                        (INT4), Apple ANE (4-bit), Jetson Thor
                                        (NVFP4 + FP8 via Tensor Cores)

So the deployment statement we can defend is not "use INT8" but: INT8 is the
only rung that exists on EVERY cheap target, and it is the rung this model was
designed for.  4-bit is a rung that exists only on parts that are either newer,
more expensive, or phone-class.

Every row carries an evidence tag.  Prices are what we could verify; where
vendor materials disagree with each other the disagreement is recorded rather
than averaged away.

CPU only, no GPU, no network.
"""
import json
import os

OUT = "experiments/phase6/deploy_profile/edge_device_matrix.json"

# class, price, NPU, native precisions, FP32 path?, toolchain, verdict for our
# graph (49 conv + 5 bilinear resize + 1 concat), evidence
MATRIX = [
    dict(name="Milk-V Duo (Sophgo CV1800B)", cls="ultra-cheap SBC / camera",
         price="¥35-53", npu="0.2-0.5 TOPS INT8", prec="INT8",
         fp32="no", mem="64 MB DDR2 (~800 MB/s)", tool="TPU SDK (MLIR quantiser)",
         verdict="Fits (0.74 MB weights, 6.3 MB peak activation) but the "
                 "toolchain's op coverage is the open question; 5 bilinear "
                 "resizes are a real risk.  Vendor materials disagree on TOPS "
                 "by 2.5x (0.2 vs 0.5) so treat any number as a range.",
         evidence="board docs + Sophgo README (they disagree); no measured anchor"),
    dict(name="K230 CanMV (Canaan KPU)", cls="cheap vision board",
         price="¥239-299", npu="vendor KPU (nncase INT8)", prec="INT8",
         fp32="no", mem="on-board DDR", tool="nncase",
         verdict="Intended target class.  One published YOLOv5s anchor only.",
         evidence="vendor spec, single anchor (see FPS table)"),
    dict(name="Rockchip RK3568", cls="cheap SBC", price="¥150-300",
         npu="~1 TOPS INT8", prec="INT8", fp32="no", mem="external DDR",
         tool="RKNN", verdict="Intended target class.",
         evidence="single published YOLOv5s anchor"),
    dict(name="Rockchip RK3588 (3 NPU cores)", cls="mid SBC", price="¥300-800",
         npu="6 TOPS INT8", prec="INT8", fp32="no", mem="external DDR",
         tool="RKNN", verdict="Intended target class; best measured anchor set.",
         evidence="two published YOLOv8 anchors, same board/harness"),
    dict(name="Google Coral Edge TPU (USB)", cls="USB accelerator", price="$60",
         npu="4 TOPS INT8", prec="INT8 ONLY", fp32="no", mem="host",
         tool="TFLite -> Edge TPU compiler",
         verdict="Runs only fully-INT8 graphs; anything else falls back to the "
                 "host CPU and the accelerator is wasted.  Stack deprecated in "
                 "2024 and the compiler tops out at TFLite.",
         evidence="vendor + 2026 market survey"),
    dict(name="Raspberry Pi AI HAT+ (Hailo-8L)", cls="accelerator HAT",
         price="$70", npu="13 TOPS INT8", prec="INT8 ONLY", fp32="no",
         mem="on-chip SRAM, no DRAM", tool="Hailo Dataflow Compiler",
         verdict="No FP16/FP32 path at all.  SRAM-only design caps model size "
                 "but our 0.74 MB is trivial.  Op coverage (bilinear resize, "
                 "the box-decode elementwise chain) must be compiled to know.",
         evidence="vendor + 2026 market survey"),
    dict(name="Hailo-8 (26 TOPS, M.2)", cls="accelerator module", price="$110-200",
         npu="26 TOPS INT8", prec="INT8 ONLY", fp32="no",
         mem="on-chip SRAM, no DRAM", tool="Hailo Dataflow Compiler",
         verdict="Same as Hailo-8L with more headroom.  Explicitly no native "
                 "FP16/FP32 compute.",
         evidence="vendor spec sheet"),
    dict(name="Hailo-10H", cls="accelerator module (newer)", price="n/a here",
         npu="40 TOPS INT4 / INT16 / INT8", prec="INT4, INT8, INT16",
         fp32="no", mem="4-8 GB LPDDR4X", tool="Hailo Dataflow Compiler",
         verdict="One of the few cheap-ish parts with a genuine 4-bit path; "
                 "aimed at small language models, not at a 0.19 M-parameter CNN.",
         evidence="vendor + independent deep dive"),
    dict(name="Arm Ethos-U55 / U85 (MCU)", cls="microNPU", price="in MCU silicon",
         npu="0.5-4 TOPS INT8", prec="INT8", fp32="no", mem="MCU SRAM",
         tool="Vela compiler",
         verdict="Our 6.3 MB peak activation does NOT fit a typical MCU SRAM "
                 "budget; this class is out of reach for a 640x640 three-task "
                 "model regardless of quantisation.",
         evidence="toolchain docs + 2026 edge-AI survey"),
    dict(name="ST STM32N6 (Neural-ART)", cls="MCU with NPU", price="~$15-25 chip",
         npu="~0.6 TOPS", prec="INT8", fp32="no", mem="MCU SRAM + external",
         tool="ST Edge AI",
         verdict="Same SRAM argument as Ethos-U; a 640x640 three-task model is "
                 "not an MCU workload.  Listed to close the question.",
         evidence="2026 edge-AI survey"),
    dict(name="Jetson Nano (2019)", cls="SBC with GPU", price="¥400-600 used",
         npu="128-core Maxwell", prec="FP16, INT8", fp32="yes (slow)",
         mem="4 GB LPDDR4", tool="TensorRT",
         verdict="Has a real floating-point path, so it can run the model "
                 "un-quantised if you insist.  Weakest of the Jetson family.",
         evidence="two published YOLOv8 INT8 anchors"),
    dict(name="Jetson Orin Nano 8GB", cls="SoM", price="$249",
         npu="40 INT8 TOPS", prec="FP16, BF16, TF32, INT8", fp32="yes",
         mem="8 GB LPDDR5", tool="TensorRT",
         verdict="Most generous precision ladder in the cheap tier; the "
                 "reference target for an INT8 deployment and the only class "
                 "here where a FP16 comparison is also measurable.",
         evidence="vendor datasheet"),
    dict(name="Jetson Orin Nano Super", cls="SoM (2025 refresh)", price="$249",
         npu="67 INT8 TOPS", prec="FP16, BF16, TF32, INT8", fp32="yes",
         mem="8 GB LPDDR5, 102 GB/s", tool="TensorRT",
         verdict="Same answer with 1.7x the throughput; memory bandwidth high "
                 "enough that our 6.3 MB working set never bottlenecks it.",
         evidence="vendor datasheet"),
    dict(name="Snapdragon 8 Gen 3 / X Elite class (Hexagon)", cls="phone/PC NPU",
         price="in device", npu="45-80 TOPS", prec="INT8 baseline, native INT4 "
         "(HMX)", fp32="GPU/CPU fallback", mem="shared LPDDR5",
         tool="QNN / QAIRT",
         verdict="The class where a 4-bit result would actually be actionable. "
                 "Also the class where our lane axis is most likely to be the "
                 "binding constraint (see the bit-width sweep).",
         evidence="platform docs + independent 2026 survey"),
    dict(name="Apple Neural Engine (A18 / M4)", cls="phone/PC NPU",
         price="in device", npu="~35-38 TOPS", prec="16-bit + 4-bit "
         "palettisation", fp32="GPU/CPU fallback", mem="unified",
         tool="Core ML",
         verdict="4-bit is a first-class weight-compression path; ANE is also "
                 "reported to be the least forgiving about non-4/8-bit shapes.",
         evidence="platform docs + independent 2026 survey"),
    dict(name="Jetson Thor (Blackwell, T5000)", cls="robotics SoM",
         price="high (100s of $)", npu="up to 2,070 FP4 TFLOPS dense",
         prec="FP4, FP6, FP8, FP16, BF16, INT8", fp32="yes",
         mem="128 GB LPDDR5X, ~273 GB/s", tool="TensorRT / CUDA",
         verdict="The only platform here with hardware FP4.  Also 40-130 W and "
                 "an order of magnitude more expensive than every other row -- "
                 "relevant as a ceiling, not as a target for a 0.19 M model.",
         evidence="vendor datasheet + wiki"),
    dict(name="Raspberry Pi 5 (4x A76, CPU only)", cls="SBC, no NPU", price="¥400-600",
         npu="none", prec="FP32 on CPU", fp32="yes", mem="4-8 GB",
         tool="ONNX Runtime / TFLite / XNNPACK",
         verdict="The fallback that always works and never accelerates.  "
                 "Measured here (scaled) at 32.5 FPS, which is the honest floor "
                 "for a 1.17 GFLOPs model with no accelerator at all.",
         evidence="scaled from this machine's measured CPU latency"),
]

# Facts our own profiling established, repeated here because every row above
# depends on them.
OURS = dict(params=192566, weights_MB={"FP32": 0.735, "FP16": 0.367, "INT8": 0.184},
            peak_activation_MB=6.25, GFLOPs_640=1.1656, GFLOPs_384x640=0.6994,
            ops="49 conv2d + 41 BatchNorm(folded) + 5 bilinear Resize + 1 Concat "
                "+ 3 Sigmoid + 3 Pow + 33 Mul/Add/Sub elementwise")


def main():
    res = {"ours": OURS, "rows": MATRIX,
           "precision_ladder": {
               "int8_only_parts": ["Coral Edge TPU", "Hailo-8", "Hailo-8L",
                                   "Arm Ethos-U", "Rockchip RKNN family",
                                   "K230 KPU", "Sophgo CV1800B"],
               "int4_or_fp4_parts": ["Hailo-10H (INT4)", "Qualcomm Hexagon HMX (INT4)",
                                     "Apple ANE (4-bit)", "Jetson Thor (NVFP4)"],
               "fp32_path_parts": ["Jetson Nano/Orin/Thor", "CPU-only boards",
                                   "x86 with iGPU"],
           },
           "_note": "Capability matrix, not a benchmark.  Every row names the "
                    "kind of evidence behind it; no FPS is asserted here -- see "
                    "phase6_edge_fps_estimate.py for the rows that have anchors."}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".tmp", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    os.replace(OUT + ".tmp", OUT)
    for r in MATRIX:
        print(f"{r['name']:<44} {r['price']:<14} {r['prec']:<28} {r['fp32']}")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()

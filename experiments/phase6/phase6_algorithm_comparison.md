# Phase 6 - cross-algorithm positioning (10 models, one harness)

> **READ THE PROTOCOL COLUMN BEFORE QUOTING ANY NUMBER HERE.**
> This table is internally comparable and externally un-comparable. Audit: `docs/PHASE6_BENCHMARK_AUDIT.md`.
> - **FLOPs** = 2 x MACs (thop) at **640x640**. Published FLOPs are MACs at **384x640**. Ratio 3.27-3.36 on the 7 models whose params also match. **Never put our FLOPs next to a published FLOPs.**
> - **Lane columns**: our value / published Lane IoU = 0.71-0.85 (spread 0.14, not a constant) => **not cross-paper comparable, and not repairable by a scale factor.**
> - **DA column**: within ~1 point of published => roughly comparable.
> - **Detection column**: reproduces published mAP50 to within 0.1 => comparable.
> - **TwinLiteNet / TwinLiteNet+ have no detection head** (2-task, not 3-task).
> - `accuracy per GFLOP` is a **secondary** indicator only - a tiny model can win it trivially. Prefer the Pareto frontier and budget-constrained accuracy.

Protocol: BDD100K `tri_val`, 10,000 images, 640x640, FP32, single-class vehicle detection.
External rows = official released weights; ours = trained by us (100 ep, seed 0).
Every row measured by `evaluation/evaluate_baseline.py`.


## Full table (sorted by cost)

| model | params (M) | FLOPs (G) | mAP50 | DA mIoU | DA fgIoU | Lane mIoU | Lane fgIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours (Model B) **(ours)** | 0.193 | 1.166 | 0.5382 | 0.8673 | 0.7888 | 0.5978 | 0.2187 |
| TriLiteNet tiny | 0.151 | 1.800 | 0.4953 | 0.8796 | 0.8045 | 0.5914 | 0.1952 |
| TwinLiteNetPlus nano | 0.033 | 1.900 | - | 0.8634 | 0.7776 | 0.5866 | 0.1859 |
| TwinLiteNetPlus small | 0.122 | 4.700 | - | 0.8986 | 0.8336 | 0.6019 | 0.2165 |
| TriLiteNet small | 0.592 | 6.600 | 0.6326 | 0.9053 | 0.8458 | 0.6037 | 0.2198 |
| TwinLiteNet | 0.440 | 14.100 | - | 0.9114 | 0.8551 | 0.6077 | 0.2281 |
| TwinLiteNetPlus medium | 0.479 | 15.400 | - | 0.9191 | 0.8672 | 0.6087 | 0.2297 |
| TriLiteNet base | 2.350 | 25.400 | 0.7235 | 0.9203 | 0.8697 | 0.6123 | 0.2371 |
| YOLOP | 7.940 | 31.300 | 0.7657 | 0.9115 | 0.8556 | 0.6038 | 0.2247 |
| TwinLiteNetPlus large | 1.944 | 58.600 | - | 0.9279 | 0.8815 | 0.6161 | 0.2450 |

## Rank of ours among the 10 models

| axis | ours | rank | #models | best |
|---|---:|---:|---:|---:|
| mAP50 | 0.5382 | **4 / 5** | 5 | 0.7657 |
| DA mIoU | 0.8673 | **9 / 10** | 10 | 0.9279 |
| Lane mIoU | 0.5978 | **8 / 10** | 10 | 0.6161 |
| Lane fgIoU | 0.2187 | **7 / 10** | 10 | 0.2450 |

Our FLOPs (1.166 G) is the **lowest of all 10** - rank 1 / 10 on cost.


## Accuracy per GFLOP (the efficiency axis)

| model | FLOPs (G) | Lane mIoU / GFLOP | DA mIoU / GFLOP |
|---|---:|---:|---:|
| Ours (Model B) **(ours)** | 1.166 | **0.5129** | 0.7441 |
| TriLiteNet tiny | 1.800 | **0.3286** | 0.4887 |
| TwinLiteNetPlus nano | 1.900 | **0.3087** | 0.4544 |
| TwinLiteNetPlus small | 4.700 | **0.1281** | 0.1912 |
| TriLiteNet small | 6.600 | **0.0915** | 0.1372 |
| TwinLiteNet | 14.100 | **0.0431** | 0.0646 |
| TwinLiteNetPlus medium | 15.400 | **0.0395** | 0.0597 |
| TriLiteNet base | 25.400 | **0.0241** | 0.0362 |
| YOLOP | 31.300 | **0.0193** | 0.0291 |
| TwinLiteNetPlus large | 58.600 | **0.0105** | 0.0158 |

## Pareto check

On `(cost, Lane mIoU)`, ours is dominated by **nobody** and strictly dominates:
- YOLOP is strictly dominated by TwinLiteNet, TwinLiteNetPlus medium, TriLiteNet base
- TwinLiteNetPlus nano is strictly dominated by TriLiteNet tiny, Ours (Model B)
- TriLiteNet tiny is strictly dominated by Ours (Model B)

On `(cost, DA mIoU)`, ours is dominated by **nobody**, and dominates:
- YOLOP is strictly dominated by TwinLiteNetPlus medium, TriLiteNet base
- TwinLiteNetPlus nano is strictly dominated by TriLiteNet tiny, Ours (Model B)

## Latency (batch 1, FP32) - and the retracted column

| model | FIXED p50 (ms) | old as-shipped p50 (ms) | note |
|---|---:|---:|---|
| Ours (Model B) **(ours)** | **2.396** | - |  |
| TriLiteNet tiny | **3.510** | 4.4 | retracted: biased instrument |
| TwinLiteNetPlus nano | **5.031** | 5.2 | retracted: biased instrument |
| TriLiteNet small | **5.384** | 6.8 | retracted: biased instrument |
| YOLOP | not measured | 14.6 | retracted: biased instrument |
| TwinLiteNet | not measured | 9.0 | retracted: biased instrument |
| TwinLiteNetPlus small | not measured | 6.1 | retracted: biased instrument |
| TwinLiteNetPlus medium | not measured | 9.3 | retracted: biased instrument |
| TwinLiteNetPlus large | not measured | 15.0 | retracted: biased instrument |
| TriLiteNet base | not measured | 9.4 | retracted: biased instrument |

The as-shipped instrument (warmup 10 / reps 3 for ours, reps 100 for baselines) returned 10.2152 ms for a model whose true p50 is 2.3964 ms - a 4.3x inflation - and scattered five identical checkpoints over 1.73-4.51x. Only the FIXED column is usable; a full sweep of the remaining 5 baselines still needs to run.


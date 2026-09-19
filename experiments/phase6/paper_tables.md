# Phase 6 — main result tables

Protocol: BDD100K `tri_val`, 10,000 images, 640x640 letterbox, FP32, single-class vehicle detection. Baselines are **official released weights**; ours are trained by us. All rows measured by `evaluation/evaluate_baseline.py`.


## Table 1 — the <=2 GFLOPs tier, plus one larger reference row

_TriLiteNet small (6.6 G) is **out of tier** and is shown only as a scaling reference. Any claim in the paper is confined to the <=2 G rows: TriLiteNet tiny (1.8 G), TLP nano (1.9 G) and ours (1.166 G)._

| model | budget | params (M) | FLOPs (G) | mAP50 | DA mIoU | Lane mIoU | Lane fgIoU |
|---|---|---:|---:|---:|---:|---:|---:|
| TriLiteNet tiny | official | 0.151 | 1.8 | 0.4953 | 0.8796 | 0.5914 | 0.1952 |
| TwinLiteNetPlus nano | official | 0.033 | 1.9 | MISSING | 0.8634 | 0.5866 | 0.1859 |
| TriLiteNet small | official | 0.592 | 6.6 | 0.6326 | 0.9053 | 0.6037 | 0.2198 |
| Ours (Model B) | 100 ep (ours, main) | 0.193 | 1.166 | 0.5382 | 0.8673 | 0.5978 | 0.2187 |
| Ours (Model B) | 40 ep (ours) | 0.193 | 1.166 | 0.5198 | 0.8594 | 0.5992 | 0.2203 |
| Ours (Model B) | 20 ep (ours) | 0.193 | 1.166 | 0.4954 | 0.8528 | 0.5946 | 0.2121 |


## Table 1b — gap vs TriLiteNet tiny, and what the epoch lever bought

| axis | 40ep gap | 100ep gap | change | 2σ band | significant? |
|---|---:|---:|---:|---:|---|
| mAP50 | +0.0245 | +0.0429 | +0.0184 | ±0.0060 | yes |
| DA mIoU | -0.0202 | -0.0123 | +0.0079 | ±0.0060 | yes |
| Lane mIoU | +0.0078 | +0.0064 | -0.0014 | ±0.0062 | yes |
| Lane fgIoU | +0.0251 | +0.0235 | -0.0016 | ±0.0100 | yes |


**Pre-registered DA decision rule** (`phase6_final100_preregistration.md` §6): 
`g = 0.0123 <= 0.0141` → **D-2: residual lies inside seed noise; seeds 1–2 required before any parity language.**


## Table 2 — the epoch lever on our own model (seed 0)

| budget | mAP50 | DA mIoU | Lane mIoU | Lane fgIoU |
|---|---:|---:|---:|---:|
| 20 ep | 0.4954 | 0.8528 | 0.5946 | 0.2121 |
| 40 ep | 0.5198 | 0.8594 | 0.5992 | 0.2203 |
| 100 ep | 0.5382 | 0.8673 | 0.5978 | 0.2187 |
| 20 ep, mean of 3 seeds | 0.4978 | 0.8562 | 0.5959 | 0.2142 |


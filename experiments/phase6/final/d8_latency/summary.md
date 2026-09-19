# D8 -- latency instrument: root cause and replacement column

Probe: `scripts/phase6_d8_latency_probe.py`.  Zero training.  GPU was held exclusively (fail-closed guard); SM clock traced, not assumed.

Device: `NVIDIA GeForce RTX 5060 Laptop GPU`, torch 2.11.0+cu130 / cuda 13.0.  Run 2026-09-14 10:08:54 -> 2026-09-14 10:13:46 (4.9 min wall).


## 1. The null replicate -- five checkpoints of the SAME architecture

192,566 params / 1.1656 GFLOPs in all five.  A good instrument must put them on top of each other; the as-shipped instrument scatters them.


| model | AS_SHIPPED_OURS p50 (10 sessions) | spread | FIXED p50 (3 sessions) | spread |
|---|---:|---:|---:|---:|
| Ours B100 (100ep s0) | 10.22 2.72 2.40 2.42 2.60 2.52 2.66 2.66 2.27 2.32 | **4.51x** | 2.40 2.49 2.45 | 1.04x |
| Ours B40 (40ep s0) | 2.59 2.48 2.41 4.78 2.21 2.55 7.93 3.17 2.27 2.47 | **3.59x** | 2.38 2.48 2.44 | 1.04x |
| Ours B20 (20ep s0) | 2.41 2.52 2.45 2.19 2.58 2.51 2.48 2.22 3.80 2.51 | **1.73x** | 2.38 2.57 2.29 | 1.12x |
| Ours B20 (20ep s1) | 2.67 2.43 2.74 2.54 2.52 2.54 2.21 8.50 2.50 2.54 | **3.85x** | 2.81 2.67 2.65 | 1.06x |
| Ours B20 (20ep s2) | 2.31 2.79 2.63 2.70 4.95 2.85 2.26 2.24 2.52 2.27 | **2.21x** | 2.32 2.66 2.52 | 1.15x |

Null-replicate spread across the identical checkpoints: **AS_SHIPPED_OURS 1.28x**, **FIXED 1.12x**.


## 2. As-shipped asymmetry -- our rows vs the baseline rows


| model | AS_SHIPPED_OURS p50 | AS_SHIPPED_BASE p50 | FIXED p50 | FIXED p95 |
|---|---:|---:|---:|---:|
| Ours B100 (100ep s0) | 10.2152 | 2.3005 | **2.3964** | 5.1143 |
| Ours B40 (40ep s0) | 2.5938 | 2.9733 | **2.3829** | 8.329 |
| Ours B20 (20ep s0) | 2.4145 | 2.4978 | **2.3843** | 7.6086 |
| Ours B20 (20ep s1) | 2.67 | 2.7691 | **2.8122** | 8.7347 |
| Ours B20 (20ep s2) | 2.3071 | 2.4481 | **2.3238** | 3.2206 |
| TriLiteNet tiny | 4.2529 | 4.3067 | **3.5096** | 9.9757 |
| TriLiteNet small | 5.6009 | 10.0174 | **5.3836** | 12.0686 |
| TwinLiteNetPlus nano | 7.1488 | 3.9734 | **5.0314** | 6.9145 |

`AS_SHIPPED_OURS` used reps=3, `AS_SHIPPED_BASE` used reps=100 -- so the two blocks of the published table were never measured by the same instrument.


## 3. SM-clock ramp trace (batch 1, sampled during a 600-iteration session)


| model | iteration -> SM clock (MHz) |
|---|---|
| Ours B100 (100ep s0) | 0@2940 50@2940 100@2310 150@2310 200@2310 250@2340 300@2340 350@2692 400@2692 450@2662 500@2662 550@2655 600@2655 |
| Ours B40 (40ep s0) | 0@2940 50@2940 100@2310 150@2310 200@2310 250@2670 300@2670 350@2670 400@2325 450@2325 500@2580 550@2580 600@2580 |
| Ours B20 (20ep s0) | 0@2940 50@2940 100@2940 150@2940 200@2685 250@2685 300@2662 350@2662 400@2670 450@2670 500@2670 550@2662 600@2662 |
| Ours B20 (20ep s1) | 0@2932 50@2932 100@2932 150@2932 200@2505 250@2505 300@2505 350@2550 400@2550 450@2550 500@2670 550@2670 600@2677 |
| Ours B20 (20ep s2) | 0@2932 50@2932 100@2932 150@2617 200@2617 250@2692 300@2692 350@2692 400@2685 450@2685 500@2685 550@2685 600@2685 |
| TriLiteNet tiny | 0@2917 50@2917 100@2685 150@2685 200@2662 250@2662 300@2887 350@2887 400@2752 450@2752 500@2790 550@2790 600@2685 |
| TriLiteNet small | 0@2910 50@2430 100@2430 150@2737 200@2895 250@2805 300@2745 350@2745 400@2737 450@2835 500@2835 550@2715 600@2812 |
| TwinLiteNetPlus nano | 0@2917 50@2917 100@2520 150@2520 200@2752 250@2752 300@2752 350@2745 400@2745 450@2827 500@2827 550@2812 600@2812 |

## 4. Replacement latency column (FIXED, batch 1, warmup 200 / reps 300)


| model | mean | p50 | p95 | p99 | sd | sd/p50 |
|---|---:|---:|---:|---:|---:|---:|
| Ours B100 (100ep s0) | 2.695 | **2.396** | 5.114 | 8.921 | 1.285 | 53.61% |
| Ours B40 (40ep s0) | 3.444 | **2.383** | 8.329 | 8.842 | 2.237 | 93.87% |
| Ours B20 (20ep s0) | 2.870 | **2.384** | 7.609 | 8.262 | 1.560 | 65.43% |
| Ours B20 (20ep s1) | 3.942 | **2.812** | 8.735 | 10.128 | 2.231 | 79.32% |
| Ours B20 (20ep s2) | 2.465 | **2.324** | 3.221 | 7.498 | 0.782 | 33.64% |
| TriLiteNet tiny | 5.472 | **3.510** | 9.976 | 11.424 | 6.403 | 182.45% |
| TriLiteNet small | 6.057 | **5.384** | 12.069 | 15.867 | 2.029 | 37.69% |
| TwinLiteNetPlus nano | 4.909 | **5.031** | 6.915 | 7.249 | 1.449 | 28.80% |

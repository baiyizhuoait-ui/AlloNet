# Consistency sweep -- input=384, gt=ours+official
canvas [384, 640], content 640x360, 10000 images

> Rebuilt from `experiments/phase6/consistency/stageB_in384.log` at 2026-09-14 13:33:35 -- the original aggregate write crashed on a numpy.bool_; values are verbatim log lines.

| model | da_mIoU_ours | da_mIoU_official | lane_fg_iou_ours | lane_fg_iou_official | lane_mIoU_ours | lane_mIoU_official | lane_line_acc_ours | lane_line_acc_official |
|---|---|---|---|---|---|---|---|---|
| OursStatic:~/ai_study/trac/experiments/phase6/final/B100/checkpoint.pt | 0.8594 | 0.8649 | 0.2173 | 0.1941 | 0.5970 | 0.5848 | 0.8773 | 0.8402 |
| TriLiteNet:tiny | 0.8840 | 0.8853 | 0.2058 | 0.2432 | 0.5956 | 0.6148 | 0.7240 | 0.7565 |
| TriLiteNet:small | 0.9084 | 0.9103 | 0.2270 | 0.2764 | 0.6062 | 0.6316 | 0.7525 | 0.7953 |
| TriLiteNet:base | 0.9224 | 0.9245 | 0.2386 | 0.2985 | 0.6120 | 0.6428 | 0.7711 | 0.8233 |
| TwinLiteNetPlus:nano | 0.8734 | 0.8742 | 0.1842 | 0.2355 | 0.5859 | 0.6122 | 0.6656 | 0.7031 |
| TwinLiteNetPlus:small | 0.9058 | 0.9065 | 0.2146 | 0.2939 | 0.6011 | 0.6417 | 0.7008 | 0.7585 |
| TwinLiteNetPlus:medium | 0.9202 | 0.9207 | 0.2322 | 0.3244 | 0.6098 | 0.6570 | 0.7244 | 0.7921 |
| TwinLiteNetPlus:large | 0.9285 | 0.9291 | 0.2440 | 0.3426 | 0.6156 | 0.6661 | 0.7453 | 0.8194 |
| TwinLiteNet | 0.9120 | 0.9127 | 0.2345 | 0.2883 | 0.6099 | 0.6376 | 0.7638 | 0.8106 |

## vs published (official GT / published protocol)

| model | metric | measured | published | delta | within +-1.0 |
|---|---|---:|---:|---:|---|
| TriLiteNet:tiny | da_mIoU | 88.53 | 88.5 | +0.03 | YES |
| TriLiteNet:tiny | lane_fg_iou | 24.32 | 24.2 | +0.12 | YES |
| TriLiteNet:small | da_mIoU | 91.03 | 90.5 | +0.53 | YES |
| TriLiteNet:small | lane_fg_iou | 27.64 | 27.6 | +0.04 | YES |
| TriLiteNet:base | da_mIoU | 92.45 | 92.0 | +0.45 | YES |
| TriLiteNet:base | lane_fg_iou | 29.85 | 29.8 | +0.05 | YES |
| TwinLiteNetPlus:nano | da_mIoU | 87.42 | 87.3 | +0.12 | YES |
| TwinLiteNetPlus:nano | lane_fg_iou | 23.55 | 23.3 | +0.25 | YES |
| TwinLiteNetPlus:nano | lane_line_acc | 70.31 | 70.2 | +0.11 | YES |
| TwinLiteNetPlus:small | da_mIoU | 90.65 | 90.6 | +0.05 | YES |
| TwinLiteNetPlus:small | lane_fg_iou | 29.39 | 29.3 | +0.09 | YES |
| TwinLiteNetPlus:small | lane_line_acc | 75.85 | 75.8 | +0.05 | YES |
| TwinLiteNetPlus:medium | da_mIoU | 92.07 | 92.0 | +0.07 | YES |
| TwinLiteNetPlus:medium | lane_fg_iou | 32.44 | 32.3 | +0.14 | YES |
| TwinLiteNetPlus:medium | lane_line_acc | 79.21 | 79.1 | +0.11 | YES |
| TwinLiteNetPlus:large | da_mIoU | 92.91 | 92.9 | +0.01 | YES |
| TwinLiteNetPlus:large | lane_fg_iou | 34.26 | 34.2 | +0.06 | YES |
| TwinLiteNetPlus:large | lane_line_acc | 81.94 | 81.9 | +0.04 | YES |

**18/18 published references reproduced within +-1.0.**

### Conformance ledger: our harness vs the number the authors published

| model | src | params mine/off | FLOPs mine/off (ratio) | mAP50 mine/off | DA mine/off | LaneIoU mine/off (ratio) | LaneAcc mine/off |
|---|---|---|---|---|---|---|---|
| TriLiteNet tiny | cited | 0.151 / 0.15 | 1.80 / 0.55 (**3.27x**) | 0.4953 / 49.6 | 0.8796 / - | 0.1952 / 24.20 (**0.81**) | 0.6790 / 76.5 |
| TriLiteNet small | cited | 0.592 / 0.59 | 6.60 / 1.99 (**3.32x**) | 0.6326 / 63.2 | 0.9053 / - | 0.2198 / 27.60 (**0.80**) | 0.7064 / 81.6 |
| TriLiteNet base | cited | 2.350 / 2.35 | 25.40 / 7.72 (**3.29x**) | 0.7235 / 72.3 | 0.9203 / - | 0.2371 / 29.80 (**0.80**) | 0.7301 / 85.6 |
| TLP nano | repo | 0.033 / 0.03 | 1.90 / 0.57 (**3.33x**) | - / - | 0.8634 / 87.3 | 0.1859 / 23.30 (**0.80**) | 0.6710 / 70.2 |
| TLP small | repo | 0.122 / 0.12 | 4.70 / 1.40 (**3.36x**) | - / - | 0.8986 / 90.6 | 0.2165 / 29.30 (**0.74**) | 0.7062 / 75.8 |
| TLP medium | repo | 0.479 / 0.48 | 15.40 / 4.63 (**3.33x**) | - / - | 0.9191 / 92.0 | 0.2297 / 32.30 (**0.71**) | 0.7175 / 79.1 |
| TLP large | repo | 1.944 / 1.94 | 58.60 / 17.58 (**3.33x**) | - / - | 0.9279 / 92.9 | 0.2450 / 34.20 (**0.72**) | 0.7448 / 81.9 |
| TwinLiteNet | repo | 0.440 / 0.44 | 14.10 / 3.90 (**3.62x**) | - / - | 0.9114 / 91.3 | 0.2281 / 31.10 (**0.73**) | 0.7203 / 77.8 |
| YOLOP | repo | 7.940 / 5.53 | 31.30 / 8.11 (**3.86x**) | 0.7657 / 76.5 | 0.9115 / 91.6 | 0.2247 / 26.50 (**0.85**) | 0.7896 / - |

**FLOPs ratio:** min 3.27, max 3.86, mean 3.41. Expected from the protocol difference alone: 2.0 (MACs -> FLOPs) x 1.6667 (384x640 -> 640x640) = **3.333**.

**LaneIoU ratio:** min 0.71, max 0.85, mean 0.77. Spread 0.14 - no single constant explains it, so this column cannot be repaired by a scale factor.


### Published reference table already vendored in-tree (`baselines/TwinLiteNetPlus/README.md`)

| Model | DA mIoU (%) | Lane Acc (%) | Lane IoU (%) | FLOPs | #Params |
|---|---|---|---|---|---|
| DeepLabV3+ | 90.9 | -- | 29.8 | 30.7G | 15.4M |
| SegForme | 92.3 | -- | 31.7 | 12.1G | 7.2M |
| R-CNNP | 90.2 | -- | 24.0 | -- | -- |
| YOLOP | 91.6 | -- | 26.5 | 8.11G | 5.53M |
| IALaneNet (ResNet-18) | 90.54 | -- | 30.39 | 89.83G | 17.05M |
| IALaneNet (ResNet-34) | 90.61 | -- | 30.46 | 139.46G | 27.16M |
| IALaneNet (ConvNeXt-tiny) | 91.29 | -- | 31.48 | 96.52G | 18.35M |
| IALaneNet (ConvNeXt-small) | 91.72 | -- | 32.53 | 200.07G | 39.97M |
| YOLOv8 (multi) | 84.2 | 81.7 | 24.3 | -- | -- |
| Sparse U-PDP | 91.5 | -- | 31.2 | -- | -- |
| TwinLiteNet | 91.3 | 77.8 | 31.1 | 3.9G | 0.44M |
| TwinLiteNet+ Nano | 87.3 | 70.2 | 23.3 | 0.57G | 0.03M |
| TwinLiteNet+ Small | 90.6 | 75.8 | 29.3 | 1.40G | 0.12M |
| TwinLiteNet+ Medium | 92.0 | 79.1 | 32.3 | 4.63G | 0.48M |
| TwinLiteNet+ Large | 92.9 | 81.9 | 34.2 | 17.58G | 1.94M |

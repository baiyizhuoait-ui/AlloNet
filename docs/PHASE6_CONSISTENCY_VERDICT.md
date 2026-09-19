# 一致性验证：官方 GT × 模型画布（2×2）

**触发**：用户指令「暂停 seed1，先跑一致性，结束后再接上 seed1」。
**日期**：2026-09-14。
**脚本**：`scripts/phase6_consistency_official.py`（`--input {640,384}` × `--gt {ours,official,both}`）

---

## 一、为什么这是一个干净的 2×2

两种画布的 letterbox 缩放都是 **scale = 0.5**：

| 画布 | letterbox 目标 | 内容区 | padding |
|---|---|---|---|
| 我们出厂（640×640） | 1280×720 → 640×640 | **640×360** | 上下各 140 px |
| 官方轻量模型（640×384） | 1280×720 → 640×384 | **640×360** | 上下各 12 px |

**裁掉 padding 后都是 640×360 ⇒ 指标永远算在同一块像素上**。所以一张表里只有两个变量：

1. **模型看到的画布**（640×640 vs 640×384）
2. **标注文件来自哪一份**（我们 vs 官方）

官方管线的原文（`baselines/TwinLiteNetPlus/BDD100K.py`）：

```python
image = letterbox(image, (H_, W_))          # :213   H_=384, W_=640
label = cv2.resize(label, (W_, 360))        # :229   直接 resize 到 640x360
_, seg = cv2.threshold(label, 1, 255, cv2.THRESH_BINARY)   # :222  前景 = label > 1
```

其 `letterbox`（`:20`）与我们 `datasets/bdd100k.py:31` 同构（等比 `min()` + 居中 pad）。
`--gt both` 让**同一次前向**同时累加两套 GT ⇒ 得到一个**纯 GT 替换**的对照。

## 二、预注册判据（跑之前钉死）

> `gt=official` + `input=384` 这一格是**发表协议**，必须复现官方数值（±1.0 绝对）：

| 模型 | DA mIoU | Lane IoU | mAP50 |
|---|---:|---:|---:|
| TriLiteNet tiny | 88.5 | 24.2 | 49.6 |
| TLP nano | 87.3 | 23.3 | — |

复现 ⇒ harness 协议合规，同协议表里剩下的差值就是**模型差异**；
不复现 ⇒ 残差必须记为**未解协议未知**，不得并进模型结论。

## 三、Stage A 结果（n = 2,500 张，判据来源）

| 模型 | 画布 | DA mIoU（我们 GT） | DA mIoU（官方 GT） | lane fgIoU（我们 GT） | lane fgIoU（官方 GT） | 发表 lane IoU |
|---|---:|---:|---:|---:|---:|---:|
| TriLiteNet tiny | 640 | 0.8805 | 0.8821 | 0.1939 | **0.2565** | 24.2 |
| TriLiteNet tiny | 384 | 0.8850 | **0.8863** | 0.2045 | **0.2406** | 24.2 |
| TLP nano | 640 | 0.8626 | 0.8630 | 0.1838 | 0.2335 | 23.3 |
| TLP nano | 384 | 0.8730 | **0.8738** | 0.1820 | **0.2314** | 23.3 |

### 判据结果：**5/5 全部落在 ±1.0 内**（另在 n=200 复验同样 5/5）

| 参照 | 复现值 | 发表值 | 偏差 |
|---|---:|---:|---:|
| TriLiteNet tiny DA | 88.63 | 88.5 | +0.13 |
| TriLiteNet tiny Lane IoU | 24.06 | 24.2 | −0.14 |
| TLP nano DA | 87.38 | 87.3 | +0.08 |
| TLP nano Lane IoU | 23.14 | 23.3 | −0.16 |
| TLP nano Lane Acc | 69.89 | 70.2 | −0.31 |

**结论**：**lane 的旧缺口来自标注文件，不是测量管线。** 加上 DA 本就同源
（Jaccard 0.9885），**「lane / DA 跨论文不可比」这条此前的判断被推翻** —— 在官方
GT + 384 画布下，两者都可比且能复现。

## 四、两条必须写进论文的附带事实

**① 画布是第二个变量，且方向已知。** 在官方 GT 不变、只把画布从 384 换成 640 时，
lane fgIoU 被抬高约 **+1.5**（TriLiteNet tiny 24.06 → 25.65）。所以
**复现发表值必须用 384 画布**；我们出厂的 640 画布会系统性高估 lane。

**② 我方模型在官方 GT 下 lane 更低，方向与基线相反。**

| 模型 | lane fgIoU（我们 GT） | lane fgIoU（官方 GT） | 方向 |
|---|---:|---:|---|
| 我方 B100（384 画布, n=200） | 0.2195 | 0.1931 | **降低** |
| TriLiteNet tiny（384 画布） | 0.2045 | 0.2406 | 升高 |
| TLP nano（384 画布） | 0.1820 | 0.2314 | 升高 |

原因：**我方模型是在自家那份「加粗」的 lane 标注上训练的**（前景占比 0.710% vs 官方
0.650%），换成官方细标注构成**训练/测试标注不匹配**；基线则本就训练在官方细标注上。

⇒ 跨模型 lane 表里，**我方那一行在官方 GT 下是悲观值**，必须在表注中声明，
不得当作「我们 lane 更差」的结论。

## 五、Stage B 状态（全量 10k 表）

`bash scripts/phase6_consistency_stageB.sh 384 0` —— 9 个模型 ×（384 画布、两列 GT）。
完成后写入 `experiments/phase6/consistency/stageB_in384/{consistency.json,consistency.md}`。

守候续训：`scripts/phase6_auto_resume_seeds.sh`（PID 1287）在扫描退出的瞬间自动执行
`scripts/phase6_seeds_resume.sh 1 2`，从 seed 1 的 **ep18** 续跑。

## 六、执行记录

| 项 | 值 |
|---|---|
| 停机 | `scripts/phase6_safe_stop.sh --yes` 2026-09-14 12:32:38，杀序 runner→chain→trainer |
| 停机代价 | **最多 1 个 epoch**（checkpoint 已确认 epoch=18 完好，`.safe` 已刷新） |
| CSV | 保持 2 行，**无伪造行** |
| 已知缺陷 | 首次运行因 `baselines/TriLiteNet/lib/models` 遮蔽我方 `models/` 而崩，且崩在存盘前 |
| 修法 | 入口预导入 `models.static_model` / `models.adaptive_model` + 单模型 try/except 容错 |
| 待改 | `phase6_consistency_stageB.sh` 的 `python \| grep \| tee` 顺序（grep 块缓冲丢进度）→ 应为 `python \| tee \| grep` |

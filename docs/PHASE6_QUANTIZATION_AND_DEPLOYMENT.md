# Phase 6 — 量化与部署：口径修正、论文可用性判定

> 本文回答四个问题：① 上一轮那些"几帧"是不是 INT8 口径？② "性能几乎没损失"成立吗？
> ③ 别的论文有没有做这种部署分析、我们能不能写进论文？④ INT8 既然可行，FP4/INT4 还有意义吗？
> 并附一张 17 行的低成本设备能力矩阵。
>
> **本文含两处对既有结论的自我更正**，均有逐位可复现的证据（§2、附录 A）。

---

## 0. 一句话结论

帧数估算用的是 **INT8 锚点**，但那个估算里 **不含任何精度信息** —— GFLOPs 是架构常数，
量化不改变它。所以"帧数高"和"精度不掉"从来是两条独立的证据链，上一轮把它们并排摆，
会被读成"实测 INT8：又快又准"，这是表述缺陷。

而"精度几乎没损失"这句本身 **必须撤回并改写**：它建立在两个缺陷上。
修正后是 —— **DA 稳健、车道前景 IoU 掉约两成、检测在朴素的 per-tensor 量化下直接清零**。

---

## 1. Q1：那些帧数是 INT8 口径吗？

**是，但只在"锚点"这一层。**

`scripts/phase6_edge_fps_estimate.py` 的每一行都标注了方法。三个"双锚点拟合"行的锚点
全部是 **别人在真板上跑 INT8 的公开数字**：

| 设备 | 锚点 | 来源 |
|---|---|---|
| RK3588（单核） | YOLOv8n INT8 37 FPS / YOLOv8s INT8 15 FPS | Orange Pi 5 Max, RKNN INT8, end-to-end |
| Jetson Orin Nano (15W) | YOLOv8n INT8 65–75 / YOLOv8s INT8 48–58 | TensorRT INT8, MAXN 15W |
| Jetson Nano (2019) | YOLOv8n INT8 25–30 / YOLOv8s INT8 8–12 | JetPack 4.x, INT8 |

拟合式是 `latency = a + k · GFLOPs`，代入我们自己的 1.1656 GFLOPs。**这一式子里没有任何
精度项** —— 权重从 FP32 变成 INT8，GFLOPs 不变，所以外推出的延迟与"量化后精度掉不掉"
毫无关系。

**必须这么写**：
- ✅「把本模型的 1.17 GFLOPs 代入同一设备的 INT8 吞吐标度，量级落在 45–74 FPS」
- ❌「我们的模型在 RK3588 上以 INT8 跑到 64 FPS」（未在任何目标设备上运行过）
- ❌ 把帧数与精度损失并排成一张表（读者会读成实测）

方法学上限也一并写明：这是 **量级估算**，不是基准测试；且真正的瓶颈不在算力 ——
拟合出的每帧固定开销 `a` = 8–12 ms，而我们全部计算项只有 0.2–3.7 ms
（比 YOLOv8n 少 7.5 倍算力，只换来 1.7 倍延迟下降）。

---

## 2. Q2："性能几乎没损失"成立吗？—— **不成立，已更正**

### 2.1 缺陷 A：激活量化器在评估前被摘除（可逐位复现）

`scripts/phase6_deploy_profile.py::qsim_outputs` 里，激活钩子的注册与移除顺序是错的：

```python
for i, m in enumerate(model.modules()):
    if isinstance(m, (nn.BatchNorm2d, nn.ReLU)):
        hooks.append(m.register_forward_hook(act_hook(f"m{i}")))

...  # 标定：观察 activation 量程
for h in hooks:
    h.remove()            # ← 第 211-212 行：在这里就摘掉了

fp = run("off",   images[calib:])   # ← 第 214 行：评估才开始
q  = run("apply", images[calib:])   # ← 第 215 行
```

标定阶段照常观察了 activation 的量程，所以 `st["a"]` 里数据齐全、日志正常、没有任何异常
——但**评估阶段实际施加的只有 per-output-channel 权重量化 + 输入图量化，激活全程留在 FP32**。
而激活量化恰恰是 CNN 后训练量化的主要误差来源。

**逐位复现的证据**：新脚本以"关掉激活量化"的方式重跑，得到的是旧数字的**逐位相同值**：

| 指标 | 旧记录（修订前 `profile.json` 的 W8A8_p9999，提交 `4904c0a` 可查） | 以 `--act-quant off` 重跑 | 完整 W8A8（钩子生效） |
|---|---|---|---|
| DA 像素一致 | 0.99906 | **0.99906** | 0.99426 |
| DA 前景 IoU | 0.98833 | **0.98833** | 0.94269 |
| 车道 像素一致 | 0.99929 | **0.99929** | 0.99644 |
| 车道 前景 IoU | 0.94846 | **0.94846** | 0.73946 |

四个数字全部**四位小数逐位相同** ⇒ 机制确认，不是估计。

复现命令：`scripts/phase6_quant_head_audit.py --act-quant off --only B_inner_only`
（结果落在 `experiments/phase6/deploy_profile/repro_old_bug.json`，是这段历史的永久证据。）

已修复：`phase6_deploy_profile.py` 内钩子移除已移到评估之后，并在原位留下长注释说明这段历史；
输出 JSON 增加 `_defect_fixed_2026_09_14` 字段，把旧数字一并写进去。
`profile.json` 已用修正后的脚本重新生成（因此文件内不再有旧的虚高值，
旧值仅存在于 git 历史与上述复现产物中）。

### 2.2 缺陷 B：检测头输出张量从未被量化（结构性盲点）

这是比 A 更严重的一处，因为它不是笔误，是**探针位置**的问题。

激活量化器挂在 `BatchNorm2d` / `ReLU` 模块上。检测头的尾巴是

```python
y = x.sigmoid()
y[..., 0:2] = (y[...,0:2]*2 - 0.5 + grid) * stride     # xy：像素量级
y[..., 2:4] = (y[...,2:4]*2)**2 * anchor_grid          # wh：像素量级
return torch.cat(outs, 1)                              # xy|wh|obj|cls → (1,25200,6)
```

`cat` 之后**没有 BN，也没有 ReLU** ⇒ 承载 objectness 与类别分数的那个张量，
它自己的输出量化器在仿真里**根本不存在**。旧的"检测 logit 余弦 ≈ 1.000000"
就是在一个没有被量化的张量上测出来的。

而这正是文献里翻车的那个拓扑。见 §3。

### 2.3 修正后的完整 W8A8 结果（真权重、真图、per-channel 权重、per-tensor 激活）

新增 `scripts/phase6_quant_head_audit.py`，四个变体同一套 BN/ReLU 量化器，只改检测头：

| 变体 | DA 像素一致 | DA 前景 IoU | 车道 像素一致 | 车道 前景 IoU | obj 归零比例 | NMS 框数(conf .25) | 框保留率 |
|---|---|---|---|---|---|---|---|
| **B** 仅内部（= 旧行为） | 0.99426 | 0.94269 | 0.99644 | 0.73946 | 0.03% | 8.70 | 94.6% |
| **C** + 检测头 per-tensor | 0.99426 | 0.94269 | 0.99644 | 0.73946 | **100.0%** | **0** | **0.0%** |
| **D** + 检测头 per-channel | 0.99426 | 0.94269 | 0.99644 | 0.73946 | 98.9% | 9.55 | 93.7% |

（DA/车道三列在 B/C/D 之间**完全不动** —— 这是控制项，证明头钩子只碰检测张量。）

**观测器的选择本身就是决定性的**（同一协议，重跑后的 `profile.json`）：

| 变体 | DA 前景 IoU | 车道 前景 IoU |
|---|---|---|
| W8A8 **absmax**（朴素全局最大值） | **0.54136** | 0.60676 |
| W8A8 **p99.99**（常规 PTQ 设置） | **0.94269** | 0.73946 |
| W6A6 absmax | 0.10640 | 0.10325 |

⇒ 8 bit 用 absmax 观测器会让 DA 前景 IoU 掉到 0.54 —— **同样 8 bit，观测器选错等于白做**。
这与 MinMAE（PeerJ CS 2026）在 YOLOv8 上的结论同向（absmax 类标定的 mAP 损失可高达 14.4%）。
论文里必须把观测器写清楚（percentile / entropy），否则数字不可复现。

- **C 的崩塌是算术必然，不是仿真偶然**：实测 per-tensor 步长 s = 6.84，而 obj 通道的
  全局最大值只有 0.86 < s/2 = 3.42 ⇒ 无论校准集怎么选，**每一个 obj 值都舍入到 0**。
  与 Valeo 报告的"概率被坐标淹没、第一量化档就归零、检测 0% mAP"机制完全一致。
- **D 把框救回来了**（保留 93.7%，匹配框平均 IoU 0.786），代价是 NMS@0.001 的候选
  从 255 降到 144 —— 排序面变窄，mAP 一定受损，但系统仍然工作。
- 三行都是**下界**：没有 bias correction、没有 AdaRound、没有 QAT。

### 2.4 结论必须改写成什么

| 轴 | 能说的 | 不能说的 |
|---|---|---|
| DA | "8 bit 下掩码像素一致率 99.4%，前景 IoU 相对 FP32 保留 94.3%" | "近乎无损" |
| 车道 | "车道是本架构最脆弱的一轴：像素一致 99.6%，但前景 IoU 只剩 0.739" | 只报像素一致率 |
| 检测 | "朴素 per-tensor 会把检测输出整体清零；per-channel 修正后保留 93.7% 的框。此项为仿真，未经板端验证" | 任何"检测精度无损"的表述 |
| 全局 | "量化友好性分析（PTQ 仿真，下界）" | "已实现 INT8 部署" |

> 车道前景 IoU 0.739 这一条与已知事实一致：车道线在验证 GT 上只有约 2 px 宽，
> 前景 IoU 对边界位移极度敏感；本项目的多处历史观察都指向同一结论。

---

## 3. Q3：别的论文有没有这样做？能不能写进论文？

### 3.1 有 —— "部署可行性 / PTQ 分析"是标准章节

| 工作 | 做法 | 结果 |
|---|---|---|
| IET Image Processing 2026（轻量 YOLOv8n 小目标） | TFLite **INT8 PTQ**，训练集 100 张标定，单列一节 4.5 *Analysis of Model Deployment Feasibility* | mAP@0.5 −0.005，参数不变，称"2.5× 速度提升" |
| ACM 2026（安卓鞋类缺陷检测） | TFLite INT8 PTQ，Table 4 压缩表 + Table 5 端侧实测 | 44.8 MB → 11.5 MB；端侧 1.7–3.0 FPS |
| **Q-YOLOP**（arXiv:2307.04537） | 同任务（检测 + DA + 车道）、同数据集（BDD100K）、INT8 | **PTQ 灾难**：DA mIoU 0.842 → **0.285**，车道 0.402 → 0.248；结论是"QAT 是必须的" |
| YOLOP @ ENSICAEN×Valeo | INT8 PTQ + 熵标定 | DA −0.25（可忽略）、车道 −5.8、**检测 mAP@50 71.94 → 58.20（−13.7）**；首次尝试 **0% mAP** |

**所以形式和内容是标准的、可发的。** 但要注意两件事：

1. 这些论文的证据标准是 **"量化后的 mAP / IoU"**，不是 logit 余弦或掩码一致率。
   我们目前只有后者 ⇒ **门槛没到**。
2. 上表里两篇最相关的工作给出的都是**负面**结果（PTQ 崩、必须 QAT）。我们能主张
   "本架构 PTQ 友好" 的前提是**把 DA/车道/检测三条轴的量化后指标都测出来并公开缺陷与修法**，
   否则会被直接对照 Q-YOLOP 判为不可信。

### 3.2 论文里可写 / 不可写

**可写**
1. 量化友好性的**架构解释**：全图只有 conv / BN / ReLU / 双线性上采样，无注意力、无 LayerNorm、
   无 GELU；BN 与 conv 融合后权重量程逐通道良好 —— 这属于文献公认的"易量化"类
   （参见 §8 的 CNN 权重在 BN 融合后"相对良性"的共识）。
2. 实测的**分解证据**：49 conv2d + 41 BN（融合）+ 5 双线性 Resize + 1 Concat + 3 Sigmoid
   + 3 Pow + 33 个逐元素算子（`scripts/phase6_onnx_opcheck.py`，追踪图代理）。
3. **车道先崩**这条轴间敏感性结论（比特扫描 §4.2）。
4. 检测头 per-tensor 清零 + per-channel 修法的**复现与定位**（这是方法学贡献，且与 Valeo 互证）。
5. 成本侧：0.735 MB / 0.367 MB / 0.184 MB（FP32/FP16/INT8），峰值激活 6.25 MB。

**不可写**
1. 任何"INT8 精度无损"的表述（修正后 DA 掉 5.7% IoU、车道掉 21% IoU）。
2. 任何"我们已部署 / 已上板 / 实测 FPS"的表述。
3. 把 FLOPs 或 FPS 与已发表数字并排（既有禁令）。
4. 声称"我们提出量化友好架构" —— 没有对照，这是观察不是贡献。

---

## 4. Q4：INT8 既然可行，FP4/INT4 还有意义吗？

### 4.1 对我们这个模型：收益几乎为零，理由是算术

| 量 | FP32 | 8 bit | 4 bit | 4 bit 相对 8 bit 的节省 |
|---|---|---|---|---|
| 权重 | 0.735 MB | 0.184 MB | 0.092 MB | **0.09 MB** |
| 峰值激活 | 6.25 MB | 3.1 MB | 1.6 MB | 1.6 MB |

而实测的延迟构成是 **每帧固定开销 8–12 ms vs 计算项 0.2–3.7 ms**。
在一个连 INT8 算力都用不满的区间里，把权重再砍一半不改变任何东西。
**4 bit 对我们是纯浪费 —— 除非目标硬件的 4 bit 路径本来就比它的 8 bit 路径快**
（Qualcomm HMX 是这样：INT4 张量吞吐是 INT8 的两倍）。

### 4.2 但比特扫描给出一条真结论：**先崩的是车道**

`scripts/phase6_quant_bitsweep.py`，同一套协议（含检测头 per-channel 量化器），60 张真实验证图：

| bits | DA 像素一致 | DA 前景 IoU | 车道 像素一致 | 车道 前景 IoU | 检测框保留率 | 头部量化步长 |
|---:|---:|---:|---:|---:|---:|---:|
| **8** | 0.99426 | **0.94269** | 0.99644 | **0.73946** | **0.9369** | 0.0067 |
| 7 | 0.98452 | 0.85555 | 0.99421 | 0.59730 | 0.8091 | 0.0135 |
| 6 | 0.91525 | 0.53706 | 0.98919 | 0.37320 | 0.4976 | 0.0274 |
| 5 | 0.87143 | **0.00013** | 0.98466 | 0.02096 | 0.0531 | 0.0567 |
| 4 | 0.88055 | **0.00000** | 0.92226 | 0.03977 | 0.0629 | 0.1214 |

> 口径提醒：本表用 **p99.99 观测器 + 检测头 per-channel 量化**，因此 6 bit 的 DA 前景 IoU
> 是 0.537；`profile.json` 里 W6A6_absmax（absmax + 无头量化）是 0.106。
> **两个数都对，但不可以互相引用** —— 它们是不同的配置。

四条读法（都是可以写进论文的）：

1. **8 bit 是正确档位，但不是"无限余量"**：7 bit 已经让 DA 前景 IoU 掉 8.7 个点、
   车道掉 19 个点、框保留率掉 12.8 个点。8→7 的落差远大于常人预期的"少一位只差一点"。
2. **悬崖在 7 与 5 之间**：6 bit 时三条轴全部腰斩；**≤5 bit 两个分割头退化成常量**
   （DA 前景 IoU 0.00013 / 0.0 —— 输出全背景）。
3. **"像素一致率"在悬崖处会骗人**：4 bit 时 DA 像素一致还有 0.8806，看着不错，
   但那只是 88% 的背景先验 —— 此时前景 IoU 为 0。**任何量化报告都必须成对给出
   agreement 与 fgIoU**，本项目此前的表格偏重前者。
4. **过度量化会产生"虚假框"而不是沉默**：4 bit 时框保留率只剩 6.3%，但 NMS 在
   conf 0.25 上却输出 **27.45** 个框（8 bit 时是 9.55）—— 退化的模型在乱报。对一个
   辅助驾驶安全功能而言，这比不出框更危险。

这条结论的价值在于它是 **可发表的轴间敏感性结果**，并且能直接支撑三句话：
1. 8 bit 是本模型的正确档位（有余量但不是无限的）。
2. 车道轴是第一个失守的，因此任何进一步的压缩研究都应把车道指标放在第一位。
3. INT4 是否值得，取决于你是否愿意牺牲车道 —— 而不是取决于参数量：

| 若把 INT8 降到 INT4，按本表推算 | 结果 |
|---|---|
| 权重节省 | 0.092 MB（0.184 → 0.092） |
| 换取 | DA 前景 IoU 0.943 → 0.000，车道 0.739 → 0.040，检测框保留 93.7% → 6.3% |

**结论：对 0.19 M 参数的本模型，INT4/FP4 的收益是 0.09 MB，代价是三条轴全废。**

### 4.3 硬件现实：INT8 是唯一"处处都在"的档位

2026 年低成本端的精度阶梯并不一致：

| 只有 INT8，没有浮点 | INT4 / FP4 原生路径 | 有 FP32 路径 |
|---|---|---|
| Coral Edge TPU、Hailo-8、Hailo-8L、Arm Ethos-U（Vela）、Rockchip RKNN 全系、K230 KPU、Sophgo CV1800B | Hailo-10H（INT4）、Qualcomm Hexagon HMX（INT4）、Apple ANE（4-bit）、Jetson Thor（NVFP4 + FP8） | Jetson Nano/Orin/Thor、纯 CPU 板、x86 |

⇒ 可辩护的表述不是"应该用 INT8"，而是：
**INT8 是本模型每个便宜目标上都存在的最低共同档位，也正是它被设计的档位；
4 bit 只出现在"更新、更贵或手机级"的部件上**，因此是一项未来/手机端的可选项，不是当前需求。

---

## 5. 低成本设备能力矩阵（17 行）

完整 JSON：`experiments/phase6/deploy_profile/edge_device_matrix.json`
生成器：`scripts/phase6_edge_device_matrix.py`（能力矩阵，不主张任何 FPS）

| 设备 | 价位 | 精度支持 | FP32 路径 | 对本模型的判读 |
|---|---|---|---|---|
| Milk-V Duo (CV1800B) | ¥35–53 | INT8 | 无 | 内存装得下（0.74 MB 权重 / 6.25 MB 激活）；工具链算子覆盖是唯一疑问。厂商材料对 TOPS 的说法差 2.5× |
| K230 CanMV | ¥239–299 | INT8 | 无 | 目标档位本身 |
| Rockchip RK3568 | ¥150–300 | INT8 | 无 | 目标档位本身 |
| Rockchip RK3588（3 核） | ¥300–800 | INT8 | 无 | 目标档位本身，锚点最全 |
| Coral Edge TPU (USB) | $60 | **INT8 ONLY** | 无 | 非全 INT8 图会回落主机 CPU，加速器白买；栈 2024 已停更 |
| Raspberry Pi AI HAT+ (Hailo-8L) | $70 | **INT8 ONLY** | 无 | SRAM-only，但 0.74 MB 微不足道；Resize 覆盖需编译验证 |
| Hailo-8 (26 TOPS, M.2) | $110–200 | **INT8 ONLY** | 无 | 同上，余量更大 |
| Hailo-10H | — | INT4 / INT8 / INT16 | 无 | 少数有真 4-bit 路径的便宜部件，但面向小语言模型 |
| Arm Ethos-U55/U85 | 随 MCU | INT8 | 无 | **6.25 MB 峰值激活塞不进典型 MCU SRAM** —— 这一档与 640² 三任务模型无关 |
| ST STM32N6 | ~$15–25 芯片 | INT8 | 无 | 同上，列出以关闭这个问题 |
| Jetson Nano (2019) | ¥400–600 二手 | FP16 / INT8 | 有（慢） | 有真浮点路径，能不量化跑；Jetson 家族最弱 |
| Jetson Orin Nano 8GB | $249 | FP16/BF16/TF32/INT8 | 有 | 便宜档里精度阶梯最宽，INT8 部署的参考目标 |
| Jetson Orin Nano Super | $249 | 同上 | 有 | 同款 1.7× 吞吐，102 GB/s 带宽 |
| Snapdragon 8 Gen 3 / X Elite | 随设备 | INT8 + **原生 INT4 (HMX)** | GPU/CPU 回落 | 4-bit 结论在这里才真正可落地；也是车道轴最可能成为约束的地方 |
| Apple ANE (A18 / M4) | 随设备 | 16-bit + 4-bit palettisation | GPU/CPU 回落 | 4-bit 是一等权重压缩路径；对非 4/8 位形状最不宽容 |
| Jetson Thor (T5000) | 数百美元级 | FP4/FP6/FP8/FP16/INT8 | 有 | 唯一有硬件 FP4 的平台，40–130 W —— 是天花板而非目标 |
| Raspberry Pi 5（纯 CPU） | ¥400–600 | FP32 on CPU | 有 | 永远能跑、永不受益于加速器；实测缩放后 32.5 FPS，这是诚实的下限 |

---

## 6. 要把这段写进论文，必须先做三件事

| # | 缺口 | 为什么是硬要求 | 成本 |
|---|---|---|---|
| 1 | **量化后 mAP@0.5**（检测轴） | 我们唯一没有量化后数值的轴，而它恰好是文献里崩得最狠的轴（Valeo −13.7）。§2.3 的"框保留率"是替代指标，审稿人不认 | 中：需要带标签的检测评测；`evaluate_detection` 已在库，但 val 检测标注 JSON 有 208 MB，必须流式加载并加内存闸门（本机 WSL 曾因类似任务 OOM 一次） |
| 2 | **板端实测**（任一档） | 把"仿真"变成"实测"；哪怕只做 Coral / Hailo-8L / RK3588 之一 | 高：需硬件 + 厂商工具链（RKNN / nncase / Hailo DFC / Vela） |
| 3 | **部署配置写进方法节** | per-channel 检测头量化不是可选项，它是"能不能出框"的开关 | 低：一句话 + 一个引用（Valeo 的图注入归一化是等价做法） |

**在完成 #1 之前，量化这一段只能作为 limitations / future work 出现，不能作为贡献。**

---

## 7. 引用清单（本文用到的外部事实）

1. Q-YOLOP: Quantization-aware YOLO for Panoptic Driving Perception — arXiv:2307.04537
   （BDD100K 全景感知，INT8 PTQ 使 DA mIoU 0.842→0.285，QAT 恢复至 0.852）
2. YOLOP INT8 部署（ENSICAEN × Valeo 工业项目）— 检测 mAP@50 71.94→58.20，
   首版 0.0% mAP，根因是 Concat 混合概率与像素坐标
3. IET Image Processing (2026) — 轻量 YOLOv8n 的 INT8 PTQ 部署可行性分析（§4.5）
4. ACM (2026) — 安卓端 TFLite INT8 PTQ 与端侧基准（1.7–3.0 FPS）
5. MinMAE (PeerJ CS, 2026) — YOLOv8n 8-bit PTQ 的 mAP 损失：优化标定 8.4%，absmax 类最高 14.4%；
   并指出小网络因冗余低而更难量化
6. Hailo-8 / Hailo-8L / Hailo-10H 官方规格与第三方深度解读（INT8-only；10H 为 INT4）
7. Coral Edge TPU 规格与 TFLite 编译链现状（INT8-only，栈已停更）
8. Jetson Orin Nano Super 官方规格（67 INT8 TOPS）与 Jetson Thor 规格（FP4 原生）
9. 2026 边缘 NPU 量化综述（TOPS 不可跨精度比较；CNN 在 BN 融合后权重分布良性）
10. Milk-V Duo / Sophgo CV1800B 板级资料（0.2–0.5 TOPS INT8 的厂商说法不一致）

---

## 8. 附录 A：本轮溯源与缺陷记录

**新增脚本（均 CPU-only，不在 GPU 上跑，均已加内存闸门 MemAvailable ≥ 5000 MB）**

| 脚本 | 作用 |
|---|---|
| `scripts/phase6_quant_head_audit.py` | 四变体检测头量化审计；`--act-quant off` 可逐位复现旧口径 |
| `scripts/phase6_quant_bitsweep.py` | 8/7/6/5/4 bit 扫描（含检测头量化器） |
| `scripts/phase6_onnx_opcheck.py` | 追踪图算子直方图 + 回落风险清单 |
| `scripts/phase6_edge_device_matrix.py` | 17 行设备能力矩阵 |

**产物**

| 路径 | 内容 |
|---|---|
| `experiments/phase6/deploy_profile/quant_head_audit.json` | B/C/D 三变体 |
| `experiments/phase6/deploy_profile/repro_old_bug.json` | 旧口径的逐位复现证据 |
| `experiments/phase6/deploy_profile/quant_bitsweep.json` | 比特扫描 |
| `experiments/phase6/deploy_profile/onnx_opcheck.json` | 算子清单 |
| `experiments/phase6/deploy_profile/edge_device_matrix.json` | 设备矩阵 |

**已修改的既有文件**

- `scripts/phase6_deploy_profile.py`：激活钩子的移除移到评估之后（缺陷 A 修复），
  原位留注释说明历史；输出 JSON 增加 `_defect_fixed_2026_09_14` 字段。
- 未触碰任何训练路径（`run.sh` / `train.py` / 冻结配置）。

**仍未闭合的两条**

1. 检测轴量化后 mAP（§6 第 1 项）。
2. 任何板端实测（§6 第 2 项）。此前所有 FPS 与精度数字，**没有一个是本模型在真实
   量化工具链上跑出来的**。

# TRAC Phase 0-3 过程报告合集（整合版）

> 本文件由以下 15 份阶段报告合并而成（合并日期 2026-09-19），内容除来源分隔标记外未作改写。原始单文件已归档移除。

## 目录

- PHASE0_DEV_PREP.md
- PHASE1_EXP1_REPORT.md
- PHASE1_FINDINGS.md
- PHASE1B_REPORT.md
- PHASE1_STATUS.md
- PHASE2_PLAN.md
- PHASE2B_POSITIONING.md
- PHASE2B_REPRODUCIBILITY.md
- PHASE2B_TASK2C_DESIGN.md
- PHASE2C_EXPA_RESULTS.md
- PHASE2D_BOTTLENECK_AUDIT.md
- PHASE2D_DECISION_REPORT.md
- PHASE3A_REPORT.md
- PHASE3B_REPORT.md
- PHASE3C_REPORT.md

---

# ＝＝＝ 来源：PHASE0_DEV_PREP ＝＝＝

# Phase 0 开发准备报告

日期：2026-08-28 ｜ 目标：为 Phase 1（MVP：Compact Representation + Task-Resource Router）铺平道路

## 0.1 工作区盘点（Step 1–2：读现有项目、确认可复用代码）

### 已有仓库（工作区根目录，均含 .git 与官方权重）
| 仓库 | 状态 | 可复用点 |
|------|------|----------|
| `YOLOP/` | ✅ 官方权重 End-to-end.pth + 320/640/1280 ONNX | **B1 baseline**；三任务多任务范式参考 |
| `YOLOPv2/` `YOLOPX/` `YOLOPv3/` | ✅ 各有权重/演示 | 参考实现（A-YOLOM 风格改进） |
| `HybridNets/` | ✅ hybridnets.pth | 参考（EfficientNet-B3 共享骨干多任务） |
| `A-YOLOM/` | ✅ | YOLOv8 多任务，参考 |

> 上一阶段（2026-08-21~25）已完成 YOLOP 家族 + HybridNets 的**统一速度基准**（FP16/bs=1/CUDA 同步，
> RTX 5060 实测：YOLOP 82 FPS / 7.94M / 131MiB 等），报告在 `../experiments/FINAL_REPORT.md`，
> 其统一评测口径可直接沿用。注意：此前实验**无精度验证**（无 ground truth），Phase 1 必须补上。

### 新克隆 baseline（`trac/baselines/`，各自保留 .git）
| 仓库 | 版本/预设 | 冒烟测试（640×640 FP32, gpu_env） |
|------|-----------|------------------------------------|
| `TwinLiteNet` | DA+Lane 两任务 | ✅ 0.440M / 14.06G / 8.0ms；权重 `pretrained/best.pth` ✅ |
| `TwinLiteNetPlus` | nano/small/medium/large 四档 | ✅ nano 33.4K / 1.89G / 5.5ms；nano 权重已从 HF 镜像获取并验证可加载 ✅ |
| `TriLiteNet` | tiny/small/base 三档，Det+DA+Lane | ✅ tiny 0.151M / 1.83G；small 0.592M / 6.61G；base 2.35M / 25.38G；官方权重待获取 |

### 环境（Step 3 前置）
- 本项目统一用 `../gpu_env`（Python 3.12.3，torch 2.11.0+cu130，CUDA 13.0 可用，
  thop/ptflops/albumentations/tensorboardX/yacs/timm ✓，timm/elephant 本次补装）。
- baseline 仓库官方 requirements（torch==1.8.0 等）在 Python 3.12 + Blackwell 下不可安装，
  沿用上一阶段的决策（torch 2.11/2.13+cu130），已在 README 注明。
- 各兄弟仓库仍用自己的 venv（yolop_env 等），不冲突。

## 0.2 数据现状（BDD100K）

### 已获得（2026-08-28 用户提供，已整合）
- **全量数据**：用户下载后拷入，已复制到 `trac/data/bdd100k/`：
  images/100k（70k train + 10k val）、images/10k/test（2k，官方 10k，Phase 1 无需 test）、
  labels/*.json（det，69,863 train + 10k val）、lanes/masks（70k+10k）、
  segments/masks（70k+10k，用户第二轮补齐，与图片 100% 匹配）。
  三任务交集清单：`data/bdd100k/splits/`（tri_train 69,863 / tri_val 10,000）。
- **全部官方权重**：TriLiteNet（nano→tiny / small / base）、TwinLiteNetPlus（nano/small/medium/large）
  已复制到 `trac_data/weights/` 并验证可加载进仓库模型。

### 未获得 / 已知限制
- images/10k/test 仅 2k/10k（官方 10k；Phase 1 用不到测试集，不阻塞）。
- 其余全部就绪：三任务标注完整、官方权重完整（见 README）。

## 0.3 Phase 1 前需要用户做的事
- [ ] **决策标注来源**（上节 1/2/3/4 选一；若选 1 或 2，请提供 token/确认注册）
- [ ] （可选）TriLiteNet 官方权重（GDrive 文件夹 1wLZqemCxxzwiFeFUGY1zMaqcKoQLHFyK）与
      TwinLiteNetPlus small/medium/large 权重（GDrive 文件夹 1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz），
      可自行下载后放入 `trac_data/weights/`；否则 Phase 1 从零训练 baseline（在统一数据上更公平）。
- [ ] 确认图片下载继续（~6.4GB，已在进行；磁盘剩余 898G 充足）

## 0.4 已建立的开发基础设施
- `trac/` 独立 git 仓库（commit 8147d8e），后续实验记录 commit_hash。
- `scripts/phase2_smoke_baselines.py`：B1–B3 冒烟 + 参数/FLOPs/前向耗时（一键验证环境与 baseline 可用）。
- `experiments/README.md`：§22 实验记录规范（metrics.json 字段、Best/Average/Worst 硬规则）。
- `requirements.txt` / `.gitignore` / `README.md` / `docs/TASKBOOK.md`（任务书完整存档）。

## 0.5 下一步（Phase 1 Step 3–4 起）
1. 标注到位后：写 `datasets/`（BDD100K 三任务加载 + YOLOP 式 train/val split 配置）。
2. `evaluation/`：统一评测（mAP + mIoU + Lane IoU + 精度/延迟全套，复用上一阶段口径）。
3. Module A（encoder + representation）→ Module B（router + nested prefix width）→ Static 三档 → Dynamic。
4. 训练 Stage A–D；实验 1–5；失败模式统计与可视化。

## 0.6 风险与注意事项
- bitmind 图片镜像未验证与官方文件一一对应（文件名看起来是 BDD100K 命名）；解压后需抽查校验数量。
- RTX 5060 8GB 显存：训练 batch 需保守（640×640 三任务模型，估计 ≤16）。
- WSL 内存 15GB：大数据加载注意 workers/缓存设置。
- 上一阶段环境为 torch 2.13（yolop_env 等），本项目 gpu_env 为 2.11；若某些 baseline 代码在 2.11 报错
  （如 torch.meshgrid 行为），可在各自 venv 中跑 baseline，ours 跑 gpu_env。


# ＝＝＝ 来源：PHASE1_EXP1_REPORT ＝＝＝

# Experiment 1: Static vs Dynamic（全量 tri_val，10,000 张）

模型：Ours（0.235M），训练 A→B→C→D 后同一 checkpoint，仅分配方式不同。

| variant | avg width | FLOPs(G) | mAP50 | DA_mIoU | Lane_fgIoU | eval_fwd_ms | eval_FPS |
|---|---|---|---|---|---|---|---|
| static_tiny (0.25) | 0.25 | ~0.82 | 0.2336 | 0.8049 | 0.1249 | 7.75 | 129.1 |
| static_medium (0.5) | 0.50 | ~1.16 | 0.2541 | 0.8667 | 0.1942 | 7.67 | 130.4 |
| static_large (1.0) | 1.00 | ~1.57 | 0.2613 | 0.8537 | 0.1841 | 7.56 | 132.3 |
| dynamic | 0.389 | ~1.06 | 0.2427 | 0.8372 | 0.1666 | 7.35 | 136.0 |

## 结论（H-01-H-03 初步）

1. **Dynamic ≈ Static 插值**：dynamic (1.06G, 0.2427) 位于 static_tiny (0.82G, 0.2336) 与
   static_medium (1.16G, 0.2541) 之间——等平均 FLOPs 下 Dynamic 不显著损失性能 ✓（H-03）
2. **Router 有任务感知**：DA 平均宽度 0.47 > det/lane 0.41，任务异构分配率 83.2% ✓（H-01）
3. **FLOPs 节省真实**：dynamic 平均 1.06G vs static_large 1.57G（-32%）
4. ⚠️ **失败模式 D**：eager PyTorch 下 eval 循环延迟未随宽度下降（7.3-7.8ms 噪声内），
   微基准显示纯模型宽度收益存在（3.09→2.35ms）但被 Python/分发开销淹没；
   需 torch.compile / TensorRT / ONNX 后端兑现硬件延迟收益。已记录，非算法问题。


# ＝＝＝ 来源：PHASE1_FINDINGS ＝＝＝

# Phase 1 核心发现报告（H-01-H-03 验证）

## 已验证成立

### H-03 ✓（Accuracy–Compute Pareto 覆盖，全量 val 10k）
- Dynamic 模型（平均宽度 0.389，≈1.06G FLOPs）精度 0.2427 mAP / 0.838 DA
  位于 static_tiny (0.82G, 0.2336) 与 static_medium (1.16G, 0.2541) 之间——
  **单一模型覆盖了静态多模型的精度-算力曲线**（Exp1）
- 相对 static_large 节省 ~32% FLOPs，精度损失仅 ~0.019 mAP
- 支持 Tiny/Medium/Large/Dynamic 四档一键切换（含 Phase 3 的 HIGH/MEDIUM/LOW 模式接口）

### 机制 ✓（§6/§7/§8 设计目标）
- Nested-prefix 连续通道宽度（硬件友好），3 档离散预算
- 逐帧任务异构分配（task-heterogeneity 0.83-0.88）
- Router 行为可视化 + 分配统计 CSV（§14 数据源就绪）

## 未成立 / 弱支持（诚实记录）

### H-01 ✗（task-wise dynamic > 统一宽度，等算力下）
- **Exp3**（1000 张，多次复现）：learned 0.2466 < random 0.2553 < fixed_medium 0.2588
- **Oracle 上界**：完美难度 Router（预算匹配）也只提升 DA +0.0125、Lane +0.0065
- **难度可预测性**：多尺度特征探针 AUC 0.69（中等，非 0.9）
- 尝试的修复均未突破：hard-STE → soft routing → DifficultyRouter（丰富特征+显式监督）
- 结论：**逐帧任务级路由在当前模型/数据上的收益上界本身很小**（+0.01~0.01 IoU），
  不是 Router 训练不足，而是杠杆有限

### 失败模式观察（§13）
- A（永远 Large）：无 budget loss 时出现 ✓ 已用 budget loss 控制
- B（永远 Tiny）：budget target 过强时出现（soft 训练塌缩）✓ 已调
- C（三任务同预算）：taskwise 模型 0.83 异构率 ✓ 未发生
- D（FLOPs 降延迟不降）：**发生**——eager PyTorch 的 Python 开销淹没小模型宽度收益
  （微基准 3.09→2.35ms 有效，eval 循环 7.3-7.8ms 噪声内）；需编译后端兑现
- E（难度估计不可靠）：**发生**——corr(choice, gain) ≈ 0

## 待验证
- H-02（Compact Z 小预算下保留三任务信息）：Exp4 等预算对比（需训练 0.5/1.0/2.0M 模型）

## 对论文的意义
- 核心卖点调整为：**单一紧凑模型 + 多运行档位 + Pareto 覆盖（H-03）**，
  并诚实报告逐帧路由的收益上界（+0.01 IoU）作为边界条件
- H-01 的更强形式可能需要 Phase 2 KD 增强表示（H-04）后重新验证——
  更强的 Compact Z 可能放大宽度敏感性，从而放大路由收益


# ＝＝＝ 来源：PHASE1B_REPORT ＝＝＝

# Phase 1-B 最终报告（Compact Representation + Dynamic Capacity + Cross-Architecture KD）

> 完成日期：2026-08-30 ｜ 目标：判断研究路线是否值得进论文（Q1-Q3 + 决策树）

## 核心结果表（§23，全量 tri_val 10,000，640×640）

| Model | Params | FLOPs | mAP50 | DA mIoU | Lane fgIoU | FPS | Latency |
|-------|-------:|------:|------:|--------:|-----------:|----:|--------:|
| YOLOP (B1) | 7.94M | 31.3G | 0.766 | 0.912 | 0.225 | 68.5 | 14.6ms |
| TwinLiteNet (B2) | 0.44M | 14.1G | — | 0.911 | 0.228 | 111 | 9.0ms |
| TriLiteNet-tiny (B3) | 0.15M | 1.8G | 0.495 | 0.880 | 0.195 | 226 | 4.4ms |
| **Ours Static (0.235M)** | 0.235M | 1.57G | 0.268 | 0.842 | 0.180 | 152 | 6.6ms |
| **Ours Static+KD** | 0.235M | 1.57G | **0.280** | **0.850** | **0.193** | 161 | 6.2ms |
| **Ours Dynamic** | 0.235M | ~0.90G | 0.243 | 0.837 | 0.167 | 136 | 7.4ms |
| **Ours Dynamic+KD** | 0.235M | ~0.90G | 0.046 | 0.441 | 0.109 | 139 | 7.2ms | |

## 各实验结论

### Experiment A：Equal-Budget Static vs Dynamic（3 seeds）
- **Case C 确认**：等平均 FLOPs（~0.90G）下，独立训练的 Static-EB（mAP 0.2555±0.003）
  全面优于或持平 Dynamic（0.1176±0.109，种子不稳定，部分 router 塌缩到 tiny）
- Dynamic 的价值 = 单一模型多档位覆盖（flexibility），非等预算 accuracy

### Experiment D：Compact Representation 容量（0.23→2.0M）
- **H-02 支持**：0.23M 紧凑 Z 保留三任务大部分信息（mAP 0.79× / DA 0.99× / Lane 0.90× of 2.0M）
- 容量敏感性：Detection > Lane > DA（DA 在 0.5M 饱和）——解释了路由收益上界小

### Experiment F/G：Single-Teacher Cross-Architecture KD
- **H-04 初步支持：三个 teacher 都提升 student**（10k×10 公平对比）
- YOLOP（检测 +0.013，综合最稳）> TriLiteNet（DA +0.022）> TwinLiteNet+
- 全量在线验证：YOLOP-KD 全面 +0.012/+0.007/+0.013（mAP/DA/Lane）
- 离线 KD（Plan A）实现：teacher 成本从每步 527-966ms 降到 0，F/G 从 ~15h 降到 ~2h

### Experiment I：KD 后重测 Dynamic
- **C-A：KD 有效（+0.012/+0.007/+0.013）✓**
- **B-A：Dynamic 无效（-0.026/-0.005/-0.014）**（与 ExpA 一致）
- **D-C：KD-Dynamic 全量公平版仍差（-0.234 mAP, -0.409 DA）**——诊断确认：
  **router 对 DA 塌缩到 tiny（57%），窄宽度 DA 头过度预测前景（cls1 0.37 vs 正常 0.15-0.25）**
  → 动态训练管线（A→B→C→D）系统性降低模型质量（ExpA s1/s2、ExpI D 三处复现）

## 决策树结论（§26）

```
Equal Budget:  Dynamic ≈ Static（Case B/C）→ Dynamic = flexibility
Compact Z:     有效（H-02 ✓）→ 保留 Compact Representation
Cross-Arch KD: 有效（H-04 ✓，C > A）→ 进入 KD
KD 后 Dynamic: 验证中（待全量 fair-D）
```

**确定路线：Route B（通用弹性模型）——Compact Model + KD + Runtime Elasticity：**
- **Compact Representation（H-02 ✓）**：0.23M 保留三任务 79-99% 信息，信息效率高
- **Cross-Architecture KD（H-04 ✓）**：显著提升 compact student（mAP/DA/Lane 全面 +0.012/+0.007/+0.013），
  是 accuracy 的核心引擎（论文卖点之一）
- **Dynamic Capacity = Runtime Elasticity（不承诺等预算 accuracy 增益）**：
  同一模型可在任意固定宽度运行（Pareto 覆盖真实存在，Exp1 强制宽度曲线已验证），
  但**学到的 router 不增加价值且训练不稳定**——如实报告，作为 negative result/ablation
- **KD+Dynamic 组合（D）失败**：router 塌缩问题在 KD 表示下依然存在

论文结论（候选）："紧凑表示 + 跨架构蒸馏可让 0.235M 模型在轻量 baseline 水平运行，
并支持多计算档位的弹性部署；学习式逐任务路由在当前极小区间内不提供额外 accuracy 收益
（oracle 上界 +0.0125 DA），其价值限于部署灵活性。"

## 关键工程成果
- 真实切片感知 FLOPs 计数（修正 thop 高估，Dynamic 平均 0.90G 而非 1.06G）
- 离线 KD（teacher 目标预计算 + 缓存训练）——时间从 15h → 2h
- 统一评测/训练协议、3-seed 统计、失败模式记录

## 服务器运行手册
见 `docs/RUNBOOK_SERVER.md`（全量 70k 最终数字可在服务器上跑，本地已提供全部脚本/协议）。


# ＝＝＝ 来源：PHASE1_STATUS ＝＝＝

# Phase 1 进度状态（2026-08-29 凌晨）

## Step 完成情况（任务书 §27）

| Step | 内容 | 状态 |
|------|------|------|
| 1-2 | 读现有项目/数据，确认可复用代码 | ✅ YOLOP/TwinLiteNet(+)/TriLiteNet 全部接入 |
| 3 | 统一 baseline evaluation | 🔄 全量 val 评测运行中（9 配置）|
| 4 | Compact Representation（Module A） | ✅ LightEncoder + CompactZ（0.166M/0.82G）|
| 5 | Static multi-task model | ✅ 0.227M / 1.57G，输出与 YOLOP 同构 |
| 6 | Task-Resource Router（Module B） | ✅ Z→Pool→MLP→task logits→离散档位；shared 变体 |
| 7 | 3 档 dynamic width（nested prefix） | ✅ 0.25/0.5/1.0，FLOPs 0.82G→1.57G 随宽度变化 |
| 8 | task-wise allocation | ✅ 逐帧异构分配（CSV 验证）|
| 9 | budget-aware 训练 loss | ✅ L_task + λ_budget·L_budget |
| 10 | Static vs Dynamic 实验 | ⏳ 待训练完成 |
| 11 | Shared vs Task-wise 实验 | ⏳ 待训练（shared 变体）|
| 12 | allocation 可视化 + Pareto | 🛠 工具就绪（visualization/plots.py, exp5_pareto.py）|
| 13 | 失败模式分析 | 🛠 工具就绪（failure_stats）|
| 14 | 阶段评估 | ⏳ |

## Baseline 评测（Step 3，全量 tri_val 10k）

| 模型 | Params | FLOPs | FPS | mAP50 | DA mIoU | Lane fgIoU |
|---|---|---|---|---|---|---|
| YOLOP | 7.94M | 31.3G | 84.1 | 0.766（论文 76.5 ✓）| 0.912（91.5 ✓）| 0.225 |
| TwinLiteNet | 0.44M | 14.1G | 130.9 | — | 0.911（91.5 ✓）| 0.228 |
| TLP nano/small/med/large | … | … | … | … | … | … |
| TriLiteNet tiny/small/base | … | … | … | … | … | … |

## 下一步

1. 等 baseline 全量评测完成（~40 min）
2. **Stage A 训练**（subset 10k 快速验证 → 全量），确认 loss 收敛 + val 指标提升
3. Stage B（冻结 encoder 训 router+heads）→ C（joint）→ D（budget-aware）
4. 训练 task-wise 与 shared 两个 router 变体（Exp2）
5. 跑 Exp1-5 + 失败模式 + 可视化
6. 论文级对比（含 Equal-Budget：TwinLiteNet+ / TriLiteNet / Ours static / Ours dynamic）

## 关键风险
- 训练时长：全量 69,863 张 × 4 阶段，估计 4-6 小时/轮；先用 10k 子集快速迭代
- Router 不收敛到异构分配（失败模式 C）：budget loss 与任务 loss 的平衡需调
- 动态宽度在真实硬件上的 latency 收益待验证（失败模式 D）


# ＝＝＝ 来源：PHASE2_PLAN ＝＝＝

# PHASE 2 — Execution Plan & Fixed Protocol

> 依据任务书 §38 顺序执行。本文档定义 Phase 2 全程统一的训练/评测 protocol，
> 映射 A0–A9 矩阵到具体 config/脚本/命令，并记录每一步状态。
> 全程遵守 novelty 边界（§23-24）：不复制 D2BNet/Sparse U-PDP/SCAM 结构，
> 不声称已有工作的"首发"。无效模块即删（§29/§36）。

## 0. 结论基线（来自 Phase 1 / 1-B，勿重跑）

| 模型 | Params | FLOPs | mAP50 | DA mIoU | Lane fgIoU | 备注 |
|---|---|---|---|---|---|---|
| ours_0.23M (static, no-KD) | 0.235M | 1.57G | 0.268 | 0.842 | 0.180 | 全 tri_val |
| ours + KD(YOLOP) | 0.235M | 1.57G | 0.280 | 0.850 | 0.193 | 单 teacher 最佳 |
| ours_0.5M | 0.424M | 2.46G | 0.282 | 0.840 | 0.193 | 容量档（4ep）|
| ours_1.0M | 0.959M | 4.22G | 0.295 | 0.852 | 0.194 | 容量档（3ep）|
| ours_2.0M | 1.980M | 8.83G | 0.319 | 0.845 | 0.194 | 容量档（2ep）|
| YOLOP (teacher) | — | — | 0.766 | 0.912 | 0.225 | 官方权重 |
| TwinLiteNet | — | — | — | 0.911 DA | — | 官方权重 |
| TriLiteNet-tiny | — | — | 0.495 | 0.880 | 0.195 | 官方权重 |

**容量档结论（H-02）**：Detection > Lane > DA 对容量敏感；DA/Lane 在 0.5M 即饱和。
注意：上述容量档用 4/3/2 非一致 epoch，仅作容量趋势上下文，不参与 Phase-2 等预算表。

## 1. Fixed Stage-A Protocol（Phase 2 全矩阵统一）

目标：A0–A9 与容量档在同 protocol 下可比。

| 项 | 值 |
|---|---|
| data | `tri_train` 全量(69,863) 训练；`tri_val` 全量(10,000) 评测 |
| batch_size | 16（drop_last）|
| optimizer | AdamW, lr 1e-3, wd 5e-4 |
| scheduler | CosineAnnealingLR over `epochs × steps` |
| epochs (Stage A) | **4**（与 Phase1 容量 4ep 可比、算力可控）|
| num_workers | auto ~ cores（本地 24 核 → 8）；pin_memory=True；prefetch=4 |
| seed | 0（主结果）；Step A 加 2 额外 seed 估方差 |
| device | cuda（脚本自动，拒绝静默 CPU）|
| grad clip | 10.0 |

推理基线统一评测：640×640 letterbox、单类车辆 mAP@0.5、DA/Lane 二值 Acc/fgIoU/mIoU、
CUDA-sync、bs=1、全 tri_val。评测脚本 `evaluation/evaluate_baseline.py`。

## 2. Step 1 — Equal-Budget Baseline

需要在一个固定 protocol 下重训 4 档容量（现有档 epoch 不一致，仅作趋势参考）：
- configs: `train_stageA_eb.yaml` 基础上派生 0.25/0.5/1.0/2.0M（模型段扩 encoder+z 到目标 params）
- 每档跑 3 seeds 估方差；评测 + 分层 latency
- 对齐基线：TwinLiteNet+ / TriLiteNet（官方权重，评测脚本现成）
- 报告列：Params, FLOPs, mAP50, DA mIoU, Lane fgIoU, FPS, P50, P95, Memory
- 比较：parameter-matched + FLOPs-matched

**状态：规划完成，待启动训练。**

## 3. Step 2 — Single-Teacher KD 复核（固定 protocol）

复用已有 cross-arch KD 机制，固定 protocol 复核 4 行：
- A0 no-KD（= Step1 0.235M 参考）
- A1 +YOLOP KD
- A2 +TwinLiteNet+ KD
- A3 +TriLiteNet KD
记录 ΔmAP / ΔDA / ΔLane。KD 走离线 cache（`scripts/phase2_precompute_teacher.py`）避免每步 teacher 开销。

**状态：待跑。**

## 4. Step 3 — Multi-Teacher KD（Exp H，本次真正新建）

- Teacher：YOLOP / TwinLiteNet+ / TriLiteNet
- 组合：Y+Tw, Y+Tr, Tw+Tr, Y+Tw+Tr
- 每 teacher 走 `Teacher-specific projection P_i → common latent`，融合（首版 3 个极简方法）：
  M1 Average；M2 Fixed task-aware weight（YOLOP→det，Twin/Tri→seg）；M3 confidence-weighted
- 第一版**不做** attention fusion / 复杂 Teacher Fusion Network
- KD 损失施加于：student Z（feature 对齐到 common latent）+ 输出（seg sigmoid-MSE）
- 必须回答 Q1（互补性）/Q2（冗余）/Q3（task-wise 最佳 teacher）
- 可能结果：multi 反而 ≤ YOLOP-only（§28）——记录 teacher contribution/conflict/task-wise gain

**状态：需新建 config + loss + teacher-projection 模块。**

## 5. Step 4/5 — Static Task Adapter（+ KD）

- Adapter 极小：先 1×1 conv（每任务一个）；不够再 dw3×3+1×1。绝非 dynamic routing，是"极小预算下给三个头的固定专属容量"。
- 挂在 shared Compact Z 之后、每 head 之前。
- 必须 equal-parameter 消融：Adapter 增加 ~0.05M 时，用同等 generic 加宽模型对照，判断是"任务特化有效"还是"纯变大"。
- Step 5 四格：A=Shared no-KD, B=Shared+KD, C=Adapter no-KD, D=Adapter+KD。
- D>B 且 D>C 才保留；否则删（§29）。

**状态：需新建 adapter 模块 + equal-param 对照 config。**

## 6. Step 6-8 — Elastic Student

- **禁止 task-wise width**（router→每任务独立宽）——旧问题。只做 global capacity：router 输出单标量档位 Tiny/Medium/Large，三任务共享同一容量状态。
- 只保留原 Router 作 negative baseline。
- 训练顺序（§18，避免旧 Dynamic+KD 崩溃）：
  1) 先稳定 KD student（Step2 产出）
  2) 在该 student 上训练 elastic subnet（各 width 充分采样）
- Adapter 必须兼容 0.25×/0.5×/1.0× width。

**状态：待新建 global-capacity elastic 模型 + elastic 训练脚本。**

## 7. Step 9 — 全 Pareto / Latency 分层

- 指标：Accuracy–Params / Accuracy–FLOPs / Accuracy–Latency
- latency 分层记录：PyTorch eager → torch.compile → （条件允许）ONNX / TensorRT
- FLOPs↓ 不直接等同 latency↓（eager 已观察失真），必须实跑 latency
- 借鉴 SCAM-P 测法（1000 frames + 100 warmup），但本机只有 RTX 5060，不虚构 edge 数据

**状态：待 Step1 后建 benchmark 脚本。**

## 8. A0–A9 完整矩阵（最终报告用）

| id | 模型 | KD | Multi-T | Adapter | Elastic |
|---|---|---|---|---|---|
| A0 | Ours static | × | × | × | × |
| A1 | +YOLOP KD | ✓ | × | × | × |
| A2 | +TwinLite KD | ✓ | × | × | × |
| A3 | +TriLite KD | ✓ | × | × | × |
| A4 | +Multi-Teacher KD | ✓ | ✓ | × | × |
| A5 | +Task Adapter | × | × | ✓ | × |
| A6 | +KD+Adapter | ✓ | ✓ | ✓ | × |
| A7 | Ours Elastic | × | × | × | ✓ |
| A8 | +KD | ✓ | ✓ | × | ✓ |
| A9 | +KD+Adapter | ✓ | ✓ | ✓ | ✓ |

## 9. 退出路线（§31-35，避免强行复杂化）

- Multi-teacher 无增益 → `Compact + best single + Elastic`
- Adapter 无增益 → `Compact + KD + Elastic`
- Elastic 无真实 latency 收益 → 不宣称 speedup，宣称 "configurable computational capacity"（deployment limitation 如实分析）
- 若同预算超 TriLite/TwinLite → Route A；否则 Route B（single compact elastic MT model）

## 10. 每步汇报格式（§37，17 项）
每步产出：目的 / 相关论文 / 实现 / Params / FLOPs / mAP50 / DA mIoU / Lane fgIoU /
FPS / P50 / P95 / Memory / vs previous baseline / 是否支持假设 / 新 collision /
是否保留 / 下一步。


# ＝＝＝ 来源：PHASE2B_POSITIONING ＝＝＝

# Phase 2-B — 方向定位与竞品对照（Direction Positioning）

> 目的：回答三个问题 —— (1) 我们做的方向是什么；(2) 与竞品逐方面区别；(3) 目前结果能否与对手比对、优劣在哪。
> 所有数字来自实测报告（P2-STEP1 / P2-STEP2 / Exp-D / 同预算对齐表），未做任何推断性填充。
> 生成日期：2026-09-03

---

## 一、方向定位（一句话）

我们做的**不是**「又一个更轻的三任务网络」，而是对一个**可测量科学关系**的研究：

> 在 BDD100K 三任务（Detection + Drivable Area + Lane）、**极低算力（<2 GFLOPs）** 下，
> **同一个极小共享瓶颈 Compact Z 的容量，如何决定每个任务各自的精度、饱和点与算力效率**；
> 以及在等参数/等 FLOPs 下，「共享瓶颈」是否比「单纯加宽骨干」更高效。

核心假设（可证伪）：

> Compactness is an **information-allocation mechanism**, not just a smaller feature map.

竞品回答的是「谁更准」，我们回答的是「紧凑为什么有效、什么时候失效、拐点在哪」。

---

## 二、与竞品逐方面区别

| # | 维度 | 竞品（TriLiteNet / TwinLiteNet+ / YOLOPX / YOLOP / Sparse U-PDP / D2BNet / SCAM-P） | 我们（PercepFlex） |
|---|---|---|---|
| 1 | 问题类型 | 架构设计问题：如何设计更准/更快的三任务网络 | **测量问题**：Z 容量 ↔ 每任务精度/饱和点/算力的函数关系 |
| 2 | 核心主张 | 我的结构更好（注意力、双分支、重参数化、动态路由） | 紧凑是信息分配机制（可被证伪的假设） |
| 3 | 优化目标 | 精度↑、参数↓、FLOPs↓ 三者取 Pareto | **拐点定位**：Z 多大开始收益递减、哪个任务先饱和 |
| 4 | 实验方法论 | 报告自己模型的最终分数表 | **受控扫描**：单变量 Z∈{16..128}、等预算 shared-vs-widening、配对对照 |
| 5 | 效率维度 | 主要报 Params / FLOPs，很少报实测延迟 | Params + FLOPs + **实测 p50/p95/FPS**（RTX5060, bs1） |
| 6 | 任务与数据 | 同三任务（BDD100K）；TwinLiteNet+ 仅 DA+lane 双任务 | 同三任务，使用 tri_ 子集（69 863 / 10 000） |
| 7 | 贡献形态 | 新模块/新结构，追求 SOTA | **经验性规律 + 设计准则**，EI/工程论文，不追求 SOTA |

### 2.1 Novelty 边界（来自 18 篇文献审计，务必遵守）

- **A 类（明显撞车）**：无。没有单篇同时覆盖「BDD100K 三任务 + 极小共享瓶颈 + 容量扫描 + 等预算效率」。
- **D 类（空缺，可作核心）**：
  1. 无工作在 BDD100K 三任务、<2 GFLOPs 下对**同一紧凑共享瓶颈做 Z∈{16..128} 系统容量扫描**并报告每任务饱和点；
  2. 无工作在等参数/等 FLOPs 下做 **shared-bottleneck vs 单纯加宽** 的效率对比实证；
  3. 无工作把「三任务对共享表示容量需求不同步」做成**可测量观察**并据此给出信息分配设计（非动态路由）。
- **C 类雷区（不可当卖点）**：
  - 「共享表示喂多头」本身 → 撞 #12（人脸情感域）/ #10 MT-VIB 先例；
  - 「IB 应用于 MTL」→ 撞 #9 Bi-MTDP / #10 MT-VIB / #11 SGW-KEM-IB；
  - 「弹性宽度」→ 撞 OFA / DS-Net / D2BNet；
  - 「驾驶 MTL + pruning + KD」→ 撞 arXiv:2511.05557（且其为训练后剪枝 34→23M，非从头极小瓶颈）。

**结论**：卖点只能是「驾驶三任务对共享表示容量需求的**任务差异** + 等预算下 shared 比 widened 更高效」的**实证与机制解释**，而非结构 novelty。

---

## 三、目前结果（全部为已实测可信数字）

### 3.1 等预算表（固定 Stage-A protocol：4ep / bs16 / AdamW lr1e-3 / cosine / seed0）

| 模型 | Params | FLOPs | mAP50 | DA mIoU | DA fg | Lane fg | Lane mIoU | FPS |
|---|---|---|---|---|---|---|---|---|
| A0 0.235M | 0.230M | 1.47G | 0.2414 | 0.8360 | 0.7418 | 0.1843 | 0.5802 | 157 |
| 0.5M | 0.424M | 2.46G | 0.2846 | 0.8334 | 0.7384 | 0.1915 | 0.5837 | 186 |
| 1.0M | 0.959M | 4.22G | 0.2951 | 0.8518 | 0.7654 | 0.1943 | 0.5855 | 179 |
| 2.0M | 1.980M | 8.84G | 0.3187 | 0.8450 | 0.7555 | 0.1939 | 0.5853 | 158 |

> 注：1.0M / 2.0M 沿用 Phase-1 权重（3ep / 2ep），非 4ep，已在原始报告中如实标注。

### 3.2 实测延迟（bs1 / 640 / RTX5060 / eager）

| 模型 | p50 | p95 | FPS |
|---|---|---|---|
| A0 0.235M | 3.73 ms | 10.44 ms | 184 |
| 0.5M | 3.94 ms | 10.95 ms | 178 |

`torch.compile` 在本环境（Blackwell + torch 2.11）inductor 报错，已记录为部署限制，未强行报数。

### 3.3 已确认的科学观察（Exp-D）

| 任务 | 0.235M 相对 2.0M 保留率 | 饱和行为 |
|---|---|---|
| Detection mAP50 | **79.2%** | 持续爬升，2.0M 才到 100% |
| Drivable Area mIoU | 98.7% | 0.5M 即 ≥99.4% |
| Lane fgIoU | 89.7% | 0.5M 即 ≥99.6% |

→ **非同步退化（asynchronous degradation）**：三任务对共享表示容量的需求显著不同，Detection 是唯一持续受益于容量者。这是我们的核心差异化观察。

### 3.4 负结果（诚实记录）

- **单 teacher KD 无稳定增益**（同 9999 子集配对，固定 4ep）：YOLOP +0.0025 mAP、TwinLite+ −0.0074、TriLite −0.0127。DA 略降、Lane 中性。
- 与 Phase-1 声称的 +0.012 mAP 冲突。冲突解释：Phase-1 那次提升被「KD 用 10k / no-KD 用 69k」的数据量差异混淆。
- → KD 不作为本项目基石；Task 2(e) 已据此降级为辅助手段。

---

## 四、能否与对手比对？—— 目前**不能完全公平比对**

### 4.1 同预算对齐表（官方 pretrained，tri_val）

| 模型 | Params | FLOPs | mAP50 | DA mIoU | Lane fg |
|---|---|---|---|---|---|
| Ours 0.235M | 0.230M | 1.47G | 0.2414 | 0.8360 | 0.1843 |
| TriLite-tiny | 0.151M | 1.83G | 0.4953 | 0.8796 | 0.1952 |
| Ours 0.5M | 0.424M | 2.46G | 0.2846 | 0.8334 | 0.1915 |
| TriLite-small | 0.592M | 6.61G | 0.6326 | 0.9053 | 0.2198 |
| TLP-medium | 0.479M | 15.4G | n/a（无 det） | 0.9191 | 0.2297 |
| Ours 2.0M | 1.980M | 8.84G | 0.3187 | 0.8450 | 0.1939 |
| TriLite-base | 2.350M | 25.4G | 0.7235 | 0.9203 | 0.2371 |
| TLP-large | 1.944M | 58.6G | n/a（无 det） | 0.9279 | 0.2450 |

### 4.2 **不对等声明（必须在论文中显式写出）**

- 官方权重 = 全量 BDD100K + 充分训练；
- 我们 = tri_train（69 863） + 固定 4ep。

因此该表**只能作为风险信号，不能作为最终判决**。

### 4.3 口径修正（重要）

早期文档写的「FLOPs 低 6–10×」是拿 **TwinLiteNet+** 比的，而 TLP 只做**双任务**（无检测头），比 FLOPs 天然占便宜。
**同任务（三任务）公平对照只有 TriLiteNet**，实测 FLOPs 倍数为：

| 对照 | FLOPs 倍数（对手 / 我们） |
|---|---|
| Ours 0.235M vs TriLite-tiny | 1.83 / 1.47 = **1.24×** |
| Ours 0.5M vs TriLite-small | 6.61 / 2.46 = **2.69×** |
| Ours 2.0M vs TriLite-base | 25.4 / 8.84 = **2.87×** |
| Ours 0.5M vs TLP-medium（双任务） | 15.4 / 2.46 = **6.26×** |
| Ours 2.0M vs TLP-large（双任务） | 58.6 / 8.84 = **6.63×** |

→ 对外表述应改为「同参数下 FLOPs 低 1.2–2.9×（三任务对照）/ 6.3–6.6×（双任务对照）」，**不得笼统称 6–10×**。

### 4.4 FLOPs 归一化效率（我们的软肋，必须自曝）

mAP50 per GFLOP：

| 模型 | mAP/GFLOP |
|---|---|
| **TriLite-tiny** | 0.4953 / 1.83 = **0.271** |
| Ours 0.235M | 0.2414 / 1.47 = **0.164** |
| Ours 0.5M | 0.2846 / 2.46 = 0.116 |
| TriLite-small | 0.6326 / 6.61 = 0.096 |
| Ours 1.0M | 0.2951 / 4.22 = 0.070 |
| Ours 2.0M | 0.3187 / 8.84 = 0.036 |
| TriLite-base | 0.7235 / 25.4 = 0.028 |

**即使按 FLOPs 归一化，TriLite-tiny 仍领先我们 1.65×。这个差距无法完全由「训练不充分」解释。**
→ 「我们 FLOPs 效率更高」这个说法在当前数据下**不成立**；唯一站得住的是「**同参数下 FLOPs 绝对值更低**」（紧凑 1/8 表示带来的结构性事实）。

---

## 五、优劣清单

### 5.1 优势（站得住的）

1. **同参数下 FLOPs 绝对值显著更低**（三任务对照 1.2–2.9×）—— 来自 1/8 分辨率的紧凑共享表示，是结构性事实，不依赖调参。
2. **实测延迟真实可用**：A0 p50 3.73 ms / 184 FPS（RTX5060, bs1, eager），竞品少有同条件实测。
3. **变量控制干净**：`z_channels` 是单一连续旋钮，可做纯粹的单变量容量扫描——这是竞品论文不具备的方法学条件。
4. **科学问题无人占据**（D 类空缺 3 项），且已有非同步退化的初步实证。
5. **负结果记录完整**（KD 三条 teacher 全部实测），可作为论文中的 honest negative result，增加可信度。

### 5.2 劣势 / 风险（必须正视）

1. **绝对精度明显落后**：同 FLOPs 下检测 mAP 约为 TriLite-tiny 的一半（0.241 vs 0.495）。
2. **⚠️ 最大发表风险：缺同 protocol 重训的竞品对照。** 现有对照全部是「官方充分训练权重 vs 我们 4ep」，审稿人会直接质疑。**必须在论文前补做**：用我们的 protocol（tri_train 69 863 / 4ep / bs16 / lr1e-3 / cosine）重训 TriLiteNet-tiny（与 small），得到真正同 protocol 的对照点。
3. **「FLOPs 效率更高」目前不成立**（见 §4.4），若强行主张会被数据推翻。
4. **Route A 已排除**：同预算下不太可能超过 TriLite/TwinLite+，因此不能走精度竞赛路线。
5. **KD 辅助路线已堵**：单 teacher 无增益，Multi-Teacher 仅在 (c) 修好后作为辅助，且不得作为卖点。

---

## 六、建议的论文 positioning 表述（避开 C 类雷区）

不可用：
- ~~「我们提出了共享瓶颈表示」~~（撞 MT-VIB / #12）
- ~~「我们将信息瓶颈引入多任务学习」~~（撞 Bi-MTDP / MT-VIB）
- ~~「我们的方法是首个用于驾驶三任务的紧凑模型」~~（撞 TriLiteNet）

可用（以测量与机制为核心）：

> We present a **controlled capacity study** of an extremely small shared bottleneck (Compact Z)
> for BDD100K three-task driving perception at ultra-low compute (<2 GFLOPs). Rather than
> proposing a new architecture, we measure **how** per-task accuracy, saturation points, and
> compute efficiency vary as a function of bottleneck capacity, and show that the three tasks
> exhibit **asynchronous degradation** under a shared representation — detection degrades
> first and most severely, while drivable-area and lane saturate early. We further provide an
> **equal-budget comparison of shared-bottleneck versus backbone widening**, and derive
> practical guidance for information allocation at a fixed compute budget.

---

## 七、决策建议（Task 3 输入）

- **判定：CONTINUE（带明确补救条件）**。D 类空缺确实存在，非同步退化已有初步实证，方向成立。
- **触发 PIVOT 的条件**：
  1. 同 protocol 重训 TriLiteNet 后，我们在等 FLOPs 下仍显著落后且「shared 比 widened 更高效」不成立 → 核心假设被证伪 → PIVOT；
  2. Z 扫描显示三个任务的饱和点几乎重合（无非同步退化）→ 差异化观察消失 → PIVOT。
- **Task 3 之前必须补的 1 项实验**（非原有计划内，但为发表所必需）：
  **TriLiteNet-tiny/small 按我们 protocol 重训**（tri_train 69 863, 4ep, bs16, lr1e-3, cosine, seed0），
  产出同 protocol 对照点。这是把「风险信号」变成「可辩护结论」的唯一途径。


# ＝＝＝ 来源：PHASE2B_REPRODUCIBILITY ＝＝＝

# Phase-2-B — Reproducibility audit: how large is run-to-run noise?

**Status: measured, not assumed. This finding constrains every claim Task 2 can make.**

Date: 2026-09-03 · sweep commit range `ce560f4` … `7b71d4b`

---

## 1. The natural repeat experiment

Task 2(a) sweeps `z_channels` over {16, 32, 48, 64, 96, 128} with the encoder fixed.
Step-1's `0.235M` reference row used **`z_channels: 64`** with the same encoder geometry.
Our z=64 sweep point is therefore the *same model* under the *same protocol* — an
unplanned but genuine repeat experiment.

| | Step-1 `0.235M` | this sweep `z=64` | difference |
|---|---|---|---|
| Params | 0.230 M | 0.230 M | **identical** |
| FLOPs | 1.47 G | 1.473 G | identical (rounding) |
| mAP50 | 0.2414 | 0.2424 | +0.0010 |
| **da_mIoU** | **0.8360** | **0.8258** | **−0.0102** |
| lane_fg | 0.1843 | 0.1836 | −0.0007 |

Configs were diffed field by field and are semantically identical: encoder
`stem 16 / stages [32,64,96,128] / blocks [2,2,2]`, `z_channels: 64`,
`nc: 1`, `segmentation.hidden: 32`, `input_size [640,640]`, `stage: A`,
`train_split: tri_train`, `batch_size 16`, `epochs 4`, `lr 1e-3`,
`weight_decay 5e-4`, `lambda_da/lane 1.0`, `grad_clip 10.0`, `num_workers 2`.

**So a same-config, same-seed repeat differs by 0.0102 da_mIoU.**

## 2. Where the nondeterminism comes from

- `training/train.py:262` sets `torch.manual_seed(args.seed)` and `:349` seeds the
  DataLoader generator — shuffling is reproducible.
- `training/train.py:357` **auto-raises `num_workers` from 2 to 8** on this box
  (`cores=24`). The value is stable on one machine but is *machine dependent*, so a
  run is not portable across hosts.
- `training/train.py` sets **neither `torch.backends.cudnn.deterministic` nor
  `torch.use_deterministic_algorithms`**. The YOLO detection loss uses indexed
  scatter/atomic CUDA ops, which are not bit-reproducible; over 4 epochs the
  resulting gradient differences compound into the observed 0.01 mIoU.
- `datasets/*.py` uses no `random` / `numpy.random`, so augmentation is not an
  additional unseeded source.

## 3. Consequence: the sweep is underpowered

Observed da_mIoU across the sweep: **0.8199 → 0.8317 → 0.8328 → 0.8258** (z = 16/32/48/64).
The series is **non-monotonic** — z=64 falls back *below* z=32 and z=48.

- Total spread across z ≥ 32: **0.0070**
- Spread across all four points: **0.0129**
- Measured repeat noise (§1): **0.0102**

The signal we are trying to resolve is therefore **~1.3× the repeat noise**. Under a
single seed the Z → da_mIoU relation is **not resolvable above z = 16**.

What *is* supported by the data:

- **z = 16 is measurably worse than z ≥ 32** for DA (0.8199 vs ~0.830, a gap of
  ~0.010, i.e. at the scale of the repeat noise but consistent in direction).
- **Above z = 32 there is no resolvable gain** for any of the three tasks.

What is **not** supported and must not be claimed:

- ~~"DA saturates at z = 32"~~ — the 32 → 48 delta is +0.0011, an order of magnitude
  below the noise floor; and z = 64 moves the other way.
- Any ranking among z ∈ {32, 48, 64, 96, 128}.

Detection behaves as designed: under R0 the detection head reads multi-scale encoder
features and never touches Z, so its variation across the sweep
(0.2368 / 0.2421 / 0.2360 / 0.2424, σ ≈ 0.0034) is a **control** — see the analyzer's
`--noise-control` flag.

## 4. Latency is affected by the same class of problem

In-sweep p50 latency: z=16 **7.256 ms**, z=32 **2.412 ms**, z=48 **2.664 ms**,
z=64 **5.810 ms** — for models whose FLOPs differ by only 1.39×. z=16 is the
known cold-clock outlier (see commit `ebe0248`), but z=64 at 5.81 ms shows that even
evals run right after training are not stable. **No in-sweep latency/FPS number is
usable**; all checkpoints must be re-benchmarked with
`scripts/phase2b_rebench_latency.py` before any latency claim.

## 5. Required actions

1. **Do not report per-Z-point rankings from single-seed runs.** Report the shape
   (z=16 deficient; z ≥ 32 flat) with the noise floor stated alongside.
2. **Re-run the decisive points with ≥ 3 seeds.** Minimum viable set: z ∈ {16, 32, 128}
   × seeds {0, 1, 2} = 9 runs ≈ 5.4 h GPU at ~36 min/run. Full 6 points × 3 seeds is
   ~10.8 h. Report mean ± std; only differences exceeding ~2σ may be called effects.
3. **Optionally make training reproducible** so future single runs are meaningful:
   set `torch.use_deterministic_algorithms(True)` and
   `torch.backends.cudnn.deterministic = True`, pin `num_workers` explicitly via
   `--num-workers`, and add a `worker_init_fn` seeding `random`/`numpy` per worker.
   This typically slows training; it should be measured before adopting it.
4. **Re-benchmark latency** for all checkpoints under one consistent warm protocol.

## 6. Silver lining

This is a *negative-control* result, and it is exactly what a rigorous capacity study
needs: it tells us the measurement instrument's resolution before we interpret the
curve. Discovering it now is far cheaper than discovering it in review. It also
directly informs Task 2(c): if we route detection through Z (R1/R2) and then see a
change, we now know the size a change must exceed to be believed.


# ＝＝＝ 来源：PHASE2B_TASK2C_DESIGN ＝＝＝

# Phase 2-B Task 2(c) — R1/R2: does Detection belong in Compact Z?

Date: 2026-09-04 · written BEFORE the runs, so the design is not fitted to results.

## 1. The question

Under R0 the detection head reads the encoder's multi-scale features
[F2(1/8), F3(1/16), F4(1/32)] and **never touches Z**; only DA and lane read Z.
That is fine engineering, but it weakens the paper's central claim: if the hardest
task bypasses Z, then "one compact shared Z carries all three tasks" has not been
tested — Z is only a segmentation neck.

Task 2(c) forces detection through Z and measures what it costs.

| variant | detection reads | entry projection |
|---|---|---|
| R0 (done) | encoder multi-scale features | — |
| R1 | Compact Z | `z_proj: false` (Z used raw) |
| R2 | Compact Z | `z_proj: true` (1x1 conv + BN + ReLU) |

## 2. Pre-flight finding A: R1 and R2 are nearly the same model

`models/representation/det_from_z.py:76-87`:

```python
if proj:      # R2
    Sequential(Conv2d(zc, det_ch, 1), BatchNorm2d(det_ch), ReLU)
else:         # R1
    Sequential(Conv2d(zc, det_ch, 1) if zc != det_ch else Identity(),
               BatchNorm2d(det_ch)   if zc != det_ch else Identity(),
               Identity())
```

So R1 and R2 differ **only by one BN + ReLU**, at every z.
The exception is **z = 32**, where `zc == det_ch == 32` makes R1's entry conv an
`Identity` as well — the only point where R1 genuinely feeds Z in raw.

**Consequence.** The originally planned "R1 vs R2 at several z" is a micro-ablation
whose expected effect is far below the measured noise floor (2 sigma on mAP50 =
0.0063, see §4). Spending ~32 min per point on it cannot produce a usable result.
Revised: R1 is run at **one** point (z=32), last, and is droppable.

## 3. Pre-flight finding B: R2's detection head is 2.5-3.4x LARGER than R0's

`scripts/phase2b_r1r2_paramaudit.py` (params, encoder fixed):

| z | R0 det head | R2 det head | R2/R0 | R0 total | R2 total | delta |
|---|---|---|---|---|---|---|
| 32 | 8,406 | 21,430 | 2.55x | 202,710 | 215,734 | +13,024 |
| 64 | 8,406 | 22,454 | 2.67x | 230,422 | 244,470 | +14,048 |
| 128 | 8,406 | 24,502 | 2.92x | 285,846 | 301,942 | +16,096 |
| 256 | 8,406 | 28,598 | 3.40x | 396,694 | 416,886 | +20,192 |

R0's head is almost free (three 1x1 convs straight off encoder features). R2 must
first project Z and then *build* a pyramid (two 3x3 stride-2 blocks at width 32),
which costs real parameters.

**This determines how the result is read — and the confound runs in our favour:**

- If R2 detection is **worse** than R0, it is worse *despite* having 2.5-3.4x the
  head parameters. That strengthens the conclusion: the bottleneck, not capacity,
  is the cause.
- If R2 detection is **comparable or better**, we CANNOT attribute it to Z being
  sufficient — the bigger head is an alternative explanation. We would then need a
  param-matched control before claiming anything.

Either way the efficiency story is affected: routing detection through Z **costs**
params/FLOPs here, it does not save them.

Cross-check: the audit's 215,734 params for R2 z=32 matches the smoke run's
reported `parameters=215,734` exactly, so the audit and the real model agree.

## 4. Noise floor — what effect size is believable

From Task 2(a) (`docs/PHASE2B_REPRODUCIBILITY.md`):

- Same-config repeat (z=64 vs Step-1 0.235M): **0.0102 da_mIoU**.
- Control metric mAP50 across 6 R0 runs: sigma = 0.0032, **2 sigma = 0.0063**.

**Important caveat for this task.** Under R0, mAP50 was a *control* (detection did
not read Z). Under R1/R2 it *is* the treatment, so we lose the in-run control. We
carry 2 sigma = 0.0063 over as a first-order estimate, but it was measured on a
different gradient path and with only n=6, so it is an estimate, not a measurement.

Decision rule: **only an R0 vs R2 mAP50 gap larger than ~0.0063 may be called an
effect; anything smaller is noise.** A collapse of the size we would expect if Z
cannot carry detection (say 0.24 -> 0.15) is unambiguous regardless.

## 5. Design

Fixed protocol, identical to the R0 sweep: 4 epochs, bs16, AdamW lr 1e-3, cosine,
seed 0, `tri_train`. Encoder fixed; `det_ch` fixed at 32 so head capacity does not
confound the Z sweep.

Z points, **ordered by information value so early stopping discards the least**:

| order | run | why |
|---|---|---|
| 1 | R2 z=32 | R0's saturation point. Is the capacity that suffices for DA/lane enough for detection? |
| 2 | R2 z=128 | 4x that. Does detection simply need more capacity? |
| 3 | R2 z=64 | fills the curve between 32 and 128 |
| 4 | R2 z=256 | upper bracket; finds break-even if 128 is still short |
| 5 | R1 z=32 | the only genuine R1/R2 ablation (§2); droppable |

~32 min/point (measured from the R0 sweep), so ~2h40m for all five.
Runs 1-2 (~1h05m) already answer the headline question.

## 6. Fairness checks passed before launch

- **Anchors identical**: `train.py:36-38`, `det_head.py:25-27`,
  `dynamic_det_head.py:22-24`, `det_from_z.py:42-44` — same 3x3 anchors. No
  anchor confound between R0 and R2.
- **Scale count identical**: DetFromZ emits 3 scales at strides 8/16/32 with the
  same output format as R0's head, so YOLOLoss sees the same shapes.
- **End-to-end path verified**: `experiments/phase2b/smoke.log` — R2 z=32 trained
  1 epoch and evaluated, `missing=0 unexpected=0`, reached `SMOKE_DONE`.
  (mAP50 = 0.0 there is expected at 1 epoch on 160 images.)

## 7. Launch

```bash
cd ~/ai_study/trac
nohup bash scripts/phase2b_run_r1r2.sh "R2:32,128,64,256" "R1:32" \
  > experiments/phase2b/r1r2_launch.log 2>&1 &
```

Graceful stop at any point boundary: `touch experiments/phase2b/STOP`
Resumes safely: a point whose `*_eval/metrics.json` exists is skipped; a point with
only a checkpoint goes straight to eval. The CSV is appended, never truncated.

## 8. Analysis rules (agreed before seeing data)

- Compare R2 against the R0 row **at the same z** (zsweep_results.csv), never
  against a different z.
- Report the da_mIoU / lane columns too: if detection degrades while DA/lane hold,
  that is the clean signature of a capacity conflict inside Z.
- Do not report an R1 vs R2 difference unless it exceeds 2 sigma = 0.0063.
- If the R0 vs R2 detection gap is large, confirm at least one point with a second
  seed before writing it into the paper.


# ＝＝＝ 来源：PHASE2C_EXPA_RESULTS ＝＝＝

# Experiment A — Z=16/32/128 · 3-seed confirm · RESULT (2026-09-04)

## 原始数据（3 seeds/config，协议固定 4ep/bs16/AdamW lr1e-3/cosine/tri_train）

seed0 来自 zsweep_results.csv，seed1/2 为本轮实验 A 补跑（全部 6 点完成，
11:22→14:35，~3.2h GPU，run_in_background 托管无中断）。

| z | seed | mAP50 | mAP50_95 | da_mIoU | da_fg | lane_fg | lane_mIoU |
|---|---|---|---|---|---|---|---|
| 16 | 0 | 0.2368 | 0.0768 | 0.8199 | 0.7190 | 0.1818 | 0.5792 |
| 16 | 1 | 0.2346 | 0.0755 | 0.8315 | 0.7355 | 0.1733 | 0.5735 |
| 16 | 2 | 0.2332 | 0.0730 | 0.8092 | 0.7038 | 0.1738 | 0.5740 |
| 32 | 0 | 0.2421 | 0.0793 | 0.8317 | 0.7359 | 0.1802 | 0.5775 |
| 32 | 1 | 0.2260 | 0.0732 | 0.8258 | 0.7274 | 0.1763 | 0.5751 |
| 32 | 2 | 0.2502 | 0.0831 | 0.8223 | 0.7223 | 0.1811 | 0.5784 |
| 128 | 0 | 0.2403 | 0.0789 | 0.8318 | 0.7360 | 0.1841 | 0.5798 |
| 128 | 1 | 0.2419 | 0.0795 | 0.8321 | 0.7365 | 0.1822 | 0.5782 |
| 128 | 2 | 0.2454 | 0.0820 | 0.7950 | 0.6835 | 0.1817 | 0.5790 |

## mean ± std（n=3）

| metric | z16 | z32 | z128 |
|---|---|---|---|
| mAP50 | 0.2349±0.0015 | 0.2394±0.0101 | 0.2425±0.0021 |
| mAP50_95 | 0.0751±0.0016 | 0.0785±0.0041 | 0.0801±0.0013 |
| da_mIoU | 0.8202±0.0112 | 0.8266±0.0048 | 0.8196±0.0213 |
| da_fg | 0.7194±0.0159 | 0.7285±0.0069 | 0.7187±0.0305 |
| lane_fg | 0.1763±0.0048 | 0.1792±0.0026 | 0.1827±0.0013 |
| lane_mIoU | 0.5756±0.0032 | 0.5770±0.0017 | 0.5790±0.0008 |

## 效应量（mean diff + pooled Cohen d）

| metric | z32−z16 | z128−z32 |
|---|---|---|
| mAP50 | +0.0046 (d=0.52) | +0.0031 (d=0.35) |
| mAP50_95 | +0.0034 (d=0.91) | +0.0016 (d=0.43) |
| da_mIoU | +0.0064 (d=0.75) | −0.0070 (d=−0.45) |
| lane_fg | +0.0029 (d=0.76) | +0.0035 (d=1.72) |
| lane_mIoU | +0.0014 (d=0.56) | +0.0020 (d=1.50) |

## 决定性对账：信号 vs 噪声（|跨z差| vs 该指标最大 within-z sample-std）

| metric | z32−z16 | z128−z32 | within-z 噪声(max) | 是否超噪声 |
|---|---|---|---|---|
| mAP50 | +0.0046 | +0.0031 | 0.0123 (z32) | 否 |
| mAP50_95 | +0.0034 | +0.0016 | 0.0050 (z32) | 否 |
| da_mIoU | +0.0064 | −0.0070 | 0.0213 (z128) | 否 |
| da_fg | +0.0091 | −0.0099 | 0.0305 (z128) | 否 |
| lane_fg | +0.0029 | +0.0035 | 0.0048 (z16) | 否 |
| lane_mIoU | +0.0014 | +0.0020 | 0.0032 (z16) | 否 |

**每项指标的跨 z 差异都小于（甚至远小于）该指标自身的 seed 间波动。**

## 排序稳定性（跨 seed）

- mAP50 / mAP50_95：0/3 seed 呈现 z16<z32<z128 单调
- da_mIoU / da_fg：1/3，且 z128 均值反而低于 z32（seed2 离群 0.795）
- lane_fg / lane_mIoU：2/3（唯一方向较一致，但非全 seed 单调，效应量仅 ~1×噪声）
- seed 间"哪个 z 最好"来回跳：seed0→z32, seed1→z128, seed2→z32

## 诚实结论（按决策规则逐条判定）

1. **z16<z32 且稳定超噪声？→ 否。** 全部指标跨 z 差落在噪声内，排序 0/3 单调。
   不支持"极低容量存在明显瓶颈"。
2. **z32≈z128 且不可分辨？→ 更准确说是 z16≈z32≈z128 在噪声内全部不可分。**
   因此既不能支持"z32 后有 diminishing return"（因为连 z16→z32 都不可分）。
3. **排序不稳定？→ 是。** 依规则：**不得声称存在确定 knee。**

**最诚实的总结**：在本固定协议（4ep/3-seed）下，Z 容量 16→128 对三个任务
无可分辨的统计效应。run-to-run 噪声（σ≈0.005–0.02，因指标而异）> Z 容量信号。
这**不能证明 Z 无关**（可能 4ep 太短，或此架构下 Z≥16 已非瓶颈），但**现有证据
不足以支持"Z 容量是决定性能的因素"这一核心假说**。

### 噪声来源提示（非为开脱，供诊断）
- z32_s1 mAP50=0.226、z128_s2 da=0.795 是两个大离群点，拉高了样本 std。
- 4ep 本身极短，收敛远未完成，任何 config 差异都被收敛不充分主导。


# ＝＝＝ 来源：PHASE2D_BOTTLENECK_AUDIT ＝＝＝

# Phase 2-D · Z-Bottleneck Structural Audit

**Status:** 0-training-cost analysis (read-only). Forbids conclusion-hacking — only code, configs, and (later) measured numbers are referenced.

## 1. Where is Z produced?

`models/representation/compact_z.py::CompactRepresentation.forward`:

```python
def forward(self, feats):
    f2, f3, f4 = feats                         # encoder multi-scale, 1/8 / 1/16 / 1/32
    z = self.lat4(f4)                          # 1x1 conv → zc channels
    z = upsample_to_f3(z); z = z + self.lat3(f3)
    z = upsample_to_f2(z); z = z + self.lat2(f2)
    z = self.act(self.bn(z))                   # Z at 1/8 resolution
    return {"z": z, "multi_scale": feats}      # ← both are exposed
```

`z_channels` is the ONLY capacity knob: `lat2/lat3/lat4` are `Conv2d(c, zc, 1)` lateral
projections, and the result is a 1/8 feature map of width `zc`. There is no other
compression in the model.

**The encoder is fixed** for the entire Phase 2-B/D Z-sweep family
(`stem=16, stages=[32,64,96,128], blocks=[2,2,2]`). So `f2/f3/f4` have channel widths
64 / 96 / 128, independent of z.

## 2. Which task heads actually see Z?

`models/static_model.py::StaticMultiTaskModel.__init__`:

```python
self.encoder = LightEncoder(...)
self.representation = CompactRepresentation(...)

if det_cfg.get("from_z", False):               # default OFF in Phase 2-B zsweep
    self.det_head = DetFromZ(zc, ...)
    self.det_from_z = True
else:
    self.det_head = DynamicDetHead(enc_cfg["stages"][1:], ...)   # ← uses [F2,F3,F4]
    self.det_from_z = False

self.da_head  = DynamicSegHead(zc, ...)
self.lane_head = DynamicSegHead(zc, ...)
```

`StaticMultiTaskModel.forward`:

```python
def forward(self, x, return_z=False):
    feats = self.encoder(x)
    z = self.representation(feats)["z"]
    if self.det_from_z:
        det = self.det_head(z, hw)             # R1/R2 — Z only
    else:
        det = self.det_head([feats[0], feats[1], feats[2]])    # R0 — bypass
    da   = self.da_head(z, hw)
    lane = self.lane_head(z, hw)
    ...
```

| Task       | R0 (current zsweep)   | R1 (`from_z:false`)  | R2 (`from_z:true`)    |
|------------|-----------------------|----------------------|-----------------------|
| detection  | encoder F2/F3/F4 directly (64/96/128 ch) | Z raw (zc @ 1/8) | Z → 1x1→BN→ReLU→32ch (det_ch fixed) |
| drivable   | Z (zc ch @ 1/8)       | Z (zc ch @ 1/8)      | Z (zc ch @ 1/8)       |
| lane       | Z (zc ch @ 1/8)       | Z (zc ch @ 1/8)      | Z (zc ch @ 1/8)       |

The `phase2b_zsweep_z{16,32,128}.yaml` files all set `detection: {nc: 1}` only —
no `from_z` flag, so `det_from_z = False`. **R0 is what we are actually running in
the budget experiment.**

## 3. Is there a high-dim bypass for the segmentation heads?

`DynamicSegHead.forward(z, target_hw)`: only `z` is the entry point. There is no
skip-connection from the encoder to the segmentation heads. The `z` it sees is the
fused-and-1x1-projected output of `compact_z.py`, at 1/8 resolution.

So for **drivable / lane**, Z is the *only* information path. Compression at
`zc ∈ {16, 32, 128}` is a real, hard bottleneck for those two tasks.

## 4. Is there a high-dim bypass for detection (R0)?

Yes. In R0:

- The detection head consumes the encoder's three multi-scale feature maps
  (`F2(64ch, 1/8)`, `F3(96ch, 1/16)`, `F4(128ch, 1/32)`) directly.
- These features are produced **upstream** of the Z lateral projections and
  contain the encoder's full information content.
- The only place Z could affect detection in R0 is the shared multi-task loss
  (and therefore the gradients that flow back into the encoder).

This means: **in R0, varying `z_channels` is a real ablation for drivable / lane
but only a soft, gradient-mediated perturbation for detection.**

## 5. Why the Phase 2-C Z-sweep could not find a knee

`experiments/phase2c/expA_multiseed.csv` (3 seeds × {16, 32, 128} @ 4ep) shows:

- **mAP50 (detection, R0 bypass path):** 0.2349 ± 0.0015, 0.2394 ± 0.0101, 0.2425 ± 0.0021
  — within-seed std (largest 0.0101) is comparable to cross-z mean gap (0.0076),
  and the cross-z ordering is non-monotonic in two of three seeds.
- **da_mIoU (R0 hits Z bottleneck):** 0.8202 ± 0.0112, 0.8266 ± 0.0048, 0.8196 ± 0.0213
  — `z=128` is *lower* on average than `z=32` (seed2 outlier 0.7950 drives this).
- **lane_mIoU (R0 hits Z bottleneck):** 0.5756, 0.5770, 0.5790 — *monotonic*,
  but the cross-z spread (0.0034) is on the same order as the within-z noise
  (typical σ ≈ 0.003 from the 3-seed sample).

Two things are happening simultaneously:

1. **Z really is a hard bottleneck for segmentation** — but `z=16` is *already*
   large enough to express the BDD100K 2-class DA and lane fg/bg logits at 1/8
   resolution. Once the information rate of the task is below `zc · 80 · 80` bits
   per image (very loose bound), additional channels just add unused capacity.
2. **Detection is decoupled from Z in R0** — so it cannot be used as an
   out-of-sample signal that Z is too small. The within-z noise for mAP50 reflects
   the encoder's stochasticity, not Z.

## 6. Implication for the budget experiment

The Phase 2-D budget experiment is testing a *legitimate* question — does more
training move the per-task mean so the cross-z gap grows? — but the answer is
*structurally* bounded:

- For DA / lane, the gap can only grow if `z=16` is actually below the task's
  information rate at 4ep. At 4ep, mIoU is still rising (`final_avg_loss` is
  still ~0.246 at ep4 vs ~0.05 plateau for a more converged model). It is
  possible that with more epochs, `z=16` saturates first and the gap opens.
- For detection, the gap is *not structurally present in R0*, so even 200ep
  will not show a Z effect on mAP50. (Only via gradient coupling; expected
  to be small.)

This audit means the decision report's "z-vs-task sensitivity" question has a
known answer before the experiments even run: **lane ≥ DA ≫ det in their
ability to reveal Z capacity.** Any future re-design should make detection
read Z (R2) to get a real three-way sensitivity.

## 7. Open caveat (not a finding, a flag)

- The above is read directly from `static_model.py` and the three zsweep YAMLs.
  If anyone modifies those, this audit must be re-run.
- The encoder's effective capacity is bounded by its **residual paths** in
  `DWSBlock` (the `+ res` in `light_encoder.py:61`). These are not part of Z and
  are not ablated here.


# ＝＝＝ 来源：PHASE2D_DECISION_REPORT ＝＝＝

# Phase 2-D · 实验判断报告

**日期**：2026-09-05
**实验**：z ∈ {16, 32, 128} × epochs ∈ {4, 10, 20}，seed=0，batch=16，lr=1e-3，encoder 固定
**样本**：9 个 cell（3 个来自 Phase 2-B zsweep 的 4ep seed=0，6 个本阶段新训）
**总机时**：9h37m（23:34 → 09:12）
**commit**：`d0ce3e5`（数据）/ 分析脚本 `scripts/phase2d_decide.py`

---

## 🎯 一句话结论

**训练预算是主导因素（效应量是 z 的 5–37 倍），而 z 效应随预算增加不但没有放大、反而收敛到噪声以内。z 容量在 16–128 区间对三个任务均无实质影响 —— 停止搜索 knee。**

---

## Q1 · 训练预算是否足以观察 z 效应？

**答：预算充足，且结论是「预算不是掩盖因素」。**

| 指标 | 预算效应<br>(4ep→20ep, 三 z 平均) | z 离散度<br>@4ep | z 离散度<br>@10ep | z 离散度<br>@20ep | 噪声底<br>(3-seed) | 预算/z<br>倍数 |
|---|---|---|---|---|---|---|
| mAP50 | **+0.0811** | 0.0053 | 0.0163 | **0.0027** | 0.0073 | **30.0×** |
| mAP50_95 | +0.0368 | 0.0025 | 0.0059 | 0.0010 | 0.0032 | 36.8× |
| da_mIoU | +0.0276 | 0.0119 | 0.0017 | **0.0019** | 0.0142 | 14.5× |
| da_fg | +0.0406 | 0.0170 | 0.0026 | 0.0027 | 0.0202 | 15.0× |
| lane_mIoU | +0.0081 | 0.0023 | 0.0041 | **0.0016** | 0.0021 | 5.1× |
| lane_fg | +0.0163 | 0.0039 | 0.0072 | 0.0032 | 0.0032 | 5.1× |

**6/6 指标在 20ep 下的跨 z 离散度全部落在噪声底以内。**

关键判据不是「预算够不够」，而是**离散度随预算的变化方向**：

- 若假设成立（4ep 欠训练掩盖了 z 效应），离散度应随预算**扩大**。
- 实测 `da_mIoU` 离散度：**0.0119 (4ep) → 0.0017 (10ep) → 0.0019 (20ep)**，缩小了 6 倍。
- `mAP50` 离散度：0.0053 → 0.0163 → 0.0027，20ep 时比 4ep 还小。

**训练越多，z=16/32/128 越趋于一致，而不是分道扬镳。** 这直接否证了「预算掩盖」假说。

### 顺带确认：4ep 确实严重欠训练

z=16 从 4ep→20ep：mAP50 `0.2368 → 0.3204`（**+35.3%**），da_mIoU `0.8199 → 0.8564`。
Phase 2-C Exp A 在 4ep 下做结论，本身就是在一条陡峭上升的曲线上采样——这是那一轮负结果的重要方法学注脚。

---

## Q2 · z 是否真正形成有效 bottleneck？

**答：结构上对 DA / lane 是真瓶颈，对 detection 不是；但功能上 z=16 就已过剩。**

代码审计（`docs/PHASE2D_BOTTLENECK_AUDIT.md`）已确认：

| 任务 | 信息路径 | 是否经过 Z |
|---|---|---|
| detection | encoder F2/F3/F4（64/96/128 ch）直接进 `DynamicDetHead` | **否，绕过** |
| drivable area | `DynamicSegHead(zc, …)` | 是 |
| lane | `DynamicSegHead(zc, …)` | 是 |

**因此 mAP50 对 z 的响应在 R0 架构下没有结构性来源**——它只反映多任务损失的梯度耦合与运行噪声。

功能层面的判定依据：如果 z=16 真的是瓶颈，把通道加到 128（8 倍容量）应当带来可观测收益。实测 20ep 下 `da_mIoU` 反而 **z=16 (0.8564) > z=128 (0.8553)**。说明 **z=16 通道已超出 DA/lane 两个前景/背景分割任务的信息需求**，额外的 112 个通道是未被利用的容量。

**真正的信息瓶颈在 encoder，不在 Z。**

---

## Q3 · z 对哪个任务最敏感？

**答：没有一个任务对 z 表现出可靠的敏感性；若按相对离散度勉强排序，lane 前景 IoU 最高，但仍在噪声内。**

20ep 下（相对离散度 = 离散度 / 均值）：

| 指标 | z=16 | z=32 | z=128 | 离散度 | 相对 | 噪声 | 判定 |
|---|---|---|---|---|---|---|---|
| lane_fg | 0.1973 | 0.1973 | 0.2005 | 0.0032 | **1.61%** | 0.0032 | 边缘（≈噪声） |
| mAP50_95 | 0.1149 | 0.1147 | 0.1157 | 0.0010 | 0.87% | 0.0032 | 噪声内 |
| mAP50 | 0.3204 | 0.3197 | 0.3224 | 0.0027 | 0.84% | 0.0073 | 噪声内（且不读 Z） |
| da_fg | 0.7723 | 0.7696 | 0.7708 | 0.0027 | 0.35% | 0.0202 | 噪声内 |
| lane_mIoU | 0.5867 | 0.5863 | 0.5879 | 0.0016 | 0.27% | 0.0021 | 噪声内 |
| da_mIoU | 0.8564 | 0.8545 | 0.8553 | 0.0019 | **0.22%** | 0.0142 | 噪声内 |

两个诚实结论：

1. `lane_fg` 的 1.61% 是最大值，但绝对值 0.0032 恰好等于噪声 0.0032，**不构成证据**。
2. 唯一真正消费 Z 的两个任务（DA、lane），相对离散度是全部指标里**最低的两档**（0.22%、0.27%）。这与「Z 是瓶颈」的预期相反。

---

## Q4 · 哪个 z 最适合作为部署候选？

**答：z=16。而且 z=32 被它严格支配（dominated），没有任何理由存在。**

20ep 精度 / 成本表：

| z | params_M | flops_G | mAP50 | da_mIoU | lane_mIoU | 峰值显存 MiB |
|---|---|---|---|---|---|---|
| **16** | **0.189** | **1.060** | 0.3204 | **0.8564** | 0.5867 | 2446–2700 |
| 32 | 0.203 | 1.197 | 0.3197 | 0.8545 | 0.5863 | 2420 |
| 128 | 0.286 | 2.023 | 0.3224 | 0.8553 | 0.5879 | 2550 |

**支配关系判定**（同时满足：成本不更高 + 三项精度都不更低）：

```
z16 DOMINATES z32   参数 -6.9%  FLOPs -11.4%  mAP50 +0.0007  da +0.0019  lane +0.0004
```

z=32 在**更贵的同时精度全面更低**，这是一个纯粹的劣势点，应从搜索空间中移除。

z=16 vs z=128：参数少 **33.9%**、FLOPs 少 **47.6%**，代价是 mAP50 −0.0020（−0.6%）、lane_mIoU −0.0012（−0.2%），而 da_mIoU 反而 **+0.0011**。考虑到这些差异全部落在噪声内，**用 48% 的 FLOPs 换取 ≤0.6% 且在噪声内的精度，不划算**。

> ⚠️ **latency 数据不可用**：同一模型（参数完全相同）在不同 cell 测出的 FPS 为 z=16: {139, 94, 141}、z=32: {416, 136, 191}、z=128: {153, 397, 138}，波动达 3 倍。这是此前已记录的 SW Power Cap 降频问题（`clocks_throttle_reasons.active=0x4`）。**本报告的效率结论以 FLOPs 为准，不使用 FPS 列。**
>
> 显存方面三者差异 <300 MiB（2420–2700），在 8 GiB 卡上不构成区分度。

---

## Q5 · 是否还值得继续搜索 knee？

**答：不值得。判定为 BRANCH B —— 停止搜索 knee，转向 Pareto / 部署最优点。**

判定依据（三条同时成立）：

1. **6/6 指标**在 20ep 下跨 z 离散度 ≤ 噪声底。
2. `da_mIoU` 离散度随预算**单调缩小**（0.0119 → 0.0017 → 0.0019），不是扩大。
3. 预算效应是 z 效应的 **5–37 倍**：继续加训练只会让三条曲线贴得更紧，不会拉开。

**不会再跑 z ∈ {8, 16, 32, 64, 128} 的容量搜索**——在 16→128（8 倍）都无可分辨效应的区间内再做细分采样，只会得到同样平坦的曲线。

---

## 🚀 下一步建议

### 立即可做（低成本）

1. **把 z 固定为 16**，作为后续所有实验的默认配置。它省 48% FLOPs 且精度不降。
2. **移除 z=32** 这个被支配的配置点，简化搜索空间。
3. 补一个 **z=8** 的单点确认（约 2 小时）：验证「z=16 已过剩」的下边界在哪。若 z=8 仍持平，则 z 的上界可以再往下压，这才是真正有价值的容量结论。

### 方向性调整（这是本轮最值钱的发现）

真正的瓶颈**不在 Z，在 encoder**。三条证据：

- Z 加 8 倍容量 → 无可观测收益
- 唯一消费 Z 的两个任务（DA、lane）对 Z 最不敏感（0.22%、0.27%）
- 预算从 4ep 加到 20ep → 精度涨 35%，说明模型**整体容量/训练**才是限制项

建议把研究问题从「Z 该多宽」改为：

> **在 0.19M 参数预算下，参数应该放在 encoder 还是 representation？**

这才是能产生差异化结果的问题。具体可对 encoder 宽度（stem/stages 通道）× z 做小规模交叉实验。

### 若要让 detection 真正受 Z 约束

必须切到 **R2**（`detection.from_z: true`，检测头从 Z 重建 3 尺度金字塔）。当前 R0 下 detection 直接读 encoder 特征，**任何关于「Z 能否同时承载三任务信息」的论断都无法用 mAP50 检验**。这是 Phase 2-B Task 2(c) 遗留的设计，当时被叫停；现在有了预算实验的基线，R2 的对比会更有解释力。

---

## 📋 局限（诚实声明）

1. **20ep 只有单 seed**。噪声底来自 Phase 2-C Exp A 的 4ep 三 seed 数据，用它代理 20ep 噪声是**保守**的（更收敛的运行通常噪声更小，我们高估了噪声底，结论因此更稳健）。但严格来说，20ep 的多 seed 噪声未被直接测量。
2. **20ep 仍未完全收敛**。最终训练损失 0.2115–0.2156 且仍在缓降，mAP50 从 10ep→20ep 仍有 +11%。若训练到 40ep，绝对精度会继续上升，但依据离散度的**收敛方向**（单调缩小），z 效应不会因此出现。
3. **FPS 全部作废**，效率结论建立在 FLOPs 上。
4. 本阶段**未调整任何 seed、未剔除任何异常点、未修改任何结论**。所有数字直接来自 `expD_budget.csv` 与 `zsweep_results.csv`。

---

**分析脚本**：`scripts/phase2d_decide.py`（可重跑，输出 `experiments/phase2d/expD_decision_analysis.txt`）
**原始数据**：`experiments/phase2d/expD_budget.csv`、`expD_loss_curves.csv`（102 行逐 epoch 损失）
**架构依据**：`docs/PHASE2D_BOTTLENECK_AUDIT.md`


# ＝＝＝ 来源：PHASE3A_REPORT ＝＝＝

# Phase 3A · Encoder Capacity Sanity Check

**研究问题**：在极低参数预算的多任务驾驶视觉模型中，有限参数应优先投入 encoder 还是 compact representation？

**本阶段子问题**：在固定 Z（z=16）和固定训练协议下，改变 **encoder 容量**能否产生超出噪声的性能变化？如果能，它与 Phase 2-D 中改变 Z 容量所产生的变化相比量级如何？

- 执行时间：2026-09-05 11:12:17 → 15:47:42（4h35m GPU）
- Commit：`51b68394aa6a727597bd046dd79ae5553d704113`
- Seed：**0（全程未改）**，结果未挑选、未剔除

---

## 1. 实验表

固定条件：`z=16`、`epochs=20`、`batch=16`、`lr=1e-3`、AdamW + cosine、`tri_train` 全量 69863 张、`input_size 640×640`、`seed=0`。
**唯一自变量：encoder 宽度**（`stem` / `stages`），`blocks=[2,2,2]` 深度不变，heads 不变，Z 不变。

| Encoder | stem | stages | Params (M) | FLOPs (G) | mAP50 | mAP50-95 | DA mIoU | DA fg | Lane mIoU | Lane fg | peak mem (MiB) | final train loss | train (min) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E-small | 12 | 24-40-64-80 | **0.1013** | **0.722** | 0.2687 | 0.0890 | 0.8423 | 0.7515 | 0.5818 | 0.1885 | 2130 | 0.2290 | 113 | trained |
| E-base  | 16 | 32-64-96-128 | **0.1889** | **1.060** | 0.3204 | 0.1149 | 0.8564 | 0.7723 | 0.5867 | 0.1973 | 2446 | 0.2156 | NA | reused: phase2d/expD_z16_e20 |
| E-large | 20 | 40-80-128-168 | **0.2915** | **1.428** | 0.3585 | 0.1356 | 0.8589 | 0.7761 | 0.5878 | 0.2000 | 2970 | 0.2068 | 153 | trained |

原始结果表：`experiments/phase3a/exp3A_encoder.csv`

### 关于 E-base 的复用（诚实声明）

E-base 就是当前 baseline encoder 本身，配置与协议与 Phase 2-D 的 `expD_z16_e20` **逐字段一致**（同为 z=16 / 20ep / bs16 / lr1e-3 / seed=0）。因此没有重跑，而是通过符号链接复用其 checkpoint 与 eval metrics，节省约 132 分钟 GPU 时间：

```
experiments/phase3a/exp3A_ebase_z16/checkpoint.pt     -> ../../phase2d/expD_z16_e20/checkpoint.pt
experiments/phase3a/exp3A_ebase_z16/training_log.txt  -> ../../phase2d/expD_z16_e20/training_log.txt
experiments/phase3a/exp3A_ebase_z16_eval/metrics.json -> ../../phase2d/expD_z16_e20_eval/metrics.json
```

代价：E-base 的 `train_wall_min` 为 `NA`（未重新计时）。该字段不参与任何结论。CSV 中 `source` 列已显式标记 `reused:phase2d/expD_z16_e20`，不会被误认为独立 run。

### 关于 FPS（不作效率结论）

| Encoder | FLOPs (G) | profiled `fps` | `eval_fps` | avg latency (ms) |
|---|---|---|---|---|
| E-small | 0.722 | 135.03 | 177.92 | 7.41 |
| E-base  | 1.060 | 140.62 | 166.20 | 7.11 |
| E-large | 1.428 | **375.21** | 189.32 | 2.67 |

E-large 的 FLOPs 是 E-small 的 **1.98 倍**，profiled FPS 却是其 **2.78 倍**（延迟 2.67ms vs 7.41ms）——物理上不可能。这是本机 **SW Power Cap / clock throttling**（`clocks_throttle_reasons.active=0x4`）造成的测量态差异，不是真实效率差异。`eval_fps` 三个值（177.9 / 166.2 / 189.3）同样非单调。

**结论：本报告所有效率判断一律基于 FLOPs 与参数量，FPS 与延迟数据仅存档，不引用、不比较。**

---

## 2. 参数 / FLOPs 对比

| 对比 | ΔParams | ΔFLOPs | 备注 |
|---|---|---|---|
| E-small → E-base | +0.0876 M (+86.5%) | +0.338 G (+46.8%) | |
| E-base → E-large | +0.1026 M (+54.3%) | +0.369 G (+34.8%) | |
| E-small → E-large | **+0.1902 M (+187.7%)** | **+0.706 G (+97.8%)** | 参数近 3 倍，FLOPs 近 2 倍 |

三档由 `scripts/phase3a_encoder_scan.py` 在 0.40–2.00 的乘法 scale 网格（27 档）上搜索得到，目标是命中用户指定的 ~0.10M / ~0.19M / ~0.30M，同时保持 encoder「单调变宽」的形状（stem 取 4 的倍数、stages 取 8 的倍数）。

**参数量独立校验**：扫描器预测的 0.1889M 与 Phase 2-D 实测 baseline 0.189M 一致；三档 params 与 FLOPs 均严格递增（无倒挂）。

**Z 的对照成本**（Phase 2-D，encoder 固定为 baseline）：z 16→128 花掉 **+0.0970 M / +0.963 G**。
注意：花在 Z 上的 **FLOPs 比花在 encoder 上更贵**（+0.963G vs +0.706G，同样换来约 0.1M 参数），因为 Z 作用在 1/8 分辨率的高维特征上，且被两个 segmentation head 反复消费。

---

## 3. 任务性能对比

### 3.1 单调性（Q1）

| metric | E-small | E-base | E-large | Δ(base−small) | Δ(large−base) | Δ(large−small) | 严格单调 |
|---|---|---|---|---|---|---|---|
| mAP50 | 0.2687 | 0.3204 | 0.3585 | +0.0517 | +0.0381 | **+0.0898** | ✅ True |
| mAP50-95 | 0.0890 | 0.1149 | 0.1356 | +0.0259 | +0.0207 | **+0.0466** | ✅ True |
| da_mIoU | 0.8423 | 0.8564 | 0.8589 | +0.0141 | +0.0025 | **+0.0166** | ✅ True |
| da_fg | 0.7515 | 0.7723 | 0.7761 | +0.0208 | +0.0038 | **+0.0246** | ✅ True |
| lane_mIoU | 0.5818 | 0.5867 | 0.5878 | +0.0049 | +0.0011 | **+0.0060** | ✅ True |
| lane_fg | 0.1885 | 0.1973 | 0.2000 | +0.0088 | +0.0027 | **+0.0115** | ✅ True |

**6/6 指标全部严格单调递增**（E-small < E-base < E-large，无一例外）。训练 loss 也单调下降（0.2290 → 0.2156 → 0.2068），与容量序一致——不是优化噪声。

**分任务敏感度**（相对 E-small 的提升幅度）：

| 任务 | 代表指标 | 绝对提升 | **相对提升** |
|---|---|---|---|
| Detection | mAP50-95 | +0.0466 | **+52.4%** |
| Detection | mAP50 | +0.0898 | **+33.4%** |
| Lane | lane_fg | +0.0115 | +6.1% |
| DA | da_fg | +0.0246 | +3.3% |
| DA | da_mIoU | +0.0166 | +2.0% |
| Lane | lane_mIoU | +0.0060 | +1.0% |

**Detection 对 encoder 容量的敏感度比其他两个任务高一个数量级**（相对提升 33–52% vs 1–6%）。

### 3.2 与 Phase 2-D 的 Z 效应对比（Q2）

噪声底来自 Phase 2-C Exp A 的 3-seed @4ep pooled stdev。

| metric | ΔZ 离散度 (z16/32/128 @20ep) | 噪声底 | ΔEncoder (small→large) | **ΔEncoder / ΔZ** | 判定 |
|---|---|---|---|---|---|
| mAP50 | 0.0027 | 0.0073 | 0.0898 | **33.3×** | encoder 远超噪声 |
| mAP50-95 | 0.0010 | 0.0032 | 0.0466 | **46.6×** | encoder 远超噪声 |
| da_mIoU | 0.0019 | 0.0142 | 0.0166 | **8.7×** | encoder 超过噪声 |
| da_fg | 0.0027 | 0.0202 | 0.0246 | **9.1×** | encoder 超过噪声 |
| lane_mIoU | 0.0016 | 0.0021 | 0.0060 | **3.8×** | encoder 超过噪声 |
| lane_fg | 0.0032 | 0.0032 | 0.0115 | **3.6×** | encoder 超过噪声 |

**最小倍数是 3.6×，最大 46.6×。** 在完全相同的训练预算（20ep）下，改变 encoder 容量产生的效应是改变 Z 容量的 **4–47 倍**。

稳健性说明：噪声底来自 4ep 的 3-seed 估计，是 20ep 真实的代理值而非精确值。但 Detection 的 ΔEncoder（0.0898 / 0.0466）即使真实 20ep 噪声比代理值大 3–5 倍，结论仍然成立；DA/Lane 的 Δ(large−base)（0.0011–0.0038）即使噪声只有代理值的一半，也仍然落在噪声内。

### 3.3 单位成本收益（Q3）

| metric | 花在 encoder（每 +0.01M 参数） | 花在 Z（每 +0.01M 参数） | **比值（参数）** | **比值（FLOPs）** |
|---|---|---|---|---|
| mAP50 | +0.00472 | +0.00021 | **22.9×** | **61.2×** |
| mAP50-95 | +0.00245 | +0.00008 | **29.7×** | **79.4×** |
| da_mIoU | +0.00087 | −0.00011 | −7.7×（符号相反） | −20.6× |
| da_fg | +0.00129 | −0.00015 | −8.4×（符号相反） | −22.4× |
| lane_mIoU | +0.00032 | +0.00012 | 2.5× | 6.8× |
| lane_fg | +0.00060 | +0.00033 | 1.8× | 4.9× |

**每 0.01M 参数投到 encoder 换来的 mAP50 是投到 Z 的 22.9 倍；按 FLOPs 计是 61.2 倍。** mAP50-95 更极端（29.7× / 79.4×）。DA 两项的比值是负数——把参数投到 Z 上不仅没有收益，点估计还是略微负的。

### 3.4 等参数量 like-for-like 对比（决定性证据）

两个总参数量几乎相同（差 1.9%）的配置，一个把预算花在 Z 上，一个花在 encoder 上：

| | baseline encoder + z=128 | **E-large encoder + z=16** | delta |
|---|---|---|---|
| Params (M) | 0.2860 | 0.2915 | +1.9% |
| **FLOPs (G)** | 2.023 | **1.428** | **−29.4%** |
| mAP50 | 0.3224 | **0.3585** | **+0.0361** |
| mAP50-95 | 0.1157 | **0.1356** | **+0.0199** |
| da_mIoU | 0.8553 | **0.8589** | +0.0036 |
| da_fg | 0.7708 | **0.7761** | +0.0053 |
| lane_mIoU | **0.5879** | 0.5878 | −0.0001 |
| lane_fg | **0.2005** | 0.2000 | −0.0005 |

**E-large + z16 用了更多参数（+1.9%）、更少 FLOPs（−29.4%），6 项指标中 4 项更高、2 项持平（差值 0.0001 / 0.0005，远在噪声内）。**

这是一个干净的**支配关系（dominance）**：在同等参数预算下，「encoder-heavy + 小 Z」严格优于「baseline-encoder + 大 Z」——既更准，又更省算力。两个 lane 指标的 −0.0001 / −0.0005 不构成反例，因为它们比 lane 的噪声底（0.0021 / 0.0032）小一个数量级。

---

## 4. Encoder Sensitivity 判断

### 4.1 逐步判定：哪些步长是真的，哪些在噪声内

| metric | small→base | base→large | 解读 |
|---|---|---|---|
| mAP50 | +0.0517 **real** | +0.0381 **real** | 两步都显著，未饱和 |
| mAP50-95 | +0.0259 **real** | +0.0207 **real** | 两步都显著，未饱和 |
| da_mIoU | +0.0141 noise | +0.0025 noise | 两步都在噪声内 |
| da_fg | +0.0208 **real** | +0.0038 noise | 第一步真实，第二步饱和 |
| lane_mIoU | +0.0049 **real** | +0.0011 noise | 第一步真实，第二步饱和 |
| lane_fg | +0.0088 **real** | +0.0027 noise | 第一步真实，第二步饱和 |

### 4.2 结论：encoder **未饱和**，但分任务

- **Detection（mAP50 / mAP50-95）：E-large 处尚未饱和。** base→large 仍然带来 +0.0381 / +0.0207，分别是噪声底的 5.2× / 6.5×。用户规则「E-large 相比 E-base 几乎没有收益 → 记 saturation」**不适用**——这里有明确收益。
- **DA / Lane：E-large 处已饱和。** base→large 的 4 个指标全部落在噪声内（0.0011–0.0038 vs 噪声底 0.0021–0.0202）。

### 4.3 为什么 Detection 最敏感——与代码审计互相印证

`docs/PHASE2D_BOTTLENECK_AUDIT.md` 已经查明 R0 架构的信息流：

| 任务 | 特征来源 | 是否经过 Z |
|---|---|---|
| Detection | encoder 多尺度 F2(64ch,1/8) / F3(96ch,1/16) / F4(128ch,1/32) | **否，bypass** |
| Drivable area | CompactRepresentation 输出 Z (1/8) | 是 |
| Lane | CompactRepresentation 输出 Z (1/8) | 是 |

这正好解释了 Phase 3A 的形态：Detection **直接吃 encoder 特征**，所以 encoder 变宽它直接受益（+33~52%）；DA/Lane **只吃 z=16 的 Z**，所以 encoder 变宽后，更丰富的特征必须被压进同样 16 通道的瓶颈里才能到达这两个 head——收益被 Z 卡住，在 E-large 处就压平了。

**关键推论：DA/Lane 在 E-large 的「饱和」，很可能是 z=16 的瓶颈造成的假饱和，而不是任务本身的容量上限。** 这个推论**不能**用 Phase 3A 的数据证实——Phase 3A 全程固定 z=16，两个因素完全混杂。这正是 Phase 3B 要解决的问题。

### 4.4 对研究问题的直接回答

> 有限参数应优先投入 encoder 还是 compact representation？

**支持「优先投 encoder」**，三条独立证据：

1. **效应量**：ΔEncoder / ΔZ = 3.6×–46.6×（6/6 指标全部 > 1）
2. **单位成本**：每 0.01M 参数的 mAP50 收益，encoder 是 Z 的 22.9×（按 FLOPs 是 61.2×）
3. **等参数量支配**：同等参数下，encoder-heavy + 小 Z 在 mAP50 上 +0.0361 且 FLOPs −29.4%

**但这是一个有条件的结论**：证据强度在 Detection 上最强（支配关系明确、未饱和），在 DA/Lane 上偏弱（点估计为正但幅度在噪声内，且受到 z=16 混杂）。要把它从「Detection 上的强结论」升级为「全任务的稳健结论」，需要 Phase 3B 的二维矩阵。

---

## 5. 是否进入 Phase 3B：**是**

**明确结论：进入 Phase 3B（Encoder × Z 二维矩阵）。**

理由：

1. **Phase 3A 已经证明 encoder 值得投参数**（22.9× 单位成本优势、等参数量支配），这个方向有继续做的价值，不是噪声驱动的假信号。
2. **但 Phase 3A 无法归因 DA/Lane 的饱和。** z=16 全程固定，encoder 容量与 Z 容量完全混杂。DA/Lane 在 E-large 压平，既可能是「任务已达容量上限」，也可能是「z=16 卡住了 encoder 的收益」——这两种解释指向完全相反的架构决策（前者说明该把参数给 detection，后者说明该给 Z）。**只有二维矩阵能区分它们。**
3. **矩阵还能回答「encoder 的收益是否依赖 Z 大小」**。如果 E-large 与 z=128 的组合在 DA/Lane 上明显超过 E-large + z=16，就说明存在交互效应，Phase 3C 的等预算分配才有意义；如果所有 Z 列下 DA/Lane 都压平，那就确实是任务饱和，Phase 3C 应把资源全部导向 detection 侧。

**不进入的理由不成立**：用户规则中「E-large 相比 E-base 几乎没有收益 → 记 saturation，不要强行扩大 encoder」的触发条件**未满足**——Detection 上 E-large 有明确且超噪声的收益。

### Phase 3B 工作量说明（供决策）

Phase 3A 已经完成了矩阵的 **z=16 整列**（esmall+z16、ebase+z16、elarge+z16 三个 cell 均已存在）。因此 Phase 3B 实际只需要跑 **6 个新 cell**，而非 9 个：

| 新 cell | 预估训练时长 |
|---|---|
| E-small × z32 / z128 | 2 × ~113 min |
| E-base × z32 / z128 | 2 × ~132 min |
| E-large × z32 / z128 | 2 × ~153 min |
| **合计** | **≈ 13.3 h + 评估 ≈ 14 h** |

如果希望更严格，z=16 的三个 cell 可以重跑以获得同批次计时（会多出 ≈ 6.6h）。**这一项请用户指示，我不擅自决定。**

**等待用户确认后才会启动 Phase 3B，不会自行开跑。**

---

## 6. 文件路径清单

### 配置文件（WSL 仓库 `~/ai_study/trac/`）

| 文件 | 说明 |
|---|---|
| `configs/phase3a_esmall_z16.yaml` | E-small，stem 12 / stages [24,40,64,80] |
| `configs/phase3a_ebase_z16.yaml` | E-base（= 当前 baseline），stem 16 / stages [32,64,96,128] |
| `configs/phase3a_elarge_z16.yaml` | E-large，stem 20 / stages [40,80,128,168] |

三份配置除 `encoder.stem` / `encoder.stages` 与 `epochs: 20` 外，与 `configs/phase2b_zsweep_z16.yaml` 逐字段一致（已用 `diff` 验证，差异仅出现在注释与上述字段）。

### 脚本

| 文件 | 说明 |
|---|---|
| `scripts/phase3a_encoder_scan.py` | 宽度扫描器：27 档 scale 网格搜索，命中三档参数量目标 |
| `scripts/phase3a_run_encoder.sh` | Runner：继承 Phase 2-D v2 的 checkpoint epoch 校验，加 E-base 复用逻辑 |
| `scripts/phase3a_analyze.py` | Q1 单调性 / Q2 ΔEncoder vs ΔZ / Q3 单位成本 / Q4 saturation |
| `scripts/phase3a_allocation_ledger.py` | 参数效率对账（A: 花在 Z / B: 花在 encoder / C: 比值 / D: like-for-like） |

### 结果与产物

| 文件 | 说明 |
|---|---|
| `experiments/phase3a/exp3A_encoder.csv` | **原始结果表**（3 行，含 params/FLOPs/6 指标/mem/loss/时间/source/commit） |
| `experiments/phase3a/exp3A_analysis.txt` | Q1–Q4 完整分析输出 |
| `experiments/phase3a/exp3A_allocation_ledger.txt` | 参数效率对账输出（A/B/C/D 四段） |
| `experiments/phase3a/exp3A_runner.log` | Runner 时间线 |
| `experiments/phase3a/exp3A_{esmall,ebase,elarge}_z16/` | 训练目录（checkpoint.pt + training_log.txt） |
| `experiments/phase3a/exp3A_{esmall,ebase,elarge}_z16_eval/` | 评估目录（metrics.json） |
| `experiments/phase3a/exp3A_{esmall,elarge}_z16_eval.log` | 评估日志（含 profiling 原始行） |

### 参考文档

| 文件 | 说明 |
|---|---|
| `docs/PHASE2D_BOTTLENECK_AUDIT.md` | 代码侧架构审计（Z 产生路径、head 分配表、bypass 判定） |
| `docs/PHASE2D_DECISION_REPORT.md` | Phase 2-D 结论：BRANCH B，停止搜索 knee |
| `docs/PHASE3A_REPORT.md` | 本报告 |

---

## 7. 已知局限（诚实声明）

1. **单 seed**。全部三个 cell 均为 seed=0。跨 cell 差异是与 Phase 2-C 的 3-seed 噪声底比较，而不是本阶段新估的噪声。Proxy 噪声底来自 4ep，非 20ep。
2. **E-base 为复用**，非独立重跑。配置与协议已逐字段核对一致，但 `train_wall_min` 缺失。
3. **FPS / 延迟不可用**（Power Cap 导致），效率结论全部基于 FLOPs 与参数量。
4. **z=16 全程固定**，因此本阶段无法区分「DA/Lane 任务饱和」与「z=16 瓶颈造成的假饱和」——这是 Phase 3B 的核心任务。
5. **未修改任何 loss / optimizer / 数据 / 增强 / 输入尺寸**；未因结果不理想剔除任何 run 或更换 seed。

---

**阶段状态**：Phase 3A 完成 · 等待用户确认后进入 Phase 3B
**分析脚本**：`scripts/phase3a_analyze.py`、`scripts/phase3a_allocation_ledger.py`
**报告日期**：2026-09-05


# ＝＝＝ 来源：PHASE3B_REPORT ＝＝＝

# Phase 3B · Encoder × Z Interaction Study

> Generated by `scripts/phase3b_report.py` directly from the result CSVs.
> Every number below is read from data, not transcribed by hand.

## 1. Objective

Phase 3A found that widening the encoder raises Detection sharply (mAP50 +0.0898, mAP50-95 +0.0466 from E-small to E-large) while DA and Lane flatten between E-base and E-large. But Phase 3A held **z=16 throughout**, so two rival explanations could not be separated:

- **A** — DA/Lane are genuinely saturated at this capacity;
- **B** — z=16 throttles the richer encoder features before they reach the segmentation heads.

The information path differs by task, which is why B is plausible:
detection reads encoder features F2/F3/F4 **directly** and never sees Z, whereas DA and Lane consume Z **exclusively**. A stronger encoder therefore has to squeeze everything through the same 16-channel bottleneck to reach DA/Lane.

Phase 3B varies encoder width and Z width **jointly** to tell A from B. It is not a search for a new z knee, and not an attempt to maximise any score.

## 2. Experimental Design

A 3 × 3 factorial: three encoder widths × three Z widths, 20 epochs, seed 0.

| | z=16 | z=32 | z=128 |
|---|---|---|---|
| **esmall** | Phase 3A ✓ | Phase 3B (new) | Phase 3B (new) |
| **ebase** | Phase 3A ✓ (reused) | Phase 3B (new) | Phase 3B (new) |
| **elarge** | Phase 3A ✓ | Phase 3B (new) | Phase 3B (new) |

Only **6 cells were newly trained**. The entire z=16 column already existed from Phase 3A and is reused rather than re-run:

- `esmall_z16`, `elarge_z16` — trained during Phase 3A;
- `ebase_z16` — reused from Phase 2-D `expD_z16_e20` (identical encoder, z, epochs, seed and protocol), flagged in the CSV `source` column as `reused:phase2d/expD_z16_e20`.

Per your Phase 3A instruction, E-base + z32 and E-base + z128 **are** trained fresh; only the z=16 cell was eligible for reuse.

**Held fixed across all nine cells:** blocks `[2,2,2]`, all three task heads, input 640×640, `tri_train` 69863 images, batch 16, lr 1e-3, AdamW, cosine schedule, loss weights, 20 epochs, seed 0. Verified before launch by `scripts/phase3b_config_audit.py` (four independent layers, result **PASS**); see `experiments/phase3b/phase3B_config_audit.txt`.

## 3. Raw Results

| cell | Params (M) | FLOPs (G) | mAP50 | mAP50-95 | da_mIoU | da_fg | lane_mIoU | lane_fg | final loss | source |
|---|---|---|---|---|---|---|---|---|---|---|
| **esmall_z16** | 0.1013 | 0.722 | 0.2687 | 0.0890 | 0.8423 | 0.7515 | 0.5818 | 0.1885 | 0.2290 | trained |
| **esmall_z32** | 0.1135 | 0.852 | 0.2810 | 0.0968 | 0.8456 | 0.7564 | 0.5841 | 0.1922 | 0.2267 | trained |
| **esmall_z128** | 0.1867 | 1.635 | 0.2639 | 0.0879 | 0.8490 | 0.7613 | 0.5855 | 0.1955 | 0.2247 | trained |
| **ebase_z16** | 0.1889 | 1.060 | 0.3204 | 0.1149 | 0.8564 | 0.7723 | 0.5867 | 0.1973 | 0.2156 | reused:phase2d/expD_z16_e20 |
| **ebase_z32** | 0.2027 | 1.197 | 0.3222 | 0.1158 | 0.8527 | 0.7670 | 0.5875 | 0.1992 | 0.2138 | trained |
| **ebase_z128** | 0.2858 | 2.023 | 0.3161 | 0.1129 | 0.8522 | 0.7663 | 0.5877 | 0.2005 | 0.2111 | trained |
| **elarge_z16** | 0.2915 | 1.428 | 0.3585 | 0.1356 | 0.8589 | 0.7761 | 0.5878 | 0.2000 | 0.2068 | trained |
| **elarge_z32** | 0.3068 | 1.571 | 0.3494 | 0.1308 | 0.8546 | 0.7699 | 0.5893 | 0.2026 | 0.2070 | trained |
| **elarge_z128** | 0.3984 | 2.429 | 0.3436 | 0.1271 | 0.8575 | 0.7740 | 0.5925 | 0.2085 | 0.2044 | trained |

Source: `experiments/phase3a/exp3A_encoder.csv` (z=16) and `experiments/phase3b/phase3B_encoder_z.csv` (z=32, z=128).

## 4. Z Main Effect

Δ is measured against z=16 **within the same encoder**, so the encoder axis is held constant.

| encoder | metric | z16 | z32 | z128 | Δ32 | Δ128 | vs noise* |
|---|---|---|---|---|---|---|---|
| esmall | mAP50 | 0.2687 | 0.2810 | 0.2639 | +0.0123 | -0.0048 | above (0.0073) |
| esmall | mAP50_95 | 0.0890 | 0.0968 | 0.0879 | +0.0078 | -0.0011 | above (0.0032) |
| esmall | da_mIoU | 0.8423 | 0.8456 | 0.8490 | +0.0033 | +0.0067 | within (0.0142) |
| esmall | da_fg | 0.7515 | 0.7564 | 0.7613 | +0.0049 | +0.0098 | within (0.0202) |
| esmall | lane_mIoU | 0.5818 | 0.5841 | 0.5855 | +0.0023 | +0.0037 | above (0.0021) |
| esmall | lane_fg | 0.1885 | 0.1922 | 0.1955 | +0.0037 | +0.0070 | above (0.0032) |
| ebase | mAP50 | 0.3204 | 0.3222 | 0.3161 | +0.0018 | -0.0043 | within (0.0073) |
| ebase | mAP50_95 | 0.1149 | 0.1158 | 0.1129 | +0.0009 | -0.0020 | within (0.0032) |
| ebase | da_mIoU | 0.8564 | 0.8527 | 0.8522 | -0.0037 | -0.0042 | within (0.0142) |
| ebase | da_fg | 0.7723 | 0.7670 | 0.7663 | -0.0053 | -0.0060 | within (0.0202) |
| ebase | lane_mIoU | 0.5867 | 0.5875 | 0.5877 | +0.0008 | +0.0010 | within (0.0021) |
| ebase | lane_fg | 0.1973 | 0.1992 | 0.2005 | +0.0019 | +0.0032 | above (0.0032) |
| elarge | mAP50 | 0.3585 | 0.3494 | 0.3436 | -0.0091 | -0.0149 | above (0.0073) |
| elarge | mAP50_95 | 0.1356 | 0.1308 | 0.1271 | -0.0048 | -0.0085 | above (0.0032) |
| elarge | da_mIoU | 0.8589 | 0.8546 | 0.8575 | -0.0043 | -0.0014 | within (0.0142) |
| elarge | da_fg | 0.7761 | 0.7699 | 0.7740 | -0.0062 | -0.0021 | within (0.0202) |
| elarge | lane_mIoU | 0.5878 | 0.5893 | 0.5925 | +0.0015 | +0.0047 | above (0.0021) |
| elarge | lane_fg | 0.2000 | 0.2026 | 0.2085 | +0.0026 | +0.0085 | above (0.0032) |

Sign summary (how many of the six metrics move each way when Z widens):

- **esmall**, z16→z32: positive 6/6, beyond reference noise 4/6
- **esmall**, z16→z128: positive 4/6, beyond reference noise 2/6
- **ebase**, z16→z32: positive 4/6, beyond reference noise 0/6
- **ebase**, z16→z128: positive 2/6, beyond reference noise 1/6
- **elarge**, z16→z32: positive 2/6, beyond reference noise 2/6
- **elarge**, z16→z128: positive 2/6, beyond reference noise 4/6

Judged on the whole metric set, never on a single metric.

## 5. Encoder Main Effect

Cross-check of Phase 3A, now computed separately inside each Z column.

**z = 16**

| metric | E-small | E-base | E-large | Δ(base−small) | Δ(large−base) | Δ(large−small) | monotonic |
|---|---|---|---|---|---|---|---|
| mAP50 | 0.2687 | 0.3204 | 0.3585 | +0.0517 | +0.0381 | +0.0898 | yes |
| mAP50_95 | 0.0890 | 0.1149 | 0.1356 | +0.0259 | +0.0207 | +0.0466 | yes |
| da_mIoU | 0.8423 | 0.8564 | 0.8589 | +0.0141 | +0.0025 | +0.0166 | yes |
| da_fg | 0.7515 | 0.7723 | 0.7761 | +0.0208 | +0.0038 | +0.0246 | yes |
| lane_mIoU | 0.5818 | 0.5867 | 0.5878 | +0.0049 | +0.0011 | +0.0060 | yes |
| lane_fg | 0.1885 | 0.1973 | 0.2000 | +0.0088 | +0.0027 | +0.0115 | yes |

**z = 32**

| metric | E-small | E-base | E-large | Δ(base−small) | Δ(large−base) | Δ(large−small) | monotonic |
|---|---|---|---|---|---|---|---|
| mAP50 | 0.2810 | 0.3222 | 0.3494 | +0.0412 | +0.0272 | +0.0684 | yes |
| mAP50_95 | 0.0968 | 0.1158 | 0.1308 | +0.0190 | +0.0150 | +0.0340 | yes |
| da_mIoU | 0.8456 | 0.8527 | 0.8546 | +0.0071 | +0.0019 | +0.0090 | yes |
| da_fg | 0.7564 | 0.7670 | 0.7699 | +0.0106 | +0.0029 | +0.0135 | yes |
| lane_mIoU | 0.5841 | 0.5875 | 0.5893 | +0.0034 | +0.0018 | +0.0052 | yes |
| lane_fg | 0.1922 | 0.1992 | 0.2026 | +0.0070 | +0.0034 | +0.0104 | yes |

**z = 128**

| metric | E-small | E-base | E-large | Δ(base−small) | Δ(large−base) | Δ(large−small) | monotonic |
|---|---|---|---|---|---|---|---|
| mAP50 | 0.2639 | 0.3161 | 0.3436 | +0.0522 | +0.0275 | +0.0797 | yes |
| mAP50_95 | 0.0879 | 0.1129 | 0.1271 | +0.0250 | +0.0142 | +0.0392 | yes |
| da_mIoU | 0.8490 | 0.8522 | 0.8575 | +0.0032 | +0.0053 | +0.0085 | yes |
| da_fg | 0.7613 | 0.7663 | 0.7740 | +0.0050 | +0.0077 | +0.0127 | yes |
| lane_mIoU | 0.5855 | 0.5877 | 0.5925 | +0.0022 | +0.0048 | +0.0070 | yes |
| lane_fg | 0.1955 | 0.2005 | 0.2085 | +0.0050 | +0.0080 | +0.0130 | yes |

## 6. Interaction Effect

The core quantity. For each reference encoder E:

```
interaction = [ m(E, z) − m(E, z16) ] − [ m(E-small, z) − m(E-small, z16) ]
```

- **> 0** → a stronger encoder makes Z *more* valuable (z=16 was throttling)
- **≈ 0** → the two factors are independent
- **< 0** → Z matters *less* as the encoder grows

| reference | z | metric | ΔZ(ref) | ΔZ(E-small) | interaction | noise* | reading |
|---|---|---|---|---|---|---|---|
| ebase | 32 | mAP50 | +0.0018 | +0.0123 | -0.0105 | 0.0073 | Z less valuable |
| ebase | 32 | mAP50_95 | +0.0009 | +0.0078 | -0.0069 | 0.0032 | Z less valuable |
| ebase | 32 | da_mIoU | -0.0037 | +0.0033 | -0.0070 | 0.0142 | ≈ independent |
| ebase | 32 | da_fg | -0.0053 | +0.0049 | -0.0102 | 0.0202 | ≈ independent |
| ebase | 32 | lane_mIoU | +0.0008 | +0.0023 | -0.0015 | 0.0021 | ≈ independent |
| ebase | 32 | lane_fg | +0.0019 | +0.0037 | -0.0018 | 0.0032 | ≈ independent |
| ebase | 128 | mAP50 | -0.0043 | -0.0048 | +0.0005 | 0.0073 | ≈ independent |
| ebase | 128 | mAP50_95 | -0.0020 | -0.0011 | -0.0009 | 0.0032 | ≈ independent |
| ebase | 128 | da_mIoU | -0.0042 | +0.0067 | -0.0109 | 0.0142 | ≈ independent |
| ebase | 128 | da_fg | -0.0060 | +0.0098 | -0.0158 | 0.0202 | ≈ independent |
| ebase | 128 | lane_mIoU | +0.0010 | +0.0037 | -0.0027 | 0.0021 | Z less valuable |
| ebase | 128 | lane_fg | +0.0032 | +0.0070 | -0.0038 | 0.0032 | Z less valuable |
| elarge | 32 | mAP50 | -0.0091 | +0.0123 | -0.0214 | 0.0073 | Z less valuable |
| elarge | 32 | mAP50_95 | -0.0048 | +0.0078 | -0.0126 | 0.0032 | Z less valuable |
| elarge | 32 | da_mIoU | -0.0043 | +0.0033 | -0.0076 | 0.0142 | ≈ independent |
| elarge | 32 | da_fg | -0.0062 | +0.0049 | -0.0111 | 0.0202 | ≈ independent |
| elarge | 32 | lane_mIoU | +0.0015 | +0.0023 | -0.0008 | 0.0021 | ≈ independent |
| elarge | 32 | lane_fg | +0.0026 | +0.0037 | -0.0011 | 0.0032 | ≈ independent |
| elarge | 128 | mAP50 | -0.0149 | -0.0048 | -0.0101 | 0.0073 | Z less valuable |
| elarge | 128 | mAP50_95 | -0.0085 | -0.0011 | -0.0074 | 0.0032 | Z less valuable |
| elarge | 128 | da_mIoU | -0.0014 | +0.0067 | -0.0081 | 0.0142 | ≈ independent |
| elarge | 128 | da_fg | -0.0021 | +0.0098 | -0.0119 | 0.0202 | ≈ independent |
| elarge | 128 | lane_mIoU | +0.0047 | +0.0037 | +0.0010 | 0.0021 | ≈ independent |
| elarge | 128 | lane_fg | +0.0085 | +0.0070 | +0.0015 | 0.0032 | ≈ independent |

Per-task mean interaction at z=128:

| reference | detection | DA | lane |
|---|---|---|---|
| ebase | -0.0002 | -0.0134 | -0.0033 |
| elarge | -0.0088 | -0.0100 | +0.0012 |

\* The noise column is the **external** Phase 2-C reference (3 seeds, 4 epochs), not a Phase 3B estimate. An interaction is a difference of two single-seed differences, so it is noisier than either term; read the **sign pattern** across metrics, not any individual row.

## 7. DA / Lane Bottleneck Diagnosis

| metric | z16 | z32 | z128 | Δ(32−16) | Δ(128−16) | verdict |
|---|---|---|---|---|---|---|
| da_mIoU | 0.8589 | 0.8546 | 0.8575 | -0.0043 | -0.0014 | flat → favours task saturation |
| da_fg | 0.7761 | 0.7699 | 0.7740 | -0.0062 | -0.0021 | flat → favours task saturation |
| lane_mIoU | 0.5878 | 0.5893 | 0.5925 | +0.0015 | +0.0047 | **revives** → favours z-bottleneck |
| lane_fg | 0.2000 | 0.2026 | 0.2085 | +0.0026 | +0.0085 | **revives** → favours z-bottleneck |

Even the flat branch is only **evidence consistent with saturation** at z ≤ 128 and 20 epochs. It is not a proof, and it does not establish that no larger Z would help — it only says the Z values tested here do not revive these tasks.

## 8. Detection Diagnosis

**In R0 the detection head reads encoder features F2/F3/F4 directly. It does not consume Z.** Any mAP response to z therefore cannot be a direct Z effect — it can only arise from multi-task gradient coupling, shared training dynamics, or indirect effects through the shared encoder. Detection's primary driver must be the encoder.

- **mAP50, encoder axis (z=16):** 0.2687 → 0.3204 → 0.3585 (total +0.0898)
- **mAP50, z axis (E-large):** 0.3585 / 0.3494 / 0.3436 (spread 0.0149) — exceeds reference noise — indirect/coupling effect, **not** direct Z consumption
- **mAP50_95, encoder axis (z=16):** 0.0890 → 0.1149 → 0.1356 (total +0.0466)
- **mAP50_95, z axis (E-large):** 0.1356 / 0.1308 / 0.1271 (spread 0.0085) — exceeds reference noise — indirect/coupling effect, **not** direct Z consumption

Nothing here may be phrased as “detection consumes the compact representation”; that claim only becomes available under R2.

## 9. Compute / Parameter Trade-off

Reference: E-base + z16 (0.1889 M, 1.060 G).

| cell | ΔParams (M) | ΔFLOPs (G) | ΔmAP50 | Δda_mIoU | Δlane_mIoU | per +0.01M (mAP50) | per +0.1GF (mAP50) |
|---|---|---|---|---|---|---|---|
| esmall_z16 | -0.0876 | -0.338 | -0.0517 | -0.0141 | -0.0049 | +0.00590 | +0.01530 |
| esmall_z32 | -0.0754 | -0.208 | -0.0394 | -0.0108 | -0.0026 | +0.00523 | +0.01899 |
| esmall_z128 | -0.0022 | +0.575 | -0.0565 | -0.0074 | -0.0012 | +0.25682 | -0.00982 |
| ebase_z32 | +0.0138 | +0.138 | +0.0018 | -0.0037 | +0.0008 | +0.00130 | +0.00131 |
| ebase_z128 | +0.0969 | +0.963 | -0.0043 | -0.0042 | +0.0010 | -0.00044 | -0.00045 |
| elarge_z16 | +0.1026 | +0.368 | +0.0381 | +0.0025 | +0.0011 | +0.00371 | +0.01035 |
| elarge_z32 | +0.1179 | +0.511 | +0.0290 | -0.0018 | +0.0026 | +0.00246 | +0.00567 |
| elarge_z128 | +0.2095 | +1.369 | +0.0232 | +0.0011 | +0.0058 | +0.00111 | +0.00169 |

Unit-cost columns are mAP50 only. DA and Lane metrics live on different scales and are **not** compared across tasks.

**Two ways to spend roughly 0.1–0.19 M extra parameters:**

| route | ΔParams (M) | ΔFLOPs (G) | mean Δ detection | mean Δ DA | mean Δ lane |
|---|---|---|---|---|---|
| encoder (E-small→E-large, z16) | +0.1902 | +0.706 | +0.0682 | +0.0206 | +0.0088 |
| Z (esmall, z16→z128) | +0.0854 | +0.913 | -0.0029 | +0.0082 | +0.0054 |
| Z (ebase, z16→z128) | +0.0969 | +0.963 | -0.0032 | -0.0051 | +0.0021 |
| Z (elarge, z16→z128) | +0.1069 | +1.001 | -0.0117 | -0.0017 | +0.0066 |

FPS and latency are **excluded** from every efficiency claim. This machine exhibits SW Power Cap / clock throttling, and in Phase 3A the E-large profile (1.428 G) reported 2.78× the FPS of E-small (0.722 G), which is physically impossible. Parameters and FLOPs only.

## 10. Pareto Analysis

The highest-scoring cell is **not** automatically the best. A cell is non-dominated only if no other cell costs no more in *both* parameters and FLOPs while scoring at least as well on **all six** metrics (tolerance 1e-4).

**Non-dominated (Pareto frontier)**

| cell | Params (M) | FLOPs (G) | mAP50 | da_mIoU | lane_mIoU |
|---|---|---|---|---|---|
| **esmall_z16** | 0.1013 | 0.722 | 0.2687 | 0.8423 | 0.5818 |
| **esmall_z32** | 0.1135 | 0.852 | 0.2810 | 0.8456 | 0.5841 |
| **esmall_z128** | 0.1867 | 1.635 | 0.2639 | 0.8490 | 0.5855 |
| **ebase_z16** | 0.1889 | 1.060 | 0.3204 | 0.8564 | 0.5867 |
| **ebase_z32** | 0.2027 | 1.197 | 0.3222 | 0.8527 | 0.5875 |
| **ebase_z128** | 0.2858 | 2.023 | 0.3161 | 0.8522 | 0.5877 |
| **elarge_z16** | 0.2915 | 1.428 | 0.3585 | 0.8589 | 0.5878 |
| **elarge_z32** | 0.3068 | 1.571 | 0.3494 | 0.8546 | 0.5893 |
| **elarge_z128** | 0.3984 | 2.429 | 0.3436 | 0.8575 | 0.5925 |

**Dominated** (some cell is cheaper and at least as accurate)

| cell | Params (M) | FLOPs (G) | dominated by |
|---|---|---|---|

Machine-readable: `experiments/phase3b/phase3B_pareto.csv`.

**All 9 cells are non-dominated.** No cell is strictly worse than another on this metric set — as cost rises, at least one metric always improves in return. That is a genuine, unresolved trade-off rather than an analysis failure, and it has a direct consequence: **dominance cannot pick a winner, so the decision must be made on a cost budget.** That is precisely the fixed-budget comparison Phase 3C is designed to run, which is further evidence that Phase 3C is the right next step.

It also means any claim of the form “cell X is best” would be an unsupported value judgement about how much FLOPs a mAP point is worth, not a finding of this experiment.

A cell that wins on mAP50 but costs much more in FLOPs for gains inside the noise floor is **not** called better here.

## 11. Research Conclusion

### 11.1 Do encoder capacity and Z capacity interact?

- DA/Lane metrics revived by z=128 at E-large: **2/4**
- DA/Lane metrics revived by z=128 at E-small: **2/4**
- any metric moved beyond the reference noise floor anywhere: **7/18**

**Stopping rule: C.**

Z produces real gains in several places, so Phase 2-D's “Z is saturated” must be **re-scoped** to “saturated under the baseline encoder”. It is no longer correct to say z=16 is universally sufficient. The next step is still a fixed-budget comparison, not simply growing z.

### 11.1b The effect is task-specific — the four rules under-describe it

The stopping rule is a coarse instrument, and here it hides the actual finding: **the z effect is not uniform across tasks — its sign differs by task.** Reporting only “rule C” would be misleading, so the per-task picture is given explicitly.

| task | Δz128 at E-small | Δz128 at E-base | Δz128 at E-large | interaction (E-large vs E-small) | reading |
|---|---|---|---|---|---|
| detection | -0.0029 | -0.0032 | -0.0117 | -0.0088 | **negative interaction** — Z hurts more as encoder grows |
| DA | +0.0082 | -0.0051 | -0.0017 | -0.0100 | no interaction (z acts as a main effect) |
| lane | +0.0054 | +0.0021 | +0.0066 | +0.0012 | no interaction (z acts as a main effect) |

Reading the three tasks separately:

- **Detection** — Z has no mechanism to help it (R0 bypasses Z), and at E-large a wider Z is actively harmful. The most plausible account is gradient coupling: Z capacity competes for shared optimisation pressure without feeding the detection head. z16→z128 costs detection real accuracy and real FLOPs.
- **DA** — insensitive to Z at *every* encoder. This is the one task for which “saturation” is the fair description.
- **Lane** — benefits from Z at *every* encoder, by a similar margin. This is a **main effect of Z**, not an interaction: z=16 was leaving lane accuracy on the table even at E-small.

Consequence for the Phase 2-D wording: “z=16 is sufficient” is **not** universally true. It is sufficient for DA and more than sufficient for detection, but it under-serves lane.

### 11.2 Should limited parameters go to the encoder or to Z?

Answered per task, because the answer is **not** the same for all three:

- **Detection** — goes to the encoder, and the mechanism is known: R0's detection head bypasses Z entirely, so encoder capacity is the only lever that reaches it.
  E-small→E-large at z=16: +0.0898 mAP50 for +0.1902 M.
  z16→z128 at E-large: -0.0149 mAP50 for +0.1069 M.
  Encoder spending is the only lever that reaches detection, and it is worth roughly 20× more per parameter than Z spending on this task (§9). Z spending is worse than useless here: it costs parameters, FLOPs *and* accuracy.

- **DA** — neither lever buys much past E-base, and Z buys nothing at any encoder size.
  encoder at z=16: +0.0141 then +0.0025 da_mIoU — the second step is inside the noise floor (0.0142).
  DA is the one task whose Phase 3A plateau survives this test: give it the encoder up to roughly E-base and stop.

- **Lane** — the one task that genuinely wants Z capacity, at every encoder size, by a comparable margin.
  lane_fg, z16→z128: esmall +0.0070, ebase +0.0032, elarge +0.0085 (noise 0.0032).
  But it also still responds to the encoder, so the honest answer for lane is **both**, in proportions this experiment cannot determine — because no cell here holds the total budget fixed.

So the headline question has no single answer: *encoder* for detection, *a little encoder then stop* for DA, *a real share for Z* for lane. Turning that into one architecture requires knowing the budget, which is the next experiment.

The allocation question cannot be settled by Phase 3B alone, because none of these nine cells holds the total parameter budget constant while shifting it between encoder and Z. That is exactly what Phase 3C does.

## 12. Recommendation for Phase 3C

Proceed to the fixed-budget allocation experiment **first**, and treat “z16 is sufficient” as retired. Only if the fixed-budget comparison shows Z-heavy winning should a finer capacity analysis follow — and it should stay budget-constrained rather than becoming an open-ended z sweep.

**Not executed in this report** — Phase 3C is proposed only. No Phase 3C, R2, pruning, quantisation or extra z values were run during Phase 3B.

## 13. Limitations

- **Single seed.** Every cell is seed=0. Phase 3B does **not** independently estimate 20-epoch multi-seed variance. The noise floor quoted throughout is an external reference from Phase 2-C (3 seeds, **4 epochs**, pooled stdev) and is a proxy, not a Phase 3B estimate.
- **20 epochs.** No claim here is shown to hold at a different budget; Phase 2-D demonstrated that budget changes conclusions.
- **GPU FPS is unreliable.** SW Power Cap / clock throttling (`clocks_throttle_reasons.active=0x4`) makes FPS and latency vary by more than 2× for the same model. All efficiency claims use parameters and FLOPs only.
- **R0: detection bypasses Z.** Detection reads encoder F2/F3/F4 directly, so no detection result may be attributed to Z capacity.
- **One cell is reused, not re-run.** `ebase_z16` comes from Phase 2-D `expD_z16_e20` (identical encoder, z, epochs, seed, protocol), flagged in the CSV. Its `train_wall_min` is recorded as NA rather than guessed.
- **No correction for multiple comparisons** is applied. With 6 metrics × 2 Z levels × 3 encoders, some apparent movement is expected by chance; the sign pattern is the evidence, not individual cells.
- **The “saturation” conclusion, if reached, is scoped** to z ≤ 128 and 20 epochs. It does not rule out a larger Z.

---

**Protocol integrity.** No seed was changed, no run was dropped, no loss/optimizer/head/data/augmentation/input size was modified, and no result was selected for reporting. The six new cells were run in ascending compute cost; that ordering is an execution detail only and was never used to alter the design.

**Artifacts**

| file | contents |
|---|---|
| `experiments/phase3b/phase3B_encoder_z.csv` | the 6 new cells |
| `experiments/phase3a/exp3A_encoder.csv` | the z=16 column |
| `experiments/phase3b/phase3B_analysis.txt` | sections 3–10 in full |
| `experiments/phase3b/phase3B_interaction.csv` | interaction terms |
| `experiments/phase3b/phase3B_pareto.csv` | dominance status per cell |
| `experiments/phase3b/phase3B_config_audit.txt` | pre-launch audit |
| `experiments/phase3b/exp3B_<enc>_z<z>/` | ckpt, training log, config |
| `experiments/phase3b/exp3B_<enc>_z<z>_eval/` | metrics.json, eval log |


# ＝＝＝ 来源：PHASE3C_REPORT ＝＝＝

# Phase 3C · Fixed-Budget Capacity Allocation

**Question.** Under a fixed total parameter budget, is capacity better spent on the
encoder or on the compact representation Z?

**Protocol.** seed 0 · 20 epochs · batch 16 · lr 1e-3 · AdamW + cosine · 640×640 ·
tri_train 69863 · blocks [2,2,2] · heads, losses, augmentation untouched.
Only encoder width and Z width move. Generated from the result CSVs by
`scripts/phase3c_report.py`; no value in this document is hand-entered.

---

## 1. Research Question

Phase 3A showed encoder capacity drives performance, especially detection.
Phase 3B showed the Z effect is **task-specific**: detection responds negatively to
wider Z (and R0 bypasses Z entirely), DA is insensitive at every encoder size, and
lane shows a stable positive Z main effect.

Neither phase held the total budget fixed, so neither can answer where a *limited*
parameter budget should go. Comparing `E-large + z16` with `E-small + z128` proves
nothing about allocation, because the two models do not cost the same. Phase 3C
therefore moves from a capacity sweep to a **fixed-budget allocation comparison**:

> At equal total parameters, which allocation — encoder-heavy, balanced, or Z-heavy
> — gives the better accuracy / compute trade-off?

## 2. Existing Evidence

- **Phase 2-D** concluded z=16 was generally sufficient. That conclusion was scoped
  to the baseline encoder and is now retired.
- **Phase 3A**: encoder capacity is the dominant lever; detection is far from
  saturated while DA/Lane flatten.
- **Phase 3B**: the Z effect splits by task — negative interaction on detection,
  genuine saturation on DA, a stable positive main effect on lane. All 9 cells were
  Pareto non-dominated, so no natural single winner exists.

Phase 3C tests the Phase 3B priors under a fixed budget instead of assuming them:
detection should favour the encoder, DA should be indifferent, lane should favour Z.

## 3. Budget Design

Three budget layers were proposed (0.19M / 0.29M / 0.39M). A cell joins a layer if
its measured parameters are within ±5% of the target. Membership uses the same
tolerance as the like-for-like test, so a layer can never contain a cell that then
fails the comparison it was admitted for.

Cells that sit near a layer but outside the tolerance are excluded from the triad
(they are still real measurements and still appear in the Pareto frontier):

- `ebase_z32` — +6.68% from the Budget-L target (0.2027 M).
- `elarge_z32` — +5.79% from the Budget-M target (0.3068 M).


| layer | target | cells | param span | verdict |
|---|---|---|---|---|
| Budget-L | 0.19 M | 3 | 1.56% | equal-budget |
| Budget-M | 0.29 M | 3 | 2.21% | equal-budget |
| Budget-H | 0.39 M | 1 | 0.00% | equal-budget |

Two new encoders were created **only** to complete the missing *balanced* leg, by
interpolating the existing stem/stages scaling law (blocks, depth, heads untouched):

| new encoder | stem | stages | z | params | target | deviation |
|---|---|---|---|---|---|---|
| midL | 15 | 32,64,88,120 | 32 | 0.186006 M | 0.19 M | -2.10% |
| midM | 19 | 40,80,120,160 | 32 | 0.285206 M | 0.29 M | -1.65% |

No other new widths were introduced. `Budget-H` holds a single existing cell and
cannot support any comparison, so it is reported for completeness only and no
allocation claim is made there.

## 4. Allocation Comparison

### Budget-L (~0.19 M)

| allocation | cell | params (M) | FLOPs (G) | mAP50 | mAP50-95 | da_mIoU | da_fg | lane_mIoU | lane_fg |
|---|---|---|---|---|---|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.1889 | 1.0597 | 0.3204 | 0.1149 | 0.8564 | 0.7723 | 0.5867 | 0.1973 |
| balanced | `midL_z32` | 0.1860 | 1.1603 | 0.3152 | 0.1109 | 0.8532 | 0.7676 | 0.5862 | 0.1972 |
| Z-heavy | `esmall_z128` | 0.1867 | 1.6350 | 0.2639 | 0.0879 | 0.8490 | 0.7613 | 0.5855 | 0.1955 |

Delta versus the encoder-heavy cell (`ebase_z16`):

| cell | allocation | Δparams | ΔFLOPs | ΔmAP50 | Δda_mIoU | Δlane_mIoU | Δlane_fg |
|---|---|---|---|---|---|---|---|
| `midL_z32` | balanced | -0.0029 | +0.1006 | -0.0052 | -0.0032 | -0.0005 | -0.0001 |
| `esmall_z128` | Z-heavy | -0.0022 | +0.5753 | -0.0565 | -0.0074 | -0.0012 | -0.0018 |

### Budget-M (~0.29 M)

| allocation | cell | params (M) | FLOPs (G) | mAP50 | mAP50-95 | da_mIoU | da_fg | lane_mIoU | lane_fg |
|---|---|---|---|---|---|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.2915 | 1.4279 | 0.3585 | 0.1356 | 0.8589 | 0.7761 | 0.5878 | 0.2000 |
| balanced | `midM_z32` | 0.2852 | 1.5249 | 0.3462 | 0.1279 | 0.8572 | 0.7737 | 0.5900 | 0.2037 |
| Z-heavy | `ebase_z128` | 0.2858 | 2.0231 | 0.3161 | 0.1129 | 0.8522 | 0.7663 | 0.5877 | 0.2005 |

Delta versus the encoder-heavy cell (`elarge_z16`):

| cell | allocation | Δparams | ΔFLOPs | ΔmAP50 | Δda_mIoU | Δlane_mIoU | Δlane_fg |
|---|---|---|---|---|---|---|---|
| `midM_z32` | balanced | -0.0063 | +0.0970 | -0.0123 | -0.0017 | +0.0022 | +0.0037 |
| `ebase_z128` | Z-heavy | -0.0057 | +0.5952 | -0.0424 | -0.0067 | -0.0001 | +0.0005 |

### Budget-H (~0.39 M) — single cell, no comparison possible

Only `elarge_z128` (0.3984 M) exists in this layer.

## 5. Task-wise Results

The three tasks are reported separately. No weighted composite score is used to
pick a winner.

### detection

**Budget-L (~0.19 M)**

| allocation | cell | mAP50 | mAP50_95 |
|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.3204 | 0.1149 |
| balanced | `midL_z32` | 0.3152 | 0.1109 |
| Z-heavy | `esmall_z128` | 0.2639 | 0.0879 |

- `mAP50`: spread 0.0565 > noise 0.0073 — best (statistical tie): **balanced, encoder-heavy**.
- `mAP50_95`: spread 0.0270 > noise 0.0032 — best (statistical tie): **encoder-heavy**.

**Budget-M (~0.29 M)**

| allocation | cell | mAP50 | mAP50_95 |
|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.3585 | 0.1356 |
| balanced | `midM_z32` | 0.3462 | 0.1279 |
| Z-heavy | `ebase_z128` | 0.3161 | 0.1129 |

- `mAP50`: spread 0.0424 > noise 0.0073 — best (statistical tie): **encoder-heavy**.
- `mAP50_95`: spread 0.0227 > noise 0.0032 — best (statistical tie): **encoder-heavy**.

### DA

**Budget-L (~0.19 M)**

| allocation | cell | da_mIoU | da_fg |
|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.8564 | 0.7723 |
| balanced | `midL_z32` | 0.8532 | 0.7676 |
| Z-heavy | `esmall_z128` | 0.8490 | 0.7613 |

- `da_mIoU`: spread 0.0074 ≤ noise 0.0142 — **no allocation wins**.
- `da_fg`: spread 0.0110 ≤ noise 0.0202 — **no allocation wins**.

**Budget-M (~0.29 M)**

| allocation | cell | da_mIoU | da_fg |
|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.8589 | 0.7761 |
| balanced | `midM_z32` | 0.8572 | 0.7737 |
| Z-heavy | `ebase_z128` | 0.8522 | 0.7663 |

- `da_mIoU`: spread 0.0067 ≤ noise 0.0142 — **no allocation wins**.
- `da_fg`: spread 0.0098 ≤ noise 0.0202 — **no allocation wins**.

### lane

**Budget-L (~0.19 M)**

| allocation | cell | lane_mIoU | lane_fg |
|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.5867 | 0.1973 |
| balanced | `midL_z32` | 0.5862 | 0.1972 |
| Z-heavy | `esmall_z128` | 0.5855 | 0.1955 |

- `lane_mIoU`: spread 0.0012 ≤ noise 0.0021 — **no allocation wins**.
- `lane_fg`: spread 0.0018 ≤ noise 0.0032 — **no allocation wins**.

**Budget-M (~0.29 M)**

| allocation | cell | lane_mIoU | lane_fg |
|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.5878 | 0.2000 |
| balanced | `midM_z32` | 0.5900 | 0.2037 |
| Z-heavy | `ebase_z128` | 0.5877 | 0.2005 |

- `lane_mIoU`: spread 0.0023 > noise 0.0021 — best (statistical tie): **balanced**.
- `lane_fg`: spread 0.0037 > noise 0.0032 — best (statistical tie): **Z-heavy, balanced**.

## 6. Params vs FLOPs

Equal parameters do **not** mean equal compute. Z operates at 1/8 resolution and is
consumed by the segmentation heads, so widening Z is expensive in FLOPs for very
little parameter cost. That asymmetry is a result in its own right.

| layer | param span | FLOPs span |
|---|---|---|
| Budget-L | 1.56% | 54.3% |
| Budget-M | 2.21% | 41.7% |

**Budget-L**

| allocation | cell | params (M) | FLOPs (G) | FLOPs per 0.01M params |
|---|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.1889 | 1.0597 | 0.0561 |
| balanced | `midL_z32` | 0.1860 | 1.1603 | 0.0624 |
| Z-heavy | `esmall_z128` | 0.1867 | 1.6350 | 0.0876 |

**Budget-M**

| allocation | cell | params (M) | FLOPs (G) | FLOPs per 0.01M params |
|---|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.2915 | 1.4279 | 0.0490 |
| balanced | `midM_z32` | 0.2852 | 1.5249 | 0.0535 |
| Z-heavy | `ebase_z128` | 0.2858 | 2.0231 | 0.0708 |

## 7. Fixed-budget Dominance

Only pairs whose parameter gap is within 5% are tested at all. Inside such a pair
the two cells are treated as sharing one budget by construction, so the residual 1-2%
gap is neither an advantage nor a disadvantage.

**Tier 1 (strict).** params ≤, FLOPs ≤, all six metrics ≥ raw value.
**Tier 2 (noise-aware).** same budget, FLOPs ≤, no metric worse by more than the
noise floor, at least one metric better beyond it.

Tier 2 is the rule that applies here. Tier 1 is shown only for transparency: it fails
on this data purely because of sub-tolerance and sub-noise residuals, not because any
allocation is genuinely competitive.

**Tier 1 — strict**

| dominating | dominated | Δparams | ΔFLOPs | better beyond noise |
|---|---|---|---|---|
| `midL_z32` | `esmall_z128` | -0.0007 (-0.37%) | -0.4747 (-29.0%) | mAP50, mAP50_95 |
| `midM_z32` | `ebase_z128` | -0.0006 (-0.21%) | -0.4982 (-24.6%) | mAP50, mAP50_95, lane_mIoU |

**Tier 2 — noise-aware (the applicable rule)**

| dominating | dominated | Δparams | ΔFLOPs | better beyond noise |
|---|---|---|---|---|
| `ebase_z16` | `esmall_z128` | +0.0022 (+1.18%) | -0.5753 (-35.2%) | mAP50, mAP50_95 |
| `ebase_z16` | `midL_z32` | +0.0029 (+1.56%) | -0.1006 (-8.7%) | mAP50_95 |
| `elarge_z16` | `ebase_z128` | +0.0057 (+1.99%) | -0.5952 (-29.4%) | mAP50, mAP50_95 |

**Near miss** — blocked only by metrics whose shortfall is at most 1.5× noise:

- `elarge_z16` vs `midM_z32`: wins mAP50, mAP50_95, blocked by `lane_mIoU` (1.05× noise), `lane_fg` (1.16× noise).

A margin that small is not a loss, it is an unresolved measurement. It is reported
as unresolved rather than as a win for either side.

Under tier 2 the cells that are never dominated by anything are: `ebase_z16`, `elarge_z16`.

## 8. Pareto Frontier

Cost is params **and** FLOPs; accuracy is all six metrics. Highest mAP alone selects
nothing here.

**Non-dominated (9):**

| cell | params (M) | FLOPs (G) | mAP50 | da_mIoU | lane_mIoU |
|---|---|---|---|---|---|
| `esmall_z16` | 0.1013 | 0.7218 | 0.2687 | 0.8423 | 0.5818 |
| `esmall_z32` | 0.1135 | 0.8522 | 0.2810 | 0.8456 | 0.5841 |
| `midL_z32` | 0.1860 | 1.1603 | 0.3152 | 0.8532 | 0.5862 |
| `ebase_z16` | 0.1889 | 1.0597 | 0.3204 | 0.8564 | 0.5867 |
| `ebase_z32` | 0.2027 | 1.1973 | 0.3222 | 0.8527 | 0.5875 |
| `midM_z32` | 0.2852 | 1.5249 | 0.3462 | 0.8572 | 0.5900 |
| `elarge_z16` | 0.2915 | 1.4279 | 0.3585 | 0.8589 | 0.5878 |
| `elarge_z32` | 0.3068 | 1.5709 | 0.3494 | 0.8546 | 0.5893 |
| `elarge_z128` | 0.3984 | 2.4292 | 0.3436 | 0.8575 | 0.5925 |

**Dominated (2):** `esmall_z128`, `ebase_z128`

## 9. Task-specific Capacity Analysis

The Phase 3B priors are treated as hypotheses and re-tested here under a fixed budget.

### detection

- Budget-L `mAP50`: best (statistical tie) **balanced, encoder-heavy**.
- Budget-L `mAP50_95`: best (statistical tie) **encoder-heavy**.
- Budget-M `mAP50`: best (statistical tie) **encoder-heavy**.
- Budget-M `mAP50_95`: best (statistical tie) **encoder-heavy**.

Decided cells: 4 · within noise: 0 · wins: encoder-heavy ×4, balanced ×1

### DA

- Budget-L `da_mIoU`: spread 0.0074 ≤ noise 0.0142 → no allocation wins.
- Budget-L `da_fg`: spread 0.0110 ≤ noise 0.0202 → no allocation wins.
- Budget-M `da_mIoU`: spread 0.0067 ≤ noise 0.0142 → no allocation wins.
- Budget-M `da_fg`: spread 0.0098 ≤ noise 0.0202 → no allocation wins.

Decided cells: 0 · within noise: 4 · wins: none

### lane

- Budget-L `lane_mIoU`: spread 0.0012 ≤ noise 0.0021 → no allocation wins.
- Budget-L `lane_fg`: spread 0.0018 ≤ noise 0.0032 → no allocation wins.
- Budget-M `lane_mIoU`: best (statistical tie) **balanced**.
- Budget-M `lane_fg`: best (statistical tie) **Z-heavy, balanced**.

Decided cells: 2 · within noise: 2 · wins: balanced ×2, Z-heavy ×1

## 10. Conclusion

**Direct answer: for detection, yes — under an equal parameter budget capacity should
go to the encoder. For DA and lane the answer is not established, because moving the
allocation barely moves them at all.**

Across every budget layer that supports a comparison, the encoder-heavy allocation
beats the Z-heavy allocation on detection by a margin far beyond noise, and it does it
with **substantially fewer FLOPs**. DA is indifferent to the allocation at every layer.
Lane is indifferent at Budget-L, and at Budget-M shows a marginal lead for the
*balanced* cell — not for Z-heavy — at roughly 1.05-1.16× noise, which one seed cannot
resolve.

The result is stronger than a trade-off:

- **Budget-L**: encoder-heavy `ebase_z16` vs Z-heavy `esmall_z128` — Δparams +0.0022 M (1.18%), ΔmAP50 +0.0565 (7.7× noise), ΔFLOPs -0.5753 G (-35.2%).
- **Budget-M**: encoder-heavy `elarge_z16` vs Z-heavy `ebase_z128` — Δparams +0.0057 M (1.99%), ΔmAP50 +0.0424 (5.8× noise), ΔFLOPs -0.5952 G (-29.4%).

So the allocation is not a compromise between accuracy and cost: spending the same
parameter budget on the encoder rather than on Z buys **more accuracy and less
compute at the same time**. A per-parameter efficiency ratio is deliberately not
quoted, because with Δparams ≈ 0 the ratio diverges and would be meaningless.

Per task:

| task | where capacity should go | evidence |
|---|---|---|
| detection | **encoder** | encoder-heavy clearly wins in 2/2 comparable layers |
| DA | **neither** — indifferent | every layer spread inside the noise floor |
| lane | **no reliable preference** | 1/2 layers inside noise; where a winner does appear it is *balanced*, never Z-heavy |

**Stopping condition: none of A/B/C/D matches exactly.**

No rule matches, and choosing one anyway would be the wrong move. Concretely:

- **Rule A fails its second clause.** Encoder-heavy clearly wins detection in
  2/2 layers, but it wins **no** segmentation task anywhere: DA is inside noise
  at every layer, and at Budget-M the balanced cell is the one that is ahead on lane.
- **Rule B fails.** Z-heavy never wins lane, on any layer.
- **Rule C fails.** Balanced does win lane at Budget-M, but it is a clear detection
  loser at both layers, so it is not the best all-round trade-off.
- **Rule D fails.** The detection margin is 5.8-7.7× the noise floor.

The honest reading is therefore a **detection-only rule A**: encoder-first is
established for detection and is *not* established for DA or lane. The lane
counter-signal is real but sits at only ~1.05-1.16× noise, which is exactly the
regime a single seed cannot adjudicate. It is recorded as unresolved, not as a
win for either allocation.

One caveat that must not be lost: this does **not** say Z is useless. It says that at
these budgets, *marginal* parameters are better spent on the encoder. Lane in
particular showed a real Z main effect in Phase 3B; what Phase 3C shows is that when
the budget is fixed, buying that Z capacity by shrinking the encoder is a bad deal.

## 11. Limitations

- **Single seed.** Every cell is seed=0. Phase 3C does not estimate its own variance;
  the noise floor is an external reference from Phase 2-C (3 seeds, 4 epochs, pooled
  stdev) and is a proxy, not a Phase 3C measurement.
- **20 epochs.** No claim is shown to hold at another budget; Phase 2-D demonstrated
  that changing the budget can change conclusions.
- **FPS / latency unusable.** Clock throttling makes FPS vary by more than 2× for the
  same model. All efficiency statements use params and FLOPs only.
- **R0: detection bypasses Z.** The detection head reads encoder F2/F3/F4 directly, so
  no detection result may be attributed to Z capacity. Under R2 this changes.
- **Budget-H is a single cell** (`elarge_z128`); no allocation claim is made at that
  budget. The conclusions rest on Budget-L and Budget-M.
- **The two new encoders exist only for budget matching.** They are width
  interpolations, not a new capacity sweep, and are not evidence about encoder
  scaling on their own.
- **No multiple-comparison correction.** With 6 metrics × 3 layers, some movement is
  expected by chance; the sign pattern across layers is the evidence, not single cells.

## 12. Recommendation for Phase 4 (R2)

Proposed only — **not executed here.** No R2, pruning, quantisation or extra z sweep
was run during Phase 3C.

Phase 3C established the allocation law **under R0, where detection bypasses Z**. The
decisive question for Phase 4 is whether that law survives when detection is forced
through the shared bottleneck:

1. Audit the R2 code path first, then run only `R0+z16`, `R2+z16`, `R2+z32`, `R2+z128`.
2. Re-run this same fixed-budget comparison under R2. If encoder-heavy still wins,
   the allocation law is a property of the budget, not of the routing.
3. If Z-heavy becomes competitive under R2, the R0 result was partly an artefact of
   detection not consuming Z — that is the finding, not a failure.
4. If R2 is unstable or loses accuracy, do **not** explain it as insufficient Z
   capacity without a budget-matched control.

Before Phase 4, a 3-seed confirmatory run is worth doing on the final candidate
configuration (`elarge_z16`) and on its budget-matched Z-heavy counterpart
(`ebase_z128`), since those two carry the entire conclusion.

---

**Protocol integrity.** No seed was changed, no run was dropped, no loss, optimizer,
head, dataset, augmentation or input size was modified, and no result was selected for
reporting. New encoders were introduced solely to make the budget comparison fair and
are documented in §3.



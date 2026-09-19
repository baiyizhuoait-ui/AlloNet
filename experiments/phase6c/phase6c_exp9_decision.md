# Phase 6C Decision Report — EXP-09（容量 × 锚框 2×2 / H-36）

**生成时刻：** 2026-09-10 22:15:31 (WSL CST)　**代码版本：** `9fb29e7`
**生成方式：** `scripts/phase6c_e9_report.py` 机械计算（阈值写死，不接受事后重定义）
**判据来源：** `docs/PHASE6C_CAPACITY_ANCHOR_PREREGISTRATION.md` §3（开跑前固定）
**噪声标尺：** σ_20 = 0.0048（§3.1，EXP-08 两臂 pooled within-arm seed-sd）⇒ 2σ_20 = 0.0096
**tier：** screening（n=1/单档）—— 只用于决定是否继续投入，**不用于对外结论**。

---

## 0. TL;DR

| 问题 | 判定 | 一句话依据 |
|---|---|---|
| **H-36**：锚框监督与表示容量是**替代**还是**可加** | **SUBSTITUTION** | interaction = -0.1435（低容量 Δ +0.1428，高容量 Δ -0.0007） |
| C4（A-uniform × k-means，判决格） | 实测 mAP50 = **0.5332** | 训练 162 min / 20ep / seed0（eval 由补评恢复，复用 checkpoint，未重训） |
| 决策点 | mAP50(C4) 对照 **0.6053 / 0.6671 / 0.6863** | ≤ 0.6053 → 替代成立；0.6053~0.6671 压缩；0.6671~0.6863 无信号（纯可加点 0.6767 ± 2σ）；≥ 0.6863 → H-36 被否 |

**分支动作：** 解锁 P6C-STEP2（C4 补 s1/s2 两格，2 × 20ep ≈ 5.4 GPU·h）。**需人确认后才启动**，本报告不自动启动。

---

## 1. 四格读数（全部实测，磁盘直读）

| 格 | 容量 | 锚框 | seed | 源目录 | mAP50 | mAP50-95 | da_mIoU | lane_mIoU | params | FLOPs(G) |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | r2_z16 | old | 0 | `experiments/phase4a/exp4A_r2_z16_e20` | **0.3543** | 0.1273 | 0.8488 | 0.5847 | 201366 | 1.0796 |
| C2 | r2_z16 | k-means | 1 | `experiments/phase4a/exp4B_danc_z16_e20_s1` | **0.4982** | 0.2171 | 0.8567 | 0.5824 | 201366 | 1.0796 |
| C2 | r2_z16 | k-means | 2 | `experiments/phase4a/exp4B_danc_z16_e20_s2` | **0.4960** | 0.2142 | 0.8555 | 0.5813 | 201366 | 1.0796 |
| C2(补seed0) | r2_z16 | k-means | 0 | `experiments/phase6c/exp9_r2z16_km` | **0.4908** | 0.2112 | 0.8420 | 0.5844 | 201366 | 1.0796 |
| C3 | A-uniform | old | 0 | `experiments/phase6/exp6_e8_unif20` | **0.5339** | 0.2420 | 0.8581 | 0.5879 | 333862 | 1.6650 |
| C4 | A-uniform | k-means | 0 | `experiments/phase6c/exp9_aunif_km` | **0.5332** | 0.2426 | 0.8557 | 0.5874 | 333862 | 1.6650 |

### 1.1 结果表行（`experiments/phase6c/phase6c_e9_factorial.csv`）

```
variant,cell,z,encoder,epochs,params_M,flops_G,fps,mAP50,mAP50_95,da_mIoU,da_fg,lane_mIoU,lane_fg,peak_gpu_mem_mib,final_train_loss,train_wall_min,seed,source,git_commit
r2,r2z16_km,16,ebase,20,0.2014,1.0796,397.23,0.4908,0.2112,0.8420,0.7511,0.5844,0.1936,NA,0.2130,134,0,trained,9fb29e727e979d9b7511c0d23e8be756a162594f
r2,aunif_km,16,ebase,20,0.3339,1.6650,419.34,0.5332,0.2426,0.8557,0.7715,0.5874,0.1999,NA,0.2019,162,0,trained,9fb29e727e979d9b7511c0d23e8be756a162594f
```

**重复行检测：** 无

---

## 2. 交互项计算（只读交互项，§3.2 纪律）

```
低容量 Δ_anchor = mean(C2 s1/s2) - C1 = 0.4971 - 0.3543 = +0.1428   [预注册基准]
低容量 Δ_anchor = mean(C2 s0/s1/s2) - C1 = 0.4950 - 0.3543 = +0.1407   [3-seed 稳健读数，§5.3]
高容量 Δ_anchor = C4 - C3(预注册基) = 0.5332 - 0.5339 = -0.0007
interaction      = -0.0007 - (+0.1428) = -0.1435      [主判定]
2σ_20            = 0.0096

交叉核对（全精度，用实测 C3 与实测低容量 Δ）：
  C3 实测 = 0.5339 (预注册写 0.5339)
  interaction = -0.1435   -> 分支 SUBSTITUTION
  与主判定一致性：一致
```

---

## 3. 分支判定（§3.3，magnitude 与显著性**同时**要求）

| mAP50(C4) 区间 | interaction | 分支 | 本次落在 |
|---|---|---|---|
| ≤ 0.6053 | ≤ -0.0096 且 Δ_高 ≤ 0.0714 | **SUBSTITUTION** | ★ |
| 0.6053 ~ 0.6671 | ≤ -0.0096，Δ_高 未腰斩 | DIMINISHING |  |
| 0.6671 ~ 0.6863 | \|interaction\| < 0.0096 | ADDITIVE |  |
| ≥ 0.6863 | ≥ +0.0096 | COMPLEMENTARY |  |

> 区间端点换算：纯可加预测点 = C3 + 低容量Δ = 0.5339 + 0.1428 = **0.6767**（即 interaction = 0）。
> ADDITIVE 带 = 纯可加点 ± 2σ = (0.6671, 0.6863)；替代门槛 = 增益腰斩点 = C3 + 0.5×Δ_low = **0.6053**。

**实测 mAP50(C4) = 0.5332 ⇒ 判定 SUBSTITUTION。**

---

## 4. 下一步（**待用户决定，本报告不自动启动任何训练**）

1. 解锁 P6C-STEP2（C4 补 s1/s2 两格，2 × 20ep ≈ 5.4 GPU·h）。需人确认后才启动，本报告不自动启动。
2. 禁止 scope drift（§4.3）：不得"顺手加一个臂"、不得因"4ep 也能出数"而补跑 4ep 臂。
3. 若需 STEP2，先确认：GPU 时间预算、是否连跑、以及 3-seed 的 seed 选择（s1/s2）。

---

## 5. 诚实性限制（§5 原文，必须随结论一并陈述）

1. **容量轴不干净**：C3/C4 = A-uniform（params 实测 +65.8%，FLOPs +54.2%），与 C1/C2 不只是"容量大小"之别，还有"容量加在哪个维度"之别 → 本 interaction 是**跨维度容量**的交互，**不能**表述为"参数量的交互"。
2. **20ep 旧锚框臂在退化**（0.3879@4ep → 0.3543@20ep）→ Δ_anchor 含"锚框影响收敛/退化"成分，**不等于**纯表示改善。
3. **跨 seed**：低容量 Δ 的预注册基准用 s1/s2，与 C1(seed0) 不同 seed；3-seed 读数见 §2，**两个读数都已报**。
4. **n=1 的格**：C3 为 seed0 单次；3σ 级结论必须等 P6C-STEP2。
5. **Screening tier**：只用于决定是否继续投入，不用于对外结论。
6. **FPS 不作为判据**（项目约定）。

**数据质量备注**：`peak_gpu_mem_mib` 列为 NA（训练日志无 `peak_mem=` 字段，格式为 `mem N/8151MiB`）；两格一致故可比，未回填。

---

## 6. 复现与审计

- 生成命令：`~/ai_study/gpu_env/bin/python scripts/phase6c_e9_report.py`
- 输入（全部只读）：§1 表内六个 `_eval/metrics.json` + `experiments/phase6c/phase6c_e9_factorial.csv`
- 判据常量（脚本内硬编码，来源 §3）：σ_20=0.0048, C3=0.5339, 低容量Δ=0.1428, gate=0.0714
- cell 1 的 eval 为**补评**（复用 checkpoint、未重训），eval 命令与 `phase6_run.sh:38` 逐字一致；`git_commit` 列保留 `9fb29e7`。
- 失败现场保留在 `experiments/phase6c/exp9_aunif_km_eval.log`（未被覆盖）。


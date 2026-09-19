# Phase 6 —— 手动选跑菜单（人工调度版）

**状态变更**：200ep 守候已于 2026-09-14 14:58 撤销。
终止范围仅限守候自身（PID 24554 及其进程组），**未触碰训练**：主进程 5840 存活、链 5763 存活、
链的 `.d8_seeds.lock` 仍由链持有、无孤儿 `sleep`。此后 GPU 调度权交回给你。

**仍然自动在跑的（不用你管）**：`phase6_seeds_resume.sh 1 2`（PID 5763）
= seed1 100ep → seed2 100ep，自己走完并写 CSV / eval。

> 本文件只列**已存在、可直接跑**的东西。凡需要先写代码的，会明说。

---

## 0 动手前先看状态

```bash
cd ~/ai_study/trac
ps -eo pid,etime,cmd | grep -E '[t]rain\.py --config|[p]hase6'   # 谁在跑
tail -1 experiments/phase6/final/B100_s1/training_log.txt        # seed1 到哪个 epoch
free -m | awk '/Mem:/{print "mem avail", $7, "MiB"}'             # 余量，需 ≥ 6000
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

## 1 六条硬规则（先读，违反会出事）

| # | 规则 | 为什么 |
|---|---|---|
| 1 | 一次只跑**一条**臂 | 8 GB 单卡。09-14 10:17 同时跑两条 → OOM 一次杀掉两个 seed |
| 2 | 启动前 `MemAvailable ≥ 6000 MiB` | 同上。**只有 G1 脚本自带这条守卫**，其他脚本没有 |
| 3 | 停任务按 **PID / 进程组**，不要 `pkill -f` | 曾因模式匹配到自己的命令行而把自己杀掉 |
| 4 | 不要编辑 **bash 正在读**的脚本 | Round-4 D10：读偏移移动 → 幻影失败 |
| 5 | 200ep **绝不能用 `--resume`** | cosine 恢复 `T_max` → 在 LR 地板上跑 → **伪造收敛** |
| 6 | D8 延迟探针只能在 **GPU 完全空闲**时跑 | 脚本 fail-closed 拒绝共享卡；且它测的就是时钟稳态 |

---

## 2 菜单

### A ⭐ 200ep 预算归因（G1）—— 24 h GPU

**回答**：DA 赤字是**架构属性**还是**训练预算**？

```bash
bash scripts/phase6_g1_budget200.sh --now      # 立刻跑；锁被占则拒绝
bash scripts/phase6_g1_budget200.sh --dry-run  # 只出计划，不等不跑
bash scripts/phase6_g1_budget200.sh            # 排队版：等链退出再自动开跑
```

- **前置**：seed 链已退出（锁空闲）。锁被占时 `--now` **直接拒绝（rc=2）**，不会偷偷排队
- **判据**（看数据前已冻结）：`|Δgap| < 2σ` ⇒ SATURATED；`≥ 2σ` ⇒ STILL_CLOSING。**两种都能发表**
- 跑完自动打印指标；再跑下面 B 出曲线
- ⚠️ 必须 **fresh**，脚本内部已写死，不要手工改成 `--resume`

### B ⭐ 零成本：预算曲线 + 误差棒（0 GPU，秒级）

**回答**：三点趋势什么形状、100ep 的误差棒多宽

```bash
~/ai_study/gpu_env/bin/python scripts/phase6_g1_budget_curve.py
```

- **前置**：无
- seed1/2 未落盘时脚本会**如实标注 n=1**，不会假装有误差棒
- 建议：seed2 落盘后再跑一次，才是完整 3-seed

### C 可选：延迟列扩到其余基线（~5 min，需 GPU 全空）

**现状**：探测**已经跑过**（`experiments/phase6/final/d8_latency/`，10:08→10:13，8 个模型
= 我们 5 个 + 基线 3 个：TL-tiny / TL-small / TLP-nano）。**缺的是其余基线**。

```bash
~/ai_study/gpu_env/bin/python scripts/phase6_d8_latency_probe.py --only <模型名> ...
```

- **前置**：**GPU 必须完全空闲**（fail-closed）；模型名与 `d8_latency/summary.md` 表内一致
- **价值：低**。延迟列已被排除在所有判定之外；扩表主要是为了表格完整，不影响任何结论

### D 跨架构复现（G3）—— 2–4 h 代码 + 7.5 h 训练

**回答**：容量分配结论能否迁移到第二个架构（**唯一需要新实验的洞**）

- **不是发个命令就行**：必须先写一个**拓扑不同**的 encoder
  （用 `stages/blocks` 缩放 = 循环论证，那正是基线 Model U 的做法；TLP lane 头已饱和，出局）
- 你要跑的话，我先出代码（2–4 h），再排训练

### E 只在 seed 链意外中断时才需要

```bash
bash scripts/phase6_seeds_resume.sh 2      # 只补 seed2（不加参数则补 1 2）
```

正常情况**不需要** —— 链是自动的。

### F 什么都不跑

完全合理。链会自己关掉 G2（补齐 seed 误差棒）。这是**成本最低、收益确定**的选项。

---

## 3 如果只跑一个，建议顺序

1. 等 seed2 落盘 → 跑 **B**（免费，立刻拿到 3-seed 误差棒）
2. 再跑 **A**（24 h，把"DA 末位"从最大弱点转成可归因的证据）
3. **C** 可跳过；**D** 取决于你想投哪个档位

## 4 一键停全部

```bash
bash scripts/phase6_safe_stop.sh --yes     # 默认干跑；--yes 才真停
```

## 5 产物落点

| 臂 | 产物 |
|---|---|
| A | `experiments/phase6/final/B200*/`、`experiments/phase6/final/g1_budget200.log` |
| B | stdout（要留档就 `> /tmp/g1curve.txt`） |
| C | `experiments/phase6/final/d8_latency/`（会被覆盖，跑前先备份） |
| D | `experiments/phase6/final/<新 encoder cells>/` |
| E | `experiments/phase6/final/seeds_resume.log`、`final_results.csv` |

---

> 从 **Windows 侧**执行需加前缀 `wsl.exe -e bash -lc '...'`；在 **WSL 终端**里按上面原样粘贴即可。

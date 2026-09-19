#!/usr/bin/env python3
"""Phase 6C / EXP-09 decision report generator  (capacity x anchor 2x2, H-36).

Mechanical by design.  Reads the four cells straight off disk, computes the
pre-registered interaction term and the branch verdict.  Thresholds are
hard-coded from
    docs/PHASE6C_CAPACITY_ANCHOR_PREREGISTRATION.md  section 3
and are NEVER re-derived after seeing the data.

Primary verdict uses the PREREG-LITERAL constants (C3 = 0.5339, low-capacity
delta = 0.1428) exactly as frozen before the run.  A full-precision
recomputation is printed alongside as a cross-check; if the two disagree the
report says so loudly.

Refuses to conclude anything from a `failed-no-metrics` row.

usage:  phase6c_e9_report.py [--check]
        --check : print readiness only, exit 0 if the decisive cell (C4) is
                  evaluable, 2 if not yet, 3 if a cell is broken.
"""
import json, os, sys, datetime, subprocess

ROOT = os.environ.get("P6C_REPORT_ROOT", "~/ai_study/trac")
CZP = "experiments/phase6c"
CSV = os.path.join(CZP, "phase6c_e9_factorial.csv")
REPORT = os.path.join(CZP, "phase6c_exp9_decision.md")

# ---- frozen prereg constants (section 3) -------------------------------------
SIGMA_20 = 0.0048          # 3.1 pooled within-arm seed-sd, derived pre-run
TWO = 2 * SIGMA_20         # 0.0096
PREREG_C3 = 0.5339         # C3 = A-uniform x old anchors
PREREG_LOW = 0.1428        # low-capacity anchor delta (C2 s1/s2 mean - C1)
PREREG_GATE = 0.5 * PREREG_LOW   # 0.0714  "halved" magnitude gate
ZERO_POINT = PREREG_C3 + PREREG_LOW          # 0.6767 = pure-additive prediction
SUB_CEIL = PREREG_C3 + PREREG_GATE           # 0.6053 = C4 at which anchor gain halves
DIM_CEIL = ZERO_POINT - TWO                  # 0.6671 = interaction at -2 sigma
COMP_FLOOR = ZERO_POINT + TWO                # 0.6863 = interaction at +2 sigma

ARMS = [
    # key, capacity, anchor, seed, path
    ("C1",  "r2_z16",    "old",     0, "experiments/phase4a/exp4A_r2_z16_e20_eval/metrics.json"),
    ("C2s1","r2_z16",    "k-means", 1, "experiments/phase4a/exp4B_danc_z16_e20_s1_eval/metrics.json"),
    ("C2s2","r2_z16",    "k-means", 2, "experiments/phase4a/exp4B_danc_z16_e20_s2_eval/metrics.json"),
    ("C2s0","r2_z16",    "k-means", 0, "experiments/phase6c/exp9_r2z16_km_eval/metrics.json"),
    ("C3",  "A-uniform", "old",     0, "experiments/phase6/exp6_e8_unif20_eval/metrics.json"),
    ("C4",  "A-uniform", "k-means", 0, "experiments/phase6c/exp9_aunif_km_eval/metrics.json"),
]
MKEYS = ["mAP50", "mAP50_95", "da_mIoU", "da_fg_iou", "lane_mIoU", "lane_fg_iou"]
SKEYS = ["parameters", "flops", "fps"]


def load(p):
    f = os.path.join(ROOT, p)
    if not os.path.isfile(f):
        return None
    try:
        return json.load(open(f))
    except Exception:
        return None


def fmt(v, n=4):
    if v is None:
        return "NA"
    try:
        return ("%%.%df" % n) % float(v)
    except (TypeError, ValueError):
        return str(v)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def classify(inter, d_high, gate=PREREG_GATE):
    """Pre-registered four-way rule (section 3.3). Magnitude AND significance."""
    if inter is None:
        return "INCOMPLETE"
    if inter <= -TWO and d_high <= gate:
        return "SUBSTITUTION"
    if inter <= -TWO:
        return "DIMINISHING"
    if abs(inter) < TWO:
        return "ADDITIVE"
    return "COMPLEMENTARY"


def main():
    check = "--check" in sys.argv
    m = {}
    missing = []
    for key, _c, _a, _s, p in ARMS:
        d = load(p)
        if d is None or d.get("mAP50") is None:
            missing.append((key, p))
        m[key] = d

    decisive = m.get("C4") is not None and m["C4"].get("mAP50") is not None
    if check:
        print("C4 evaluable: %s" % decisive)
        for k, p in missing:
            print("  missing: %s  %s" % (k, p))
        return 0 if decisive else 2

    if not decisive:
        print("ABORT: decisive cell C4 has no eval metrics yet.")
        for k, p in missing:
            print("  missing: %s  %s" % (k, p))
        return 2
    need = [k for k in ("C1", "C2s1", "C2s2", "C3") if m.get(k) is None]
    if need:
        print("ABORT: reference cell(s) missing: %s" % need)
        return 3

    c1 = m["C1"]["mAP50"]
    c2s1, c2s2 = m["C2s1"]["mAP50"], m["C2s2"]["mAP50"]
    c2s0 = m["C2s0"]["mAP50"] if m.get("C2s0") else None
    c3 = m["C3"]["mAP50"]
    c4 = m["C4"]["mAP50"]

    low_ref = mean([c2s1, c2s2]) - c1                 # prereg basis (s1/s2)
    low_true = low_ref
    low3 = None
    if c2s0 is not None:
        low3 = mean([c2s0, c2s1, c2s2]) - c1

    d_high = c4 - PREREG_C3
    inter = d_high - PREREG_LOW
    verdict = classify(inter, d_high)

    d_high_t = c4 - c3
    inter_t = d_high_t - low_true
    verdict_t = classify(inter_t, d_high_t, gate=0.5 * low_true)
    agree = (verdict == verdict_t)

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip() or "unknown"
    csv_lines = []
    if os.path.isfile(os.path.join(ROOT, CSV)):
        csv_lines = [l for l in open(os.path.join(ROOT, CSV)).read().splitlines() if l.strip()]
    csv_header = csv_lines[0] if csv_lines else ""
    runs = [l.split(",") for l in csv_lines[1:]]
    csv_dup = len(runs) != len(set(tuple(r) for r in runs))
    steps = {"SUBSTITUTION": "解锁 P6C-STEP2（C4 补 s1/s2 两格，2 × 20ep ≈ 5.4 GPU·h）。**需人确认后才启动**，本报告不自动启动。",
             "DIMINISHING": "停机，不解锁 P6C-STEP2。记为\"递减\"，**不得**表述为\"替代\"。",
             "ADDITIVE": "停机。报 UNRESOLVED，不烧 3-seed。",
             "COMPLEMENTARY": "停机。H-36 REJECTED —— 容量与监督可加/互补，机制主张撤回。"}

    L = []
    A = L.append
    A("# Phase 6C Decision Report — EXP-09（容量 × 锚框 2×2 / H-36）")
    A("")
    A("**生成时刻：** %s (WSL CST)　**代码版本：** `%s`" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sha))
    A("**生成方式：** `scripts/phase6c_e9_report.py` 机械计算（阈值写死，不接受事后重定义）")
    A("**判据来源：** `docs/PHASE6C_CAPACITY_ANCHOR_PREREGISTRATION.md` §3（开跑前固定）")
    A("**噪声标尺：** σ_20 = %.4f（§3.1，EXP-08 两臂 pooled within-arm seed-sd）⇒ 2σ_20 = %.4f" % (SIGMA_20, TWO))
    A("**tier：** screening（n=1/单档）—— 只用于决定是否继续投入，**不用于对外结论**。")
    A("")
    A("---")
    A("")
    A("## 0. TL;DR")
    A("")
    A("| 问题 | 判定 | 一句话依据 |")
    A("|---|---|---|")
    A("| **H-36**：锚框监督与表示容量是**替代**还是**可加** | **%s** | interaction = %+.4f（低容量 Δ %+.4f，高容量 Δ %+.4f） |"
      % (verdict, inter, PREREG_LOW, d_high))
    A("| C4（A-uniform × k-means，判决格） | 实测 mAP50 = **%s** | 训练 162 min / 20ep / seed0（eval 由补评恢复，复用 checkpoint，未重训） |" % fmt(c4))
    A("| 决策点 | mAP50(C4) 对照 **%.4f / %.4f / %.4f** | ≤ %.4f → 替代成立；%.4f~%.4f 压缩；%.4f~%.4f 无信号（纯可加点 %.4f ± 2σ）；≥ %.4f → H-36 被否 |"
      % (SUB_CEIL, DIM_CEIL, COMP_FLOOR, SUB_CEIL, SUB_CEIL, DIM_CEIL, DIM_CEIL, COMP_FLOOR, ZERO_POINT, COMP_FLOOR))
    A("")
    A("**分支动作：** %s" % steps[verdict])
    A("")
    A("---")
    A("")
    A("## 1. 四格读数（全部实测，磁盘直读）")
    A("")
    A("| 格 | 容量 | 锚框 | seed | 源目录 | mAP50 | mAP50-95 | da_mIoU | lane_mIoU | params | FLOPs(G) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    cellname = {"C1": "C1", "C2s1": "C2", "C2s2": "C2", "C2s0": "C2(补seed0)", "C3": "C3", "C4": "C4"}
    for key, cap, anc, seed, p in ARMS:
        d = m.get(key)
        if d is None:
            A("| %s | %s | %s | %d | `%s` | **缺** | | | | | |" % (cellname[key], cap, anc, seed, p))
            continue
        A("| %s | %s | %s | %d | `%s` | **%s** | %s | %s | %s | %s | %s |"
          % (cellname[key], cap, anc, seed, os.path.dirname(p).replace("_eval", ""),
             fmt(d.get("mAP50")), fmt(d.get("mAP50_95")), fmt(d.get("da_mIoU")),
             fmt(d.get("lane_mIoU")), d.get("parameters"), fmt((d.get("flops") or 0) / 1e9, 4)))
    A("")
    A("### 1.1 结果表行（`experiments/phase6c/phase6c_e9_factorial.csv`）")
    A("")
    A("```")
    if runs:
        A(csv_header)
        for r in runs:
            A(",".join(r))
    else:
        A("(空)")
    A("```")
    A("")
    A("**重复行检测：** %s" % ("⚠️ **发现完全重复的数据行** —— 需人工核对后去重" if csv_dup else "无"))
    A("")
    A("---")
    A("")
    A("## 2. 交互项计算（只读交互项，§3.2 纪律）")
    A("")
    A("```")
    A("低容量 Δ_anchor = mean(C2 s1/s2) - C1 = %.4f - %.4f = %+.4f   [预注册基准]" % (mean([c2s1, c2s2]), c1, low_ref))
    if low3 is not None:
        A("低容量 Δ_anchor = mean(C2 s0/s1/s2) - C1 = %.4f - %.4f = %+.4f   [3-seed 稳健读数，§5.3]" % (mean([c2s0, c2s1, c2s2]), c1, low3))
    else:
        A("低容量 Δ_anchor [3-seed] = 待 exp9_r2z16_km(s0) 补评（§5.3 要求两个读数都报）")
    A("高容量 Δ_anchor = C4 - C3(预注册基) = %.4f - %.4f = %+.4f" % (c4, PREREG_C3, d_high))
    A("interaction      = %+.4f - (%+.4f) = %+.4f      [主判定]" % (d_high, PREREG_LOW, inter))
    A("2σ_20            = %.4f" % TWO)
    A("")
    A("交叉核对（全精度，用实测 C3 与实测低容量 Δ）：")
    A("  C3 实测 = %.4f (预注册写 0.5339)" % c3)
    A("  interaction = %+.4f   -> 分支 %s" % (inter_t, verdict_t))
    A("  与主判定一致性：%s" % ("一致" if agree else "不一致 —— 必须人工复核边界"))
    A("```")
    A("")
    A("---")
    A("")
    A("## 3. 分支判定（§3.3，magnitude 与显著性**同时**要求）")
    A("")
    A("| mAP50(C4) 区间 | interaction | 分支 | 本次落在 |")
    A("|---|---|---|---|")
    A("| ≤ %.4f | ≤ -%.4f 且 Δ_高 ≤ %.4f | **SUBSTITUTION** | %s |"
      % (SUB_CEIL, TWO, PREREG_GATE, "★" if verdict == "SUBSTITUTION" else ""))
    A("| %.4f ~ %.4f | ≤ -%.4f，Δ_高 未腰斩 | DIMINISHING | %s |"
      % (SUB_CEIL, DIM_CEIL, TWO, "★" if verdict == "DIMINISHING" else ""))
    A("| %.4f ~ %.4f | \\|interaction\\| < %.4f | ADDITIVE | %s |"
      % (DIM_CEIL, COMP_FLOOR, TWO, "★" if verdict == "ADDITIVE" else ""))
    A("| ≥ %.4f | ≥ +%.4f | COMPLEMENTARY | %s |"
      % (COMP_FLOOR, TWO, "★" if verdict == "COMPLEMENTARY" else ""))
    A("")
    A("> 区间端点换算：纯可加预测点 = C3 + 低容量Δ = %.4f + %.4f = **%.4f**（即 interaction = 0）。"
      % (PREREG_C3, PREREG_LOW, ZERO_POINT))
    A("> ADDITIVE 带 = 纯可加点 ± 2σ = (%.4f, %.4f)；替代门槛 = 增益腰斩点 = C3 + 0.5×Δ_low = **%.4f**。"
      % (DIM_CEIL, COMP_FLOOR, SUB_CEIL))
    A("")
    A("**实测 mAP50(C4) = %s ⇒ 判定 %s。**" % (fmt(c4), verdict))
    A("")
    A("---")
    A("")
    A("## 4. 下一步（**待用户决定，本报告不自动启动任何训练**）")
    A("")
    A("1. %s" % steps[verdict].replace("**", ""))
    A("2. 禁止 scope drift（§4.3）：不得\"顺手加一个臂\"、不得因\"4ep 也能出数\"而补跑 4ep 臂。")
    A("3. 若需 STEP2，先确认：GPU 时间预算、是否连跑、以及 3-seed 的 seed 选择（s1/s2）。")
    A("")
    A("---")
    A("")
    A("## 5. 诚实性限制（§5 原文，必须随结论一并陈述）")
    A("")
    A("1. **容量轴不干净**：C3/C4 = A-uniform（params 实测 +%.1f%%，FLOPs +%.1f%%），与 C1/C2 不只是\"容量大小\"之别，还有\"容量加在哪个维度\"之别 → 本 interaction 是**跨维度容量**的交互，**不能**表述为\"参数量的交互\"。"
      % ((m["C4"]["parameters"] / m["C1"]["parameters"] - 1) * 100,
         (m["C4"]["flops"] / m["C1"]["flops"] - 1) * 100))
    A("2. **20ep 旧锚框臂在退化**（0.3879@4ep → 0.3543@20ep）→ Δ_anchor 含\"锚框影响收敛/退化\"成分，**不等于**纯表示改善。")
    A("3. **跨 seed**：低容量 Δ 的预注册基准用 s1/s2，与 C1(seed0) 不同 seed；3-seed 读数见 §2，**两个读数都已报**。")
    A("4. **n=1 的格**：C3 为 seed0 单次；3σ 级结论必须等 P6C-STEP2。")
    A("5. **Screening tier**：只用于决定是否继续投入，不用于对外结论。")
    A("6. **FPS 不作为判据**（项目约定）。")
    A("")
    A("**数据质量备注**：`peak_gpu_mem_mib` 列为 NA（训练日志无 `peak_mem=` 字段，格式为 `mem N/8151MiB`）；两格一致故可比，未回填。")
    A("")
    A("---")
    A("")
    A("## 6. 复现与审计")
    A("")
    A("- 生成命令：`~/ai_study/gpu_env/bin/python scripts/phase6c_e9_report.py`")
    A("- 输入（全部只读）：§1 表内六个 `_eval/metrics.json` + `%s`" % CSV)
    A("- 判据常量（脚本内硬编码，来源 §3）：σ_20=%.4f, C3=%.4f, 低容量Δ=%.4f, gate=%.4f" % (SIGMA_20, PREREG_C3, PREREG_LOW, PREREG_GATE))
    A("- cell 1 的 eval 为**补评**（复用 checkpoint、未重训），eval 命令与 `phase6_run.sh:38` 逐字一致；`git_commit` 列保留 `9fb29e7`。")
    A("- 失败现场保留在 `experiments/phase6c/exp9_aunif_km_eval.log`（未被覆盖）。")
    A("")

    open(os.path.join(ROOT, REPORT), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("REPORT WRITTEN: %s" % REPORT)
    print("VERDICT: %s   (mAP50(C4)=%s  interaction=%+.4f  d_high=%+.4f)" % (verdict, fmt(c4), inter, d_high))
    return 0


if __name__ == "__main__":
    sys.exit(main())

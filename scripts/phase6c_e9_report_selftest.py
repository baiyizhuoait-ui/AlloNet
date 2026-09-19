#!/usr/bin/env python3
"""Reverse-case selftest for phase6c_e9_report.py.

Synthetic metrics in a sandbox tree; asserts the mechanical verdict equals the
hand-computed branch for each of the four branches plus the two boundary cases.
A gate that cannot fail is not a gate -- so this also asserts the generator
REFUSES to emit a report when the decisive cell is missing.
"""
import json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "phase6c_e9_report.py")

REF = {
 "experiments/phase4a/exp4A_r2_z16_e20_eval/metrics.json":      0.3543,
 "experiments/phase4a/exp4B_danc_z16_e20_s1_eval/metrics.json": 0.4982,
 "experiments/phase4a/exp4B_danc_z16_e20_s2_eval/metrics.json": 0.4960,
 "experiments/phase6/exp6_e8_unif20_eval/metrics.json":         0.5339,
 "experiments/phase6c/exp9_r2z16_km_eval/metrics.json":         0.4975,
}
C4 = "experiments/phase6c/exp9_aunif_km_eval/metrics.json"

CASES = [   # (C4 mAP50, expected branch)
 (0.5900, "SUBSTITUTION"),
 (0.6053, "SUBSTITUTION"),   # 腰斩边界：Δ_高 = 0.0714 = gate -> 仍算替代
 (0.6054, "DIMINISHING"),    # 刚过边界
 (0.6200, "DIMINISHING"),
 (0.6671, "DIMINISHING"),    # interaction = -0.0096 = -2σ 仍算
 (0.6672, "ADDITIVE"),
 (0.6720, "ADDITIVE"),
 (0.6767, "ADDITIVE"),       # 纯可加预测点本身 -> interaction = 0 -> 无信号
 (0.6862, "ADDITIVE"),       # +2σ 之下
 (0.6864, "COMPLEMENTARY"),  # +2σ 之上（精确边界 0.6863 浮点敏感，两侧夹测）
 (0.7000, "COMPLEMENTARY"),
]

def payload(v, params=333862, flops=1664992000.0):
    return dict(mAP50=v, mAP50_95=0.24, da_mIoU=0.85, da_fg_iou=0.77,
                lane_mIoU=0.58, lane_fg_iou=0.20, parameters=params, flops=flops, fps=400.0)

def build(root, c4=None):
    for rel, v in REF.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        pr = 201366 if "z16" in rel else 333862
        fl = 1079628800.0 if "z16" in rel else 1664992000.0
        json.dump(payload(v, pr, fl), open(p, "w"))
    if c4 is not None:
        p = os.path.join(root, C4)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(payload(c4), open(p, "w"))

def run(root):
    r = subprocess.run([sys.executable, GEN], capture_output=True, text=True,
                       env=dict(os.environ, P6C_REPORT_ROOT=root))
    return r.returncode, r.stdout + r.stderr

fails = 0
for c4v, want in CASES:
    root = tempfile.mkdtemp()
    build(root, c4v)
    rc, out = run(root)
    got = "?"
    for line in out.splitlines():
        if line.startswith("VERDICT:"):
            got = line.split()[1]
    ok = (rc == 0 and got == want)
    rep = os.path.join(root, "experiments/phase6c/phase6c_exp9_decision.md")
    ok = ok and os.path.isfile(rep) and os.path.getsize(rep) > 1500
    print("%-6s C4=%.4f  expect=%-14s got=%-14s rc=%d  report=%s"
          % ("PASS" if ok else "FAIL", c4v, want, got, rc,
             os.path.getsize(rep) if os.path.isfile(rep) else "MISSING"))
    fails += 0 if ok else 1
    shutil.rmtree(root, ignore_errors=True)

# 反例：缺 C4 必须拒发报告
root = tempfile.mkdtemp(); build(root, None)
rc, out = run(root)
rep = os.path.join(root, "experiments/phase6c/phase6c_exp9_decision.md")
ok = (rc != 0 and not os.path.isfile(rep))
print("%-6s missing-C4 must refuse (rc=%d, no report=%s)" % ("PASS" if ok else "FAIL", rc, not os.path.isfile(rep)))
fails += 0 if ok else 1
shutil.rmtree(root, ignore_errors=True)

print("\nSELFTEST %s (%d failure(s))" % ("PASS" if fails == 0 else "FAIL", fails))
sys.exit(1 if fails else 0)

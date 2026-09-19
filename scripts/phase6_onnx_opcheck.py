"""What operators would a vendor compiler actually have to place?

Deployment risk is not FLOPs.  It is: (a) does every operator in the graph have
a hardware path on the target NPU, and (b) does the graph contain an op that
forces a fallback to the host CPU (Hailo's compiler, Arm's Vela and RKNN all
reject or fall back on unsupported ops, and a fallback usually costs more than
the whole model -- measured fixed overheads on these parts are 8-12 ms).

This script traces the trained model and prints the ATen operator histogram,
grouped into the deployment ops a toolchain would emit.  It is a proxy for an
ONNX export, NOT a vendor compiler result: a real answer needs RKNN / nncase /
Hailo DFC / Vela on the target.  What it does establish is the shape of the
graph, which is what decides whether that call is worth making.

CPU only, a few seconds, ~400 MB.
"""
import argparse
import json
import os
import sys
from collections import Counter

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import models.static_model  # noqa: E402,F401
from models.static_model import StaticMultiTaskModel  # noqa: E402

# aten op -> the deployment op a toolchain emits.  `folded` = routinely fused
# away by conv+BN folding and never reaches the scheduler as its own op.
GROUP = {
    "conv2d": ("Conv", False),
    "batch_norm": ("BatchNorm(folded into Conv)", True),
    "relu": ("ReLU", False),
    "upsample_bilinear2d": ("Resize(bilinear)", False),
    "cat": ("Concat", False),
    "sigmoid": ("Sigmoid", False),
    "mul": ("Mul(elementwise)", False),
    "add": ("Add(elementwise)", False),
    "sub": ("Sub(elementwise)", False),
    "div": ("Div(elementwise)", False),
    "pow": ("Pow(elementwise)", False),
    "view": ("Reshape", True),
    "reshape": ("Reshape", True),
    "permute": ("Transpose", True),
    "contiguous": ("(layout, no-op)", True),
    "to": ("(cast, no-op)", True),
    "clone": ("(copy, no-op)", True),
    "expand": ("Broadcast", True),
    "arange": ("(const)", True),
    "meshgrid": ("(const)", True),
    "stack": ("(const)", True),
    "zeros": ("(const)", True),
    "flatten": ("Reshape", True),
    "type_as": ("(cast, no-op)", True),
    "size": ("(shape)", True),
}

# ops that are known to be unsupported or slow on at least one mainstream
# low-cost edge toolchain -> a first-order fallback risk
RISK = {
    "Resize(bilinear)": "Hailo-8/8L and some Ethos-U configs support nearest/"
                        "limited resize; bilinear upsample to large sizes is a "
                        "known placement risk and a common CPU-fallback trigger.",
    "Pow(elementwise)": "`**2` on the box branch; some compilers refuse non-"
                        "integer exponents and fall back.  Equivalent to Mul "
                        "with itself and is trivially rewritable if rejected.",
    "Concat": "supported everywhere, but it is the op where our detection "
              "probabilities share a quantisation scale with box coordinates "
              "(see phase6_quant_head_audit.py).",
}


def collect(mod, out, depth=0):
    try:
        g = mod.graph
        if callable(g):        # top-level ScriptModule exposes .graph as a method
            g = g()
        nodes = g.nodes
        if callable(nodes):    # some versions: nodes() is a method
            nodes = nodes()
        for n in nodes:
            out[n.kind().replace("aten::", "")] += 1
    except Exception:          # noqa: BLE001 -- leaf modules carry no graph
        pass
    for child in mod.children():
        collect(child, out, depth + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/phase6_r4_R2_thin14_z16.yaml")
    ap.add_argument("--hw", default="640")
    ap.add_argument("--outdir", default="experiments/phase6/deploy_profile")
    args = ap.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    model = StaticMultiTaskModel(cfg["model"]).eval()
    h = int(args.hw)
    x = torch.randn(1, 3, h, h)

    res = {"cfg": args.cfg, "hw": [h, h]}
    try:
        with torch.no_grad():
            model(x)          # warm up: the head caches its decode grid, and the
                              # dynamic-width layers specialise on the first call.
                              # Without this the two trace invocations differ and
                              # the sanity check (correctly) refuses the trace.
            traced = torch.jit.trace(model, x, strict=False, check_trace=False)
        raw = Counter()
        collect(traced, raw)
        res["traced"] = True
        res["aten_histogram"] = dict(raw.most_common())
        grouped = Counter()
        for k, v in raw.items():
            g = GROUP.get(k, (f"aten::{k}", False))[0]
            grouped[g] += v
        res["deployment_ops"] = dict(grouped.most_common())
        risk_hits = {g: n for g, n in res["deployment_ops"].items() if g in RISK}
        res["fallback_risk_ops"] = risk_hits
        res["risk_notes"] = {g: RISK[g] for g in risk_hits}
        res["unknown_ops"] = [k for k in raw if k not in GROUP]
    except Exception as e:                                # noqa: BLE001
        import traceback
        res["traced"] = False
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-2000:]

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "onnx_opcheck.json")
    with open(out + ".tmp", "w") as f:
        json.dump(res, f, indent=2)
    os.replace(out + ".tmp", out)
    print(json.dumps(res, indent=2)[:4000])
    print("saved ->", out)


if __name__ == "__main__":
    main()

"""Where does the bit-width cliff actually sit for Model B?

The question this answers is "if INT8 works, is INT4/FP4 worth anything to us?"
and it is answered by measurement, not by preference.

For each bit-width we run the same PTQ fake-quantisation protocol with the
DETECTION-HEAD quantiser present (per-channel, i.e. the best realistic
configuration found by phase6_quant_head_audit.py) and report every axis, so
the cliff can be read off directly rather than inferred.

What we already know from the 8-bit and 6-bit probes in phase6_deploy_profile.py:
DA held up at 6 bits (99.47% agreement) while lane foreground IoU collapsed
(0.745 vs 0.948).  So the interesting region is 8 -> 4 and the interesting
question is which axis breaks FIRST.

Reality check that the paper has to repeat: no cheap edge accelerator we can
find offers a native 4-bit path (Coral Edge TPU and Hailo-8 are INT8-only;
RKNN/Ethos-U are INT8).  INT4 appears on Hailo-10H, Qualcomm Hexagon HMX and
Apple ANE; NVFP4 appears on Jetson Thor (Blackwell).  So a 4-bit result is
evidence about a FUTURE/phone-class target, not about the target this model was
designed for.

CPU only, <= ~900 MB, does not touch the GPU.
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import models.static_model  # noqa: E402,F401  (pin before any baseline tree)
from phase6_quant_head_audit import (  # noqa: E402
    Sim, load_model, mask_stats, det_stats, mem_available_mb, MEM_FLOOR_MB)

from datasets.bdd100k import BDD100KDataset  # noqa: E402


def argmax_maps(outs):
    return [(o[1].argmax(1)[0].to(torch.uint8).numpy(),
             o[2].argmax(1)[0].to(torch.uint8).numpy()) for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/phase6_r4_R2_thin14_z16.yaml")
    ap.add_argument("--ckpt", default="experiments/phase6/final/B100/checkpoint.pt")
    ap.add_argument("--data", default="data/bdd100k")
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--bits", default="8,7,6,5,4")
    ap.add_argument("--outdir", default="experiments/phase6/deploy_profile")
    args = ap.parse_args()

    avail = mem_available_mb()
    if avail is not None and avail < MEM_FLOOR_MB:
        print(f"[guard] MemAvailable {avail} MB < {MEM_FLOOR_MB} MB -> refuse",
              file=sys.stderr)
        sys.exit(2)
    torch.set_num_threads(4)

    model, info = load_model(args.cfg, args.ckpt)
    print(f"[load] params={info['params']} missing={info.get('missing')} "
          f"epoch={info.get('epoch')} mem_avail={avail} MB")

    ds = BDD100KDataset(args.data, split="tri_val", img_size=640,
                        with_det=False, with_da=False, with_lane=False)
    n = min(len(ds), max(args.n, 40))
    imgs = [ds[i]["image"] for i in range(n)]
    calib, ev = imgs[:30], imgs[30:]

    s_fp = Sim(model, 8).install()
    fp = s_fp.run(ev, "off")
    s_fp.close()
    ref_det = torch.cat([o[0] for o in fp], 0)
    ref_ag = argmax_maps(fp)
    del fp
    gc.collect()

    bits_list = [int(b) for b in args.bits.split(",")]
    rows = []
    for bits in bits_list:
        s = Sim(model, nbits=bits, observer="p9999",
                head_mode="per_channel").install()
        s.run(calib, "calib")
        outs = s.run(ev, "apply")
        step = getattr(s, "last_head_step", None)
        ag = argmax_maps(outs)
        acc = {}
        for i, o in enumerate(outs):
            st = det_stats(o[0], ref_det[i:i + 1])
            for k, val in st.items():
                if val is not None:
                    acc.setdefault(k, []).append(val)
        d = [mask_stats(ref_ag[i][0], ag[i][0]) for i in range(len(ev))]
        l = [mask_stats(ref_ag[i][1], ag[i][1]) for i in range(len(ev))]
        row = {
            "bits": bits,
            "da_pixel_agreement": round(float(np.mean([x[0] for x in d])), 5),
            "da_fg_iou": round(float(np.nanmean([x[1] for x in d])), 5),
            "lane_pixel_agreement": round(float(np.mean([x[0] for x in l])), 5),
            "lane_fg_iou": round(float(np.nanmean([x[1] for x in l])), 5),
            "det_obj_zero_frac": round(float(np.mean(acc["obj_frac_rounded_to_zero"])), 5),
            "det_boxes_preserved": round(float(np.mean(acc["boxes_preserved_frac"])), 5),
            "det_nms_boxes_c25": round(float(np.mean(acc["nms_boxes_c25"])), 2),
            "head_output_step": None if step is None else round(step, 6),
        }
        rows.append(row)
        s.close()
        del outs, ag
        gc.collect()
        print(json.dumps(row, ensure_ascii=False))

    res = {"model": info, "n_calib": len(calib), "n_eval": len(ev),
           "head_mode": "per_channel", "observer": "p9999",
           "mem_avail_mb": avail, "rows": rows,
           "_note": "PTQ fake-quant with the detection-head quantiser present; "
                    "no bias correction / AdaRound / QAT, so every row is a "
                    "LOWER bound.  Simulation, not a vendor toolchain."}
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "quant_bitsweep.json")
    with open(out + ".tmp", "w") as f:
        json.dump(res, f, indent=2)
    os.replace(out + ".tmp", out)
    print("saved ->", out)


if __name__ == "__main__":
    main()

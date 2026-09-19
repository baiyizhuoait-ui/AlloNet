"""Inverted-residual multi-scale encoder (Module A, second architecture).

WHY THIS FILE EXISTS
--------------------
Phase 6 / G3 asks whether the "marginal price -> allocation rule" result
transfers to a *second architecture*.  Holding the encoder family fixed and
only rescaling `stages` / `blocks` does NOT answer that: uniform rescaling is
exactly the baseline practice (Model U = uniform encoder x1.40), so a rescaled
encoder would re-measure the same thing and the argument would be circular.

This encoder is therefore a different *topology*, from a different lineage:

    light_encoder.py   ESPNet / depthwise-separable lineage
                       - each stage: DW+PW then DW+PW, ReLU after the add
                       - residual lives on the WIDE (output) tensor
                       - downsample = stride-2 on the first DW conv of a block
                       - stem goes 1/2 then a stride-2 block reaches 1/4

    ir_encoder.py      MobileNetV2/V3 inverted-residual lineage  (this file)
                       - each stage: 1x1 EXPAND -> kxk DW -> 1x1 PROJECT
                       - residual lives on the NARROW (bottleneck) tensor and
                         there is NO activation after the residual add
                       - downsample = stride on the depthwise conv, and the
                         shortcut is dropped whenever stride != 1
                       - stem reaches 1/4 in one shot (two stride-2 convs)
                       - mixed 3x3 / 5x5 depthwise kernels across stages

The activation is plain ReLU on purpose: it keeps the operator set inside the
{conv, BN, ReLU, resize, concat} family that ships on INT8-only NPUs, so a
quantisation conclusion drawn here stays comparable to the one drawn on the
ESPNet-lineage encoder.  No attention, no SE, no hard-swish.

INTERFACE CONTRACT (must match LightEncoder exactly)
----------------------------------------------------
    __init__(cfg)                cfg = {"stem": int, "stages": [c1,c2,c3,c4],
                                        "blocks": [...], ...}
    forward(x, highres=False)    -> [F2, F3, F4]                      (1/8, 1/16, 1/32)
                                 -> ([F2, F3, F4], {"f0":.., "f1":..}) when highres

`stages` keeps the SAME meaning as in light_encoder.py -- it is the channel
width of the four exported feature levels:

    stages[0] -> f1, 1/4   (consumed by lane_lat / da_lat / DetFromZ p2)
    stages[1] -> F2, 1/8   (CompactRepresentation)
    stages[2] -> F3, 1/16
    stages[3] -> F4, 1/32

StaticMultiTaskModel indexes enc_cfg["stages"] directly, so the exports must
respect these widths exactly or the model will not build.
"""
import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """MobileNetV2-style block: expand -> depthwise -> project (linear).

    t=1 degenerates to a plain depthwise-separable conv (no expand/project
    round trip), which is kept so a config can dial the block down.
    """

    def __init__(self, cin, cout, k=3, s=1, expand=4):
        super().__init__()
        mid = int(round(cin * expand))
        self.use_res = (s == 1 and cin == cout)
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, mid, 1, bias=False),
                       nn.BatchNorm2d(mid),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(mid, mid, k, s, k // 2, groups=mid, bias=False),
                   nn.BatchNorm2d(mid),
                   nn.ReLU(inplace=True)]
        # projection: linear bottleneck -- NO activation after it
        layers += [nn.Conv2d(mid, cout, 1, bias=False),
                   nn.BatchNorm2d(cout)]
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        out = self.f(x)
        return out + x if self.use_res else out


class InvertedResidualEncoder(nn.Module):
    """Multi-scale inverted-residual encoder.

    cfg keys
      stem    : int, width at 1/4 (the second stem conv output).       default 16
      stages  : [c1, c2, c3, c4] widths of f1(1/4), F2, F3, F4.        default [32,64,96,128]
      blocks  : either 3 entries (stages 2..4, like light_encoder) or 4
                entries (stage 1..4).  Missing stage-1 count comes from
                `blocks_first` (default 1).
      expand  : int or list of 4 ints, expansion ratio per stage.      default 2
      kernels : list of kernel sizes used by the depthwise convs;
                the stage index picks from it.                        default [3, 3, 5, 5]
    """

    def __init__(self, cfg):
        super().__init__()
        c_stem = int(cfg.get("stem", 16))
        c1, c2, c3, c4 = cfg.get("stages", [32, 64, 96, 128])
        out_w = [int(c1), int(c2), int(c3), int(c4)]

        blocks = list(cfg.get("blocks", [2, 2, 2]))
        if len(blocks) == 3:
            blocks = [int(cfg.get("blocks_first", 1))] + [int(b) for b in blocks]
        elif len(blocks) == 4:
            blocks = [int(b) for b in blocks]
        else:
            raise ValueError(
                "`blocks` must have 3 entries (stages 2..4) or 4 entries "
                "(stages 1..4); got %r" % (blocks,))

        expand = cfg.get("expand", 2)
        if isinstance(expand, (list, tuple)):
            ex = [float(e) for e in expand]
            if len(ex) != 4:
                raise ValueError("`expand` as a list needs 4 entries")
        else:
            ex = [float(expand)] * 4

        kern = list(cfg.get("kernels", [3, 3, 5, 5]))
        if len(kern) != 4:
            raise ValueError("`kernels` needs 4 entries")

        # ---- stem: 3 -> c_stem//2 -> c_stem, both stride 2 -> reaches 1/4 ----
        c_half = max(4, c_stem // 2)
        self.stem0 = ConvBNAct(3, c_half, 3, 2)      # f0, 1/2
        self.stem1 = ConvBNAct(c_half, c_stem, 3, 2)  # 1/4

        # ---- four stages, outputs at 1/4, 1/8, 1/16, 1/32 ----
        # stage 1 does NOT downsample (the stem already delivered 1/4), which is
        # the second structural difference from light_encoder.
        cin = c_stem
        stages = []
        for si in range(4):
            stride_first = 1 if si == 0 else 2
            layers = []
            for bi in range(blocks[si]):
                s = stride_first if bi == 0 else 1
                layers.append(InvertedResidual(cin, out_w[si],
                                               k=int(kern[si]), s=s,
                                               expand=ex[si]))
                cin = out_w[si]
            stages.append(nn.Sequential(*layers))
        self.s1, self.s2, self.s3, self.s4 = stages

    def forward(self, x, highres=False):
        """highres=True also returns the 1/2 and 1/4 maps (contract parity
        with LightEncoder: StaticMultiTaskModel reads extra["f1"])."""
        f0 = self.stem0(x)    # 1/2
        x4 = self.stem1(f0)   # 1/4
        f1 = self.s1(x4)      # 1/4, stages[0] channels
        f2 = self.s2(f1)      # 1/8
        f3 = self.s3(f2)      # 1/16
        f4 = self.s4(f3)      # 1/32
        if highres:
            return [f2, f3, f4], {"f0": f0, "f1": f1}
        return [f2, f3, f4]

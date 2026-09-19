#!/usr/bin/env python3
"""Phase 6 Round 3 -- EXP-9B: lane spatial-vs-channel arms on TwinLiteNetPlus.

WHY THIS FILE EXISTS (measured premise, see phase6_round3_architecture_audit.md)
------------------------------------------------------------------------------
TwinLiteNetPlus' lane head receives THREE inputs:
    out_caam  1/8  -- the only LEARNED semantic feature
    inp2      1/4  -- 3-channel AVERAGE-POOLED RAW RGB (encoder.sample2)
    inp1      1/2  -- 3-channel AVERAGE-POOLED RAW RGB (encoder.sample1)
So the lane head does already see 1/2 and 1/4 *resolution*, but only as unlearned
RGB averages -- no high-resolution LEARNED feature reaches it.  The reachable
"spatial" headroom is therefore not more resolution but a LEARNED 1/4 tap.

Design (both interventions sit at the SAME 1/4 stage, on the SAME tensor):

  B0 baseline  stock.
  B1 spatial   sub2 = inp2 + lat(f14)          lat = 1x1 Conv(C14 -> 3)
               f14 = encoder.b2 output (1/4, learned), captured by a forward hook
               -- a NEW higher-resolution LEARNED input source.
  B2 channel   ll  = ll + widen(ll)            widen = 1x1 C0->dc->C0 bottleneck
               at the 1/4 stage, NO new input source -- pure channel capacity.

FAIRNESS, mechanically enforced:
  * The last layer of every added path is ZERO-INITIALISED, so at step 0 both arms
    are bit-identical to baseline.  Verified by `selfcheck()` below, not asserted
    in prose.
  * `dc` is solved so FLOPs(B2) = FLOPs(B1) within the project's +/-5 % budget-
    equivalence tolerance, then VERIFIED with the real FLOPs counter (not with the
    analytic formula alone).
  * DA branch is untouched in every arm (spec: no new DA intervention).
  * All pretrained lane weights load unchanged -- no weight surgery, so every arm
    starts from the same released checkpoint.
"""
import argparse

import torch
import torch.nn as nn

from model import config as tlp_cfg            # noqa: E402  (baselines/TwinLiteNetPlus)
from model.model import TwinLiteNetPlus        # noqa: E402

ARMS = ("baseline", "spatial", "channel")


def channel_plan(preset):
    """Return (C14, C0) -- the two widths the parity formula needs."""
    c = tlp_cfg.sc_ch_dict[preset]["chanels"]
    return c[3] + tlp_cfg.chanel_img, c[0]


def solve_dc(preset):
    """dc such that 2*C0*dc == C14*3  (the 1/4-stage spatial area cancels)."""
    C14, C0 = channel_plan(preset)
    return max(1, int(round(C14 * 3.0 / (2.0 * C0))))


class Round3TLP(nn.Module):
    def __init__(self, preset, arm="baseline", dc=None):
        super().__init__()
        assert arm in ARMS, arm
        self.preset = preset
        self.arm = arm
        self.net = TwinLiteNetPlus(argparse.Namespace(config=preset))
        C14, C0 = channel_plan(preset)
        self.C14, self.C0 = C14, C0
        self._cap = None
        self.dc = 0

        if arm == "spatial":
            self.lat = nn.Conv2d(C14, 3, 1)
            nn.init.zeros_(self.lat.weight)
            nn.init.zeros_(self.lat.bias)
            self.net.encoder.b2.register_forward_hook(self._hook)
        elif arm == "channel":
            self.dc = int(dc if dc is not None else solve_dc(preset))
            self.widen = nn.Sequential(
                nn.Conv2d(C0, self.dc, 1, bias=False),
                nn.BatchNorm2d(self.dc),
                nn.PReLU(self.dc),
                nn.Conv2d(self.dc, C0, 1, bias=False),
                nn.BatchNorm2d(C0),
            )
            nn.init.zeros_(self.widen[-1].weight)   # zero-init block => identity at step 0
            # BatchNorm2d(C0) default init: weight=1, bias=0, running_var=1 -> a zero
            # conv output stays exactly zero through BN in train and eval mode.

    def _hook(self, module, inp, out):
        self._cap = out

    def load_pretrained(self, state_dict, strict=True):
        """Load the released checkpoint into the wrapped net (new paths excluded)."""
        cur = self.net.state_dict()
        compat = {k: v for k, v in state_dict.items() if k in cur and cur[k].shape == v.shape}
        missing, unexpected = self.net.load_state_dict(compat, strict=False)
        if strict:
            assert not unexpected, "unexpected keys: %s" % list(unexpected)[:5]
        return len(compat), list(missing)

    def forward(self, x):
        net = self.net
        oe, inp1, inp2 = net.encoder(x)              # hook fills self._cap for 'spatial'
        oc = net.conv_caam(net.caam(oe))

        da = net.out_da(net.up_2_da(net.up_1_da(oc, inp2), inp1))     # DA: untouched

        sub2 = inp2
        if self.arm == "spatial":
            sub2 = inp2 + self.lat(self._cap)
        ll = net.up_1_ll(oc, sub2)
        if self.arm == "channel":
            ll = ll + self.widen(ll)
        ll = net.out_ll(net.up_2_ll(ll, inp1))
        return da, ll

    def extra_params(self):
        if self.arm == "spatial":
            return sum(p.numel() for p in self.lat.parameters())
        if self.arm == "channel":
            return sum(p.numel() for p in self.widen.parameters())
        return 0


def selfcheck(preset="small", tol=1e-6):
    """Reverse-case gate: the added paths must be exact no-ops at step 0.

    The arms must be compared under IDENTICAL backbone weights -- freshly built
    modules each get their own random init, so comparing raw constructions would
    measure the init difference, not the intervention.  (The first version of this
    function did exactly that and failed; the gate caught its own author.)  Real
    arms all load weights/TwinLiteNetPlus_<preset>.pth, so sharing the baseline's
    state dict is also the faithful test.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    base = Round3TLP(preset, "baseline")
    ref_sd = {k: v.clone() for k, v in base.net.state_dict().items()}
    outs = {}
    for arm in ARMS:
        m = Round3TLP(preset, arm)
        n, miss = m.load_pretrained(ref_sd)
        assert not miss, "unexpected missing keys: %s" % miss[:5]
        m.eval()
        with torch.no_grad():
            outs[arm] = [t.clone() for t in m(x)]
    ok = True
    for arm in ("spatial", "channel"):
        dmax = max((outs[arm][i] - outs["baseline"][i]).abs().max().item() for i in (0, 1))
        flag = dmax <= tol
        ok &= flag
        print("  selfcheck %-8s max|delta| vs baseline = %.3e  %s" % (arm, dmax, "PASS" if flag else "FAIL"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        print("Round3TLP step-0 equivalence (preset=%s):" % a.preset)
        raise SystemExit(0 if selfcheck(a.preset) else 1)
    C14, C0 = channel_plan(a.preset)
    print("preset=%s C14=%d C0=%d -> dc=%d" % (a.preset, C14, C0, solve_dc(a.preset)))

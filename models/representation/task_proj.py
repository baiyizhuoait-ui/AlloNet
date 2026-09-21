"""Per-task projection off a shared Compact Z (Phase 4A Level 2, probe A).

Under R2, detection already owns a 1x1 projection while DA and lane read the
raw shared Z, so the shared Z width leaks straight into those two heads.
`TaskProj` gives each segmentation task its own explicit projection so the
width a task sees is independent of the shared Z width.

Design rules:
1. Widths are ABSOLUTE targets (e.g. {det:32, lane:32, da:16}) at every shared-Z
   width, so across a z16/z32 sweep only the shared Z itself changes.
2. in_ch == out_ch uses nn.Identity: a no-change projection is a true no-op,
   which keeps R3 comparable to R2 on arms where nothing should move.
3. Default off: without `task_proj` in the config, R0/R1/R2 build unchanged.

Cost is negligible (~800 params at z=16, <1% of the model).
"""
import torch
import torch.nn as nn


class TaskProj(nn.Module):
    """1x1 conv + BN + ReLU from the shared Z to one task's own width.

    Identity when in_ch == out_ch (see rule 2 above).
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        if self.in_ch == self.out_ch:
            self.op = nn.Identity()
        else:
            self.op = nn.Sequential(
                nn.Conv2d(self.in_ch, self.out_ch, 1, bias=False),
                nn.BatchNorm2d(self.out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, z):
        return self.op(z)

    def extra_repr(self):
        return "in=%d out=%d%s" % (
            self.in_ch, self.out_ch,
            " (identity)" if self.in_ch == self.out_ch else "")

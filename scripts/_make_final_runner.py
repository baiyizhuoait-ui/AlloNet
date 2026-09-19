#!/usr/bin/env python3
"""Create the paper-facing 100-epoch runner as a FROZEN COPY of the Round-4 runner.

Why a copy instead of editing phase6_round4_run.sh:
  Round 4 is closed, committed and archived. Appending the paper's main run into the
  round4 CSVs would muddy tables that are already final. The 100-epoch run is a
  different question ("same budget as the published baselines") and deserves its own
  directory and its own tables.

The copy is deliberately MINIMAL: four path/id substitutions, nothing else. The VRAM
guard carries the D6 (regex) and D9 (per-process `self` field) fixes verbatim; those
took a whole session to get right and must not be re-derived.

Assertions: every substitution must match EXACTLY once, so a future edit to the
source runner that invalidates an anchor fails loudly here instead of producing a
silently wrong script.
"""
import pathlib
import sys

SRC = pathlib.Path("scripts/phase6_round4_run.sh")
DST = pathlib.Path("scripts/phase6_final100_run.sh")

SUBS = [
    # (old, new, label)
    ("OUT=experiments/phase6/round4\n",
     "OUT=experiments/phase6/final\n",
     "output dir -> experiments/phase6/final"),
    ("CSV=experiments/phase6/phase6_round4_results.csv\n",
     "CSV=experiments/phase6/final_results.csv\n",
     "narrow table -> final_results.csv"),
    ("WIDE=experiments/phase6/phase6_round4_metrics.csv\n",
     "WIDE=experiments/phase6/final_metrics.csv\n",
     "wide table -> final_metrics.csv"),
    ('TR="$OUT/r4_${TAG}"\n',
     'TR="$OUT/${TAG}"\n',
     "arm dir drops the round4 prefix"),
]

s = SRC.read_text()
for old, new, label in SUBS:
    n = s.count(old)
    if n != 1:
        print("ANCHOR FAIL: %s  matched %d times (expected 1)" % (label, n))
        sys.exit(1)
    s = s.replace(old, new, 1)
    print("ok: %s" % label)

# The chain-control flag and the round4 chain log belong to the round-4 orchestrator.
# Keep STOP_CHAIN (harmless, and it is our abort lever) but rename the log so the two
# runners can never interleave writes into one file.
s = s.replace('"$OUT/round4_chain.log"', '"$OUT/final_chain.log"')
print("ok: chain log -> final_chain.log (%d occurrences)" % s.count("final_chain.log"))

DST.write_text(s)
print("written: %s  (%d bytes)" % (DST, len(s)))

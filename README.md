# AlloNet: Ultra-Lightweight Multi-Task Driving Perception

**AlloNet** (internal codename **TRAC**, *Task-Resource Adaptive Compact representation*) is a
0.19M-parameter multi-task network for driving scene understanding that jointly performs
**object detection**, **drivable area segmentation**, and **lane marking segmentation** on a
single compact shared representation.

The repository is built around a capacity-allocation study: under a fixed parameter budget,
*how should capacity be split between the shared encoder and the compact bottleneck (Z)?*
All experiments follow a **preregistered, audited protocol** — every run's config and metrics
are committed under `experiments/`, and headline numbers are backed by conformance ledgers.

## Headline Results

Final model (Protocol A: 640×640 input, our ground-truth annotations, 200-epoch training run):

| Metric | Value |
|---|---|
| Parameters | **0.193 M** (0.77 MB fp32) |
| FLOPs @ 640×640 | **1.17 G** |
| Detection mAP@0.5 | **0.545** |
| Drivable area mIoU | **0.877** |
| Lane mIoU | **0.600** |
| GPU memory (inference) | 34 MiB |

A second evaluation protocol (Protocol B: 384×384 input, official BDD100K ground truth)
is reported side-by-side in `experiments/phase6/paper_tables.md` to guard against
annotation-protocol bias. Lane metrics are protocol-sensitive and are therefore always
reported under the official GT.

Key findings of the capacity study (see `docs/PHASE0-3_CONSOLIDATED_REPORT.md` and the
Phase 4–6 reports in `docs/`):

- Task performance scales **monotonically** with encoder capacity across all six metrics,
  but at **different rates**: detection is the most sensitive task, while drivable area and
  lane segmentation saturate at the largest encoder.
- Re-allocating capacity toward the encoder at a fixed parameter budget (wider encoder,
  z=16 bottleneck) matches a wider-bottleneck baseline at **29.4% lower FLOPs**.
- The allocation result replicates across architectures (cross-architecture ledger in
  `experiments/phase6/g3/g3_ledger.txt`).

## Repository Layout

```
AlloNet/
├── models/               # Model code
│   ├── encoder/          #   Lightweight shared encoders (factory + IR/Light variants)
│   ├── representation/   #   Compact driving representation Z (+ det-from-Z path)
│   ├── router/           #   Task-resource router / dynamic conv modules
│   ├── heads/            #   Detection / DA / Lane heads (static + dynamic)
│   ├── adaptive_model.py #   Single-model multi-profile assembly
│   └── static_model.py   #   Static allocation model (headline config)
├── losses/               # Task losses + budget-aware regularization
├── datasets/             # BDD100K three-task dataset loader
├── training/             # Training pipeline (train.py entry point)
├── evaluation/           # Unified metrics: mAP / mIoU / lane IoU / params / FLOPs / latency
├── profiling/            # FLOPs / latency / GPU memory measurement
├── visualization/        # Router behavior, Pareto curves, failure-mode statistics
├── configs/              # Per-experiment YAML configs (model / training / data)
├── scripts/              # Reproduction chains, smoke tests, repo gates, weight downloads
├── experiments/          # Committed evidence: per-run config.yaml + metrics.json,
│                         # result tables (paper_tables.md), ledgers, figures
├── trained_models/       # Our released checkpoints (~3 MB each)
├── docs/                 # Dataset guide, task book, phase reports (0–6)
├── baselines/            # Third-party baseline integrations (weights fetched separately)
└── requirements.txt
```

## Installation

Requires Python 3.11+ and a CUDA GPU (verified on torch 2.11.0 + cu130 / Blackwell;
any recent CUDA build works).

```bash
git clone https://github.com/baiyizhuoait-ui/AlloNet.git
cd AlloNet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Note: third-party baselines pin older torch versions in their own repos; those pins are not
installable on Python 3.12 + sm_120 (Blackwell) GPUs. The versions above are verified working
for both this codebase and the baseline integrations.

## Data Preparation

This project uses the [BDD100K](https://www.bdd100k.com/) dataset (images + drivable area +
lane annotations). **The dataset is not included in this repository** due to its license —
please download it from the official source and arrange it as described in
[`docs/DATASET.md`](docs/DATASET.md).

## Pretrained Weights

- **Ours**: `trained_models/` ships three released checkpoints
  (`Static_noKD.pt`, `Static_KD_YOLOP.pt`, `Dynamic_noKD.pt`, ~3 MB each).
- **Baselines**: third-party pretrained weights are **not** committed (license + size).
  Run `bash scripts/download_weights.sh` to fetch them into `weights/` from the upstream
  repositories (YOLOP, TwinLiteNetPlus, TriLiteNet), following each project's own license.

## Quick Start

Smoke-test the training pipeline on a small image subset:

```bash
python training/train.py --config configs/phase1b_train_stage_a_1.0m.yaml --num-images 200
```

Full training run (writes checkpoints + metrics to `--outdir`):

```bash
python training/train.py --config configs/phase1b_train_stage_a_1.0m.yaml --epochs 200
```

Useful `train.py` flags: `--num-images N` (subset), `--epochs N` (override config),
`--seed N`, `--init <ckpt>` (initialize weights), `--resume <ckpt>` (resume training),
`--device cuda:0` / `--allow-cpu`.

Dual-protocol evaluation (the script that produced the paper's Protocol B numbers):

```bash
python scripts/phase6_consistency_official.py --help
```

Repository integrity gates (naming convention + hygiene, wired as a pre-commit hook):

```bash
bash scripts/check_all.sh
```

## Reproducibility & Audit Protocol

Every experiment directory under `experiments/` contains the exact `config.yaml` used and
the resulting `metrics.json`, so any number in the reports can be traced to its run. The
study follows a preregistered phase protocol:

- Phase 0–3: development, capacity sweep, allocation ledger
  (`docs/PHASE0-3_CONSOLIDATED_REPORT.md`)
- Phase 4–5: architecture and bottleneck studies (`docs/PHASE4*_`, `docs/PHASE5_*.md`)
- Phase 6: final model, dual-protocol evaluation, seed variance, and cross-architecture
  replication (`docs/PHASE6_*.md`, `experiments/phase6/`)

Headline claims are additionally gated by an 18/18 conformance audit and a ±5%
compute-budget equivalence rule; see `experiments/phase6/phase6_conformance_ledger.md`
and the G3 replication ledger.

## Documentation

| Document | Content |
|---|---|
| [`docs/DATASET.md`](docs/DATASET.md) | BDD100K download & directory layout |
| [`docs/TASKBOOK.md`](docs/TASKBOOK.md) | Algorithm development task book |
| [`docs/PHASE0-3_CONSOLIDATED_REPORT.md`](docs/PHASE0-3_CONSOLIDATED_REPORT.md) | Consolidated Phase 0–3 findings |
| [`docs/PHASE6_FINAL_REPORT.md`](docs/PHASE6_FINAL_REPORT.md) | Final model & benchmark report |
| [`experiments/phase6/paper_tables.md`](experiments/phase6/paper_tables.md) | Dual-protocol result tables |

## License

Code is released under the MIT License (see [LICENSE](LICENSE)).
The BDD100K dataset and third-party baseline weights are subject to their own licenses;
obtain them from their official sources.

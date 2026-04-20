# NOVA3R Evaluation Guide

This guide explains how to reproduce the benchmark results reported in the paper.

## Prerequisites

1. **Install NOVA3R**: Follow [INSTALL.md](INSTALL.md)
2. **Download checkpoints**: `bash scripts/download_checkpoints.sh`
3. **Download datasets**: See [DATASETS.md](DATASETS.md)

## Evaluation Datasets

NOVA3R is evaluated on the following benchmark:

- **SCRREAM**: Scene-level amodal reconstruction with occlusions

## Running Evaluations

All evaluation scripts require `--data_root` pointing to your dataset directory.

### SCRREAM Evaluation

**1-View Model (518×392)**

```bash
bash scripts/eval/eval_scrream_n1_stage2.sh --data_root /path/to/datasets
```

**2-View Model (518×392)**

```bash
bash scripts/eval/eval_scrream_n2_stage2.sh --data_root /path/to/datasets
```

## Dataset Directory Structure

The `--data_root` should contain:

```
/path/to/datasets/
└── eval_scrream/    # SCRREAM evaluation data
```

## Output

Results are saved to `eval_results/` with per-sample metrics and summary statistics.


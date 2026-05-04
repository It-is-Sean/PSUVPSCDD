# Integrated workspace map

This repo is now a single research workspace built around **NOVA3R** as the base codebase.

## Main layers

### 1. Upstream base
- `nova3r/`
- `demo_*.py`
- `eval/`
- `scripts/download_*.sh`

This remains the base implementation for reconstruction / evaluation.

### 2. Structured probe workspace
- `PROJECT.md`
- `PROPOSAL.md`
- `configs/probe/`
- `docs/probe/`
- `nova3r/probe/`
- `scripts/probe/`
- `experiments/templates/`

This is the cleaner research layer for running the shared complete-3D decoding agenda.

### 3. Collaborator-side direct experiment path
- `experiments/probe3d/`

This contains the more concrete and fast-moving probe experiments, especially the VGGT/NOVA adapter work.

## Vendored dependencies

### VGGT
- `third_party/vggt/`

Used by both the structured probe path and `experiments/probe3d/`.

### DUSt3R dataset loaders
- `dust3r/datasets/`

Copied in so dataset loading no longer has to rely on an external CUT3R checkout by default.

### Dataset preprocessing helpers
- `datasets_preprocess/`

Includes the ScanNet preprocessing scripts referenced by `experiments/probe3d/prepare_scannet_large.py`.

## Still external by nature

These are not vendored into git and should stay local/runtime-provided:
- checkpoints under `checkpoints/`
- datasets on local disks
- experiment outputs under `artifacts/`, `experiments/probe3d/result/`, `runs/`, etc.

## Practical rule of thumb

- If you want a **clean reproducible research interface**, start from `README.md`, `PROJECT.md`, `configs/probe/`, and `scripts/probe/`.
- If you want the **latest concrete adapter experiments**, go straight to `experiments/probe3d/`.
- If you need third-party model code, look in `third_party/` before reaching outside the repo.

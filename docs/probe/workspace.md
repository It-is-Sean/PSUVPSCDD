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

This contains the more concrete and fast-moving probe experiments, especially the VGGT/NOVA and WAN/NOVA adapter work.

## Third-party dependencies

### VGGT
- `third_party/vggt/`

Submodule for `facebookresearch/vggt`, used by both the structured probe path and `experiments/probe3d/`.
Initialize it before adapter training; an empty submodule checkout causes `ModuleNotFoundError: No module named 'vggt.models.vggt'`.

### Wan2.1
- `third_party/Wan2.1/`

Submodule for `Wan-Video/Wan2.1`. Its dependencies are not merged into the root `requirements.txt` or `environment.yml`.

For the SCRREAM WAN Route2 probe, use the isolated dependency file in the existing `nova3r` environment:

```bash
pip install -r experiments/probe3d/requirements-wan-t2v.txt
```

The WAN Route2 training path does not import Wan2.1 during every adapter step. WAN is loaded during feature precompute and `experiments/probe3d/train_wan_t2v_nova_adapter.py` trains from cache files. Hidden-state caches use `[3120,1536]` tensors under `tXXX/layerYY/`; pred-x0 audit caches use `--feature_kind pred_x0_latent`, write `t499/pred_x0_latent/`, and have complete `[12480,16]` tensors from job `86411`. Train pred-x0 with `--adapter_type conv2d_mlp`, because the original MLP readout hit a CUDA adaptive-pooling backward assert. Fixed hidden-grid readouts are available as `--adapter_type grid2d_pool` and `--adapter_type grid2d_conv`; both completed as negative audits, so the next WAN readout branch should be a learned Perceiver / cross-attention resampler.

### VidFM3D
- `third_party/VidFM3D/`

Submodule for `zxhuang1698/VidFM3D`, used as a reference for WAN feature extraction conventions. The current project does not import VidFM3D's Lightning training stack.

After a fresh clone, initialize third-party submodules with:

```bash
git submodule update --init --recursive
```

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
- If you need the current SCRREAM run state, read `docs/probe/handoff_2026-05-07.md`.
- If you need WAN Route2 / Route2.1 / pred-x0 / hidden-grid execution state, read the `2026-05-16 22:26`, `2026-05-16 14:05`, `2026-05-15 23:10`, `2026-05-15 12:30`, `2026-05-12`, `2026-05-11`, and `2026-05-10` update sections in `docs/probe/handoff_2026-05-07.md`.

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

The WAN Route2 training path does not import Wan2.1 during every adapter step. WAN is loaded during feature precompute and `experiments/probe3d/train_wan_t2v_nova_adapter.py` trains from cache files. Hidden-state caches use `[3120,1536]` tensors under `tXXX/layerYY/`; latent tensor-choice caches use `[12480,16]` tensors under `tXXX/<feature_kind>/`. The full WAN audit is paused and summarized in `docs/probe/wan_summary_2026-05-19.md`. The main technical lesson is that `wan_cross_attn_resampler` is the only clearly positive WAN readout; the historical peak is `t499/layer14` F@0.10 `0.5012956284974035`, repeated settings are closer to `0.49`, and WAN remains far below clean-GT VGGT layer `16`. Optional future WAN work should be appendix-only.

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
- If you need WAN Route2 / Route2.1 / pred-x0 / hidden-grid / CA-resampler / multi-source fusion execution state, start with `docs/probe/wan_summary_2026-05-19.md`; use the dated update sections in `docs/probe/handoff_2026-05-07.md` only for the detailed evidence trail.

# NOVA3R Installation Guide

## Requirements

- **GPU**: NVIDIA GPU with ≥11GB VRAM (24GB recommended)
- **Python**: 3.10 or 3.11
- **CUDA**: 12.1+

## Quick Install

```bash
git clone --recursive https://github.com/wrchen530/nova3r.git
cd nova3r
bash setup.sh
```

## Reproducible setup for this research fork

This fork is now wired to use the machine's existing conda installation
(``/data1/jcd_data/miniconda3`` on this node) and create/update the `nova3r`
environment there. We intentionally do **not** bootstrap a project-local
Miniconda anymore.

```bash
cd /path/to/3dprobe

# create/update env `nova3r` using the system conda install
make probe-env

# verify imports and CUDA
make probe-env-verify
```

Initialize third-party submodules after cloning:

```bash
git submodule update --init --recursive
```

Current submodules:

- `third_party/vggt` -> `https://github.com/facebookresearch/vggt.git`
- `third_party/Wan2.1` -> `https://github.com/Wan-Video/Wan2.1.git`
- `third_party/VidFM3D` -> `https://github.com/zxhuang1698/VidFM3D.git`

Wan2.1 and VidFM3D dependencies are intentionally not merged into the root `requirements.txt` or `environment.yml`. The SCRREAM WAN Route2 probe uses the current `nova3r` environment plus the isolated dependency file below:

```bash
conda activate nova3r
pip install -r experiments/probe3d/requirements-wan-t2v.txt
```

This keeps the VGGT/NOVA adapter environment stable while adding only the WAN T2V feature-extraction requirements (`transformers`, `tokenizers`, `sentencepiece`, `protobuf`, `ftfy`, and `huggingface-hub[cli]`).

The workflow is driven by:

- `environment.yml`
- `scripts/probe/setup_env.sh`
- `scripts/probe/verify_env.py`

Notes:

- the environment is created as `nova3r`
- if `pytorch3d` / `chamferdist` compilation fails, the script keeps going in
  best-effort mode and reports the missing optional pieces at verification time
- the current probe visualization workflow can fall back to a matplotlib backend
  even when `pytorch3d` is unavailable
- if `conda` cannot be found in the known system locations, the setup now fails
  explicitly instead of installing another Miniconda under the repo

## Manual Install

### 1. Clone and create environment

```bash
git clone --recursive https://github.com/wrchen530/nova3r.git
cd nova3r
conda create -n nova3r python=3.10 -y
conda activate nova3r
```

### 2. Install PyTorch

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install torch-cluster and pytorch3d

These require CUDA for compilation. Load CUDA first if on an HPC cluster:

```bash
module load cuda/12.1.1  # HPC clusters only

# torch-cluster
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# pytorch3d (builds from source, takes a few minutes)
FORCE_CUDA=1 MAX_JOBS=4 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
```

### 5. Compile CroCo RoPE kernels (optional, ~2-3x faster inference)

```bash
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

### 6. Install chamferdist (required for evaluation)

```bash
cd third_party
git clone https://github.com/wrchen530/chamferdist_custom.git
cd chamferdist_custom
python setup.py install
cd ../../
```

### 7. Download checkpoints

```bash
bash scripts/download_checkpoints.sh
```

Current local research-workspace state on 2026-05-19:

- `checkpoints/scene_n1/checkpoint-last.pth` and `.hydra/config.yaml` are present.
- `checkpoints/scene_n2/checkpoint-last.pth` and `.hydra/config.yaml` are present.
- `checkpoints/scene_ae/checkpoint-last.pth` and `.hydra/config.yaml` are present.
- `checkpoints/vggt/model.pt` is present.
- WAN checkpoint target `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers` is present locally (~27 GB); future downloads/preflight are handled by `slurm/scrream_wan_t2v_download.sbatch`.
- Proxy `http://127.0.0.1:7896` is the working non-WAN route for checkpoint / HuggingFace access on this machine.
- WAN repo/checkpoint jobs use proxy `http://127.0.0.1:17890`; the WAN Slurm scripts create a compute-node SSH tunnel back to `air-server:127.0.0.1:17890` by default.
- WAN sanity job `86316`, full Route2 pack job `86307`, Route2.1 clean / low-noise jobs `86371` / `86372`, targeted normalization job `86395`, and `pair_tiled81` jobs `86396 -> 86397 -> 86400 -> 86401` completed; the route is non-degenerate but remains below the clean-GT VGGT baseline.
- The pred-x0 cache script supports `--feature_kind pred_x0_latent`; full cache job `86411` completed `329/329` `[12480,16]` samples, and the training script supports `--wan_feature_kind pred_x0_latent`.
- The original pred-x0 MLP readout failed on a CUDA adaptive-pooling backward assert. `WAN_ADAPTER_TYPE=conv2d_mlp` fixed the crash path, but formal job `86422` did not beat old WAN hidden (`F@0.10=0.41336424426176693`, `pred_to_gt_p90=0.6272750149170557`).
- Hidden-grid readouts `WAN_ADAPTER_TYPE=grid2d_pool` and `WAN_ADAPTER_TYPE=grid2d_conv` completed and did not beat old WAN hidden. Learned readout `WAN_ADAPTER_TYPE=wan_cross_attn_resampler` completed its first layer sweep as job `86429`; `t499/layer14` is the historical WAN peak with `F@0.10=0.5012956284974035`, `pred_to_gt_p90=0.4229188362757365`, and `Chamfer=0.10908368105689685`. The original low-LR stability job `86444` was cancelled because its `128G` memory request blocked scheduling; replacement job `86453` completed but was negative (`F@0.10=0.4292316005720653`). Full-grid job `86468` completed all `249/499/749 x 9/14/19/24/29` CA-resampler runs; the strongest repeated settings are `t249/layer14` by F@0.10 (`0.48913902331806663`) and `t249/layer09` by p90 / Chamfer (`0.410525918006897` / `0.11981592203179996`).
- Multi-source hidden fusion is available through `WAN_ADAPTER_TYPE=wan_cross_attn_multilayer_resampler` with `slurm/scrream_wan_t2v_multilayer_pack_train.sbatch` and `WAN_ADAPTER_TYPE=wan_cross_attn_multitime_resampler` with `slurm/scrream_wan_t2v_multitime_pack_train.sbatch`. Multi-layer jobs `86485` / `86487` and multi-timestep formal job `86494` were negative versus single-layer CA baselines.
- `--feature_kind model_output_latent` / `--wan_feature_kind model_output_latent` is implemented for raw WAN transformer `output.sample` latent slices; full cache / smoke / formal jobs `86534 -> 86535 -> 86536` completed. Conv2d and latent-CA readouts for model-output/pred-x0 latent features remained below hidden CA baselines.
- Hidden CA stability/capacity jobs `86547` / `86548` / `86549` / `86550` / `86553` / `86554` completed. L4 `t249/layer14` is the best recent repeated-F WAN result (`F@0.10=0.4919946248489035`), but it remains below the historical `t499/layer14` CA peak and far below clean-GT VGGT layer `16`.
- WAN is paused as the main branch after the `2026-05-19` audit; final summary is `docs/probe/wan_summary_2026-05-19.md`. Future WAN runs should be appendix-only unless explicitly requested.
- `train_wan_t2v_nova_adapter.py` catches `swanlab.finish()` exceptions so a network/proxy teardown failure does not invalidate already-written metrics and checkpoints.
- `swanlab==0.7.16` is installed and importable in conda env `nova3r`.

### 8. Verify

```bash
python -c "from demo_nova3r import predict; print('OK')"
```

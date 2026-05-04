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

Current local research-workspace state on 2026-05-03:

- `checkpoints/scene_n1/checkpoint-last.pth` and `.hydra/config.yaml` are present.
- `checkpoints/scene_n2/checkpoint-last.pth` and `.hydra/config.yaml` are present.
- `checkpoints/scene_ae/checkpoint-last.pth` and `.hydra/config.yaml` are present.
- `checkpoints/vggt/model.pt` is present.
- Proxy `http://127.0.0.1:7896` was the working route for checkpoint / HuggingFace access on this machine.
- `swanlab==0.7.16` is installed and importable in conda env `nova3r`.

### 8. Verify

```bash
python -c "from demo_nova3r import predict; print('OK')"
```

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

### 8. Verify

```bash
python -c "from demo_nova3r import predict; print('OK')"
```

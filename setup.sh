#!/bin/bash
# NOVA3R Installation Script
# This script automates the installation of NOVA3R and its dependencies
#
# Prerequisites:
#   - conda (Miniconda or Anaconda)
#   - CUDA 12.1 (via module load or system install)
#
# Usage:
#   bash setup.sh              # Interactive setup
#   ENV_NAME=my_env bash setup.sh  # Custom environment name

set -e

echo "========================================="
echo "NOVA3R Installation Script"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "${GREEN}✓${NC} Found conda: $(conda --version)"

# Check CUDA
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
    echo -e "${GREEN}✓${NC} Found CUDA: $NVCC_VERSION"
else
    echo -e "${YELLOW}Warning: nvcc not found in PATH${NC}"
    echo "CUDA is required for compilation steps. Try:"
    echo "  module load cuda/12.1.1    (on HPC clusters)"
    echo "  export CUDA_HOME=/usr/local/cuda-12.1"
    echo ""
    read -p "Continue without CUDA? (compilation steps will be skipped) (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create conda environment
ENV_NAME="${ENV_NAME:-nova3r}"
echo ""
echo "Creating conda environment: $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME "; then
    echo -e "${YELLOW}Warning: Environment $ENV_NAME already exists${NC}"
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
    else
        echo "Using existing environment"
    fi
fi

if ! conda env list | grep -q "^$ENV_NAME "; then
    conda create -n $ENV_NAME python=3.10 -y
    echo -e "${GREEN}✓${NC} Created environment: $ENV_NAME"
else
    echo -e "${GREEN}✓${NC} Environment exists: $ENV_NAME"
fi

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo -e "${RED}Error: Failed to activate environment${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Activated: $ENV_NAME"

# Install PyTorch
echo ""
echo "Installing PyTorch with CUDA 12.1 support..."
if python -c "import torch" &> /dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "${YELLOW}PyTorch $TORCH_VERSION already installed${NC}"
    read -p "Reinstall? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
    fi
else
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
fi

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
echo -e "${GREEN}✓${NC} PyTorch $TORCH_VERSION (CUDA available: $CUDA_AVAILABLE)"

# Initialize submodules
echo ""
echo "Initializing submodules (dust3r, croco)..."
if [ -d "dust3r/.git" ] && [ -d "croco/.git" ]; then
    echo -e "${GREEN}✓${NC} Submodules already initialized"
else
    git submodule update --init --recursive
    echo -e "${GREEN}✓${NC} Submodules initialized"
fi

# Install pip dependencies
echo ""
echo "Installing pip dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}✓${NC} Pip dependencies installed"

# Install torch-cluster (needs special index for CUDA builds)
echo ""
echo "Installing torch-cluster..."
TORCH_SHORT=$(python -c "import torch; v=torch.__version__.split('+')[0].rsplit('.',1)[0]; print(v.replace('.',''))")
pip install torch-cluster -f "https://data.pyg.org/whl/torch-${TORCH_SHORT}.0+cu121.html" 2>/dev/null || \
pip install torch-cluster
echo -e "${GREEN}✓${NC} torch-cluster installed"

# Install pytorch3d (requires CUDA for compilation)
echo ""
echo "Installing pytorch3d (this may take several minutes)..."
if python -c "import pytorch3d" &> /dev/null; then
    echo -e "${GREEN}✓${NC} pytorch3d already installed"
else
    if command -v nvcc &> /dev/null; then
        FORCE_CUDA=1 MAX_JOBS=4 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git" || \
            echo -e "${YELLOW}Warning: pytorch3d build failed. Evaluation will not work without it.${NC}"
    else
        echo -e "${YELLOW}Skipping pytorch3d (requires CUDA). Install manually later:${NC}"
        echo "  FORCE_CUDA=1 pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git'"
    fi
fi

# Compile CroCo RoPE CUDA kernels
echo ""
echo "Compiling CroCo RoPE CUDA kernels..."
if command -v nvcc &> /dev/null; then
    cd croco/models/curope/
    if python setup.py build_ext --inplace 2>&1 | tee /tmp/rope_build.log | tail -1; then
        echo -e "${GREEN}✓${NC} RoPE kernels compiled"
    else
        echo -e "${YELLOW}Warning: RoPE compilation failed (inference will be slower)${NC}"
        echo "See /tmp/rope_build.log for details"
    fi
    cd ../../../
else
    echo -e "${YELLOW}Skipping RoPE compilation (CUDA not available)${NC}"
fi

# Install chamferdist
echo ""
echo "Installing chamferdist..."
if python -c "import chamferdist" &> /dev/null; then
    echo -e "${GREEN}✓${NC} chamferdist already installed"
else
    mkdir -p third_party
    cd third_party
    if [ ! -d "chamferdist_custom" ]; then
        git clone https://github.com/wrchen530/chamferdist_custom.git
    fi
    cd chamferdist_custom
    if python setup.py install 2>&1 | tee /tmp/chamfer_build.log | tail -1; then
        echo -e "${GREEN}✓${NC} chamferdist installed"
    else
        echo -e "${YELLOW}Warning: chamferdist build failed${NC}"
        echo "See /tmp/chamfer_build.log for details"
    fi
    cd ../../
fi

# Download VGGT checkpoint
echo ""
read -p "Download VGGT-1B checkpoint (~4.5GB)? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    mkdir -p checkpoints/vggt
    if [ -f "checkpoints/vggt/model.pt" ]; then
        echo -e "${YELLOW}VGGT checkpoint already exists${NC}"
    else
        echo "Downloading VGGT-1B..."
        cd checkpoints/vggt
        if wget -q --show-progress https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt; then
            echo -e "${GREEN}✓${NC} VGGT-1B downloaded"
        else
            echo -e "${YELLOW}Warning: Failed to download VGGT-1B${NC}"
        fi
        cd ../..
    fi
fi

# Verify installation
echo ""
echo "========================================="
echo "Verifying installation..."
echo "========================================="

PASS=true
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null && \
    echo -e "${GREEN}✓${NC} PyTorch + CUDA" || \
    { echo -e "${YELLOW}✗${NC} PyTorch CUDA (CPU only)"; }

python -c "from nova3r.models.nova3r_img_cond import Nova3rImgCond" 2>/dev/null && \
    echo -e "${GREEN}✓${NC} Nova3r models" || \
    { echo -e "${RED}✗${NC} Nova3r models"; PASS=false; }

python -c "from nova3r.inference import inference_nova3r" 2>/dev/null && \
    echo -e "${GREEN}✓${NC} Nova3r inference" || \
    { echo -e "${RED}✗${NC} Nova3r inference"; PASS=false; }

python -c "import pytorch3d" 2>/dev/null && \
    echo -e "${GREEN}✓${NC} pytorch3d" || \
    echo -e "${YELLOW}✗${NC} pytorch3d (eval will not work)"

python -c "import chamferdist" 2>/dev/null && \
    echo -e "${GREEN}✓${NC} chamferdist" || \
    echo -e "${YELLOW}✗${NC} chamferdist (eval will not work)"

# Summary
echo ""
echo "========================================="
if [ "$PASS" = true ]; then
    echo -e "${GREEN}Installation Complete!${NC}"
else
    echo -e "${RED}Installation had errors. Check above for details.${NC}"
fi
echo "========================================="
echo ""
echo "Environment: $ENV_NAME"
echo "Python: $(python --version 2>&1)"
echo "PyTorch: $TORCH_VERSION"
echo "CUDA: $CUDA_AVAILABLE"
echo ""
echo "Next steps:"
echo "  conda activate $ENV_NAME"
echo "  python demo_nova3r.py --images demo/examples/scene_1.png \\"
echo "    --ckpt checkpoints/scene_n1/checkpoint-last.pth --resolution 518 392"
echo ""
echo "For evaluation datasets: bash scripts/download_datasets.sh"
echo "See docs/INSTALL.md for more information."
echo ""

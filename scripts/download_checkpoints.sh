#!/bin/bash
# Download NOVA3R checkpoints from HuggingFace
#
# Usage:
#   bash scripts/download_checkpoints.sh           # Download all available
#   bash scripts/download_checkpoints.sh --model scene_n1  # Download specific model

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

HF_REPO="wrchen530/nova3r"

MODEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash $0 [--model scene_n1|scene_n2|scene_ae|all]"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

download_hf_model() {
    local name=$1
    local ckpt_name=$2
    local target_dir="checkpoints/${name}"

    if [ -f "${target_dir}/${ckpt_name}" ]; then
        echo -e "${YELLOW}${name} already exists. Skipping.${NC}"
        return 0
    fi

    echo "Downloading ${name}..."
    mkdir -p "$target_dir"
    huggingface-cli download $HF_REPO "${name}/${ckpt_name}" --local-dir checkpoints/ || \
        { echo "Failed to download ${name}. Try: huggingface-cli login"; return 1; }

    # Download hydra config
    huggingface-cli download $HF_REPO "${name}/.hydra/config.yaml" --local-dir checkpoints/ 2>/dev/null || true

    echo -e "${GREEN}✓${NC} ${name} downloaded to ${target_dir}/"
}

echo "========================================"
echo "NOVA3R Checkpoint Download"
echo "========================================"
echo ""

# Download based on selection
if [ -z "$MODEL" ] || [ "$MODEL" = "all" ]; then
    download_hf_model "scene_n1" "checkpoint-last.pth"
    echo ""
    download_hf_model "scene_n2" "checkpoint-last.pth"
    echo ""
    download_hf_model "scene_ae" "checkpoint-last.pth"
    echo ""
elif [ "$MODEL" = "scene_n1" ]; then
    download_hf_model "scene_n1" "checkpoint-last.pth"
elif [ "$MODEL" = "scene_n2" ]; then
    download_hf_model "scene_n2" "checkpoint-last.pth"
elif [ "$MODEL" = "scene_ae" ]; then
    download_hf_model "scene_ae" "checkpoint-last.pth"
else
    echo "Unknown model: $MODEL"
    echo "Available: scene_n1, scene_n2, scene_ae, all"
    exit 1
fi

echo ""
echo "========================================"
echo "Checkpoint Status"
echo "========================================"
[ -f "checkpoints/scene_n1/checkpoint-last.pth" ] && echo -e "${GREEN}✓${NC} Scene N=1" || echo "  ✗ Scene N=1"
[ -f "checkpoints/scene_n2/checkpoint-last.pth" ] && echo -e "${GREEN}✓${NC} Scene N=2" || echo "  ✗ Scene N=2"
[ -f "checkpoints/scene_ae/checkpoint-last.pth" ] && echo -e "${GREEN}✓${NC} Scene AE" || echo "  ✗ Scene AE"
echo ""

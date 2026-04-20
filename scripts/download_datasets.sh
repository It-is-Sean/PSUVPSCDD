#!/bin/bash
# Download evaluation datasets for NOVA3R
# Datasets are hosted on HuggingFace (same source as LaRI)
#
# Usage:
#   bash scripts/download_datasets.sh                    # Download all
#   bash scripts/download_datasets.sh --dataset scrream   # Download SCRREAM only
#   bash scripts/download_datasets.sh --output /path/to   # Custom output directory

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Defaults
OUTPUT_DIR="datasets"
DATASET="all"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: bash $0 [--dataset scrream|all] [--output /path/to/dir]"
            echo ""
            echo "Options:"
            echo "  --dataset   Dataset to download: scrream or all (default: all)"
            echo "  --output    Output directory (default: datasets/)"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; echo "Usage: bash $0 [--dataset scrream|all] [--output /path/to/dir]"; exit 1 ;;
    esac
done

# HuggingFace URLs (same source as LaRI)
SCRREAM_URL="https://huggingface.co/datasets/ruili3/LaRI_dataset/resolve/main/eval/eval_scrream.zip?download=true"

echo "========================================"
echo "NOVA3R Evaluation Dataset Download"
echo "========================================"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""

mkdir -p "$OUTPUT_DIR"

download_dataset() {
    local name=$1
    local url=$2
    local target_dir="${OUTPUT_DIR}/eval_${name}"
    local zip_file="${OUTPUT_DIR}/eval_${name}.zip"

    if [ -d "$target_dir" ] && [ "$(ls -A $target_dir 2>/dev/null)" ]; then
        echo -e "${YELLOW}${name} already exists at ${target_dir}. Skipping.${NC}"
        echo "  (Delete the directory to re-download)"
        return 0
    fi

    echo "Downloading ${name} dataset..."
    if wget --progress=bar:force -O "$zip_file" "$url"; then
        echo "Extracting ${name}..."
        unzip -q "$zip_file" -d "$OUTPUT_DIR"
        rm "$zip_file"
        echo -e "${GREEN}✓${NC} ${name} downloaded and extracted to ${target_dir}"
    else
        echo -e "${RED}✗${NC} Failed to download ${name}"
        echo "  Please download manually from:"
        echo "  ${url}"
        rm -f "$zip_file"
        return 1
    fi
}

# Download requested datasets
if [ "$DATASET" = "all" ] || [ "$DATASET" = "scrream" ]; then
    download_dataset "scrream" "$SCRREAM_URL"
    echo ""
fi

# Verify
echo "========================================"
echo "Dataset Status"
echo "========================================"

check_dataset() {
    local name=$1
    local dir="${OUTPUT_DIR}/eval_${name}"
    if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
        local count=$(ls -d ${dir}/*/ 2>/dev/null | wc -l)
        echo -e "${GREEN}✓${NC} ${name}: ${count} entries in ${dir}"
    else
        echo -e "✗ ${name}: Not found at ${dir}"
    fi
}

check_dataset "scrream"

echo ""
echo "To run evaluation:"
echo "  bash scripts/eval/eval_scrream_n1_stage2.sh --data_root ${OUTPUT_DIR}"
echo ""

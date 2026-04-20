#!/bin/bash
# Evaluate Nova3r (N=2) on SCRREAM scene dataset
# Usage: bash scripts/eval/eval_scrream_n2_stage2.sh --data_root /path/to/datasets

DATA_ROOT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; echo "Usage: bash $0 --data_root /path/to/datasets"; exit 1 ;;
    esac
done

if [ -z "$DATA_ROOT" ]; then
    echo "Error: --data_root is required"
    echo "Usage: bash $0 --data_root /path/to/datasets"
    exit 1
fi

CKPT_DIR=checkpoints/scene_n2
CKPT_NAME=checkpoint-last.pth
WORKDIR=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")

echo "Evaluating: SCRREAM N=2 (scene-level, multi-view)"
echo "Checkpoint: ${CKPT_DIR}/${CKPT_NAME}"

python eval/mv_recon/test_nova3r.py \
    --config-path=$WORKDIR/$CKPT_DIR/.hydra \
    --config-name=config \
    +experiment.ckpt_path=${CKPT_DIR}/${CKPT_NAME} \
    +experiment.test_dataset_name=scrream_n2 \
    +experiment.data_root=$DATA_ROOT

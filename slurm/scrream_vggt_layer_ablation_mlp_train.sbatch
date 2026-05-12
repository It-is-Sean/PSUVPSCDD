#!/usr/bin/env bash
#SBATCH --job-name=scrream_vggt_layer_ablation
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-6

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/JJ_Group/lih2511/3dprobe}"
mkdir -p slurm_out experiments/probe3d/result experiments/probe3d/feature_cache

LAYERS_CSV="${SCRREAM_ABLATION_LAYERS:-0,4,8,12,16,20,24}"
IFS=',' read -r -a LAYERS <<< "$LAYERS_CSV"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$TASK_ID" -lt 0 || "$TASK_ID" -ge "${#LAYERS[@]}" ]]; then
  echo "SLURM_ARRAY_TASK_ID=$TASK_ID is out of range for SCRREAM_ABLATION_LAYERS=$LAYERS_CSV" >&2
  exit 2
fi

LAYER="${LAYERS[$TASK_ID]}"
LAYER_PADDED="$(printf '%02d' "$LAYER")"
RUN_PREFIX="${SCRREAM_ABLATION_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17_vggt_layer}"

export SCRREAM_VGGT_LAYER="$LAYER"
export SCRREAM_ADAPTER_DATA="${SCRREAM_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt}"
export SCRREAM_NUM_QUERIES="${SCRREAM_NUM_QUERIES:-20000}"
export SCRREAM_EPOCHS="${SCRREAM_EPOCHS:-50}"
export SCRREAM_SAVE_EVERY="${SCRREAM_SAVE_EVERY:-500}"
export SCRREAM_VAL_EVERY="${SCRREAM_VAL_EVERY:-500}"
export SCRREAM_EVAL_BATCHES="${SCRREAM_EVAL_BATCHES:-2}"
export SCRREAM_VAL_FLOW_T_BINS="${SCRREAM_VAL_FLOW_T_BINS:-10}"
export SCRREAM_OUTPUT_DIR="${SCRREAM_OUTPUT_DIR:-experiments/probe3d/result/${RUN_PREFIX}${LAYER_PADDED}}"
export SCRREAM_FEATURE_CACHE_DIR="${SCRREAM_FEATURE_CACHE_DIR:-experiments/probe3d/feature_cache/${RUN_PREFIX}${LAYER_PADDED}}"
export SCRREAM_SWANLAB_EXPERIMENT="${SCRREAM_SWANLAB_EXPERIMENT:-${RUN_PREFIX}${LAYER_PADDED}}"

echo "SCRREAM VGGT layer ablation"
echo "  task_id=$TASK_ID"
echo "  layers=$LAYERS_CSV"
echo "  selected_layer=$LAYER"
echo "  output_dir=$SCRREAM_OUTPUT_DIR"
echo "  feature_cache_dir=$SCRREAM_FEATURE_CACHE_DIR"

bash slurm/scrream_mesh_complete_mlp_train.sbatch

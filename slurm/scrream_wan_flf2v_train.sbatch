#!/usr/bin/env bash
#SBATCH --job-name=scrream_wan_flf2v_train
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/JJ_Group/lih2511/3dprobe}"
mkdir -p slurm_out experiments/probe3d/result

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate nova3r

if [[ -f slurm/.secrets.env ]]; then
  set -a
  source slurm/.secrets.env
  set +a
fi

ADAPTER_DATA="${WAN_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt}"
NOVA_CKPT="${WAN_NOVA_CKPT:-checkpoints/scene_ae/checkpoint-last.pth}"
CACHE_DIR="${WAN_FLF2V_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_wan_flf2v14b_pair_endpoint81_480}"
TIMESTEP="${WAN_FLF2V_TIMESTEP:-249}"
LAYER="${WAN_FLF2V_LAYER:-14}"
ADAPTER_TYPE="${WAN_ADAPTER_TYPE:-wan_cross_attn_resampler}"
RUN_PREFIX="${WAN_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_wan_flf2v14b_pair_endpoint81_}"
LAYER_PADDED="$(printf '%02d' "$LAYER")"
RUN_NAME="${WAN_RUN_NAME:-${RUN_PREFIX}t${TIMESTEP}_layer${LAYER_PADDED}}"
OUTPUT_DIR="${WAN_OUTPUT_BASE:-experiments/probe3d/result}/${RUN_NAME}"

NUM_QUERIES="${WAN_NUM_QUERIES:-20000}"
EPOCHS="${WAN_EPOCHS:-50}"
BATCH_SIZE="${WAN_BATCH_SIZE:-1}"
TRAIN_COUNT="${WAN_TRAIN_COUNT:-317}"
MAX_STEPS="${WAN_MAX_STEPS:-$(
  TRAIN_COUNT="$TRAIN_COUNT" EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" python - <<'PY'
import math
import os
print(max(1, math.ceil(float(os.environ["TRAIN_COUNT"]) * float(os.environ["EPOCHS"]) / max(1, int(os.environ["BATCH_SIZE"])))))
PY
)}"

if [[ "$ADAPTER_TYPE" != "wan_cross_attn_resampler" ]]; then
  echo "FLF2V first run expects WAN_ADAPTER_TYPE=wan_cross_attn_resampler, got $ADAPTER_TYPE" >&2
  exit 2
fi
if [[ ! -f "$ADAPTER_DATA" ]]; then
  echo "Missing SCRREAM adapter dataset: $ADAPTER_DATA" >&2
  exit 2
fi
if [[ ! -f "$NOVA_CKPT" ]]; then
  echo "Missing NOVA checkpoint: $NOVA_CKPT" >&2
  exit 2
fi
if [[ ! -d "$CACHE_DIR" ]]; then
  echo "Missing WAN FLF2V feature cache dir: $CACHE_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

echo "SCRREAM WAN FLF2V hidden probe train"
echo "  timestep=$TIMESTEP"
echo "  layer=$LAYER"
echo "  adapter_data=$ADAPTER_DATA"
echo "  cache_dir=$CACHE_DIR"
echo "  output_dir=$OUTPUT_DIR"
echo "  adapter_type=$ADAPTER_TYPE"
echo "  adapter_layers=${WAN_ADAPTER_LAYERS:-2}"
echo "  adapter_hidden_dim=${WAN_ADAPTER_HIDDEN_DIM:-512}"
echo "  adapter_heads=${WAN_ADAPTER_HEADS:-8}"
echo "  adapter_mlp_ratio=${WAN_ADAPTER_MLP_RATIO:-2.0}"
echo "  max_steps=$MAX_STEPS"
echo "  num_queries=$NUM_QUERIES"
nvidia-smi || true

python -u experiments/probe3d/train_wan_t2v_nova_adapter.py \
  --dataset scrream_adapter \
  --data_root "$ADAPTER_DATA" \
  --nova_ckpt "$NOVA_CKPT" \
  --wan_feature_cache_dir "$CACHE_DIR" \
  --wan_timestep "$TIMESTEP" \
  --wan_layer "$LAYER" \
  --wan_feature_kind hidden \
  --wan_feature_norm "${WAN_FEATURE_NORM:-none}" \
  --wan_feature_norm_eps "${WAN_FEATURE_NORM_EPS:-1e-6}" \
  --adapter_type "$ADAPTER_TYPE" \
  --adapter_layers "${WAN_ADAPTER_LAYERS:-2}" \
  --adapter_hidden_dim "${WAN_ADAPTER_HIDDEN_DIM:-512}" \
  --adapter_heads "${WAN_ADAPTER_HEADS:-8}" \
  --adapter_mlp_ratio "${WAN_ADAPTER_MLP_RATIO:-2.0}" \
  --loss_type "${WAN_LOSS_TYPE:-nova_flow}" \
  --num_queries "$NUM_QUERIES" \
  --num_views 2 \
  --train_split "${WAN_TRAIN_SPLIT:-train}" \
  --val_split "${WAN_VAL_SPLIT:-val}" \
  --test_split "${WAN_TEST_SPLIT:-test}" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "${WAN_NUM_WORKERS:-4}" \
  --lr "${WAN_LR:-1e-4}" \
  --max_steps "$MAX_STEPS" \
  --save_every "${WAN_SAVE_EVERY:-500}" \
  --val_every "${WAN_VAL_EVERY:-500}" \
  --eval_batches "${WAN_EVAL_BATCHES:-0}" \
  --val_flow_t_bins "${WAN_VAL_FLOW_T_BINS:-10}" \
  --val_metric_max_points "${WAN_VAL_METRIC_MAX_POINTS:-20000}" \
  --val_preview_samples "${WAN_VAL_PREVIEW_SAMPLES:-12}" \
  --val_preview_queries "${WAN_VAL_PREVIEW_QUERIES:-40960}" \
  --save_ply_queries "${WAN_SAVE_PLY_QUERIES:-40960}" \
  --output_dir "$OUTPUT_DIR" \
  --swanlab \
  --swanlab_project "${WAN_SWANLAB_PROJECT:-PSUVPSC3DD}" \
  --swanlab_experiment "$RUN_NAME"

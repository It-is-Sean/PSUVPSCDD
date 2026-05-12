#!/usr/bin/env bash
#SBATCH --job-name=scrream_wan_sanity
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

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

if [[ "${WAN_SANITY_DISABLE_PROXY:-0}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
else
  export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7896}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7896}"
  export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7896}"
  export http_proxy="$HTTP_PROXY"
  export https_proxy="$HTTPS_PROXY"
  export all_proxy="$ALL_PROXY"
fi

ADAPTER_DATA="${WAN_SANITY_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt}"
NOVA_CKPT="${WAN_SANITY_NOVA_CKPT:-checkpoints/scene_ae/checkpoint-last.pth}"
CACHE_DIR="${WAN_SANITY_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81}"
TIMESTEP="${WAN_SANITY_TIMESTEP:-499}"
LAYER="${WAN_SANITY_LAYER:-9}"
MAX_STEPS="${WAN_SANITY_MAX_STEPS:-3000}"
VAL_EVERY="${WAN_SANITY_VAL_EVERY:-500}"
SAVE_EVERY="${WAN_SANITY_SAVE_EVERY:-500}"
NUM_QUERIES="${WAN_SANITY_NUM_QUERIES:-20000}"
PREVIEW_SAMPLES="${WAN_SANITY_VAL_PREVIEW_SAMPLES:-4}"
PREVIEW_QUERIES="${WAN_SANITY_VAL_PREVIEW_QUERIES:-40960}"
OUTPUT_BASE="${WAN_SANITY_OUTPUT_BASE:-experiments/probe3d/result/scrream_wan_t2v_sanity_metafilter_seed17}"
SENSITIVITY_CONFIGS="${WAN_SANITY_SENSITIVITY_CONFIGS:-249:9,249:29,499:9,749:9}"
SENSITIVITY_SAMPLES="${WAN_SANITY_SENSITIVITY_SAMPLES:-4}"

if [[ ! -f "$ADAPTER_DATA" ]]; then
  echo "Missing adapter data: $ADAPTER_DATA" >&2
  exit 2
fi
if [[ ! -f "$NOVA_CKPT" ]]; then
  echo "Missing NOVA checkpoint: $NOVA_CKPT" >&2
  exit 2
fi
if [[ ! -d "$CACHE_DIR" ]]; then
  echo "Missing WAN cache dir: $CACHE_DIR" >&2
  exit 2
fi

echo "SCRREAM WAN sanity"
echo "  adapter_data=$ADAPTER_DATA"
echo "  cache_dir=$CACHE_DIR"
echo "  primary=t${TIMESTEP}/layer${LAYER}"
echo "  max_steps=$MAX_STEPS"
echo "  sensitivity_configs=$SENSITIVITY_CONFIGS"
echo "  output_base=$OUTPUT_BASE"
echo "  proxy=${HTTP_PROXY:-<disabled>}"
nvidia-smi || true

python -u experiments/probe3d/scripts/check_wan_feature_sensitivity.py \
  --adapter_data "$ADAPTER_DATA" \
  --nova_ckpt "$NOVA_CKPT" \
  --wan_feature_cache_dir "$CACHE_DIR" \
  --configs "$SENSITIVITY_CONFIGS" \
  --num_samples "$SENSITIVITY_SAMPLES" \
  --num_queries "$NUM_QUERIES" \
  --output_path "$OUTPUT_BASE/feature_sensitivity.json"

for mode in zero sample_shuffle; do
  layer_padded="$(printf '%02d' "$LAYER")"
  run_name="scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_wan_t2v1p3b_ctx81_t${TIMESTEP}_layer${layer_padded}_sanity_${mode}_step${MAX_STEPS}"
  output_dir="$OUTPUT_BASE/$run_name"
  log_file="slurm_out/scrream_wan_sanity_t${TIMESTEP}_layer${layer_padded}_${mode}_${SLURM_JOB_ID:-manual}.log"
  echo "Launching sanity mode=$mode output=$output_dir log=$log_file"
  python -u experiments/probe3d/train_wan_t2v_nova_adapter.py \
    --dataset scrream_adapter \
    --data_root "$ADAPTER_DATA" \
    --nova_ckpt "$NOVA_CKPT" \
    --wan_feature_cache_dir "$CACHE_DIR" \
    --wan_timestep "$TIMESTEP" \
    --wan_layer "$LAYER" \
    --wan_feature_mode "$mode" \
    --wan_feature_shuffle_seed "${WAN_SANITY_SHUFFLE_SEED:-17001}" \
    --adapter_type mlp \
    --adapter_layers "${WAN_SANITY_ADAPTER_LAYERS:-4}" \
    --adapter_hidden_dim "${WAN_SANITY_ADAPTER_HIDDEN_DIM:-1024}" \
    --loss_type "${WAN_SANITY_LOSS_TYPE:-nova_flow}" \
    --num_queries "$NUM_QUERIES" \
    --num_views 2 \
    --train_split "${WAN_SANITY_TRAIN_SPLIT:-train}" \
    --val_split "${WAN_SANITY_VAL_SPLIT:-val}" \
    --test_split "${WAN_SANITY_TEST_SPLIT:-test}" \
    --batch_size "${WAN_SANITY_BATCH_SIZE:-1}" \
    --num_workers "${WAN_SANITY_NUM_WORKERS:-4}" \
    --lr "${WAN_SANITY_LR:-1e-4}" \
    --max_steps "$MAX_STEPS" \
    --save_every "$SAVE_EVERY" \
    --val_every "$VAL_EVERY" \
    --eval_batches "${WAN_SANITY_EVAL_BATCHES:-0}" \
    --val_flow_t_bins "${WAN_SANITY_VAL_FLOW_T_BINS:-10}" \
    --val_metric_max_points "${WAN_SANITY_VAL_METRIC_MAX_POINTS:-20000}" \
    --val_preview_samples "$PREVIEW_SAMPLES" \
    --val_preview_queries "$PREVIEW_QUERIES" \
    --save_ply_queries "$PREVIEW_QUERIES" \
    --output_dir "$output_dir" \
    --swanlab \
    --swanlab_project "${WAN_SANITY_SWANLAB_PROJECT:-PSUVPSC3DD}" \
    --swanlab_experiment "$run_name" \
    >"$log_file" 2>&1
done

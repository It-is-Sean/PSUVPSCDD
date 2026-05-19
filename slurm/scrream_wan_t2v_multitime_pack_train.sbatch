#!/usr/bin/env bash
#SBATCH --job-name=scrream_wan_t2v_multitime
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/JJ_Group/lih2511/3dprobe}"
mkdir -p slurm_out experiments/probe3d/result

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate nova3r

TIMESTEP_GROUPS_CSV="${WAN_TIMESTEP_GROUPS:-249+499+749}"
LAYER_CSV="${WAN_LAYERS:-14}"
ADAPTER_TYPE="${WAN_ADAPTER_TYPE:-wan_cross_attn_multitime_resampler}"
FEATURE_KIND="${WAN_FEATURE_KIND:-hidden}"
if [[ "$FEATURE_KIND" != "hidden" ]]; then
  echo "Multi-timestep WAN fusion only supports WAN_FEATURE_KIND=hidden, got $FEATURE_KIND" >&2
  exit 2
fi
if [[ "$ADAPTER_TYPE" != "wan_cross_attn_multitime_resampler" ]]; then
  echo "Multi-timestep WAN fusion requires WAN_ADAPTER_TYPE=wan_cross_attn_multitime_resampler, got $ADAPTER_TYPE" >&2
  exit 2
fi
IFS=',' read -r -a TIMESTEP_GROUPS <<< "$TIMESTEP_GROUPS_CSV"
IFS=',' read -r -a LAYERS <<< "$LAYER_CSV"

VISIBLE_RAW="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "$VISIBLE_RAW" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$VISIBLE_RAW"
else
  GPU_COUNT="${SLURM_GPUS_ON_NODE:-${WAN_PACK_GPUS:-1}}"
  GPU_IDS=()
  for ((i = 0; i < GPU_COUNT; i++)); do
    GPU_IDS+=("$i")
  done
fi
MAX_PARALLEL="${WAN_PACK_PARALLEL:-${#GPU_IDS[@]}}"
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "No GPUs visible for WAN multi-timestep pack." >&2
  exit 2
fi

ADAPTER_DATA="${WAN_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt}"
NOVA_CKPT="${WAN_NOVA_CKPT:-checkpoints/scene_ae/checkpoint-last.pth}"
CACHE_DIR="${WAN_CACHE_DIR:-experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81}"
RUN_PREFIX="${WAN_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_wan_t2v1p3b_ctx81_ca_resampler_multitime_}"
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

if [[ ! -f "$ADAPTER_DATA" ]]; then
  echo "Missing SCRREAM adapter dataset: $ADAPTER_DATA" >&2
  exit 2
fi
if [[ ! -f "$NOVA_CKPT" ]]; then
  echo "Missing NOVA checkpoint: $NOVA_CKPT" >&2
  exit 2
fi
if [[ ! -d "$CACHE_DIR" ]]; then
  echo "Missing WAN feature cache dir: $CACHE_DIR" >&2
  exit 2
fi

echo "Packed SCRREAM WAN T2V multi-timestep fusion"
echo "  timestep_groups=$TIMESTEP_GROUPS_CSV"
echo "  layers=$LAYER_CSV"
echo "  visible_gpus=${GPU_IDS[*]}"
echo "  max_parallel=$MAX_PARALLEL"
echo "  adapter_data=$ADAPTER_DATA"
echo "  cache_dir=$CACHE_DIR"
echo "  adapter_type=$ADAPTER_TYPE"
echo "  adapter_layers=${WAN_ADAPTER_LAYERS:-2}"
echo "  adapter_hidden_dim=${WAN_ADAPTER_HIDDEN_DIM:-512}"
echo "  adapter_heads=${WAN_ADAPTER_HEADS:-8}"
echo "  adapter_mlp_ratio=${WAN_ADAPTER_MLP_RATIO:-2.0}"
echo "  max_steps=$MAX_STEPS"
echo "  feature_norm=${WAN_FEATURE_NORM:-none}"
nvidia-smi || true

declare -a PIDS=()

wait_for_slot() {
  while [[ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]]; do
    local still=()
    local pid
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        still+=("$pid")
      else
        wait "$pid"
      fi
    done
    PIDS=("${still[@]}")
    if [[ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]]; then
      sleep 10
    fi
  done
}

job_idx=0
for layer in "${LAYERS[@]}"; do
  for group in "${TIMESTEP_GROUPS[@]}"; do
    wait_for_slot
    timesteps_csv="${group//+/,}"
    IFS=',' read -r -a GROUP_TIMESTEPS <<< "$timesteps_csv"
    timesteps_slug=""
    for group_timestep in "${GROUP_TIMESTEPS[@]}"; do
      group_timestep="$(printf '%d' "$group_timestep")"
      if [[ -z "$timesteps_slug" ]]; then
        timesteps_slug="$group_timestep"
      else
        timesteps_slug="${timesteps_slug}-${group_timestep}"
      fi
    done
    layer_padded="$(printf '%02d' "$layer")"
    first_timestep="$(printf '%s' "$timesteps_csv" | cut -d',' -f1)"
    run_name="${RUN_PREFIX}timesteps${timesteps_slug}_layer${layer_padded}"
    gpu="${GPU_IDS[$((job_idx % ${#GPU_IDS[@]}))]}"
    output_dir="${WAN_OUTPUT_BASE:-experiments/probe3d/result}/${run_name}"
    log_file="slurm_out/scrream_wan_t2v_multitime_timesteps${timesteps_slug}_layer${layer_padded}_${SLURM_JOB_ID:-manual}.log"
    mkdir -p "$output_dir"
    echo "Launching timesteps=$timesteps_csv layer=$layer on CUDA_VISIBLE_DEVICES=$gpu output=$output_dir log=$log_file"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python -u experiments/probe3d/train_wan_t2v_nova_adapter.py \
        --dataset scrream_adapter \
        --data_root "$ADAPTER_DATA" \
        --nova_ckpt "$NOVA_CKPT" \
        --wan_feature_cache_dir "$CACHE_DIR" \
        --wan_timestep "$first_timestep" \
        --wan_timesteps "$timesteps_csv" \
        --wan_layer "$layer" \
        --wan_feature_kind "$FEATURE_KIND" \
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
        --output_dir "$output_dir" \
        --swanlab \
        --swanlab_project "${WAN_SWANLAB_PROJECT:-PSUVPSC3DD}" \
        --swanlab_experiment "$run_name"
    ) >"$log_file" 2>&1 &
    PIDS+=("$!")
    job_idx=$((job_idx + 1))
  done
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"

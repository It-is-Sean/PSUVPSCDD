#!/usr/bin/env bash
#SBATCH --job-name=scrream_omega_vggt_pad_preproc
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# Direction 2 of the 2026-05-20 VGGT-Omega vs VGGT-1B audit.
#
# Hypothesis: the third_party VGGT-Omega preprocess at image_resolution=512 may
# differ from VGGT-1B's mode="pad" preprocess (e.g. center-crop vs aspect-pad,
# fill color, normalization), giving Omega a structurally different field-of-view
# input. This launcher controls the preprocess variable by forcing Omega through
# a local aspect-preserving longest-side resize + center pad to a square 512
# canvas (16-aligned for Omega's patch_embed). The third_party Omega loader is
# bypassed; both arms below use the exact same preprocess.
#
# Compares the two best Omega readouts from the 2026-05-20 audit:
# - dense full-token MLP-L4 layer16
# - dense full-token CA-L4 layer16
# Each one is rerun with the controlled preprocess so we can read off whether
# preprocess accounts for the Omega gap.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/JJ_Group/lih2511/3dprobe}"
mkdir -p slurm_out experiments/probe3d/result experiments/probe3d/feature_cache

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate nova3r

if [[ -f slurm/.secrets.env ]]; then
  set -a
  source slurm/.secrets.env
  set +a
fi

if [[ "${SCRREAM_DISABLE_PROXY:-0}" == "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
else
  export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7896}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7896}"
  export ALL_PROXY="${ALL_PROXY:-http://127.0.0.1:7896}"
fi

# Each spec is "adapter_type:layer:suffix". We only sweep the Omega readouts
# that previously came closest to VGGT-1B.
RUN_SPECS_CSV="${SCRREAM_OMEGA_PAD_RUN_SPECS:-mlp:16:omega_aspect_pad_mlp_l4_layer16,cross_attention:16:omega_aspect_pad_ca_l4_layer16}"
IFS=',' read -r -a RUN_SPECS <<< "$RUN_SPECS_CSV"

VISIBLE_RAW="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "$VISIBLE_RAW" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$VISIBLE_RAW"
else
  GPU_COUNT="${SLURM_GPUS_ON_NODE:-${SCRREAM_PACK_GPUS:-1}}"
  GPU_IDS=()
  for ((i = 0; i < GPU_COUNT; i++)); do
    GPU_IDS+=("$i")
  done
fi
MAX_PARALLEL="${SCRREAM_PACK_PARALLEL:-${#GPU_IDS[@]}}"
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "No GPUs visible for VGGT-pad Omega rerun." >&2
  exit 2
fi

ADAPTER_DATA="${SCRREAM_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt}"
NOVA_CKPT="${SCRREAM_NOVA_CKPT:-checkpoints/scene_ae/checkpoint-last.pth}"
VGGT_OMEGA_WEIGHTS="${SCRREAM_VGGT_OMEGA_WEIGHTS:-checkpoints/vggt_omega/vggt_omega_1b_512.pt}"
VGGT_OMEGA_IMAGE_RESOLUTION="${SCRREAM_VGGT_OMEGA_IMAGE_RESOLUTION:-512}"
VGGT_OMEGA_IMAGE_PREPROC="${SCRREAM_VGGT_OMEGA_IMAGE_PREPROC:-aspect_pad_square}"
RUN_PREFIX="${SCRREAM_ABLATION_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_nova_flow_robustval_metafilter_seed17_}"
VGGT_TOKEN_MODE="${SCRREAM_VGGT_TOKEN_MODE:-full}"
NUM_QUERIES="${SCRREAM_NUM_QUERIES:-20000}"
EPOCHS="${SCRREAM_EPOCHS:-50}"
BATCH_SIZE="${SCRREAM_BATCH_SIZE:-1}"
TRAIN_COUNT="${SCRREAM_TRAIN_COUNT:-317}"
MAX_STEPS="${SCRREAM_MAX_STEPS:-$(
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
if [[ ! -f "$VGGT_OMEGA_WEIGHTS" ]]; then
  echo "Missing VGGT-Omega weights: $VGGT_OMEGA_WEIGHTS" >&2
  exit 2
fi

echo "Controlled-preprocess VGGT-Omega rerun (Direction 2)"
echo "  run_specs=$RUN_SPECS_CSV"
echo "  visible_gpus=${GPU_IDS[*]}"
echo "  max_parallel=$MAX_PARALLEL"
echo "  adapter_data=$ADAPTER_DATA"
echo "  vggt_omega_image_resolution=$VGGT_OMEGA_IMAGE_RESOLUTION"
echo "  vggt_omega_image_preproc=$VGGT_OMEGA_IMAGE_PREPROC"
echo "  vggt_token_mode=$VGGT_TOKEN_MODE"
echo "  max_steps=$MAX_STEPS"
echo "  num_queries=$NUM_QUERIES"
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

for idx in "${!RUN_SPECS[@]}"; do
  wait_for_slot
  spec="${RUN_SPECS[$idx]}"
  IFS=':' read -r adapter_type layer suffix <<< "$spec"
  if [[ -z "${adapter_type:-}" || -z "${layer:-}" || -z "${suffix:-}" ]]; then
    echo "Bad run spec '$spec'. Expected adapter_type:layer:suffix" >&2
    exit 2
  fi
  gpu="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
  output_dir="${SCRREAM_OUTPUT_BASE:-experiments/probe3d/result}/${RUN_PREFIX}${suffix}"
  # Different preproc means a fresh cache, otherwise features would mix two preprocesses.
  feature_cache_dir="${SCRREAM_FEATURE_CACHE_BASE:-experiments/probe3d/feature_cache}/${RUN_PREFIX}${suffix}_${VGGT_OMEGA_IMAGE_PREPROC}"
  mkdir -p "$output_dir" "$feature_cache_dir"
  log_file="slurm_out/scrream_omega_pad_preproc_${suffix}_${SLURM_JOB_ID:-manual}.log"
  echo "Launching ${suffix} (adapter=$adapter_type layer=$layer) on CUDA_VISIBLE_DEVICES=$gpu output=$output_dir log=$log_file"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export FEATURE_MEMORY_CACHE_MAX="${FEATURE_MEMORY_CACHE_MAX:-0}"
    python -u experiments/probe3d/train_vggt_nova_adapter.py \
      --backbone vggt_omega \
      --dataset scrream_adapter \
      --data_root "$ADAPTER_DATA" \
      --nova_ckpt "$NOVA_CKPT" \
      --vggt_omega_weights "$VGGT_OMEGA_WEIGHTS" \
      --vggt_omega_image_resolution "$VGGT_OMEGA_IMAGE_RESOLUTION" \
      --vggt_omega_image_preproc "$VGGT_OMEGA_IMAGE_PREPROC" \
      --vggt_token_mode "$VGGT_TOKEN_MODE" \
      --adapter_type "$adapter_type" \
      --adapter_layers "${SCRREAM_ADAPTER_LAYERS:-4}" \
      --adapter_hidden_dim "${SCRREAM_ADAPTER_HIDDEN_DIM:-1024}" \
      --adapter_heads "${SCRREAM_ADAPTER_HEADS:-8}" \
      --adapter_mlp_ratio "${SCRREAM_ADAPTER_MLP_RATIO:-2.0}" \
      --loss_type "${SCRREAM_LOSS_TYPE:-nova_flow}" \
      --num_queries "$NUM_QUERIES" \
      --vggt_layer "$layer" \
      --num_views 2 \
      --train_split "${SCRREAM_TRAIN_SPLIT:-train}" \
      --val_split "${SCRREAM_VAL_SPLIT:-val}" \
      --test_split "${SCRREAM_TEST_SPLIT:-test}" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "${SCRREAM_NUM_WORKERS:-4}" \
      --lr "${SCRREAM_LR:-1e-4}" \
      --max_steps "$MAX_STEPS" \
      --save_every "${SCRREAM_SAVE_EVERY:-500}" \
      --val_every "${SCRREAM_VAL_EVERY:-500}" \
      --eval_batches "${SCRREAM_EVAL_BATCHES:-0}" \
      --val_flow_t_bins "${SCRREAM_VAL_FLOW_T_BINS:-10}" \
      --val_metric_max_points "${SCRREAM_VAL_METRIC_MAX_POINTS:-20000}" \
      --val_preview_samples "${SCRREAM_VAL_PREVIEW_SAMPLES:-12}" \
      --val_preview_queries "${SCRREAM_VAL_PREVIEW_QUERIES:-40960}" \
      --save_ply_queries "${SCRREAM_SAVE_PLY_QUERIES:-40960}" \
      --feature_cache_dir "$feature_cache_dir" \
      --output_dir "$output_dir" \
      --swanlab \
      --swanlab_project "${SCRREAM_SWANLAB_PROJECT:-PSUVPSC3DD}" \
      --swanlab_experiment "${RUN_PREFIX}${suffix}"
  ) >"$log_file" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"

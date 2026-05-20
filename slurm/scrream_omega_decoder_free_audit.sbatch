#!/usr/bin/env bash
#SBATCH --job-name=scrream_omega_decoder_free_audit
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# Direction 1 of the 2026-05-20 VGGT-Omega vs VGGT-1B audit.
#
# Hypothesis: Omega underperforms VGGT-1B through the NOVA/FM decoder because
# the small adapter cannot remap Omega tokens onto NOVA's condition-token
# manifold, not because Omega's frozen representation is spatially worse.
#
# This launcher bypasses the NOVA decoder entirely. Both backbones go through
# the same DirectPointCrossAttentionReadout (CA-L4-H512 -> [N,3]) trained with
# Chamfer-L2 on the same SCRREAM mesh-complete .pt and split as the NOVA probe.
#
# Interpretation gate:
# - Omega clearly beats VGGT-1B here  => Omega is a stronger spatial backbone
#   and the NOVA path is the bottleneck. Reframe the project line as measuring
#   NOVA-decoder compatibility, not pure representation quality.
# - Omega still trails VGGT-1B here   => Omega's frozen tokens are actually
#   weaker on this SCRREAM target under the same readout budget.
# - Outcomes are within noise          => decoder-free CA budget itself is the
#   bottleneck; consider adapter-capacity / token-mode follow-ups.

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

# Focused audit grid: VGGT-1B and Omega at the two layers where VGGT-1B's NOVA
# probe peaked (MLP layer16, CA layer20). All runs use the same dense full-token
# CA readout that directly regresses 3D points.
RUN_SPECS_CSV="${SCRREAM_DIRECT_RUN_SPECS:-vggt:full:16:none:vggt1_layer16,vggt:full:20:none:vggt1_layer20,vggt_omega:full:16:none:omega_dense_layer16,vggt_omega:full:20:none:omega_dense_layer20}"
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
  echo "No GPUs visible for decoder-free audit pack." >&2
  exit 2
fi

ADAPTER_DATA="${SCRREAM_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt}"
VGGT_WEIGHTS="${SCRREAM_VGGT_WEIGHTS:-checkpoints/vggt/model.pt}"
VGGT_OMEGA_WEIGHTS="${SCRREAM_VGGT_OMEGA_WEIGHTS:-checkpoints/vggt_omega/vggt_omega_1b_512.pt}"
VGGT_OMEGA_IMAGE_RESOLUTION="${SCRREAM_VGGT_OMEGA_IMAGE_RESOLUTION:-512}"
RUN_PREFIX="${SCRREAM_DIRECT_RUN_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_decoder_free_audit_point_ca_l4_h512_seed17_}"
FEATURE_CACHE_PREFIX="${SCRREAM_DIRECT_FEATURE_CACHE_PREFIX:-scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_decoder_free_audit_feat_}"
POINT_QUERIES="${SCRREAM_POINT_QUERIES:-4096}"
TARGET_TRAIN_POINTS="${SCRREAM_TARGET_TRAIN_POINTS:-4096}"
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
if [[ ! -f "$VGGT_WEIGHTS" ]]; then
  echo "Missing VGGT weights: $VGGT_WEIGHTS" >&2
  exit 2
fi
if [[ ! -f "$VGGT_OMEGA_WEIGHTS" ]]; then
  echo "Missing VGGT-Omega weights: $VGGT_OMEGA_WEIGHTS" >&2
  exit 2
fi
if [[ ! -d third_party/vggt-omega ]]; then
  echo "Missing third_party/vggt-omega. Run git submodule update --init --recursive." >&2
  exit 2
fi

echo "Decoder-free SCRREAM probe-validity audit (Direction 1)"
echo "  run_specs=$RUN_SPECS_CSV"
echo "  visible_gpus=${GPU_IDS[*]}"
echo "  max_parallel=$MAX_PARALLEL"
echo "  adapter_data=$ADAPTER_DATA"
echo "  run_prefix=$RUN_PREFIX"
echo "  max_steps=$MAX_STEPS"
echo "  point_queries=$POINT_QUERIES"
echo "  target_train_points=$TARGET_TRAIN_POINTS"
echo "  val_metric_max_points=${SCRREAM_VAL_METRIC_MAX_POINTS:-20000}"
echo "  val_preview_samples=${SCRREAM_VAL_PREVIEW_SAMPLES:-12}"
echo "  val_preview_queries=${SCRREAM_VAL_PREVIEW_QUERIES:-4096}"
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
  IFS=':' read -r backbone token_mode layer control suffix <<< "$spec"
  if [[ -z "${backbone:-}" || -z "${token_mode:-}" || -z "${layer:-}" || -z "${control:-}" || -z "${suffix:-}" ]]; then
    echo "Bad run spec '$spec'. Expected backbone:token_mode:layer:feature_control:suffix" >&2
    exit 2
  fi
  gpu="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
  output_dir="${SCRREAM_OUTPUT_BASE:-experiments/probe3d/result}/${RUN_PREFIX}${suffix}"
  feature_cache_dir="${SCRREAM_FEATURE_CACHE_BASE:-experiments/probe3d/feature_cache}/${FEATURE_CACHE_PREFIX}${suffix}"
  mkdir -p "$output_dir" "$feature_cache_dir"
  log_file="slurm_out/scrream_omega_decoder_free_${suffix}_${SLURM_JOB_ID:-manual}.log"
  echo "Launching decoder-free spec=$spec on CUDA_VISIBLE_DEVICES=$gpu output=$output_dir log=$log_file"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export FEATURE_MEMORY_CACHE_MAX="${FEATURE_MEMORY_CACHE_MAX:-0}"
    extra_weights=()
    if [[ "$backbone" == "vggt_omega" ]]; then
      extra_weights+=(--vggt_omega_weights "$VGGT_OMEGA_WEIGHTS" --vggt_omega_image_resolution "$VGGT_OMEGA_IMAGE_RESOLUTION")
    else
      extra_weights+=(--vggt_weights "$VGGT_WEIGHTS")
    fi
    python -u experiments/probe3d/train_vggt_direct_point_adapter.py \
      --backbone "$backbone" \
      "${extra_weights[@]}" \
      --dataset scrream_adapter \
      --data_root "$ADAPTER_DATA" \
      --vggt_layer "$layer" \
      --vggt_token_mode "$token_mode" \
      --feature_control "$control" \
      --readout_type point_ca \
      --point_queries "$POINT_QUERIES" \
      --adapter_layers "${SCRREAM_ADAPTER_LAYERS:-4}" \
      --adapter_hidden_dim "${SCRREAM_ADAPTER_HIDDEN_DIM:-512}" \
      --adapter_heads "${SCRREAM_ADAPTER_HEADS:-8}" \
      --adapter_mlp_ratio "${SCRREAM_ADAPTER_MLP_RATIO:-2.0}" \
      --target_train_points "$TARGET_TRAIN_POINTS" \
      --target_norm_mode "${SCRREAM_TARGET_NORM_MODE:-median_3}" \
      --num_views 2 \
      --train_split "${SCRREAM_TRAIN_SPLIT:-train}" \
      --val_split "${SCRREAM_VAL_SPLIT:-val}" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "${SCRREAM_NUM_WORKERS:-4}" \
      --lr "${SCRREAM_LR:-1e-4}" \
      --max_steps "$MAX_STEPS" \
      --save_every "${SCRREAM_SAVE_EVERY:-500}" \
      --val_every "${SCRREAM_VAL_EVERY:-500}" \
      --eval_batches "${SCRREAM_EVAL_BATCHES:-0}" \
      --val_metric_max_points "${SCRREAM_VAL_METRIC_MAX_POINTS:-20000}" \
      --val_preview_samples "${SCRREAM_VAL_PREVIEW_SAMPLES:-12}" \
      --val_preview_queries "${SCRREAM_VAL_PREVIEW_QUERIES:-4096}" \
      --save_ply_queries "${SCRREAM_SAVE_PLY_QUERIES:-4096}" \
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

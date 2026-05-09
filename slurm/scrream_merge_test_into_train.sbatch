#!/usr/bin/env bash
#SBATCH --job-name=scrream_split_merge
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/home/JJ_Group/lih2511/3dprobe}"
mkdir -p slurm_out experiments/probe3d/adapter_data

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate nova3r

SOURCE_PATH="${SCRREAM_SOURCE_ADAPTER_DATA:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.pt}"
OUTPUT_PATH="${SCRREAM_ADAPTER_OUT:-experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_trainplus_test.pt}"
FROM_SPLIT="${SCRREAM_MERGE_FROM_SPLIT:-test}"
TO_SPLIT="${SCRREAM_MERGE_TO_SPLIT:-train}"
KEEP_SPLIT="${SCRREAM_KEEP_SPLIT:-val}"
NUM_QUERIES="${SCRREAM_NUM_QUERIES:-10000}"

if [[ ! -f "$SOURCE_PATH" ]]; then
  echo "Missing source adapter dataset: $SOURCE_PATH" >&2
  exit 2
fi

export SOURCE_PATH OUTPUT_PATH FROM_SPLIT TO_SPLIT KEEP_SPLIT NUM_QUERIES

python - <<'PY'
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

source_path = Path(os.environ["SOURCE_PATH"]).expanduser()
output_path = Path(os.environ["OUTPUT_PATH"]).expanduser()
from_split = os.environ["FROM_SPLIT"]
to_split = os.environ["TO_SPLIT"]
keep_split = os.environ["KEEP_SPLIT"]
num_queries = int(os.environ["NUM_QUERIES"])

payload = torch.load(source_path, map_location="cpu")
targets = payload["target_points"]
if targets.ndim != 3 or targets.shape[-1] != 3:
    raise ValueError(f"Expected target_points [N,P,3], got {tuple(targets.shape)}")
if targets.shape[1] < num_queries:
    raise ValueError(
        f"target_points has only {targets.shape[1]} points, requested {num_queries}"
    )

splits = list(payload.get("splits", ["train"] * int(targets.shape[0])))
if len(splits) != int(targets.shape[0]):
    raise ValueError("splits length does not match target_points")

old_counts = Counter(splits)
new_splits = [to_split if split == from_split else split for split in splits]
new_counts = Counter(new_splits)
if old_counts.get(from_split, 0) <= 0:
    raise ValueError(f"No samples found for from_split={from_split!r}: {dict(old_counts)}")
if new_counts.get(to_split, 0) <= old_counts.get(to_split, 0):
    raise ValueError(
        f"Merge did not increase {to_split!r}: old={dict(old_counts)} new={dict(new_counts)}"
    )
if keep_split and old_counts.get(keep_split, 0) != new_counts.get(keep_split, 0):
    raise ValueError(
        f"Expected split {keep_split!r} to be preserved: old={dict(old_counts)} new={dict(new_counts)}"
    )

payload["splits"] = new_splits
meta = dict(payload.get("meta", {}))
meta["split_policy"] = f"merge_{from_split}_into_{to_split}_keep_{keep_split}"
meta["source_adapter_data"] = str(source_path)
meta["old_split_counts"] = dict(old_counts)
meta["new_split_counts"] = dict(new_counts)
payload["meta"] = meta

output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, output_path)

metadata = payload.get("metadata", [{} for _ in new_splits])
scene_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
for item, split in zip(metadata, new_splits):
    scene_id = item.get("scene_id") or str(item.get("sample_id", "")).split("/")[0]
    scene_counts_by_split[split][scene_id] += 1

manifest = {
    "source_path": str(source_path),
    "output_path": str(output_path),
    "policy": meta["split_policy"],
    "target_points_shape": list(targets.shape),
    "old_split_counts": dict(old_counts),
    "new_split_counts": dict(new_counts),
    "new_scene_counts_by_split": {
        split: dict(counter) for split, counter in sorted(scene_counts_by_split.items())
    },
    "created_at": datetime.now(timezone.utc).isoformat(),
}
manifest_path = output_path.with_suffix(".manifest.json")
with manifest_path.open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")

print(json.dumps(manifest, indent=2, sort_keys=True))
PY

python experiments/probe3d/scripts/check_scrream_adapter_training_inputs.py \
  --adapter_data "$OUTPUT_PATH" \
  --num_queries "$NUM_QUERIES" \
  --split "$TO_SPLIT"

if [[ -n "$KEEP_SPLIT" ]]; then
  python experiments/probe3d/scripts/check_scrream_adapter_training_inputs.py \
    --adapter_data "$OUTPUT_PATH" \
    --num_queries "$NUM_QUERIES" \
    --split "$KEEP_SPLIT"
fi

# Git / workspace organization — 2026-04-29

This repo is currently a research execution workspace with many uncommitted changes. Do not interpret the dirty tree as one logical patch.

## Current direction

- ScanNet artifacts remain under `experiments/probe3d/result/` and are ignored by the existing `experiments/probe3d/.gitignore` result rule.
- The next data direction is an InteriorGS pilot on this server.
- Generated InteriorGS subsets, renders, point clouds, and checkpoints should stay out of git unless they are tiny documentation artifacts.

## Intended code/config groups

### A. ScanNet dataset / preprocessing plumbing

Files such as:

- `dust3r/datasets/scannet.py`
- `datasets_preprocess/preprocess_scannet.py`
- `experiments/probe3d/prepare_scannet_large.py`
- `experiments/probe3d/build_scannet_complete_gt.py`
- `experiments/probe3d/scripts/make_scannet_train_val_split.py`

Purpose: ScanNet mesh-first complete/amodal target construction and corrected `scannet_max_interval=1` handling.

### B. VGGT → NOVA adapter code

Files such as:

- `experiments/probe3d/probe/adapter.py`
- `experiments/probe3d/train_vggt_nova_adapter.py`
- `experiments/probe3d/train_vggt_nova_attention_adapter.py`
- `experiments/probe3d/train_vggt_nova_cross_attention_adapter.py`
- `experiments/probe3d/train_vggt_nova_self_attention_adapter.py`
- `experiments/probe3d/vggt_nova_adapter_common.py`
- `experiments/probe3d/vggt_nova_adapter_common_raw.py`
- `experiments/probe3d/visualize_vggt_nova_adapter.py`

Purpose: frozen VGGT-style backbone + lightweight/structured adapter + frozen NOVA3R-style decoder experiments.

### C. Probe harness

Directory:

- `experiments/probe3d/probe_trials/`

Purpose: config-driven trial harness, immutable result rows, robust fixed-sample evaluation tools, and current-state notes.

Important files:

- `CURRENT_STATE.md`
- `robust_ply_metrics.py`
- `eval_checkpoint_robust.py`
- `run_trial.py`
- `results.tsv`
- `configs/*.json`

### D. Documentation

Files:

- `README.md`
- `PROJECT.md`
- `docs/probe/*.md`
- `experiments/probe3d/README.md`

Purpose: keep the proposal-facing state honest after corrections: SCRREAM invalidation, ScanNet interval correction, metric/oracle unreliability, robust visual-first eval, and the InteriorGS data pivot.

## Do not commit as one blob unless explicitly desired

Suggested future commit split if we decide to commit:

1. docs: corrected current state and proposal-facing notes
2. data: ScanNet mesh/frustum target and interval plumbing
3. training: adapter/training/common code changes
4. eval: probe robust metrics/eval harness
5. data-pivot: InteriorGS loader / inspection / documentation

## Working branch

The local cleanup branch is `wip/psuvpsc3dd-probe-20260429` so future commits can be split away from `master`. The branch currently points to the old `master` HEAD plus the dirty working tree; no commit was made during cleanup.

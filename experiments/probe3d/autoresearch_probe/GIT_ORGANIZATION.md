# Git / workspace organization — 2026-04-29

This repo is currently a research execution workspace with many uncommitted changes. Do not interpret the dirty tree as one logical patch.

## Active processes

- AutoResearchClaw run artifacts live under `experiments/probe3d/result/researchclaw/` and are ignored by the existing `experiments/probe3d/.gitignore` result rule.
- Local ResearchClaw runtime cache `.researchclaw_cache/` is ignored.
- Local LLM compatibility proxy source is tracked-intended under `researchclaw/openclaw_stream_proxy.js`; it contains no API key and reads the OpenClaw config at runtime.

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

### C. Autoresearch probe harness

Directory:

- `experiments/probe3d/autoresearch_probe/`

Purpose: config-driven trial harness, immutable result rows, robust fixed-sample evaluation tools, current-state notes, and ResearchClaw supervision notes.

Important files:

- `CURRENT_STATE.md`
- `robust_ply_metrics.py`
- `eval_checkpoint_robust.py`
- `run_trial.py`
- `results.tsv`
- `configs/*.json`

### D. ResearchClaw integration

Directory:

- `researchclaw/`

Purpose: project-specific ResearchClaw config/guardrails and localhost Responses proxy for model access.

Files:

- `config.arc.yaml`
- `prompts.psuvpsc3dd.yaml`
- `openclaw_stream_proxy.js`

### E. Documentation

Files:

- `README.md`
- `PROJECT.md`
- `docs/probe/*.md`
- `experiments/probe3d/README.md`

Purpose: keep the proposal-facing state honest after corrections: SCRREAM invalidation, ScanNet interval correction, metric/oracle unreliability, robust visual-first eval, and ResearchClaw supervision.

## Do not commit as one blob unless explicitly desired

Suggested future commit split if we decide to commit:

1. docs: corrected current state and proposal-facing notes
2. data: ScanNet mesh/frustum target and interval plumbing
3. training: adapter/training/common code changes
4. eval: autoresearch robust metrics/eval harness
5. infra: ResearchClaw config/proxy/supervisor notes

## Current caution

ResearchClaw stage-04/05 external literature search had broad-query contamination. Supervisor guidance was injected at stage 7; downstream ResearchClaw artifacts should still be audited before being treated as authoritative.

## Working branch

Created/switched to `wip/psuvpsc3dd-autoresearch-20260429` so future cleanup commits can be split away from `master`. The branch currently points to the old `master` HEAD plus the dirty working tree; no commit was made during cleanup.

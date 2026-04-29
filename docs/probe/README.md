# Probe Docs

## Current canonical status — 2026-04-29 afternoon

For the latest autonomous/research-loop state, read:

- `../../experiments/probe3d/autoresearch_probe/CURRENT_STATE.md`
- `../../experiments/probe3d/autoresearch_probe/heartbeat_log.md`
- `../../researchclaw/config.arc.yaml`

Key corrections:

- `scannet_max_interval=1` is now the intended setting because ScanNet preprocessing already uses `frame_skip=20`. Old `max_interval=30` K-view conclusions are invalid/confounded.
- CD-only and two-sample oracle results are not claim-level evidence. Use fixed-sample robust metrics and visual audits.
- The current MLP baseline is mostly a failure-mode baseline: recall is moderate, precision/sharpness are poor.
- AutoResearchClaw is active as a proposal-aligned organizer, but its outputs are audited by a 15-minute supervisor for direction and code cleanliness.


This folder records the proposal-facing execution state for the current adapter / decoder experiments.

## Read this first

### 1. What is active now?
The current active formal branch is the **ScanNet v2 mesh-first extension line**.

It is:
- a NOVA3R-style extension / transfer probe
- based on reliable mesh-first complete-GT supervision
- launched through DDP / `torchrun`

It is **not** a literal reproduction of official NOVA3R training on `3D-FRONT + ScanNet++V2`.

### 2. What about SCRREAM?
The older local `eval_scrream` experiments are now treated as **invalid for formal claims**, because they used only the released eval subset rather than the official full-data setup.

The SCRREAM branch remains important, but only after correct full data becomes available locally.

## Core documents

- `scannet_mesh_first_plan.md`
  - the current formal ScanNet v2 plan, assumptions, implementation status, and launch semantics

- `experiment_history.md`
  - the honest history, including corrections and invalidated branches

- `experiment_plan.md`
  - phased execution plan from the current state forward

- `todo.md`
  - current actionable task list

## Current active probe baseline

The current ScanNet branch has shifted from a long formal MLP run to a short autoresearch-style probe loop, because that isolated the failure mode faster.

Current best numeric baseline:

- target: `anchor_frustum`
- adapter: `MLP-L4, hidden=1024`
- objective: direct sampled rollout Chamfer (`loss_type=chamfer_sample`)
- best validation CD: `0.08745259`
- output dir: `experiments/probe3d/result/autoresearch_probe/p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step2500`

Interpretation:

- direct Chamfer fixed a large train/eval objective mismatch compared with `nova_flow`
- the result is still not visually clean: prediction recall is reasonable, but precision / outlier control is poor
- next experiments should track precision-aware loss and GT-vs-pred videos, not just lower symmetric CD
### Paper-aligned NOVA3R reset

After user review, the active plan is to align the ScanNet target/loss more closely with NOVA3R: complete / amodal points inside the selected input-view frustum, FPS-style target sampling through `src_complete_fps_*`, and native flow matching as the primary loss. The new phase-2 config is:

- `experiments/probe3d/autoresearch_probe/configs/phase2_nova_aligned.json`

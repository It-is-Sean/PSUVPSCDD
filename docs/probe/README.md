# Probe Docs

## Current canonical status — 2026-05-03

For the latest project state, read:

- `handoff_2026-05-03.md`
- `../../experiments/probe3d/README.md`
- `experiment_plan.md`

Key corrections:

- `scannet_max_interval=1` is now the intended setting because ScanNet preprocessing already uses `frame_skip=20`. Old `max_interval=30` K-view conclusions are invalid/confounded.
- CD-only and two-sample oracle results are not claim-level evidence. Use fixed-sample robust metrics and visual audits.
- The current MLP baseline is mostly a failure-mode baseline: recall is moderate, precision/sharpness are poor.
- The old local `eval_scrream` branch is invalid for claims, but full SCRREAM is now downloaded at `~/datasets/SCRREAM`.
- The active next branch is SCRREAM full-data mesh-complete adapter training.
- Long data generation and training should use `slurm/` scripts with logs in `slurm_out/`.
- On 2026-05-03 02:26 CST, Slurm job `85773` was running full mesh-complete adapter prep and dependent job `85774` was pending for the first MLP baseline.
- NOVA `scene_n1`, `scene_n2`, `scene_ae`, and VGGT weights are staged under `checkpoints/`; SwanLab is installed in `nova3r`.
- The local cleanup branch is `wip/psuvpsc3dd-probe-20260429`; push it before using it for a fresh remote checkout.

This folder records the proposal-facing execution state for the current adapter / decoder experiments.

## Read this first

### 1. What is active now?
The current active formal branch is the **SCRREAM full-data mesh-complete adapter line**.

It is:
- a corrected rerun of the SCRREAM adapter idea using the full dataset at `~/datasets/SCRREAM`
- based on registered SCRREAM scene meshes, not the invalid `eval_scrream` subset
- constrained to the selected two-input-view union frustum
- launched through Slurm scripts in `slurm/`

It is **not** a claim yet. The next required steps are full `.pt` generation, MLP baseline training, and visual / metric inspection.
The full `.pt` generation had been submitted as job `85773` at the current handoff timestamp; monitor it instead of starting a duplicate default prep job.

### 2. What about ScanNet?
The ScanNet v2 mesh-first line remains the diagnostic baseline.

It is:
- a NOVA3R-style extension / transfer probe
- based on reliable mesh-first complete-GT supervision
- useful for understanding failure modes and metric reliability

It is **not** a literal reproduction of official NOVA3R training on `3D-FRONT + ScanNet++V2`.

### 3. What about old SCRREAM?
The older local `eval_scrream` experiments are treated as **invalid for formal claims**, because they used only the released eval subset rather than the official full-data setup.

### 4. What about InteriorGS?
InteriorGS remains a deferred high-quality data option. It is no longer the immediate next training branch after the full SCRREAM dataset became available locally.


## Core documents

- `handoff_2026-05-03.md`
  - current machine handoff for SCRREAM full mesh-complete data prep and training

- `scannet_mesh_first_plan.md`
  - the current formal ScanNet v2 plan, assumptions, implementation status, and launch semantics

- `experiment_history.md`
  - the honest history, including corrections and invalidated branches

- `experiment_plan.md`
  - phased execution plan from the current state forward

- `interiorgs_training_plan.md`
  - deferred high-quality indoor dataset migration path

- `todo.md`
  - current actionable task list

## Current ScanNet probe baseline

The current ScanNet branch has shifted from a long formal MLP run to a short probe loop, because that isolated the failure mode faster.

Current best numeric baseline:

- target: `anchor_frustum`
- adapter: `MLP-L4, hidden=1024`
- objective: direct sampled rollout Chamfer (`loss_type=chamfer_sample`)
- best validation CD: `0.08745259`
- output dir: `experiments/probe3d/result/probe_trials/p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step2500`

Interpretation:

- direct Chamfer fixed a large train/eval objective mismatch compared with `nova_flow`
- the result is still not visually clean: prediction recall is reasonable, but precision / outlier control is poor
- the next project direction is to test whether corrected SCRREAM full mesh-complete supervision gives a cleaner training signal than the current ScanNet transfer setup

### Paper-aligned NOVA3R reset

After user review, the active plan is to align the ScanNet target/loss more closely with NOVA3R: complete / amodal points inside the selected input-view frustum, FPS-style target sampling through `src_complete_fps_*`, and native flow matching as the primary loss. The new phase-2 config is:

- `experiments/probe3d/probe_trials/configs/phase2_nova_aligned.json`

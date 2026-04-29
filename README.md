# PSUVPSC3DD / Probe Workspace

## Current canonical status — 2026-04-29 late afternoon

This branch is now a **local-autopilot / handoff-ready research workspace**. The source of truth is:

- `PROPOSAL.md`
- `experiments/probe3d/autoresearch_probe/CURRENT_STATE.md`
- `experiments/probe3d/autoresearch_probe/heartbeat_log.md`
- `docs/probe/handoff_2026-04-29.md`

Important corrections that override older sections below:

1. **ResearchClaw retired as executor:** AutoResearchClaw/ResearchClaw full-pipeline execution drifted into irrelevant CIFAR/KD/FitNet experiments and is no longer used to drive this project. Current automation is the local OpenClaw autopilot over `experiments/probe3d/autoresearch_probe/`.
2. **ScanNet view interval:** processed data already uses `frame_skip=20`; corrected experiments use `scannet_max_interval=1` (adjacent processed frames, roughly 20 raw frames). Older `max_interval=30` runs are interval-confounded.
3. **Metric reliability:** single symmetric CD and two-sample oracle averages are diagnostic only. Current comparisons use fixed samples, pred→GT precision, GT→pred recall, F-score thresholds, trimmed CD, and representative renders.
4. **Current MLP baseline:** K2/interval=1 `anchor_frustum + MLP-L4 + chamfer_sample` is recall-heavy but precision/outlier-poor on fixed-30 robust eval: F@0.05 mean/median `0.291/0.275`, precision@0.05 mean `0.204`, recall@0.05 mean `0.532`.
5. **Latest structured-adapter check:** K2/interval=1 `anchor_frustum + cross_attention L2/H512 + chamfer_sample` completed 1000 steps with validation CD `0.54222615`, which is not better than the MLP baseline by scalar validation CD. Its fixed-30 robust eval was launched under the same protocol and should be inspected before any claim.
6. **Remote handoff:** the current branch was pushed to GitHub at `dongjiacheng06/3dprobe`, branch `wip/psuvpsc3dd-autoresearch-20260429`, currently at commit `2f0e210` before the `/neat` cleanup commit.

This repository is currently a **research execution workspace** around a simple question:

> can frozen visual backbones (currently VGGT-first) drive a NOVA3R-style decoder through a lightweight adapter, and learn useful complete 3D reconstruction behavior under reliable supervision?

It contains:

- the NOVA3R / DUST3R style reconstruction stack used here as the decoder/data backbone
- the probe / adapter experiments under `experiments/probe3d/`
- proposal-facing documentation under `docs/probe/`

## Current experimental status

There are now **two active lines**, and they should be interpreted differently.

### 1. SCRREAM line

A retrospective audit found that the earlier local `eval_scrream` package was only the released **evaluation subset** (~1.6 GB), not the official full training-scale dataset.

So:

- the older SCRREAM quantitative results are **invalid as formal scientific evidence**
- they are still useful as **pipeline/debugging history**
- they should not be used as proposal feasibility claims

The correct full-data SCRREAM rerun remains a separate pending branch once the official full dataset is available locally.

### 2. ScanNet v2 line

The current active scale-up path is a **ScanNet v2 mesh-first extension line**.

Important interpretation note:

- this is a **NOVA3R-style extension / transfer probe** on ScanNet v2
- it is **not** a literal reproduction of the official NOVA3R scene-training recipe, which is described around `3D-FRONT + ScanNet++V2`

Still, the training structure is intentionally aligned with the proposal direction:

- frozen visual backbone
- lightweight adapter
- NOVA3R-style decoder / flow-matching training path
- reliable **complete-GT style** supervision

## Active ScanNet v2 mesh-first plan

The current formal ScanNet training path uses:

- geometry source: `vh_clean.ply`
- frame sampling: `frame_skip=20`
- GT definition: `mesh surface ∩ sparse-input-view union frustum`
- no extra visible/occluded auxiliary labels for now
- scene-level reservoir: `500k` mesh-sampled points per scene
- true **4-view input**
- DDP / `torchrun` training launchers

Formal plan details live in:

- `docs/probe/scannet_mesh_first_plan.md`

## What is already implemented

### Data / supervision
- full ScanNet v2 preprocess completed to:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k`
- formal split created at:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17`
- split scene counts:
  - `train=1362`
  - `val=151`
  - `test=100`

### Model / training plumbing
- ScanNet now truly feeds **4 input views**
- `pts3d_complete` is available from scene-level mesh reservoirs
- MLP / CA / SA training scripts were converted to **DDP / torchrun**
- launchers were updated accordingly

### Validation status
- visible-depth ScanNet smoke passed earlier
- complete-GT ScanNet smoke passed
- full-root `torchrun --nproc_per_node=1` MLP DDP preflight also passed

## Current active result — autoresearch-style ScanNet probe

The long formal 50-epoch MLP run is no longer the most informative immediate path. A short autoresearch-style harness under `experiments/probe3d/autoresearch_probe/` isolated the main failure mode more quickly.

Current best numeric run:

- target: `anchor_frustum`
- adapter: `MLP-L4, hidden=1024`
- objective: direct sampled rollout Chamfer, `loss_type=chamfer_sample`
- schedule: `lr=5e-5` to step 2000, then `lr=1e-5` refinement to step 2500
- best validation CD: `0.08745259`
- output dir: `experiments/probe3d/result/autoresearch_probe/p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step2500`

Important interpretation:

- switching from `nova_flow` to direct rollout Chamfer is what moved CD down from roughly `0.30–0.35` to `<0.1`
- this mostly fixes the training/evaluation objective mismatch
- the visual result is still not good enough: the prediction covers the GT but contains many loose / thick-shell / outlier points
- the problem has shifted from **recall / target reachability** to **precision / sharpness**


### Paper-aligned NOVA3R reset

After user review, the active plan is to align the ScanNet target/loss more closely with NOVA3R: complete / amodal points inside the selected input-view frustum, FPS-style target sampling through `src_complete_fps_*`, and native flow matching as the primary loss. The new phase-2 config is:

- `experiments/probe3d/autoresearch_probe/configs/phase2_nova_aligned.json`

## Research direction right now

The practical near-term plan is:

1. keep the SCRREAM correction in mind and do not reuse the old invalid claims
2. treat the ScanNet v2 mesh-first branch as a transfer-probe line, not official NOVA3R reproduction
3. keep `anchor_frustum + direct Chamfer` as the current numeric baseline
4. replace symmetric Chamfer-only training with a precision-aware objective, e.g. weighted `pred→GT` plus `GT→pred`, trimmed Chamfer, or outlier penalties
5. use visual point-cloud videos alongside CD before claiming progress
6. once correct full SCRREAM data arrives, rerun that branch cleanly as a separate line

## Documentation map

- `PROJECT.md` — current project-level status and next steps
- `docs/probe/README.md` — probe-doc entry point
- `docs/probe/scannet_mesh_first_plan.md` — current ScanNet formal plan
- `docs/probe/experiment_history.md` — what really happened, including corrections
- `docs/probe/experiment_plan.md` — phased execution plan from here
- `docs/probe/todo.md` — current actionable task list

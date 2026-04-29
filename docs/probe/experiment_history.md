# Experiment history summary

## 2026-04-29 correction block — interval, metric, and ResearchClaw

A major audit corrected the interpretation of the ScanNet runs:

- The processed ScanNet data already uses `frame_skip=20`. The inherited dataloader `max_interval=30` therefore sampled views up to roughly 600 raw frames apart. All old K-view conclusions from those runs are interval-confounded.
- Corrected local-overlap experiments use `scannet_max_interval=1`, equivalent to adjacent processed frames / roughly 20 raw frames.
- GT-only visual audit showed the three ScanNet target modes (`anchor_frustum`, `covered_by_ge2`, `nova_input_frustum`) are not drastically different by eye.
- Oracle CD rankings were dominated by tiny sample counts, stochastic decoder sampling, flow-vs-CD mismatch, and outlier samples; they are no longer treated as reliable target-mode rankings.
- Robust evaluation of the interval-corrected MLP baseline shows moderate recall but poor precision/outlier control, matching the qualitative failure.

AutoResearchClaw was installed and configured as a research-loop organizer. A local proxy bridges ResearchClaw to the OpenClaw-configured model provider, and a separate 15-minute OpenClaw supervisor audits ResearchClaw outputs for proposal alignment and code cleanliness.


This document records what actually happened, including corrections.

## 1. Major correction — old SCRREAM results are invalid for claims

A later audit found that the local `eval_scrream` package used in the earlier SCRREAM line was only the released evaluation subset, not the official full-data setup.

Therefore:

- the older SCRREAM quantitative results are **invalid as formal evidence**
- they remain useful as **engineering/debugging history** only
- they must not be used as proposal feasibility claims

## 2. What remains valid from the earlier history

Even after that correction, the repo still accumulated real engineering progress:

- adapter branches were implemented and trained
- NOVA3R-style decoder integration was established
- caching / logging / export paths were exercised
- debugging knowledge compounded across multiple runs

So the history is not wasted — it just must be interpreted honestly.

## 3. New active branch — ScanNet v2 mesh-first extension

After the SCRREAM correction, a new formal branch became the active scale-up path.

### Interpretation
This branch is:
- a **NOVA3R-style extension / transfer probe** on ScanNet v2
- not a literal reproduction of official NOVA3R scene training on `3D-FRONT + ScanNet++V2`

### Formal data decisions
- mesh source: `vh_clean.ply`
- frame sampling: `frame_skip=20`
- GT: `mesh surface ∩ sparse-input-view union frustum`
- no extra visible/occluded auxiliary labels
- reservoir density: `500k / scene`

### Formal processed dataset
- processed root:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k`
- formal split root:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17`
- scene counts:
  - `train=1362`
  - `val=151`
  - `test=100`

### Key implementation milestones
- true 4-view input wiring for ScanNet batches
- `pts3d_complete` dataset path from scene-level mesh reservoirs
- complete-GT builder script added
- DDP / `torchrun` conversion for MLP / CA / SA

## 4. Smoke / preflight milestones

### Complete-GT ScanNet smoke
- output dir:
  - `experiments/probe3d/result/scannet_mlp_complete_smoke_seed17`
- SwanLab run id:
  - `30iyh29o1orq7mnm097u4`

### Full-root MLP DDP preflight
- output dir:
  - `experiments/probe3d/result/scannet_mlp_ddp_fullroot_preflight_seed17`
- SwanLab run id:
  - `f7nivqbjlilypp5o9u4sz`

These runs matter because they show the current formal branch is not just a paper plan:

- full preprocess exists
- full split exists
- complete GT exists
- DDP launch path exists
- the full-root training path actually starts and runs

## 5. Superseded formal baseline run

The earlier formal run target was the **ScanNet MLP baseline**. This remains part of the history, but the active direction was later superseded by the shorter autoresearch-style probe in Section 6.

### Purpose
This is not meant as the final headline method.
Its purpose is to establish whether:

- frozen VGGT features
- a lightweight MLP adapter
- and a NOVA3R-style decoder

can learn stable complete 3D behavior under reliable mesh-first supervision at scale.

### Formal schedule
- 8 GPUs
- batch size `1 / rank`
- 50 epochs
- `steps_per_epoch = 13158`
- `max_steps = 657900`
- validation/checkpoint every 5 epochs

### Formal output dir
- `experiments/probe3d/result/scannet_mlp_adapter_l4_lr5e-5_seed17_epoch50_formal`

## 6. Autoresearch-style ScanNet probe update — objective mismatch isolated

A short harness was added under:

- `experiments/probe3d/autoresearch_probe/`

The purpose was to stop relying on a long formal run and instead test target modes / objectives quickly with immutable logging in `results.tsv`.

### Target-mode findings

Oracle token optimization showed that target support matters:

- `anchor_frustum`: oracle CD `0.23668508`
- `anchor_frustum_margin1.5`: oracle CD `0.81146899` — too broad / out of generator support
- `covered_by_ge2`: oracle CD `0.18723274`
- `covered_by_ge2_anchorfb`: oracle CD `0.20196436`

`anchor_frustum` became the stable adapter target because it avoided empty-support / fallback contamination and stayed within the anchor-camera generator domain.

### Objective findings

With `anchor_frustum`, the old `nova_flow` objective underperformed:

- `MLP-L4 + nova_flow`, step1000: `0.34512938`
- `MLP-L2-h512 + nova_flow`, step1000: `0.30276422`

Switching to direct sampled rollout Chamfer was the decisive change:

- `MLP-L4 + chamfer_sample`, step1000: `0.11552816`
- continued to step2000 at `lr=5e-5`: `0.09181590`
- continuing at `lr=5e-5` became unstable / overfit by step2500: `0.35143724`, so it was stopped
- resuming from step2000 with `lr=1e-5` refined to step2500: `0.08745259` — current best
- continuing the same low-LR branch to step3000 slightly worsened: `0.09132496`

### Visual / diagnostic finding

The numeric best is not yet visually satisfying. A GT-vs-pred video showed that the model covers much of the target support but produces a thick / loose / outlier-heavy point cloud.

Nearest-neighbor diagnostic on the visualized sample:

- `GT→Pred` mean distance: `0.0582`
- `Pred→GT` mean distance: `0.1777`

This means recall is acceptable but precision is poor. The current bottleneck is no longer simply “make CD go down”; it is prediction sharpness / outlier suppression.

### Resulting conclusion

The key learning is: direct rollout Chamfer fixes a large train/eval objective mismatch, but symmetric Chamfer alone can be gamed by noisy coverage. The next method change should be precision-aware: overweight `pred→GT`, use trimmed Chamfer, or add an outlier penalty.

## 7. Bottom line

The cleanest honest reading of the repo today is:

- old SCRREAM eval-subset claims are invalid
- the engineering stack survived that correction
- the ScanNet v2 mesh-first line is now the main trustworthy scale-up path
- the current best numeric baseline is `anchor_frustum + MLP-L4 + direct Chamfer`, CD `0.08745259`
- that numeric result is not visually clean enough: recall is acceptable, precision / outlier control is poor
- the next milestone is a precision-aware loss and visual improvement, before CA / SA comparison
## 8. Paper-aligned GT / loss correction after user review

Jiacheng clarified that NOVA3R's completion target is not a full-room point cloud. It is a complete / amodal point cloud **within the selected input-view frustum**: for example, if a table is in view, the model should recover surfaces such as the underside of the table within that frustum, not hallucinate the entire room.

This changes the interpretation of the previous `anchor_frustum` result. It was useful because it partially aligned with NOVA3R's target construction, but for K-view training the paper-aligned target should be the union of the selected input frusta, not just the first view. Direct Chamfer remains a diagnostic, while the main representation-probe objective should return to NOVA-native flow matching.

Implementation started:

- added explicit `nova_input_frustum` alias for union-of-selected-input-frusta complete targets
- added `nova_anchor_frustum` alias for K=1 / first-view debug targets
- exposed `scannet_complete_points` through loader, oracle, adapter, and autoresearch harness
- added `query_source` passthrough in the harness so trials can use `src_complete_fps_4096`
- created `experiments/probe3d/autoresearch_probe/configs/phase2_nova_aligned.json` with K=1/K=2 oracle and MLP-L4 native-flow adapter trials

Next result to record: K=1/K=2 oracle support for `nova_input_frustum + src_complete_fps_4096`.


## 9. Phase-2 paper-aligned frustum oracle controls

After Jiacheng's clarification that NOVA3R predicts complete/amodal geometry constrained to input-view frusta rather than whole-room completion, I added explicit target modes and controls for more paper-literal target construction.

Implementation notes:

- `nova_input_frustum`: union of the selected input-view frusta, collapsed into the existing `pts3d_complete` target path.
- `nova_per_view_frustum`: each input view contributes its own complete frustum crop, matching NOVA's `get_complete_pts3d()` per-view stacking convention more literally.
- `nova_per_view_frustum_anchor_zpos`: same as per-view but clipped to positive anchor-camera z for ScanNet stability.
- Exposed `scannet_complete_points` and `query_source` through the autoresearch harness.
- Fixed two runtime issues discovered by controls: ScanNet `num_views=1` sequence sampling crashed in `get_seq_from_start_id`, and local PyTorch3D FPS requires an explicit `max_K` argument.

Oracle results on the first two val samples were not promising:

- `nova_input_frustum`, K1, `src_complete_fps_4096`: mean CD `0.4766`
- `nova_input_frustum`, K2, `src_complete_fps_4096`: mean CD `1.3288`
- `nova_input_frustum`, K4, `src_complete`: mean CD `0.7692`
- `nova_input_frustum`, K4, `src_complete_fps_4096`: mean CD `1.3315`
- `nova_per_view_frustum`, K4, `src_complete_fps_4096`: mean CD `1.6023`
- `nova_per_view_frustum_anchor_zpos`, K4, `src_complete_fps_4096`: mean CD `0.9611`

Key interpretation: K1 controls are not directly comparable to the real 4-view probe because the NOVA decoder receives `num_views=1` conditioning, which appears out-of-domain. For the actual 4-view probe, the paper-literal union/per-view input-frustum targets fail oracle gating. The earlier 4-view `anchor_frustum` / `covered_by_ge2` targets remain more credible generator-domain targets (`anchor_frustum` oracle around `0.2367`, `covered_by_ge2` around `0.1872`). Therefore the next adapter experiments should not use `nova_input_frustum` yet; they should use oracle-supported 4-view target definitions and focus on native-flow or hybrid objective design.

## 10. Phase-4 feasibility pivot — K=2, MLP-L4, loss and GT sweep

After additional user review, the current priority was changed from exact NOVA3R GT reproduction to a cleaner **feasibility proof**: show that frozen VGGT representations can be adapted into a latent/token form that the NOVA generator can consume and roll out into meaningful 3D.

### Why the pivot happened

Two observations motivated the new sweep:

- The exact official NOVA3R GT construction is not fully observable from the released code/data path, so exact alignment should not block feasibility validation.
- Jiacheng pointed out that increasing the number of input views can itself reduce / empty the valid complete-target support under some target packagings. Quick loader stats supported this concern: K=4 often introduced empty view slots or lower average valid target counts than K=2.

### Completed quick controls before the overnight run

- `p4_oracle_norm_nova_per_view_ldi4_k2_fps4096_s2_step600`: CD `0.78956747`
- `p4_oracle_norm_anchor_frustum_k2_fps2048_s2_step600`: CD `0.10993152`

Interpretation:

- K=2 did **not** rescue the current LDI-style per-view target.
- K=2 made the stable `anchor_frustum` feasibility target much more generator-reachable than the previous K=4 setting.

### Active overnight loss ablation

The current active run is the K=2 / MLP-L4-H1024 / `anchor_frustum` loss ablation:

- `p4_k2_anchor_mlp_l4_flow_step1000` — native `nova_flow`
- `p4_k2_anchor_mlp_l4_hybrid005_step1000` — `nova_flow + 0.05 * rollout_chamfer`
- `p4_k2_anchor_mlp_l4_chamfer_step1000` — direct rollout Chamfer diagnostic

Driver log:

- `experiments/probe3d/result/autoresearch_probe/p4_k2_mlp_l4_loss_ablation_driver.out`

As of the documentation update, the first run was active around step ~676.

### Queued overnight GT-construction sweep

A continuation script waits for the active MLP-L4 loss ablation to finish and then launches additional K=2 controls.

Script / driver:

- script: `experiments/probe3d/result/autoresearch_probe/p4_overnight_after_l4_ablation.sh`
- driver log: `experiments/probe3d/result/autoresearch_probe/p4_overnight_after_l4_ablation_driver.out`
- launcher PID at creation time: `339478`

Queued oracle GT candidates:

- `nova_input_frustum`
- `covered_by_ge2`
- `anchor_frustum_margin`, margin `1.5`
- `nova_per_view_frustum_anchor_zpos`
- `nova_per_view_ldi2`
- `nova_per_view_ldi4`
- `nova_per_view_ldi8`

Queued adapter target/loss candidates:

- targets: `covered_by_ge2`, `nova_input_frustum`, `anchor_frustum_margin1.5`
- losses: native `nova_flow` and `flow_chamfer_hybrid` with `chamfer_weight=0.05`
- adapter: `MLP-L4-H1024`
- views: K=2
- queries: `2048`

### How to interpret tomorrow's results

The primary feasibility claim should come from the best K=2 result that satisfies both conditions:

1. The target has a plausible oracle ceiling.
2. The VGGT adapter approaches that ceiling enough to show the representation can be consumed by the NOVA generator.

Direct Chamfer remains a metric/visual diagnostic. Native flow and the small hybrid loss are more relevant to a generator-native representation claim.

### Step-count adjustment

Jiacheng requested that the training runs use slightly more steps. The short K=2 MLP-L4 native-flow run had already completed at step1000 with CD `0.91071419`, which is weak and should be treated as a partial / short-run diagnostic.

The overnight plan was updated to longer adapter schedules:

- resume `p4_k2_anchor_mlp_l4_flow_step1000` to `p4_k2_anchor_mlp_l4_flow_resume_step2000`
- run `p4_k2_anchor_mlp_l4_hybrid005_step2000`
- run `p4_k2_anchor_mlp_l4_chamfer_step2000`
- run the queued adapter GT/loss sweep at `2000` steps per trial instead of `1000`

New long-suite driver:

- `experiments/probe3d/result/autoresearch_probe/p4_overnight_k2_mlp_l4_long_driver.out`

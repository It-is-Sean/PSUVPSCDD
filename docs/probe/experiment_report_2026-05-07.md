# Probe Experiment Results Report — 2026-05-07

Generated at `2026-05-07 23:25 CST`.

Supersession note updated `2026-05-16`: this report predates the SCRREAM sequence-meta GT correction and the later WAN Route2 audit. The robust VGGT layer ablation job `86149` and its layer-20 recommendation are now **pre-meta-filter history**, not clean-GT claims. The clean-GT rerun `86286` completed successfully; current default VGGT layer is `16`, with layer `24` as the main comparison point. WAN Route2 sanity job `86316` and full pack job `86307` completed later; best WAN was `t499/layer09`, still far weaker than clean-GT VGGT. Route2.1 `no_noise` / `low_noise`, targeted `t499/layer09 + token_layernorm`, `pair_tiled81`, pred-x0 Conv2d readout, and simple hidden-grid readouts completed later and did not beat old Route2 best. The next WAN branch should test a generator-preserving Perceiver / cross-attention resampler on best WAN hidden tokens. Use this document only to understand the old evidence trail.

This report summarizes the experiment evidence currently available in this workspace. It separates valid current results from diagnostic history and invalidated branches, because several earlier conclusions were corrected after data and sampling audits.

Primary source files checked for this report:

- `experiments/probe3d/probe_trials/results.tsv`
- `experiments/probe3d/result/*/{config.json,final_metrics.json,validation_metrics.json}`
- `docs/probe/experiment_history.md`
- `docs/probe/experiment_plan.md`
- `experiments/probe3d/probe_trials/CURRENT_STATE.md`
- `PROJECT.md`
- `README.md`
- `experiments/probe3d/README.md`
- Slurm accounting for jobs `85773`, `85774`, `86128`, `86134`, `86138`, `86140`, `86147`

## Executive Summary

The strongest current branch is now the full-SCRREAM mesh-complete adapter line, not the old local `eval_scrream` line and not the older ScanNet-only line.

Valid-at-report-time SCRREAM baseline, now pre-meta-filter history:

- data: full SCRREAM from `~/datasets/SCRREAM`
- GT at the time: registered scene meshes, sampled by surface area, cropped to the two-input-view union frustum, stored in first input camera coordinates
- adapter: frozen VGGT features -> MLP-L4-H1024 -> frozen NOVA scene decoder
- objective: `nova_flow`
- best completed full run: 20k target points / 500k mesh reservoir / trainplus-test split
- best validation Chamfer-L2: `0.5149603486061096`

The 20k SCRREAM run is a real completed baseline, but not a final scientific claim yet:

- it merged the old test split into train, so it has `train=317`, `val=12`, and no held-out test split;
- its best validation metric is better than the 10k run (`0.51496` vs `0.52927`), but only slightly;
- the latest validation metric is worse than the best metric (`0.67792` at step `9500`), so checkpoint selection and visual inspection matter;
- representative PLYs should be inspected from `best.pth`, not only from the latest checkpoint.

The older ScanNet probe remains useful, but mostly as diagnosis:

- direct rollout Chamfer produced the best scalar ScanNet CD (`0.08745259`);
- robust fixed-sample evaluation showed the visual failure mode: decent recall but poor prediction-side precision and many outliers;
- ScanNet numbers with old `max_interval=30` are interval-confounded because preprocessing already used `frame_skip=20`;
- old local SCRREAM `eval_scrream` results are invalid for claims because the data was the released evaluation subset, not the full dataset.

The robust VGGT-layer ablation completed before the sequence-meta GT correction:

- Slurm job: `86149` / `scrream_vggt_layer_pack`
- layers tested: `0,4,8,12,16,20,24`
- validation: full 12-sample val split, robust one-way/F-score/trimmed-CD metrics, and `40960`-point visual exports
- historical pre-meta-filter best layer: `20`
- layer-20 metrics: `val_fscore_tau_0.10=0.672747712053826`, `val_pred_to_gt_p90=0.23175348962346712`, `val_gt_to_pred_p90=0.15852033160626888`, `val_chamfer_l2=0.03473521831134955`
- main qualitative comparison: layer `16`, which has the lowest prediction-side p90 distance (`0.20785426969329515`)

## Evidence Tiers At Report Time

### Tier 1 — Valid Evidence At Report Time

These were the results that could guide the next experiment before the 2026-05-11 sequence-meta GT correction:

- full SCRREAM mesh-complete `.pt` datasets generated from `~/datasets/SCRREAM`;
- completed SCRREAM MLP baseline `86140`;
- completed pre-meta-filter SCRREAM robust VGGT layer ablation, with layer `20` selected only as the historical default before the sequence-meta correction;
- ScanNet corrected-interval robust evaluation as a failure-mode diagnostic.

### Tier 2 — Useful Diagnostics

These are useful for engineering decisions but should not be proposal-facing claims by themselves:

- ScanNet `probe_trials/results.tsv` scalar CD sweep;
- two-sample oracle checks;
- direct Chamfer runs that optimize the same metric used for validation;
- cross-attention p7 first pass;
- SCRREAM 10k baseline before the 20k / trainplus-test rerun.

### Tier 3 — Invalid Or Retired Evidence

These should not be used for formal claims:

- old local `eval_scrream` experiments;
- ScanNet K-view conclusions from runs using old `max_interval=30`;
- `p1_adapter_covered_ge2_mlp_l4_step1000`, invalidated by `src_complete` fallback contamination;
- first stopped `chamfer_sample` branch before DDP gradient synchronization was fixed;
- local `scrream_official_depth_mix_*` artifacts unless a future explicit depth-mix ablation is requested.

## SCRREAM Data Construction At Report Time

The active adapter data bridge is:

- `experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py`

The report-time GT source was `mesh_complete`, not dense-depth aggregation. The current post-2026-05-11 version adds sequence `meta.txt` filtering before mesh sampling.

Pipeline:

1. Read official SCRREAM two-view pairs from `data/scrream/scrream_n2_list.json`.
2. Use the two RGB frames in each pair as adapter inputs.
3. Read registered `sceneXX/meshes/*.obj` meshes.
4. Sample mesh surfaces proportional to estimated surface area.
5. Voxel-deduplicate the scene reservoir.
6. Crop world points to the union frustum of the two input views.
7. Transform target points into the first input camera coordinate frame.
8. FPS sample to `target_points`, or pad with replacement if the crop has fewer points.

This construction is the closest current match to the intended NOVA-style target: complete / amodal geometry within the selected input-view frustum, not full-room reconstruction outside the input support.

Important distinction from the older dense-depth idea:

- `depth_gt_dense` aggregates observed `depth_gt` frames between the two input frames. It can contain surfaces invisible in the first input view if intermediate frames saw them, but it is still limited to observed depth surfaces.
- `mesh_complete` samples registered scene meshes. In the current corrected version, those meshes are first filtered by the sequence `meta.txt`. It can provide surfaces that are not directly visible in the input RGB/depth frames but lie inside the input-view frustum. This is better aligned with training an adapter for completion.

## Generated SCRREAM Adapter Datasets

The following `.pt` files were verified directly with `torch.load`.

| file | target shape | split | target source | mesh reservoir |
|---|---:|---|---|---:|
| `scrream_mesh_complete_n2_adapter_seed17.pt` | `[329,10000,3]` | `train=223,val=12,test=94` | `mesh_complete` | `250000` |
| `scrream_mesh_complete_n2_adapter_seed17_trainplus_test.pt` | `[329,10000,3]` | `train=317,val=12` | `mesh_complete` | `250000` |
| `scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000.pt` | `[329,20000,3]` | `train=223,val=12,test=94` | `mesh_complete` | `500000` |
| `scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt` | `[329,20000,3]` | `train=317,val=12` | `mesh_complete` | `500000` |

The `trainplus_test` files are split-label rewrites:

- old `test` samples are relabeled to `train`;
- `val` remains unchanged;
- these files are useful for fitting a stronger baseline on all available non-val pairs;
- they cannot support held-out test claims.

The first sample metadata was also checked. Its RGB `frame_paths` point to existing files in the full SCRREAM tree.

## SCRREAM Training Results

### 10k / 250k Baseline

Run directory:

- `experiments/probe3d/result/scrream_mesh_complete_n2_mlp_l4_nova_flow_seed17`

Configuration:

- dataset: `scrream_mesh_complete_n2_adapter_seed17.pt`
- target points: `10000`
- mesh sample points: `250000`
- split: `train=223,val=12,test=94`
- adapter: `mlp`, `4` layers, hidden dim `1024`
- loss: `nova_flow`
- num queries: `10000`
- max steps: `6690`

Metrics:

| metric | value |
|---|---:|
| first train loss | `1.4251021146774292` |
| final train loss | `0.9666423201560974` |
| best train loss | `0.6534841060638428` |
| best val Chamfer-L2 | `0.5292698293924332` |
| latest val step | `6500` |
| latest val Chamfer-L2 | `0.6747700572013855` |

Interpretation:

- The model trains and validation gets below `0.6` at best.
- The latest validation is worse than the best validation, so this run is not monotonically improving.
- This was the first full mesh-complete baseline and is useful as a reference, but the later 20k run supersedes it for current comparisons.

### 10k Retry Smoke

Run directory:

- `experiments/probe3d/result/scrream_mesh_complete_n2_mlp_l4_nova_flow_seed17_retry_smoke`

Configuration:

- same 10k dataset
- max steps: `20`

Metrics:

| metric | value |
|---|---:|
| first loss | `1.4251021146774292` |
| final loss | `1.1968979835510254` |
| best loss | `1.0065158605575562` |
| step-20 val Chamfer-L2 | `0.5783020257949829` |

Interpretation:

- This is only a smoke test.
- It validates that the loader/training path can start, not model quality.

### 20k / 500k Trainplus-Test Baseline

Slurm job:

- job id: `86140`
- job name: `scrream_mesh_mlp`
- state: `COMPLETED`
- exit code: `0:0`
- node: `air-node-02`
- GPU: `1 x A100`
- elapsed: `00:26:26`
- start: `2026-05-07T22:13:56`
- end: `2026-05-07T22:40:22`

Run directory:

- `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17`

Configuration:

- dataset: `scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
- target points: `20000`
- mesh sample points: `500000`
- split: `train=317,val=12`
- adapter: `mlp`, `4` layers, hidden dim `1024`
- loss: `nova_flow`
- num queries: `20000`
- max steps: `9510`
- SwanLab run: `https://swanlab.cn/@JiachengDong/PSUVPSC3DD/runs/eoiupi3bv2g11dvd21ypm`

Metrics:

| metric | value |
|---|---:|
| first train loss | `1.4599288702011108` |
| final train loss | `0.8398033976554871` |
| best train loss | `0.5998285412788391` |
| best val Chamfer-L2 | `0.5149603486061096` |
| latest val step | `9500` |
| latest val Chamfer-L2 | `0.6779176592826843` |

Interpretation:

- Compared with the 10k baseline, best validation Chamfer improves from `0.52927` to `0.51496`, about a small `2.7%` relative improvement.
- This is not a clean one-variable ablation: target density, reservoir density, and train sample count all changed.
- The latest validation metric is again worse than the best validation metric, so the run should be interpreted through `best.pth`.
- The result is strong enough to justify follow-up ablations, but not yet strong enough for a final claim without visual and robust metrics.

### SCRREAM Robust VGGT Layer Ablation

Slurm job:

- job id: `86149`
- job name: `scrream_vggt_layer_pack`
- state: `COMPLETED`
- exit code: `0:0`
- GPUs: `4 x A100`
- elapsed: `02:00:08`
- layers tested: `0,4,8,12,16,20,24`

Configuration:

- dataset: `scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
- split: `train=317,val=12`
- target points: `20000`
- mesh sample points: `500000`
- adapter: `mlp`, `4` layers, hidden dim `1024`
- loss: `nova_flow`
- num queries: `20000`
- max steps: `9510`
- validation: full val split (`eval_batches=0`)
- robust metric points: `20000`
- visual preview points: `40960`

Final robust metrics:

| layer | F@0.10 ↑ | F@0.05 ↑ | pred→GT p90 ↓ | GT→pred p90 ↓ | trimmed CD95 ↓ | Chamfer ↓ | velocity MSE ↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `0.360445` | `0.158992` | `0.757063` | `0.217211` | `0.160015` | `0.813673` | `1.219393` |
| `4` | `0.426661` | `0.200347` | `0.523990` | `0.295402` | `0.091500` | `0.545035` | `1.234664` |
| `8` | `0.430191` | `0.197705` | `0.513064` | `0.209427` | `0.079848` | `0.271475` | `1.229198` |
| `12` | `0.467699` | `0.221818` | `0.430590` | `0.236178` | `0.063575` | `0.100242` | `1.213175` |
| `16` | `0.649718` | `0.347173` | **`0.207854`** | `0.187817` | `0.021746` | `0.034842` | `1.166274` |
| `20` | **`0.672748`** | **`0.361666`** | `0.231753` | **`0.158520`** | **`0.020939`** | **`0.034735`** | **`1.154195`** |
| `24` | `0.661291` | `0.360759` | `0.241899` | `0.170757` | `0.022651` | `0.038710` | `1.171148` |

Interpretation:

- Layers `16/20/24` are in a clearly better regime than `0/4/8/12`.
- Layer `20` is the best default choice by F-score, GT coverage, trimmed CD, Chamfer, and velocity MSE.
- Layer `16` is still worth visual comparison because it has the lowest prediction-to-GT p90 distance and may produce fewer outlier points.
- Layer `24` is close to layer `20`, but does not win any primary aggregate metric.

Visual inspection directories:

- `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_seed17_vggt_layer16/val_visual_40960/step_009500/`
- `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_seed17_vggt_layer20/val_visual_40960/step_009500/`
- `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_seed17_vggt_layer24/val_visual_40960/step_009500/`
- The new auxiliary metrics are useful: `val_velocity_mse` directly monitors NOVA flow target quality, while `loss_per_t_bin` shows whether errors concentrate at early or late diffusion/flow times.

## SCRREAM Operational History

Relevant Slurm accounting:

| job | name | state | exit | elapsed | note |
|---:|---|---|---|---:|---|
| `85773` | `scrream_mesh_prep` | `COMPLETED` | `0:0` | `00:26:47` | first full 10k data generation |
| `85774` | `scrream_mesh_mlp` | `FAILED` | `1:0` | `00:01:24` | missing `torch_cluster`; SwanLab API not configured at that time |
| `86128` | `scrream_mesh_prep` | `COMPLETED` | `0:0` | `00:48:46` | 20k / 500k data generation |
| `86134` | `scrream_split_merge` | `COMPLETED` | `0:0` | `00:00:24` | rewrote `test -> train`, kept `val` |
| `86138` | `scrream_mesh_mlp` | `FAILED` | `1:0` | `00:01:04` | VGGT submodule was empty after conversion |
| `86140` | `scrream_mesh_mlp` | `COMPLETED` | `0:0` | `00:26:26` | completed 20k trainplus-test baseline |
| `86147` | `scrream_vggt_layer_pack` | `RUNNING` | `0:0` so far | running | VGGT layer ablation |

Operational lessons:

- Submodules must be initialized before training: `git submodule update --init --recursive`.
- Long runs should keep using Slurm scripts in `slurm/` with logs in `slurm_out/`.
- Network-dependent training should keep proxy defaults at `http://127.0.0.1:7896`.
- SwanLab API should stay in ignored local secrets, not in tracked scripts or docs.

## ScanNet Probe Results

The ScanNet branch is now diagnostic, but it produced the most detailed controlled ablations so far.

### Major Correction

Processed ScanNet data already used `frame_skip=20`.

Therefore:

- old `max_interval=30` meant views could be around `600` raw frames apart;
- those old K-view conclusions are interval-confounded;
- corrected local-overlap tests use `scannet_max_interval=1`.

This correction changes how the old ScanNet table should be read. The experiments are still informative about objective mismatch, target support, and outlier failure, but the old K-view comparison is not clean.

### Best Scalar ScanNet Adapter Result

From `experiments/probe3d/probe_trials/results.tsv`:

| trial | target | adapter | loss | steps | val CD |
|---|---|---|---|---:|---:|
| `p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step2500` | `anchor_frustum` | `mlp_l4_h1024` | `chamfer_sample` | `2500` | `0.08745259` |
| `p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step3000` | `anchor_frustum` | `mlp_l4_h1024` | `chamfer_sample` | `3000` | `0.09132496` |
| `p1_adapter_anchor_frustum_mlp_l4_chamfer_8gpu_resume_step2000` | `anchor_frustum` | `mlp_l4_h1024` | `chamfer_sample` | `2000` | `0.09181590` |
| `p1_adapter_anchor_frustum_mlp_l4_chamfer_ddpfix_8gpu_resume_step1000` | `anchor_frustum` | `mlp_l4_h1024` | `chamfer_sample` | `1000` | `0.11552816` |

Interpretation:

- Direct rollout Chamfer dramatically lowered scalar validation CD.
- Continuing with high LR became unstable, while low-LR refinement gave the best number.
- The best scalar CD is not equivalent to a good visual reconstruction.

### Robust ScanNet Evaluation

The fixed-30 robust eval of the K2 / interval=1 / MLP-L4 / Chamfer checkpoint found:

| metric | value |
|---|---:|
| symmetric CD mean / median / p90 | `0.0504 / 0.0381 / 0.0639` |
| trimmed CD95 mean / median | `0.0335 / 0.0265` |
| pred-to-GT mean distance mean / median | `0.151 / 0.142` |
| GT-to-pred mean distance mean / median | `0.069 / 0.067` |
| F@0.05 mean / median | `0.291 / 0.275` |
| precision@0.05 mean | `0.204` |
| recall@0.05 mean | `0.532` |
| worst row | `scene0000_02_00154/00155`, F@0.05 `0.0266` |

Failure audit:

- median recall-minus-precision gap at `tau=0.05`: `0.3126`
- pred-to-GT p90 correlation with F@0.05: `r=-0.8263`

Interpretation:

- The model has moderate recall: it covers a meaningful part of GT support.
- Precision is poor: many predicted points are far from GT.
- Symmetric CD alone hides this because broad noisy predictions can partially cover the GT while still being visually bad.
- This is why future reports should include one-way distances, F-score thresholds, and representative renders.

## ScanNet Target-Mode Findings

### Early Oracle Checks

Representative oracle results:

| trial | target | val CD | interpretation |
|---|---|---:|---|
| `p1_oracle_covered_ge2_s2_step600` | `covered_by_ge2` | `0.18723274` | good early target support |
| `p1_oracle_covered_ge2_anchorfb_s2_step600` | `covered_by_ge2_anchorfb` | `0.20196436` | fallback version plausible |
| `p1_oracle_anchor_frustum_s2_step600` | `anchor_frustum` | `0.23668508` | stable and generator-reachable |
| `p1_oracle_anchor_frustum_margin1.5_s2_step600` | expanded anchor frustum | `0.81146899` | too broad / out of support |

Early conclusion:

- `anchor_frustum` was chosen as a stable feasibility target because it avoided empty support and stayed in the generator domain.
- `covered_by_ge2` had good oracle numbers but was more fragile in adapter training due to fallback contamination.

### Paper-Aligned Frustum / LDI Reset

The user clarified that NOVA3R-style completion should be complete / amodal geometry inside selected input-view frusta, not the full room.

Implemented target ideas:

- `nova_input_frustum`
- `nova_per_view_frustum`
- `nova_per_view_frustum_anchor_zpos`
- LDI-style per-view variants

Representative results:

| trial | target | K | steps | val CD |
|---|---|---:|---:|---:|
| `p2_oracle_nova_input_frustum_k1_fps4096_s2_step400` | `nova_input_frustum` | `1` | `400` | `0.47663970` |
| `p2_oracle_nova_input_frustum_k2_fps4096_s2_step400` | `nova_input_frustum` | `2` | `400` | `1.32879451` |
| `p2_oracle_nova_per_view_frustum_k4_fps4096_s2_step400` | `nova_per_view_frustum` | `4` | `400` | `1.60226032` |
| `p2_oracle_nova_per_view_frustum_anchor_zpos_k4_fps4096_s2_step400` | anchor-z positive per-view | `4` | `400` | `0.96107247` |
| `p3_oracle_norm_nova_input_frustum_k4_fps4096_s2_step2000` | `nova_input_frustum` | `4` | `2000` | `0.24589425` |
| `p3_oracle_norm_nova_per_view_ldi4_k4_fps4096_s2_step2000` | `nova_per_view_ldi4` | `4` | `2000` | `0.29746810` |
| `p3_adapter_nova_per_view_ldi4_k4_mlp_l4_flow_fps4096_step1000` | `nova_per_view_ldi4` | `4` | `1000` | `1.04477810` |

Interpretation:

- The paper-aligned targets were conceptually better, but many were hard for the current NOVA generator / adapter path.
- Restoring the decoder checkpoint `norm_mode=median_3` helped oracle support.
- Adapter performance on these targets remained weak, so exact target alignment did not immediately solve the problem.

### K2 Pivot

The later K2 sweep showed that K2 was often more stable than K4 for this probe.

Representative oracle results:

| trial | target | val CD |
|---|---|---:|
| `p4_oracle_norm_anchor_frustum_k2_fps2048_s2_step600` | `anchor_frustum` | `0.10993152` |
| `p4_oracle_norm_covered_ge2_k2_fps2048_s2_step600` | `covered_by_ge2` | `0.12180443` |
| `p4_oracle_norm_anchor_margin15_k2_fps2048_s2_step600` | `anchor_frustum_margin` | `0.23195412` |
| `p4_oracle_norm_nova_input_frustum_k2_fps2048_s2_step600` | `nova_input_frustum` | `0.76464893` |

Representative adapter results:

| trial | target | loss | steps | val CD |
|---|---|---|---:|---:|
| `p4_k2_anchor_mlp_l4_flow_step1000` | `anchor_frustum` | `nova_flow` | `1000` | `0.91071419` |
| `p4_k2_anchor_mlp_l4_flow_resume_step2000` | `anchor_frustum` | `nova_flow` | `2000` | `0.85699082` |
| `p4_k2_anchor_mlp_l4_hybrid005_step2000` | `anchor_frustum` | `flow_chamfer_hybrid` | `2000` | `0.66678675` |
| `p4_k2_anchor_mlp_l4_chamfer_step2000` | `anchor_frustum` | `chamfer_sample` | `2000` | `0.54158844` |
| `p4_k2_input_frustum_mlp_l4_hybrid005_step2000` | `nova_input_frustum` | `flow_chamfer_hybrid` | `2000` | `0.78693782` |
| `p4_k2_anchor_margin15_mlp_l4_hybrid005_step2000` | `anchor_frustum_margin` | `flow_chamfer_hybrid` | `2000` | `0.76487390` |

Interpretation:

- K2 anchor oracle looked very reachable.
- Adapter still had a large gap to oracle under `nova_flow`.
- Adding a small Chamfer auxiliary helped.
- Direct Chamfer again gave better scalar CD, but this repeats the precision/outlier risk seen in robust eval.

### Corrected Interval Sweep

With `scannet_max_interval=1`, representative oracle results were:

| trial | target | K | val CD |
|---|---|---:|---:|
| `p6_k2_i1_oracle_covered_ge2_s2_step600` | `covered_by_ge2` | `2` | `0.06886154` |
| `p6_k2_i1_oracle_input_frustum_s2_step600` | `nova_input_frustum` | `2` | `0.11214603` |
| `p6_k2_i1_oracle_anchor_s2_step600` | `anchor_frustum` | `2` | `0.13334342` |
| `p6_k4_i1_oracle_anchor_s2_step600` | `anchor_frustum` | `4` | `0.11520538` |
| `p6_k4_i1_oracle_covered_ge2_s2_step600` | `covered_by_ge2` | `4` | `0.85016728` |
| `p6_k4_i1_oracle_input_frustum_s2_step600` | `nova_input_frustum` | `4` | `1.37526729` |

Adapter results:

| trial | target | loss | val CD |
|---|---|---|---:|
| `p6_k2_i1_anchor_mlp_l4_flow_step1000` | `anchor_frustum` | `nova_flow` | `0.75662778` |
| `p6_k2_i1_anchor_mlp_l4_hybrid005_step1000` | `anchor_frustum` | `flow_chamfer_hybrid` | `0.72299022` |

Interpretation:

- Correcting the interval changed the oracle story substantially.
- K2 corrected-interval targets can be very generator-reachable.
- The adapter gap remains large under flow / hybrid losses.
- K4 target construction can still become unstable or out-of-domain.

## Loss Function Findings

### `nova_flow`

Pros:

- It is closest to the NOVA decoder training semantics.
- It is the right default for proposal-aligned adapter experiments.

Cons observed:

- On ScanNet anchor-frustum, it underperformed direct rollout Chamfer by a large margin in scalar CD.
- On SCRREAM, it trains but validation can drift upward after the best checkpoint.

### `chamfer_sample`

Pros:

- It directly optimizes the sampled point-cloud validation metric.
- It produced the best ScanNet scalar CD.

Cons observed:

- It can create broad/noisy point clouds that cover GT but have poor precision.
- Robust metrics showed low precision@0.05 despite decent recall.
- It should be treated as a diagnostic or auxiliary, not the sole proposal-facing objective.

### `flow_chamfer_hybrid`

Pros:

- It improved over pure flow in K2 ScanNet ablations.

Cons observed:

- It did not close the gap to direct Chamfer or oracle.
- The tested `0.05` Chamfer weight is not proven optimal.

Practical conclusion:

- Keep `nova_flow` as the current SCRREAM baseline objective.
- Add robust validation metrics so we can see whether flow is improving the right behavior.
- Use Chamfer or trimmed/precision-aware losses as controlled ablations, not as headline evidence until visual quality is verified.

## Adapter Architecture Findings

### MLP

The MLP adapter is the current baseline:

- simple;
- stable enough to run on SCRREAM;
- enough to test whether frozen VGGT features carry usable reconstruction signal.

Current limitation:

- it does not yet prove strong geometry recovery;
- ScanNet robust eval suggests it may produce recall-heavy but imprecise predictions.

### Cross-Attention

The p7 cross-attention candidate:

- trial: `p7_k2_i1_anchor_ca_l2_h512_chamfer_step1000`
- target: `anchor_frustum`
- K / interval: `K=2`, `scannet_max_interval=1`
- validation CD: `0.54222615`

Interpretation:

- It was not a scalar improvement over the MLP Chamfer baseline.
- It still needs robust fixed-sample eval and visual inspection before any architectural conclusion.

### VGGT Layer Ablation

The completed robust layer ablation answered the immediate representation-layer question:

- layer `0` tests DINO-only features;
- layers `4,8,12,16,20,24` test progressively deeper VGGT representations;
- validation logs `val_velocity_mse`, `loss_per_t_bin`, one-way distance, F-score, and trimmed CD.

Historical conclusion before the sequence-meta correction:

- late VGGT layers are clearly stronger than DINO-only / early VGGT layers;
- layer `20` was the best pre-meta-filter representation;
- layer `16` should remain in visual comparisons because it has the lowest prediction-side p90 distance.

## GT Construction Conclusions

### What Worked Best Conceptually

For SCRREAM, `mesh_complete` remains the preferred GT construction, but after 2026-05-11 it must use the sequence-meta-filtered mesh set before limiting supervision to the union of the two input-view frusta.

This avoids two bad extremes:

- only visible depth, which does not train completion well;
- whole-room completion, which asks the model to predict geometry outside the input support.

### What Remains Risky

Even with mesh-complete GT:

- mesh registration quality must be trusted scene by scene;
- frustum crop correctness matters;
- target density can change training behavior but does not automatically solve adapter representation alignment;
- `trainplus_test` improves training sample count but removes held-out test evidence.

### Density Question

Increasing SCRREAM targets from `10000` to `20000` points and mesh reservoirs from `250000` to `500000` gave a small best-val improvement:

- 10k best val: `0.5292698293924332`
- 20k best val: `0.5149603486061096`

This suggests density helps somewhat, but density is not the main bottleneck by itself. The bigger remaining issue is whether the adapter produces precise, well-distributed decoder conditioning.

## Current Research Read

The honest read of all experiments so far is:

1. The infrastructure now works end to end on full SCRREAM.
2. Mesh-complete SCRREAM GT is visually accepted and better aligned with completion than depth aggregation.
3. A simple MLP adapter can train, but current metrics do not yet prove high-quality reconstruction.
4. Increasing target/query density from 10k to 20k helped only slightly.
5. ScanNet showed that low Chamfer can hide visually bad precision/outlier behavior.
6. Future SCRREAM claims must include robust metrics and PLY/render inspection, not only `best_val_chamfer_l2`.
7. The pre-meta-filter VGGT layer ablation suggested layer `20` as the best old representation, with layer `16` as the main outlier-sensitive comparison. This must be re-checked on the clean sequence-meta-filtered GT.

## Recommended Next Steps

1. Use clean-GT VGGT layer `16` as the current default representation and layer `24` as the closest comparison point.
2. Compare the clean-GT layer `16` and `24` PLYs before making a qualitative visual claim.
3. Expand SCRREAM training pairs beyond the current 329 official pairs.
4. Keep `trainplus_test` for fitting experiments, but regenerate a clean held-out split before making any final claim.
5. Run a controlled density ablation after the layer-16 baseline is established:
   - same split;
   - same layer;
   - same steps;
   - compare 10k vs 20k vs possibly 40k if memory permits.
6. Test more adapter/model variants after the data-scale question is addressed.
7. Test a precision-aware auxiliary only after the layer-16 baseline is stable:
   - trimmed Chamfer;
   - stronger pred-to-GT penalty;
   - outlier clipping / bounded radius penalty.
8. Do not revive `depth_mix` or old `eval_scrream` unless explicitly running a historical ablation.

## Bottom Line

The project is past the data-bridge stage: full SCRREAM mesh-complete data generation and MLP adapter training now run successfully. The SCRREAM data bridge is V0.5: it is wired end to end and visually plausible, but still needs larger training scale and broader model coverage.

The pre-meta-filter completed VGGT probe baseline used layer `20`, which achieved `val_fscore_tau_0.10=0.672747712053826` on the 12-sample scene08 val split. After the 2026-05-11 sequence-meta GT correction, this is a historical comparison point rather than the current default. The clean-GT rerun now favors layer `16` (`best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`), with layer `24` close on F-score. WAN Route2 job `86307` later completed; best WAN (`t499/layer09`, `F@0.10=0.46988987902779306`) remained far below clean-GT VGGT. Route2.1 `no_noise` / `low_noise`, targeted `t499/layer09 + token_layernorm`, and `pair_tiled81` completed later and did not beat old Route2 best. The active WAN tensor-choice check is `pred_x0_latent` with Conv2d readout; SCRREAM training data scale expansion follows after this WAN audit while keeping robust metrics and `val_visual_40960/` inspection enabled.

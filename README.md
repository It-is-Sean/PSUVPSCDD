# PSUVPSC3DD / Probe Workspace

## Current canonical status — 2026-05-20

This branch is now a **server-side research workspace**. The source of truth is:

- `PROPOSAL.md`
- `PROJECT.md`
- `experiments/probe3d/README.md`
- `docs/probe/wan_summary_2026-05-19.md`
- `docs/probe/handoff_2026-05-07.md`
- `docs/probe/handoff_2026-05-03.md`
- `docs/probe/experiment_plan.md`

Important corrections that override older sections below:

1. **ScanNet view interval:** processed data already uses `frame_skip=20`; corrected experiments use `scannet_max_interval=1` (adjacent processed frames, roughly 20 raw frames). Older `max_interval=30` runs are interval-confounded.
2. **Metric reliability:** single symmetric CD and two-sample oracle averages are diagnostic only. Current comparisons use fixed samples, pred-to-GT precision, GT-to-pred recall, F-score thresholds, trimmed CD, and representative renders.
3. **Current MLP baseline:** K2/interval=1 `anchor_frustum + MLP-L4 + chamfer_sample` is recall-heavy but precision/outlier-poor on fixed-30 robust eval: F@0.05 mean/median `0.291/0.275`, precision@0.05 mean `0.204`, recall@0.05 mean `0.532`.
4. **Latest structured-adapter check:** K2/interval=1 `anchor_frustum + cross_attention L2/H512 + chamfer_sample` completed 1000 steps with validation CD `0.54222615`, which is not better than the MLP baseline by scalar validation CD. Its fixed-30 robust eval should be inspected before any claim if result artifacts are available.
5. **SCRREAM full-data status:** the old `eval_scrream` package is still invalid for claims, but the full SCRREAM tree is now available locally at `~/datasets/SCRREAM`.
6. **Current SCRREAM GT path:** the active data bridge uses registered sequence-filtered scene meshes as the complete target source. For `mesh_complete`, it reads each SCRREAM sequence `meta.txt`, samples only the listed object meshes from `sceneXX/meshes/*.obj` proportional to surface area, crops to the selected two-view union frustum, transforms targets into the first input camera frame, and exports fixed-size adapter targets.
7. **Slurm convention:** long data generation and training jobs should be launched through scripts in `slurm/`, with logs in `slurm_out/`.
8. **SCRREAM baseline status:** job `86140` completed the old 20k/500k SCRREAM mesh-complete MLP baseline, and job `86149` completed the pre-meta-filter robust VGGT layer ablation. A sequence-meta-filtered clean GT was regenerated on `2026-05-10`; clean-GT VGGT job `86286` completed successfully on `2026-05-11`. The current clean-GT default is now VGGT1 MLP layer `16` at 50 epochs from job `86571`, `F@0.10=0.702544731989172`; VGGT1 CA layer `20` from job `86572` is the strongest readout comparison, `F@0.10=0.7014525243138799`.
9. **Local weights:** NOVA3R `scene_n1`, `scene_n2`, `scene_ae`, and VGGT weights are staged under `checkpoints/`; non-WAN Slurm scripts default network proxy variables to `http://127.0.0.1:7896`.
10. **Third-party source:** VGGT, VGGT-Omega, Wan2.1, and VidFM3D are Git submodules under `third_party/`; run `git submodule update --init --recursive` after a fresh clone. Wan2.1 / WAN probe dependencies stay separate from the root env.
11. **WAN Route2 status:** WAN is now a paused / closed model-coverage branch as of `2026-05-19`; see `docs/probe/wan_summary_2026-05-19.md`. It used WAN2.1 T2V video-context features on the same clean SCRREAM `.pt`, split, adapter/decoder family, and robust validation setup as the VGGT ablation. The learned hidden CA resampler was the only clearly positive WAN readout: historical best was `t499/layer14 + wan_cross_attn_resampler` with `F@0.10=0.5012956284974035`, `pred_to_gt_p90=0.4229188362757365`, `Chamfer=0.10908368105689685`, and user visual inspection improved. Repeated settings were closer to `0.49`, and L4 CA `t249/layer14` reached `F@0.10=0.4919946248489035`. WAN remains below clean-GT VGGT layer `16` (`F@0.10=0.6860468604251301`), and no broad WAN sweeps are recommended before testing the next backbone. WAN repo/checkpoint jobs use proxy `http://127.0.0.1:17890` through the Slurm SSH tunnel logic in `slurm/scrream_wan_*`.
12. **VGGT-Omega status:** VGGT-Omega is wired as a new backbone through `third_party/vggt-omega` and `--backbone vggt_omega`. Use `checkpoints/vggt_omega/vggt_omega_1b_512.pt`; `checkpoints/vggt_omega/model.pt` is an old-VGGT duplicate and invalid for Omega claims. Omega dense/full-token MLP, dense/full-token CA, and register-only / "Frozen Scene Tokens" CA have completed for layers `12,16,20,24`. Best Omega is dense CA layer `16` (`F@0.10=0.6576072630447046`), still well below VGGT1 MLP layer `16` (`0.702544731989172`) and VGGT1 CA layer `20` (`0.7014525243138799`).
13. **VGGT1 50ep baseline completion:** jobs `86571` (MLP-L4-H1024) and `86572` (cross_attention-L4-H1024) completed `0:0` on `air-node-02`. MLP layer16 is the new official VGGT1 baseline (`F@0.10=0.702544731989172`, `p90=0.19675995161135992`, `Chamfer=0.027118226668486994`). CA layer20 is nearly tied on F-score (`0.7014525243138799`) and has the best Chamfer among the 50ep readouts (`0.027010128212471802`).

This repository is currently a **research execution workspace** around a simple question:

> can frozen visual backbones (currently VGGT-first) drive a NOVA3R-style decoder through a lightweight adapter, and learn useful complete 3D reconstruction behavior under reliable supervision?

It contains:

- the NOVA3R / DUST3R style reconstruction stack used here as the decoder/data backbone
- the probe / adapter experiments under `experiments/probe3d/`
- proposal-facing documentation under `docs/probe/`

## Current experimental status

There are now four lines, and they should be interpreted differently.

### 1. SCRREAM full-data line

A retrospective audit found that the earlier local `eval_scrream` package was only the released **evaluation subset** (~1.6 GB), not the official full training-scale dataset.

So:

- the older SCRREAM quantitative results are **invalid as formal scientific evidence**
- they are still useful as **pipeline/debugging history**
- they should not be used as proposal feasibility claims

The corrected full-data branch is now active because the full SCRREAM tree is present at:

- `~/datasets/SCRREAM`

The current target source is **mesh-complete**, not the earlier eval-subset pseudo-GT and not the depth-only bridge by default:

- input samples come from `data/scrream/scrream_n2_list.json`
- RGB inputs are the two frames in each official pair
- GT points are sampled from `sceneXX/meshes/*.obj` after filtering by the current sequence's `meta.txt`
- mesh sampling is surface-area proportional, so walls / room structure are not underweighted relative to small objects
- targets are cropped to the union frustum of the two input views
- final `target_points` are stored in the first input camera coordinate frame
- the output `.pt` is consumed through `--dataset scrream_adapter --data_root <adapter.pt>`
- for `nova_flow`, SCRREAM targets are normalized with the NOVA `scene_ae` checkpoint `norm_mode`; the local `scene_ae` config reports `median_3`

Current data / job state recorded on 2026-05-18:

- 10k original adapter data exists:
  - `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.pt`
  - shape `[329, 10000, 3]`, split `train=223`, `val=12`, `test=94`
- 10k trainplus-test data exists:
  - `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_trainplus_test.pt`
  - split `train=317`, `val=12`
- 20k / 500k data exists:
  - `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000.pt`
  - shape `[329, 20000, 3]`, split `train=223`, `val=12`, `test=94`
- current formal training input exists:
  - `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
  - shape `[329, 20000, 3]`, split `train=317`, `val=12`, `mesh_sequence_meta_filter=True`
  - regenerated by Slurm jobs `86283` and `86284` after deleting old scene-level mesh cache; old contaminated `.pt` files were moved to `experiments/probe3d/adapter_data/deprecated_meta_filter_bug/`
- the sequence-meta filter removes scene-level phantom objects that are not listed in a sequence `meta.txt`; for example `scene08/scene08_reduced_00_000220_000260` now selects 22 mesh files and excludes the three mannequin OBJ files
- `86140` / `scrream_mesh_mlp`: `COMPLETED` on `air-node-02`
  - output: `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17`
  - SwanLab: `https://swanlab.cn/@JiachengDong/PSUVPSC3DD/runs/eoiupi3bv2g11dvd21ypm`
  - final metrics: first loss `1.4599288702011108`, final loss `0.8398033976554871`, best train loss `0.5998285412788391`, best validation Chamfer-L2 `0.5149603486061096`
- `86149` / `scrream_vggt_layer_pack`: `COMPLETED` on `2026-05-08`, but it used the pre-meta-filter GT and is now historical rather than the clean-GT baseline
- `86286` / `scrream_vggt_layer_pack`: `COMPLETED` on `air-node-04`, exit `0:0`, elapsed `02:23:11`; output prefix is `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_vggt_layerXX`
- clean-GT layer ranking now uses robust validation, not the old pre-meta-filter result:
  - layer `16`: `best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`, `best_val_chamfer_l2=0.026904070439438026`
  - layer `24`: `best_val_fscore_tau_0.10=0.6802513448244976`, `best_val_pred_to_gt_p90=0.25152526050806046`, `best_val_chamfer_l2=0.03554012098660072`
  - layer `20`: `best_val_fscore_tau_0.10=0.548158719135383`, `best_val_pred_to_gt_p90=0.36711231619119644`, `best_val_chamfer_l2=0.0767408860847354`

### 2. WAN2.1 T2V Route2 line

The WAN Route2 branch was the first post-VGGT model-coverage expansion after the clean-GT VGGT rerun. It is now paused after the `2026-05-19` audit; use `docs/probe/wan_summary_2026-05-19.md` for the frozen conclusion. The old completed VGGT ablation is pre-meta-filter history, so WAN comparisons should use the clean-GT layer `16` default and layer `24` comparison point. WAN should be read as a **video-context representation probe**, not a pair-only image backbone probe.

Fixed setting:

- adapter data: `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
- train / val split: `317 / 12`
- GT: SCRREAM registered mesh-complete targets, union-frustum cropped, first input camera frame, `20000` points
- adapter / decoder / validation: same MLP-L4, NOVA `scene_ae`, `nova_flow`, and robust validation path as the VGGT layer ablation
- WAN feature cache: 81-frame RGB windows around each pair, empty prompt, model `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- ablation grid: timesteps `249,499,749` x code layers `9,14,19,24,29`

Implementation entrypoints:

- dependency pins: `experiments/probe3d/requirements-wan-t2v.txt`
- feature cache: `experiments/probe3d/scripts/prepare_scrream_wan_t2v_feature_cache.py`
- training: `experiments/probe3d/train_wan_t2v_nova_adapter.py`
- Slurm: `slurm/scrream_wan_t2v_download.sbatch`, `slurm/scrream_wan_t2v_precompute.sbatch`, `slurm/scrream_wan_t2v_ablation_pack_train.sbatch`, `slurm/scrream_wan_t2v_multilayer_pack_train.sbatch`, `slurm/scrream_wan_t2v_multitime_pack_train.sbatch`

Status on `2026-05-19 14:14 CST`: the WAN checkpoint is downloaded under `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers` (~27 GB). Full feature precompute completed as jobs `86292`, `86293`, and `86294`, writing `4937` `.pt` files / about `45G` under `experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81`. The full WAN audit through hidden CA, Route2.1, pair_tiled81, latent tensors, grid readouts, multi-source fusion, FLF2V hidden, gated CA, and L4 capacity has completed. Final summary is in `docs/probe/wan_summary_2026-05-19.md`.

WAN Route2 and readout results:

- old best WAN hidden baseline: `t499/layer09`, `best_val_fscore_tau_0.10=0.46988987902779306`, `best_val_pred_to_gt_p90=0.48718947172164917`, `best_val_chamfer_l2=0.21040735269586244`
- historical best WAN interface: `t499/layer14 + wan_cross_attn_resampler`, `best_val_fscore_tau_0.10=0.5012956284974035`, `best_val_pred_to_gt_p90=0.4229188362757365`, `best_val_chamfer_l2=0.10908368105689685`
- CA-resampler stability / full-grid checks: low-LR `t499/layer14` rerun `86453` was negative (`F@0.10=0.4292316005720653`); full-grid job `86468` completed all `249/499/749 x 9/14/19/24/29` settings and found the best repeated F-score at `t249/layer14` (`0.48913902331806663`) and the best repeated p90 / Chamfer at `t249/layer09` (`0.410525918006897` / `0.11981592203179996`)
- clean-GT VGGT layer `16`: `best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`, `best_val_chamfer_l2=0.026904070439438026`
- interpretation: WAN Route2 is live but setting-limited; learned WAN-to-NOVA token resampling helps materially, but WAN is still not competitive with clean-GT VGGT
- Route2.1 `no_noise` / `low_noise` audit completed for layers `9,14,29`; best clean/low-noise result (`no_noise/layer29`, `F@0.10=0.44840173`) did not beat old Route2 best `t499/layer09`
- feature-normalization follow-up: `t499/layer09 + token_layernorm` completed as job `86395`; it improved Chamfer slightly but did not beat the baseline F-score (`F@0.10=0.45476213575933216`)
- pair-context follow-up: `pair_tiled81` completed as jobs `86396 -> 86397 -> 86400 -> 86401`; it improved Chamfer but worsened F-score / pred-to-GT precision (`F@0.10=0.4429200036613287`)
- pred-x0 follow-up: `pred_x0_latent` / denoised WAN latent feature, using `x0 = sample - sigma * model_output`; full cache job `86411` completed `329/329` tensors of shape `[12480,16]`
- original MLP-L4 pred-x0 readout is not a completed result: `86412` failed on a PyTorch CUDA adaptive-pooling backward assert from pooling the large `[12480,128]` token sequence
- replacement readout: `--adapter_type conv2d_mlp` reshapes `[B,12480,16]` to `[B,2,60,104,16]`, Conv2d stride-2 reads out `[B,3120,128]`, then pools to NOVA scene tokens `[B,768,128]`; smoke job `86421` and formal job `86422` completed, but the formal result (`F@0.10=0.41336424426176693`, `pred_to_gt_p90=0.6272750149170557`) did not beat old WAN hidden
- structured hidden readout follow-up: `--adapter_type grid2d_pool` and `--adapter_type grid2d_conv` keep the `[2,30,52]` hidden grid before pooling to `[2,24,16]`; formal jobs `86424` and `86426` completed with `F@0.10=0.43326753863230877` and `0.39855373112050535`, so simple grid pooling/conv does not close the WAN gap
- learned hidden readout follow-up: `--adapter_type wan_cross_attn_resampler` keeps WAN hidden as `[2,30,52]`, adds temporal/row/column position embeddings, and maps to `768` NOVA query tokens by cross-attention; layer sweep `86429` found the historical best WAN at `t499/layer14`, but exact repeats are unstable
- full-grid interpretation: `t249/layer09` and `t249/layer14` are the most reliable CA-resampler settings from job `86468`; `layer24/29` remain weak and should not be prioritized as single-layer probes
- multi-source learned readout follow-up: `--adapter_type wan_cross_attn_multilayer_resampler` with `--wan_layers` and `--adapter_type wan_cross_attn_multitime_resampler` with `--wan_timesteps` are implemented in `train_wan_t2v_nova_adapter.py`. Multi-layer `t249 layers 9+14` reached `F@0.10=0.46487238144059545`, `t249 layers 9+14+19` reached `0.4724058254433277`, and multi-timestep `249+499+749/layer14` reached `0.45213117002164466`; none beat the single-layer full-grid baselines, so simple hidden fusion is stopped for now
- model-output latent follow-up: `--feature_kind model_output_latent` / `--wan_feature_kind model_output_latent` caches raw WAN transformer `output.sample` latent slices with expected shape `[12480,16]`. Conv2d formal job `86536` reached `F@0.10=0.4199299775478907`; learned latent-CA formal job `86540` reached `0.4289946537162989`. Both were negative versus hidden CA baselines.
- hidden CA stability/capacity follow-up: seed23 standard CA reached `0.48706357506233194` at `t249/layer14` and `0.4591392560862819` at `t499/layer14`; gated CA reached `0.4505522726705196` at `t249/layer14` and `0.4866296484297818` at `t499/layer14`; L4 CA reached `0.4919946248489035` at `t249/layer14` and `0.46280321302050603` at `t499/layer14`. L4 `t249/layer14` is a small repeated-F improvement over full-grid `t249/layer14`, but gated CA and `t499` repeats do not justify expansion.

### 3. WAN2.1 FLF2V Route

FLF2V is a separate high-cost branch. Do not describe it as T2V `ctx81`: its window mode is **`pair_endpoint81`**, where slot `0` is the first SCRREAM pair frame, slot `80` is the second pair frame, and slots `1..79` linearly sample frame IDs between them. The extracted pair tokens are the endpoint temporal hidden slices `(0, 20)`, aligned with the first/last-frame conditioning.

First setting:

- model: `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers`
- local checkpoint target: `checkpoints/wan2.1/Wan2.1-FLF2V-14B-720P-diffusers`
- cache root: `experiments/probe3d/feature_cache/scrream_wan_flf2v14b_pair_endpoint81_480`
- feature cache: `experiments/probe3d/scripts/prepare_scrream_wan_flf2v_feature_cache.py`
- Slurm: `slurm/scrream_wan_flf2v_download.sbatch`, `slurm/scrream_wan_flf2v_precompute.sbatch`, `slurm/scrream_wan_flf2v_train.sbatch`
- first formal probe: `t249/layer14`, empty prompt, `--adapter_type wan_cross_attn_resampler`, same clean SCRREAM `.pt`, NOVA `scene_ae`, `nova_flow`, robust validation, 50 epochs
- expected 480P pair feature shape: `[3120,5120]`; if 480P is rejected and the run falls back to 720P, expected shape is `[7200,5120]`

Implementation status on `2026-05-19`: local static checks passed, `WanHiddenCrossAttentionResamplerAdapter` accepts metadata-derived hidden grids and `input_dim=5120`, and local window-only smoke validated all `329` SCRREAM samples. Download/preflight retry `86500`, feature-cache shard jobs `86515` / `86516` / `86517` / `86518`, smoke train `86519`, and formal train `86520` all completed. The cache has `329/329` files under `t249/layer14`, shape `[3120,5120]`. Formal FLF2V result is `F@0.10=0.42897984457682026`, `pred_to_gt_p90=0.5051772321263949`, and `Chamfer=0.14052311765650907`; it is weaker than T2V CA baselines, so do not expand FLF2V hidden broadly yet.

### 4. ScanNet v2 line

The ScanNet v2 mesh-first extension remains the current diagnostic baseline.

Important interpretation note:

- this is a **NOVA3R-style extension / transfer probe** on ScanNet v2
- it is **not** a literal reproduction of the official NOVA3R scene-training recipe, which is described around `3D-FRONT + ScanNet++V2`

The training structure remains aligned with the proposal direction:

- frozen visual backbone
- lightweight adapter
- NOVA3R-style decoder / flow-matching training path
- reliable **complete-GT style** supervision

### 5. InteriorGS line

InteriorGS remains a plausible future data-quality migration path, but it is no longer the immediate next step. The current priority is to inspect the completed SCRREAM full mesh-complete 20k / 500k MLP baseline before choosing the next data-quality migration or adapter branch.

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

## Current ScanNet probe result

The long formal 50-epoch MLP run is no longer the most informative immediate path. A short probe harness under `experiments/probe3d/probe_trials/` isolated the main failure mode more quickly.

Current best numeric run:

- target: `anchor_frustum`
- adapter: `MLP-L4, hidden=1024`
- objective: direct sampled rollout Chamfer, `loss_type=chamfer_sample`
- schedule: `lr=5e-5` to step 2000, then `lr=1e-5` refinement to step 2500
- best validation CD: `0.08745259`
- output dir: `experiments/probe3d/result/probe_trials/p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step2500`

Important interpretation:

- switching from `nova_flow` to direct rollout Chamfer is what moved CD down from roughly `0.30–0.35` to `<0.1`
- this mostly fixes the training/evaluation objective mismatch
- the visual result is still not good enough: the prediction covers the GT but contains many loose / thick-shell / outlier points
- the problem has shifted from **recall / target reachability** to **precision / sharpness**


### Paper-aligned NOVA3R reset

After user review, the active plan is to align the ScanNet target/loss more closely with NOVA3R: complete / amodal points inside the selected input-view frustum, FPS-style target sampling through `src_complete_fps_*`, and native flow matching as the primary loss. The new phase-2 config is:

- `experiments/probe3d/probe_trials/configs/phase2_nova_aligned.json`

## Research direction right now

The practical near-term plan is:

1. keep the old `eval_scrream` correction in mind and do not reuse those invalid claims
2. use clean-GT VGGT1 MLP layer `16` 50ep as the current default representation, with VGGT1 CA layer `20` 50ep as the closest readout comparison
3. treat WAN as paused after `docs/probe/wan_summary_2026-05-19.md`; optional WAN follow-ups are appendix-only
4. stop treating another Omega sweep as the main line: dense MLP, dense CA, and register-only CA are all below VGGT1 under the NOVA/FM probe
5. run a NOVA/FM probe-validity audit with a decoder-free SCRREAM spatial readout, so we can distinguish representation quality from compatibility with the fixed generator condition-token interface
6. expand SCRREAM training sample scale beyond the current 329 official pairs after the next-backbone pilot is defined
7. keep the fixed-30 ScanNet metrics as a failure-mode baseline

### VGGT-Omega probe setup

VGGT-Omega is the next backbone probe after WAN. The source lives at `third_party/vggt-omega`, and `experiments/probe3d/train_vggt_nova_adapter.py` supports `--backbone vggt_omega` for dense/full-token comparison against clean-GT VGGT. Use the official 512 checkpoint path `checkpoints/vggt_omega/vggt_omega_1b_512.pt`; the file currently named `checkpoints/vggt_omega/model.pt` was verified on `2026-05-19` to be byte-identical to the old VGGT checkpoint and must not be used as a VGGT-Omega result source.

Omega formal runs now cover dense/full-token MLP, dense/full-token CA, and paper-faithful register-only / "Frozen Scene Tokens" CA for layers `12,16,20,24`. Best Omega result is dense CA layer `16`: `F@0.10=0.6576072630447046`, `p90=0.23843258867661157`, `Chamfer=0.033605430430422224`. Dense MLP layer `16` is close but lower (`F@0.10=0.6550556579677611`), and register-only CA is much weaker (best register layer `16`, `F@0.10=0.4930976482166729`). Since all Omega branches are below VGGT1 under the NOVA/FM decoder probe, the current risk is no longer just an Omega readout choice; it is whether this fixed generator interface is a valid cross-backbone representation evaluator.

Next experiment should therefore be a probe-validity audit: train a decoder-free direct SCRREAM spatial readout on the same frozen VGGT1 and Omega features. If Omega beats VGGT1 there but loses through NOVA/FM, the project should frame the current metric as generator-compatibility rather than pure spatial representation quality.

## Documentation map

- `AGENTS.md` — project-level instructions for future coding agents
- `PROJECT.md` — current project-level status and next steps
- `docs/probe/README.md` — probe-doc entry point
- `docs/probe/handoff_2026-05-07.md` — current SCRREAM data / Slurm / submodule handoff
- `docs/probe/handoff_2026-05-03.md` — historical machine handoff for the first SCRREAM mesh-complete prep chain
- `docs/probe/scannet_mesh_first_plan.md` — current ScanNet formal plan
- `docs/probe/experiment_history.md` — what really happened, including corrections
- `docs/probe/experiment_plan.md` — phased execution plan from here
- `docs/probe/interiorgs_training_plan.md` — deferred high-quality dataset migration plan
- `docs/probe/todo.md` — current actionable task list
- `slurm/` — Slurm job scripts; logs should go to `slurm_out/`

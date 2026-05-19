# Probe Docs

## Current canonical status — 2026-05-19

For the latest project state, read:

- `handoff_2026-05-07.md`
- `wan_summary_2026-05-19.md`
- `../../experiments/probe3d/README.md`
- `experiment_plan.md`

Key corrections:

- `scannet_max_interval=1` is now the intended setting because ScanNet preprocessing already uses `frame_skip=20`. Old `max_interval=30` K-view conclusions are invalid/confounded.
- CD-only and two-sample oracle results are not claim-level evidence. Use fixed-sample robust metrics and visual audits.
- The current MLP baseline is mostly a failure-mode baseline: recall is moderate, precision/sharpness are poor.
- The old local `eval_scrream` branch is invalid for claims, but full SCRREAM is now downloaded at `~/datasets/SCRREAM`.
- The active baseline branch is SCRREAM full-data mesh-complete VGGT adapter training on sequence-meta-filtered clean GT.
- The WAN2.1 T2V Route2 branch is paused after the `2026-05-19` audit; use `wan_summary_2026-05-19.md` as the final evidence trail. WAN kept the VGGT ablation's data/GT/split/decoder/validation fixed and changed only the representation/readout. The learned hidden CA resampler was the only clearly positive WAN hidden readout: historical best `t499/layer14` reached `F@0.10=0.5012956284974035`, `pred_to_gt_p90=0.4229188362757365`, and `Chamfer=0.10908368105689685`; repeated settings are closer to `0.49`, and L4 `t249/layer14` reached `0.4919946248489035`. WAN is still below clean-GT VGGT layer `16`, so the next active model-coverage work should move to a new backbone rather than another broad WAN sweep.
- Long data generation and training should use `slurm/` scripts with logs in `slurm_out/`.
- Slurm job `86140` completed the 20k / 500k trainplus-test MLP baseline on `air-node-02` with exit `0:0`; `final_metrics.json` reports `best_val_chamfer_l2=0.5149603486061096`.
- Slurm job `86149` completed the robust VGGT layer ablation on `2026-05-08`, but it used the pre-meta-filter GT and is now historical. Clean-GT VGGT ablation job `86286` completed successfully; the current default is layer `16`, with layer `24` as the main comparison point.
- NOVA `scene_n1`, `scene_n2`, `scene_ae`, and VGGT weights are staged under `checkpoints/`; SwanLab is installed in `nova3r`.
- VGGT, Wan2.1, and VidFM3D are Git submodules under `third_party/`; initialize them with `git submodule update --init --recursive`.
- WAN Route2 uses proxy `http://127.0.0.1:17890` through compute-node SSH tunnel logic in `slurm/scrream_wan_t2v_*.sbatch`; non-WAN jobs keep the `7896` proxy default.

This folder records the proposal-facing execution state for the current adapter / decoder experiments.

## Read this first

### 1. What is active now?
The current active formal data/GT line is the **SCRREAM full-data mesh-complete adapter line**.

It is:
- a corrected rerun of the SCRREAM adapter idea using the full dataset at `~/datasets/SCRREAM`
- based on sequence-meta-filtered registered SCRREAM scene meshes, not the invalid `eval_scrream` subset
- constrained to the selected two-input-view union frustum
- launched through Slurm scripts in `slurm/`

Full `.pt` generation is complete for both 10k and 20k target variants. The current formal data file is `scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`, which has `train=317`, `val=12`, and `mesh_sequence_meta_filter=True`. The previous 20k / 500k MLP baseline and VGGT layer ablation were run before the sequence-level object filter; use them as historical diagnostics, not final clean-GT claims.

The clean-GT VGGT layer ablation completed as Slurm job `86286`. The current default VGGT layer is `16` (`best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`); layer `24` is the closest comparison point.

WAN is no longer the active follow-up. The branch remains implemented and documented, but broad expansion is paused. Optional future WAN appendix checks are limited to L4 hidden CA around `t249/layer09/14`; the main next step is selecting and probing the next backbone under the same clean SCRREAM / NOVA protocol.

WAN Route2 entrypoints:

- dependency pins: `../../experiments/probe3d/requirements-wan-t2v.txt`
- feature cache: `../../experiments/probe3d/scripts/prepare_scrream_wan_t2v_feature_cache.py`
- training: `../../experiments/probe3d/train_wan_t2v_nova_adapter.py`
- Slurm: `../../slurm/scrream_wan_t2v_download.sbatch`, `../../slurm/scrream_wan_t2v_precompute.sbatch`, `../../slurm/scrream_wan_t2v_ablation_pack_train.sbatch`, `../../slurm/scrream_wan_t2v_route21_pack_train.sbatch`, `../../slurm/scrream_wan_t2v_multilayer_pack_train.sbatch`, `../../slurm/scrream_wan_t2v_multitime_pack_train.sbatch`

Status on `2026-05-19 13:13 CST`: the WAN checkpoint is present under `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers`. Full WAN feature precompute completed as jobs `86292`, `86293`, and `86294`, writing `4937` `.pt` files / about `45G`. Training pack job `86307` completed on `air-node-04`, exit `0:0`, elapsed `13:36:04`. Learned CA-resampler layer sweep job `86429`, low-LR rerun job `86453`, full-grid job `86468`, multi-layer jobs `86485` / `86487`, multi-timestep job `86494`, FLF2V job `86520`, latent tensor-choice jobs `86536` / `86540` / `86543`, and hidden CA stability/capacity jobs `86547` / `86548` / `86549` / `86550` / `86553` / `86554` have completed.

Route2 result:

- old best WAN hidden baseline: `t499/layer09`, `best_val_fscore_tau_0.10=0.46988987902779306`, `best_val_pred_to_gt_p90=0.48718947172164917`, `best_val_chamfer_l2=0.21040735269586244`
- historical best WAN interface: `t499/layer14 + wan_cross_attn_resampler`, `best_val_fscore_tau_0.10=0.5012956284974035`, `best_val_pred_to_gt_p90=0.4229188362757365`, `best_val_chamfer_l2=0.10908368105689685`; layer sweep F@0.10 was layer09 `0.46751068`, layer14 `0.50129563`, layer19 `0.45041048`, layer24 `0.40883680`, layer29 `0.38334162`
- CA-resampler repeat/full-grid checks: `86453` low-LR `t499/layer14` was negative (`F@0.10=0.4292316005720653`); `86468` full grid completed with best repeated F at `t249/layer14` (`0.48913902331806663`) and best repeated p90 / Chamfer at `t249/layer09` (`0.410525918006897` / `0.11981592203179996`)
- clean-GT VGGT layer `16`: `best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`, `best_val_chamfer_l2=0.026904070439438026`
- Route2.1 result: layers `9,14,29` with `no_noise` and `low_noise` completed through `86342/86343 -> 86350/86351 -> 86357 -> 86371/86372`; best clean/low-noise run did not beat old Route2 `t499/layer09`
- normalization follow-up: `t499/layer09 + token_layernorm` completed as job `86395`; it did not beat old Route2 best (`F@0.10=0.45476213575933216`)
- pair-context follow-up: `pair_tiled81` completed as jobs `86396 -> 86397 -> 86400 -> 86401`; it did not beat old Route2 best (`F@0.10=0.4429200036613287`)
- pred-x0 follow-up: `pred_x0_latent` uses `x0 = sample - sigma * model_output`; full cache job `86411` wrote `329/329` files under `t499/pred_x0_latent/` with feature shape `[12480,16]`
- original pred-x0 MLP-L4 readout failed on CUDA adaptive-pooling backward in job `86412`; it should not be treated as an experiment result
- replacement readout: `--adapter_type conv2d_mlp` restores `[B,2,60,104,16]`, applies a stride-2 Conv2d readout to `[B,3120,128]`, and pools to `[B,768,128]`; smoke job `86421` passed and formal job `86422` completed with `F@0.10=0.41336424426176693`, `pred_to_gt_p90=0.6272750149170557`, `Chamfer-L2=0.35522504647572833`, below old WAN hidden
- hidden-grid readouts: `--adapter_type grid2d_pool` and `--adapter_type grid2d_conv` preserve `[2,30,52]` before mapping to `[2,24,16]`; formal jobs `86424` and `86426` completed with F@0.10 `0.43326753863230877` and `0.39855373112050535`, so fixed grid pooling/conv did not close the WAN gap
- learned hidden readout: `--adapter_type wan_cross_attn_resampler` restores `[B,3120,1536]` as `[B,2,30,52,1536]`, adds temporal/row/column position embeddings, and cross-attends `768` learned NOVA query tokens to WAN tokens; it is the only positive WAN readout branch so far, but exact repeats are noisy
- multi-source hidden readout: `--adapter_type wan_cross_attn_multilayer_resampler` with `--wan_layers` and `--adapter_type wan_cross_attn_multitime_resampler` with `--wan_timesteps` are implemented. Multi-layer `t249 layers 9+14` reached `F@0.10=0.46487238144059545`; multi-layer `9+14+19` reached `0.4724058254433277`; same-layer multi-timestep `249+499+749/layer14` reached `0.45213117002164466`; all are below single-layer CA baselines
- model-output latent readout: `--feature_kind model_output_latent` and `--wan_feature_kind model_output_latent` cache raw WAN transformer `output.sample` latent slices as `[12480,16]`. Conv2d job `86536` reached F@0.10 `0.4199299775478907`; learned latent-CA job `86540` reached `0.4289946537162989`; pred-x0 learned latent-CA job `86543` reached `0.43766060222664477`. These are negative versus hidden CA.
- hidden CA stability/capacity: seed23 standard CA reached `0.48706357506233194` at `t249/layer14` and `0.4591392560862819` at `t499/layer14`; gated CA reached `0.4505522726705196` at `t249/layer14` and `0.4866296484297818` at `t499/layer14`; L4 CA reached `0.4919946248489035` at `t249/layer14` and `0.46280321302050603` at `t499/layer14`. L4 `t249/layer14` is the best recent repeated-F setting, but it does not beat the historical `t499/layer14` peak.

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
  - historical machine handoff for the first SCRREAM full mesh-complete prep chain

- `handoff_2026-05-07.md`
  - current SCRREAM data / Slurm / submodule handoff

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

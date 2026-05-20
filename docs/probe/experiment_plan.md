# Experiment plan from the current state

## Current plan override — 2026-05-19

Older phase labels below remain useful history, but the next valid plan is now:

1. **Use full SCRREAM, not `eval_scrream`.** The old local SCRREAM subset remains invalid for claims. The corrected branch uses `~/datasets/SCRREAM`.
2. **Use sequence-filtered mesh-complete SCRREAM targets first.** The current default GT source reads the current sequence `meta.txt`, samples only the listed registered scene meshes proportional to surface area, crops to the selected two-view union frustum, and stores targets in the first input camera frame.
3. **Launch through Slurm.** Data generation and training scripts live in `slurm/`; logs go to `slurm_out/`.
4. **Clean GT regeneration completed.** Slurm jobs `86283` and `86284` regenerated the 20k / 500k `trainplus_test` adapter data with `mesh_sequence_meta_filter=True`.
5. **Clean-GT VGGT reruns completed.** The previous job `86149` is pre-meta-filter history. Clean-GT job `86286` completed successfully and is now the 9510-step historical baseline. VGGT1 50ep jobs `86571` / `86572` completed; use VGGT1 MLP layer `16` 50ep as the current default and VGGT1 CA layer `20` 50ep as the main readout comparison.
6. **WAN Route2 completed and is setting-limited.** Keep SCRREAM data, mesh-complete GT, split, MLP adapter, NOVA decoder, and robust validation matched to the clean-GT VGGT ablation; replace only the representation with cached WAN2.1 T2V video-context features. Full feature precompute and 15-run training pack `86307` completed. Best WAN (`t499/layer09`) is above zero/sample-shuffle controls but still far weaker than clean-GT VGGT.
7. **WAN Route2.1 clean / low-noise, normalization, and pair_tiled81 audits completed.** None of these beat old Route2 best `t499/layer09`; pair_tiled81 improved Chamfer but worsened F-score / pred-to-GT precision. Treat these as evidence that simple noise-level, scale, or context-dilution explanations are insufficient.
8. **WAN Route2 is paused after the 2026-05-19 audit.** VidFM3D remains useful as an extraction/probe reference, but the project goal is not to replace the NOVA/FM generator with a dense pointmap probe. The learned CA-resampler branch improved WAN and has a historical peak at `t499/layer14` (`F@0.10=0.5012956284974035`), but repeated settings are around `0.49`, L4 hidden CA is only a small positive, and all tested noise/context/latent/fusion/FLF2V/gated variants remain below clean-GT VGGT. Use `wan_summary_2026-05-19.md` as the frozen WAN conclusion.
9. **VGGT-Omega is wired, but the NOVA/FM probe ranking is now under audit.** Omega dense/full-token MLP, dense/full-token CA, and register-only / "Frozen Scene Tokens" CA completed for layers `12,16,20,24`. Best Omega is dense CA layer16 (`F@0.10=0.6576072630447046`), below VGGT1 MLP layer16 (`0.702544731989172`) and VGGT1 CA layer20 (`0.7014525243138799`). Do not treat this as a final claim that Omega has worse spatial representation; the next branch is a decoder-free SCRREAM direct spatial readout to test whether the fixed NOVA/FM generator interface is distorting cross-backbone representation ranking.
10. **Keep ScanNet as a diagnostic baseline.** All new ScanNet K-view trials must set `scannet_max_interval=1` unless the experiment explicitly studies wider baselines. Compare ScanNet checkpoints with fixed robust metrics before claims.
11. **Defer InteriorGS.** InteriorGS remains a plausible data-quality migration path, but it is not the immediate next branch.

This plan is intentionally short and tied to what is already real in the repo.

## Phase 0 — Corrections already absorbed

### SCRREAM correction
- older local `eval_scrream` runs are invalid for formal claims
- keep them only as engineering/debug history
- do not use them as feasibility evidence

### ScanNet path correction
- do not describe the ScanNet v2 line as official NOVA3R recipe reproduction
- describe it as a **NOVA3R-style extension / transfer probe**

## Phase 1 — Data and training infrastructure

### Completed
- full ScanNet v2 mesh-first preprocess
- formal `train / val / test` split
- scene-level `500k` mesh reservoirs
- true 4-view input wiring
- `pts3d_complete` dataset path
- DDP / `torchrun` conversion for MLP / CA / SA
- complete-GT smoke validation
- full-root DDP preflight validation

## Phase 2 — Short ScanNet probe baseline

### Goal
Quickly identify whether the failure comes from target definition, adapter capacity, or objective mismatch.

### Current best baseline
- dataset root:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17`
- target mode:
  - `anchor_frustum`
- adapter:
  - `MLP-L4, hidden=1024`
- loss:
  - direct sampled rollout Chamfer (`loss_type=chamfer_sample`)
- best schedule:
  - `lr=5e-5` to step2000
  - `lr=1e-5` refinement to step2500
- best validation CD:
  - `0.08745259`

### Interpretation
This is a real numeric improvement, but not a visually clean reconstruction. The model has acceptable GT coverage but poor prediction precision / outlier control.

## Phase 3 — Paper-aligned NOVA3R target / loss reset

Jiacheng clarified the key methodological point: NOVA3R itself trains on complete / amodal point clouds **inside the selected input-view frustum**, not the entire room. The current `anchor_frustum` result was useful because it partially matched that idea, but for K-view training the target should be the union of the selected input frusta, not only the first view.

Immediate implementation plan:

1. add explicit target aliases:
   - `nova_input_frustum`: mesh surface points inside the union of selected input-view frusta
   - `nova_anchor_frustum`: first-view-only debug / K=1 equivalent
2. use NOVA-native target sampling through `src_complete_fps_*`, starting with `src_complete_fps_4096`
3. run paper-aligned K=1 and K=2 oracle sanity checks:
   - `p2_oracle_nova_input_frustum_k1_fps4096_s2_step400`
   - `p2_oracle_nova_input_frustum_k2_fps4096_s2_step400`
4. only after oracle support is plausible, run native-flow adapter probes:
   - `p2_adapter_nova_input_frustum_k1_mlp_l4_flow_fps4096_step1000`
   - `p2_adapter_nova_input_frustum_k2_mlp_l4_flow_fps4096_step1000`
5. keep direct Chamfer as a diagnostic / metric-chasing upper check, not the main claim

## Phase 3b — Precision-aware metrics after native-flow baseline

After the paper-aligned native-flow baseline is established, add precision-aware metrics/losses only as diagnostics or small auxiliaries:

1. log one-way distances:
   - `pred→GT` for precision / outlier control
   - `GT→pred` for completeness / recall
2. use GT-vs-pred videos for qualitative checks
3. consider weighted / trimmed Chamfer only if it does not replace the NOVA-native flow objective as the primary training signal

## Phase 4 — Adapter comparison on the same objective

After the paper-aligned native-flow MLP baseline is meaningful:

1. launch CA on the same processed data / target / objective
2. launch SA on the same processed data / target / objective
3. compare them only after all branches have comparable runs and visualizations

## Phase 5 — InteriorGS high-quality data pilot

InteriorGS-style high-quality indoor 3DGS data is deferred until after the completed corrected SCRREAM full-data baseline is inspected and understood.

Immediate steps:

1. download or stage a small InteriorGS subset on the server;
2. inspect scene assets (`3dgs_compressed.ply`, `labels.json`, occupancy files,
   and `structure.json`);
3. document coordinate transforms and metric units;
4. export a small target-point/render sanity set in the current probe convention;
5. create a fixed tiny train/val/test split and only then launch a training smoke.

See `interiorgs_training_plan.md`.

## Phase 6 — Corrected SCRREAM full-data branch

Current status:

1. full SCRREAM data is available locally at `~/datasets/SCRREAM`
2. data bridge exists at `experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py`
3. loader honors `--dataset scrream_adapter --data_root <adapter.pt>`
4. mesh-complete preview for `scene09` has been visually accepted
5. Slurm scripts exist for data preparation and MLP adapter training
6. checkpoints are staged under `checkpoints/` for NOVA `scene_n1`, `scene_n2`, `scene_ae`, and VGGT
7. SwanLab is installed in `nova3r`, and the full MLP Slurm script enables it by default
8. 10k and 20k adapter `.pt` datasets are generated
9. job `86140` completed the 20k / 500k trainplus-test MLP baseline

GT construction:

1. read official two-view pairs from `data/scrream/scrream_n2_list.json`
2. use the two RGB frames as adapter inputs
3. read the current sequence `meta.txt` and select only listed registered `sceneXX/meshes/*.obj` surfaces
4. sample selected mesh surfaces proportional to surface area
5. voxel-deduplicate and cache a per-sequence/object-set mesh reservoir
6. crop points to the union frustum of the two input views
7. transform the target to the first input camera coordinate frame
8. FPS sample or pad to the requested target count

Completed baseline:

- job: `86140` / `scrream_mesh_mlp`
- state: `COMPLETED`, exit `0:0`
- node: `air-node-02`
- elapsed: `00:26:26`
- Slurm end time: `2026-05-07 22:40:22 CST`
- output: `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17`
- SwanLab: `https://swanlab.cn/@JiachengDong/PSUVPSC3DD/runs/eoiupi3bv2g11dvd21ypm`
- `final_metrics.json`: `first_loss=1.4599288702011108`, `final_loss=0.8398033976554871`, `best_loss=0.5998285412788391`, `best_val_chamfer_l2=0.5149603486061096`
- latest `validation_metrics.json`: step `9500`, `val_chamfer_l2=0.6779176592826843`

Inspect `best.pth`, `latest.pth`, and the exported PLYs before making any adapter claim from this run.

Smoke variants can override script defaults through environment variables:

```bash
SCRREAM_ADAPTER_OUT=experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_smoke8.pt \
SCRREAM_EXTRA_ARGS="--max_samples 8 --save_preview_dir experiments/probe3d/adapter_data/scrream_mesh_complete_smoke8_preview" \
sbatch slurm/scrream_mesh_complete_prepare.sbatch

SCRREAM_ADAPTER_DATA=experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_smoke8.pt \
SCRREAM_MAX_STEPS=100 \
SCRREAM_OUTPUT_DIR=experiments/probe3d/result/scrream_mesh_complete_smoke8_mlp \
sbatch slurm/scrream_mesh_complete_mlp_train.sbatch
```

Future follow-up runs should use the multi-GPU Slurm path when idle GPUs are available:

```bash
SCRREAM_GPUS_PER_NODE=4 \
SCRREAM_EPOCHS=50 \
SCRREAM_ADAPTER_DATA=experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt \
SCRREAM_OUTPUT_DIR=experiments/probe3d/result/<new-run-name> \
SCRREAM_FEATURE_CACHE_DIR=experiments/probe3d/feature_cache/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_vggt23 \
SCRREAM_NUM_QUERIES=20000 \
sbatch --qos=high --nodelist=<node-with-free-a100s> --gres=gpu:a100:4 --mem=96G \
  slurm/scrream_mesh_complete_mlp_train.sbatch
```

## Phase 6b — WAN2.1 T2V Route2 representation probe

Goal: compare WAN2.1 T2V video-context features against the clean-GT VGGT rerun without changing the training problem. The completed job `86149` is only pre-meta-filter history; the active VGGT comparison is clean-GT layer `16` plus layer `24`.

Fixed SCRREAM setting:

1. adapter data: `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
2. split: `train=317`, `val=12`
3. GT: sequence-meta-filtered registered SCRREAM mesh-complete surface points, `target_points=20000`, `mesh_sample_points=500000`, union-frustum crop, first input camera frame
4. adapter: `MLP-L4`, hidden dim `1024`
5. loss: `nova_flow`
6. decoder: NOVA `scene_ae`
7. validation: robust metrics over full val split, `val_visual_40960/` previews

WAN representation:

1. model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
2. prompt: empty string
3. context: 81 RGB frames centered/clamped around the SCRREAM pair
4. timesteps: `249,499,749`
5. code layers: `9,14,19,24,29`
6. cached feature shape: `[3120,1536]` per sample / timestep / layer

Execution plan:

1. checkpoint download/preflight is complete under `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers`
2. 2-sample feature smoke is complete for `WAN_TIMESTEPS=749 WAN_LAYERS=20 WAN_MAX_SAMPLES=2`
3. full 15-grid feature precompute completed through `slurm/scrream_wan_t2v_precompute.sbatch` as jobs `86292`, `86293`, and `86294`
4. training pack attempt `86306` failed immediately on scalar final-loss reporting; `train_wan_t2v_nova_adapter.py` was fixed
5. replacement 15-grid training pack `86307` completed through `slurm/scrream_wan_t2v_ablation_pack_train.sbatch`, exit `0:0`, elapsed `13:36:04`
6. best WAN Route2 result is `t499/layer09`: `best_val_fscore_tau_0.10=0.46988987902779306`, `best_val_pred_to_gt_p90=0.48718947172164917`, `best_val_chamfer_l2=0.21040735269586244`
7. current clean-GT VGGT1 MLP layer `16` 50ep remains much stronger: `best_val_fscore_tau_0.10=0.702544731989172`, `best_val_pred_to_gt_p90=0.19675995161135992`, `best_val_chamfer_l2=0.027118226668486994`; the older 9510-step VGGT layer16 reference was `F@0.10=0.6860468604251301`
8. current WAN evidence is frozen in `docs/probe/wan_summary_2026-05-19.md`: learned CA resampler beats old hidden MLP in historical peak and repeated `t249` settings, but simple multi-layer hidden fusion, same-layer multi-timestep fusion, FLF2V hidden, latent tensor-choice readouts, and gated CA did not beat single-layer hidden CA. L4 `t249/layer14` is the only recent small positive, at F@0.10 `0.4919946248489035`.

WAN repo/checkpoint network jobs use proxy `http://127.0.0.1:17890` through the compute-node SSH tunnel logic in `slurm/scrream_wan_t2v_*.sbatch`. The checkpoint, 2-sample feature smoke, window validation, and full cache generation have completed.

### Phase 6c — WAN Route2.1 setting audit

Goal: test whether current Route2 is weak because it probes noisy T2V denoising hidden states rather than clean visual-condition features, then check whether feature scale / normalization is masking useful WAN signal.

Immediate settings:

1. implemented feature modes:
   - `no_noise`: encode the 81-frame video with the WAN VAE and run the transformer on the clean latent while still passing the low-noise timestep embedding at scheduler index `999`;
   - `low_noise`: add scheduler noise at index `999` and use the matching low-noise timestep embedding;
2. cache metadata records `noise_mode`, `requested_timestep_index`, `scheduler_timestep`, `latent_noise_applied`, and `low_noise_index`, because the current `249/499/749` values are scheduler indices rather than guaranteed raw noise levels;
3. layers: `9,14,29`;
4. execution completed: cache smokes `86342` / `86343`, full caches `86350` / `86351`, smoke train `86357`, and replacement 1-GPU formal packs `86371` / `86372`;
5. training: same SCRREAM `.pt`, same `train=317` / `val=12`, same MLP-L4, same NOVA decoder, full robust validation and `val_visual_40960/`;
6. normalization check completed: targeted `t499/layer09 + token_layernorm` ran as job `86395` against the existing best WAN route and did not beat the baseline F-score.

Deferred settings:

1. I2V / FLF2V conditioning;
2. broader adapter-capacity sweeps. Low-priority MLP capacity probes can test `MLP-L6-H1024` or `MLP-L4-H2048`; the current MLP adapter already uses `GELU`, so this is a capacity/readout check rather than the main suspected cause.

Completed follow-up audits:

1. `t499/layer09 + token_layernorm` completed as job `86395`: F@0.10 `0.45476213575933216`, pred-to-GT p90 `0.484821617603302`, Chamfer-L2 `0.1771892917652925`. Feature scale / token normalization helped Chamfer slightly but did not beat baseline F-score.
2. `pair_tiled81` versus `ctx81` completed as jobs `86396 -> 86397 -> 86400 -> 86401`: F@0.10 `0.4429200036613287`, pred-to-GT p90 `0.5104739194115003`, Chamfer-L2 `0.16478415516515574`. Replacing real 81-frame context with tiled pair frames did not improve the main precision/F-score metrics.

### Phase 6d — WAN generator-compatible representation search

Goal: preserve the NOVA/FM decoder as the probe generator while searching for WAN internal states that are easier to translate into NOVA condition tokens. Do not switch this project line to VidFM3D's dense pointmap probe; use VidFM3D only to audit WAN extraction details.

Baseline A is already complete and should not be rerun unless an exact reproducibility repeat is explicitly requested:

| ID | WAN representation | Adapter / decoder | Status | Key result |
| --- | --- | --- | --- | --- |
| A | current WAN block hidden, ctx81, normal `t499/layer09` | MLP-L4 -> NOVA/FM | completed | F@0.10 `0.46988987902779306`, pred-to-GT p90 `0.48718947172164917`, Chamfer-L2 `0.21040735269586244` |
| B / G | raw T2V transformer `model_output_latent`, ctx81, normal `t499` | Conv2d readout -> NOVA/FM | completed negative | `86536` F@0.10 `0.4199299775478907`, p90 `0.6285491685072581`, Chamfer `0.44791099180777866` |
| B2 / G2 | raw T2V transformer `model_output_latent`, ctx81, normal `t499` | latent-grid CA resampler -> NOVA/FM | completed negative | `86540` F@0.10 `0.4289946537162989`, p90 `0.5942087918519974`, Chamfer `0.19963246708114943` |
| C | predicted clean latent / x0 estimate from WAN denoising state | Conv2d readout -> NOVA/FM | completed negative | F@0.10 `0.41336424426176693`, pred-to-GT p90 `0.6272750149170557`, Chamfer-L2 `0.35522504647572833` |
| C2 | predicted clean latent / x0 estimate from WAN denoising state | latent-grid CA resampler -> NOVA/FM | completed negative | `86543` F@0.10 `0.43766060222664477`, p90 `0.5175856028993925`, Chamfer `0.20880577837427458` |
| D0 | current block hidden | fixed 2D pool / tiny 2D conv -> NOVA/FM | completed negative | grid2d_pool F@0.10 `0.43326753863230877`; grid2d_conv F@0.10 `0.39855373112050535` |
| D | current block hidden | learned cross-attention / Perceiver-style resampler -> NOVA/FM | positive but unstable | historical `t499/layer14` reaches F@0.10 `0.5012956284974035`; full-grid repeat favors `t249/layer14` / `t249/layer09`; seed/gated/L4 jobs `86547`-`86554` completed, with L4 `t249/layer14` reaching `0.4919946248489035` |
| E1 | multi-layer hidden fusion, layers `9+14` and `9+14+19` at `t249` | multi-source CA resampler -> NOVA/FM | completed negative | `9+14` F@0.10 `0.46487238144059545`; `9+14+19` F@0.10 `0.4724058254433277`; neither beats single-layer `t249/layer14` |
| E2 | same-layer multi-timestep hidden fusion, `249+499+749` at layer `14` | multi-source CA resampler -> NOVA/FM | completed negative | formal job `86494` F@0.10 `0.45213117002164466`, below single-layer CA |
| F | FLF2V first/last-frame hidden, `pair_endpoint81`, `t249/layer14` | CA resampler -> NOVA/FM | completed negative | formal job `86520` F@0.10 `0.42897984457682026`, below T2V CA |

Near-term policy after WAN pause:

1. Treat A as the completed baseline for this branch.
2. C is complete: `--feature_kind pred_x0_latent` / `--wan_feature_kind pred_x0_latent` produced full cache `86411`; original MLP readout failed on CUDA adaptive-pooling backward; Conv2d readout smoke `86421` and formal job `86422` completed, but the result did not beat A.
3. D0 is complete: fixed 2D hidden pooling and tiny 2D conv did not beat A.
4. D is complete for the first `t499` sweep, low-LR rerun, and full `249/499/749 x 9/14/19/24/29` grid.
5. E1 and E2 are complete and negative, so stop simple hidden fusion for now.
6. F is complete and negative for hidden tokens; do not expand FLF2V hidden broadly unless later tensor-choice evidence motivates it.
7. Stop B/G and C2 for now: model-output and pred-x0 latent learned-readout results are below hidden CA.
8. Pause D as the main line. Optional appendix-only checks are L4 `t249/layer09` and one L4 `t249/layer14` seed repeat. The active work has moved past the Omega readout audit to the NOVA/FM probe-validity audit below; do not launch another broad WAN sweep without explicit new evidence.

## Phase 6b — NOVA/FM probe-validity audit

Current issue: under the same SCRREAM/NOVA protocol, VGGT-Omega ranks below VGGT1 even after dense MLP, dense CA, and register-only CA readouts. Since Omega should plausibly have stronger scene representations, the project must now test whether the fixed NOVA/FM generator can be used as a cross-backbone representation evaluator.

First audit:

- add a decoder-free direct spatial readout from frozen features to SCRREAM mesh-complete targets
- keep the same `.pt`, train/val split, robust metrics, and `val_visual_40960` style outputs
- compare VGGT1 layer16 / CA-layer20 reference features against Omega dense layer16 / layer20 features
- use a controlled readout capacity, preferably a learned query cross-attention readout plus a small point head, with the same trainable budget across backbones

Decision rule:

- if Omega beats VGGT1 in the decoder-free readout but loses through NOVA/FM, the current project line measures compatibility with the NOVA condition-token manifold, not pure spatial representation quality
- if Omega also loses in the decoder-free readout, the local claim that Omega's selected frozen tokens are stronger for this SCRREAM geometry target is not established
- if a stronger but controlled NOVA bridge makes Omega catch VGGT1, the issue is the adapter/token contract rather than the generator itself

## Phase 7 — Proposal-facing interpretation

Only after the above:

- decide whether MLP remains only a baseline or becomes a stronger proposal anchor
- decide whether CA / SA materially strengthen the proposal
- update the proposal narrative with only the valid, corrected runs

## Phase 3c — Overnight K=2 feasibility sweep

Current tactical priority: produce a feasibility result rather than block on exact hidden NOVA3R GT construction.

### Fixed settings for the main overnight branch

- input views: `K=2`
- adapter: `MLP-L4-H1024`
- queries: `2048`
- main stable target: `anchor_frustum`
- training length: `1000` steps per trial
- validation: steps `500` and `1000`

### Loss ablation now running

1. `p4_k2_anchor_mlp_l4_flow_step1000`
2. `p4_k2_anchor_mlp_l4_hybrid005_step1000`
3. `p4_k2_anchor_mlp_l4_chamfer_step1000`

The purpose is to compare:

- generator-native flow matching
- flow matching with small rollout Chamfer auxiliary
- direct Chamfer as a diagnostic / upper-bound metric chaser

### GT-construction candidates queued after the loss ablation

Oracle sweep:

- `nova_input_frustum`
- `covered_by_ge2`
- `anchor_frustum_margin1.5`
- `nova_per_view_frustum_anchor_zpos`
- `nova_per_view_ldi2`
- `nova_per_view_ldi4`
- `nova_per_view_ldi8`

Adapter sweep:

- `covered_by_ge2`: flow + hybrid005
- `nova_input_frustum`: flow + hybrid005
- `anchor_frustum_margin1.5`: flow + hybrid005

### Morning checklist

1. Inspect `experiments/probe3d/probe_trials/results.tsv` rows beginning with `p4_`.
2. Check whether K=2 MLP-L4 improves over previous K=4 feasibility runs.
3. Compare oracle ceiling vs adapter result for each target.
4. Inspect generated PLYs for the best metric candidates.
5. Select one result as the feasibility proof and keep the rest as ablations / diagnostics.

### Step-count update

Per user request, overnight adapter trials were increased from `1000` to `2000` steps where possible. The first short native-flow K=2 / MLP-L4 run completed at step1000 with CD `0.91071419`; it is now only a short-run diagnostic. The active overnight suite uses:

- `p4_k2_anchor_mlp_l4_flow_resume_step2000`
- `p4_k2_anchor_mlp_l4_hybrid005_step2000`
- `p4_k2_anchor_mlp_l4_chamfer_step2000`
- `p4_k2_{covered_ge2,input_frustum,anchor_margin15}_mlp_l4_{flow,hybrid005}_step2000`

Driver log:

- `experiments/probe3d/result/probe_trials/p4_overnight_k2_mlp_l4_long_driver.out`

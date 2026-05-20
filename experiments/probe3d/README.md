# NOVA3R 3D Probe

## Current status override — 2026-05-19

The active probe state has moved to the corrected full SCRREAM branch. Important current constraints:

- use corrected `scannet_max_interval=1` for the frame_skip=20 ScanNet data;
- treat old `max_interval=30` runs as confounded;
- do not interpret single symmetric CD or two-sample oracle averages as claim-level evidence;
- use fixed-sample robust eval / videos before launching new adapter claims;
- treat old `eval_scrream` runs as invalid for claims because they used the released eval subset;
- use the downloaded full SCRREAM tree at `~/datasets/SCRREAM` for the corrected branch;
- default SCRREAM full-data target source is now sequence-filtered registered mesh-complete, not dense depth aggregation;
- submit long data-prep and training work through `slurm/`; Slurm logs go to `slurm_out/`.
- job `86140` completed the pre-meta-filter 20k / 500k trainplus-test MLP baseline on `air-node-02` with exit `0:0`; treat it as historical after the sequence-meta GT correction.
- job `86286` completed the older 9510-step VGGT layer ablation on clean sequence-meta-filtered GT under `*robustval_metafilter_seed17_vggt_layerXX`; it is now superseded by 50-epoch VGGT1 baselines. The current default is VGGT1 MLP layer `16` from job `86571`, `F@0.10=0.702544731989172`, and the strongest readout comparison is VGGT1 CA layer `20` from job `86572`, `F@0.10=0.7014525243138799`.
- WAN is paused as a main branch after the 2026-05-19 audit; see `docs/probe/wan_summary_2026-05-19.md`. WAN sanity job `86316` completed; WAN pack job `86307` completed all 15 Route2 hidden-state runs. Old best WAN hidden baseline was `t499/layer09`, above zero/sample-shuffle controls but much weaker than clean-GT VGGT. Route2.1 `no_noise` / `low_noise`, targeted `t499/layer09 + token_layernorm`, `pair_tiled81`, latent tensor-choice probes, simple hidden-grid readouts, multi-layer fusion, multi-timestep fusion, FLF2V hidden, and gated CA completed and did not beat the best single-layer T2V CA baselines. The learned hidden CA resampler is the best WAN readout family: historical best `t499/layer14` reached `F@0.10=0.5012956284974035`; repeated settings are around `0.49`. WAN remains below clean-GT VGGT and should not be expanded broadly before the next backbone probe.
- initialize third-party submodules with `git submodule update --init --recursive`; VGGT training imports from `third_party/vggt`, and the VGGT-Omega probe imports from `third_party/vggt-omega`.
- VGGT-Omega dense/full-token MLP, dense/full-token CA, and register-only / "Frozen Scene Tokens" CA completed layers `12,16,20,24` using official `checkpoints/vggt_omega/vggt_omega_1b_512.pt`. Best Omega is dense CA layer16 (`F@0.10=0.6576072630447046`), still below VGGT1 MLP layer16 (`0.702544731989172`) and VGGT1 CA layer20 (`0.7014525243138799`). Treat this as a NOVA/FM probe-validity warning, not a final model-quality conclusion.
- VGGT1 50-epoch baseline completion finished as jobs `86571` (MLP-L4-H1024, elapsed `02:08:56`) and `86572` (cross_attention-L4-H1024, elapsed `02:13:51`), both with exit `0:0` on `air-node-02`.


Minimal collaborator-side probing experiment for decoding complete 3D geometry from frozen NOVA3R / VGGT features.

This repo now also contains the more structured `docs/probe/`, `configs/probe/`, `nova3r/probe/`, and `scripts/probe/` workspace. The files in `experiments/probe3d/` are kept as the direct experimental path.

The backbone is frozen. Only the lightweight adapter / decoder heads are trained.

## Critical data warning

Do **not** treat the local `eval_scrream` package as training data for formal SCRREAM experiments.

That package is the released **evaluation subset** (~1.6 GB on the current machine), not the official full SCRREAM dataset (~200 GB scale). Historical runs that trained from `eval_scrream` should be treated as **debug / invalid-for-claim** runs, not final evidence.

## SCRREAM-full mesh-complete adapter bridge

For the full SCRREAM layout at `~/datasets/SCRREAM`, use `prepare_scrream_full_adapter_data.py` with `--target_source mesh_complete`. This path reads the official two-view pair list, uses the two RGB frames as adapter inputs, reads the current sequence `meta.txt`, samples only the listed registered scene meshes, crops the complete point cloud to the selected input-pair union frustum, and stores fixed-size target point clouds in the first input view coordinate frame.

Current target semantics:

- pair list: `data/scrream/scrream_n2_list.json`
- input images: the two pair frames, for example `scene09/scene09_full_00 200 275`
- mesh source: `sceneXX/meshes/*.obj` filtered by the current sequence `meta.txt`
- mesh sampling: surface-area proportional across the selected sequence object meshes
- cache: sequence/object-set mesh reservoirs under `experiments/probe3d/adapter_data/mesh_cache/`
- frustum crop: keep mesh points inside at least one selected input-view frustum with positive depth
- target frame: first input camera coordinates
- final target: deterministic FPS to the requested target count, with replacement padding only when the crop is undersized
- output schema: `scene_ids`, `target_points`, `splits`, `metadata`, and global `meta`

The earlier equal-points-per-OBJ preview under `scrream_mesh_complete_n2_preview/` underweighted room-scale surfaces. The accepted current preview uses area-proportional sampling and lives under:

- `experiments/probe3d/adapter_data/scrream_mesh_complete_area_n2_preview/`

## SCRREAM WAN-T2V video-context probe

The WAN route2 probe keeps the current SCRREAM N2 adapter data and mesh-complete GT unchanged, but replaces VGGT features with precomputed WAN2.1 T2V features. This is a **video-context** representation: each two-view pair is embedded by loading WAN T2V on an 81-frame RGB window around the pair, then keeping only the two temporal slices corresponding to the original pair frames. Adapter training reads the cached WAN features and does not run WAN inside every training step.

Route2 hidden-state defaults:

- model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- local checkpoint: `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers`
- timesteps: `249,499,749`
- code layer ids: `9,14,19,24,29` (human layers 10/15/20/25/30)
- cache shape per sample: `[3120,1536]`
- network proxy for repo/checkpoint fetches: `http://127.0.0.1:17890`

The cache generator also supports latent feature kinds. `--feature_kind pred_x0_latent` keeps the same 81-frame WAN forward pass but caches the scheduler clean-latent estimate `x0 = sample - sigma * model_output` instead of a block hidden state. `--feature_kind model_output_latent` caches the raw WAN transformer `output.sample` tensor. Both write one cache per timestep under `t499/<feature_kind>/<sample>.pt`, record scheduler/latent metadata, and produce pair features of shape `[12480,16]`.

The original pred-x0 MLP-L4 readout failed during training because pooling `[12480,128] -> [768,128]` triggered a PyTorch CUDA adaptive-pooling backward assert. Two latent readouts are now supported: `--adapter_type conv2d_mlp` restores `[B,2,60,104,16]`, applies a small stride-2 Conv2d stem to `[B,3120,128]`, and pools to NOVA scene tokens; `--adapter_type wan_latent_cross_attn_resampler` restores the same latent grid, adds temporal/row/column position embeddings, and cross-attends `768` NOVA query tokens to the full latent grid. The learned latent CA path is the fairer comparison against the successful hidden CA readout.

On this Slurm cluster the `17890` proxy is bound to login-node loopback. WAN Slurm jobs therefore open an SSH local tunnel from the compute node back to `air-server:127.0.0.1:17890`, then export `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` to the compute node's local forwarded port. Override with `WAN_PROXY_SSH_HOST`, `WAN_PROXY_REMOTE_PORT`, `WAN_PROXY_LOCAL_PORT`, or set `WAN_DISABLE_PROXY=1`.

WAN-specific Python dependencies are isolated in `experiments/probe3d/requirements-wan-t2v.txt`. They are not merged into the root environment files. Job `slurm/scrream_wan_t2v_download.sbatch` installs/validates them by default before downloading the checkpoint.

Status on `2026-05-19 14:14 CST`: the WAN checkpoint is present under `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers` (~27 GB). Full feature precompute completed as jobs `86292`, `86293`, and `86294`; cache root `experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81` contains `4937` `.pt` files / about `45G`. The first full training pack `86306` failed immediately on a final-loss scalar bug, `train_wan_t2v_nova_adapter.py` was patched, and replacement pack job `86307` completed all 15 runs on `air-node-04`, exit `0:0`, elapsed `13:36:04`. Learned hidden CA-resampler jobs `86429`, `86453`, `86468`, `86485`, `86487`, `86494`, `86547`, `86548`, `86549`, `86550`, `86553`, and `86554` completed. Latent tensor-choice jobs `86536`, `86540`, and `86543` completed negative. The final WAN conclusion is frozen in `docs/probe/wan_summary_2026-05-19.md`.

Route2 result:

- old best WAN hidden baseline: `t499/layer09`, `best_val_fscore_tau_0.10=0.46988987902779306`, `best_val_pred_to_gt_p90=0.48718947172164917`, `best_val_chamfer_l2=0.21040735269586244`
- historical best WAN interface: `t499/layer14 + wan_cross_attn_resampler`, `best_val_fscore_tau_0.10=0.5012956284974035`, `best_val_pred_to_gt_p90=0.4229188362757365`, `best_val_chamfer_l2=0.10908368105689685`; user visual inspection confirmed improvement
- repeated CA-resampler signal: low-LR `t499/layer14` rerun `86453` was negative (`F@0.10=0.4292316005720653`); full-grid `86468` best F@0.10 is `t249/layer14=0.48913902331806663`, while best p90 / Chamfer is `t249/layer09=0.410525918006897 / 0.11981592203179996`
- current VGGT1 MLP layer `16` 50ep: `best_val_fscore_tau_0.10=0.702544731989172`, `best_val_pred_to_gt_p90=0.19675995161135992`, `best_val_chamfer_l2=0.027118226668486994`
- historical clean-GT VGGT layer `16` from job `86286`: `best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`, `best_val_chamfer_l2=0.026904070439438026`
- Route2.1 result: `no_noise` / `low_noise` WAN cache modes for layers `9,14,29` completed, but the best clean/low-noise run (`no_noise/layer29`, `F@0.10=0.44840173`) did not beat old Route2 `t499/layer09`
- targeted norm check: `t499/layer09 + token_layernorm` completed as job `86395`; it did not beat old Route2 best (`F@0.10=0.45476213575933216`, `pred_to_gt_p90=0.484821617603302`, `Chamfer=0.1771892917652925`)
- pair-context check: `pair_tiled81` completed as jobs `86396 -> 86397 -> 86400 -> 86401`; it did not beat old Route2 best (`F@0.10=0.4429200036613287`, `pred_to_gt_p90=0.5104739194115003`, `Chamfer=0.16478415516515574`)
- pred-x0 cache job `86411` completed `329/329` full-cache samples under `experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81_pred_x0_latent/t499/pred_x0_latent/`
- pred-x0 conv smoke job `86421` completed successfully on `air-node-02`, exit `0:0`, and wrote robust validation metrics
- pred-x0 conv formal job `86422` completed on `air-node-02`, exit `0:0`, elapsed `00:52:13`; result: `F@0.10=0.41336424426176693`, `pred_to_gt_p90=0.6272750149170557`, `Chamfer-L2=0.35522504647572833`
- structured hidden readout jobs completed on `air-node-04`: `grid2d_pool` formal job `86424` yielded `F@0.10=0.43326753863230877`, `pred_to_gt_p90=0.5651116619507471`, `Chamfer-L2=0.1759416777640581`; `grid2d_conv` formal job `86426` yielded `F@0.10=0.39855373112050535`, `pred_to_gt_p90=0.7689011543989182`, `Chamfer-L2=1.159957120815913`
- learned hidden readout jobs completed on `air-node-02`: `wan_cross_attn_resampler` layer09 smoke/formal jobs `86427` / `86428` nearly matched old WAN hidden F@0.10 while improving p90/Chamfer; layer sweep job `86429` found a historical high point at layer14; full-grid job `86468` shifted the most repeatable settings toward `t249/layer09/14`
- multi-source learned readouts are implemented: `wan_cross_attn_multilayer_resampler` concatenates multiple layers at one timestep via `--wan_layers`, and `wan_cross_attn_multitime_resampler` concatenates multiple timesteps at one layer via `--wan_timesteps`
- multi-layer and multi-timestep fusion did not close the gap: `t249 layers9-14` reached `F@0.10=0.46487238144059545`, `t249 layers9-14-19` reached `0.4724058254433277`, and `249+499+749/layer14` reached `0.45213117002164466`; all are below single-layer `t249/layer14` and the historical CA peak
- latent tensor-choice readouts: `model_output_latent + conv2d_mlp` job `86536` reached F@0.10 `0.4199299775478907`; `model_output_latent + wan_latent_cross_attn_resampler` job `86540` reached `0.4289946537162989`; `pred_x0_latent + wan_latent_cross_attn_resampler` job `86543` reached `0.43766060222664477`. Learned latent CA helps over Conv2d but remains below hidden CA, so stop this tensor-choice branch for now.
- hidden CA stability/capacity results: seed23 standard CA reached `0.48706357506233194` at `t249/layer14` and `0.4591392560862819` at `t499/layer14`; gated CA reached `0.4505522726705196` at `t249/layer14` and `0.4866296484297818` at `t499/layer14`; L4 CA reached `0.4919946248489035` at `t249/layer14` and `0.46280321302050603` at `t499/layer14`.
- current interpretation: fixed pred-x0 Conv2d readout, fixed model-output Conv2d readout, learned latent CA readout, fixed 2D hidden pooling, tiny fixed 2D conv readout, simple hidden fusion, gated CA, and FLF2V hidden do not close the WAN gap; a learned WAN-hidden-to-NOVA token interface is the only clearly positive branch. WAN is paused as the main line. Optional appendix checks are L4 `t249/layer09` or a seed repeat of L4 `t249/layer14`; the active direction is the next backbone probe.

Download/preflight WAN dependencies and checkpoint:

```bash
sbatch slurm/scrream_wan_t2v_download.sbatch
```

Validate all SCRREAM pair windows without loading WAN:

```bash
WAN_WINDOW_ONLY=1 sbatch slurm/scrream_wan_t2v_precompute.sbatch
```

Feature smoke:

```bash
WAN_TIMESTEPS=749 WAN_LAYERS=20 WAN_MAX_SAMPLES=2 \
  sbatch slurm/scrream_wan_t2v_precompute.sbatch
```

Full feature cache:

```bash
sbatch slurm/scrream_wan_t2v_precompute.sbatch
```

Full 15-run adapter ablation:

```bash
sbatch slurm/scrream_wan_t2v_ablation_pack_train.sbatch
```

Multi-source learned resampler audits:

```bash
WAN_TIMESTEPS=249 \
WAN_LAYER_GROUPS=9+14,9+14+19 \
sbatch slurm/scrream_wan_t2v_multilayer_pack_train.sbatch

WAN_TIMESTEP_GROUPS=249+499+749 \
WAN_LAYERS=14 \
sbatch slurm/scrream_wan_t2v_multitime_pack_train.sbatch
```

Predicted-x0 latent smoke and formal chain:

```bash
WAN_FEATURE_KIND=pred_x0_latent \
WAN_TIMESTEPS=499 \
WAN_LAYERS=0 \
WAN_MAX_SAMPLES=2 \
sbatch slurm/scrream_wan_t2v_precompute.sbatch

WAN_FEATURE_KIND=pred_x0_latent \
WAN_ADAPTER_TYPE=conv2d_mlp \
WAN_TIMESTEPS=499 \
WAN_LAYERS=0 \
sbatch slurm/scrream_wan_t2v_ablation_pack_train.sbatch
```

Model-output latent audit:

```bash
WAN_FEATURE_KIND=model_output_latent \
WAN_TIMESTEPS=499 \
WAN_LAYERS=0 \
sbatch slurm/scrream_wan_t2v_precompute.sbatch

WAN_FEATURE_KIND=model_output_latent \
WAN_ADAPTER_TYPE=conv2d_mlp \
WAN_TIMESTEPS=499 \
WAN_LAYERS=0 \
sbatch slurm/scrream_wan_t2v_ablation_pack_train.sbatch
```

## SCRREAM WAN-FLF2V pair-endpoint probe

FLF2V is intentionally separate from the T2V `ctx81` route. It uses first/last-frame conditioning and a `pair_endpoint81` window:

- slot `0`: first SCRREAM pair frame `f0`
- slot `80`: second SCRREAM pair frame `f1`
- slots `1..79`: linearly sampled SCRREAM sequence frame IDs between `f0` and `f1`
- extracted hidden temporal indices: `[0,20]`

First probe setting:

- model: `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers`
- local checkpoint: `checkpoints/wan2.1/Wan2.1-FLF2V-14B-720P-diffusers`
- cache root: `experiments/probe3d/feature_cache/scrream_wan_flf2v14b_pair_endpoint81_480`
- feature script: `experiments/probe3d/scripts/prepare_scrream_wan_flf2v_feature_cache.py`
- training script: reuse cache-backed `experiments/probe3d/train_wan_t2v_nova_adapter.py`
- Slurm: `slurm/scrream_wan_flf2v_download.sbatch`, `slurm/scrream_wan_flf2v_precompute.sbatch`, `slurm/scrream_wan_flf2v_train.sbatch`
- first formal run: `t249/layer14`, empty prompt, `--adapter_type wan_cross_attn_resampler`, 50 epochs

The 480P expected pair cache shape is `[3120,5120]`. If a future run must fall back to 720P, expected pair cache shape is `[7200,5120]`. The CA-resampler now reads hidden-grid shape from cache metadata and supports `input_dim=5120`.

Status on `2026-05-19`: checkpoint download/preflight completed as job `86500` after an initial transient incomplete-read failure in job `86497`; the local checkpoint is about `84G`. Static checks, `bash -n`, dummy `5120` adapter forward/backward, and 329-sample window-only smoke passed. Cache shard jobs `86515` / `86516` / `86517` / `86518` completed `329/329` samples at `[3120,5120]`; smoke/formal training jobs `86519` / `86520` completed. Formal FLF2V result is `F@0.10=0.42897984457682026`, `pred_to_gt_p90=0.5051772321263949`, and `Chamfer=0.14052311765650907`, below T2V CA baselines.

FLF2V smoke commands:

```bash
sbatch slurm/scrream_wan_flf2v_download.sbatch

WAN_WINDOW_ONLY=1 \
sbatch slurm/scrream_wan_flf2v_precompute.sbatch

WAN_MAX_SAMPLES=2 \
WAN_FLF2V_TIMESTEPS=249 \
WAN_FLF2V_LAYERS=14 \
WAN_CPU_OFFLOAD=1 \
sbatch --gres=gpu:a100:1 --mem=128G \
  slurm/scrream_wan_flf2v_precompute.sbatch
```

## SCRREAM VGGT-Omega probe

VGGT-Omega is the next backbone probe after the WAN audit. It keeps the clean SCRREAM mesh-complete `.pt`, split, MLP-L4 adapter, NOVA `scene_ae`, `nova_flow`, and robust validation fixed.

- source: `third_party/vggt-omega`
- official checkpoint path: `checkpoints/vggt_omega/vggt_omega_1b_512.pt`
- important audit note: `checkpoints/vggt_omega/model.pt` was checked on `2026-05-19` and is byte-identical to `checkpoints/vggt/model.pt`; it has old VGGT 4-register / 14-patch / `global_blocks` keys and must not be used as a VGGT-Omega checkpoint
- training entry: `experiments/probe3d/train_vggt_nova_adapter.py --backbone vggt_omega`
- download/preflight Slurm: `slurm/scrream_vggt_omega_download.sbatch`
- layer pack Slurm: `slurm/scrream_vggt_omega_layer_ablation_pack_train.sbatch`

Dense/full aggregator tokens use `aggregator.cached_layer_indices` forced to all layers so old VGGT comparison layers are available. The first dense MLP job `86564` completed with:

| backbone / layer | F@0.10 | pred-to-GT p90 | Chamfer-L2 |
| --- | ---: | ---: | ---: |
| old VGGT layer16 | `0.6860468604251301` | `0.20770130679011345` | `0.026904070439438026` |
| old VGGT layer24 | `0.6802513448244976` | `0.25152526050806046` | `0.03554012098660072` |
| Omega dense layer16 | `0.6550556579677611` | `0.23105039075016975` | `0.03261705581098795` |
| Omega dense layer24 | `0.6133813288610513` | `0.25702105338374776` | `0.03972778360669812` |

The later Omega CA and register-only runs completed the immediate feature/readout audit:

| Omega token / adapter | best layer | F@0.10 | pred-to-GT p90 | Chamfer-L2 |
| --- | ---: | ---: | ---: | ---: |
| dense MLP | 16 | `0.6550556579677611` | `0.23105039075016975` | `0.03261705581098795` |
| dense CA | 16 | `0.6576072630447046` | `0.23843258867661157` | `0.033605430430422224` |
| register-only CA | 16 | `0.4930976482166729` | `0.4645070905486743` | `0.20077569534381232` |

This is a live integration but a negative result versus VGGT1 across the tested Omega readout branches. The completed VGGT1 50ep table is:

| adapter / layer | F@0.10 | pred-to-GT p90 | Chamfer-L2 |
| --- | ---: | ---: | ---: |
| MLP layer12 | `0.5316993988168482` | `0.37930816908677417` | `0.07861695190270741` |
| MLP layer16 | `0.702544731989172` | `0.19675995161135992` | `0.027118226668486994` |
| MLP layer20 | `0.519013588803542` | `0.38937361538410187` | `0.07316385923574369` |
| MLP layer24 | `0.6690686277634136` | `0.2389497272670269` | `0.0331160935262839` |
| CA layer12 | `0.5065650562192365` | `0.33501065025726956` | `0.07443799295773108` |
| CA layer16 | `0.6835200371527321` | `0.23028291016817093` | `0.030130337458103895` |
| CA layer20 | `0.7014525243138799` | `0.21181315431992212` | `0.027010128212471802` |
| CA layer24 | `0.6204781376731087` | `0.2675088259081046` | `0.03806558856740594` |

Use MLP layer16 as the main VGGT1 baseline. CA layer20 is the key readout comparison because it nearly ties MLP layer16 and rescues the layer20 representation. Since Omega remains below VGGT1 after dense MLP, dense CA, and register-only CA, the next experiment should not be another broad Omega sweep. It should be a decoder-free SCRREAM direct spatial readout that tests whether Omega ranks above VGGT1 without the NOVA/FM generator interface.

Run these commands from the `nova3r` conda environment.

Small smoke directly from shell if you only need a quick local check:

```bash
python3 experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py \
  --data_root ~/datasets/SCRREAM \
  --pair_list data/scrream/scrream_n2_list.json \
  --output_path experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_smoke.pt \
  --target_source mesh_complete \
  --target_points 10000 \
  --max_samples 2 \
  --save_preview_dir experiments/probe3d/adapter_data/scrream_mesh_complete_n2_preview
```

Preferred Slurm smoke:

```bash
SCRREAM_ADAPTER_OUT=experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_smoke8.pt \
SCRREAM_EXTRA_ARGS="--max_samples 8 --save_preview_dir experiments/probe3d/adapter_data/scrream_mesh_complete_smoke8_preview" \
sbatch slurm/scrream_mesh_complete_prepare.sbatch
```

The prep script runs `check_scrream_adapter_training_inputs.py` after writing the `.pt`, so schema / split / frame-path issues fail before training.

Full adapter dataset after the smoke preview looks correct:

```bash
sbatch slurm/scrream_mesh_complete_prepare.sbatch
```

Generated formal datasets now available locally:

- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.pt`
  - shape `[329, 10000, 3]`
  - split `train=223`, `val=12`, `test=94`
- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_trainplus_test.pt`
  - split `train=317`, `val=12`
- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000.pt`
  - shape `[329, 20000, 3]`
  - `mesh_sample_points=500000`
  - split `train=223`, `val=12`, `test=94`
- `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
  - split `train=317`, `val=12`, `mesh_sequence_meta_filter=True`

The `trainplus_test` files rewrite split labels only: `test` becomes `train`, and `val` is preserved. They have no held-out test split.

The 20k / 500k data was regenerated on `2026-05-10` with sequence-level object filtering from each sequence `meta.txt`. Old pre-meta-filter `.pt` files were moved to `experiments/probe3d/adapter_data/deprecated_meta_filter_bug/`. The corresponding mesh cache now uses sequence/meta-hash labels such as `scene08_scene08_reduced_00_meta7c4e1a9bf084_...npz`, so it no longer reuses scene-level all-object reservoirs.

Training smoke after a smoke `.pt` exists:

```bash
SCRREAM_ADAPTER_DATA=experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_smoke8.pt \
SCRREAM_MAX_STEPS=20 \
SCRREAM_OUTPUT_DIR=experiments/probe3d/result/scrream_mesh_complete_smoke8_mlp_smoke \
sbatch slurm/scrream_mesh_complete_mlp_smoke.sbatch
```

The smoke script uses the same training entrypoint and loss as the full run. It checks the adapter `.pt`, loads the NOVA scene autoencoder checkpoint, extracts VGGT features, verifies adapter gradients, and writes a short checkpoint / PLY export.

First full MLP baseline after the full `.pt` exists:

```bash
sbatch slurm/scrream_mesh_complete_mlp_train.sbatch
```

Current 20k / 500k trainplus-test run:

```bash
SCRREAM_ADAPTER_DATA=experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt \
SCRREAM_OUTPUT_DIR=experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17 \
SCRREAM_FEATURE_CACHE_DIR=experiments/probe3d/feature_cache/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_vggt23 \
SCRREAM_SWANLAB=1 \
SCRREAM_NUM_QUERIES=20000 \
SCRREAM_MAX_STEPS=9510 \
SCRREAM_SAVE_EVERY=500 \
SCRREAM_VAL_EVERY=500 \
SCRREAM_EVAL_BATCHES=2 \
sbatch --qos=high --nodelist=air-node-02 --gres=gpu:a100:1 --mem=32G \
  slurm/scrream_mesh_complete_mlp_train.sbatch
```

This submitted job `86140`, which completed on `air-node-02` with exit `0:0`, elapsed `00:26:26`, and Slurm end time `2026-05-07 22:40:22 CST`.

Important: job `86140` and the old robust VGGT layer ablation job `86149` used the pre-meta-filter GT. The clean-GT rerun is completed Slurm job `86286`, with output prefix:

- `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_vggt_layerXX`

Historical clean-GT VGGT ranking from 9510-step job `86286`:

- layer `16`: old best 9510-step VGGT reference, `best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`, `best_val_chamfer_l2=0.026904070439438026`
- layer `24`: old close comparison point, `best_val_fscore_tau_0.10=0.6802513448244976`, `best_val_pred_to_gt_p90=0.25152526050806046`, `best_val_chamfer_l2=0.03554012098660072`
- layer `20`: no longer the clean-GT default, `best_val_fscore_tau_0.10=0.548158719135383`

Current VGGT1 reference after the 50-epoch completion is MLP layer `16` from job `86571`: `best_val_fscore_tau_0.10=0.702544731989172`, `best_val_pred_to_gt_p90=0.19675995161135992`, `best_val_chamfer_l2=0.027118226668486994`. The main readout comparison is CA layer `20` from job `86572`: `F@0.10=0.7014525243138799`, `pred_to_gt_p90=0.21181315431992212`, `Chamfer=0.027010128212471802`.

Completed run artifacts:

- output: `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_seed17`
- SwanLab: `https://swanlab.cn/@JiachengDong/PSUVPSC3DD/runs/eoiupi3bv2g11dvd21ypm`
- `final_metrics.json`: `first_loss=1.4599288702011108`, `final_loss=0.8398033976554871`, `best_loss=0.5998285412788391`, `best_val_chamfer_l2=0.5149603486061096`
- latest `validation_metrics.json`: step `9500`, `val_chamfer_l2=0.6779176592826843`

Future multi-GPU follow-up runs should use the script's DDP path:

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

When `SCRREAM_GPUS_PER_NODE>1`, the Slurm script launches `torchrun --standalone`; with `SCRREAM_EPOCHS` it computes the correct distributed step count from `train_count / (batch_size_per_gpu * gpu_count)`.

If checkpoints are not in the default repo paths, pass them as environment variables:

```bash
SCRREAM_NOVA_CKPT=/path/to/scene_ae/checkpoint-last.pth \
SCRREAM_VGGT_WEIGHTS=/path/to/VGGT-1B/model.pt \
sbatch slurm/scrream_mesh_complete_mlp_train.sbatch
```

Training defaults follow the current NOVA-style adapter baseline:

- `--dataset scrream_adapter`
- `--num_views 2`
- `--adapter_type mlp`
- `--adapter_layers 4`
- `--adapter_hidden_dim 1024`
- `--loss_type nova_flow`
- `--num_queries 10000`
- persistent VGGT layer-23 cache under `experiments/probe3d/feature_cache/`
- target normalization follows the NOVA `scene_ae` decoder checkpoint `norm_mode`; local `scene_ae` currently reports `median_3`
- full MLP Slurm training enables SwanLab by default with project `PSUVPSC3DD`

Local weights currently staged under `checkpoints/`:

- `scene_n1/checkpoint-last.pth`
- `scene_n2/checkpoint-last.pth`
- `scene_ae/checkpoint-last.pth`
- `vggt/model.pt`

The Slurm scripts default `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` to `http://127.0.0.1:7896` for network-dependent HuggingFace / SwanLab access.

The older `prepare_scrream_adapter_data.py` remains a legacy LDI / pseudo-GT path. It expects `ldi/` and `*_ldi.npz`, so it is not the right entrypoint for the downloaded full SCRREAM tree.

### Historical depth-GT / depth-mix branches

`prepare_scrream_full_adapter_data.py` still supports `--target_source depth_gt_dense` as an alternate baseline. That mode aggregates `depth_gt` frames between the input pair, voxel-filters duplicates, crops to the input union frustum, and stores targets in the first input camera frame.

Use it only when intentionally comparing depth aggregation against mesh-complete targets. It does not produce truly mesh-complete surfaces; it can include surfaces seen by nearby dense frames that were not visible in the first input, but it cannot recover unobserved mesh backsides except where other depth frames observed them.

Local `scrream_official_depth_mix_*` `.pt` and preview artifacts under `experiments/probe3d/adapter_data/` are historical ablations. They are not part of the current adapter-training line and should be ignored unless the user explicitly asks for a depth-mix ablation.

## Legacy LDI / pseudo-GT adapter path

The following path is kept for historical reference and only applies to SCRREAM-style data that has `ldi/` and `*_ldi.npz` files. It is not the path for `~/datasets/SCRREAM`.

## Step 1: Build Adapter Data

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/scripts/prepare_scrream_adapter_data.py \
  --data_root /path/to/official_full_scrream \
  --output_root experiments/probe3d/adapter_data \
  --group_size 4 \
  --sample_stride 1 \
  --pad_short_scenes \
  --pseudo_gt_views 2 \
  --pseudo_gt_queries 20000 \
  --feature_ckpt checkpoints/scene_n2/checkpoint-last.pth \
  --pseudo_gt_ckpt checkpoints/scene_n2/checkpoint-last.pth \
  --skip_failures
```

This creates:

- a scene-level `train` / `val` / `test` split
- 4-frame groups per scene
- a manifest JSON with per-sample frame lists
- an adapter-training `.pt` dataset with `features`, `target_points`, `splits`, and per-sample `metadata`

`pseudo_gt_views=2` keeps pseudo GT on the official NOVA3R inference path while each adapter sample still carries 4 frames.

Before running this step, verify that `--data_root` points to the **official full SCRREAM release** and contains the expected full-scene metadata (for example `camera_pose/` where required). If it only looks like `eval_scrream`, stop and fix the data source first.

## Step 2: Run Official Baseline

Run the official NOVA3R SCRREAM baseline first to verify checkpoints, config, dataset paths, and CUDA environment.

## Step 3: Inspect Outputs

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/scripts/inspect_nova3r_outputs.py \
  +experiment.ckpt_path=checkpoints/scene_n1/checkpoint-last.pth \
  +experiment.test_dataset_name=scrream_n1 \
  +experiment.data_root=/path/to/datasets
```

The script prints output keys and tensor shapes. If model config is not found, run it with the same Hydra model config used by `eval/mv_recon/test_nova3r.py`.

## Step 4: Extract Features

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/scripts/extract_nova3r_features.py \
  +experiment.ckpt_path=checkpoints/scene_n1/checkpoint-last.pth \
  +experiment.test_dataset_name=scrream_n1 \
  +experiment.data_root=/path/to/datasets \
  --output_path experiments/probe3d/features/nova3r_scrream_n1.pt
```

If `--feature_key` is omitted, the script prints candidates and uses the first feature-like tensor with a warning. Use `--target_key` only after inspecting batch tensor shapes if automatic complete-point extraction fails.

## Step 5: Train Probe

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/train_probe.py \
  --feature_path experiments/probe3d/adapter_data/scrream_adapter_dataset.pt \
  --save_path experiments/probe3d/checkpoints/probe_nova3r_d1.pt \
  --train_split train \
  --val_split val \
  --adapter_depth 1 \
  --latent_dim 512 \
  --num_points 8192 \
  --batch_size 4 \
  --epochs 100
```

## Step 6: Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/probe3d/eval_probe.py \
  --feature_path experiments/probe3d/adapter_data/scrream_adapter_dataset.pt \
  --checkpoint experiments/probe3d/checkpoints/probe_nova3r_d1.pt \
  --split test
```

Use `--save_predictions` to write predicted point clouds to `experiments/probe3d/outputs/`.

## Notes

- Keep `adapter_depth` small. A shallow adapter is the point of this representation probe.
- If `pytorch3d` is unavailable, this code uses a simple `torch.cdist` Chamfer L2 implementation.
- VGGT code is now available in-repo at `third_party/vggt`.
- `dust3r/datasets/` and `datasets_preprocess/` are also vendored locally, so the common data-loading / ScanNet-prep path is less dependent on an external CUT3R checkout.
- For the structured shared-decoding sanity path, see `scripts/probe/run_vggt_to_nova3r_decoder.py` and `docs/probe/`.
- Feature files, checkpoints, outputs, and datasets are ignored by git.
## Current ScanNet probe status

The current best short-run ScanNet result is not from the old long formal MLP schedule, but from the compact probe harness:

- harness: `experiments/probe3d/probe_trials/`
- best run: `p1_adapter_anchor_frustum_mlp_l4_chamfer_lr1e5_refine_step2500`
- target: `anchor_frustum`
- adapter: `MLP-L4, hidden=1024`
- loss: direct sampled rollout Chamfer (`loss_type=chamfer_sample`)
- best validation CD: `0.08745259`

Interpretation: direct rollout Chamfer fixed the large train/eval objective mismatch seen with `nova_flow`, but the visual point cloud is still loose / thick / outlier-heavy. The next active data task is to inspect the completed SCRREAM full mesh-complete adapter baseline; InteriorGS is deferred until this corrected full-data branch is understood.
### Paper-aligned NOVA3R reset

After user review, the active plan is to align the ScanNet target/loss more closely with NOVA3R: complete / amodal points inside the selected input-view frustum, FPS-style target sampling through `src_complete_fps_*`, and native flow matching as the primary loss. The new phase-2 config is:

- `experiments/probe3d/probe_trials/configs/phase2_nova_aligned.json`

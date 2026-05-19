# TODO

## Active now — 2026-05-19

### SCRREAM full mesh-complete adapter line
- [x] download full SCRREAM to `~/datasets/SCRREAM`
- [x] confirm the old `eval_scrream` subset must not be used for formal claims
- [x] add `experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py`
- [x] add `--target_source mesh_complete`
- [x] sample sequence-selected SCRREAM scene meshes with surface-area-proportional budgets
- [x] crop mesh-complete targets to the selected two-view union frustum
- [x] store adapter targets in the first input camera frame
- [x] verify a scene09 mesh-complete area-proportional PLY preview
- [x] update `scrream_adapter` loading so `--data_root` can point to the generated `.pt`
- [x] add Slurm scripts under `slurm/`
- [x] send Slurm logs to `slurm_out/`
- [x] download NOVA3R `scene_n1`, `scene_n2`, `scene_ae`, and VGGT weights under `checkpoints/`
- [x] install and verify `swanlab` in conda env `nova3r`
- [x] set Slurm network proxy defaults to `http://127.0.0.1:7896`
- [x] validate SCRREAM training-input precheck against the preview `.pt` and `scene_ae` checkpoint
- [x] submit full adapter data generation as Slurm job `85773`
- [x] submit first MLP baseline as dependent Slurm job `85774`
- [x] generate `scrream_mesh_complete_n2_adapter_seed17.pt` with shape `[329, 10000, 3]`
- [x] generate `scrream_mesh_complete_n2_adapter_seed17_trainplus_test.pt` with split `train=317`, `val=12`
- [x] generate `scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000.pt` with shape `[329, 20000, 3]`
- [x] generate `scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt` with split `train=317`, `val=12`
- [x] add `slurm/scrream_merge_test_into_train.sbatch`
- [x] initialize the VGGT submodule after submodule conversion
- [x] launch 20k / 500k trainplus-test MLP baseline as Slurm job `86140`
- [x] verify job `86140` loads VGGT weights, starts SwanLab, writes checkpoints, and reaches validation at step `3000`
- [x] add multi-GPU DDP launch support to `slurm/scrream_mesh_complete_mlp_train.sbatch`
- [x] confirm job `86140` completed successfully with exit `0:0`
- [x] record job `86140` final metrics in the handoff docs
- [x] implement robust validation metrics and `val_visual_40960/` exports
- [x] run packed VGGT layer ablation as Slurm job `86149`
- [x] confirm job `86149` completed successfully with exit `0:0`
- [x] record pre-meta-filter VGGT layer `20` as the historical best layer before the sequence-meta GT correction
- [x] record robust layer-ablation results in the handoff docs
- [x] identify and fix sequence-level phantom object contamination in SCRREAM mesh-complete GT
- [x] regenerate 20k / 500k adapter data with `mesh_sequence_meta_filter=True` as Slurm jobs `86283` and `86284`
- [x] archive old pre-meta-filter `.pt` files under `experiments/probe3d/adapter_data/deprecated_meta_filter_bug/`
- [x] launch clean-GT VGGT layer rerun as Slurm job `86286`
- [x] confirm job `86286` completed successfully with exit `0:0`
- [x] record clean-GT VGGT layer `16` as the current default and layer `24` as the main comparison point
- [ ] inspect clean-GT layer `16` vs layer `24` visual PLYs before making a qualitative claim
- [ ] expand SCRREAM training data scale beyond the current 329 official pairs
- [ ] test more adapter/model variants after the clean-GT VGGT baseline is established

### Current execution convention
- [x] use conda env `nova3r`
- [x] keep generated adapter `.pt`, previews, and mesh caches under ignored `experiments/probe3d/adapter_data/`
- [x] keep Slurm scripts under `slurm/`
- [x] keep Slurm logs under `slurm_out/`
- [x] use SwanLab by default for the formal SCRREAM MLP Slurm script unless `SCRREAM_SWANLAB=0`
- [x] keep SwanLab API key in ignored local `slurm/.secrets.env`
- [x] use `SCRREAM_GPUS_PER_NODE` + `SCRREAM_EPOCHS` for future multi-GPU Slurm training
- [x] change formal SCRREAM VGGT/WAN ablation defaults from 30 epochs to 50 epochs
- [x] ignore `scrream_official_depth_mix_*` artifacts unless explicitly running a depth-mix ablation
- [x] keep robust validation and `val_visual_40960/` enabled for formal runs
- [x] add project-level VSCode exclusions so large result/cache/checkpoint directories remain visible but are not watched/searched/indexed

### SCRREAM WAN2.1 T2V Route2 probe
- [x] add `third_party/VidFM3D` as a reference submodule
- [x] keep WAN/VidFM3D dependencies out of the root `requirements.txt` and `environment.yml`
- [x] add isolated dependency pins in `experiments/probe3d/requirements-wan-t2v.txt`
- [x] add WAN feature cache generator at `experiments/probe3d/scripts/prepare_scrream_wan_t2v_feature_cache.py`
- [x] add separate WAN training entrypoint at `experiments/probe3d/train_wan_t2v_nova_adapter.py`
- [x] keep `train_vggt_nova_adapter.py` as the VGGT entrypoint
- [x] update shared adapter helpers so trainable-parameter checks allow `vggt=None`
- [x] add WAN Slurm scripts under `slurm/`
- [x] route WAN repo/checkpoint jobs through `http://127.0.0.1:17890`
- [x] add compute-node SSH tunnel fallback to `air-server:127.0.0.1:17890` for WAN Slurm scripts
- [x] validate WAN import/dependency preflight in conda env `nova3r`
- [x] validate all `329` SCRREAM samples can build legal 81-frame WAN windows without loading WAN
- [x] submit WAN checkpoint download/preflight job `86267`
- [x] confirm local checkpoint is complete under `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers`
- [x] run 2-sample WAN feature smoke with `WAN_TIMESTEPS=749 WAN_LAYERS=20 WAN_MAX_SAMPLES=2` as job `86282`
- [x] run full WAN feature precompute for timesteps `249,499,749` and code layers `9,14,19,24,29` as jobs `86292`, `86293`, and `86294`
- [x] run WAN feature-sensitivity sanity and zero/sample-shuffle controls as job `86316`
- [x] fix the WAN training final-loss scalar bug exposed by failed pack job `86306`
- [x] launch replacement full WAN Route2 ablation pack as job `86307`
- [x] confirm job `86307` finished all 15 runs: timesteps `249,499,749` x code layers `9,14,19,24,29`
- [x] compare WAN Route2 results against clean-GT VGGT layer `16` and layer `24`
- [x] record best WAN Route2 as `t499/layer09` and interpret the current route as setting-limited rather than claim-level
- [x] implement WAN Route2.1 `no_noise` cache mode and record `requested_timestep_index`, `scheduler_timestep`, `latent_noise_applied`, and `low_noise_index`
- [x] implement WAN Route2.1 `low_noise` cache mode and record the same cache metadata fields
- [x] finish Route2.1 cache chain for layers `9,14,29` under `no_noise` and `low_noise` (`86342/86343 -> 86350/86351`)
- [x] finish Route2.1 smoke/formal adapter runs (`86357`, replacement 1-GPU formal packs `86371` / `86372`)
- [x] finish targeted `t499/layer09 + token_layernorm` setting to test feature-scale / feature-normalization effects (`86395`)
- [x] finish `pair_tiled81` vs `ctx81` audit as jobs `86396 -> 86397 -> 86400 -> 86401`
- [x] conclude simple noise-level, token-normalization, and pair-context-dilution explanations do not explain the WAN gap
- [x] record generator-preserving WAN representation-search plan; VidFM3D is a reference, but this project keeps NOVA/FM as the probe generator
- [x] treat A (`ctx81 normal t499/layer09 block hidden + MLP-L4 -> NOVA/FM`) as the completed baseline, not a new run
- [x] implement tensor-choice audit B: final WAN transformer output / model-output latent (`--feature_kind model_output_latent`, `--wan_feature_kind model_output_latent`) -> `conv2d_mlp` -> NOVA/FM
- [x] implement C: predicted clean latent / x0 estimate -> MLP-L4 -> NOVA/FM (`--feature_kind pred_x0_latent`, `--wan_feature_kind pred_x0_latent`)
- [x] finish pred-x0 full cache job `86411` for all 329 SCRREAM samples; cache files under `t499/pred_x0_latent/` have feature shape `[12480,16]` and metadata fields `sigma`, `x0_formula`, and latent/model-output shapes
- [x] record original pred-x0 MLP readout failure: `86412` failed on PyTorch CUDA adaptive-pooling backward and should not be treated as a result
- [x] implement `conv2d_mlp` pred-x0 readout: restore `[B,2,60,104,16]`, Conv2d stride-2 to `[B,3120,128]`, then pool to `[B,768,128]`
- [x] verify pred-x0 conv smoke job `86421`: loss, robust val metrics, checkpoints, and no CUDA pooling crash
- [x] finish pred-x0 conv formal train job `86422` and compare against old WAN `t499/layer09`; result did not beat old hidden
- [x] implement and run fixed hidden `grid2d_pool` readout (`86423` / `86424`); result did not beat old hidden
- [x] implement and run fixed hidden `grid2d_conv` readout (`86425` / `86426`); result did not beat old hidden or `grid2d_pool`
- [x] implement D adapter: current block hidden -> learned cross-attention / Perceiver-style resampler -> NOVA/FM (`--adapter_type wan_cross_attn_resampler`)
- [x] submit D smoke/formal chain for `ctx81 normal t499/layer09`, adapter layers `2`, hidden dim `512`, heads `8`: smoke `86427`, formal `86428` with `afterok:86427`
- [x] inspect D smoke job `86427` and formal job `86428` metrics: `86428` nearly matches old WAN hidden F@0.10 and improves p90 / Chamfer
- [x] submit D layer sweep job `86429` for `t499 x layers 14,19,24,29`; layer09 is covered by `86428`
- [x] inspect D layer sweep job `86429`: layer14 is the historical WAN peak with `F@0.10=0.5012956284974035`, `pred_to_gt_p90=0.4229188362757365`, `Chamfer=0.10908368105689685`
- [x] inspect CA resampler layer14 `val_visual_40960/step_011500`; user confirmed visual improvement
- [x] submit low-LR stability rerun for CA resampler `t499/layer14` as job `86444` (`WAN_LR=5e-5`, `WAN_VAL_EVERY=250`)
- [x] record that low-LR job `86444` was cancelled because its `128G` memory request blocked scheduling
- [x] inspect replacement low-LR rerun job `86453`: negative result for `t499/layer14` (`F@0.10=0.4292316005720653`, `pred_to_gt_p90=0.5338096991181374`, `Chamfer=0.15617707930505276`)
- [x] inspect CA-resampler full-grid job `86468`: completed all `249/499/749 x 9/14/19/24/29`; best repeated F@0.10 is `t249/layer14=0.48913902331806663`, best repeated p90 / Chamfer is `t249/layer09=0.410525918006897 / 0.11981592203179996`
- [ ] inspect full-grid `val_visual_40960` outputs for `t249/layer09` and `t249/layer14`
- [x] implement multi-layer hidden fusion (`--adapter_type wan_cross_attn_multilayer_resampler`, `--wan_layers`) and Slurm launcher `slurm/scrream_wan_t2v_multilayer_pack_train.sbatch`
- [x] run multi-layer smoke/formal jobs: smoke `86483`, formal `86485` for `t249 layers 9+14`, and formal `86487` for `t249 layers 9+14+19`
- [x] record multi-layer conclusion: `9+14` F@0.10 `0.46487238144059545` and `9+14+19` F@0.10 `0.4724058254433277` did not beat the single-layer CA baselines; stop simple same-timestep layer fusion for now
- [x] implement same-layer multi-timestep fusion (`--adapter_type wan_cross_attn_multitime_resampler`, `--wan_timesteps`) and Slurm launcher `slurm/scrream_wan_t2v_multitime_pack_train.sbatch`
- [x] run multi-timestep smoke job `86493` for `249+499+749/layer14`; it completed with selected feature shape `[1,9360,1536]`
- [x] inspect formal multi-timestep job `86494` for `249+499+749/layer14`: final result `F@0.10=0.45213117002164466`, `pred_to_gt_p90=0.49119475732247037`, `Chamfer=0.12659800921877226`; same-layer multi-timestep fusion is not promising
- [x] implement F: FLF2V first/last-frame hidden route with `pair_endpoint81`, `t249/layer14`, cache script `prepare_scrream_wan_flf2v_feature_cache.py`, and Slurm scripts `scrream_wan_flf2v_{download,precompute,train}.sbatch`
- [x] download/preflight FLF2V checkpoint: initial job `86497` failed on transient HF incomplete-read, retry/resume job `86500` completed; local checkpoint is `checkpoints/wan2.1/Wan2.1-FLF2V-14B-720P-diffusers` (~84G)
- [x] run local FLF2V static/window checks: py_compile, bash syntax, CA-resampler `[1,3120,5120] -> [1,768,128]` and `[1,7200,5120] -> [1,768,128]` dummy backward, plus all 329 `pair_endpoint81` windows
- [x] inspect FLF2V feature smoke / cache shards; final cache has `329/329` files under `scrream_wan_flf2v14b_pair_endpoint81_480/t249/layer14` with shape `[3120,5120]`
- [x] run full FLF2V feature precompute for 329 samples at `t249/layer14` via shard jobs `86515` / `86516` / `86517` / `86518`
- [x] run FLF2V smoke/formal training jobs `86519` / `86520`; formal result `F@0.10=0.42897984457682026`, `pred_to_gt_p90=0.5051772321263949`, `Chamfer=0.14052311765650907`, below T2V CA baselines
- [x] submit tensor-choice audit B Slurm chain: `86534` full `model_output_latent` cache, `86535` one-step smoke, `86536` formal 50-epoch train
- [x] inspect `model_output_latent` cache job `86534`: `329/329` files under `t499/model_output_latent/`, shape `[12480,16]`, metadata `model_output_formula=transformer_output_sample`
- [x] inspect `model_output_latent` smoke job `86535`: loss, robust val metrics, checkpoints, and `val_visual_40960` were written
- [x] inspect `model_output_latent` formal job `86536`; result `F@0.10=0.4199299775478907`, `pred_to_gt_p90=0.6285491685072581`, `Chamfer=0.44791099180777866`, negative versus old hidden MLP and hidden CA baselines
- [x] implement latent learned readout (`WanLatentCrossAttentionResamplerAdapter` / `--adapter_type wan_latent_cross_attn_resampler`) for `pred_x0_latent` and `model_output_latent`
- [x] run local shape/backward smoke for latent CA readout: `[1,12480,16] -> [1,768,128]`, wrong hidden shape raises a clear error
- [x] run Slurm smoke `86539` for `model_output_latent + wan_latent_cross_attn_resampler`; it completed `0:0` and wrote loss / robust val / previews
- [x] inspect formal job `86540` for `model_output_latent + wan_latent_cross_attn_resampler`; result `F@0.10=0.4289946537162989`, `pred_to_gt_p90=0.5942087918519974`, `Chamfer=0.19963246708114943`, still negative versus hidden CA
- [x] inspect replacement Slurm smoke `86542` for `pred_x0_latent + wan_latent_cross_attn_resampler`; it completed `0:0` with loss, robust val, and previews
- [x] submit formal `pred_x0_latent + wan_latent_cross_attn_resampler` as job `86543` with `48G` memory
- [x] inspect formal job `86543` for `pred_x0_latent + wan_latent_cross_attn_resampler`; result `F@0.10=0.43766060222664477`, `pred_to_gt_p90=0.5175856028993925`, `Chamfer=0.20880577837427458`, best latent learned-readout result but still below hidden CA
- [x] stop latent tensor-choice branch for now: `86536`, `86540`, and `86543` are all negative versus hidden CA baselines
- [x] add `--adapter_gated` / `WAN_ADAPTER_GATED` and `WAN_SEED` support for WAN CA stability experiments
- [x] run static checks and local dummy hidden CA gated/ungated forward-backward after adding gated support
- [x] submit hidden CA seed/gated stability jobs `86547`-`86550` for `t249/layer14` and `t499/layer14`
- [x] submit hidden CA L4 capacity jobs `86553`-`86554` for `t249/layer14` and `t499/layer14`
- [x] inspect hidden CA seed23 repeat job `86547` (`t249/layer14`): F@0.10 `0.48706357506233194`, p90 `0.4482837840914726`, Chamfer `0.11202508273224036`
- [x] inspect hidden CA seed23 repeat job `86548` (`t499/layer14`): F@0.10 `0.4591392560862819`, p90 `0.4479780395825704`, Chamfer `0.14107579924166203`
- [x] inspect gated hidden CA job `86549` (`t249/layer14`): F@0.10 `0.4505522726705196`, p90 `0.46173084527254105`, Chamfer `0.11595033543805282`
- [x] inspect gated hidden CA job `86550` (`t499/layer14`): F@0.10 `0.4866296484297818`, p90 `0.45513606319824856`, Chamfer `0.12565729891260466`
- [x] inspect hidden CA L4 capacity job `86553` (`t249/layer14`): F@0.10 `0.4919946248489035`, p90 `0.4724609777331352`, Chamfer `0.11948200377325217`
- [x] inspect hidden CA L4 capacity job `86554` (`t499/layer14`): F@0.10 `0.46280321302050603`, p90 `0.44610939423243207`, Chamfer `0.13269949393967786`
- [x] record conclusion from `86547`-`86554`: hidden CA remains best WAN branch; L4 `t249/layer14` is a small repeated-F improvement, but gated CA and `t499` repeats should not be expanded
- [x] create final WAN summary at `docs/probe/wan_summary_2026-05-19.md`
- [x] pause WAN as the active model-coverage branch after the 2026-05-19 audit
- [ ] optional WAN appendix only: inspect full-grid `t249/layer09`, full-grid `t249/layer14`, and L4 `t249/layer14` `val_visual_40960` outputs side by side
- [ ] optional WAN appendix only: run L4 hidden-CA follow-up `t249/layer09 + adapter_layers=4`
- [ ] optional WAN appendix only: repeat L4 `t249/layer14` with a second seed before calling the L4 gain stable
- [ ] defer low-priority MLP-only capacity variants (`MLP-L6-H1024`, `MLP-L4-H2048`); note the current MLP already uses `GELU`, so this is probably not the root cause

### Next backbone probe
- [ ] choose next frozen backbone to probe after WAN
- [ ] define the next-backbone feature cache schema, preserving native grid / temporal metadata
- [ ] run cache shape / metadata sanity on 2 SCRREAM samples
- [ ] add zero or sample-shuffle control before full training
- [ ] run one-step adapter training smoke with robust val previews
- [ ] start with a small layer/readout pilot before any broad ablation

## Deferred after SCRREAM baseline

### InteriorGS data-quality migration
- [ ] stage a small InteriorGS pilot subset on this server
- [ ] inspect one to three InteriorGS scenes and verify coordinate frames / units
- [ ] build a minimal InteriorGS data bridge for target points or rendered views
- [ ] create a fixed tiny InteriorGS split and visual sanity sheet
- [ ] run an InteriorGS adapter-training smoke only after the data bridge passes visual checks

### ScanNet diagnostic follow-up
- [ ] inspect the current cross-attention candidate's fixed-30 robust eval once it finishes
- [ ] compare cross-attention vs MLP using F@0.05 / precision / representative renders, not symmetric CD alone
- [ ] archive or rename stale phase configs once the next stable experiment plan is chosen

---

## Older TODO history below


## Active now — 2026-04-29 evening

### Server handoff / data pivot
- [ ] push branch `wip/psuvpsc3dd-probe-20260429` to `dongjiacheng06/3dprobe` once the cleanup changes are committed
- [x] remove the stale external automation notes from the active project docs
- [x] document the InteriorGS high-quality dataset migration plan
- [ ] stage a small InteriorGS pilot subset on this server
- [ ] inspect one to three InteriorGS scenes and verify coordinate frames / units
- [ ] build a minimal InteriorGS data bridge for target points or rendered views
- [ ] create a fixed tiny InteriorGS split and visual sanity sheet
- [ ] run an InteriorGS adapter-training smoke only after the data bridge passes visual checks

### Corrected evaluation protocol
- [x] mark old `max_interval=30` ScanNet runs as interval-confounded
- [x] set intended ScanNet interval to `scannet_max_interval=1`
- [x] add `robust_ply_metrics.py`
- [x] add `eval_checkpoint_robust.py`
- [x] evaluate current MLP baseline on fixed 30-sample manifest
- [x] render representative success/failure cases from fixed-30 robust eval
- [x] add fixed-30 failure-case audit
- [ ] inspect the current cross-attention candidate's fixed-30 robust eval once it finishes
- [ ] compare cross-attention vs MLP using F@0.05 / precision / representative renders, not symmetric CD alone

### Candidate experiments
- [x] enable corrected-interval ScanNet controls for cross-attention adapter
- [x] run one-batch cross-attention smoke test
- [x] launch and finish `p7_k2_i1_anchor_ca_l2_h512_chamfer_step1000` (`val_chamfer_l2=0.54222615`)
- [ ] inspect p7 fixed-30 eval/renders only if result artifacts are available locally
- [ ] prioritize InteriorGS data migration before additional ScanNet capacity sweeps

### Code/documentation hygiene
- [x] move temporary DDP helper to `experiments/probe3d/probe_trials/debug/init_distributed_smoke.py`
- [x] add `experiments/probe3d/probe_trials/CURRENT_STATE.md`
- [x] keep `README.md`, `PROJECT.md`, and `docs/probe/*` synchronized for the 2026-04-29 handoff
- [ ] archive or rename stale phase configs once the next stable experiment plan is chosen

---


## Overnight running plan — 2026-04-29 early morning

- [x] switch active feasibility line to K=2 after user hypothesis about target sparsity / empty view slots
- [x] switch adapter capacity to `MLP-L4-H1024`
- [x] launch K=2 `anchor_frustum` loss ablation: `nova_flow`, `flow_chamfer_hybrid@0.05`, `chamfer_sample`
- [x] queue K=2 oracle GT sweep: `nova_input_frustum`, `covered_by_ge2`, `anchor_frustum_margin1.5`, `nova_per_view_frustum_anchor_zpos`, `nova_per_view_ldi{2,4,8}`
- [x] queue K=2 adapter GT/loss sweep for `covered_by_ge2`, `nova_input_frustum`, and `anchor_frustum_margin1.5`
- [ ] 2026-04-29 follow-up: summarize all `p4_` rows from `experiments/probe3d/probe_trials/results.tsv`
- [ ] 2026-04-29 follow-up: inspect PLYs for the best K=2 loss/target candidates
- [ ] 2026-04-29 follow-up: decide which run becomes the proposal-facing feasibility result

## Active now

### Reset to paper-aligned NOVA3R GT / loss
- [x] identify that direct Chamfer can overfit the metric and is not the main representation-probe claim
- [x] clarify NOVA3R target semantics: complete / amodal surface inside selected input-view frustum, not full-room completion
- [x] add explicit `nova_input_frustum` / `nova_anchor_frustum` target aliases
- [x] expose `scannet_complete_points` and `query_source` in the probe harness
- [x] create `phase2_nova_aligned.json` with K=1/K=2 oracle and native-flow adapter probes
- [ ] run K=1 oracle sanity with `nova_input_frustum + src_complete_fps_4096`
- [ ] run K=2 oracle sanity with `nova_input_frustum + src_complete_fps_4096`
- [ ] if oracle support is plausible, run K=1/K=2 MLP-L4 native-flow adapter probes
- [ ] log one-way distances (`pred→GT`, `GT→pred`) during validation
- [ ] compare visual sharpness, not just symmetric CD

### Keep docs and state in sync
- [x] record current probe results in docs
- [ ] record future precision-aware loss results and output paths

## Next after paper-aligned native-flow MLP stabilizes

### Launch CA on the same target/objective
- [ ] reuse the same processed split
- [ ] reuse the same `nova_input_frustum` target
- [ ] reuse the same precision-aware objective
- [ ] keep interpretation baseline-first, not headline-first

### Launch SA on the same target/objective
- [ ] same constraints as CA

## Deferred but still important

### Correct SCRREAM rerun
- [x] official full SCRREAM data root is now available at `~/datasets/SCRREAM`
- [x] reconnect the corrected training path through the mesh-complete bridge
- [ ] rerun adapter branches under valid data assumptions

### Performance optimization
- [ ] optimize online union-frustum crop if dataloader becomes the bottleneck
- [ ] consider deeper reservoir / sample construction optimization only if the formal run shows it is necessary

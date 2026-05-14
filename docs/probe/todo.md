# TODO

## Active now — 2026-05-12

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
- [ ] finish queued Route2.1 cache chain for layers `9,14,29` under `no_noise` and `low_noise` (`86342/86343 -> 86350/86351`)
- [ ] finish queued Route2.1 smoke/formal adapter runs (`86357 -> 86358/86359`)
- [ ] run targeted `t499/layer09 + norm` setting to test feature-scale / feature-normalization effects
- [ ] defer `pair_tiled81` vs `ctx81` until after Route2.1
- [ ] defer I2V / FLF2V conditioning until after Route2.1
- [ ] defer broader normalization / adapter-capacity sweeps until after Route2.1

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

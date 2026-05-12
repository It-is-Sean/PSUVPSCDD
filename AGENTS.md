# Agent Notes

This repo is the active PSUVPSC3DD / NOVA3R probe workspace. Start from these files before changing experiment logic:

- `README.md`
- `PROJECT.md`
- `experiments/probe3d/README.md`
- `docs/probe/handoff_2026-05-07.md`
- `docs/probe/handoff_2026-05-03.md`
- `docs/probe/experiment_plan.md`
- `docs/probe/todo.md`

Current execution rules:

- Use conda env `nova3r`.
- Long data preparation and training jobs run through Slurm scripts in `slurm/`.
- Slurm logs go to `slurm_out/`.
- Do not run full SCRREAM data generation or training directly in an interactive shell.
- Non-WAN network-dependent Slurm jobs should default to `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` at `http://127.0.0.1:7896`.
- WAN Route2 repo/checkpoint jobs use `http://127.0.0.1:17890`; on compute nodes the WAN Slurm scripts create an SSH local tunnel back to `air-server:127.0.0.1:17890`.

Current SCRREAM adapter branch:

- Full SCRREAM data root is `~/datasets/SCRREAM`.
- Old local `eval_scrream` runs are invalid for formal claims.
- Active GT source is `mesh_complete`: read the current sequence `meta.txt`, sample only the listed registered `sceneXX/meshes/*.obj` surfaces proportional to surface area, crop to the two-input-view union frustum, transform to the first input camera frame, then FPS/pad to fixed target points.
- Adapter `.pt` generation script is `experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py`.
- Training entrypoint is `experiments/probe3d/train_vggt_nova_adapter.py --dataset scrream_adapter --data_root <adapter.pt>`.
- The SCRREAM `nova_flow` path applies the NOVA decoder checkpoint target `norm_mode`; current `scene_ae` config reports `norm_mode=median_3`.
- Ignore local `scrream_official_depth_mix_*` adapter artifacts unless the user explicitly asks for a depth-mix ablation.

Current machine state recorded on 2026-05-12 16:32 CST:

- downloaded checkpoints are under `checkpoints/scene_n1/`, `checkpoints/scene_n2/`, `checkpoints/scene_ae/`, `checkpoints/vggt/model.pt`, and `checkpoints/wan2.1/Wan2.1-T2V-1.3B-Diffusers`;
- SwanLab is installed in `nova3r`;
- current branch is `master`;
- `third_party/vggt`, `third_party/Wan2.1`, and `third_party/VidFM3D` are Git submodules; run `git submodule update --init --recursive` after a fresh clone;
- current formal data is `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt` with shape `[329, 20000, 3]`, split `train=317`, `val=12`, and `mesh_sequence_meta_filter=True`;
- old pre-meta-filter `.pt` files are archived under `experiments/probe3d/adapter_data/deprecated_meta_filter_bug/`;
- SCRREAM data bridge status is V0.5: mesh-complete GT from registered SCRREAM meshes is wired end to end for adapter training, but future work should expand training data scale and model coverage;
- robust VGGT layer ablation job `86149` completed successfully on `2026-05-08`, but it used the pre-meta-filter GT and should be treated as historical;
- clean-GT VGGT layer ablation job `86286` completed successfully on `air-node-04` with exit `0:0`, elapsed `02:23:11`; outputs use prefix `experiments/probe3d/result/scrream_mesh_complete_n2_trainplus_test_tp20000_ms500000_mlp_l4_nova_flow_robustval_metafilter_seed17_vggt_layerXX`;
- current clean-GT VGGT default is layer `16` (`best_val_fscore_tau_0.10=0.6860468604251301`, `best_val_pred_to_gt_p90=0.20770130679011345`, `best_val_chamfer_l2=0.026904070439438026`); layer `24` is the main comparison point, and old layer `20` is historical/pre-meta-filter or a weaker clean-GT result;
- visual validation outputs are under `experiments/probe3d/result/*robustval*/val_visual_40960/`;
- `slurm/scrream_mesh_complete_mlp_train.sbatch` supports single-node DDP via `SCRREAM_GPUS_PER_NODE`; `slurm/scrream_vggt_layer_ablation_pack_train.sbatch` is the preferred packed layer-ablation launcher.
- WAN Route2 implementation exists: feature precompute is `experiments/probe3d/scripts/prepare_scrream_wan_t2v_feature_cache.py`, training is `experiments/probe3d/train_wan_t2v_nova_adapter.py`, dependencies are isolated in `experiments/probe3d/requirements-wan-t2v.txt`, and Slurm launchers are `slurm/scrream_wan_t2v_download.sbatch`, `slurm/scrream_wan_t2v_precompute.sbatch`, `slurm/scrream_wan_t2v_ablation_pack_train.sbatch`, and `slurm/scrream_wan_t2v_sanity.sbatch`.
- WAN Route2 keeps the clean SCRREAM mesh-complete `.pt`, split, MLP adapter, NOVA decoder, and robust validation setup from the VGGT ablation, but replaces VGGT features with cached WAN2.1 T2V video-context features from 81-frame windows. The formal grid is timesteps `249,499,749` x code layers `9,14,19,24,29`; checkpoint download, window validation, 2-sample feature smoke, and full feature precompute have completed.
- WAN feature cache root is `experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81`, currently `4937` `.pt` files / about `45G`; official grid files are complete at `329` samples per `(timestep, layer)` plus a 2-file historical smoke cache at `t749/layer20`.
- WAN sanity job `86316` completed; the real WAN features differ materially from zero/sample-shuffle controls, so the cache path is live, but the route is still much weaker than clean-GT VGGT and may be setting-limited.
- WAN training pack job `86306` failed immediately due a reduced-loss float `.item()` bug; `experiments/probe3d/train_wan_t2v_nova_adapter.py` has been patched so `final_loss = reduced_loss`. Replacement job `86307` completed on `air-node-04` with exit `0:0`, elapsed `13:36:04`, `1 x A100`, serial `WAN_PACK_PARALLEL=1`, 50 epochs / `15850` steps per run.
- WAN Route2 best formal run is `t499/layer09` with `best_val_fscore_tau_0.10=0.46988987902779306`, `best_val_pred_to_gt_p90=0.48718947172164917`, and `best_val_chamfer_l2=0.21040735269586244`; this is above zero/sample-shuffle controls but far below clean-GT VGGT layer `16` (`F@0.10=0.6860468604251301`, `pred_to_gt_p90=0.20770130679011345`, `Chamfer=0.026904070439438026`).
- Next WAN setting audit is Route2.1: implement/cache `no_noise` and `low_noise` modes for layers `9,14,29`, then train short probes; also run a targeted `t499/layer09` feature-normalization setting. Defer `pair_tiled81` vs `ctx81`, I2V/FLF2V conditioning, and broader adapter-normalization sweeps until after Route2.1.
- Formal SCRREAM and WAN ablation Slurm defaults now use 50 epochs unless explicitly overridden; completed clean-GT VGGT job `86286` was submitted before that change and remains a 30-epoch-equivalent `9510`-step comparison.

Preserve user changes. The worktree may be dirty with unrelated edits and intentionally deleted retired-automation traces.

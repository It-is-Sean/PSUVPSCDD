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

Current machine state recorded on 2026-05-16 22:26 CST:

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
- WAN Route2 implementation exists: feature precompute is `experiments/probe3d/scripts/prepare_scrream_wan_t2v_feature_cache.py`, training is `experiments/probe3d/train_wan_t2v_nova_adapter.py`, dependencies are isolated in `experiments/probe3d/requirements-wan-t2v.txt`, and Slurm launchers are `slurm/scrream_wan_t2v_download.sbatch`, `slurm/scrream_wan_t2v_precompute.sbatch`, `slurm/scrream_wan_t2v_ablation_pack_train.sbatch`, `slurm/scrream_wan_t2v_route21_pack_train.sbatch`, and `slurm/scrream_wan_t2v_sanity.sbatch`.
- WAN Route2 keeps the clean SCRREAM mesh-complete `.pt`, split, MLP adapter, NOVA decoder, and robust validation setup from the VGGT ablation, but replaces VGGT features with cached WAN2.1 T2V video-context features from 81-frame windows. The formal grid is timesteps `249,499,749` x code layers `9,14,19,24,29`; checkpoint download, window validation, 2-sample feature smoke, and full feature precompute have completed.
- WAN feature cache root is `experiments/probe3d/feature_cache/scrream_wan_t2v1p3b_ctx81`, currently `4937` `.pt` files / about `45G`; official grid files are complete at `329` samples per `(timestep, layer)` plus a 2-file historical smoke cache at `t749/layer20`.
- WAN sanity job `86316` completed; the real WAN features differ materially from zero/sample-shuffle controls, so the cache path is live, but the route is still much weaker than clean-GT VGGT and may be setting-limited.
- WAN training pack job `86306` failed immediately due a reduced-loss float `.item()` bug; `experiments/probe3d/train_wan_t2v_nova_adapter.py` has been patched so `final_loss = reduced_loss`. Replacement job `86307` completed on `air-node-04` with exit `0:0`, elapsed `13:36:04`, `1 x A100`, serial `WAN_PACK_PARALLEL=1`, 50 epochs / `15850` steps per run.
- WAN Route2 best formal run is `t499/layer09` with `best_val_fscore_tau_0.10=0.46988987902779306`, `best_val_pred_to_gt_p90=0.48718947172164917`, and `best_val_chamfer_l2=0.21040735269586244`; this is above zero/sample-shuffle controls but far below clean-GT VGGT layer `16` (`F@0.10=0.6860468604251301`, `pred_to_gt_p90=0.20770130679011345`, `Chamfer=0.026904070439438026`).
- WAN Route2.1 clean / low-noise audit has completed: smoke cache jobs `86342` / `86343`, full cache jobs `86350` / `86351`, smoke train `86357`, and replacement formal packs `86371` / `86372` all exited `0:0`; no-noise and low-noise caches are complete for layers `9,14,29` at `329` samples each.
- Route2.1 did not beat old WAN Route2 best `t499/layer09`: best clean/low-noise result is `no_noise/layer29` with `best_val_fscore_tau_0.10=0.44840173` and `best_val_pred_to_gt_p90=0.48779308`, below old Route2 `F@0.10=0.46988987902779306`.
- `experiments/probe3d/train_wan_t2v_nova_adapter.py` supports `--wan_feature_norm none|token_layernorm|sample_standardize|token_l2`; targeted normalization job `86395` completed with `F@0.10=0.45476213575933216`, `pred_to_gt_p90=0.484821617603302`, and `Chamfer=0.1771892917652925`, so token normalization did not beat old Route2 best.
- `pair_tiled81` vs `ctx81` completed as jobs `86396 -> 86397 -> 86400 -> 86401`; result `F@0.10=0.4429200036613287`, `pred_to_gt_p90=0.5104739194115003`, `Chamfer=0.16478415516515574`, so replacing real video context with tiled pair frames did not fix WAN.
- WAN predicted-x0 / denoised latent probing is implemented: `prepare_scrream_wan_t2v_feature_cache.py --feature_kind pred_x0_latent` writes `t499/pred_x0_latent/<sample>.pt` using `x0 = sample - sigma * model_output`; full cache job `86411` completed with `329/329` samples and feature shape `[12480,16]`.
- Original pred-x0 MLP readout is not a valid completed result: smoke job `86409` failed because the 2-sample cache did not cover the sampled train batch, and formal job `86412` failed on the first backward with a PyTorch CUDA `AdaptiveAveragePooling` shared-memory assert from pooling `[12480,128] -> [768,128]`.
- Current pred-x0 readout fix is `WanPredX0LatentConvAdapter` / `--adapter_type conv2d_mlp`: reshape `[B,12480,16]` to `[B,2,60,104,16]`, Conv2d stride-2 to `[B,3120,128]`, then pool to NOVA scene tokens `[B,768,128]`. `slurm/scrream_wan_t2v_ablation_pack_train.sbatch` exposes this as `WAN_ADAPTER_TYPE=conv2d_mlp`.
- Pred-x0 conv formal job `86422` completed on `air-node-02`, exit `0:0`, elapsed `00:52:13`: `best_val_fscore_tau_0.10=0.41336424426176693`, `best_val_pred_to_gt_p90=0.6272750149170557`, `best_val_chamfer_l2=0.35522504647572833`. It did not beat old WAN hidden `t499/layer09`.
- WAN hidden structured readout audits are implemented in `experiments/probe3d/probe/adapter.py` and `train_wan_t2v_nova_adapter.py`: `--adapter_type grid2d_pool` maps hidden `[B,3120,1536]` as `[B,2,30,52,1536]` to `[B,768,128]` via 2D pooling, and `--adapter_type grid2d_conv` adds a small same-resolution 3x3 Conv2d readout before pooling. Both are restricted to `--wan_feature_kind hidden`.
- Grid readout jobs completed on `air-node-04`: `grid2d_pool` smoke/formal `86423` / `86424` yielded `F@0.10=0.43326753863230877`, `pred_to_gt_p90=0.5651116619507471`, `Chamfer=0.1759416777640581`; `grid2d_conv` smoke/formal `86425` / `86426` yielded `F@0.10=0.39855373112050535`, `pred_to_gt_p90=0.7689011543989182`, `Chamfer=1.159957120815913`. Simple structured pooling/conv does not explain the WAN gap.
- Current recommended WAN next step is a generator-preserving Perceiver / cross-attention resampler on the best WAN hidden setting (`ctx81 normal t499/layer09`) before broader MLP capacity sweeps. The final WAN transformer output / noise-pred-like tensor remains a possible later tensor-choice audit. The current MLP already uses `GELU`.
- Formal SCRREAM and WAN ablation Slurm defaults now use 50 epochs unless explicitly overridden; completed clean-GT VGGT job `86286` was submitted before that change and remains a 30-epoch-equivalent `9510`-step comparison.

Preserve user changes. The worktree may be dirty with unrelated edits and intentionally deleted retired-automation traces.

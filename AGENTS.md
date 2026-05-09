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
- Network-dependent Slurm jobs should default to `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` at `http://127.0.0.1:7896`.

Current SCRREAM adapter branch:

- Full SCRREAM data root is `~/datasets/SCRREAM`.
- Old local `eval_scrream` runs are invalid for formal claims.
- Active GT source is `mesh_complete`: sample registered `sceneXX/meshes/*.obj` surfaces proportional to surface area, crop to the two-input-view union frustum, transform to the first input camera frame, then FPS/pad to fixed target points.
- Adapter `.pt` generation script is `experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py`.
- Training entrypoint is `experiments/probe3d/train_vggt_nova_adapter.py --dataset scrream_adapter --data_root <adapter.pt>`.
- The SCRREAM `nova_flow` path applies the NOVA decoder checkpoint target `norm_mode`; current `scene_ae` config reports `norm_mode=median_3`.
- Ignore local `scrream_official_depth_mix_*` adapter artifacts unless the user explicitly asks for a depth-mix ablation.

Current machine state recorded on 2026-05-09 23:10 CST:

- downloaded checkpoints are under `checkpoints/scene_n1/`, `checkpoints/scene_n2/`, `checkpoints/scene_ae/`, and `checkpoints/vggt/model.pt`;
- SwanLab is installed in `nova3r`;
- current branch is `senior-vggt-submodule`;
- `third_party/vggt` and `third_party/Wan2.1` are Git submodules; run `git submodule update --init --recursive` after a fresh clone;
- current formal data is `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt` with shape `[329, 20000, 3]`, split `train=317`, `val=12`;
- SCRREAM data bridge status is V0.5: mesh-complete GT from registered SCRREAM meshes is wired end to end for adapter training, but future work should expand training data scale and model coverage;
- robust VGGT layer ablation job `86149` completed successfully on `2026-05-08`; best current probe layer is VGGT layer `20` with `val_fscore_tau_0.10=0.672747712053826`, `val_pred_to_gt_p90=0.23175348962346712`, and `val_chamfer_l2=0.03473521831134955`;
- layer `16` has the lowest prediction-side p90 outlier distance (`val_pred_to_gt_p90=0.20785426969329515`) and is the main qualitative comparison point against layer `20`;
- visual validation outputs are under `experiments/probe3d/result/*robustval*/val_visual_40960/step_009500/`;
- `slurm/scrream_mesh_complete_mlp_train.sbatch` supports single-node DDP via `SCRREAM_GPUS_PER_NODE`; `slurm/scrream_vggt_layer_ablation_pack_train.sbatch` is the preferred packed layer-ablation launcher.

Preserve user changes. The worktree may be dirty with unrelated edits and intentionally deleted retired-automation traces.

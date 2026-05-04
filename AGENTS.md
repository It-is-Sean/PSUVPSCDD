# Agent Notes

This repo is the active PSUVPSC3DD / NOVA3R probe workspace. Start from these files before changing experiment logic:

- `README.md`
- `PROJECT.md`
- `experiments/probe3d/README.md`
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
- Active GT source is `mesh_complete`: sample registered `sceneXX/meshes/*.obj` surfaces proportional to surface area, crop to the two-input-view union frustum, transform to the first input camera frame, then FPS/pad to `10000` target points.
- Adapter `.pt` generation script is `experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py`.
- Training entrypoint is `experiments/probe3d/train_vggt_nova_adapter.py --dataset scrream_adapter --data_root <adapter.pt>`.
- The SCRREAM `nova_flow` path applies the NOVA decoder checkpoint target `norm_mode`; current `scene_ae` config reports `norm_mode=median_3`.

Machine state recorded on 2026-05-03 02:26 CST:

- downloaded checkpoints are under `checkpoints/scene_n1/`, `checkpoints/scene_n2/`, `checkpoints/scene_ae/`, and `checkpoints/vggt/model.pt`;
- SwanLab is installed in `nova3r`;
- full prep job `85773` was running as `scrream_mesh_prep`;
- dependent MLP job `85774` was pending as `scrream_mesh_mlp`;
- full adapter output `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.pt` had not yet been written.

Preserve user changes. The worktree may be dirty with unrelated edits and intentionally deleted retired-automation traces.

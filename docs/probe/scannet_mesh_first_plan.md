# ScanNet v2 mesh-first training plan (2026-04-28)

This document records the currently agreed **formal ScanNet v2 extension line**.

## Interpretation

This branch is:

- a **NOVA3R-style extension / transfer probe** on ScanNet v2
- not a literal reproduction of official NOVA3R scene training on `3D-FRONT + ScanNet++V2`

Still, it is now serious enough to serve as the proposal’s **reliable-target baseline**.

## Locked decisions

### Data source
- raw dataset root: `/data1/jcd_data/scannerv2_paraell_w48`
- train = official `scans`
- test = official `scans_test`
- val = 10% split carved from official train

### Geometry source
- formal mesh source: `vh_clean.ply`
- do not use `vh_clean_2.ply` as the main geometry source

### Frame extraction
- formal frame sampling rate: `frame_skip=20`

### GT definition
For each sample:

1. start from a dense mesh-sampled scene reservoir
2. keep only points lying inside the **union frustum** of the sparse input views
3. transform them to the **first input view coordinate frame**
4. use the resulting points as complete GT

No extra visible/occluded auxiliary labels are added in this version.
No extra depth-based clipping heuristic is used in this version.

### Reservoir density
- `500k` points / scene
- measured compressed cache size: about `5.2 MB / scene`
- full-scale cache size: about `8–9 GB`

## Processed dataset status

### Full preprocess
Completed successfully.

- processed root:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k`
- split root:
  - `/data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17`

### Split sizes
- `train = 1362 scenes`
- `val = 151 scenes`
- `test = 100 scenes`

### Sample counts with the current 4-view loader
- `train = 105259`
- `val = 12121`
- `test = 10199`

## Implementation status

### Already implemented
- true 4-view input wiring for ScanNet batches
- automatic loading of `pts3d_complete`
- scene-level mesh reservoir builder
- DDP / `torchrun` launchers for MLP / CA / SA
- complete-GT training path validated by smoke runs

### Key files
- `dust3r/datasets/scannet.py`
- `experiments/probe3d/build_scannet_complete_gt.py`
- `experiments/probe3d/prepare_scannet_large.py`
- `experiments/probe3d/train_vggt_nova_adapter.py`
- `experiments/probe3d/train_vggt_nova_cross_attention_adapter.py`
- `experiments/probe3d/train_vggt_nova_self_attention_adapter.py`

## Preflight results

### Data integrity
- all train/test scenes have `vh_clean.ply`
- full preprocess finished without failures
- `pts3d_complete` is present across sampled train/val/test batches

### Mesh / pose alignment
- sampled scene checks indicate `vh_clean.ply` and `.sens` poses are aligned well enough
- no obvious global coordinate mismatch was observed

### Runtime considerations
- online union-frustum crop on one `500k` reservoir scene is about `0.58 s / sample`
- this means dataloader workers matter for the formal multi-GPU run
- not a blocker, but something to monitor

## Validated smoke milestones

### Complete-GT smoke
- output dir:
  - `experiments/probe3d/result/scannet_mlp_complete_smoke_seed17`
- SwanLab run id:
  - `30iyh29o1orq7mnm097u4`

### Full-root DDP preflight
- output dir:
  - `experiments/probe3d/result/scannet_mlp_ddp_fullroot_preflight_seed17`
- SwanLab run id:
  - `f7nivqbjlilypp5o9u4sz`

## Formal MLP run

The first formal run on this branch is the **MLP baseline**.

### Role in the proposal
This run is intended to answer:

> can frozen VGGT features + a lightweight MLP adapter + a NOVA3R-style decoder learn stable complete 3D reconstruction behavior under reliable mesh-first supervision at scale?

It is a **baseline for feasibility**, not the final headline method.

### Launch schedule
- 8 GPUs
- batch size `1 / rank`
- 50 epochs
- `steps_per_epoch = 13158`
- `max_steps = 657900`
- validation every 5 epochs = `65790` steps
- checkpoint every 5 epochs = `65790` steps

### Formal output dir
- `experiments/probe3d/result/scannet_mlp_adapter_l4_lr5e-5_seed17_epoch50_formal`

## Next steps after MLP baseline

1. verify the formal MLP run starts and remains stable
2. use the same processed split + DDP stack for CA / SA comparisons
3. keep the corrected full-data SCRREAM rerun as a separate branch once that data is available

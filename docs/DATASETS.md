# Dataset Guide

## Download

```bash
# Download dataset
bash scripts/download_datasets.sh

# Or download individually
bash scripts/download_datasets.sh --dataset scrream
```

Datasets are downloaded to `datasets/` by default. Use `--output /path/to/dir` for a custom location.

## Datasets

Both datasets are hosted on [HuggingFace](https://huggingface.co/datasets/ruili3/LaRI_dataset) (same source as [LaRI](https://ruili3.github.io/lari/)):

| Dataset | Type | Download Size | Objects/Scenes |
|---------|------|---------------|----------------|
| **SCRREAM** | Scene-level | ~1.5 GB | 11 scenes |

Important distinction:

- `datasets/eval_scrream` is the released evaluation subset and must not be used as formal training data for the SCRREAM adapter branch.
- The current full SCRREAM training tree used by the adapter work is staged locally at `~/datasets/SCRREAM`.
- For the full tree, use `experiments/probe3d/scripts/prepare_scrream_full_adapter_data.py`; do not use the legacy LDI prep script unless the data explicitly contains `ldi/` and `*_ldi.npz`.
- The current primary full-SCRREAM target source is `--target_source mesh_complete`, using registered `sceneXX/meshes/*.obj` assets cropped to the selected input-pair frustum.
- Full mesh-complete adapter generation was submitted as Slurm job `85773` on 2026-05-03. The expected full output is `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17.pt` plus `.manifest.json`; at 2026-05-03 02:26 CST those files were not yet written.

### Directory structure after download

```
datasets/
└── eval_scrream/
    ├── scene01/
    │   └── scene01_full_00/
    │       ├── rgb/                    # RGB images
    │       ├── camera_pose/            # Per-frame camera poses
    │       └── intrinsics.txt          # Camera intrinsics
    └── ... (11 scenes)
```

## Acknowledgments

The evaluation datasets are hosted and distributed by the [LaRI](https://ruili3.github.io/lari/) project. We thank the LaRI team for making these datasets publicly available.

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

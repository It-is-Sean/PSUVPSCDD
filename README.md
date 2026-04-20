<p align="center">
  <img src="assets/nova3r_logo.png" alt="NOVA3R logo" width="224">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.04179"><img src="https://img.shields.io/badge/arXiv-2603.04179-b31b1b.svg" alt="arXiv"></a>
  <a href="https://wrchen530.github.io/nova3r/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
</p>

# NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction

**[ICLR 2026]** The repository contains the official implementation of [NOVA3R](https://wrchen530.github.io/nova3r/). Given unposed multi-view images, **NOVA3R** recovers complete, non-overlapping 3D geometry, reconstructing visible and occluded regions with physical plausibility.


> **NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction**<br> 
> [Weirong Chen](https://wrchen530.github.io/), [Chuanxia Zheng](https://physicalvision.github.io/people/~chuanxia), [Ganlin Zhang](https://ganlinzhang.xyz/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), [Daniel Cremers](https://cvg.cit.tum.de/members/cremers) <br> 
> ICLR 2026

**[[Paper](https://arxiv.org/abs/2603.04179)] [[Project Page](https://wrchen530.github.io/nova3r/)]**

![NOVA3R teaser](assets/nova3r_teaser.gif)

## Requirements

- **Python**: 3.10
- **PyTorch**: 2.2+ with CUDA 12.1+
- **GPU**: NVIDIA GPU with ≥24GB VRAM (48GB recommended). Evaluated on NVIDIA L40s GPU.

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/wrchen530/nova3r.git
cd nova3r

# Automated setup
bash setup.sh

# Download checkpoints
bash scripts/download_checkpoints.sh
```

See [docs/INSTALL.md](docs/INSTALL.md) for manual installation.

## Demo

Run 3D reconstruction on your own images:

```bash
conda activate nova3r

# Single image (scene-level)
python demo_nova3r.py \
  --images demo/examples/scene_1.png \
  --ckpt checkpoints/scene_n1/checkpoint-last.pth

# Two images (multi-view, scene-level)
python demo_nova3r.py \
  --images demo/examples/scrream_scene09_200.png demo/examples/scrream_scene09_275.png \
  --ckpt checkpoints/scene_n2/checkpoint-last.pth
```

Output `.ply` point clouds and `.mp4` 360° videos are saved to `demo/outputs/<image_name>/` (configurable with `--output_dir`).

### Point Cloud AE

Reconstruct a point cloud using the Stage 1 point-conditioned autoencoder:

```bash
# Point cloud autoencoding from a SCRREAM scene
python demo_nova3r_ae.py \
  --input_ply demo/examples/scrream_scene09.ply \
  --ckpt checkpoints/scene_ae/checkpoint-last.pth \
  --num_queries 50000
```

### Python API

```python
from demo_nova3r import predict

# Single image → 3D point cloud
pts3d = predict(
    ckpt_path="checkpoints/scene_n1/checkpoint-last.pth",
    image_paths=["path/to/image.png"],
    resolution=(518, 392),
    output_path="output.ply",
)
# pts3d is a numpy array of shape (N, 3)
```

## Checkpoints

Download all checkpoints:
```bash
bash scripts/download_checkpoints.sh
```

| Model | Training Dataset | Input | Checkpoint | Size |
|-------|---------|-------|------------|------|
| Pts2Pts (AE) | 3DFront + Scannetpp | point cloud | `checkpoints/scene_ae/checkpoint-last.pth` | 262 MB |
| Img2Pts (N=1) | 3DFront + Scannetpp | 1 image | `checkpoints/scene_n1/checkpoint-last.pth` | 5.8 GB |
| Img2Pts (N=2) | 3DFront + Scannetpp | 2 images | `checkpoints/scene_n2/checkpoint-last.pth` | 5.8 GB |

Checkpoints are hosted on [HuggingFace](https://huggingface.co/wrchen530/nova3r).

## Evaluation

Reproduce benchmark results:

```bash
# Download datasets
bash scripts/download_datasets.sh

# SCRREAM evaluation (1-view / 2-view)
bash scripts/eval/eval_scrream_n1_stage2.sh --data_root /path/to/datasets
bash scripts/eval/eval_scrream_n2_stage2.sh --data_root /path/to/datasets
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for detailed instructions.




## BibTeX

If you find NOVA3R useful for your research and applications, please cite us using this BibTex:

```bibtex
@inproceedings{chennova3r,
  title={NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction},
  author={Chen, Weirong and Zheng, Chuanxia and Zhang, Ganlin and Vedaldi, Andrea and Cremers, Daniel},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for full terms. Third-party code (e.g., [DUSt3R](https://github.com/naver/dust3r), [CroCo](https://github.com/naver/croco), [VGGT](https://huggingface.co/facebook/VGGT-1B), [TripoSG](https://github.com/VAST-AI-Research/TripoSG)) retains its original license.


## Acknowledgments

We build on prior advances in multi-view 3D reconstruction, global scene representations, and flow-based generative models. Our codebase utilizes code from [VGGT](https://huggingface.co/facebook/VGGT-1B), [DUSt3R](https://github.com/naver/dust3r), [TripoSG](https://github.com/VAST-AI-Research/TripoSG), and [LaRI](https://ruili3.github.io/lari/). We sincerely appreciate the authors for their wonderful work and for releasing their code and data processing scripts.

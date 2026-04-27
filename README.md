<p align="center">
  <img src="assets/nova3r_logo.png" alt="NOVA3R logo" width="224">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.04179"><img src="https://img.shields.io/badge/arXiv-2603.04179-b31b1b.svg" alt="arXiv"></a>
  <a href="https://wrchen530.github.io/nova3r/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
</p>

# PSUVPSC3DD / NOVA3R integrated research repo

This repo is the merged working tree for two parallel attempts:

1. the original `PSUVPSC3DD_repo` branch centered on `experiments/probe3d/`
2. the separate `probe` research fork that added proposal docs, configs, probe modules, and launch scaffolding

## Current experimental status

The current story is simpler than the full proposal scope:

- on **SCREAM**, we have already trained **2-layer and 4-layer MLP adapters**
- the result is promising enough to support **initial feasibility**, with the best run reaching roughly **CD ≈ 0.18**
- a **4-layer attention adapter** has also been tested, but the result is currently **not as good / not as convincing**
- the next real milestone is to move to **ScanNet v2** and test whether the same idea still works on a larger dataset

So the repo should currently be read as a **feasibility-stage research workspace**, not a finished full-scope benchmark platform.

## What to read first

If you only want the essential project state, read:

1. `PROPOSAL.md` — core idea
2. `PROJECT.md` — actual current status and next step
3. `experiments/probe3d/README.md` — the path where the real experiments currently live

Supporting notes under `docs/probe/` are secondary.

## Repo reality

There are two layers in this repo:

- a **cleaner scaffold**: `configs/probe/`, `scripts/probe/`, `nova3r/probe/`
- a **more executable experimental path**: `experiments/probe3d/`

Right now the real run history is mostly under `experiments/probe3d/`.

## Key layout

- `PROJECT.md` — project memory, decisions, current status
- `PROPOSAL.md` — proposal copy for the shared complete-3D decoding direction
- `experiments/probe3d/` — current main experiment path
- `docs/probe/` — supporting notes and planning docs
- `third_party/vggt/` — vendored VGGT code
- `dust3r/datasets/` — vendored DUSt3R dataset loaders
- `datasets_preprocess/` — vendored preprocessing helpers including ScanNet scripts

## Quick start

```bash
cd /home/jcd/PSUVPSC3DD_repo

# create / update the shared conda env
make probe-env

# verify critical imports and CUDA
make probe-env-verify
```

## Research direction right now

The practical next step is:

- keep the first paper path focused on **image / geometry backbones**
- treat **video** as a later extension
- prove feasibility more convincingly on **ScanNet v2**

## License

This project is licensed under the Apache License 2.0. Third-party code keeps its original license terms, including vendored code under `third_party/vggt/`.

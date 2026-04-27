# Shared Complete-3D Decoding Probe (NOVA3R research fork)

- Upstream repo: `https://github.com/wrchen530/nova3r`
- Local path: `/home/jcd/PSUVPSC3DD_repo`
- Working tree: integrated local merge of the original `PSUVPSC3DD_repo` and the separate `probe` research fork
- Vendored external dependency: `third_party/vggt`

## Goal

Use NOVA3R's Stage-1 latent-to-point decoder family as the **shared canonical decoder**, then test the proposal:

> Probing Image and Video Foundation Representations via Shared Complete-3D Decoding

The key comparison is still:

1. **Shared decoding probe**: frozen representation -> lightweight adapter -> shared complete-3D decoder
2. **Direct 3D readout baseline**: frozen representation -> shallow point/depth/pose heads

## Current status

The real experimental status right now is:

- on **SCREAM**, we have already trained **2-layer and 4-layer MLP adapters**
- those runs are good enough to support **initial feasibility** of the proposal on a smaller dataset
- the current best result is roughly **CD ≈ 0.18**
- a **4-layer attention adapter** has also been tested, but its effect is currently **not as convincing** as the MLP branch

So the current conclusion is not “the full proposal is solved,” but:

- the basic idea appears **viable on SCREAM**
- **MLP adapters currently look like the better working direction**
- the next important question is whether this still holds on a **larger dataset**

## Current repo interpretation

The repo has two layers:

- a cleaner research scaffold: `configs/probe/`, `scripts/probe/`, `nova3r/probe/`
- the more executable experimental path: `experiments/probe3d/`

At the moment, the **actual run history is mostly under `experiments/probe3d/`**.

## What has already been set up

### Repo / infra

- the separate probe workspace has been merged into this repo
- `third_party/vggt/` has been vendored in
- `dust3r/datasets/` and `datasets_preprocess/` have been copied in so the experiment path is less dependent on external checkouts
- `Makefile`, environment checks, and supporting docs are in place

### Decoder / backbone wiring

- the NOVA3R Stage-1 decoder family is available as the canonical decoder base
- VGGT code and frozen-feature extraction support have been wired in
- the repo can support the current SCRREAM experiments and the next scaling step

## Main strategic reading

Even though the proposal mentions image / geometry / video together, the practical first-paper path should stay narrow:

- first prove the idea well on **image / geometry backbones**
- avoid widening to video too early
- prioritize a clear **feasibility result on ScanNet v2**

## Immediate next step

The next main experiment should be:

1. move from **SCREAM** to **ScanNet v2**
2. train/test the current best adapter direction there
3. use that result to argue that the proposal is not just a toy small-dataset effect, but a plausible research direction

## Optional supporting files

- `PROPOSAL.md` — core proposal text
- `experiments/probe3d/README.md` — where the current real experiments live
- `docs/probe/experiment_history.md` — detailed log-based history summary

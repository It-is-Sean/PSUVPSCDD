# Experiment plan

## Phase 0 — Integrated workspace setup (done)

- import the upstream NOVA3R base into the current repo
- merge the separate proposal/probe attempt into the same working tree
- vendor the minimum external code needed for the first probe loop (`third_party/vggt/`, `dust3r/datasets/`, `datasets_preprocess/`)
- scaffold docs / configs / probe package / launch scripts

## Phase 1 — Canonical decoder inventory

Deliverables:
- identify the exact Stage-1 checkpoint(s) to freeze
- document the latent token shape (`N_s`, `d`)
- document query budget (`N_q`) and integration schedule
- confirm whether the existing encoder must be reused or only the decoder family

## Phase 2 — Unified feature extraction interface

Deliverables:
- one common representation container for all backbone families
- layer-selection protocol for image / geometry models
- limited timestep protocol for video models
- padding / masking conventions for variable token length

## Phase 3 — Probe heads

Deliverables:
- small scene-token adapter
- direct baseline A (linear / MLP)
- optional stronger direct baseline B for robustness only
- minimal metric suite for shared decoding + direct outputs

## Phase 4 — Phase-1 reliable target experiments

Deliverables:
- pick reliable complete-3D targets
- implement visible / unseen region split
- run small-adapter sanity experiments
- produce the first within-family rankings

## Phase 5 — Video extensions (only after image/geometry path is credible)

Deliverables:
- timestep sweep in a limited early/mid window
- within-video-family comparison
- first cross-family comparison with explicit caveats

Scope discipline note:
- the proposal is intentionally broad, but the executable first paper path should probably stabilize on **image / geometry backbones first**
- video should be treated as a second-wave extension unless the shared-decoder protocol is already convincing on the image/geometry side

## Phase 6 — Paper-facing outputs

Deliverables:
- main tables
- low-view / unseen-region degradation plots
- layer/timestep plots
- concise claim-to-figure mapping

## Open decisions

- Which datasets count as Phase-1 reliable targets?
- Do we reuse the existing Stage-1 encoder exactly, or only the decoder family?
- What is the minimal adapter width / depth that still trains stably?
- Which backbones are realistic for the first month of implementation?

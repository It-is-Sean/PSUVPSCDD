# Experiment plan

## Phase 0 — Branch setup (done)

- clone upstream NOVA3R into `projects/`
- create a dedicated proposal branch
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

## Phase 5 — Video extensions

Deliverables:
- timestep sweep in a limited early/mid window
- within-video-family comparison
- first cross-family comparison with explicit caveats

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

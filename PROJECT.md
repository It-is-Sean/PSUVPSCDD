# Shared Complete-3D Decoding Probe (NOVA3R research fork)

- Upstream repo: `https://github.com/wrchen530/nova3r`
- Local path: `/home/jcd/PSUVPSC3DD_repo`
- Working tree: integrated local merge of the original `PSUVPSC3DD_repo` and the separate `probe` research fork
- Vendored external dependency: `third_party/vggt`

## Goal

Use NOVA3R's Stage-1 latent-to-point decoder family as the **shared canonical decoder**, then build a research fork for the proposal:

> Probing Image and Video Foundation Representations via Shared Complete-3D Decoding

The fork compares two probe signals under a unified protocol:

1. **Shared decoding probe**: frozen representation -> lightweight scene-token adapter -> shared complete-3D decoder.
2. **Direct 3D readout baseline**: frozen representation -> shallow point/depth/pose heads.

## Workspace map

- `docs/probe/` — proposal digest, method mapping, experiment plan, TODOs
- `configs/probe/` — stage configs, model family lists, stress-test/sweep settings
- `nova3r/probe/` — proposal-specific probe modules (adapter, direct readout, metrics, registries)
- `scripts/probe/` — launch / planning scripts, sweeps, sanity runs, evaluation scaffolding
- `experiments/` — run templates and logging conventions
- `artifacts/` — generated manifests, tables, figures, reports, checkpoints (runtime outputs)
- `data/probe/` — dataset protocol notes for Phase 1 / Phase 2

## Key design decisions

- Keep the **canonical decoder decoder-conditional** and anchored to NOVA3R Stage 1 instead of redesigning a new decoder family.
- Treat this branch as a **research fork**, not an upstream-clean minimal patch. Documentation and experiment scaffolding are first-class.
- Keep the adapter in a **small-adapter regime** by default, so the probe still measures representation quality instead of adapter capacity.
- Compare **within-family rankings** and **cross-family trends**, but do not over-interpret a single global leaderboard.
- For project setup, prefer extending an obvious upstream repo on a dedicated branch instead of starting from an empty repo skeleton.

## Current status memory

### Repo / scaffold

- Upstream `nova3r` has been cloned and converted into the working research fork.
- The research workspace scaffold is in place: `docs/probe/`, `configs/probe/`, `scripts/probe/`, `experiments/`, `artifacts/`, `data/probe/`, and `nova3r/probe/`.
- The collaborator-side `experiments/probe3d/` path has been preserved inside the same repo instead of living in a separate checkout.
- `Makefile` targets exist for workspace prep, planning, sweeps, sanity runs, and run visualization.
- A reusable visualization workflow now exists via `scripts/probe/visualize_run.py`, which can materialize `preview.png`, `turntable.(mp4|gif)`, and `visualization_manifest.json` from a probe run directory.

### Canonical decoder side

- The NOVA3R Stage-1 AE checkpoint has been downloaded locally:
  - `checkpoints/scene_ae/checkpoint-last.pth`
  - `checkpoints/scene_ae/.hydra/config.yaml`
- The Stage-1 decoder has been extracted into a callable frozen wrapper:
  - `nova3r/probe/canonical_decoder.py`
- This wrapper loads only the `pts3d_head` weights from the AE checkpoint and exposes a minimal interface:
  - `scene tokens -> decoder step`
  - `scene tokens -> Euler sampling -> point cloud`
- The decoder interface has already been smoke-tested with random tokens.

### VGGT side

- The official VGGT code has been vendored locally under `third_party/vggt`.
- A VGGT frozen-feature extractor has been added:
  - `nova3r/probe/backbones/vggt_extractor.py`
- A temporary **training-free bridge** has been added for the first sanity pass:
  - `nova3r/probe/bridges.py`
- The first cross-model sanity script has been added:
  - `scripts/probe/run_vggt_to_nova3r_decoder.py`

### What has not been finished yet

- The current VGGT -> NOVA3R path is only a **sanity bridge**, not the final learned small adapter from the proposal.
- The last sanity run was interrupted during / around VGGT pretrained weight download and should be re-run from the new project path.
- No claim should be made yet from the current cross-model output until the sanity path finishes and the geometry is inspected.

## Immediate next steps

1. Re-run the `VGGT final-layer features -> frozen NOVA3R decoder` sanity pass from the integrated repo root.
2. Check whether the resulting point cloud is non-trivial / stable enough to justify continuing this route.
3. Replace the current training-free bridge with the intended small learned adapter.
4. Add a NOVA3R-native feature path as an internal control.
5. Start measuring visible vs unseen-region behavior once the first runnable path is stable.

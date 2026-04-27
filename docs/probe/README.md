# Probe workspace overview

This branch turns NOVA3R into the base repository for the proposal **Shared Complete-3D Decoding Probe**.

## What this fork adds

- a proposal-oriented docs layer (`docs/probe/`)
- a reproducible config layer (`configs/probe/`)
- a new probe package (`nova3r/probe/`) for lightweight adapters, direct readout, metrics, and backbone registries
- run / sweep / evaluation launcher stubs (`scripts/probe/`)
- experiment logging templates (`experiments/`)

## Conceptual mapping

- **Stage 1** uses NOVA3R's point-latent interface as the shared canonical decoder family.
- **Stage 2** adds a lightweight scene-token adapter from frozen representations to canonical scene tokens.
- **Stage 3** compares shared complete-3D decoding against a shallow direct readout baseline.

## Repository conventions for this fork

- Keep proposal-specific code under `nova3r/probe/` instead of scattering it through upstream modules.
- Use `configs/probe/` as the single source of truth for model families, data phases, budgets, and sweeps.
- Put generated outputs in `artifacts/` rather than committing them.
- Log every substantial experiment with `experiments/templates/run_card.md`.

## Visualization workflow

Probe runs can now be rendered through a reusable visualization entrypoint:

```bash
# Latest run under artifacts/reports/vggt_to_nova3r/
make probe-viz

# Explicit run directory
make probe-viz ARGS="--run-dir artifacts/reports/vggt_to_nova3r/20260424-092748"
```

This workflow:

- resolves a probe run directory
- loads `pointcloud.npy` / `pointcloud.ply`
- writes `visualization/preview.png`
- writes `visualization/turntable.mp4` or `visualization/turntable.gif`
- records rendering settings in `visualization/visualization_manifest.json`
- falls back to a lightweight matplotlib renderer when the heavier PyTorch3D stack is unavailable

Implementation entrypoint:

- `scripts/probe/visualize_run.py`

Reusable rendering utilities remain in:

- `demo/visualization/render_points.py`

## Current state

The workspace is ready for research planning and incremental implementation.
The launcher scripts currently materialize run manifests / sweep plans rather than launching full training. That is deliberate: the method framing is in place, but actual backbone wiring and trainer loops still need to be filled in.

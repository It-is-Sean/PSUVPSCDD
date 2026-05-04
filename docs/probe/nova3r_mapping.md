# Mapping the proposal onto the upstream NOVA3R repository

## Reused upstream pieces

### Stage 1: canonical latent-to-point interface

Proposal role | Upstream code
--- | ---
point-latent encoder | `nova3r/heads/pts3d_encoder/transformer_encoder.py`
latent-to-point decoder | `nova3r/heads/pts3d_decoder/flowm_decoder_point_joint_v2.py`
point-conditioned model wrapper | `nova3r/models/nova3r_pts_cond.py`
AE demo / checkpoint loading | `demo_nova3r_ae.py`
flow matching solver utilities | `nova3r/flow_matching/`

### Existing image-conditioned pieces that may inspire feature hooks

Proposal need | Existing code
--- | ---
scene token aggregation reference | `nova3r/models/aggregator_pts3d.py`
image-conditioned model | `nova3r/models/nova3r_img_cond.py`
inference / evaluation helpers | `nova3r/inference.py`

## New fork-specific additions

### Stage 2: adapter layer

- `nova3r/probe/adapters.py`
- `nova3r/probe/backbones/`
- `configs/probe/stage2/shared_adapter.yaml`

### Stage 3: direct baseline + evaluation

- `nova3r/probe/direct_readout.py`
- `nova3r/probe/metrics.py`
- `configs/probe/stage3/direct_baseline.yaml`
- `configs/probe/stress/low_view_unseen.yaml`

### Planning / experiment management

- `PROJECT.md`
- `docs/probe/experiment_plan.md`
- `docs/probe/todo.md`
- `scripts/probe/`
- `experiments/`

## Recommended implementation order

1. Verify how Stage-1 NOVA3R checkpoints are loaded and frozen.
2. Expose a clean decoder-facing interface: `scene_tokens -> complete point cloud`.
3. Build the small adapter around that interface.
4. Add direct readout heads on the same frozen representations.
5. Wire model-family-specific feature extractors one by one.

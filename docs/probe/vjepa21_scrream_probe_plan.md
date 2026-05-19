# V-JEPA 2.1 SCRREAM Probe Plan

## Goal

Evaluate whether frozen V-JEPA 2.1 encoder features can drive the current NOVA decoder on SCRREAM under the same clean-GT probe protocol used by the current VGGT and WAN lines.

The question is:

`f0, f1 -> video bridge -> frozen V-JEPA 2.1 encoder tokens -> current MLP adapter -> frozen NOVA decoder`

Can this path learn useful 3D reconstruction on the current SCRREAM setup?

## Fixed Control Variables

These must remain unchanged in the first-stage comparison:

- adapter data:
  - `experiments/probe3d/adapter_data/scrream_mesh_complete_n2_adapter_seed17_tp20000_ms500000_trainplus_test.pt`
- split:
  - `train=317`, `val=12`
- GT semantics:
  - sequence-meta-filtered SCRREAM mesh-complete target
  - two-view union-frustum crop
  - first input camera frame
- target points:
  - `20000`
- decoder checkpoint:
  - `checkpoints/scene_ae/checkpoint-last.pth`
- decoder:
  - same frozen NOVA decoder
- adapter:
  - current `MLP` adapter only
- loss:
  - `nova_flow`
- queries:
  - `20000`
- validation:
  - same robust metrics
  - same `val_visual_40960`

## Representation Change Only

The only change is the representation source:

- VGGT:
  - online two-frame visual features
- WAN:
  - offline cached video-context features from a diffusion model internal state
- V-JEPA 2.1:
  - offline cached frozen encoder tokens

The V-JEPA 2.1 branch therefore follows the WAN engineering pattern, but its feature semantics should stay closer to the VGGT branch.

## First-Stage Conditions

Run these four conditions:

- `V-JEPA2.1 pair_exact16 -> MLP -> NOVA`
- `V-JEPA2.1 ctx_anchor16 -> MLP -> NOVA`
- `V-JEPA2.1 ctx_shuffle16 -> MLP -> NOVA`
- `V-JEPA2.1 ctx_anchor32 -> MLP -> NOVA`

Interpretation:

- `pair_exact16` vs `ctx_anchor16`
  - whether real context helps
- `ctx_anchor16` vs `ctx_shuffle16`
  - whether ordered temporal context helps beyond extra frames
- `ctx_anchor16` vs `ctx_anchor32`
  - whether longer context continues to help

## Window Definition

The V-JEPA branch should use pair-local context directly.

There is no requirement to inherit WAN's `ctx81` logical window.

Instead:

1. build a pair-local clip directly from the source sequence
2. run the frozen encoder on that clip
3. keep only the temporal tubelet slices relevant to the pair

## Precise Clip Modes

### `pair_exact16`

- no real context
- construct `16` frames exactly as:
  - `8 x f0 + 8 x f1`

### `ctx_anchor16`

- build a pair-local `16`-frame clip directly from the source sequence
- use `8` frames ending at `f0`, then `8` frames starting at `f1`
- keep the pair at fixed positions `7` and `8`

### `ctx_shuffle16`

- start from the exact `ctx_anchor16` selected local frames
- keep pair frame positions fixed
- deterministically shuffle only the remaining context frames

### `ctx_anchor32`

- same local-anchor logic as `ctx_anchor16`
- use `16` frames ending at `f0`, then `16` frames starting at `f1`

## V-JEPA 2.1 Feature Definition

Use only frozen encoder output tokens.

Do not use:

- predictor output
- pretraining loss code
- attentive classifier / probe
- any JEPA downstream head

The branch should test encoder representation quality only.

## Temporal Slice Selection

Do not pass full-window tokens directly to the current adapter.

Instead:

1. run the frozen encoder on the selected clip
2. reshape encoder output tokens back to temporal-spatial grid
3. keep only the temporal tubelet slices containing `f0` and `f1`
4. flatten the retained slices into `[tokens, dim]`

Because `tubelet_size=2`, the retained slices correspond to pair-relevant temporal tubelets, not necessarily exact individual frames.

If `f0` and `f1` fall into the same temporal tubelet, record that fact in metadata.

## Cache File Format

Each sample cache file should store:

- `features`
- `metadata`

### `features`

Shape:

- `[tokens, dim]`

Conceptually:

- `selected_temporal_slices x spatial_tokens`

### `metadata`

At minimum:

- `feature_backbone`
- `model_name`
- `checkpoint_path`
- `sample_id`
- `scene_id`
- `sequence_id`
- `window_mode`
- `window_size_raw`
- `window_paths_raw`
- `resampled_num_frames`
- `resampled_frame_indices`
- `pair_frame_ids`
- `pair_frame_paths`
- `pair_temporal_indices_raw`
- `pair_temporal_indices_resampled`
- `tubelet_size`
- `selected_temporal_token_indices`
- `selected_feature_shape`
- `full_encoder_token_shape`
- `crop_size`
- `patch_size`
- `num_frames`

## Files To Add

- `experiments/probe3d/scripts/prepare_scrream_vjepa21_feature_cache.py`
- `experiments/probe3d/train_vjepa21_nova_adapter.py`
- `slurm/scrream_vjepa21_precompute.sbatch`
- `slurm/scrream_vjepa21_ablation_pack_train.sbatch`
- `experiments/probe3d/requirements-vjepa21.txt`

## Shared Code To Reuse

Reuse from the current probe stack:

- SCRREAM adapter `.pt` loader
- image root remap logic
- target normalization logic
- frozen NOVA decoder loading
- `nova_flow`
- robust validation metrics
- `val_visual_40960`
- final metrics json structure

## Feature Cache Script Responsibilities

The cache script should:

1. read SCRREAM adapter metadata
2. build the requested pair-local clip directly from the source sequence
3. derive one of:
   - `pair_exact16`
   - `ctx_anchor16`
   - `ctx_shuffle16`
   - `ctx_anchor32`
4. preprocess frames for V-JEPA 2.1
5. run frozen encoder
6. reshape encoder tokens into temporal-spatial grid
7. keep pair-relevant temporal tubelet slices only
8. write per-sample cache files

## Training Script Responsibilities

The training script should:

1. reuse current SCRREAM loader
2. read V-JEPA 2.1 cache by `scene_id` or `sample_id`
3. feed cached features into the current MLP adapter
4. use the current frozen NOVA decoder
5. train with the same `nova_flow`
6. validate with the same robust metrics and preview export

## Smoke Plan

### Smoke A: cache precompute

- `ctx_anchor16`
- `max_samples=2`

Checks:

- cache files are written
- feature shape is stable
- metadata correctly records pair-to-temporal mapping

### Smoke B: shuffled control

- `ctx_shuffle16`
- `max_samples=2`

Checks:

- shuffled path runs
- feature shape matches the non-shuffled path

### Smoke C: train integration

- smoke cache
- `max_steps=20`

Checks:

- loss is finite
- gradients flow through adapter
- checkpoint writes
- preview export works

## Formal Runs

First-stage formal runs:

- `V-JEPA2.1 pair_exact16 -> MLP -> NOVA`
- `V-JEPA2.1 ctx_anchor16 -> MLP -> NOVA`
- `V-JEPA2.1 ctx_shuffle16 -> MLP -> NOVA`
- `V-JEPA2.1 ctx_anchor32 -> MLP -> NOVA`

## Risks

1. checkpoint loading:
   - local checkpoint loading must be explicit; do not depend on the current torch.hub URL defaults
2. temporal tubelet alignment:
   - pair frames may map to the same temporal token slice
3. backbone mismatch:
   - video representation quality may not transfer cleanly to static two-view geometry
4. fairness drift:
   - the adapter must continue to receive only pair-relevant tokens in first-stage runs

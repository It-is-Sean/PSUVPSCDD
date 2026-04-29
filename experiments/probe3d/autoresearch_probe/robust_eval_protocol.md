# Robust visual-first eval protocol (draft)

This protocol replaces two-sample oracle/CD leaderboard readings with a sample-stable, visual-first audit. It is meant for the proposal-aligned path: frozen VGGT visual backbone -> lightweight/structured adapter -> frozen NOVA-style decoder -> complete/amodal reconstruction inside sparse input-view frusta.

## Why this exists

Current oracle/CD rankings are not reliable enough to drive architecture decisions:

- the recent oracle gates averaged only two validation samples;
- CD is stochastic because decoder rollout sampling changes with seed/step;
- oracle optimizes NOVA flow loss while we often rank by rollout Chamfer;
- target modes that are visually similar (`anchor_frustum`, `covered_by_ge2`, `nova_input_frustum`) can receive very different CD due to outlier samples;
- MLP adapter outputs can look shapeless even when a single-sample CD appears deceptively low.

So every meaningful comparison should report robust stats and include renders before changing the research direction.

## Fixed manifest

Use a fixed ScanNet val manifest rather than "first 2 samples". Initial metadata-only helper:

```bash
python3 experiments/probe3d/autoresearch_probe/make_fixed_eval_manifest.py \
  --data_root /data1/jcd_data/scannet_processed_large_f20_vhclean500k_split_seed17 \
  --split val \
  --num_views 2 \
  --max_interval 1 \
  --scene_count 10 \
  --samples_per_scene 3 \
  --out experiments/probe3d/result/autoresearch_probe/fixed_eval_manifest_i1_k2_s10x3.json
```

This yields 30 fixed K=2 / interval=1 samples spread over 10 sorted val scenes. `max_interval=1` is intentional: the processed ScanNet data already uses `frame_skip=20`, so adjacent processed frames are the corrected NOVA-style spacing baseline.

## Metrics to log per manifest row

For each row and decoder seed, log:

- symmetric CD-L2 (legacy, for continuity only);
- one-sided `pred -> GT` CD and `GT -> pred` CD separately;
- trimmed CD (e.g. 90% or 95%) to expose outlier domination;
- failure flags: NaN/empty target, CD above fixed threshold, visibly collapsed output;
- target stats: valid point count, bbox, local plane residual summary if cheap;
- sample key / labels / decoder seed / checkpoint hash.

Aggregate with median, p75, p90, failure rate, and worst-5 sample table. Avoid claiming from means alone.

## Visual audit requirements

Before trusting a comparison, save for representative rows:

- input RGB contact sheet;
- GT-only rotating MP4 or multi-view PNG;
- adapter/oracle prediction vs GT triptych;
- overlay using consistent colors and camera/bbox settings;
- at least: median sample, p75 sample, and worst failure sample.

For adapter renders, load `best.pth` directly and infer/render; do not retrain or optimize an oracle just to visualize a checkpoint.

## Decision rule

- If target modes are visually similar but metrics diverge, treat the metric/protocol as suspect and inspect one-sided/trimmed metrics before selecting a target.
- If oracle fits a target but adapter fails, do not launch wider MLP sweeps by default; first test representation-alignment mechanisms (query-conditioned/cross-attention/ray-pos adapter or oracle-token distillation).
- Direct Chamfer is diagnostic/auxiliary; generator-native NOVA flow semantics remain central for proposal claims.

## Next implementation step

Add manifest-aware eval scripts so oracle and adapter eval can run on the same fixed rows and decoder seeds, then generate a compact `robust_eval_summary.json` under `experiments/probe3d/result/autoresearch_probe/<eval_id>/`.

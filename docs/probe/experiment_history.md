# Experiment history summary

This document consolidates the **actual recorded experiment history** currently present in the repo.

## 0. High-level judgment

The current history is real but unevenly documented.

- The **real executable history** is concentrated in `experiments/probe3d/`.
- The cleaner `scripts/probe/` + `configs/probe/` workspace is still mostly a scaffold and has **not yet accumulated real run history**.
- The strongest currently documented signal is **VGGT -> NOVA decoder via the flow adapter branch**.
- The cross-attention branch has been tried multiple times, but the current logged evidence does **not** make it look like the leading direction yet.
- The old direct probe path has one reasonably documented full run, but the earlier tiny/medium stages are only partially recorded.

So the repo history currently tells a coherent story:

1. build SCRREAM adapter data
2. run a direct probe baseline from precomputed NOVA features
3. switch to a VGGT -> NOVA decoding path
4. compare adapter variants
5. start probing ScanNet-scale preprocessing, but that branch is currently blocked by permissions

---

## 1. Where the real history lives

### Main recorded artifacts
- `experiments/probe3d/result/`
- `experiments/probe3d/runs/`
- `experiments/probe3d/adapter_data_test/`
- `experiments/probe3d/cache/`
- `experiments/probe3d/result/scannet_logs/`

### Important absence
There are currently **no real experiment records** under the cleaner `artifacts/` workspace. That means the structured probe framework is documented, but the live experiment trail still belongs to `probe3d`.

---

## 2. Chronological reconstruction

## Phase A — SCRREAM adapter-data construction and small dry runs

### A1. Initial manifest build
- Artifact: `experiments/probe3d/adapter_data_test/scrream_adapter_manifest.json`
- Generated at: `2026-04-26T13:05:27+00:00`
- Sample counts: `train=388, val=43, test=1` (total 432)
- Config clues:
  - `group_size=4`
  - `sample_stride=1`
  - `pad_short_scenes=true`
  - `pseudo_gt_views=2`

### Judgment
This looks like the **first successful data-preparation pass**, but the split is clearly awkward (`test=1`). Good enough for bringing the pipeline up, not good enough as a final experimental protocol.

### A2. Tiny prepared dataset
- Artifact: `experiments/probe3d/adapter_data_test/scrream_adapter_tiny.pt`
- Modified: `2026-04-26 22:11`
- Size: ~8.9 MB

### A3. Tiny direct-probe checkpoint
- Artifact: `experiments/probe3d/runs/adapter_tiny_epoch10.pt`
- Modified: `2026-04-26 22:11`

### Judgment
This was almost certainly a **bring-up / smoke-test stage**. Useful historically, but there is not enough metadata to treat it as a serious comparable run.

### A4. Medium selected subset
- Artifact: `experiments/probe3d/adapter_data_test/scrream_adapter_medium_seed17_selection.json`
- Modified: `2026-04-26 22:21`
- Source manifest: `/home/wdh/PSUVPSC3DD/experiments/probe3d/adapter_data_seed17/scrream_adapter_manifest_seed17.json`
- Limits:
  - `train=128`
  - `val=32`
  - `test=32`
- Selected samples: 192 total

### A5. Medium prepared dataset + checkpoint
- Artifact: `experiments/probe3d/adapter_data_test/scrream_adapter_medium_seed17.pt`
- Modified: `2026-04-26 22:21`
- Size: ~82 MB

- Artifact: `experiments/probe3d/runs/adapter_medium_seed17_epoch15.pt`
- Modified: `2026-04-26 22:22`

### Judgment
This is a sensible intermediate step: move from tiny bring-up to a **bounded balanced subset** before committing to full-data training. But again, the bookkeeping is incomplete: the checkpoint exists, but there is no matching explicit metrics/log bundle beside it.

---

## 3. Phase B — Full direct probe baseline on prepared SCRREAM data

### B1. Full prepared dataset
- Artifact: `experiments/probe3d/result/scrream_adapter_manifest_seed17_full.json`
- Generated at: `2026-04-26T14:26:29+00:00`
- Prepared dataset path: `experiments/probe3d/result/scrream_adapter_full_seed17.pt`
- Sample counts: `train=241, val=55, test=136` (total 432)
- Feature shape per sample: `[768, 128]`
- Target shape per sample: `[4096, 3]`
- Dataset file size: ~191 MB

### Judgment
This is the first dataset version that looks like a **real experimental base** instead of just a pipeline test. The split is much more usable than the original manifest.

### B2. Full direct-probe training run
- Checkpoint: `experiments/probe3d/result/adapter_full_seed17_epoch30.pt`
- Modified: `2026-04-26 22:38`
- Train log: `experiments/probe3d/result/adapter_full_seed17_epoch30_train.log`
- Test log: `experiments/probe3d/result/adapter_full_seed17_epoch30_test.log`
- Summary: `experiments/probe3d/result/adapter_full_seed17_epoch30_summary.txt`

### Recorded metrics
- Final test Chamfer Distance: `0.865267`
- Best observed validation loss: `0.213123` at epoch `29/30`
- Training loss fell from `3.707373` at epoch 1 to roughly the `0.175~0.18` range later

### B3. Qualitative outputs
- Directory: `experiments/probe3d/result/full_seed17_test_ply/`
- Stored:
  - `predictions.pt`
  - `136` GT PLY files
  - `136` predicted PLY files
  - `4` compare PNGs

### Judgment
This is the **best-documented direct baseline** in the repo right now. It is not glamorous, but it is real: dataset -> train log -> test metric -> qualitative exports. If we ever need a historical baseline anchor, this is the one to cite first.

---

## 4. Phase C — VGGT -> NOVA decoding experiments

All runs in this phase share the same overall pattern:
- frozen VGGT features
- selected intermediate feature index `22` (23rd block in human counting)
- cached real-image features under `experiments/probe3d/cache/vggt23_realimg_seed17_l4/`
- NOVA Stage-1 decoder supervision via `nova_flow` loss

### Shared cache footprint
- Cache directory: `experiments/probe3d/cache/vggt23_realimg_seed17_l4/`
- Cached feature files: `257`
- Total size: ~`5.52 GB`

This means the project has already paid the cost of a meaningful amount of VGGT feature extraction, and the later runs are not just toy stubs.

### C1. Flow adapter branch — strongest current result
- Run: `vggt23_nova_adapter_flow_l4_realimg_cached_lr5e-5_seed17`
- Modified: `2026-04-27 01:50`
- Key config:
  - adapter hidden dim: `1024`
  - adapter layers: `4`
  - adapter params: `4,334,720`
  - loss: `nova_flow`
  - lr: `5e-5`
  - batch size: `8`
  - max steps: `3000`
  - num queries: `2048`
- Final metrics:
  - `first_loss = 1.6231`
  - `final_loss = 0.8839`
  - `best_loss = 0.6908`
- Validation history:
  - best logged validation Chamfer L2 = `0.1631` at step `2000`
  - final logged validation Chamfer L2 = `0.1874` at step `3000`
- Extra outputs:
  - checkpoints every 500 steps
  - PLY snapshots during training
  - separate nohup log

### Judgment
This is currently the **most promising branch** in the recorded history.

Not because it is perfect, but because:
- it ran to completion
- it has repeated checkpoints
- it has a plausible downward training curve
- its validation metric is much better than the logged attention-branch runs

If I had to pick one historical run to continue from, it would be this one.

### C2. Cross-attention debug smoke test
- Run: `vggt23_nova_attention_l4_scrream_debug_seed17`
- Modified: `2026-04-27 02:07`
- Key config:
  - adapter type: `cross_attention`
  - hidden dim: `512`
  - layers: `4`
  - params: `9,924,224`
  - batch size: `1`
  - max steps: `1`
  - `debug_one_batch = true`
- Metrics:
  - `first_loss = final_loss = best_loss = 1.7893`
  - validation Chamfer L2 = `52.3702`

### Judgment
This is not a meaningful scientific result. It is a **plumbing check**: the attention adapter can run one step and save outputs.

### C3. Cross-attention full-size attempt (4 layers, h=512)
- Run: `vggt23_nova_attention_l4_scrream_cached_lr2e-5_seed17`
- Modified: `2026-04-27 02:14`
- Key config:
  - adapter type: `cross_attention`
  - hidden dim: `512`
  - layers: `4`
  - params: `9,924,224`
  - lr: `2e-5`
  - batch size: `8`
  - max steps planned: `3000`
- What actually happened:
  - only checkpointed up to `step_000500.pth`
  - training log continues only to around step `883`
  - no `final_metrics.json`
- Best logged training-side number: `best=0.83555222`
- Best/only validation record in log bundle: `val_chamfer_l2 = 0.6273` at step `500`

### Judgment
This run is **incomplete / interrupted**. It still tells us something: the larger attention adapter did not obviously beat the flow branch early, and the run stopped before becoming a trustworthy candidate.

### C4. Smaller cross-attention run (2 layers, h=256)
- Run: `vggt23_nova_attention_l2_h256_scrream_cached_lr1e-5_seed17`
- Modified: `2026-04-27 02:38`
- Key config:
  - adapter type: `cross_attention`
  - hidden dim: `256`
  - layers: `2`
  - heads: `4`
  - params: `1,809,792`
  - lr: `1e-5`
  - batch size: `8`
  - max steps: `3000`
- Final metrics:
  - `first_loss = 2.2715`
  - `final_loss = 0.9747`
  - `best_loss = 0.8164`
- Validation:
  - final / best logged validation Chamfer L2 = `1.2136` at step `3000`

### Judgment
This is a completed and well-logged run, but the result is **clearly weaker than the flow adapter branch**. Right now it argues against making cross-attention the default path unless there is a strong conceptual reason or later evidence reverses the story.

---

## 5. Phase D — ScanNet-scale preprocessing attempt

### D1. ScanNet extraction attempt
- Log: `experiments/probe3d/result/scannet_logs/prepare_scannet_large_300_50.log`
- Modified: `2026-04-27 02:01`
- Intent:
  - select `300` training scenes
  - select `50` test scenes
  - sample frames with `frame_skip=10`
- Failure:
  - `PermissionError: [Errno 13] Permission denied`
  - blocked on reading `.sens` files, e.g. `scene0000_01.sens`

### Judgment
This is **infrastructure history, not experiment history**. It matters because it shows the project was already trying to expand beyond SCRREAM, but right now this branch is blocked by file-permission issues rather than model behavior.

---

## 6. What the history currently supports

## Strongly supported
- The repo has already moved past pure planning into real probe experimentation.
- A full SCRREAM-based direct baseline exists.
- A VGGT -> NOVA decoding path exists and has been trained seriously enough to compare variants.
- The flow adapter branch is the best currently documented direction.

## Weakly supported / not yet convincing
- Cross-attention as the main adapter family.
- Any claim about video backbones.
- Any claim about large-scale ScanNet experiments.
- Any claim that the structured `scripts/probe/` workspace is already the place where real experiments are being logged.

---

## 7. Reproducibility debt found while reading the history

Several history files still contain stale absolute paths from earlier environments:

- `/home/wdh/nova3r/...`
- `/home/wdh/PSUVPSC3DD/...`
- `/home/jcd/.openclaw/workspace/projects/probe/...`

This shows up in:
- dataset manifests
- some run configs
- old checkpoint references

### Why this matters
The experiments themselves are still useful, but the history is **not yet fully self-contained as a paper artifact trail**. Anyone replaying the logs later may not know which paths are semantically important versus just historical machine residue.

---

## 8. Suggested cleanup priority

If the goal is to make the history truly paper-usable, the next documentation step should be:

1. keep this summary file as the canonical history overview
2. retro-tag each serious run as one of:
   - data-prep
   - direct-baseline
   - VGGT->NOVA flow adapter
   - VGGT->NOVA attention adapter
   - infra attempt / failed preprocessing
3. normalize stale absolute paths in copied summaries / manifests where possible
4. start writing future serious runs into a single structured run-card or experiment-index file

---

## 9. Bottom line

The current experiment history is not random clutter. It already tells a fairly sharp story:

- **data-prep matured from tiny -> medium -> full**
- **a direct baseline was trained and evaluated**
- **a VGGT -> NOVA probe line was established**
- **the flow adapter currently beats the attention variants in the recorded logs**
- **the project has not yet earned a broader cross-backbone or video-level claim**

That is probably the most honest reading of the history right now.

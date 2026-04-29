# TODO

## Active now — 2026-04-29 late afternoon

### Local autopilot / handoff
- [x] retire ResearchClaw full-pipeline execution after CIFAR/KD/FitNet drift
- [x] install 15-minute local OpenClaw autopilot for PSUVPSC3DD
- [x] make Feishu autopilot follow-ups Chinese by default
- [x] push branch `wip/psuvpsc3dd-autoresearch-20260429` to `dongjiacheng06/3dprobe`
- [x] create this `/neat` handoff cleanup

### Corrected evaluation protocol
- [x] mark old `max_interval=30` ScanNet runs as interval-confounded
- [x] set intended ScanNet interval to `scannet_max_interval=1`
- [x] add `robust_ply_metrics.py`
- [x] add `eval_checkpoint_robust.py`
- [x] evaluate current MLP baseline on fixed 30-sample manifest
- [x] render representative success/failure cases from fixed-30 robust eval
- [x] add fixed-30 failure-case audit
- [ ] inspect the current cross-attention candidate's fixed-30 robust eval once it finishes
- [ ] compare cross-attention vs MLP using F@0.05 / precision / representative renders, not symmetric CD alone

### Candidate experiments
- [x] enable corrected-interval ScanNet controls for cross-attention adapter
- [x] run one-batch cross-attention smoke test
- [x] launch and finish `p7_k2_i1_anchor_ca_l2_h512_chamfer_step1000` (`val_chamfer_l2=0.54222615`)
- [ ] decide whether cross-attention has any qualitative precision benefit after fixed-30 eval/renders
- [ ] if cross-attention is not better, prioritize token-distillation / pseudo-GT diagnostics over more adapter capacity sweeps

### Code/documentation hygiene
- [x] move temporary DDP helper to `experiments/probe3d/autoresearch_probe/debug/init_distributed_smoke.py`
- [x] add `.researchclaw_cache/` and local ResearchClaw secret patterns to `.gitignore`
- [x] add `experiments/probe3d/autoresearch_probe/CURRENT_STATE.md`
- [x] keep `README.md`, `PROJECT.md`, and `docs/probe/*` synchronized for the 2026-04-29 handoff
- [ ] archive or rename stale phase configs once the next stable experiment plan is chosen

---

## Older TODO history below



## Overnight running plan — 2026-04-29 early morning

- [x] switch active feasibility line to K=2 after user hypothesis about target sparsity / empty view slots
- [x] switch adapter capacity to `MLP-L4-H1024`
- [x] launch K=2 `anchor_frustum` loss ablation: `nova_flow`, `flow_chamfer_hybrid@0.05`, `chamfer_sample`
- [x] queue K=2 oracle GT sweep: `nova_input_frustum`, `covered_by_ge2`, `anchor_frustum_margin1.5`, `nova_per_view_frustum_anchor_zpos`, `nova_per_view_ldi{2,4,8}`
- [x] queue K=2 adapter GT/loss sweep for `covered_by_ge2`, `nova_input_frustum`, and `anchor_frustum_margin1.5`
- [ ] tomorrow morning: summarize all `p4_` rows from `experiments/probe3d/autoresearch_probe/results.tsv`
- [ ] tomorrow morning: inspect PLYs for the best K=2 loss/target candidates
- [ ] tomorrow morning: decide which run becomes the proposal-facing feasibility result

## Active now

### Reset to paper-aligned NOVA3R GT / loss
- [x] identify that direct Chamfer can overfit the metric and is not the main representation-probe claim
- [x] clarify NOVA3R target semantics: complete / amodal surface inside selected input-view frustum, not full-room completion
- [x] add explicit `nova_input_frustum` / `nova_anchor_frustum` target aliases
- [x] expose `scannet_complete_points` and `query_source` in the autoresearch harness
- [x] create `phase2_nova_aligned.json` with K=1/K=2 oracle and native-flow adapter probes
- [ ] run K=1 oracle sanity with `nova_input_frustum + src_complete_fps_4096`
- [ ] run K=2 oracle sanity with `nova_input_frustum + src_complete_fps_4096`
- [ ] if oracle support is plausible, run K=1/K=2 MLP-L4 native-flow adapter probes
- [ ] log one-way distances (`pred→GT`, `GT→pred`) during validation
- [ ] compare visual sharpness, not just symmetric CD

### Keep docs and state in sync
- [x] record current autoresearch results in docs
- [ ] record future precision-aware loss results and output paths

## Next after paper-aligned native-flow MLP stabilizes

### Launch CA on the same target/objective
- [ ] reuse the same processed split
- [ ] reuse the same `nova_input_frustum` target
- [ ] reuse the same precision-aware objective
- [ ] keep interpretation baseline-first, not headline-first

### Launch SA on the same target/objective
- [ ] same constraints as CA

## Deferred but still important

### Correct SCRREAM rerun
- [ ] wait for official full SCRREAM data root
- [ ] reconnect the corrected training path
- [ ] rerun adapter branches under valid data assumptions

### Performance optimization
- [ ] optimize online union-frustum crop if dataloader becomes the bottleneck
- [ ] consider deeper reservoir / sample construction optimization only if the formal run shows it is necessary
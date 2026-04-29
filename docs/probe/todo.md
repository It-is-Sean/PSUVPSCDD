# TODO

## Active now — 2026-04-29 afternoon

### Workspace / ResearchClaw supervision
- [x] install AutoResearchClaw in isolated data-disk env
- [x] run `researchclaw setup` and install OpenCode
- [x] configure project-specific `researchclaw/config.arc.yaml`
- [x] add local Responses proxy `researchclaw/openclaw_stream_proxy.js` for ResearchClaw LLM access
- [x] start ResearchClaw run under `experiments/probe3d/result/researchclaw/`
- [x] start 15-minute supervisor cron to audit ResearchClaw direction and code diffs
- [ ] if ResearchClaw keeps drifting into irrelevant external search, redirect it to local repo audit / fixed eval protocol

### Corrected evaluation protocol
- [x] mark old `max_interval=30` ScanNet runs as interval-confounded
- [x] set intended ScanNet interval to `scannet_max_interval=1`
- [x] add `robust_ply_metrics.py`
- [x] add `eval_checkpoint_robust.py`
- [x] evaluate current MLP baseline on 5 fixed samples
- [ ] expand robust eval to the fixed 30-sample manifest
- [ ] render representative success/failure cases from robust eval
- [ ] compare at least one structured adapter / pseudo-GT candidate under the same protocol

### Code/documentation hygiene
- [x] move temporary DDP helper to `experiments/probe3d/autoresearch_probe/debug/init_distributed_smoke.py`
- [x] add `.researchclaw_cache/` and local ResearchClaw secret patterns to `.gitignore`
- [x] add `experiments/probe3d/autoresearch_probe/CURRENT_STATE.md`
- [ ] split or archive stale phase configs once ResearchClaw produces the next valid experiment plan
- [ ] keep `README.md`, `PROJECT.md`, and `docs/probe/*` synchronized after each major correction

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
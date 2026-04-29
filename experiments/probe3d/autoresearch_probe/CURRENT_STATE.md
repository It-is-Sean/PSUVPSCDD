# Current PSUVPSC3DD probe state — 2026-04-29

This file is the compact current-state pointer for autonomous / autoresearch work.
It supersedes older numeric-only notes when conflicts appear.

## Non-negotiable corrections

1. **ScanNet interval correction**
   - Processed ScanNet root uses `frame_skip=20`.
   - Therefore `scannet_max_interval=1` means adjacent processed frames, i.e. roughly 20 raw-frame spacing.
   - Old runs using `max_interval=30` are interval-confounded and cannot support K-view conclusions.

2. **Metric correction**
   - Single scalar symmetric Chamfer and two-sample oracle averages are not reliable enough for claims.
   - Use fixed-sample robust metrics: pred→GT precision side, GT→pred recall side, F-score thresholds, trimmed CD, and visual videos.

3. **MLP baseline status**
   - The K2 interval=1 MLP-L4/chamfer checkpoint has some recall but poor precision/outlier behavior.
   - Fixed-5 robust eval for `p5_k2_adjacent_anchor_mlp_l4_chamfer_step1000/best.pth`:
     - symmetric CD mean/median: `0.0411 / 0.0326`
     - pred→GT mean distance median: `0.130`
     - GT→pred mean distance median: `0.0628`
     - F@0.05 mean/median: `0.328 / 0.355`
     - precision@0.05 mean: `0.238`
     - recall@0.05 mean: `0.536`
   - Interpretation: not just missing coverage; predictions are loose/outlier-heavy and visually poor.

4. **GT quality correction**
   - GT-only visual audit showed the three ScanNet target modes are not dramatically different by eye.
   - Current failure should not be blamed solely on dirty GT.

## Active ResearchClaw integration

- Config: `researchclaw/config.arc.yaml`
- Prompt guardrails: `researchclaw/prompts.psuvpsc3dd.yaml`
- Local Responses compatibility proxy: `researchclaw/openclaw_stream_proxy.js`
- Current run: `experiments/probe3d/result/researchclaw/run_20260429_135015`
- Supervisor cron: `PSUVPSC3DD AutoResearchClaw supervisor`, every 15 minutes.

ResearchClaw is allowed to help organize the work, but its outputs must be audited against the proposal:

- frozen visual backbone;
- frozen NOVA3R-style complete-3D decoder;
- train only lightweight/structured adapters;
- complete/amodal sparse-input-frustum reconstruction;
- visual-first robust evaluation;
- no claims from old `max_interval=30` runs or single-sample CD.

If ResearchClaw external web search drifts into irrelevant sources, redirect it to local repo audit and proposal-specific evidence.

## Next clean research step

1. Expand robust eval from 5 samples to the fixed 30-sample manifest.
2. Render representative success/failure cases.
3. Compare a structured adapter or pseudo-GT checkpoint under the same robust protocol.

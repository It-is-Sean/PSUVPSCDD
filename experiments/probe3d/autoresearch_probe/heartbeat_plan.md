# PSUVPSC3DD autonomous research heartbeat

Purpose: every ~15 minutes, make one small, proposal-aligned research step without cluttering the repo or spamming Jiacheng.

## Hard constraints

- Stay inside the proposal direction: frozen visual backbone (VGGT-first) -> lightweight/structured adapter -> frozen NOVA3R-style decoder -> complete/amodal reconstruction inside selected sparse input-view frusta.
- Treat ScanNet as a NOVA3R-style transfer probe, not an official NOVA3R reproduction.
- Do not resurrect invalid old SCRREAM eval-subset claims.
- Do not make formal claims from current oracle/CD numbers alone; current evidence says the oracle protocol and symmetric CD are biased/unstable.
- Render-first validation: visual GT/pred/input audits are mandatory before interpreting leaderboard rows.
- Keep code tidy: prefer small scripts/configs under `experiments/probe3d/autoresearch_probe/`; put large/generated outputs under `experiments/probe3d/result/autoresearch_probe/`; use `/tmp` for throwaway scratch.
- Avoid overlapping GPU jobs. If a run is already active, monitor/summarize it instead of launching another.
- GPU use is allowed, including roughly 30-minute experiments. Jiacheng explicitly said not to worry about burning GPU. Avoid overlapping duplicate jobs, keep runs bounded, and log trial_id/output path/why here.
- Send a concise Feishu progress update after every 15-minute heartbeat run, because Jiacheng explicitly requested per-heartbeat feedback.

## Current research state

Latest user-approved interpretation:

- Corrected dataloader interval is `scannet_max_interval=1`, which means adjacent processed frames and roughly 20 raw-frame spacing because preprocessing used `frame_skip=20`.
- GT-only audit suggests current raw ScanNet target modes (`anchor_frustum`, `covered_by_ge2`, `nova_input_frustum`) are visually similar and not obviously too dirty.
- The current oracle CD rankings are not trustworthy: they use only two samples, optimize flow loss but evaluate stochastic CD, and are dominated by outlier samples.
- MLP-L4 adapter visual output is poor; this may be representation alignment, objective/metric bias, or both.

## Priority queue

1. Replace fragile oracle/CD interpretation with a visual-first, sample-stable evaluation protocol.
   - fixed sample list (20-50 val samples)
   - fixed target FPS seed(s)
   - repeated decoder eval seeds
   - report median/p75/failure rate, not 2-sample mean
   - save input/GT/pred videos for representative success/failure cases
2. Audit whether adapter predictions are bad because of adapter representation alignment or because the eval/metric is misleading.
   - compare pred->GT and GT->pred separately
   - add trimmed/thresholded metrics and visual crops
3. If metric protocol is stable, test a structured adapter rather than more blind MLP sweeps.
   - cross-attention/query-conditioned adapter
   - optional camera/ray/view positional encoding
   - consider oracle-token distillation only after verifying optimized tokens are stable across restarts
4. Keep documentation in sync with corrections.

## Heartbeat step format

Each heartbeat should append to `heartbeat_log.md`:

```text
## YYYY-MM-DD HH:MM
- Checked: ...
- Action: ...
- Result: ...
- Next: ...
```

## Autoresearch reference pattern

Jiacheng suggested referencing the autoresearch library/style. Treat the local `experiments/probe3d/autoresearch_probe/` harness as the immediate pattern unless a separate upstream library path is provided. Reuse these ideas:

- explicit trial configs rather than ad-hoc shell commands
- immutable `results.tsv` rows with output directories
- small hypothesis-driven trials with keep/discard criteria
- one-step loop: inspect state -> choose next minimal experiment/audit -> run or write proposed command -> log result
- avoid changing validation protocol silently while comparing trials

If a separate autoresearch repo/library is later identified, inspect it and port only clean abstractions that fit this proposal.

## Delivery rule

Jiacheng explicitly requested a Feishu progress update every 15 minutes. Each heartbeat must return a concise summary of what was checked, what changed, key result, and next step. Avoid long reports unless a decision/blocker/milestone requires detail.

## GPU budget rule

Jiacheng explicitly allowed GPU-burning experiments, including ~15-minute cadence with runs allowed to continue up to ~30 minutes when useful. Use GPUs when they are the right next step; do not block on excessive caution. Still keep commands/configs logged, avoid duplicate overlap, and prefer bounded trials with render/metric outputs.

## AutoResearchClaw reference

Jiacheng asked to use AutoResearchClaw as a reference. Local install: `/home/jcd/.openclaw/workspace/projects/external/AutoResearchClaw`, env `/data1/jcd_data/envs/researchclaw-py311`, CLI `/data1/jcd_data/envs/researchclaw-py311/bin/researchclaw`. Read `autoresearchclaw_notes.md` before changing the autonomous loop. Use its stage/contract/checkpoint/HITL patterns, but do not run powerful ResearchClaw commands (`setup`, full `run`, Docker/SSH/Overleaf/server) without config review and a clear proposal-aligned reason.

## Local autopilot replaces ResearchClaw full pipeline

ResearchClaw full pipeline is no longer used for automatic project execution because it drifted into unrelated CIFAR/KD/FitNet experiments. Future autonomous progress should use `local_autopilot_prompt.md` and the local `autoresearch_probe` harness. The only retained ResearchClaw value is high-level planning inspiration and the archived artifacts under ignored result directories.
## Feishu reporting language

All Feishu progress follow-ups to Jiacheng must be in Chinese unless Jiacheng explicitly asks for another language.

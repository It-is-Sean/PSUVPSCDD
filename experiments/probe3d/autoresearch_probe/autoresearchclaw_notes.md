# AutoResearchClaw reference notes for PSUVPSC3DD

Installed reference repo:

- Source: `https://github.com/aiming-lab/AutoResearchClaw/tree/main`
- Local clone: `/home/jcd/.openclaw/workspace/projects/external/AutoResearchClaw`
- Python env: `/data1/jcd_data/envs/researchclaw-py311`
- CLI: `/data1/jcd_data/envs/researchclaw-py311/bin/researchclaw`
- Installed editable package: `researchclaw==0.3.1`

Safety/vetting summary:

- Reviewed `pyproject.toml`, `README.md`, CLI entry, config example, and source scan for high-risk patterns.
- The project is a full research pipeline with LLM/web/Docker/SSH/Overleaf integrations. These are intentional but powerful.
- Do **not** run `researchclaw setup`, `researchclaw run`, Docker/SSH/Overleaf/server commands, or external-publish features without explicit reason/config review.
- The CLI setup path can install `opencode-ai` globally via npm if interactive; avoid that path for now.
- Use the repo mainly as a design/reference library and possibly for offline components (project management, result tables, pipeline contracts, checkpoint/logging patterns) unless a specific ResearchClaw run is justified.

Patterns to port into this proposal project:

1. Stage/contract discipline
   - Each autonomous step should declare input artifacts, output artifacts, acceptance/failure criteria, and next decision.
2. Immutable run directories
   - Keep all generated artifacts under `experiments/probe3d/result/autoresearch_probe/<trial_id>/`.
3. Human-readable checkpointing
   - Every trial/eval should write a small JSON summary and append to `heartbeat_log.md`.
4. Robust validation before experiments
   - Validate dataloader, sample list, metric protocol, and render outputs before launching larger adapter runs.
5. HITL-style gates
   - Ask Jiacheng only at major direction gates; otherwise keep progress moving and report concise 15-minute updates.
6. Anti-fabrication/anti-metric-gaming
   - Do not claim success from single CD rows; require visual evidence + robust metrics + fixed sample protocol.

Immediate application to PSUVPSC3DD:

- Convert current ad-hoc evaluation into a small staged loop:
  1. `eval_protocol`: fixed sample list, robust metrics, render plan.
  2. `baseline_eval`: evaluate existing MLP checkpoint under protocol.
  3. `candidate_design`: choose one structured adapter or pseudo-GT candidate.
  4. `candidate_run`: bounded GPU run with explicit acceptance criteria.
  5. `candidate_audit`: render representative success/failure and compare robust metrics.
- Keep the current 15-minute OpenClaw heartbeat as the orchestration layer rather than running full ResearchClaw end-to-end.

## Setup result — 2026-04-29

Ran:

```bash
/data1/jcd_data/envs/researchclaw-py311/bin/researchclaw setup
```

Result:

- OpenCode installed globally via npm: `/home/jcd/.openclaw/tools/node-v22.22.0/bin/opencode`, version `1.14.29`.
- Docker available.
- LaTeX/pdflatex unavailable; ResearchClaw can still export `.tex` but not compile PDFs here.
- Doctor with example config passes Python/YAML/config, but fails LLM connectivity/API-key because no real ResearchClaw config/API key is set; sandbox python points to missing `.venv/bin/python3` in example config; matplotlib missing in ResearchClaw env. Doctor report: `experiments/probe3d/result/autoresearch_probe/researchclaw_doctor_after_setup.json`.

Implication: usable as a reference/tooling library now. Before running a full ResearchClaw pipeline, create a project-specific config instead of using the example config.

# Probe workspace overview

This repo is trying to answer one concrete question:

> Can a shared complete-3D decoder reveal 3D structure in frozen representations that shallow direct readout misses?

## Current experimental status

Right now the real progress is simpler than the full proposal scope:

- on **SCREAM**, we have already trained **2-layer and 4-layer MLP adapters**
- the result is **promising enough to support feasibility**, with the best run reaching roughly **CD ≈ 0.18**
- switching to a **4-layer attention adapter** did **not** improve things and currently looks less convincing
- the next real milestone is to move to a **larger dataset (ScanNet v2)** and test whether this proposal still works there

So the current story is:

1. the proposal looks **feasible on a smaller dataset**
2. MLP-style adapters are currently the stronger direction
3. the next job is **scale validation**, not expanding scope

## What matters most right now

If you only want the essential documents, read these in order:

1. `PROPOSAL.md` — the core idea
2. `PROJECT.md` — current status / decisions / next step
3. `experiments/probe3d/README.md` — the path where the real experiments currently live

Everything else under `docs/probe/` should be treated as supporting notes, not the main project narrative.

## Repo reality check

The repo currently has two layers:

- a **cleaner research scaffold**: `configs/probe/`, `scripts/probe/`, `nova3r/probe/`
- a **more executable experiment path**: `experiments/probe3d/`

At the moment, the **actual experimental history is mostly in `experiments/probe3d/`**.

## Current strategic interpretation

The proposal is conceptually broad, but the practical first-paper path should stay narrow:

- focus on **image / geometry backbones first**
- treat **video** as a later extension
- prioritize a strong **feasibility story on ScanNet v2** over adding more model families too early

## Optional supporting notes

Use these only when needed:

- `docs/probe/experiment_history.md` — consolidated run history read from logs/results
- `docs/probe/nova3r_mapping.md` — mapping from proposal ideas to NOVA3R code
- `docs/probe/experiment_plan.md` — phased implementation plan
- `docs/probe/todo.md` — loose TODO list

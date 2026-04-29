# autoresearch_probe

Autoresearch-style loop for the frozen VGGT -> adapter -> frozen NOVA decoder probe.

## Fixed constraints

- VGGT is frozen.
- NOVA decoder is frozen.
- Only adapter/tokens/target preprocessing/search settings may change.
- The final selection metric is rollout Chamfer-L2; lower is better.
- Prefer lightweight adapters when Chamfer is close.
- Do not silently change validation protocol while comparing trials.

## Search order

1. Target/domain search first. Run oracle-token trials before adapter trials. If frozen NOVA cannot fit a target mode, discard it before spending adapter compute.
2. Adapter search second, on the best target mode(s).
3. Loss search third, after target + adapter are stable.

## Keep/discard rule

Keep a trial if it improves best Chamfer by >=5%, or keeps Chamfer close while reducing adapter params by >=30%, or gives a clear qualitative improvement. Discard if oracle is bad, metric regresses, or complexity increases without payoff.

## Logging

Every trial appends one row to `results.tsv`:

```text
trial_id	kind	target_mode	adapter	loss	steps	val_cd	best_cd	runtime_sec	status	notes	out_dir
```

Logs and artifacts live under `experiments/probe3d/result/autoresearch_probe/<trial_id>/`.

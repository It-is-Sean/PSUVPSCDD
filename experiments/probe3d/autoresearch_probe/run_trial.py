#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_ROOT = Path(__file__).resolve().parent
RESULTS_TSV = HARNESS_ROOT / "results.tsv"
HEADER = "trial_id\tkind\ttarget_mode\tadapter\tloss\tsteps\tval_cd\tbest_cd\truntime_sec\tstatus\tnotes\tout_dir\n"


def load_config(path: Path):
    payload = json.loads(path.read_text())
    defaults = payload.get("defaults", {})
    trials = payload.get("trials", [])
    return defaults, {t["trial_id"]: t for t in trials}


def ensure_results():
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(HEADER)


def append_result(row):
    ensure_results()
    values = [
        row.get("trial_id", ""), row.get("kind", ""), row.get("target_mode", ""),
        row.get("adapter", ""), row.get("loss", ""), str(row.get("steps", "")),
        f"{row.get('val_cd', 0.0):.8f}" if row.get("val_cd") is not None else "",
        f"{row.get('best_cd', 0.0):.8f}" if row.get("best_cd") is not None else "",
        f"{row.get('runtime_sec', 0.0):.1f}", row.get("status", ""),
        row.get("notes", "").replace("\t", " ").replace("\n", " "), row.get("out_dir", ""),
    ]
    with RESULTS_TSV.open("a", encoding="utf-8") as f:
        f.write("\t".join(values) + "\n")


def target_args(trial):
    mode = trial.get("target_mode", "complete_zpos")
    args = []
    if mode == "src_view":
        args += ["--query_source", "src_view", "--scannet_target_mode", "complete_zpos"]
    else:
        args += ["--scannet_target_mode", mode]
    if "frustum_margin" in trial:
        args += ["--scannet_frustum_margin", str(trial["frustum_margin"])]
    if "min_views" in trial:
        args += ["--scannet_min_views", str(trial["min_views"])]
    if "complete_points" in trial:
        args += ["--scannet_complete_points", str(trial["complete_points"])]
    if "max_interval" in trial:
        args += ["--scannet_max_interval", str(trial["max_interval"])]
    if "query_source" in trial:
        args += ["--query_source", str(trial["query_source"])]
    return args


def summarize_oracle(out_dir: Path):
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    bests = [float(s.get("best_cd", s.get("final_cd", 0.0))) for s in summary.get("samples", [])]
    if not bests:
        return None
    return sum(bests) / len(bests)


def summarize_adapter(out_dir: Path):
    log_path = out_dir / "training.log"
    if not log_path.exists():
        return None
    vals = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = re.search(r"validation step=\d+ val_chamfer_l2=([0-9.]+)", line)
        if m:
            vals.append(float(m.group(1)))
    return min(vals) if vals else None


def build_command(defaults, trial, out_dir: Path):
    python_bin = trial.get("python_bin", defaults.get("python_bin", sys.executable))
    data_root = trial.get("data_root", defaults["data_root"])
    nova_ckpt = trial.get("nova_ckpt", defaults["nova_ckpt"])
    num_views = str(trial.get("num_views", defaults.get("num_views", 4)))
    num_queries = str(trial.get("num_queries", defaults.get("num_queries", 2048)))
    kind = trial.get("kind", "oracle")
    if kind == "oracle":
        return [
            python_bin, "experiments/probe3d/fit_nova_tokens_scannet_oracle.py",
            "--data_root", data_root,
            "--nova_ckpt", nova_ckpt,
            "--split", trial.get("split", "val"),
            "--max_scenes", str(trial.get("max_scenes", 1)),
            "--num_views", num_views,
            "--num_queries", num_queries,
            "--num_samples", str(trial.get("num_samples", 2)),
            "--steps", str(trial.get("steps", 600)),
            "--lr", str(trial.get("lr", 1e-2)),
            "--out_dir", str(out_dir),
        ] + target_args(trial)
    if kind == "adapter":
        torchrun_bin = trial.get("torchrun_bin", defaults.get("torchrun_bin", "torchrun"))
        nproc = str(trial.get("nproc_per_node", 1))
        return [
            torchrun_bin, "--standalone", "--nproc_per_node", nproc,
            "experiments/probe3d/train_vggt_nova_adapter.py",
            "--dataset", "scannet",
            "--data_root", data_root,
            "--train_split", "train",
            "--val_split", "val",
            "--test_split", "test",
            "--max_val_scenes", str(trial.get("max_val_scenes", defaults.get("max_val_scenes", 10))),
            "--max_test_scenes", "0",
            "--eval_batches", str(trial.get("eval_batches", 0)),
            "--test_eval_batches", "0",
            "--nova_ckpt", nova_ckpt,
            "--adapter_layers", str(trial.get("adapter_layers", 4)),
            "--adapter_hidden_dim", str(trial.get("adapter_hidden_dim", 1024)),
            "--adapter_type", str(trial.get("adapter_type", "mlp")),
            "--adapter_heads", str(trial.get("adapter_heads", 8)),
            "--adapter_mlp_ratio", str(trial.get("adapter_mlp_ratio", 2.0)),
            "--lr", str(trial.get("lr", 5e-5)),
            "--batch_size", str(trial.get("batch_size", 1)),
            "--num_workers", str(trial.get("num_workers", 4)),
            "--max_steps", str(trial.get("steps", 1000)),
            "--save_every", str(trial.get("save_every", 500)),
            "--val_every", str(trial.get("val_every", 500)),
            "--num_views", num_views,
            "--num_queries", num_queries,
            "--save_ply_queries", str(trial.get("save_ply_queries", 40960)),
            "--loss_type", trial.get("loss", "nova_flow"),
        ] + (["--chamfer_weight", str(trial["chamfer_weight"])] if "chamfer_weight" in trial else []) + [
            "--amp",
            "--output_dir", str(out_dir),
        ] + (["--resume", str(trial["resume"])] if trial.get("resume") else []) + target_args(trial)
    raise ValueError(f"Unsupported trial kind={kind!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HARNESS_ROOT / "configs/phase1_targets.json"))
    ap.add_argument("--trial", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    defaults, trials = load_config(Path(args.config))
    if args.trial not in trials:
        raise SystemExit(f"Unknown trial {args.trial}; available: {sorted(trials)}")
    trial = {**defaults, **trials[args.trial]}
    out_dir = REPO_ROOT / trial.get("result_root", defaults.get("result_root", "experiments/probe3d/result/autoresearch_probe")) / trial["trial_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.out"
    cmd = build_command(defaults, trial, out_dir)
    (out_dir / "trial.json").write_text(json.dumps(trial, indent=2) + "\n")
    (out_dir / "command.txt").write_text(" ".join(cmd) + "\n")

    if args.dry_run:
        print(" ".join(cmd))
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(trial.get("cuda_visible_devices", defaults.get("cuda_visible_devices", "0")))
    if trial.get("distributed_backend"):
        env["TORCH_DISTRIBUTED_BACKEND"] = str(trial["distributed_backend"])
    if trial.get("kind") == "adapter" and int(trial.get("nproc_per_node", 1)) > 1:
        env.setdefault("TORCH_DISTRIBUTED_BACKEND", "gloo")
    start = time.time()
    with run_log.open("w", encoding="utf-8") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    runtime = time.time() - start
    kind = trial.get("kind")
    metric = summarize_oracle(out_dir) if kind == "oracle" else summarize_adapter(out_dir)
    status = "keep" if proc.returncode == 0 and metric is not None else "crash"
    append_result({
        "trial_id": trial["trial_id"], "kind": kind, "target_mode": trial.get("target_mode", ""),
        "adapter": trial.get("adapter", "tokens" if kind == "oracle" else ""), "loss": trial.get("loss", "oracle_flow" if kind == "oracle" else ""),
        "steps": trial.get("steps", ""), "val_cd": metric, "best_cd": metric,
        "runtime_sec": runtime, "status": status, "notes": trial.get("notes", ""), "out_dir": str(out_dir),
    })
    print(json.dumps({"trial_id": trial["trial_id"], "status": status, "metric": metric, "runtime_sec": runtime, "out_dir": str(out_dir)}, indent=2))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()

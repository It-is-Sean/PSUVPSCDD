from __future__ import annotations

import argparse
import csv
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the configured layer sweep plan.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    repo_root = Path(args.config).resolve().parents[2]
    models_cfg = OmegaConf.load(repo_root / str(cfg.configs.models))
    sweep_cfg = OmegaConf.load(repo_root / str(cfg.configs.sweeps))

    out_dir = repo_root / str(cfg.artifacts.tables)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer_sweep_plan.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "model", "layer"])
        for family, entries in models_cfg.models.items():
            for entry in entries:
                layers = entry.get("candidate_layers", sweep_cfg.sweeps.layers.report_priority)
                for layer in layers:
                    writer.writerow([family, entry["name"], layer])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

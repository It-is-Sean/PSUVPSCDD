from __future__ import annotations

import argparse
import csv
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the configured video timestep sweep plan.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    repo_root = Path(args.config).resolve().parents[2]
    models_cfg = OmegaConf.load(repo_root / str(cfg.configs.models))
    sweep_cfg = OmegaConf.load(repo_root / str(cfg.configs.sweeps))

    out_dir = repo_root / str(cfg.artifacts.tables)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "video_timestep_sweep_plan.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "timestep"])
        for entry in models_cfg.models.video:
            for t in sweep_cfg.sweeps.video_timesteps.search_space:
                writer.writerow([entry["name"], t])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

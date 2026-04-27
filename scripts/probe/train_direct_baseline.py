from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a direct-baseline training plan.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="nova3r")
    parser.add_argument("--layer", default="mid")
    parser.add_argument("--tag", default="dev")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    repo_root = Path(args.config).resolve().parents[2]
    run_dir = repo_root / str(cfg.artifacts.manifests) / "direct_probe" / f"{args.model}-{args.layer}-{args.tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "mode": "direct_probe",
        "model": args.model,
        "layer": args.layer,
        "tag": args.tag,
        "direct_config": str(cfg.configs.direct),
        "dataset_config": str(cfg.configs.datasets),
        "note": "Launcher stub only: wire actual baseline training here once feature extraction is implemented.",
    }
    (run_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Wrote direct baseline plan to {run_dir / 'plan.json'}")


if __name__ == "__main__":
    main()

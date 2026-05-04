from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize an evaluation plan for the probe workspace.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", default="dev")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    repo_root = Path(args.config).resolve().parents[2]
    out_dir = repo_root / str(cfg.artifacts.manifests) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "tag": args.tag,
        "metrics": [
            "chamfer_distance",
            "completeness_coverage",
            "normal_consistency",
            "visible_unseen_split",
            "pointmap_depth_pose_errors",
        ],
        "stress_config": str(cfg.configs.stress),
        "models_config": str(cfg.configs.models),
        "note": "Evaluation harness scaffold only. Replace with real checkpoint / prediction loading.",
    }
    path = out_dir / f"eval-{args.tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Wrote evaluation plan to {path}")


if __name__ == "__main__":
    main()

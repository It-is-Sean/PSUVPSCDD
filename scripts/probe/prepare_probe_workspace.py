from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the probe artifact tree and a manifest.")
    parser.add_argument("--config", required=True, help="Path to configs/probe/defaults.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    repo_root = Path(args.config).resolve().parents[2]

    created = []
    for _, rel_path in cfg.artifacts.items():
        path = repo_root / str(rel_path)
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path.relative_to(repo_root)))

    manifest_dir = repo_root / str(cfg.artifacts.manifests)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"workspace-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    manifest = {
        "project": OmegaConf.to_container(cfg.project, resolve=True),
        "created": created,
        "config": args.config,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created artifact tree and wrote {manifest_path}")


if __name__ == "__main__":
    main()

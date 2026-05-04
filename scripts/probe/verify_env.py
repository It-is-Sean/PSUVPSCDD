from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODULES = [
    "torch",
    "torchvision",
    "numpy",
    "einops",
    "omegaconf",
    "hydra",
    "accelerate",
    "diffusers",
    "flow_matching",
    "open3d",
    "imageio",
    "safetensors",
]

OPTIONAL_MODULES = [
    "pytorch3d",
    "torch_cluster",
    "chamferdist",
    "croco.models.curope",
]

IMPORT_PATHS = [
    "nova3r.models.nova3r_img_cond:Nova3rImgCond",
    "nova3r.inference:inference_nova3r",
    "nova3r.probe.canonical_decoder:FrozenCanonicalPointDecoder",
    "dust3r.datasets:get_data_loader",
    "scripts.probe.visualize_run:main",
]


def check_module(name: str) -> dict:
    try:
        mod = importlib.import_module(name)
        return {
            "name": name,
            "ok": True,
            "version": getattr(mod, "__version__", None),
            "file": getattr(mod, "__file__", None),
        }
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "ok": False, "error": repr(exc)}



def check_import_path(spec: str) -> dict:
    module_name, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        getattr(module, attr)
        return {"spec": spec, "ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"spec": spec, "ok": False, "error": repr(exc)}



def main() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    report = {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "repo_root": str(REPO_ROOT),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "modules": [check_module(name) for name in MODULES],
        "optional_modules": [check_module(name) for name in OPTIONAL_MODULES],
        "imports": [check_import_path(spec) for spec in IMPORT_PATHS],
    }

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    except Exception as exc:  # noqa: BLE001
        report["torch"] = {"error": repr(exc)}

    print(json.dumps(report, indent=2))

    hard_fail = []
    hard_fail.extend([m["name"] for m in report["modules"] if not m["ok"]])
    hard_fail.extend([s["spec"] for s in report["imports"] if not s["ok"]])
    if report.get("torch", {}).get("cuda_available") is False:
        hard_fail.append("torch.cuda")

    if hard_fail:
        print("\n[verify-env] missing critical pieces:", ", ".join(hard_fail), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

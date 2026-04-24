import os
import sys
from typing import Any, Dict, Iterable, List, Tuple

import torch
from omegaconf import OmegaConf


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_hydra_like_cli(argv: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    dotlist = []
    rest = []
    config_path = None
    config_name = "config"
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--config-path="):
            config_path = arg.split("=", 1)[1]
            i += 1
        elif arg == "--config-path" and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
        elif arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
            i += 1
        elif arg == "--config-name" and i + 1 < len(argv):
            config_name = argv[i + 1]
            i += 2
        elif arg.startswith("+") or (("=" in arg) and not arg.startswith("--")):
            dotlist.append(arg[1:] if arg.startswith("+") else arg)
            i += 1
        else:
            rest.append(arg)
            if arg.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith(("-", "+")):
                rest.append(argv[i + 1])
                i += 2
            else:
                i += 1
    cfg = OmegaConf.create()
    if config_path is not None:
        config_file = os.path.join(config_path, f"{config_name}.yaml")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Hydra config file not found: {config_file}")
        cfg = OmegaConf.load(config_file)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return cfg.get("experiment", cfg), rest


def ensure_eval_defaults(args):
    from eval.mv_recon.test_nova3r import _apply_eval_defaults

    cfg = OmegaConf.create(args)
    _apply_eval_defaults(cfg)
    if "amp_dtype" not in cfg:
        cfg.amp_dtype = "fp16"
    if "amp" not in cfg:
        cfg.amp = False
    return cfg


def build_nova3r_model(args, device):
    from nova3r.models.nova3r_img_cond import Nova3rImgCond  # noqa: F401

    if "model" not in args:
        raise KeyError(
            "No experiment.model config found. Run from the same Hydra config used by "
            "eval/mv_recon/test_nova3r.py or include the model config in overrides."
        )
    model_config = args.model
    model = eval(model_config["name"])(**model_config["params"])
    ckpt = torch.load(args.ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    print(model.load_state_dict(state, strict=False))
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def scrream_dataset_string(args):
    data_root = args.data_root
    datasets = {
        "scrream_n1": (
            "SCRREAM(split='train', ROOT='"
            f"{data_root}/eval_scrream', train_list_path='data/scrream/scrream_n1_list.json', "
            "test_list_path='data/scrream/scrream_n1_list.json', resolution=518, aug_crop=0, "
            "input_n=1, n_ldi_layers=0, enforce_img_reso_for_eval=[518,392], max_pts=100000)"
        ),
        "scrream_n2": (
            "SCRREAM_MULTI(split='train', ROOT='"
            f"{data_root}/eval_scrream', train_list_path='data/scrream/scrream_n2_list.json', "
            "test_list_path='data/scrream/scrream_n2_list.json', resolution=518, aug_crop=0, "
            "input_n=2, n_ldi_layers=1, enforce_img_reso_for_eval=[518,392], max_pts=100000)"
        ),
    }
    return datasets[args.test_dataset_name]


def build_scrream_loader(args, batch_size=1, num_workers=0):
    from eval.mv_recon.test_nova3r import build_dataset

    return build_dataset(args, scrream_dataset_string(args), batch_size, num_workers, test=True)


def move_batch_to_device(batch, device):
    ignore = {"dataset", "label", "instance", "idx", "true_shape", "rng", "view_label"}
    for view in batch:
        for key, value in list(view.items()):
            if key not in ignore and torch.is_tensor(value):
                view[key] = value.to(device, non_blocking=True)
    return batch


def images_from_batch(batch):
    imgs = []
    for view in batch:
        labels = view.get("view_label", [""])
        label = labels[0] if isinstance(labels, (list, tuple)) else str(labels)
        if "input" in label:
            imgs.append(view["img"] * 0.5 + 0.5)
    if not imgs:
        print("No input views found from view_label; using the first view image as a fallback.")
        imgs.append(batch[0]["img"] * 0.5 + 0.5)
    return torch.stack(imgs, dim=1)


def iter_named_tensors(obj: Any, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(obj):
        yield prefix.rstrip("."), obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from iter_named_tensors(value, f"{prefix}{key}.")
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield from iter_named_tensors(value, f"{prefix}{idx}.")


def print_tensor_tree(obj: Any, prefix: str = "output"):
    for name, tensor in iter_named_tensors(obj, prefix):
        print(f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")


def feature_candidates(obj: Any) -> List[Tuple[str, torch.Tensor]]:
    candidates = []
    for name, tensor in iter_named_tensors(obj):
        if tensor.ndim in (2, 3, 4) and tensor.shape[-1] >= 16 and tensor.is_floating_point():
            candidates.append((name, tensor))
    return candidates


def get_by_path(obj: Any, path: str):
    cur = obj
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(cur, dict):
            cur = cur[part]
        elif isinstance(cur, (list, tuple)):
            cur = cur[int(part)]
        else:
            raise KeyError(f"Cannot descend into {type(cur)} at {part} for path {path}")
    return cur


def scene_ids_from_batch(batch, start_idx: int, batch_size: int) -> List[str]:
    first = batch[0]
    for key in ("instance", "label", "idx"):
        value = first.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            return [str(x.item()) for x in value.view(-1)]
        if isinstance(value, (list, tuple)):
            return [str(x) for x in value]
        return [str(value)]
    return [f"sample_{start_idx + i:06d}" for i in range(batch_size)]


def extract_targets_from_batch(args, batch, target_key=None, max_points=None):
    if target_key:
        tensor = get_by_path(batch, target_key)
        if not torch.is_tensor(tensor):
            raise TypeError(f"--target_key {target_key} did not resolve to a tensor")
        return tensor

    try:
        from nova3r.inference import get_all_pts3d

        query_src = args.model.params.cfg.pts3d_head.params.get("query_source", "src_complete")
        targets, valid = get_all_pts3d(batch, mode=query_src)
        compact = []
        for pts, mask in zip(targets, valid):
            pts = pts[mask]
            if max_points is not None and pts.shape[0] > max_points:
                pts = pts[:max_points]
            compact.append(pts)
        min_len = min(p.shape[0] for p in compact)
        return torch.stack([p[:min_len] for p in compact], dim=0)
    except Exception as exc:
        print(f"Could not infer target points via nova3r.inference.get_all_pts3d: {exc}")
        print("Available batch tensors:")
        print_tensor_tree(batch, "batch")
        raise RuntimeError("Pass --target_key explicitly after inspecting the printed batch tensors.") from exc

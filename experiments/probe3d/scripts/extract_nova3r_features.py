import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from probe_nova3r_common import (  # noqa: E402
    build_nova3r_model,
    build_scrream_loader,
    ensure_eval_defaults,
    extract_targets_from_batch,
    feature_candidates,
    get_by_path,
    images_from_batch,
    move_batch_to_device,
    parse_hydra_like_cli,
    print_tensor_tree,
    scene_ids_from_batch,
)


def parse_args():
    exp, rest = parse_hydra_like_cli(sys.argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--feature_key", default=None)
    parser.add_argument("--target_key", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--max_target_points", type=int, default=8192)
    args = parser.parse_args(rest)
    return ensure_eval_defaults(exp), args


def main():
    exp, args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_nova3r_model(exp, device)
    loader = build_scrream_loader(exp, batch_size=args.batch_size, num_workers=args.num_workers)

    all_features = []
    all_targets = []
    scene_ids = []
    selected_key = args.feature_key

    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            images = images_from_batch(batch)
            outputs = {
                "forward": model(images),
                "encode": model._encode(images=images, test=True),
            }

        if selected_key is None:
            candidates = feature_candidates(outputs)
            print("Feature candidates:")
            for name, tensor in candidates:
                print(f"  {name}: shape={tuple(tensor.shape)}")
            if not candidates:
                print_tensor_tree(outputs)
                raise RuntimeError("No feature-like tensors found. Pass --feature_key after inspecting outputs.")
            selected_key = candidates[0][0]
            print(f"WARNING: --feature_key was not provided; using first candidate: {selected_key}")

        features = get_by_path(outputs, selected_key)
        if features.ndim == 4:
            features = features.flatten(1, 2)
        targets = extract_targets_from_batch(
            exp, batch, target_key=args.target_key, max_points=args.max_target_points
        )
        all_features.append(features.detach().float().cpu())
        all_targets.append(targets.detach().float().cpu())
        scene_ids.extend(scene_ids_from_batch(batch, len(scene_ids), features.shape[0]))
        print(f"Extracted batch {batch_idx}: features={tuple(features.shape)} targets={tuple(targets.shape)}")

    data = {
        "scene_ids": scene_ids,
        "features": torch.cat(all_features, dim=0),
        "target_points": torch.cat(all_targets, dim=0),
        "meta": {
            "backbone": "nova3r",
            "feature_key": selected_key,
            "note": "Frozen NOVA3R features extracted for probe.",
        },
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(data, args.output_path)
    print(f"Saved {len(scene_ids)} samples to {args.output_path}")


if __name__ == "__main__":
    main()


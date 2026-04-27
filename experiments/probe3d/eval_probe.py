import argparse
import os

import torch
from torch.utils.data import DataLoader

try:
    from probe.adapter import SmallAdapter
    from probe.dataset import FeaturePointDataset
    from probe.decoder import PointDecoder
    from probe.losses import chamfer_l2
except ImportError:
    from experiments.probe3d.probe.adapter import SmallAdapter
    from experiments.probe3d.probe.dataset import FeaturePointDataset
    from experiments.probe3d.probe.decoder import PointDecoder
    from experiments.probe3d.probe.losses import chamfer_l2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--output_dir", default="experiments/probe3d/outputs")
    return parser.parse_args()


def write_point_cloud_ply(path, points):
    points = points.detach().cpu().float()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for point in points.tolist():
            handle.write(f"{point[0]} {point[1]} {point[2]}\n")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt["args"]
    try:
        dataset = FeaturePointDataset(
            args.feature_path,
            pool_features=not train_args.get("no_pool_features", False),
            split=args.split,
        )
    except ValueError:
        dataset = FeaturePointDataset(args.feature_path, pool_features=not train_args.get("no_pool_features", False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    adapter = SmallAdapter(
        input_dim=ckpt["input_dim"],
        latent_dim=train_args["latent_dim"],
        depth=train_args["adapter_depth"],
    ).to(device)
    decoder = PointDecoder(
        latent_dim=train_args["latent_dim"],
        hidden_dim=train_args["hidden_dim"],
        num_points=train_args["num_points"],
    ).to(device)
    adapter.load_state_dict(ckpt["adapter"])
    decoder.load_state_dict(ckpt["decoder"])
    adapter.eval()
    decoder.eval()

    total = 0.0
    count = 0
    predictions = []
    target_points = []
    scene_ids = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device).flatten(1)
            targets = batch["target_points"].to(device)
            pred = decoder(adapter(features))
            loss = chamfer_l2(pred, targets)
            total += loss.item() * features.shape[0]
            count += features.shape[0]
            if args.save_predictions:
                predictions.append(pred.cpu())
                target_points.append(targets.cpu())
                scene_ids.extend(batch["scene_id"])

    avg = total / max(count, 1)
    print(f"Average Chamfer Distance: {avg:.6f}")
    if args.save_predictions:
        os.makedirs(args.output_dir, exist_ok=True)
        path = os.path.join(args.output_dir, "predictions.pt")
        pred_tensor = torch.cat(predictions, dim=0)
        target_tensor = torch.cat(target_points, dim=0)
        torch.save({"scene_ids": scene_ids, "pred_points": pred_tensor, "target_points": target_tensor}, path)
        print(f"Saved predictions to {path}")
        for scene_id, pred_points, gt_points in zip(scene_ids, pred_tensor, target_tensor):
            write_point_cloud_ply(os.path.join(args.output_dir, f"{scene_id}_pred.ply"), pred_points)
            write_point_cloud_ply(os.path.join(args.output_dir, f"{scene_id}_gt.ply"), gt_points)
        print(f"Saved PLY point clouds to {args.output_dir}")


if __name__ == "__main__":
    main()

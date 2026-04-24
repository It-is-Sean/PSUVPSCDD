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
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--adapter_depth", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_pool_features", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = FeaturePointDataset(args.feature_path, pool_features=not args.no_pool_features)
    sample = dataset[0]["features"]
    input_dim = sample.numel() if sample.ndim > 1 else sample.shape[-1]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    adapter = SmallAdapter(input_dim=input_dim, latent_dim=args.latent_dim, depth=args.adapter_depth).to(device)
    decoder = PointDecoder(args.latent_dim, args.hidden_dim, args.num_points).to(device)
    optim = torch.optim.AdamW(
        list(adapter.parameters()) + list(decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        adapter.train()
        decoder.train()
        total = 0.0
        count = 0
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["target_points"].to(device)
            features = features.flatten(1)
            pred = decoder(adapter(features))
            loss = chamfer_l2(pred, targets)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += loss.item() * features.shape[0]
            count += features.shape[0]
        print(f"epoch {epoch + 1:04d}/{args.epochs} loss={total / max(count, 1):.6f}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "decoder": decoder.state_dict(),
            "args": vars(args),
            "input_dim": input_dim,
        },
        args.save_path,
    )
    print(f"Saved probe checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()


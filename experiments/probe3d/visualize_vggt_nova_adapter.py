from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from probe.adapter import VGGTToNovaAdapter
from vggt_nova_adapter_common import (
    DEFAULT_NOVA_CKPT,
    build_decoder,
    build_loader,
    chamfer_l2,
    extract_vggt_features,
    get_targets,
    images_from_batch,
    load_vggt,
    move_batch_to_device,
    resolve_device,
    sample_decoder,
    save_json,
    scene_ids_from_batch,
    select_vggt_layer23,
    set_seed,
    write_point_cloud_ply,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--nova_ckpt", default=str(DEFAULT_NOVA_CKPT))
    parser.add_argument("--dataset", default="scrream_adapter", choices=("scrream_adapter", "scannet"))
    parser.add_argument("--data_root", default=None, help="Dataset root for --dataset scannet; default /data1/jcd_data/scannet_processed_large")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--num_queries", type=int, default=4096)
    return parser.parse_args()


def save_input_images(images: torch.Tensor, out_dir: Path, scene_id: str) -> None:
    imgs = images[0].detach().cpu().clamp(0, 1)
    for i, img in enumerate(imgs):
        arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(arr).save(out_dir / f"{scene_id}_input{i}.png")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    config = ckpt["config"]
    decoder, meta, cfg = build_decoder(device, args.nova_ckpt)
    loader, _ = build_loader(cfg, args.batch_size, args.num_workers, test=True, dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views)
    vggt = load_vggt(device)
    adapter = VGGTToNovaAdapter(
        input_dim=config["selected_vggt_feature_shape"][-1],
        output_dim=meta["token_dim"],
        output_tokens=meta["num_scene_tokens"],
        hidden_dim=config["adapter_hidden_dim"],
        adapter_layers=config["adapter_layers"],
    ).to(device)
    adapter.load_state_dict(ckpt["adapter"])
    adapter.eval()

    metrics = []
    saved = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if saved >= args.num_samples:
                break
            batch = move_batch_to_device(batch, device)
            images = images_from_batch(batch)
            features, _ = extract_vggt_features(vggt, images, amp=args.amp)
            selected, selected_idx, reason = select_vggt_layer23(features)
            tokens = adapter(selected)
            pred = sample_decoder(decoder, tokens, args.num_queries, meta["fm_step_size"], args.seed + batch_idx, images.shape[1])
            target = get_targets(batch, meta["query_source"], max_points=args.num_queries)
            loss = chamfer_l2(pred, target)
            scene_ids = scene_ids_from_batch(batch, saved)
            scene_id = scene_ids[0]
            write_point_cloud_ply(output_dir / f"{scene_id}_pred.ply", pred[0])
            write_point_cloud_ply(output_dir / f"{scene_id}_pseudo_gt.ply", target[0])
            save_input_images(images, output_dir, scene_id)
            metrics.append(
                {
                    "scene_id": scene_id,
                    "chamfer_l2": float(loss.item()),
                    "pred_ply": str(output_dir / f"{scene_id}_pred.ply"),
                    "gt_ply": str(output_dir / f"{scene_id}_pseudo_gt.ply"),
                    "selected_vggt_feature_index": selected_idx,
                    "selection_reason": reason,
                }
            )
            print(f"saved {scene_id}: chamfer_l2={loss.item():.8f}")
            saved += 1
    save_json(output_dir / "metrics.json", {"samples": metrics})
    print(f"Saved visualization outputs to {output_dir}")


if __name__ == "__main__":
    main()

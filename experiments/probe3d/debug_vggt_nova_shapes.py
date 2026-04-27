from __future__ import annotations

import argparse
from pathlib import Path

from probe.adapter import VGGTToNovaAdapter
from vggt_nova_adapter_common import (
    DEFAULT_NOVA_CKPT,
    assert_only_adapter_trainable,
    build_decoder,
    build_loader,
    count_parameters,
    extract_vggt_features,
    get_targets,
    images_from_batch,
    load_vggt,
    move_batch_to_device,
    print_feature_shapes,
    resolve_device,
    sample_decoder,
    select_vggt_layer23,
    set_seed,
    trainable_parameter_names,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--adapter_layers", type=int, default=2, )
    parser.add_argument("--adapter_hidden_dim", type=int, default=1024)
    parser.add_argument("--nova_ckpt", default=str(DEFAULT_NOVA_CKPT))
    parser.add_argument("--dataset", default="scrream_adapter", choices=("scrream_adapter", "scannet"))
    parser.add_argument("--data_root", default=None, help="Dataset root for --dataset scannet; default /data1/jcd_data/scannet_processed_large")
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--num_queries", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    decoder, meta, cfg = build_decoder(device, args.nova_ckpt)
    loader, data_args = build_loader(cfg, args.batch_size, args.num_workers, test=True, dataset_name=args.dataset, data_root=args.data_root, seed=args.seed, num_views=args.num_views)
    vggt = load_vggt(device)

    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)
    images = images_from_batch(batch)
    print(f"Input images shape: {tuple(images.shape)}")

    features, patch_start_idx = extract_vggt_features(vggt, images, amp=args.amp)
    print(f"VGGT patch_start_idx: {patch_start_idx}")
    print_feature_shapes(features)
    selected, selected_idx, reason = select_vggt_layer23(features)
    print(f"Selected VGGT feature index: {selected_idx}")
    print(f"Selection reason: {reason}")
    print(f"Selected VGGT feature shape: {tuple(selected.shape)}")

    adapter = VGGTToNovaAdapter(
        input_dim=selected.shape[-1],
        output_dim=meta["token_dim"],
        output_tokens=meta["num_scene_tokens"],
        hidden_dim=args.adapter_hidden_dim,
        adapter_layers=args.adapter_layers,
    ).to(device)
    adapter.train()
    adapter_out = adapter(selected)
    print(f"Adapter output shape: {tuple(adapter_out.shape)}")

    pred = sample_decoder(
        decoder,
        adapter_out,
        num_queries=args.num_queries,
        step_size=meta["fm_step_size"],
        seed=args.seed,
        num_views=images.shape[1],
    )
    print(f"Decoder sampled output shape: {tuple(pred.shape)}")

    targets = get_targets(batch, meta["query_source"], max_points=args.num_queries)
    print(f"Target pointcloud shape: {tuple(targets.shape)}")

    names = trainable_parameter_names({"adapter": adapter, "vggt": vggt, "decoder": decoder})
    print("Trainable parameter names:")
    for name in names:
        print(f"  {name}")
    print(f"Trainable adapter parameter count: {count_parameters(adapter)}")
    assert_only_adapter_trainable(adapter, vggt, decoder)
    print("Assertion passed: only adapter parameters are trainable.")
    print(f"Reference NOVA checkpoint: {Path(args.nova_ckpt)}")
    print(f"Dataset root: {data_args.data_root}, dataset={data_args.test_dataset_name}")


if __name__ == "__main__":
    main()

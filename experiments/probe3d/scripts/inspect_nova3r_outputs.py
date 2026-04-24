import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from probe_nova3r_common import (  # noqa: E402
    build_nova3r_model,
    build_scrream_loader,
    ensure_eval_defaults,
    feature_candidates,
    images_from_batch,
    parse_hydra_like_cli,
    print_tensor_tree,
)


def main():
    exp, _ = parse_hydra_like_cli(sys.argv[1:])
    args = ensure_eval_defaults(exp)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_nova3r_model(args, device)
    loader = build_scrream_loader(args, batch_size=1, num_workers=0)
    batch = next(iter(loader))

    with torch.no_grad():
        for view in batch:
            for key, value in list(view.items()):
                if torch.is_tensor(value):
                    view[key] = value.to(device)
        images = images_from_batch(batch)
        outputs = {
            "forward": model(images),
            "encode": model._encode(images=images, test=True),
        }

    print("Output tensor tree:")
    print_tensor_tree(outputs)
    print("Feature candidates:")
    for name, tensor in feature_candidates(outputs):
        print(f"  {name}: shape={tuple(tensor.shape)}")


if __name__ == "__main__":
    main()


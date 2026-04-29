#!/usr/bin/env python3
"""Smoke-test DDP initialization for probe3d training scripts.

This was originally a temporary helper. Keep it under autoresearch_probe/debug/
so one-off diagnostics do not clutter the main experiment package.
"""

from vggt_nova_adapter_common_raw import cleanup_distributed, init_distributed_mode


if __name__ == "__main__":
    ctx = init_distributed_mode()
    print("rank", ctx["rank"], "world", ctx["world_size"], "device", ctx["device"], flush=True)
    cleanup_distributed()

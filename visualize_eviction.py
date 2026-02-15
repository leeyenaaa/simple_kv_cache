#!/usr/bin/env python3
"""
Visualize KV cache eviction patterns as a heatmap.

Y-axis: layer index (0 = bottom)
X-axis: position index (token slot in cache)
Blue  = kept
Gray  = evicted

This script simulates eviction without requiring GPU — it only needs the
eviction logic from simple_evict.py.

Usage:
    # Default: 32 layers, 4096 total tokens, sink=256, recent=1024, mid=2816
    python visualize_eviction.py

    # Custom parameters
    python visualize_eviction.py --total_len 8192 --num_layers 32 \\
        --strategy sink_recent_uniform \\
        --sink_tokens 256 --recent_tokens 1024 --middle_budget 2816

    # Sink + Recent only (no middle)
    python visualize_eviction.py --strategy sink_recent

    # Save to file instead of showing
    python visualize_eviction.py --save eviction_pattern.png
"""

import argparse
import os
import sys

import numpy as np

# Add LaCache path for eviction imports
LACACHE_LONGBENCH_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "LaCache", "LongBench")
)
sys.path.insert(0, LACACHE_LONGBENCH_DIR)

try:
    import torch
    from evict.simple_evict import SimpleEvictConfig, build_simple_keep_token_idx
except ImportError as e:
    print(f"ERROR: {e}")
    print("Please activate the LaCache venv: source ../LaCache/.venv/bin/activate")
    sys.exit(1)


BLOCK_N = 128


def simulate_eviction(
    total_len: int,
    num_layers: int,
    cfg: SimpleEvictConfig,
    block_n: int = BLOCK_N,
) -> np.ndarray:
    """
    Simulate eviction and return a binary mask [num_layers, total_len].

    1 = kept, 0 = evicted.
    Currently all layers share the same keep_idx.
    """
    keep_idx = build_simple_keep_token_idx(
        total_len=total_len,
        block_n=block_n,
        cfg=cfg,
        device=torch.device("cpu"),
    )
    keep_set = set(keep_idx.tolist())

    mask = np.zeros((num_layers, total_len), dtype=np.uint8)
    for pos in keep_set:
        mask[:, pos] = 1

    return mask


def plot_heatmap(
    mask: np.ndarray,
    cfg: SimpleEvictConfig,
    total_len: int,
    save_path: str | None = None,
) -> None:
    """
    Render the eviction heatmap.

    Blue = kept, light gray = evicted.
    Annotates sink / middle / recent regions.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    num_layers, seq_len = mask.shape

    # Colors: 0 = evicted (light gray), 1 = kept (blue)
    cmap = ListedColormap(["#D9D9D9", "#3274A1"])

    fig, ax = plt.subplots(figsize=(16, max(4, num_layers * 0.22)))
    ax.imshow(mask, aspect="auto", cmap=cmap, interpolation="none", origin="lower")

    # --- Region boundary lines ---
    sink_end = min(total_len, cfg.sink_tokens)
    recent_start = max(0, total_len - cfg.recent_tokens)

    ax.axvline(x=sink_end - 0.5, color="#E74C3C", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.axvline(x=recent_start - 0.5, color="#2ECC71", linewidth=1.5, linestyle="--", alpha=0.8)

    # --- Labels ---
    ax.set_xlabel("Position Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)

    strategy_label = cfg.middle_strategy
    if strategy_label == "uniform":
        mid_info = f"mid={cfg.middle_budget_tokens}"
        if cfg.uniform_stride > 0:
            mid_info = f"stride={cfg.uniform_stride}"
    else:
        mid_info = "no mid"

    kept_count = int(mask[0].sum())
    evicted_count = total_len - kept_count

    ax.set_title(
        f"KV Cache Eviction Pattern\n"
        f"total={total_len}  sink={cfg.sink_tokens}  recent={cfg.recent_tokens}  "
        f"{mid_info}  |  kept={kept_count}  evicted={evicted_count}",
        fontsize=13,
        fontweight="bold",
    )

    # Y-axis ticks: show every 4 layers
    if num_layers <= 16:
        ax.set_yticks(range(num_layers))
    else:
        step = max(1, num_layers // 8)
        ax.set_yticks(range(0, num_layers, step))

    # X-axis ticks: reasonable spacing
    if total_len <= 2048:
        ax.set_xticks(range(0, total_len, 256))
    elif total_len <= 8192:
        ax.set_xticks(range(0, total_len, 512))
    else:
        ax.set_xticks(range(0, total_len, 1024))

    # Legend
    legend_elements = [
        Patch(facecolor="#3274A1", label="Kept"),
        Patch(facecolor="#D9D9D9", label="Evicted"),
        Patch(facecolor="none", edgecolor="#E74C3C", linestyle="--", label=f"Sink boundary ({sink_end})"),
        Patch(facecolor="none", edgecolor="#2ECC71", linestyle="--", label=f"Recent boundary ({recent_start})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize KV cache eviction pattern as a heatmap",
    )
    parser.add_argument("--total_len", type=int, default=4096,
                        help="Total number of tokens before eviction (default: 4096)")
    parser.add_argument("--num_layers", type=int, default=32,
                        help="Number of transformer layers (default: 32)")
    parser.add_argument("--strategy", type=str,
                        choices=["sink_recent", "sink_recent_uniform"],
                        default="sink_recent_uniform",
                        help="Eviction strategy (default: sink_recent_uniform)")
    parser.add_argument("--sink_tokens", type=int, default=256,
                        help="Sink tokens (default: 256)")
    parser.add_argument("--recent_tokens", type=int, default=1024,
                        help="Recent tokens (default: 1024)")
    parser.add_argument("--middle_budget", type=int, default=2816,
                        help="Middle budget tokens for uniform (default: 2816)")
    parser.add_argument("--uniform_stride", type=int, default=0,
                        help="Keep every N-th block in middle (default: 0 = use budget)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save to file instead of showing (e.g. eviction.png)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.strategy == "sink_recent":
        cfg = SimpleEvictConfig(
            sink_tokens=args.sink_tokens,
            recent_tokens=args.recent_tokens,
            middle_strategy="none",
        )
    else:
        cfg = SimpleEvictConfig(
            sink_tokens=args.sink_tokens,
            recent_tokens=args.recent_tokens,
            middle_strategy="uniform",
            middle_budget_tokens=args.middle_budget,
            uniform_stride=args.uniform_stride,
        )

    target = cfg.sink_tokens + cfg.recent_tokens
    if cfg.middle_strategy == "uniform":
        target += cfg.middle_budget_tokens

    if args.total_len <= target:
        print(f"total_len ({args.total_len}) <= target ({target}), no eviction needed.")
        print("Increase --total_len or decrease sink/recent/middle to see eviction.")
        return 1

    print(f"Simulating eviction: {args.total_len} tokens, {args.num_layers} layers")
    print(f"  Strategy: {args.strategy}")
    print(f"  Sink: {args.sink_tokens}, Recent: {args.recent_tokens}", end="")
    if args.strategy == "sink_recent_uniform":
        print(f", Mid budget: {args.middle_budget}", end="")
    print()

    mask = simulate_eviction(
        total_len=args.total_len,
        num_layers=args.num_layers,
        cfg=cfg,
    )

    kept = int(mask[0].sum())
    print(f"  Kept: {kept}/{args.total_len} ({100*kept/args.total_len:.1f}%)")
    print(f"  Evicted: {args.total_len - kept}/{args.total_len}")

    plot_heatmap(mask, cfg, args.total_len, save_path=args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())

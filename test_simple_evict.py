#!/usr/bin/env python3
"""
Test script for KV cache eviction strategies.

Demonstrates both eviction strategies from LaCache:
1. Sink + Recent only (middle_strategy="none")
2. Sink + Recent + Uniform Middle (middle_strategy="uniform")

This script uses dummy data to test the token index selection logic
without requiring GPU or model loading.
"""

import sys
sys.path.insert(0, '../LaCache/LongBench')

from evict.simple_evict import SimpleEvictConfig, build_simple_keep_token_idx


def format_indices(indices, max_display=10):
    """Format indices for readable output, showing ranges and samples."""
    if len(indices) == 0:
        return "[]"
    
    indices_list = sorted(indices.tolist())
    
    # If small enough, show all
    if len(indices_list) <= max_display:
        return str(indices_list)
    
    # Otherwise show first few, last few, and count
    first = indices_list[:3]
    last = indices_list[-3:]
    return f"[{first[0]}-{first[-1]}, ..., {last[0]}-{last[-1]}] ({len(indices_list)} total)"


def test_strategy_1():
    """Test Strategy 1: Sink + Recent only."""
    print("=" * 70)
    print("STRATEGY 1: Sink + Recent Only (middle_strategy='none')")
    print("=" * 70)
    
    # Configuration: keep first 256 tokens + last 512 tokens
    config = SimpleEvictConfig(
        sink_tokens=256,
        recent_tokens=512,
        middle_strategy="none"
    )
    
    total_tokens = 2000
    block_size = 128
    
    # Build indices of tokens to keep
    keep_idx = build_simple_keep_token_idx(
        total_len=total_tokens,
        block_n=block_size,
        cfg=config
    )
    
    kept_count = keep_idx.numel()
    kept_percent = (kept_count / total_tokens) * 100
    saved_percent = 100 - kept_percent
    
    print(f"\nConfiguration:")
    print(f"  Sink tokens:   {config.sink_tokens}")
    print(f"  Recent tokens: {config.recent_tokens}")
    print(f"  Middle strategy: {config.middle_strategy}")
    
    print(f"\nResults:")
    print(f"  Total tokens:     {total_tokens}")
    print(f"  Kept tokens:      {kept_count} ({kept_percent:.1f}%)")
    print(f"  Evicted tokens:   {total_tokens - kept_count} ({saved_percent:.1f}%)")
    print(f"  Kept indices:     {format_indices(keep_idx)}")
    
    return keep_idx


def test_strategy_2():
    """Test Strategy 2: Sink + Recent + Uniform Middle."""
    print("\n" + "=" * 70)
    print("STRATEGY 2: Sink + Recent + Uniform Middle (middle_strategy='uniform')")
    print("=" * 70)
    
    # Configuration: keep sink + recent + uniformly-spaced middle tokens
    config = SimpleEvictConfig(
        sink_tokens=256,
        recent_tokens=512,
        middle_strategy="uniform",
        middle_budget_tokens=256  # Keep ~256 tokens from middle region
    )
    
    total_tokens = 2000
    block_size = 128
    
    # Build indices of tokens to keep
    keep_idx = build_simple_keep_token_idx(
        total_len=total_tokens,
        block_n=block_size,
        cfg=config
    )
    
    kept_count = keep_idx.numel()
    kept_percent = (kept_count / total_tokens) * 100
    saved_percent = 100 - kept_percent
    
    print(f"\nConfiguration:")
    print(f"  Sink tokens:        {config.sink_tokens}")
    print(f"  Recent tokens:      {config.recent_tokens}")
    print(f"  Middle strategy:    {config.middle_strategy}")
    print(f"  Middle budget:      {config.middle_budget_tokens}")
    
    print(f"\nResults:")
    print(f"  Total tokens:     {total_tokens}")
    print(f"  Kept tokens:      {kept_count} ({kept_percent:.1f}%)")
    print(f"  Evicted tokens:   {total_tokens - kept_count} ({saved_percent:.1f}%)")
    print(f"  Kept indices:     {format_indices(keep_idx)}")
    
    return keep_idx


def compare_strategies(keep_idx_1, keep_idx_2):
    """Compare the two strategies."""
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    count_1 = keep_idx_1.numel()
    count_2 = keep_idx_2.numel()
    
    print(f"\nStrategy 1 (Sink+Recent):")
    print(f"  Kept tokens: {count_1}")
    
    print(f"\nStrategy 2 (Sink+Recent+Uniform):")
    print(f"  Kept tokens: {count_2}")
    
    print(f"\nDifference:")
    print(f"  Strategy 2 keeps {count_2 - count_1} more tokens")
    print(f"  Strategy 2 memory overhead: {((count_2 - count_1) / count_1) * 100:.1f}%")
    
    # Find unique tokens in each strategy
    set_1 = set(keep_idx_1.tolist())
    set_2 = set(keep_idx_2.tolist())
    
    only_in_1 = set_1 - set_2
    only_in_2 = set_2 - set_1
    in_both = set_1 & set_2
    
    print(f"\nToken overlap:")
    print(f"  In both strategies: {len(in_both)}")
    print(f"  Only in Strategy 1: {len(only_in_1)}")
    print(f"  Only in Strategy 2: {len(only_in_2)}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("KV CACHE EVICTION STRATEGY TEST")
    print("=" * 70)
    
    # Test both strategies
    keep_idx_1 = test_strategy_1()
    keep_idx_2 = test_strategy_2()
    
    # Compare results
    compare_strategies(keep_idx_1, keep_idx_2)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

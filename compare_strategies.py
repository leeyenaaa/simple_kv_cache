#!/usr/bin/env python3
"""
Compare KV cache eviction strategy results.

This script loads JSON results from run_longbench_example.py and compares metrics
between different eviction strategies. It generates a formatted comparison table
and optional visualizations.

Usage:
    # Compare two strategies
    python compare_strategies.py results_sink_recent.json results_sink_recent_uniform.json
    
    # Compare with visualizations
    python compare_strategies.py results_sr.json results_sru.json --plot
    
    # Compare multiple strategies
    python compare_strategies.py sr.json sru.json stride2.json --plot

Expected JSON Format (from run_longbench_example.py):
    {
        "config": {
            "model": "...",
            "strategy": "sink_recent" or "sink_recent_uniform",
            "sink_tokens": 256,
            "recent_tokens": 512,
            "middle_budget_tokens": 256
        },
        "summary": {
            "total_examples": 10,
            "successful": 10,
            "avg_prompt_length": 4567.8,
            "avg_output_length": 123.4,
            "avg_prefill_sec": 1.23,
            "avg_decode_sec": 0.45,
            "avg_total_sec": 1.68
        },
        "results": [...]
    }
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def load_results(file_path: str) -> Dict[str, Any]:
    """Load JSON results from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def compute_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute comparison metrics from a result dictionary.
    
    Returns:
        Dictionary with computed metrics:
        - kept_tokens: Number of tokens kept by eviction strategy
        - retention_rate: Percentage of tokens retained
        - memory_saved: Percentage of memory saved
        - avg_prefill_sec: Average prefill time
        - avg_decode_sec: Average decode time
        - avg_total_sec: Average total time
        - tokens_per_sec: Throughput metric
    """
    config = result.get('config', {})
    summary = result.get('summary', {})
    
    # Calculate kept tokens based on strategy
    sink = config.get('sink_tokens', 0)
    recent = config.get('recent_tokens', 0)
    middle_budget = config.get('middle_budget_tokens', 0)
    
    kept_tokens = sink + recent + middle_budget
    avg_prompt = summary.get('avg_prompt_length', 1)
    
    retention_rate = (kept_tokens / avg_prompt * 100) if avg_prompt > 0 else 0
    memory_saved = 100 - retention_rate
    
    avg_prefill = summary.get('avg_prefill_sec', 0)
    avg_decode = summary.get('avg_decode_sec', 0)
    avg_total = summary.get('avg_total_sec', 0)
    avg_output = summary.get('avg_output_length', 1)
    
    tokens_per_sec = (avg_output / avg_total) if avg_total > 0 else 0
    
    return {
        'kept_tokens': kept_tokens,
        'retention_rate': retention_rate,
        'memory_saved': memory_saved,
        'avg_prefill_sec': avg_prefill,
        'avg_decode_sec': avg_decode,
        'avg_total_sec': avg_total,
        'tokens_per_sec': tokens_per_sec,
        'avg_prompt_length': avg_prompt,
        'avg_output_length': avg_output,
    }


def format_strategy_name(result: Dict[str, Any]) -> str:
    """Format a human-readable strategy name."""
    config = result.get('config', {})
    strategy = config.get('strategy', 'unknown')
    
    if strategy == 'sink_recent':
        return 'Sink+Recent (SR)'
    elif strategy == 'sink_recent_uniform':
        middle = config.get('middle_budget_tokens', 0)
        return f'Sink+Recent+Uniform (SRU, mid={middle})'
    else:
        return strategy


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table of strategies."""
    if not results:
        print("No results to compare.", file=sys.stderr)
        return
    
    # Compute metrics for all results
    metrics_list = [compute_metrics(r) for r in results]
    strategy_names = [format_strategy_name(r) for r in results]
    
    # Get model name
    model = results[0].get('config', {}).get('model', 'Unknown')
    
    print("\n" + "=" * 100)
    print(f"Strategy Comparison - Model: {model}")
    print("=" * 100)
    
    # Header
    header_parts = ["Metric"]
    for name in strategy_names:
        header_parts.append(f"| {name:20s}")
    if len(results) > 1:
        header_parts.append("| Delta (vs first)")
    
    header = " ".join(header_parts)
    print(header)
    print("-" * len(header))
    
    # Helper function to format values with delta
    def format_row(label: str, values: List[str], deltas: Optional[List[str]] = None) -> str:
        row_parts = [f"{label:30s}"]
        for val in values:
            row_parts.append(f"| {val:20s}")
        if deltas:
            for delta in deltas:
                row_parts.append(f"| {delta:15s}")
        return " ".join(row_parts)
    
    # Model
    print(format_row("Model", [model] * len(results)))
    
    # Prompt length
    prompt_vals = [f"{m['avg_prompt_length']:.0f} tokens" for m in metrics_list]
    print(format_row("Avg Prompt Length", prompt_vals))
    
    # Kept tokens
    kept_vals = []
    kept_deltas = []
    for i, m in enumerate(metrics_list):
        kept_vals.append(f"{m['kept_tokens']:.0f} ({m['retention_rate']:.1f}%)")
        if i > 0:
            delta_tokens = m['kept_tokens'] - metrics_list[0]['kept_tokens']
            delta_pct = (delta_tokens / metrics_list[0]['kept_tokens'] * 100) if metrics_list[0]['kept_tokens'] > 0 else 0
            kept_deltas.append(f"+{delta_tokens:.0f} ({delta_pct:+.1f}%)" if delta_tokens >= 0 else f"{delta_tokens:.0f} ({delta_pct:+.1f}%)")
    print(format_row("Kept Tokens", kept_vals, kept_deltas if len(results) > 1 else None))
    
    # Memory saved
    mem_vals = [f"{m['memory_saved']:.1f}%" for m in metrics_list]
    mem_deltas = []
    for i, m in enumerate(metrics_list):
        if i > 0:
            delta = m['memory_saved'] - metrics_list[0]['memory_saved']
            mem_deltas.append(f"{delta:+.1f}%")
    print(format_row("Memory Saved", mem_vals, mem_deltas if len(results) > 1 else None))
    
    # Prefill time
    prefill_vals = [f"{m['avg_prefill_sec']:.3f}s" for m in metrics_list]
    prefill_deltas = []
    for i, m in enumerate(metrics_list):
        if i > 0:
            delta = m['avg_prefill_sec'] - metrics_list[0]['avg_prefill_sec']
            pct = (delta / metrics_list[0]['avg_prefill_sec'] * 100) if metrics_list[0]['avg_prefill_sec'] > 0 else 0
            prefill_deltas.append(f"{delta:+.3f}s ({pct:+.1f}%)")
    print(format_row("Avg Prefill Time", prefill_vals, prefill_deltas if len(results) > 1 else None))
    
    # Decode time
    decode_vals = [f"{m['avg_decode_sec']:.3f}s" for m in metrics_list]
    decode_deltas = []
    for i, m in enumerate(metrics_list):
        if i > 0:
            delta = m['avg_decode_sec'] - metrics_list[0]['avg_decode_sec']
            pct = (delta / metrics_list[0]['avg_decode_sec'] * 100) if metrics_list[0]['avg_decode_sec'] > 0 else 0
            decode_deltas.append(f"{delta:+.3f}s ({pct:+.1f}%)")
    print(format_row("Avg Decode Time", decode_vals, decode_deltas if len(results) > 1 else None))
    
    # Total time
    total_vals = [f"{m['avg_total_sec']:.3f}s" for m in metrics_list]
    total_deltas = []
    for i, m in enumerate(metrics_list):
        if i > 0:
            delta = m['avg_total_sec'] - metrics_list[0]['avg_total_sec']
            pct = (delta / metrics_list[0]['avg_total_sec'] * 100) if metrics_list[0]['avg_total_sec'] > 0 else 0
            total_deltas.append(f"{delta:+.3f}s ({pct:+.1f}%)")
    print(format_row("Avg Total Time", total_vals, total_deltas if len(results) > 1 else None))
    
    # Tokens per second
    tps_vals = [f"{m['tokens_per_sec']:.2f} tok/s" for m in metrics_list]
    tps_deltas = []
    for i, m in enumerate(metrics_list):
        if i > 0:
            delta = m['tokens_per_sec'] - metrics_list[0]['tokens_per_sec']
            pct = (delta / metrics_list[0]['tokens_per_sec'] * 100) if metrics_list[0]['tokens_per_sec'] > 0 else 0
            tps_deltas.append(f"{delta:+.2f} ({pct:+.1f}%)")
    print(format_row("Tokens/Second", tps_vals, tps_deltas if len(results) > 1 else None))

    # Evaluation score (if available)
    scores = []
    for r in results:
        eval_data = r.get('evaluation', {})
        score = eval_data.get('score') or r.get('summary', {}).get('score')
        scores.append(score)

    if any(s is not None for s in scores):
        metric_name = None
        for r in results:
            eval_data = r.get('evaluation', {})
            metric_name = eval_data.get('metric') or r.get('summary', {}).get('metric')
            if metric_name:
                break
        label = f"Score ({metric_name or '?'})"

        score_vals = [f"{s:.2f}" if s is not None else "N/A" for s in scores]
        score_deltas = []
        for i, s in enumerate(scores):
            if i > 0 and s is not None and scores[0] is not None:
                delta = s - scores[0]
                score_deltas.append(f"{delta:+.2f}")
            elif i > 0:
                score_deltas.append("N/A")
        print("-" * len(header))
        print(format_row(label, score_vals, score_deltas if len(results) > 1 else None))

    print("=" * 100 + "\n")


def plot_comparison(results: List[Dict[str, Any]]) -> None:
    """Generate comparison visualizations using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available. Skipping visualizations.", file=sys.stderr)
        return
    
    if not results:
        return
    
    metrics_list = [compute_metrics(r) for r in results]
    strategy_names = [format_strategy_name(r) for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('KV Cache Eviction Strategy Comparison', fontsize=16, fontweight='bold')
    
    # 1. Token Retention
    ax = axes[0, 0]
    retention_rates = [m['retention_rate'] for m in metrics_list]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(results)]
    ax.bar(strategy_names, retention_rates, color=colors)
    ax.set_ylabel('Retention Rate (%)')
    ax.set_title('Token Retention Rate')
    ax.set_ylim(0, 100)
    for i, v in enumerate(retention_rates):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    
    # 2. Memory Saved
    ax = axes[0, 1]
    memory_saved = [m['memory_saved'] for m in metrics_list]
    ax.bar(strategy_names, memory_saved, color=colors)
    ax.set_ylabel('Memory Saved (%)')
    ax.set_title('Memory Efficiency')
    ax.set_ylim(0, 100)
    for i, v in enumerate(memory_saved):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
    
    # 3. Latency Breakdown
    ax = axes[1, 0]
    x = np.arange(len(strategy_names))
    width = 0.35
    prefill_times = [m['avg_prefill_sec'] for m in metrics_list]
    decode_times = [m['avg_decode_sec'] for m in metrics_list]
    ax.bar(x - width/2, prefill_times, width, label='Prefill', color='#1f77b4')
    ax.bar(x + width/2, decode_times, width, label='Decode', color='#ff7f0e')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Latency Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names)
    ax.legend()
    
    # 4. Score or Total Time
    ax = axes[1, 1]
    scores = []
    for r in results:
        eval_data = r.get('evaluation', {})
        score = eval_data.get('score') or r.get('summary', {}).get('score')
        scores.append(score)

    if any(s is not None for s in scores):
        score_vals = [s if s is not None else 0 for s in scores]
        ax.bar(strategy_names, score_vals, color=colors)
        ax.set_ylabel('Score (0-100)')
        ax.set_title('LongBench Score')
        ax.set_ylim(0, 100)
        for i, v in enumerate(score_vals):
            if scores[i] is not None:
                ax.text(i, v + 2, f'{v:.2f}', ha='center', va='bottom')
    else:
        total_times = [m['avg_total_sec'] for m in metrics_list]
        ax.bar(strategy_names, total_times, color=colors)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Total Inference Time')
        for i, v in enumerate(total_times):
            ax.text(i, v + 0.05, f'{v:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'strategy_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    try:
        plt.show()
    except Exception:
        pass  # Headless environment


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compare KV cache eviction strategy results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two strategies
  python compare_strategies.py results_sr.json results_sru.json
  
  # Compare with visualizations
  python compare_strategies.py sr.json sru.json --plot
  
  # Compare multiple strategies
  python compare_strategies.py sr.json sru.json stride2.json --plot
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='JSON result files to compare (from run_longbench_example.py)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots (requires matplotlib)'
    )
    
    args = parser.parse_args()
    
    if len(args.files) < 2:
        print("Error: At least 2 result files required for comparison.", file=sys.stderr)
        sys.exit(1)
    
    # Load all results
    results = [load_results(f) for f in args.files]
    
    # Print comparison table
    print_comparison_table(results)
    
    # Generate plots if requested
    if args.plot:
        plot_comparison(results)


if __name__ == '__main__':
    main()

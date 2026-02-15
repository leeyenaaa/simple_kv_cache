# 🚀 Simple KV Cache Eviction Strategies

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A demonstration of efficient KV cache eviction strategies from [LaCache](https://github.com/LaCache/LaCache), designed for long-context language model inference without importance scoring overhead.

## 📋 Overview

Large Language Models (LLMs) using KV caching can run out of memory when processing long contexts. This project implements **baseline KV cache eviction strategies** that reduce memory consumption during long-context inference by selectively keeping only the most important tokens:

- **Strategy 1 (Sink + Recent)**: Keep the first N tokens (sink) and last M tokens (recent), dropping everything in between
- **Strategy 2 (Sink + Recent + Uniform Middle)**: Keep sink tokens, recent tokens, plus uniformly-spaced tokens from the middle region

### ✨ Key Features

- ✅ **Zero overhead**: No computational cost for importance scoring (unlike attention-based methods)
- ✅ **Memory efficient**: Save 50-80% memory on long sequences
- ✅ **Flash Attention 2 compatible**: Works seamlessly with optimized attention kernels
- ✅ **Benchmarked**: Tested on LongBench dataset with multiple QA tasks
- ✅ **Visualizations**: Includes tools to visualize eviction patterns and compare strategies

## Architecture

### Strategy 1: Sink + Recent (`middle_strategy="none"`)

```
Original Cache:
┌─────────────────────────────────────────────────────────┐
│ Sink (first N) │ Middle (dropped) │ Recent (last M)     │
└─────────────────────────────────────────────────────────┘

After Eviction:
┌──────────────┬──────────────┐
│ Sink (N)     │ Recent (M)   │
└──────────────┴──────────────┘
```

**Rationale**: The first tokens (sink) provide context for the entire sequence, while recent tokens are needed for next-token prediction. Middle tokens are less critical.

### Strategy 2: Sink + Recent + Uniform Middle (`middle_strategy="uniform"`)

```
Original Cache:
┌─────────────────────────────────────────────────────────┐
│ Sink │ Middle (uniformly sampled) │ Recent              │
└─────────────────────────────────────────────────────────┘

After Eviction:
┌──────────┬──────┬──────┬──────┬──────────┐
│ Sink (N) │ Mid1 │ Mid2 │ Mid3 │ Recent(M)│
└──────────┴──────┴──────┴──────┴──────────┘
```

**Rationale**: Adds uniformly-spaced middle tokens to capture important information throughout the sequence while maintaining a fixed memory budget.

## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sink_tokens` | int | 256 | Number of initial tokens to always keep |
| `recent_tokens` | int | 512 | Number of recent tokens to always keep |
| `middle_strategy` | str | "none" | Eviction strategy: `"none"` or `"uniform"` |
| `middle_budget_tokens` | int | 0 | Token budget for uniform middle selection (used if `uniform_stride=0`) |
| `uniform_stride` | int | 0 | Keep every N-th block in middle region (overrides budget if > 0) |

### Block Size

- **block_n**: 128 tokens per block (fixed in implementation)
- Blocks are the unit of uniform sampling in Strategy 2

## Installation

### Using LaCache Virtual Environment

```bash
# Activate the LaCache virtual environment
source ../LaCache/.venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Dependencies

- **PyTorch**: 2.4.1
- **Transformers**: 4.43.4
- **Flash Attention 2**: 2.6.3 (for optimized attention computation)
- **Block Sparse Flash Attention**: 2.8.3 (optional, for sparse attention patterns)

## Flash Attention Integration

This project leverages **Flash Attention 2** for efficient attention computation:

```python
# Enable Flash Attention in model initialization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

Flash Attention 2 provides:
- **Memory efficiency**: Reduced intermediate tensor allocations
- **Speed**: Faster attention computation via kernel fusion
- **Compatibility**: Works seamlessly with KV cache eviction

## Usage Examples

### Example 1: Sink + Recent Only

```python
from evict.simple_evict import SimpleEvictConfig, build_simple_keep_token_idx

# Configuration: keep first 256 + last 512 tokens
config = SimpleEvictConfig(
    sink_tokens=256,
    recent_tokens=512,
    middle_strategy="none"
)

# Build indices of tokens to keep
keep_idx = build_simple_keep_token_idx(
    total_len=4096,      # Total tokens in cache
    block_n=128,         # Block size
    cfg=config
)
# Output: [0, 1, ..., 255, 3584, 3585, ..., 4095]
# Keeps 768 tokens total (256 + 512)
```

### Example 2: Sink + Recent + Uniform Middle

```python
# Configuration: keep sink + recent + uniformly-spaced middle tokens
config = SimpleEvictConfig(
    sink_tokens=256,
    recent_tokens=512,
    middle_strategy="uniform",
    middle_budget_tokens=512,  # Keep ~512 tokens from middle region
    uniform_stride=0            # Use budget (not stride)
)

keep_idx = build_simple_keep_token_idx(
    total_len=4096,
    block_n=128,
    cfg=config
)
# Output: [0, ..., 255, 512, 1024, 1536, ..., 3584, ..., 4095]
# Keeps ~1280 tokens total (256 + 512 + 512)
```

### Example 3: Using Stride Instead of Budget

```python
# Configuration: keep every 2nd block in middle region
config = SimpleEvictConfig(
    sink_tokens=256,
    recent_tokens=512,
    middle_strategy="uniform",
    uniform_stride=2  # Keep every 2nd block (overrides budget)
)

keep_idx = build_simple_keep_token_idx(
    total_len=4096,
    block_n=128,
    cfg=config
)
# Keeps blocks 2, 4, 6, ... from middle region
```

### Example 4: In-Place Cache Eviction

```python
from evict.simple_evict import evict_dynamic_cache_inplace

# After generating keep_idx, apply eviction to HuggingFace cache
evict_dynamic_cache_inplace(past_key_values, keep_idx)

# Cache is now compacted:
# - Key/value tensors sliced along sequence dimension
# - _seen_tokens metadata updated
# - RoPE position tracking (_ub_abs_pos) maintained
```

## Key Implementation Details

### Token Index Construction

The `build_simple_keep_token_idx()` function:
1. Identifies sink region: `[0, sink_tokens)`
2. Identifies recent region: `[total_len - recent_tokens, total_len)`
3. For uniform strategy, samples middle region uniformly by block
4. Returns sorted, deduplicated indices

### In-Place Eviction

The `evict_dynamic_cache_inplace()` function:
1. Slices key/value cache tensors using `index_select()` on sequence dimension
2. Updates `_seen_tokens` to prevent causal mask corruption
3. Maintains RoPE position tracking via `_ub_abs_pos` attribute

### RoPE Position Tracking

When using position-shift tricks in attention (e.g., storing unrotated K and rotating on-the-fly):
- Original absolute positions are stored in `_ub_abs_pos`
- Eviction slices this tensor consistently with cache
- Prevents RoPE corruption after cache packing

## 📁 Project Structure

```
simple_kv_eviction/
├── compare_strategies.py      # Compare results between different strategies
├── run_longbench_example.py   # Run LongBench evaluation with eviction
├── test_simple_evict.py       # Unit tests for eviction logic
├── visualize_eviction.py      # Visualize eviction patterns as heatmaps
├── run_examples.sh            # Batch script for running experiments
├── requirements.txt           # Python dependencies
├── results/                   # Experiment results (JSON)
└── README.md                  # This file
```

## 🧪 Experimental Results

This project includes benchmark results from the **LongBench** dataset across multiple question-answering tasks:

| Dataset | Task Type | Avg Context Length |
|---------|-----------|-------------------|
| NarrativeQA | Reading Comprehension | ~18K tokens |
| Qasper | Scientific QA | ~3.5K tokens |
| MultiFieldQA-en | Multi-domain QA (EN) | ~4.5K tokens |
| MultiFieldQA-zh | Multi-domain QA (ZH) | ~6K tokens |
| HotpotQA | Multi-hop Reasoning | ~9K tokens |
| 2WikiMQA | Multi-hop Reasoning | ~4.5K tokens |
| MuSiQue | Multi-hop Reasoning | ~11K tokens |

### Visualization Examples

The project includes visualization tools that generate heatmaps showing which tokens are kept/evicted:

- **Blue regions**: Kept tokens
- **Gray regions**: Evicted tokens
- **Red dashed line**: Sink boundary
- **Green dashed line**: Recent boundary

Example visualizations are saved as PNG files showing eviction patterns across all transformer layers.

## 🔬 Running Experiments

### Quick Test (No GPU Required)

```bash
# Test eviction logic with dummy data
python test_simple_evict.py
```

### Visualize Eviction Patterns

```bash
# Visualize default configuration
python visualize_eviction.py

# Visualize sink + recent only (no middle)
python visualize_eviction.py --strategy sink_recent --save eviction_sink_recent.png

# Visualize with custom parameters
python visualize_eviction.py \
    --total_len 8192 \
    --strategy sink_recent_uniform \
    --sink_tokens 256 \
    --recent_tokens 1024 \
    --middle_budget 2816 \
    --save eviction_custom.png
```

### Run LongBench Evaluation (Requires GPU)

```bash
# Demo mode (small model, 2 examples)
python run_longbench_example.py --demo_mode

# Full evaluation with specific strategy
python run_longbench_example.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --strategy sink_recent \
    --sink_tokens 256 \
    --recent_tokens 1024 \
    --dataset narrativeqa \
    --num_examples 10 \
    --output results/sr_narrativeqa.json

# Evaluate with uniform middle strategy
python run_longbench_example.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --strategy sink_recent_uniform \
    --sink_tokens 256 \
    --recent_tokens 1024 \
    --middle_budget 512 \
    --dataset hotpotqa \
    --num_examples 10 \
    --output results/sru_hotpotqa.json
```

### Compare Strategies

```bash
# Compare two strategies
python compare_strategies.py results/sr_narrativeqa.json results/sru_narrativeqa.json

# Compare with visualizations
python compare_strategies.py results/sr_narrativeqa.json results/sru_narrativeqa.json --plot
```

## 📊 Performance Characteristics

| Strategy | Memory Saved | Computation Overhead | Best For |
|----------|--------------|---------------------|----------|
| Sink + Recent | High (50-80%) | **Zero** | Long sequences, strict memory budget |
| Sink + Recent + Uniform | Medium (30-60%) | **Zero** | Long sequences, better quality |

**Comparison with other methods:**
- ❌ **Attention-based eviction** (e.g., H2O): Requires computing attention scores → overhead
- ❌ **EMA-based importance** (e.g., StreamingLLM): Tracks running statistics → overhead
- ✅ **This approach**: Position-based eviction → **zero overhead**

## 🎯 Use Cases

1. **Long-document question answering**: Process documents beyond model's context window
2. **Multi-turn conversations**: Keep conversation history within memory limits
3. **Streaming inference**: Real-time processing with controlled memory growth
4. **Research**: Baseline comparison for advanced eviction strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📚 References

- **LaCache**: [Long-context inference framework](https://github.com/LaCache/LaCache)
- **Flash Attention 2**: [Dao et al., 2023 - Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- **LongBench**: [Bai et al., 2023 - A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508)
- **StreamingLLM**: [Xiao et al., 2023 - Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)

## 📄 License

This project is based on the LaCache framework. See the [LaCache repository](https://github.com/LaCache/LaCache) for license information.

## 🙏 Acknowledgments

- LaCache team for the original eviction implementation
- HuggingFace Transformers for the model interface
- LongBench authors for the evaluation benchmark

---

**⭐ If you find this project helpful, please consider giving it a star!**

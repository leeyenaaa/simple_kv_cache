#!/usr/bin/env python3
"""
LongBench KV Cache Eviction — Self-Contained Example

Chunk-based streaming prefill with KV cache eviction.
No dependency on gated_pred.py or Block Sparse Flash Attention.

Eviction flow:
  1. Feed prompt into model chunk by chunk (streaming prefill)
  2. After each chunk, if cache length > target → evict
  3. Eviction keeps sink (front) + recent (back) [+ uniform mid]
  4. Greedy decode from the compacted cache

RoPE correctness after eviction:
  - Cached keys already have correct rotary embeddings from their original positions
  - New queries/keys use absolute position_ids for rotation
  - cache_position (for causal mask) is auto-derived from _seen_tokens
  - evict_dynamic_cache_inplace() syncs _seen_tokens with actual cache length

Usage:
    # Demo mode (small model, 2 examples)
    python run_longbench_example.py --demo_mode

    # Full evaluation
    python run_longbench_example.py \\
        --model "meta-llama/Meta-Llama-3-8B-Instruct" \\
        --strategy sink_recent \\
        --sink_tokens 256 \\
        --recent_tokens 1024 \\
        --dataset narrativeqa \\
        --num_examples 0 \\
        --output results/full_sr.json

    # num_examples=0 means ALL examples in the dataset
"""

import argparse
import json
import os
import sys
import time
from typing import Optional, Dict, Any, List, Tuple

import torch
from tqdm import tqdm

# Add LaCache path for eviction imports only — no gated_pred dependency
LACACHE_LONGBENCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LaCache', 'LongBench'))
sys.path.insert(0, LACACHE_LONGBENCH_DIR)

try:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from evict.simple_evict import (
        SimpleEvictConfig,
        build_simple_keep_token_idx,
        evict_dynamic_cache_inplace,
    )
    from gated_eval import scorer, dataset2metric, LONG_BENCH_DATASETS
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("\nPlease ensure you have activated the LaCache virtual environment:")
    print("  source ../LaCache/.venv/bin/activate")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Block size for uniform middle sampling.  Only matters for strategy="uniform";
# strategy="none" ignores this entirely.  128 matches Flash Attention's typical
# block size and the LaCache convention.
# ---------------------------------------------------------------------------
BLOCK_N = 128

# LongBench official max generation lengths per dataset.
# Source: LaCache/LongBench/config/dataset2maxlen.json
DATASET2MAXLEN: Dict[str, int] = {}
_d2m_path = os.path.join(LACACHE_LONGBENCH_DIR, "config", "dataset2maxlen.json")
if os.path.isfile(_d2m_path):
    with open(_d2m_path) as _f:
        DATASET2MAXLEN = json.load(_f)
else:
    # Fallback hardcoded from LongBench defaults
    DATASET2MAXLEN = {
        "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
        "multifieldqa_zh": 64, "hotpotqa": 32, "2wikimqa": 32,
        "musique": 32, "dureader": 128, "gov_report": 512,
        "qmsum": 512, "multi_news": 512, "vcsum": 512,
        "trec": 64, "triviaqa": 32, "samsum": 128, "lsht": 64,
        "passage_count": 32, "passage_retrieval_en": 32,
        "passage_retrieval_zh": 32, "lcc": 64, "repobench-p": 64,
    }

# LongBench official prompt templates per dataset.
# Source: LaCache/LongBench/config/dataset2prompt.json
DATASET2PROMPT: Dict[str, str] = {}
_d2p_path = os.path.join(LACACHE_LONGBENCH_DIR, "config", "dataset2prompt.json")
if os.path.isfile(_d2p_path):
    with open(_d2p_path) as _f:
        DATASET2PROMPT = json.load(_f)

# Datasets that should NOT use chat template (few-shot / code completion).
NO_CHAT_TEMPLATE_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

# Chinese datasets — use Chinese system prompt.
ZH_DATASETS = {"multifieldqa_zh", "dureader", "vcsum", "passage_retrieval_zh"}


# ===================== Model Patch: Raw KV + On-the-fly RoPE ===============


def _apply_rotary_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a single tensor (Q or K). x: [..., seq_len, head_dim]."""
    # cos, sin: [1, seq_len, head_dim] from rotary_emb()
    # x: [bsz, n_heads, seq_len, head_dim]
    cos = cos.unsqueeze(1)  # [1, 1, seq_len, hd]
    sin = sin.unsqueeze(1)  # [1, 1, seq_len, hd]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def patch_model_raw_kv(model) -> None:
    """
    Monkey-patch LlamaFlashAttention2 (or LlamaSdpaAttention) to store RAW
    (unrotated) keys in the DynamicCache and apply RoPE on-the-fly at
    attention time using cache-slot positions [0..kv_len-1].

    This makes eviction safe: after eviction the surviving raw keys simply
    get new cache-slot positions, keeping everything within the model's
    trained RoPE range.
    """
    from transformers.models.llama.modeling_llama import (
        LlamaFlashAttention2,
        apply_rotary_pos_emb,
        repeat_kv,
    )
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except ImportError:
        _flash_attention_forward = None

    def _patched_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # --- RoPE for Q only (using position_ids from the caller) ---
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states = _apply_rotary_single(query_states, cos, sin)

        # --- Store RAW (unrotated) K in cache ---
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, kwargs,
            )

        # --- On-the-fly RoPE for ALL cached K using cache-slot positions ---
        kv_len = key_states.shape[2]
        slot_pos = torch.arange(kv_len, device=key_states.device, dtype=torch.long).unsqueeze(0)
        cos_k, sin_k = self.rotary_emb(value_states, slot_pos)
        key_states_rot = _apply_rotary_single(key_states, cos_k, sin_k)

        # --- GQA: repeat KV heads to match Q heads ---
        key_states_rot = repeat_kv(key_states_rot, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # --- Attention (Flash Attention 2 preferred, SDPA fallback) ---
        if _flash_attention_forward is not None:
            query_states = query_states.transpose(1, 2)   # [bsz, q_len, n_heads, hd]
            key_states_rot = key_states_rot.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            attn_output = _flash_attention_forward(
                query_states, key_states_rot, value_states,
                attention_mask, q_len,
                dropout=0.0,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
                is_causal=self.is_causal,
            )
        else:
            # SDPA fallback — query/key/value are [bsz, n_heads, seq_len, hd]
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states_rot, value_states,
                attn_mask=attention_mask,
                is_causal=(attention_mask is None and q_len > 1),
            )
            attn_output = attn_output.transpose(1, 2)  # [bsz, q_len, n_heads, hd]

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    # Patch all LlamaFlashAttention2 modules
    patched = 0
    for name, module in model.named_modules():
        if isinstance(module, LlamaFlashAttention2):
            module.forward = _patched_forward.__get__(module, type(module))
            patched += 1
    if patched > 0:
        print(f"✓ Patched {patched} attention layers for raw KV + on-the-fly RoPE")
    else:
        print("WARNING: No LlamaFlashAttention2 layers found to patch. "
              "raw_rel mode requires Flash Attention 2.")


# ======================== Eviction Heatmap =================================


def save_eviction_heatmap(
    evict_cfg: SimpleEvictConfig,
    total_len: int,
    num_layers: int,
    output_path: str,
    block_n: int = BLOCK_N,
) -> None:
    """
    Simulate eviction and save a heatmap PNG.

    Y-axis: layer (0 = bottom), X-axis: position index.
    Blue = kept, gray = evicted.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        import numpy as np
    except ImportError:
        print("  ⚠ matplotlib/numpy not available, skipping heatmap")
        return

    # Simulate
    keep_idx = build_simple_keep_token_idx(
        total_len=total_len,
        block_n=block_n,
        cfg=evict_cfg,
        device=torch.device("cpu"),
    )
    keep_set = set(keep_idx.tolist())
    mask = np.zeros((num_layers, total_len), dtype=np.uint8)
    for pos in keep_set:
        mask[:, pos] = 1

    # Plot
    cmap = ListedColormap(["#D9D9D9", "#3274A1"])
    fig, ax = plt.subplots(figsize=(16, max(4, num_layers * 0.22)))
    ax.imshow(mask, aspect="auto", cmap=cmap, interpolation="none", origin="lower")

    sink_end = min(total_len, evict_cfg.sink_tokens)
    recent_start = max(0, total_len - evict_cfg.recent_tokens)
    ax.axvline(x=sink_end - 0.5, color="#E74C3C", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.axvline(x=recent_start - 0.5, color="#2ECC71", linewidth=1.5, linestyle="--", alpha=0.8)

    ax.set_xlabel("Position Index", fontsize=12)
    ax.set_ylabel("Layer Index", fontsize=12)

    mid_info = "no mid"
    if evict_cfg.middle_strategy == "uniform":
        mid_info = (f"stride={evict_cfg.uniform_stride}" if evict_cfg.uniform_stride > 0
                    else f"mid={evict_cfg.middle_budget_tokens}")

    kept_count = int(mask[0].sum())
    ax.set_title(
        f"KV Cache Eviction Pattern\n"
        f"total={total_len}  sink={evict_cfg.sink_tokens}  "
        f"recent={evict_cfg.recent_tokens}  {mid_info}  |  "
        f"kept={kept_count}  evicted={total_len - kept_count}",
        fontsize=13, fontweight="bold",
    )

    if num_layers <= 16:
        ax.set_yticks(range(num_layers))
    else:
        ax.set_yticks(range(0, num_layers, max(1, num_layers // 8)))

    xtick_step = 256 if total_len <= 2048 else (512 if total_len <= 8192 else 1024)
    ax.set_xticks(range(0, total_len, xtick_step))

    legend_elements = [
        Patch(facecolor="#3274A1", label="Kept"),
        Patch(facecolor="#D9D9D9", label="Evicted"),
        Patch(facecolor="none", edgecolor="#E74C3C", linestyle="--",
              label=f"Sink boundary ({sink_end})"),
        Patch(facecolor="none", edgecolor="#2ECC71", linestyle="--",
              label=f"Recent boundary ({recent_start})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved to: {output_path}")


# ========================= Prompt Formatting ===============================


def format_longbench_prompt(example: Dict[str, Any], dataset: str) -> str:
    """
    Apply the official LongBench prompt template for the given dataset.

    Each template uses {context} and/or {input} placeholders.
    Falls back to raw example["input"] if no template is found.
    """
    fmt = DATASET2PROMPT.get(dataset)
    if not fmt:
        return example.get("input", "")

    # Build substitution dict — LongBench examples have context, input, etc.
    obj = {
        "context": example.get("context", ""),
        "input": example.get("input", ""),
    }
    try:
        return fmt.format(**obj)
    except (KeyError, IndexError):
        return example.get("input", "")


def build_input_ids(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    dataset: str,
) -> torch.Tensor:
    """
    Tokenize prompt_text with chat template + system instruction.

    No truncation — the full prompt is fed through streaming_prefill(),
    and KV cache eviction manages the memory budget (sink + recent [+ mid]).

    For instruction-tuned models: wraps in chat template with a system prompt
    that says "answer concisely / output only the answer."
    For few-shot datasets (trec, triviaqa, etc.): uses raw prompt (no chat template).
    """
    is_zh = dataset in ZH_DATASETS
    use_chat = dataset not in NO_CHAT_TEMPLATE_DATASETS

    if not use_chat:
        # Few-shot / code tasks — raw prompt, no chat wrapping.
        return tokenizer.encode(
            prompt_text, return_tensors="pt",
            truncation=False,
        )

    # Build system + user messages for chat template.
    if is_zh:
        system = "你是一个有帮助的助手。请直接回答用户问题。不要复述问题。不要添加额外说明。只输出答案。"
        user_content = f"{prompt_text}\n\n指令：只输出答案，不要复述问题。"
    else:
        system = (
            "You are a helpful assistant. "
            "Answer the user's question directly. "
            "Do not repeat the question. "
            "Do not add extra commentary. "
            "Output only the answer."
        )
        user_content = f"{prompt_text}\n\nInstruction: Output only the answer. Do not repeat the question."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    try:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        return ids
    except Exception:
        # Fallback: plain instruction prefix.
        text = f"{system}\n\nUser: {prompt_text}\nAssistant:"
        return tokenizer.encode(
            text, return_tensors="pt",
            truncation=False,
        )


# ============================= Core Engine =================================


@torch.no_grad()
def streaming_prefill(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    evict_cfg: SimpleEvictConfig,
    chunk_size: int = 512,
    block_n: int = BLOCK_N,
    rope_mode: str = "abs",
    verbose: bool = False,
) -> Tuple[DynamicCache, torch.Tensor, int, dict]:
    """
    Feed *input_ids* into *model* chunk by chunk, evicting whenever the
    KV cache exceeds the target length derived from *evict_cfg*.

    Args:
        model:      Causal LM (already on device, eval mode)
        input_ids:  [1, seq_len] prompt token ids
        evict_cfg:  Eviction configuration
        chunk_size: Tokens per chunk (≥1)
        block_n:    Block size for uniform mid sampling
        rope_mode:  "abs" = absolute document positions (correct RoPE distances)
                    "rel" = cache-relative positions (stays in trained range)
                    "raw_rel" = cache-relative positions; model is patched to
                                store raw K and apply RoPE on-the-fly
        verbose:    Print per-chunk stats

    Returns:
        cache:       DynamicCache with (possibly evicted) KV states
        last_logits: [1, chunk_len, vocab] logits from the last chunk
        prompt_len:  Original prompt length (before any eviction)
        stats:       Dict with eviction statistics
    """
    device = input_ids.device
    seq_len = int(input_ids.shape[1])
    chunk_size = max(1, int(chunk_size))

    # Target cache length — eviction fires when cache exceeds this.
    target = (
        evict_cfg.sink_tokens
        + evict_cfg.recent_tokens
        + (evict_cfg.middle_budget_tokens if evict_cfg.middle_strategy == "uniform" else 0)
    )

    cache = DynamicCache()
    last_logits = None
    evict_count = 0
    total_evicted_tokens = 0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = input_ids[:, start:end]
        chunk_len = end - start

        # ---- position_ids: depends on rope_mode ----
        if rope_mode == "abs":
            # Absolute document positions.  Cached keys already carry their
            # original rotary embeddings, so new keys must use their true
            # absolute position for correct RoPE relative distances.
            position_ids = torch.arange(start, end, device=device).unsqueeze(0)
        else:
            # rel / raw_rel: Cache-relative positions = [cur_cache_len, ...).
            # For "rel": keeps values within trained range; after eviction,
            #   RoPE relative distances become approximate (K already rotated).
            # For "raw_rel": positions are used by the patched forward to
            #   rotate Q only.  Cached K is stored raw (unrotated), and RoPE
            #   is applied on-the-fly with slot positions [0..kv_len-1],
            #   so eviction is fully safe.
            cur_cache_len = cache.get_seq_length()
            position_ids = torch.arange(
                cur_cache_len, cur_cache_len + chunk_len, device=device,
            ).unsqueeze(0)

        out = model(
            input_ids=chunk,
            past_key_values=cache,
            use_cache=True,
            position_ids=position_ids,
        )
        cache = out.past_key_values
        last_logits = out.logits

        # ---- Evict if needed ----
        cur_len = cache.get_seq_length()
        if cur_len > target:
            before = cur_len
            keep_idx = build_simple_keep_token_idx(
                total_len=cur_len,
                block_n=block_n,
                cfg=evict_cfg,
                device=device,
            )
            evict_dynamic_cache_inplace(cache, keep_idx)
            after = cache.get_seq_length()
            evict_count += 1
            total_evicted_tokens += before - after
            if verbose:
                print(
                    f"  [EVICT] chunk [{start}:{end}]  "
                    f"cache {before} → {after}  "
                    f"dropped {before - after} tokens"
                )
        elif verbose:
            print(f"  [CHUNK] [{start}:{end}]  cache {cur_len} (no eviction)")

    stats = {
        "eviction_count": evict_count,
        "total_evicted_tokens": total_evicted_tokens,
        "final_cache_len": cache.get_seq_length(),
        "target_cache_len": target,
    }

    return cache, last_logits, seq_len, stats


@torch.no_grad()
def greedy_decode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cache: DynamicCache,
    last_logits: torch.Tensor,
    prompt_len: int,
    max_new_tokens: int = 128,
    rope_mode: str = "abs",
) -> Tuple[str, List[int], int]:
    """
    Greedy decode from *cache* produced by streaming_prefill().

    Args:
        model:          Causal LM
        tokenizer:      Tokenizer (for EOS detection & decoding)
        cache:          DynamicCache (post-eviction prefill state)
        last_logits:    [1, chunk_len, vocab] from last prefill chunk
        prompt_len:     Original (pre-eviction) prompt length
        max_new_tokens: Maximum tokens to generate
        rope_mode:      "abs" = absolute positions, "rel"/"raw_rel" = cache-relative

    Returns:
        text:           Decoded string
        token_ids:      List of generated token ids
        n_generated:    Number of tokens actually generated
    """
    device = last_logits.device
    eos_id = tokenizer.eos_token_id

    # First generated token comes from last prefill logits.
    next_token = last_logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
    generated = [int(next_token.item())]

    if next_token.item() == eos_id:
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text, generated, len(generated)

    for step in range(max_new_tokens - 1):
        if rope_mode == "abs":
            # Absolute position: prompt_len + step
            # (step 0 = processing g0 at document position prompt_len)
            position_ids = torch.tensor([[prompt_len + step]], device=device)
        else:
            # Cache-relative: current cache length (before this forward call)
            position_ids = torch.tensor(
                [[cache.get_seq_length()]], device=device,
            )

        out = model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
            position_ids=position_ids,
        )
        cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))

        if next_token.item() == eos_id:
            break

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, generated, len(generated)


# ============================= CLI / Eval ==================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LongBench evaluation with self-contained KV cache eviction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct",
                        help="Model name or path (default: SmolLM2-135M for demo)")

    # Eviction strategy
    parser.add_argument("--strategy", type=str, choices=["sink_recent", "sink_recent_uniform"],
                        default="sink_recent_uniform",
                        help="Eviction strategy (default: sink_recent_uniform)")

    # Eviction parameters
    parser.add_argument("--sink_tokens", type=int, default=256,
                        help="Number of initial tokens to keep (default: 256)")
    parser.add_argument("--recent_tokens", type=int, default=512,
                        help="Number of recent tokens to keep (default: 512)")
    parser.add_argument("--middle_budget", type=int, default=256,
                        help="Middle region token budget for uniform strategy (default: 256)")
    parser.add_argument("--uniform_stride", type=int, default=0,
                        help="Keep every N-th block in middle; overrides budget if >0 (default: 0)")

    # Dataset
    parser.add_argument("--dataset", type=str, default="narrativeqa",
                        help="LongBench dataset name, or 'all' to run every dataset "
                             "(default: narrativeqa)")
    parser.add_argument("--num_examples", type=int, default=2,
                        help="Number of examples (0 = ALL, default: 2)")
    parser.add_argument("--max_new_tokens", type=int, default=0,
                        help="Max tokens to generate. 0 = auto from LongBench per-dataset config (recommended)")

    # Streaming prefill
    parser.add_argument("--prefill_chunk_size", type=int, default=512,
                        help="Chunk size for streaming prefill (default: 512)")

    # RoPE position mode
    parser.add_argument("--rope_mode", type=str, choices=["abs", "rel", "raw_rel"], default="abs",
                        help="Position ID strategy for RoPE. "
                             "abs = absolute document positions (correct RoPE distances, "
                             "but positions may exceed model's trained range for very long inputs). "
                             "rel = cache-relative positions (stays in trained range, "
                             "but RoPE relative distances become approximate after eviction). "
                             "raw_rel = store RAW (unrotated) K in cache, apply RoPE on-the-fly "
                             "with cache-slot positions [0..kv_len-1]. Eviction-safe: surviving "
                             "keys always get correct positions within trained range. "
                             "(default: abs)")

    # Output
    parser.add_argument("--output", type=str, default="longbench_results.json",
                        help="Output JSON file path (default: longbench_results.json)")

    # Mode / device
    parser.add_argument("--demo_mode", action="store_true",
                        help="Small model, 2 examples, verbose output")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device (default: auto)")

    args = parser.parse_args()

    if args.demo_mode:
        print("[DEMO MODE] Using small model and limited examples for quick testing")
        args.model = "HuggingFaceTB/SmolLM2-135M-Instruct"
        args.num_examples = 2

    return args


def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model with Flash Attention 2 (fallback to standard)."""
    print(f"\nLoading model: {model_name}")
    print(f"Device: {device}")

    if device == "auto" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
    }
    if device == "auto":
        model_kwargs["device_map"] = "auto"

    # Try Flash Attention 2
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="flash_attention_2", **model_kwargs
        )
        print("✓ Flash Attention 2 enabled")
    except Exception as e:
        print(f"Flash Attention 2 not available ({e}), using standard attention")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device != "auto":
        model = model.to(device)

    model.eval()
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    return model, tokenizer


def load_longbench_dataset(dataset_name: str, num_examples: int) -> List[Dict[str, Any]]:
    """
    Load LongBench dataset.

    num_examples:
        >0 → take exactly that many examples
         0 → take ALL examples in the split
    """
    print(f"\nLoading LongBench dataset: {dataset_name}")

    try:
        dataset = load_dataset("THUDM/LongBench", dataset_name, split="test")
        if num_examples > 0:
            dataset = dataset.select(range(min(num_examples, len(dataset))))
        print(f"✓ Loaded {len(dataset)} examples" + (" (all)" if num_examples == 0 else ""))
        return list(dataset)
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        print("Falling back to dummy data...")
        n = max(num_examples, 1)
        return [
            {
                "input": ("This is a test document. " * 200
                          + "\nQuestion: What is this about?"),
                "answers": ["A test document"],
                "context": "This is a test document. " * 200,
                "all_classes": None,
                "length": 1000,
            }
            for _ in range(n)
        ]


def build_eviction_config(
    strategy: str,
    sink_tokens: int,
    recent_tokens: int,
    middle_budget: int,
    uniform_stride: int,
) -> SimpleEvictConfig:
    """Build SimpleEvictConfig from CLI arguments."""
    if strategy == "sink_recent":
        return SimpleEvictConfig(
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            middle_strategy="none",
        )
    else:  # sink_recent_uniform
        return SimpleEvictConfig(
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            middle_strategy="uniform",
            middle_budget_tokens=middle_budget,
            uniform_stride=uniform_stride,
        )


def evaluate_example(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    example: Dict[str, Any],
    evict_cfg: SimpleEvictConfig,
    dataset: str,
    max_new_tokens: int,
    chunk_size: int,
    rope_mode: str,
    verbose: bool,
) -> Dict[str, Any]:
    """Evaluate a single LongBench example with streaming prefill + eviction."""
    answers = example.get("answers", [])
    all_classes = example.get("all_classes", None)

    # Step 1: Apply LongBench official prompt template.
    prompt_text = format_longbench_prompt(example, dataset)
    # Step 2: Tokenize with chat template + "answer only" system instruction.
    #         No truncation — eviction manages the cache budget.
    input_ids = build_input_ids(tokenizer, prompt_text, dataset)
    input_ids = input_ids.to(model.device)
    prompt_len = int(input_ids.shape[1])

    target = (
        evict_cfg.sink_tokens + evict_cfg.recent_tokens
        + (evict_cfg.middle_budget_tokens if evict_cfg.middle_strategy == "uniform" else 0)
    )

    if verbose:
        print(f"  Prompt: {prompt_len} tokens → target cache: {target}")

    try:
        # ---- Prefill ----
        t0 = time.time()
        cache, last_logits, seq_len, prefill_stats = streaming_prefill(
            model=model,
            input_ids=input_ids,
            evict_cfg=evict_cfg,
            chunk_size=chunk_size,
            rope_mode=rope_mode,
            verbose=verbose,
        )
        t_prefill = time.time() - t0

        if verbose:
            print(f"  Prefill: {t_prefill:.2f}s  "
                  f"cache={prefill_stats['final_cache_len']}  "
                  f"evictions={prefill_stats['eviction_count']}")

        # ---- Decode ----
        t1 = time.time()
        output_text, gen_ids, n_gen = greedy_decode(
            model=model,
            tokenizer=tokenizer,
            cache=cache,
            last_logits=last_logits,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            rope_mode=rope_mode,
        )
        t_decode = time.time() - t1

        if verbose:
            preview = output_text[:120] + ("..." if len(output_text) > 120 else "")
            print(f"  Decode: {t_decode:.2f}s  generated={n_gen} tokens")
            print(f"  Output: {preview}")

        return {
            "prompt_length": prompt_len,
            "output_text": output_text,
            "output_length": n_gen,
            "answers": answers,
            "all_classes": all_classes,
            "timing": {
                "prefill_sec": t_prefill,
                "decode_sec": t_decode,
                "generated_tokens": n_gen,
                "eviction_count": prefill_stats["eviction_count"],
                "total_evicted_tokens": prefill_stats["total_evicted_tokens"],
                "final_cache_len": prefill_stats["final_cache_len"],
            },
            "success": True,
        }

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        if verbose:
            traceback.print_exc()
        return {
            "prompt_length": prompt_len,
            "error": str(e),
            "success": False,
        }


# ================================= Main ====================================


def _make_output_path(base_output: str, dataset: str) -> str:
    """
    Derive a per-dataset output path from the base output path.

    'results/foo.json' + 'narrativeqa' → 'results/foo_narrativeqa.json'
    'results/foo.json' + None          → 'results/foo.json'  (single dataset)
    """
    if dataset is None:
        return base_output
    root, ext = os.path.splitext(base_output)
    return f"{root}_{dataset}{ext}"


def run_single_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    dataset: str,
    output_path: str,
    evict_cfg: SimpleEvictConfig,
) -> Dict[str, Any]:
    """
    Run evaluation on a single LongBench dataset.

    Returns the output_data dict (config + summary + results + evaluation).
    """
    # ---- Load dataset ----
    examples = load_longbench_dataset(dataset, args.num_examples)

    # ---- Resolve max_new_tokens ----
    max_new_tokens = args.max_new_tokens
    if max_new_tokens <= 0:
        max_new_tokens = DATASET2MAXLEN.get(dataset, 128)
        print(f"  max_new_tokens = {max_new_tokens} (auto, from LongBench config for '{dataset}')")
    else:
        print(f"  max_new_tokens = {max_new_tokens} (manual override)")

    # ---- Evaluate ----
    print(f"\n{'=' * 80}")
    print(f"Running Evaluation: {dataset} ({len(examples)} examples)")
    print("=" * 80)

    results: List[Dict[str, Any]] = []
    for idx, example in enumerate(tqdm(examples, desc=f"Eval {dataset}")):
        if args.demo_mode:
            print(f"\n[Example {idx + 1}/{len(examples)}]")

        result = evaluate_example(
            model=model,
            tokenizer=tokenizer,
            example=example,
            evict_cfg=evict_cfg,
            dataset=dataset,
            max_new_tokens=max_new_tokens,
            chunk_size=args.prefill_chunk_size,
            rope_mode=args.rope_mode,
            verbose=args.demo_mode,
        )
        results.append(result)

    # ---- Summary ----
    ok = [r for r in results if r["success"]]
    summary: Dict[str, Any] = {
        "total_examples": len(results),
        "successful": len(ok),
        "failed": len(results) - len(ok),
    }
    if ok:
        summary["avg_prompt_length"] = sum(r["prompt_length"] for r in ok) / len(ok)
        summary["avg_output_length"] = sum(r["output_length"] for r in ok) / len(ok)
        summary["avg_prefill_sec"] = sum(r["timing"]["prefill_sec"] for r in ok) / len(ok)
        summary["avg_decode_sec"] = sum(r["timing"]["decode_sec"] for r in ok) / len(ok)
        summary["avg_total_sec"] = summary["avg_prefill_sec"] + summary["avg_decode_sec"]
        summary["avg_eviction_count"] = sum(r["timing"]["eviction_count"] for r in ok) / len(ok)

    # ---- Evaluation scoring ----
    eval_score: Optional[float] = None
    eval_metric_name: Optional[str] = None
    per_example_scores: List[Optional[float]] = []

    if dataset in dataset2metric and ok:
        metric_fn = dataset2metric[dataset]
        eval_metric_name = metric_fn.__name__

        predictions = [r["output_text"] for r in ok]
        answers_list = [r["answers"] for r in ok]
        all_classes = ok[0].get("all_classes") or []

        has_answers = any(len(a) > 0 for a in answers_list)

        if has_answers:
            eval_score = scorer(dataset, predictions, answers_list, all_classes)
            summary["score"] = eval_score
            summary["metric"] = eval_metric_name

            for r in results:
                if not r["success"] or not r.get("answers"):
                    per_example_scores.append(None)
                    continue
                pred = r["output_text"]
                gts = r["answers"]
                if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
                    pred = pred.lstrip("\n").split("\n")[0]
                s = max(metric_fn(pred, gt, all_classes=all_classes) for gt in gts) if gts else 0.0
                per_example_scores.append(round(s, 4))
                r["score"] = round(s, 4)
        else:
            print("\n  ⚠ No ground truth answers found; skipping scoring.")
    elif dataset not in dataset2metric:
        print(f"\n  ⚠ Dataset '{dataset}' has no registered metric; skipping scoring.")

    # ---- Save JSON ----
    output_data: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "strategy": args.strategy,
            "sink_tokens": args.sink_tokens,
            "recent_tokens": args.recent_tokens,
            "middle_budget_tokens": args.middle_budget if args.strategy == "sink_recent_uniform" else 0,
            "uniform_stride": args.uniform_stride,
            "dataset": dataset,
            "num_examples": args.num_examples,
            "max_new_tokens": max_new_tokens,
            "prefill_chunk_size": args.prefill_chunk_size,
            "rope_mode": args.rope_mode,
        },
        "summary": summary,
        "results": results,
    }
    if eval_score is not None:
        output_data["evaluation"] = {
            "score": eval_score,
            "metric": eval_metric_name,
            "num_scored": sum(1 for s in per_example_scores if s is not None),
        }

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # ---- Print summary ----
    print(f"\n{'─' * 60}")
    print(f"  Dataset:   {dataset}")
    print(f"  Total:     {summary['total_examples']}")
    print(f"  Success:   {summary['successful']}")
    print(f"  Failed:    {summary['failed']}")
    if ok:
        print(f"  Avg prompt:     {summary['avg_prompt_length']:.0f} tokens")
        print(f"  Avg output:     {summary['avg_output_length']:.0f} tokens")
        print(f"  Avg prefill:    {summary['avg_prefill_sec']:.2f}s")
        print(f"  Avg decode:     {summary['avg_decode_sec']:.2f}s")
        print(f"  Avg evictions:  {summary['avg_eviction_count']:.1f}")
    if eval_score is not None:
        print(f"  Score:          {eval_score} ({eval_metric_name})")
    print(f"  Saved to:       {output_path}")
    print(f"{'─' * 60}")

    # ---- Auto-generate eviction heatmap ----
    # Use a representative total_len: average prompt length of successful examples,
    # clamped to at least target+1 so eviction actually fires in simulation.
    if ok:
        avg_prompt = int(summary["avg_prompt_length"])
        target = (
            evict_cfg.sink_tokens + evict_cfg.recent_tokens
            + (evict_cfg.middle_budget_tokens if evict_cfg.middle_strategy == "uniform" else 0)
        )
        sim_total = max(avg_prompt, target + 1)

        # Infer num_layers from model config
        num_layers = getattr(model.config, "num_hidden_layers", 32)

        heatmap_path = os.path.splitext(output_path)[0] + "_eviction.png"
        save_eviction_heatmap(
            evict_cfg=evict_cfg,
            total_len=sim_total,
            num_layers=num_layers,
            output_path=heatmap_path,
        )

    return output_data


def main() -> int:
    args = parse_args()

    # ---- Resolve dataset list ----
    if args.dataset == "all":
        datasets = list(LONG_BENCH_DATASETS)
    else:
        datasets = [args.dataset]

    print("=" * 80)
    print("LongBench KV Cache Eviction (self-contained)")
    print("=" * 80)
    print(f"\n  Model:        {args.model}")
    print(f"  Strategy:     {args.strategy}")
    print(f"  Sink:         {args.sink_tokens}")
    print(f"  Recent:       {args.recent_tokens}")
    if args.strategy == "sink_recent_uniform":
        if args.uniform_stride > 0:
            print(f"  Stride:       {args.uniform_stride}")
        else:
            print(f"  Mid budget:   {args.middle_budget}")
    print(f"  Dataset(s):   {', '.join(datasets)} ({len(datasets)} total)")
    print(f"  Examples:     {args.num_examples} ({'all' if args.num_examples == 0 else 'subset'})")
    print(f"  Chunk size:   {args.prefill_chunk_size}")
    print(f"  RoPE mode:    {args.rope_mode}")

    # ---- Load model (once) ----
    try:
        model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    except Exception:
        print("\nFATAL: Could not load model. Exiting.")
        return 1

    # ---- Apply raw KV patch if requested ----
    if args.rope_mode == "raw_rel":
        patch_model_raw_kv(model)

    # ---- Build eviction config (shared across datasets) ----
    evict_cfg = build_eviction_config(
        args.strategy, args.sink_tokens, args.recent_tokens,
        args.middle_budget, args.uniform_stride,
    )

    # ---- Run each dataset ----
    all_results: Dict[str, Dict[str, Any]] = {}
    for ds_idx, dataset in enumerate(datasets):
        print(f"\n{'=' * 80}")
        print(f"  [{ds_idx + 1}/{len(datasets)}] Dataset: {dataset}")
        print("=" * 80)

        # Per-dataset output path: base_{dataset}.json when multi-dataset
        if len(datasets) > 1:
            output_path = _make_output_path(args.output, dataset)
        else:
            output_path = args.output

        ds_result = run_single_dataset(
            model=model,
            tokenizer=tokenizer,
            args=args,
            dataset=dataset,
            output_path=output_path,
            evict_cfg=evict_cfg,
        )
        all_results[dataset] = ds_result

    # ---- Aggregate summary (multi-dataset) ----
    if len(datasets) > 1:
        print(f"\n{'=' * 80}")
        print("Aggregate Summary (all datasets)")
        print("=" * 80)

        agg_rows: List[str] = []
        scored_datasets: List[Tuple[str, float]] = []
        avg_score = 0.0

        for ds in datasets:
            ds_data = all_results.get(ds, {})
            summary = ds_data.get("summary", {})
            evaluation = ds_data.get("evaluation", {})
            score = evaluation.get("score") or summary.get("score")
            metric = evaluation.get("metric") or summary.get("metric", "")
            total = summary.get("total_examples", 0)
            ok_n = summary.get("successful", 0)

            score_str = f"{score:.2f}" if score is not None else "N/A"
            agg_rows.append(f"  {ds:<25s}  {ok_n:>4d}/{total:<4d}  score={score_str}  ({metric})")

            if score is not None:
                scored_datasets.append((ds, score))

        for row in agg_rows:
            print(row)

        if scored_datasets:
            avg_score = sum(s for _, s in scored_datasets) / len(scored_datasets)
            print(f"\n  Average score across {len(scored_datasets)} scored datasets: {avg_score:.2f}")

        # Save aggregate JSON
        agg_path = _make_output_path(args.output, "aggregate")
        agg_data = {
            "config": {
                "model": args.model,
                "strategy": args.strategy,
                "sink_tokens": args.sink_tokens,
                "recent_tokens": args.recent_tokens,
                "middle_budget_tokens": args.middle_budget if args.strategy == "sink_recent_uniform" else 0,
                "uniform_stride": args.uniform_stride,
                "datasets": datasets,
                "num_examples": args.num_examples,
                "prefill_chunk_size": args.prefill_chunk_size,
                "rope_mode": args.rope_mode,
            },
            "per_dataset": {
                ds: {
                    "score": all_results[ds].get("evaluation", {}).get("score")
                             or all_results[ds].get("summary", {}).get("score"),
                    "metric": all_results[ds].get("evaluation", {}).get("metric")
                              or all_results[ds].get("summary", {}).get("metric"),
                    "total_examples": all_results[ds].get("summary", {}).get("total_examples"),
                    "successful": all_results[ds].get("summary", {}).get("successful"),
                }
                for ds in datasets
            },
        }
        if scored_datasets:
            agg_data["average_score"] = round(avg_score, 2)
            agg_data["num_scored_datasets"] = len(scored_datasets)

        with open(agg_path, "w") as f:
            json.dump(agg_data, f, indent=2)
        print(f"\n✓ Aggregate saved to: {agg_path}")
    else:
        print(f"\n✓ Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

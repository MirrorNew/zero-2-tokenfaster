"""A teaching experiment: recompute a prefix versus reuse its K/V tensors.

Random weights and predetermined input tokens: this is not a useful text model.
Only the final position's logits are projected in both execution paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.head_dim = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x, cache=None, start_pos=0):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [
            t.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        ]
        if cache is not None:
            if start_pos != 0 and length != 1:
                raise ValueError("This lab supports only prefill or one-token decode.")
            end = start_pos + length
            if end > cache[0].shape[2]:
                raise ValueError("KV cache capacity exceeded.")
            cache[0][:, :, start_pos:end, :].copy_(k)
            cache[1][:, :, start_pos:end, :].copy_(v)
            k, v = cache[0][:, :, :end, :], cache[1][:, :, :end, :]

        # In single-token incremental decode every stored position is legal.
        # is_causal=True with unequal Q/K lengths would mask out legal history.
        causal = cache is None or start_pos == 0
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        x = x + self.proj(attn.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class TinyDecoder(nn.Module):
    def __init__(self, vocab=4096, width=512, layers=4, heads=8, max_length=2048):
        super().__init__()
        self.vocab, self.width, self.layers, self.heads = vocab, width, layers, heads
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab, width)
        self.position_embedding = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab, bias=False)

    def make_cache(self, batch_size, capacity):
        if capacity > self.max_length:
            raise ValueError("Cache capacity exceeds position embedding capacity.")
        weight = self.token_embedding.weight
        shape = (batch_size, self.heads, capacity, self.width // self.heads)
        return [
            (
                torch.empty(shape, dtype=weight.dtype, device=weight.device),
                torch.empty(shape, dtype=weight.dtype, device=weight.device),
            )
            for _ in self.blocks
        ]

    def forward(self, tokens, caches=None, start_pos=0):
        if caches is None and start_pos != 0:
            raise ValueError("Uncached execution must contain the full prefix.")
        positions = torch.arange(
            start_pos, start_pos + tokens.shape[1], device=tokens.device
        )
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for index, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[index], start_pos)
        return self.lm_head(self.final_norm(x[:, -1, :]))


@torch.inference_mode()
def check_equivalence(model, tokens, prompt_length, decode_steps):
    """Check prefill AND all decode logits, rather than just final argmax."""
    cache = model.make_cache(tokens.shape[0], prompt_length + decode_steps)
    cached = model(tokens[:, :prompt_length], cache)
    baseline = model(tokens[:, :prompt_length])
    max_abs = (baseline - cached).abs().max().item()
    all_close = torch.allclose(baseline, cached, rtol=1e-4, atol=1e-5)
    for step in range(decode_steps):
        pos = prompt_length + step
        cached = model(tokens[:, pos : pos + 1], cache, start_pos=pos)
        baseline = model(tokens[:, : pos + 1])
        max_abs = max(max_abs, (baseline - cached).abs().max().item())
        all_close = all_close and torch.allclose(
            baseline, cached, rtol=1e-4, atol=1e-5
        )
    return {
        "allclose": bool(all_close),
        "max_abs_logit_error": max_abs,
        "rtol": 1e-4,
        "atol": 1e-5,
        "positions_checked": decode_steps + 1,
    }


@torch.inference_mode()
def run_once(model, tokens, prompt_length, decode_steps, use_cache):
    """CUDA event time; cache allocation excluded, cache writes included."""
    torch.cuda.synchronize()
    base_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    cache = (
        model.make_cache(tokens.shape[0], prompt_length + decode_steps)
        if use_cache
        else None
    )
    start, after_prefill, finish = [
        torch.cuda.Event(enable_timing=True) for _ in range(3)
    ]
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    output = model(tokens[:, :prompt_length], cache)
    after_prefill.record()
    for step in range(decode_steps):
        pos = prompt_length + step
        if use_cache:
            output = model(tokens[:, pos : pos + 1], cache, start_pos=pos)
        else:
            output = model(tokens[:, : pos + 1])
    finish.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    # Keep output alive until synchronization. Its values are checked separately.
    assert output.shape == (tokens.shape[0], model.vocab)
    return {
        "prefill_ms": start.elapsed_time(after_prefill),
        "decode_total_ms": after_prefill.elapsed_time(finish),
        "decode_ms_per_step": after_prefill.elapsed_time(finish) / decode_steps,
        "prefill_plus_decode_wall_ms": wall_ms,
        "base_allocated_mib": base_allocated / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_increment_mib": (
            torch.cuda.max_memory_allocated() - base_allocated
        ) / 2**20,
        "reserved_after_run_mib": torch.cuda.memory_reserved() / 2**20,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("Use at least 3 repeats for this small benchmark.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this GPU benchmark.")

    torch.manual_seed(20261020)
    # FP32 reduces numerical ambiguity; this is not peak mixed-precision speed.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = TinyDecoder().to(device="cuda", dtype=torch.float32).eval()
    config = dict(
        vocab=4096, width=512, layers=4, heads=8, max_length=2048,
        dtype="float32", decode_steps=32, repeats=args.repeats, warmups_per_path=2,
        seed=20261020, tf32=False, sdpa_backend="PyTorch automatic dispatch",
    )
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Random-weight teaching microbenchmark; no text quality claim.",
        "system": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "gpu_total_memory_mib": (
                torch.cuda.get_device_properties(0).total_memory / 2**20
            ),
        },
        "config": config,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "cases": [],
    }
    print(json.dumps(result["system"], ensure_ascii=False), flush=True)
    for case_index, (batch, prompt_length) in enumerate(
        (b, p) for b in (1, 4) for p in (128, 512, 1024)
    ):
        tokens = torch.randint(
            model.vocab, (batch, prompt_length + config["decode_steps"]),
            device="cuda",
        )
        correctness = check_equivalence(
            model, tokens, prompt_length, config["decode_steps"]
        )
        if not correctness["allclose"]:
            raise RuntimeError(f"Logit mismatch: {correctness}")
        for _ in range(config["warmups_per_path"]):
            for use_cache in (False, True):
                run_once(
                    model, tokens, prompt_length, config["decode_steps"], use_cache
                )
        measurements = {"no_kv": [], "kv": []}
        for repeat in range(args.repeats):
            # Counterbalance drift: reverse first path across repeats and cases.
            order = (False, True) if (repeat + case_index) % 2 == 0 else (True, False)
            for use_cache in order:
                measurements["kv" if use_cache else "no_kv"].append(
                    run_once(
                        model, tokens, prompt_length, config["decode_steps"], use_cache
                    )
                )
        summary = {
            name: {
                key: statistics.median(run[key] for run in runs)
                for key in runs[0]
            }
            for name, runs in measurements.items()
        }
        speedup = (
            summary["no_kv"]["decode_ms_per_step"] /
            summary["kv"]["decode_ms_per_step"]
        )
        kv_bytes = (
            2 * model.layers * batch * (prompt_length + config["decode_steps"])
            * model.width * model.token_embedding.weight.element_size()
        )
        case = {
            "batch": batch,
            "prompt_tokens": prompt_length,
            "decode_steps": config["decode_steps"],
            "correctness": correctness,
            "kv_cache_capacity_mib": kv_bytes / 2**20,
            "median": summary,
            "decode_speedup_vs_recompute": speedup,
            "raw_runs": measurements,
        }
        result["cases"].append(case)
        print(
            f"B={batch} prompt={prompt_length}: "
            f"noKV={summary['no_kv']['decode_ms_per_step']:.3f} ms/step, "
            f"KV={summary['kv']['decode_ms_per_step']:.3f} ms/step, "
            f"ratio={speedup:.2f}x, max_logit_error="
            f"{correctness['max_abs_logit_error']:.3g}", flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results_kv.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "results_kv.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "batch", "prompt_tokens", "decode_steps", "no_kv_ms_per_step",
                "kv_ms_per_step", "speedup_vs_recompute",
                "max_abs_logit_error", "kv_capacity_mib",
                "no_kv_peak_allocated_mib", "kv_peak_allocated_mib",
            ],
        )
        writer.writeheader()
        for case in result["cases"]:
            writer.writerow({
                "batch": case["batch"],
                "prompt_tokens": case["prompt_tokens"],
                "decode_steps": case["decode_steps"],
                "no_kv_ms_per_step": case["median"]["no_kv"]["decode_ms_per_step"],
                "kv_ms_per_step": case["median"]["kv"]["decode_ms_per_step"],
                "speedup_vs_recompute": case["decode_speedup_vs_recompute"],
                "max_abs_logit_error": case["correctness"]["max_abs_logit_error"],
                "kv_capacity_mib": case["kv_cache_capacity_mib"],
                "no_kv_peak_allocated_mib": case["median"]["no_kv"]["peak_allocated_mib"],
                "kv_peak_allocated_mib": case["median"]["kv"]["peak_allocated_mib"],
            })
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

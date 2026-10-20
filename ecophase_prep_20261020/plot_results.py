"""Render teaching figures from saved results; does not rerun experiments."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def plot_kv():
    report = json.loads((ROOT / "results_kv.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for batch, axis in zip((1, 4), axes):
        cases = [case for case in report["cases"] if case["batch"] == batch]
        for path, label, color in (
            ("no_kv", "Recompute full prefix", "#c26b29"),
            ("kv", "Reuse KV cache", "#2166ac"),
        ):
            medians = [case["median"][path]["decode_ms_per_step"] for case in cases]
            low, high = [], []
            for case, median in zip(cases, medians):
                values = [run["decode_ms_per_step"] for run in case["raw_runs"][path]]
                low.append(median - min(values))
                high.append(max(values) - median)
            axis.errorbar(
                [case["prompt_tokens"] for case in cases],
                medians,
                yerr=[low, high],
                marker="o",
                capsize=4,
                color=color,
                label=label,
            )
        axis.set_title(f"Batch = {batch}")
        axis.set_xlabel("Prompt length (tokens)")
        axis.set_xticks([128, 512, 1024])
        axis.set_yscale("log")
        axis.grid(alpha=0.2, which="both")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Decode time (ms / batch step, log scale)")
    axes[1].legend(frameon=False)
    fig.suptitle("KV cache teaching experiment: random 17.85M model, RTX 5080 Laptop")
    fig.text(
        0.5, 0.015,
        "Median of 5 runs; bars show min-max. Baseline is naive prefix recomputation, not vLLM.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(ROOT / "kv_results.png", dpi=180)
    plt.close(fig)


def plot_early_stop():
    report = json.loads((ROOT / "results_early_stop.json").read_text(encoding="utf-8"))
    curves = report["seed_0_example_curves"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    upper_loss = max(max(curve) for curve in curves.values()) * 1.05
    for axis, (curve_type, curve) in zip(axes, curves.items()):
        axis.plot(
            range(1, len(curve) + 1), curve,
            color="#777777", linewidth=1.6, label="Full synthetic trace",
        )
        axis.axhline(
            report["protocol"]["target_loss_at_most"],
            color="#4d9221", linestyle="--", label="Frozen target",
        )
        stop_indices = []
        for policy, label, color in (
            ("raw_patience", "Raw stop", "#c26b29"),
            ("ema_patience", "EMA stop", "#2166ac"),
        ):
            row = next(
                row for row in report["runs"]
                if row["curve_type"] == curve_type
                and row["seed"] == 0 and row["policy"] == policy
            )
            axis.axvline(
                row["stop_evaluation"], color=color, linestyle=":",
                linewidth=2, label=label,
            )
            stop_indices.append(row["stop_evaluation"])
        axis.set_title(
            f'{curve_type.replace("_", " ").title()}\n'
            f'Raw / EMA end: {stop_indices[0]} / {stop_indices[1]}',
            fontsize=11,
        )
        axis.set_xlabel("Validation observation index")
        axis.set_ylim(0, upper_loss)
        axis.grid(alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Synthetic loss (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Early-stop counterexamples: synthetic curves, seed 0; not LLM training")
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(ROOT / "early_stop_results.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_kv()
    plot_early_stop()
    print("Rendered kv_results.png and early_stop_results.png from saved data.")

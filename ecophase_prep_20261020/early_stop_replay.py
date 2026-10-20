"""Synthetic early-stopping replay. No model training or EcoTrain implementation.

All policies observe only the current prefix. Full future curves are used solely
for retrospective false-stop diagnosis. Run with Python 3.10+; standard library.
"""

import csv
import json
import math
import random
from pathlib import Path
from statistics import mean


# Frozen teaching protocol, chosen before the first run.
BUDGET = 120  # validation evaluations, NOT training steps or elapsed seconds
SEEDS = tuple(range(30))
TARGET_LOSS = 0.25
PATIENCE = 10
MIN_DELTA = 0.005
EMA_ALPHA = 0.25
CURVE_TYPES = ("smooth_decay", "noisy_decay", "delayed_improvement")
POLICIES = ("fixed_budget", "raw_patience", "ema_patience")


def make_curve(curve_type, seed):
    """Generate intentionally stylized loss traces; lower is better."""
    type_offset = CURVE_TYPES.index(curve_type) * 10000
    rng = random.Random(type_offset + seed)
    amplitude = rng.uniform(0.85, 0.95)
    decay_scale = rng.uniform(27.0, 33.0)
    curve = []
    for evaluation in range(1, BUDGET + 1):
        if curve_type == "smooth_decay":
            loss = 0.12 + amplitude * math.exp(-evaluation / decay_scale)
        elif curve_type == "noisy_decay":
            loss = 0.12 + amplitude * math.exp(-evaluation / decay_scale)
            loss += rng.gauss(0.0, 0.035)
        elif curve_type == "delayed_improvement":
            # Deliberately constructed counterexample: a long plateau precedes
            # another improvement phase. This is NOT evidence about real LLMs.
            if evaluation <= 25:
                loss = 0.48 + amplitude * math.exp(-evaluation / 5.0)
            elif evaluation <= 75:
                loss = 0.48
            else:
                loss = 0.14 + 0.34 * math.exp(-(evaluation - 75) / 10.0)
            loss += rng.gauss(0.0, 0.003)
        else:
            raise ValueError(f"Unknown curve type: {curve_type}")
        curve.append(max(loss, 0.001))
    return curve


def replay(curve, policy):
    """Replay an online stopping rule and select its lowest-loss checkpoint."""
    best_loss = float("inf")
    best_checkpoint = None
    best_monitor = float("inf")
    stale_evaluations = 0
    ema = None
    first_target_evaluation = None
    stop_evaluation = len(curve)
    for evaluation, loss in enumerate(curve, start=1):
        if loss < best_loss:
            best_loss = loss
            best_checkpoint = evaluation
        if first_target_evaluation is None and loss <= TARGET_LOSS:
            first_target_evaluation = evaluation
        if policy == "fixed_budget":
            continue
        ema = loss if ema is None else EMA_ALPHA * loss + (1 - EMA_ALPHA) * ema
        monitor = loss if policy == "raw_patience" else ema
        if monitor < best_monitor - MIN_DELTA:
            best_monitor = monitor
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        if stale_evaluations >= PATIENCE:
            stop_evaluation = evaluation
            break

    attained = best_loss <= TARGET_LOSS
    # These two quantities are computed AFTER the online decision. No policy
    # sees the future; a real deployment cannot know this diagnosis in advance.
    full_curve_attains_target = min(curve) <= TARGET_LOSS
    false_stop = (
        stop_evaluation < len(curve) and not attained and full_curve_attains_target
    )
    return {
        "policy": policy,
        "stop_evaluation": stop_evaluation,
        "best_checkpoint_evaluation": best_checkpoint,
        "best_observed_loss": best_loss,
        "target_attained": attained,
        "evaluations_to_target": first_target_evaluation,
        "saved_evaluation_fraction": 1 - stop_evaluation / len(curve),
        "false_stop": false_stop,
        "full_curve_attains_target_ex_post": full_curve_attains_target,
    }


def summarize(rows):
    summaries = []
    for curve_type in CURVE_TYPES:
        for policy in POLICIES:
            selected = [
                row for row in rows
                if row["curve_type"] == curve_type and row["policy"] == policy
            ]
            attained_times = [
                row["evaluations_to_target"] for row in selected
                if row["target_attained"]
            ]
            summaries.append({
                "curve_type": curve_type,
                "policy": policy,
                "runs": len(selected),
                "target_attained_count": sum(r["target_attained"] for r in selected),
                "target_attainment_rate": mean(r["target_attained"] for r in selected),
                "mean_stop_evaluation": mean(r["stop_evaluation"] for r in selected),
                "mean_saved_evaluation_fraction": mean(
                    r["saved_evaluation_fraction"] for r in selected
                ),
                "false_stop_count": sum(r["false_stop"] for r in selected),
                "false_stop_rate": mean(r["false_stop"] for r in selected),
                "mean_evaluations_to_target_among_attained": (
                    mean(attained_times) if attained_times else None
                ),
            })
    return summaries


def self_checks(rows):
    """Three behavior checks covering the core measurement claims."""
    fixed = [row for row in rows if row["policy"] == "fixed_budget"]
    assert all(
        row["stop_evaluation"] == BUDGET
        and row["saved_evaluation_fraction"] == 0
        and not row["false_stop"]
        for row in fixed
    ), "Fixed-budget runs must not stop early."

    delayed = [row for row in rows if row["curve_type"] == "delayed_improvement"]
    assert any(
        row["policy"] != "fixed_budget" and row["false_stop"] for row in delayed
    ), "The designed delayed-improvement counterexample must expose a false stop."
    assert all(
        row["target_attained"]
        for row in delayed if row["policy"] == "fixed_budget"
    ), "The complete delayed trajectory must reach the frozen target."

    failures = [row for row in rows if not row["target_attained"]]
    assert failures, "The null check must cover actual failed runs."
    assert all(
        row["evaluations_to_target"] is None for row in failures
    ), "Unattained targets must have null evaluations-to-target, never zero/budget."
    assert all(
        1 <= row["evaluations_to_target"] <= row["stop_evaluation"]
        for row in rows if row["target_attained"]
    ), "Attained target times must be inside the observed prefix."
    return {
        "fixed_budget_never_stops_early": "PASS",
        "constructed_delayed_case_exposes_false_stop": "PASS",
        "unattained_target_time_is_null": "PASS",
    }


def main():
    rows = []
    examples = {}
    for curve_type in CURVE_TYPES:
        for seed in SEEDS:
            curve = make_curve(curve_type, seed)
            if seed == 0:
                examples[curve_type] = curve
            for policy in POLICIES:
                rows.append({"curve_type": curve_type, "seed": seed, **replay(curve, policy)})

    checks = self_checks(rows)
    summaries = summarize(rows)
    report = {
        "scope": (
            "All curves are synthetic. This is a teaching replay, not LLM training, "
            "a real GPU cost benchmark, or a reproduction of EcoTrain."
        ),
        "protocol": {
            "budget_evaluations": BUDGET,
            "seed_values": list(SEEDS),
            "seeds_per_curve_type": len(SEEDS),
            "target_loss_at_most": TARGET_LOSS,
            "patience_evaluations": PATIENCE,
            "minimum_monitor_improvement": MIN_DELTA,
            "ema_alpha": EMA_ALPHA,
            "checkpoint_selection": "lowest observed raw loss before or at stop",
            "false_stop_definition": (
                "Stopped early without target attainment, while the complete "
                "same synthetic curve attains target; retrospective diagnosis only."
            ),
            "unit_note": (
                "Evaluation counts are not training steps, wall-clock time, GPU-hours "
                "or energy. Unattained evaluations_to_target is JSON null / CSV null."
            ),
        },
        "self_checks": checks,
        "summary": summaries,
        "seed_0_example_curves": examples,
        "runs": rows,
    }
    output_dir = Path(__file__).resolve().parent
    json_path = output_dir / "results_early_stop.json"
    csv_path = output_dir / "results_early_stop.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: "null" if value is None else value for key, value in row.items()
            })
    for item in summaries:
        print(
            f'{item["curve_type"]:20} {item["policy"]:14} '
            f'attained={item["target_attained_count"]:2}/{item["runs"]} '
            f'saved_evals={item["mean_saved_evaluation_fraction"]:.1%} '
            f'false_stop={item["false_stop_count"]:2}/{item["runs"]}'
        )
    print("Self-checks:", checks)
    print("Wrote:", json_path)
    print("Wrote:", csv_path)


if __name__ == "__main__":
    main()

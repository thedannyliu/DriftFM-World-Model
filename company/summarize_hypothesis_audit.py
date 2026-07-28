#!/usr/bin/env python3
"""Combine per-checkpoint hypothesis audits into compact diagnostic gates."""

import argparse
import json
from pathlib import Path

import numpy as np


TRANSITIONS = ((1, 2), (2, 4), (4, 8))
EXPECTED_RUNS = 16


def parse_result(value):
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("results must use NAME=PATH")
    return name, Path(path)


def number(value):
    if value is None:
        return "NA"
    return f"{value:.6g}"


def load_result(name, path):
    with path.open() as input_file:
        result = json.load(input_file)
    if result.get("status") != "complete":
        raise ValueError(f"{path} is not a complete audit")
    return name, result


def median(values):
    return float(np.median(values)) if values else None


def positive_fraction(values):
    return float(np.mean(np.asarray(values) > 0.0)) if values else None


def signal(values, threshold=0.30, required_fraction=0.75):
    if not values:
        return "missing"
    if median(values) >= threshold and positive_fraction(values) >= required_fraction:
        return "pass"
    return "fail"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", action="append", type=parse_result, required=True
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = [load_result(name, path) for name, path in args.result]
    composition_values = {transition: [] for transition in TRANSITIONS}
    off_manifold_values = {4: [], 8: []}
    motion_ratios = {
        transition: {"mse": [], "pose_vertex": []}
        for transition in ((2, 4), (4, 8))
    }
    compact_runs = []

    for name, result in runs:
        metrics = result["metrics"]
        correlations = result["correlations"]
        strata = result["motion_strata"]
        for transition in TRANSITIONS:
            shallow, deep = transition
            value = correlations.get(
                f"defect_vs_degradation/{shallow}_to_{deep}/mse"
            )
            if value is not None:
                composition_values[transition].append(value)
        for nfe, transition in ((4, (2, 4)), (8, (4, 8))):
            shallow, deep = transition
            if metrics[f"degradation/{shallow}_to_{deep}/mse"] > 0.0:
                off_manifold_values[nfe].append(
                    metrics[
                        f"off_manifold/nfe{nfe}/later_relative_penalty"
                    ]
                )
        for transition in ((2, 4), (4, 8)):
            shallow, deep = transition
            for risk in ("mse", "pose_vertex"):
                key = f"degradation/{shallow}_to_{deep}/{risk}"
                low = strata["low"][key]
                high = strata["high"][key]
                denominator = max(abs(low), 1e-12)
                motion_ratios[transition][risk].append(
                    (high - low) / denominator
                )

        risk_mse = [
            metrics[f"risk/nfe{nfe}/mse"] for nfe in (1, 2, 4, 8)
        ]
        risk_pose = [
            metrics[f"risk/nfe{nfe}/pose_vertex"] for nfe in (1, 2, 4, 8)
        ]
        corr_24 = correlations.get("defect_vs_degradation/2_to_4/mse")
        corr_48 = correlations.get("defect_vs_degradation/4_to_8/mse")
        print(
            f"AUDIT name={name} step={result['checkpoint_step']} "
            f"mse_1/2/4/8={'/'.join(number(value) for value in risk_mse)} "
            f"pose_1/2/4/8={'/'.join(number(value) for value in risk_pose)} "
            f"corr_2to4={number(corr_24)} corr_4to8={number(corr_48)} "
            f"family={result['family']} "
            f"offmanifold_4/8="
            f"{number(metrics['off_manifold/nfe4/later_relative_penalty'])}/"
            f"{number(metrics['off_manifold/nfe8/later_relative_penalty'])} "
            f"wandb={result.get('wandb_run_id', 'NA')}"
        )
        compact_runs.append({
            "name": name,
            "family": result["family"],
            "checkpoint_step": result["checkpoint_step"],
            "mse": dict(zip(("nfe1", "nfe2", "nfe4", "nfe8"), risk_mse)),
            "pose_vertex": dict(
                zip(("nfe1", "nfe2", "nfe4", "nfe8"), risk_pose)
            ),
            "corr_defect_degradation": {
                "2_to_4": corr_24,
                "4_to_8": corr_48,
            },
            "off_manifold_relative_penalty": {
                "nfe4": metrics[
                    "off_manifold/nfe4/later_relative_penalty"
                ],
                "nfe8": metrics[
                    "off_manifold/nfe8/later_relative_penalty"
                ],
            },
            "wandb_run_id": result.get("wandb_run_id"),
        })

    composition_gates = {}
    for transition in ((2, 4), (4, 8)):
        values = composition_values[transition]
        key = f"{transition[0]}_to_{transition[1]}"
        composition_gates[key] = {
            "median_correlation": median(values),
            "positive_fraction": positive_fraction(values),
            "decision": signal(values),
        }
    composition_decisions = [
        item["decision"] for item in composition_gates.values()
    ]
    if all(decision == "pass" for decision in composition_decisions):
        composition_decision = "pass"
    elif any(decision == "pass" for decision in composition_decisions):
        composition_decision = "partial"
    else:
        composition_decision = "fail"

    off_manifold_gates = {}
    for nfe, values in off_manifold_values.items():
        median_penalty = median(values)
        fraction_above = (
            float(np.mean(np.asarray(values) >= 0.20)) if values else None
        )
        off_manifold_gates[f"nfe{nfe}"] = {
            "median_relative_penalty": median_penalty,
            "fraction_at_least_20pct": fraction_above,
            "decision": (
                "pass"
                if (
                    median_penalty is not None
                    and median_penalty >= 0.20
                    and fraction_above >= 0.75
                )
                else ("missing" if not values else "fail")
            ),
        }
    off_manifold_decision = (
        "pass"
        if all(
            item["decision"] == "pass"
            for item in off_manifold_gates.values()
        )
        else "fail"
    )

    motion_gates = {}
    family_motion = {}
    for _, result in runs:
        family = result["family"]
        family_motion.setdefault(family, {})
        for transition in ((2, 4), (4, 8)):
            key = f"{transition[0]}_to_{transition[1]}"
            family_motion[family].setdefault(
                key, {"mse": [], "pose_vertex": []}
            )
            for risk in ("mse", "pose_vertex"):
                low = result["motion_strata"]["low"][
                    f"degradation/{key}/{risk}"
                ]
                high = result["motion_strata"]["high"][
                    f"degradation/{key}/{risk}"
                ]
                family_motion[family][key][risk].append(
                    (high - low) / max(abs(low), 1e-12)
                )
    for transition, risks in motion_ratios.items():
        key = f"{transition[0]}_to_{transition[1]}"
        motion_gates[key] = {}
        for risk, ratios in risks.items():
            fraction_above = float(np.mean(np.asarray(ratios) >= 0.25))
            motion_gates[key][risk] = {
                "median_high_vs_low_relative_increase": median(ratios),
                "fraction_at_least_25pct": fraction_above,
            }
    passing_families = []
    for family, transitions in family_motion.items():
        if any(
            median(risks["mse"]) >= 0.25
            and median(risks["pose_vertex"]) >= 0.25
            for risks in transitions.values()
        ):
            passing_families.append(family)
    motion_decision = "pass" if len(passing_families) >= 2 else "fail"

    coverage_complete = len(runs) >= EXPECTED_RUNS
    if not coverage_complete:
        next_step = "collect_remaining_audits"
    elif (
        composition_decision == "pass"
        and off_manifold_decision == "pass"
        and motion_decision == "pass"
    ):
        next_step = "implement_minimal_advantage_aligned_training"
    elif composition_decision in {"pass", "partial"} and off_manifold_decision == "pass":
        next_step = "audit_dynamics_relevance_before_training"
    else:
        next_step = "reject_or_refine_composition_amplification_hypothesis"

    summary = {
        "status": "complete",
        "num_runs": len(runs),
        "expected_runs": EXPECTED_RUNS,
        "coverage": "complete" if coverage_complete else "partial",
        "gates": {
            "composition": {
                "decision": composition_decision,
                "transitions": composition_gates,
            },
            "off_manifold": {
                "decision": off_manifold_decision,
                "nfes": off_manifold_gates,
            },
            "motion": {
                "decision": motion_decision,
                "passing_families": sorted(passing_families),
                "transitions": motion_gates,
            },
        },
        "next_step": next_step,
        "runs": compact_runs,
    }
    output = json.dumps(summary, separators=(",", ":"))
    print(output)
    if args.output:
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()

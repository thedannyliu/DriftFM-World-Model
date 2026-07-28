#!/usr/bin/env python3
"""Summarize baseline or pilot rollout JSON files without verbose logs."""

import argparse
import json
from pathlib import Path

import numpy as np


LOWER_IS_BETTER = ("mse", "lpips", "final_block_vertex_error")


def read_metrics(path):
    metrics = json.loads(path.read_text())
    keys = (
        "mse",
        "ssim",
        "psnr",
        "lpips",
        "seconds_per_frame",
        "final_block_vertex_error",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def paired_analysis(paths_by_nfe):
    payloads = {
        nfe: json.loads(path.read_text())
        for nfe, path in paths_by_nfe.items()
    }
    if 1 not in payloads:
        return None
    difficulty_name = "action_step_mean"
    action_path = payloads[1].get("per_video", {}).get(difficulty_name)
    if not action_path:
        difficulty_name = "action_path"
        action_path = payloads[1].get("per_video", {}).get(difficulty_name)
    if not action_path:
        return None

    action_path = np.asarray(action_path, dtype=np.float64)
    count = action_path.size
    for nfe, payload in payloads.items():
        candidate = payload.get("per_video", {}).get(difficulty_name)
        if candidate is None or len(candidate) != count:
            raise ValueError(f"NFE{nfe} action-path records are not aligned")
        if not np.allclose(action_path, candidate, rtol=0.0, atol=1e-8):
            raise ValueError(f"NFE{nfe} action-path records differ from NFE1")

    result = {
        "num_examples": int(count),
        "difficulty_name": difficulty_name,
        "action_path_mean": float(action_path.mean()),
        "transitions": {},
    }
    for shallow, deep in ((1, 2), (2, 4), (4, 8)):
        if shallow not in payloads or deep not in payloads:
            continue
        transition = {}
        for metric in LOWER_IS_BETTER:
            shallow_values = payloads[shallow].get("per_video", {}).get(metric)
            deep_values = payloads[deep].get("per_video", {}).get(metric)
            if shallow_values is None or deep_values is None:
                continue
            shallow_values = np.asarray(shallow_values, dtype=np.float64)
            deep_values = np.asarray(deep_values, dtype=np.float64)
            benefit = shallow_values - deep_values
            correlation = None
            if action_path.std() >= 1e-12 and benefit.std() >= 1e-12:
                correlation = float(np.corrcoef(action_path, benefit)[0, 1])
            transition[metric] = {
                "mean_delta": float((deep_values - shallow_values).mean()),
                "relative_change": float(
                    deep_values.mean() / max(shallow_values.mean(), 1e-12) - 1.0
                ),
                "fraction_improved": float(np.mean(benefit > 0.0)),
                "action_path_vs_benefit_correlation": correlation,
            }
        result["transitions"][f"{shallow}_to_{deep}"] = transition

    if 2 in payloads and count >= 4:
        dev_indices = np.arange(0, count, 2)
        test_indices = np.arange(1, count, 2)
        threshold = float(np.median(action_path[dev_indices]))
        test_action = action_path[test_indices]
        use_nfe2 = test_action >= threshold
        routing = {
            "split": "even_index_dev_odd_index_test",
            "dev_examples": int(dev_indices.size),
            "test_examples": int(test_indices.size),
            "action_path_threshold": threshold,
            "nfe2_fraction": float(use_nfe2.mean()),
            "metrics": {},
        }
        for metric in LOWER_IS_BETTER:
            nfe1 = payloads[1].get("per_video", {}).get(metric)
            nfe2 = payloads[2].get("per_video", {}).get(metric)
            if nfe1 is None or nfe2 is None:
                continue
            nfe1 = np.asarray(nfe1, dtype=np.float64)[test_indices]
            nfe2 = np.asarray(nfe2, dtype=np.float64)[test_indices]
            adaptive = float(np.where(use_nfe2, nfe2, nfe1).mean())
            selected_fraction = float(use_nfe2.mean())
            random_expected = float(
                (1.0 - selected_fraction) * nfe1.mean()
                + selected_fraction * nfe2.mean()
            )
            routing["metrics"][metric] = {
                "adaptive": adaptive,
                "random_expected_same_budget": random_expected,
                "adaptive_vs_random_relative_change": float(
                    adaptive / max(random_expected, 1e-12) - 1.0
                ),
            }
        result["action_routing_nfe1_or_2"] = routing
    return result


def numeric_items(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else key
            yield from numeric_items(child, child_prefix)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--driftflow-dir", type=Path)
    parser.add_argument("--variant-dir", type=Path)
    parser.add_argument("--nfes", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.baseline_dir:
        result = {
            "status": "complete",
            "baseline_64": read_metrics(args.baseline_dir / "rollout_len-64_nfe-1.json"),
            "baseline_full": read_metrics(args.baseline_dir / "rollout_len-full_nfe-1.json"),
        }
    elif args.variant_dir:
        result = {"status": "complete"}
        for length in ("64", "full"):
            paths_by_nfe = {}
            for nfe in args.nfes:
                path = args.variant_dir / f"rollout_len-{length}_nfe-{nfe}.json"
                paths_by_nfe[nfe] = path
                result[f"variant_{length}_nfe{nfe}"] = read_metrics(path)
            analysis = paired_analysis(paths_by_nfe)
            if analysis is not None:
                result[f"paired_{length}"] = analysis
    else:
        result = {"status": "complete"}
        for length in ("64", "full"):
            result[f"control_{length}"] = read_metrics(
                args.control_dir / f"rollout_len-{length}_nfe-1.json"
            )
            for nfe in (1, 2, 4):
                result[f"driftflow_{length}_nfe{nfe}"] = read_metrics(
                    args.driftflow_dir / f"rollout_len-{length}_nfe-{nfe}.json"
                )
    if args.wandb_project:
        import wandb

        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name,
            job_type="rollout-eval",
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
        )
        wandb.log(dict(numeric_items(result)))
        result["wandb_run_id"] = run.id
        run.finish()
    payload = json.dumps(result, separators=(",", ":"))
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()

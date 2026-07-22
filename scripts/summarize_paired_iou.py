#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def paired_summary(baseline, method, bootstrap_samples=10000, seed=0):
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    method = np.asarray(method, dtype=np.float64).reshape(-1)
    if baseline.shape != method.shape:
        raise ValueError(f"Paired arrays differ: {baseline.shape} != {method.shape}")
    if baseline.size == 0:
        raise ValueError("Paired arrays are empty")

    differences = method - baseline
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, (bootstrap_samples, differences.size))
    bootstrap_means = differences[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "pairs": int(differences.size),
        "baseline_mean_iou": float(baseline.mean()),
        "method_mean_iou": float(method.mean()),
        "paired_mean_delta": float(differences.mean()),
        "paired_median_delta": float(np.median(differences)),
        "paired_bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "primary_gate_passed": bool(lower > 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("method", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = paired_summary(
        np.load(args.baseline),
        np.load(args.method),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

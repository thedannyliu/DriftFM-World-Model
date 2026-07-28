#!/usr/bin/env python3
"""Print compact progress and metrics for the locked advantage frontier."""

import argparse
import csv
import json
from pathlib import Path


def compact(values):
    return "/".join(f"{value:.6g}" for value in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scope", choices=("node-a", "node-b", "all"), default="all", nargs="?")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("/group-volume/danny-dataset/driftworld"),
    )
    args = parser.parse_args()

    plan_path = Path(__file__).with_name("advantage_frontier_plan.tsv")
    with plan_path.open(newline="") as plan_file:
        rows = [
            row
            for row in csv.reader(
                (line for line in plan_file if not line.startswith("#")),
                delimiter="\t",
            )
            if args.scope == "all" or row[0] == args.scope
        ]

    completed = 0
    for role, name, tag, seed, checkpoint, family in rows:
        marker = (
            args.asset_root
            / "checkpoints"
            / "experiments"
            / (
                f"eval-{tag}-seed{seed}-{checkpoint}-endpoint_normalized-"
                "advantage-locked1000.json"
            )
        )
        if not marker.is_file() or marker.stat().st_size == 0:
            print(
                f"WAIT queue={role} name={name} family={family} marker={marker}"
            )
            continue
        result = json.loads(marker.read_text())
        lpips = [
            result[f"variant_full_nfe{nfe}"]["lpips"]
            for nfe in (1, 2, 4, 8)
        ]
        mse = [
            result[f"variant_full_nfe{nfe}"]["mse"]
            for nfe in (1, 2, 4, 8)
        ]
        vertex = [
            result[f"variant_full_nfe{nfe}"]["final_block_vertex_error"]
            for nfe in (1, 2, 4, 8)
        ]
        routing = result.get("paired_full", {}).get(
            "action_routing_nfe1_or_2", {}
        )
        routing_metrics = routing.get("metrics", {})
        route_mse = routing_metrics.get("mse", {}).get(
            "adaptive_vs_random_relative_change"
        )
        route_vertex = routing_metrics.get(
            "final_block_vertex_error", {}
        ).get("adaptive_vs_random_relative_change")
        print(
            f"DONE queue={role} name={name} family={family} "
            f"lpips_1/2/4/8={compact(lpips)} "
            f"mse_1/2/4/8={compact(mse)} "
            f"vertex_1/2/4/8={compact(vertex)} "
            f"route_mse={route_mse if route_mse is not None else 'NA'} "
            f"route_vertex={route_vertex if route_vertex is not None else 'NA'} "
            f"wandb={result.get('wandb_run_id', 'NA')}"
        )
        completed += 1
    print(
        f"advantage_frontier_status scope={args.scope} "
        f"completed={completed}/{len(rows)}"
    )


if __name__ == "__main__":
    main()

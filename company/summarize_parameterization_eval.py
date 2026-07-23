#!/usr/bin/env python3
"""Combine endpoint-normalized rollout evaluations into a short JSON."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True, metavar="NAME=PATH")
    args = parser.parse_args()
    runs = []
    for item in args.result:
        name, separator, path = item.partition("=")
        if not separator:
            parser.error("--result must use NAME=PATH")
        result = json.loads(Path(path).read_text())
        lpips = {
            f"nfe{nfe}": result[f"variant_full_nfe{nfe}"]["lpips"]
            for nfe in (1, 2, 4)
        }
        runs.append({
            "name": name,
            "full_lpips": lpips,
            "nfe2_vs_nfe1": lpips["nfe2"] / lpips["nfe1"],
            "nfe4_vs_nfe1": lpips["nfe4"] / lpips["nfe1"],
            "wandb": result.get("wandb_run_id"),
        })
    print(json.dumps({"status": "complete", "runs": runs}, separators=(",", ":")))


if __name__ == "__main__":
    main()

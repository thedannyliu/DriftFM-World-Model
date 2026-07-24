#!/usr/bin/env python3
"""Select a rollout candidate with a preregistered quality/monotonicity score."""

import argparse
import json
import math
from pathlib import Path


def read_candidate(name, path):
    result = json.loads(path.read_text())
    lpips = [
        result[f"variant_full_nfe{nfe}"]["lpips"]
        for nfe in (1, 2, 4)
    ]
    vertex = [
        result[f"variant_full_nfe{nfe}"]["final_block_vertex_error"]
        for nfe in (1, 2, 4)
    ]
    if not all(math.isfinite(value) for value in (*lpips, *vertex)):
        raise ValueError(f"Non-finite rollout metric for {name}")
    lpips_worsening = max(0.0, lpips[1] - lpips[0]) + max(
        0.0, lpips[2] - lpips[1]
    )
    vertex_worsening = max(0.0, vertex[1] - vertex[0]) + max(
        0.0, vertex[2] - vertex[1]
    )
    score = (
        2.0 * lpips[0]
        + lpips[1]
        + lpips[2]
        + 0.01 * (2.0 * vertex[0] + vertex[1] + vertex[2])
        + 2.0 * lpips_worsening
        + 0.02 * vertex_worsening
    )
    return {
        "name": name,
        "score": score,
        "full_lpips": dict(zip(("nfe1", "nfe2", "nfe4"), lpips)),
        "full_vertex_error": dict(zip(("nfe1", "nfe2", "nfe4"), vertex)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for item in args.result:
        name, separator, raw_path = item.partition("=")
        if not separator:
            parser.error("--result must use NAME=PATH")
        candidates.append(read_candidate(name, Path(raw_path)))
    candidates.sort(key=lambda candidate: (candidate["score"], candidate["name"]))
    payload = {
        "status": "complete",
        "selected": candidates[0]["name"],
        "score": (
            "2*LPIPS1+LPIPS2+LPIPS4"
            "+0.01*(2*vertex1+vertex2+vertex4)"
            "+2*LPIPS_worsening+0.02*vertex_worsening"
        ),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, separators=(",", ":"))
    args.output.write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()

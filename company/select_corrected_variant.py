#!/usr/bin/env python3
"""Select a rollout candidate with a preregistered quality/monotonicity score."""

import argparse
import json
import math
from pathlib import Path


def read_candidate(name, path, nfes=(1, 2, 4)):
    result = json.loads(path.read_text())
    lpips = [
        result[f"variant_full_nfe{nfe}"]["lpips"]
        for nfe in nfes
    ]
    vertex = [
        result[f"variant_full_nfe{nfe}"]["final_block_vertex_error"]
        for nfe in nfes
    ]
    if not all(math.isfinite(value) for value in (*lpips, *vertex)):
        raise ValueError(f"Non-finite rollout metric for {name}")
    lpips_worsening = sum(
        max(0.0, right - left) for left, right in zip(lpips, lpips[1:])
    )
    vertex_worsening = sum(
        max(0.0, right - left) for left, right in zip(vertex, vertex[1:])
    )
    score = (
        2.0 * lpips[0]
        + sum(lpips[1:])
        + 0.01 * (2.0 * vertex[0] + sum(vertex[1:]))
        + 2.0 * lpips_worsening
        + 0.02 * vertex_worsening
    )
    return {
        "name": name,
        "score": score,
        "full_lpips": dict(zip((f"nfe{nfe}" for nfe in nfes), lpips)),
        "full_vertex_error": dict(zip((f"nfe{nfe}" for nfe in nfes), vertex)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--nfes", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.nfes[0] != 1 or any(
        right <= left for left, right in zip(args.nfes, args.nfes[1:])
    ):
        parser.error("--nfes must start at 1 and be strictly increasing")

    candidates = []
    for item in args.result:
        name, separator, raw_path = item.partition("=")
        if not separator:
            parser.error("--result must use NAME=PATH")
        candidates.append(read_candidate(name, Path(raw_path), tuple(args.nfes)))
    candidates.sort(key=lambda candidate: (candidate["score"], candidate["name"]))
    payload = {
        "status": "complete",
        "selected": candidates[0]["name"],
        "nfes": args.nfes,
        "score": (
            "2*LPIPS1+sum(LPIPS_at_higher_NFE)"
            "+0.01*(2*vertex1+sum(vertex_at_higher_NFE))"
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

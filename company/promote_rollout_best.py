#!/usr/bin/env python3
"""Keep a rollout-selected best checkpoint without retaining milestone snapshots."""

import argparse
import json
import os
import shutil
from pathlib import Path

from select_corrected_variant import read_candidate


def copy_atomic(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def write_json_atomic(payload, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--latest", required=True, type=Path)
    parser.add_argument("--best", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--nfes", nargs="+", type=int, default=(1, 2, 4))
    args = parser.parse_args()

    candidate = read_candidate(args.name, args.result, tuple(args.nfes))
    previous = json.loads(args.state.read_text()) if args.state.exists() else None
    improved = previous is None or candidate["score"] < previous["score"]
    if improved:
        copy_atomic(args.latest, args.best)
        payload = {
            "status": "complete",
            "name": args.name,
            "step": args.step,
            "score": candidate["score"],
            "nfes": args.nfes,
            "result": str(args.result),
            "checkpoint": str(args.best),
            "metrics": candidate,
        }
        write_json_atomic(payload, args.state)
    else:
        payload = previous

    print(json.dumps({
        "status": "promoted" if improved else "retained",
        "name": args.name,
        "candidate_step": args.step,
        "best_step": payload["step"],
        "candidate_score": candidate["score"],
        "best_score": payload["score"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()

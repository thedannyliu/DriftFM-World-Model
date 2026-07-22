#!/usr/bin/env python3
"""Print a short, pasteable training summary."""

import argparse
import json
import re
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_path = args.output_dir / "ckpt-latest.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_id_path = args.output_dir / "wandb_run_id.json"
    run_id = json.loads(run_id_path.read_text())["run_id"] if run_id_path.exists() else None
    log_text = args.log.read_text(errors="replace") if args.log.exists() else ""
    losses = re.findall(r"loss_backprop: ([0-9.eE+-]+)", log_text)

    print(json.dumps({
        "status": "complete",
        "role": args.role,
        "step": checkpoint["step"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_mib": round(checkpoint_path.stat().st_size / 2**20, 1),
        "wandb_run_id": run_id,
        "last_logged_loss": float(losses[-1]) if losses else None,
        "full_log": str(args.log),
    }))


if __name__ == "__main__":
    main()

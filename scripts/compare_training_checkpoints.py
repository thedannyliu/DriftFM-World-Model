#!/usr/bin/env python3
"""Compare uninterrupted and resumed training checkpoints quantitatively."""

import argparse
import json

import numpy as np
import torch


def compare_values(left, right, path, tensor_differences, value_differences, atol, rtol):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            value_differences.append({"path": path, "reason": "type mismatch"})
            return
        if left.shape != right.shape or left.dtype != right.dtype:
            value_differences.append({
                "path": path,
                "reason": f"tensor metadata mismatch: {left.shape}/{left.dtype} vs {right.shape}/{right.dtype}",
            })
            return
        if torch.equal(left, right):
            return
        delta = (left.detach().to(torch.float64) - right.detach().to(torch.float64)).abs()
        scale = torch.maximum(
            left.detach().to(torch.float64).abs(),
            right.detach().to(torch.float64).abs(),
        )
        relative = torch.where(scale > 0, delta / scale, delta)
        is_float = left.is_floating_point() or left.is_complex()
        tensor_differences.append({
            "path": path,
            "dtype": str(left.dtype),
            "numel": left.numel(),
            "different": int(torch.count_nonzero(delta).item()),
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
            "max_rel": float(relative.max().item()),
            "within_tolerance": bool(
                is_float and torch.allclose(left, right, atol=atol, rtol=rtol)
            ),
        })
    elif isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.shape != right.shape:
            value_differences.append({"path": path, "reason": "array metadata mismatch"})
        elif not np.array_equal(left, right):
            value_differences.append({"path": path, "reason": "array values differ"})
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            value_differences.append({"path": path, "reason": "dictionary keys differ"})
            return
        for key in left:
            compare_values(
                left[key], right[key], f"{path}.{key}",
                tensor_differences, value_differences, atol, rtol,
            )
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            value_differences.append({"path": path, "reason": "sequence metadata mismatch"})
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_values(
                left_item, right_item, f"{path}[{index}]",
                tensor_differences, value_differences, atol, rtol,
            )
    elif left != right:
        value_differences.append({"path": path, "reason": "values differ"})


def compare_checkpoints(continuous, resumed, world_size, atol=0.0, rtol=0.0):
    tensor_differences = []
    value_differences = []
    sections = {}
    for key in ("step", "model", "optimizer", "scheduler", "rng_state_by_rank"):
        tensor_start = len(tensor_differences)
        value_start = len(value_differences)
        compare_values(
            continuous[key], resumed[key], key,
            tensor_differences, value_differences, atol, rtol,
        )
        section_tensors = tensor_differences[tensor_start:]
        section_values = value_differences[value_start:]
        sections[key] = {
            "exact": not section_tensors and not section_values,
            "tensor_mismatches": len(section_tensors),
            "value_mismatches": len(section_values),
            "within_tolerance": (
                not section_values
                and all(item["within_tolerance"] for item in section_tensors)
            ),
        }

    rank_states = resumed["rng_state_by_rank"]
    if len(rank_states) != world_size:
        value_differences.append({
            "path": "rng_state_by_rank",
            "reason": f"expected {world_size} ranks, found {len(rank_states)}",
        })
    elif world_size > 1:
        cuda_states = [state["cuda"][rank] for rank, state in enumerate(rank_states)]
        if torch.equal(cuda_states[0], cuda_states[1]):
            value_differences.append({
                "path": "rng_state_by_rank.cuda",
                "reason": "rank-local CUDA RNG streams are identical",
            })

    top_differences = sorted(
        tensor_differences,
        key=lambda item: item["max_abs"],
        reverse=True,
    )[:12]
    exact = not tensor_differences and not value_differences
    within_tolerance = (
        not value_differences
        and all(item["within_tolerance"] for item in tensor_differences)
    )
    return {
        "status": "exact" if exact else "mismatch",
        "step": resumed["step"],
        "atol": atol,
        "rtol": rtol,
        "exact": exact,
        "within_tolerance": within_tolerance,
        "sections": sections,
        "tensor_mismatches": len(tensor_differences),
        "value_mismatches": len(value_differences),
        "top_tensor_differences": top_differences,
        "value_differences": value_differences[:12],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("continuous")
    parser.add_argument("resumed")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    continuous = torch.load(args.continuous, map_location="cpu", weights_only=False)
    resumed = torch.load(args.resumed, map_location="cpu", weights_only=False)
    report = compare_checkpoints(
        continuous,
        resumed,
        args.world_size,
        atol=args.atol,
        rtol=args.rtol,
    )
    print(json.dumps(report, separators=(",", ":")))
    if not args.report_only and not report["within_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Require uninterrupted and resumed training checkpoints to match exactly."""

import argparse

import numpy as np
import torch


def assert_equal(left, right, path):
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError(f"Tensor mismatch at {path}")
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise AssertionError(f"Array mismatch at {path}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"Key mismatch at {path}")
        for key in left:
            assert_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"Length mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            assert_equal(left_item, right_item, f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"Value mismatch at {path}: {left!r} != {right!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("continuous")
    parser.add_argument("resumed")
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()

    continuous = torch.load(args.continuous, map_location="cpu", weights_only=False)
    resumed = torch.load(args.resumed, map_location="cpu", weights_only=False)
    for key in ("step", "model", "optimizer", "scheduler", "rng_state_by_rank"):
        assert_equal(continuous[key], resumed[key], key)

    rank_states = resumed["rng_state_by_rank"]
    if len(rank_states) != args.world_size:
        raise AssertionError(
            f"Expected {args.world_size} rank RNG states, found {len(rank_states)}"
        )
    current_cuda_states = [state["cuda"][rank] for rank, state in enumerate(rank_states)]
    if args.world_size > 1 and torch.equal(current_cuda_states[0], current_cuda_states[1]):
        raise AssertionError("Rank-local CUDA RNG streams are identical")

    print(
        f"Exact checkpoint match at step {resumed['step']} across "
        f"{len(resumed['model'])} model tensors and {args.world_size} RNG states"
    )


if __name__ == "__main__":
    main()

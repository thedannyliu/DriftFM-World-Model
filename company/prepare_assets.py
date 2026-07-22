#!/usr/bin/env python3
"""Download the minimal Push-T data and checkpoints from Hugging Face."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

from huggingface_hub import hf_hub_download, snapshot_download
from requests import RequestException


MODEL_PATTERNS = [
    "pusht_checkpoints/pushT_driftworld/ckpt_save/ckpt-step1180500.pth",
    "pusht_checkpoints/reward/reward_predictor_xy.pth",
    "pusht_checkpoints/reward/reward_predictor_angle.pth",
    "pusht_checkpoints/diffusion_policy_v1/ckpt_save/ckpt-ep100.pth",
    "pusht_checkpoints/diffusion_policy_v1/ckpt_save/ckpt-ep300.pth",
]
DATASET_LFS_FILES = [
    "world_model_data/dataset_domain/all_data/"
    "domain18_single_boundary_v2.zarr/data/keypoint/0.0.0",
    "world_model_data/dataset_domain/all_data/"
    "domain18_single_long_v2.zarr/data/keypoint/0.0.0",
]
MAX_DOWNLOAD_WORKERS = 16


def load_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        print("[auth] Hugging Face token loaded from HF_TOKEN", flush=True)
        return token

    candidates = []
    if os.environ.get("HF_TOKEN_PATH"):
        candidates.append(Path(os.environ["HF_TOKEN_PATH"]))
    candidates.append(Path.home() / ".cache" / "huggingface" / "token")
    for path in candidates:
        if path.is_file():
            token = path.read_text().strip()
            if token:
                print("[auth] Hugging Face login token loaded", flush=True)
                return token

    raise RuntimeError("Hugging Face token not found; run `hf auth login` first")


def download_snapshot(
    *, label: str, local_dir: Path, max_workers: int, attempts: int = 5, **kwargs
):
    """Download concurrently and resume after transient company-network failures."""
    for attempt in range(1, attempts + 1):
        for metadata_dir in (
            local_dir / ".huggingface",
            local_dir / ".cache" / "huggingface",
        ):
            metadata_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[download] {label}: attempt {attempt}/{attempts}, workers={max_workers}",
            flush=True,
        )
        try:
            result = snapshot_download(
                local_dir=local_dir,
                max_workers=max_workers,
                **kwargs,
            )
            print(f"[download] {label}: complete", flush=True)
            return result
        except Exception as error:
            is_download_error = isinstance(error, (RequestException, OSError)) or type(
                error
            ).__module__.startswith(("huggingface_hub", "httpx"))
            if not is_download_error:
                raise
            if attempt == attempts:
                raise
            delay = min(2 ** attempt, 30)
            print(
                f"[download] {label}: {type(error).__name__}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)


def download_dataset_via_git(*, root: Path, data_root: Path, token: str):
    clone_root = root / "cache" / "git" / "gpc_pushT_data"
    clone_root.parent.mkdir(parents=True, exist_ok=True)
    git_env = os.environ.copy()
    git_env["GIT_LFS_SKIP_SMUDGE"] = "1"
    git_env["GIT_TERMINAL_PROMPT"] = "0"

    if (clone_root / ".git").is_dir():
        print("[download] Push-T dataset: updating cached Git checkout", flush=True)
        subprocess.run(
            ["git", "-C", str(clone_root), "pull", "--ff-only"],
            check=True,
            env=git_env,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(clone_root),
                "restore",
                "--source=HEAD",
                "--worktree",
                "--",
                "world_model_data",
            ],
            check=True,
            env=git_env,
        )
    elif clone_root.exists():
        raise RuntimeError(f"Incomplete Git cache exists at {clone_root}")
    else:
        print("[download] Push-T dataset: cloning 0.30 GiB Git pack", flush=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--branch",
                "main",
                "--progress",
                "https://huggingface.co/datasets/han2019/gpc_pushT_data",
                str(clone_root),
            ],
            check=True,
            env=git_env,
        )

    print("[download] Push-T dataset: copying Git pack contents", flush=True)
    shutil.copytree(
        clone_root / "world_model_data",
        data_root / "world_model_data",
        dirs_exist_ok=True,
    )
    print("[download] Push-T dataset: fetching 2 Git LFS objects", flush=True)
    for metadata_dir in (
        data_root / ".huggingface",
        data_root / ".cache" / "huggingface",
    ):
        metadata_dir.mkdir(parents=True, exist_ok=True)
    for filename in DATASET_LFS_FILES:
        hf_hub_download(
            repo_id="han2019/gpc_pushT_data",
            repo_type="dataset",
            filename=filename,
            local_dir=data_root,
            token=token,
        )
    print("[download] Push-T dataset: complete", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("/group-volume/danny-dataset/driftworld"),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    max_workers = min(args.max_workers, MAX_DOWNLOAD_WORKERS)
    if max_workers != args.max_workers:
        print(
            f"[download] limiting workers from {args.max_workers} to {max_workers}",
            flush=True,
        )
    root = args.asset_root.resolve()
    model_root = root / "checkpoints" / "official"
    data_root = root / "data"
    cache_root = root / "cache" / "huggingface"
    model_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    token = load_hf_token()

    download_snapshot(
        label="official checkpoints",
        max_workers=max_workers,
        repo_id="Susie-Lu/driftworld",
        local_dir=model_root,
        cache_dir=cache_root,
        allow_patterns=MODEL_PATTERNS,
        token=token,
    )
    try:
        download_dataset_via_git(root=root, data_root=data_root, token=token)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(
            f"[download] Git path failed ({type(error).__name__}); using HF snapshot fallback",
            flush=True,
        )
        download_snapshot(
            label="Push-T dataset fallback",
            max_workers=max_workers,
            repo_id="han2019/gpc_pushT_data",
            repo_type="dataset",
            local_dir=data_root,
            cache_dir=cache_root,
            allow_patterns=["world_model_data/**"],
            token=token,
        )

    expected = [model_root / pattern for pattern in MODEL_PATTERNS]
    data_dir = data_root / "world_model_data" / "dataset_domain" / "all_data"
    expected.append(data_dir)
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Incomplete Hugging Face download: {missing}")

    zarr_count = sum(path.name.endswith(".zarr") for path in data_dir.iterdir())
    if zarr_count != 16:
        raise RuntimeError(f"Expected 16 Push-T Zarr datasets, found {zarr_count}")

    print(json.dumps({
        "status": "ready",
        "asset_root": str(root),
        "zarr_datasets": zarr_count,
        "checkpoint_files": len(MODEL_PATTERNS),
    }))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download the minimal Push-T data and checkpoints from Hugging Face."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

from huggingface_hub import snapshot_download
from requests import RequestException


MODEL_PATTERNS = [
    "pusht_checkpoints/pushT_driftworld/ckpt_save/ckpt-step1180500.pth",
    "pusht_checkpoints/reward/reward_predictor_xy.pth",
    "pusht_checkpoints/reward/reward_predictor_angle.pth",
    "pusht_checkpoints/diffusion_policy_v1/ckpt_save/ckpt-ep100.pth",
    "pusht_checkpoints/diffusion_policy_v1/ckpt_save/ckpt-ep300.pth",
]


def download_snapshot(*, label: str, local_dir: Path, attempts: int = 5, **kwargs):
    """Download serially and resume after transient company-network failures."""
    for attempt in range(1, attempts + 1):
        for metadata_dir in (
            local_dir / ".huggingface",
            local_dir / ".cache" / "huggingface",
        ):
            metadata_dir.mkdir(parents=True, exist_ok=True)
        print(f"[download] {label}: attempt {attempt}/{attempts}", flush=True)
        try:
            result = snapshot_download(local_dir=local_dir, max_workers=1, **kwargs)
            print(f"[download] {label}: complete", flush=True)
            return result
        except (RequestException, OSError) as error:
            if attempt == attempts:
                raise
            delay = min(2 ** attempt, 30)
            print(
                f"[download] {label}: {type(error).__name__}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("/group-volume/danny-dataset/driftworld"),
    )
    args = parser.parse_args()
    root = args.asset_root.resolve()
    model_root = root / "checkpoints" / "official"
    data_root = root / "data"
    cache_root = root / "cache" / "huggingface"
    model_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    download_snapshot(
        label="official checkpoints",
        repo_id="Susie-Lu/driftworld",
        local_dir=model_root,
        cache_dir=cache_root,
        allow_patterns=MODEL_PATTERNS,
    )
    download_snapshot(
        label="Push-T dataset",
        repo_id="han2019/gpc_pushT_data",
        repo_type="dataset",
        local_dir=data_root,
        cache_dir=cache_root,
        allow_patterns=["world_model_data/**"],
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

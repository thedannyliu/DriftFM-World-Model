#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path("/storage/project/r-agarg35-0/eliu354/projects/driftworld")


def ensure_link(link, target):
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise RuntimeError(f"Refusing to replace existing path: {link}")
    link.symlink_to(target, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/storage/scratch1/9/eliu354/driftflowworld"),
    )
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    model_root = root / "checkpoints" / "official"
    data_root = root / "data"
    cache_root = root / "cache" / "huggingface"
    for path in (model_root, cache_root, root / "runs", root / "slurm_logs"):
        path.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="Susie-Lu/driftworld",
        local_dir=model_root,
        cache_dir=cache_root,
    )
    if not (data_root / ".git").exists():
        if data_root.exists():
            raise RuntimeError(f"Refusing to replace non-Git data directory: {data_root}")
        environment = os.environ.copy()
        environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://huggingface.co/datasets/han2019/gpc_pushT_data",
                str(data_root),
            ],
            check=True,
            env=environment,
        )
    subprocess.run(
        [
            "git", "lfs", "pull",
            "--include=world_model_data/**",
            "--exclude=diffusion_policy_data/**",
        ],
        check=True,
        cwd=data_root,
    )

    expected = [
        model_root / "pusht_checkpoints" / "pushT_driftworld" / "ckpt_save" / "ckpt-step1180500.pth",
        model_root / "pusht_checkpoints" / "reward" / "reward_predictor_xy.pth",
        data_root / "world_model_data" / "dataset_domain" / "all_data",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Downloaded snapshot is incomplete: {missing}")

    ensure_link(REPO_ROOT / "driftworld" / "pusht_checkpoints", model_root / "pusht_checkpoints")
    ensure_link(REPO_ROOT / "driftworld" / "pusht_data", data_root)
    print(f"Assets ready under {root}")


if __name__ == "__main__":
    main()

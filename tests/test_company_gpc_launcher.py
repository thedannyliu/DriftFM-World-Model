from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gpc_launcher_uses_monolithic_policy_checkpoints():
    launcher = (
        REPO_ROOT / "company" / "run_gpc_budget_eval.sh"
    ).read_text()
    queue = (
        REPO_ROOT / "company" / "run_unordered_fidelity_queue.sh"
    ).read_text()

    assert "diffusion_policy_v1/ckpt_save/ckpt-${POLICY}.pth" in launcher
    assert "ckpt.use_official=false" in launcher
    assert 'train.seed="${POLICY_SEED}"' in launcher
    assert "[[ ! -f ${POLICY_CHECKPOINT}" in launcher
    assert "torch.load(path, map_location=\"cpu\", weights_only=False)" in queue
    assert '{"model", "ema"} - set(checkpoint)' in queue

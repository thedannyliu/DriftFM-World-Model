from pathlib import Path
import csv


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


def test_gpc_launcher_supports_held_out_trial_offsets():
    launcher = (
        REPO_ROOT / "company" / "run_gpc_budget_eval.sh"
    ).read_text()

    assert "TRIAL_OFFSET=${GPC_TRIAL_OFFSET:-0}" in launcher
    assert "START=$((TRIAL_OFFSET + GPU * SHARD_SIZE))" in launcher


def test_confirmation_plan_balances_families_and_policies():
    with (
        REPO_ROOT / "company" / "unordered_confirmation_plan.tsv"
    ).open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))

    assert len(rows) == 32
    for role in ("node-a", "node-b", "node-c", "node-d"):
        role_rows = [row for row in rows if row[0] == role]
        assert len(role_rows) == 8
        assert {row[6] for row in role_rows} == {"ep100", "ep300"}
        assert {int(row[8]) for row in role_rows} == {32}
        assert {int(row[9]) for row in role_rows} == {1, 2, 4, 8}
        assert all(row[1].startswith("a-confirm80-") for row in role_rows)


def test_decision_fidelity_plan_separates_evidence_and_smoke():
    with (
        REPO_ROOT / "company" / "decision_fidelity_pusht_plan.tsv"
    ).open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))

    assert all(len(row) == 13 for row in rows)
    node_a = [row for row in rows if row[0] == "node-a"]
    node_b = [row for row in rows if row[0] == "node-b"]
    assert len(node_a) == 8
    assert {row[5] for row in node_a} == {
        "k1-grid25",
        "k32",
        "joint-k16",
        "deep-base-k16",
    }
    assert {row[6] for row in node_a} == {"ep100", "ep300"}
    assert {(int(row[8]), int(row[9])) for row in node_a} == {(64, 1)}
    assert len(node_b) == 4
    assert {row[5] for row in node_b} == {"k32"}
    assert {row[6] for row in node_b} == {"ep100", "ep300"}
    assert {int(row[9]) for row in node_b} == {1, 4}

    queue = (
        REPO_ROOT / "company" / "run_decision_fidelity_pusht.sh"
    ).read_text()
    assert "GPC_CANDIDATE_ANCHOR_COUNT=32" in queue
    assert "GPC_EXECUTION_STRATEGY=policy_first" in queue
    assert "GPC_AUDIT_MAX_DECISIONS=4" in queue
    assert "GPC_AUDIT_REPEAT_GROUND_TRUTH=true" in queue

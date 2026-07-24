import json
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "company"))

from select_corrected_variant import read_candidate


def write_result(path, lpips, vertex=None):
    vertex = vertex or lpips
    payload = {}
    for nfe, lpips_value, vertex_value in zip((1, 2, 4, 8), lpips, vertex):
        payload[f"variant_full_nfe{nfe}"] = {
            "lpips": lpips_value,
            "final_block_vertex_error": vertex_value,
        }
    path.write_text(json.dumps(payload))


def test_candidate_score_includes_nfe_eight(tmp_path):
    stable = tmp_path / "stable.json"
    unstable = tmp_path / "unstable.json"
    write_result(stable, (0.01, 0.009, 0.008, 0.007))
    write_result(unstable, (0.01, 0.009, 0.008, 0.1))

    stable_score = read_candidate("stable", stable, (1, 2, 4, 8))["score"]
    unstable_score = read_candidate("unstable", unstable, (1, 2, 4, 8))["score"]

    assert stable_score < unstable_score


def test_rollout_best_retains_lower_score_checkpoint(tmp_path):
    script = Path(__file__).parents[1] / "company" / "promote_rollout_best.py"
    latest = tmp_path / "ckpt-latest.pth"
    best = tmp_path / "ckpt-best.pth"
    state = tmp_path / "rollout-best.json"
    result = tmp_path / "result.json"
    latest.write_bytes(b"first")
    write_result(result, (0.01, 0.009, 0.008, 0.007))

    command = [
        sys.executable,
        str(script),
        "--name", "candidate",
        "--step", "10000",
        "--result", str(result),
        "--latest", str(latest),
        "--best", str(best),
        "--state", str(state),
        "--nfes", "1", "2", "4", "8",
    ]
    subprocess.run(command, check=True)
    assert best.read_bytes() == b"first"

    latest.write_bytes(b"second")
    write_result(result, (0.02, 0.02, 0.02, 0.02))
    command[command.index("10000")] = "30000"
    subprocess.run(command, check=True)

    assert best.read_bytes() == b"first"
    assert json.loads(state.read_text())["step"] == 10000

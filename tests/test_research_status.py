import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]


def run_status(script, *args):
    result = subprocess.run(
        (sys.executable, str(REPO_ROOT / "company" / script), *map(str, args)),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def test_weekend_status_lists_every_unstarted_run(tmp_path):
    output = run_status(
        "status_weekend_research.py",
        "all",
        "--asset-root",
        tmp_path,
    )

    assert output.count("  WAIT ") == 54
    assert output.count("state=pending") == 4
    assert "planned_tags=12 active_tags=0" in output
    assert "planned_tags=6 active_tags=0" in output


def test_weekend_status_expands_selected_winner_and_runner_seeds(tmp_path):
    experiment_root = tmp_path / "checkpoints" / "experiments"
    experiment_root.mkdir(parents=True)
    selection = {
        "candidates": [
            {"name": "wknd-a-base", "score": 0.1},
            {"name": "wknd-a-grid25", "score": 0.2},
        ]
    }
    (experiment_root / "weekend-selection-node-a-step30000.json").write_text(
        json.dumps(selection)
    )

    output = run_status(
        "status_weekend_research.py",
        "node-a",
        "--asset-root",
        tmp_path,
    )

    assert "SELECTION winner=wknd-a-base@0.1 runner_up=wknd-a-grid25@0.2" in output
    assert "WAIT tag=wknd-a-base seed=5 target=400000" in output
    assert "WAIT tag=wknd-a-grid25 seed=3 target=200000" in output
    assert "WAIT tag=wknd-a-sr25 seed=1 target=30000" in output


def test_unified_status_runs_all_reporters_and_inventory(tmp_path):
    output = run_status(
        "status_all_experiments.py",
        "--asset-root",
        tmp_path,
        "--runtime-root",
        tmp_path / "runtime",
    )

    assert "===== OVERNIGHT_BASELINE_AND_ORIGINAL_DFM =====" in output
    assert "===== CORRECTED_LONG_RESEARCH =====" in output
    assert "===== AGGRESSIVE_WEEKEND_RESEARCH =====" in output
    assert "===== SHARED_ARTIFACT_INVENTORY =====" in output
    assert "status=complete failed_reporters=none" in output

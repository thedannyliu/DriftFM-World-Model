from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_company_environment_uses_compatible_opencv_and_pymunk():
    requirements = (
        REPO_ROOT / "company" / "requirements.txt"
    ).read_text().splitlines()
    pinned = {
        name: version
        for line in requirements
        if line and not line.startswith("#")
        for name, version in [line.split("==", 1)]
    }

    assert pinned["opencv-python-headless"] == "4.10.0.84"
    assert "opencv-python" not in pinned
    assert pinned["pymunk"] == "7.3.0"

    setup = (REPO_ROOT / "company" / "setup.sh").read_text()
    queue = (
        REPO_ROOT / "company" / "run_unordered_fidelity_queue.sh"
    ).read_text()
    assert "pip uninstall -y" in setup
    assert "opencv-contrib-python-headless" in setup
    assert "import cv2" in setup
    assert "import cv2" in queue
    assert 'hasattr(pymunk.Space, "on_collision")' in setup
    assert 'hasattr(pymunk.Space, "on_collision")' in queue

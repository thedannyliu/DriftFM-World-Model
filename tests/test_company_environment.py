from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_company_environment_uses_headless_opencv_and_preflights_cv2():
    requirements = (
        REPO_ROOT / "company" / "requirements.txt"
    ).read_text().splitlines()
    installed = {
        line.split("==", 1)[0]
        for line in requirements
        if line and not line.startswith("#")
    }

    assert "opencv-python-headless" in installed
    assert "opencv-python" not in installed

    setup = (REPO_ROOT / "company" / "setup.sh").read_text()
    queue = (
        REPO_ROOT / "company" / "run_unordered_fidelity_queue.sh"
    ).read_text()
    assert "pip uninstall -y" in setup
    assert "opencv-contrib-python-headless" in setup
    assert "import cv2" in setup
    assert "import cv2" in queue

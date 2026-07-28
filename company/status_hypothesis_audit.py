#!/usr/bin/env python3
"""Print one combined report from the latest four hypothesis-audit nodes."""

import os
from pathlib import Path
import subprocess
import sys


NODES = ("node-a", "node-b", "node-c", "node-d")


def main():
    runtime_root = Path(os.environ.get(
        "DRIFTFLOWWORLD_RUNTIME_ROOT", "/user-volume/driftworld"
    ))
    audit_root = runtime_root / "results" / "hypothesis-audit"
    command = [
        sys.executable,
        str(Path(__file__).with_name("summarize_hypothesis_audit.py")),
    ]
    missing = []
    selected = []
    for node in NODES:
        candidates = sorted(
            path for path in audit_root.glob(f"{node}-*")
            if (path / "summary.json").is_file()
        )
        if not candidates:
            missing.append(node)
            continue
        directory = candidates[-1]
        results = sorted(
            path for path in directory.glob("*.json")
            if path.name != "summary.json"
        )
        if len(results) != 4:
            missing.append(f"{node}(results={len(results)})")
            continue
        selected.append(directory)
        for result in results:
            command.extend((
                "--result",
                f"{node}:{result.stem}={result}",
            ))

    print(
        "hypothesis_audit_status "
        f"selected={','.join(map(str, selected)) or 'none'} "
        f"missing={','.join(missing) or 'none'}"
    )
    if not selected:
        raise SystemExit(1)
    completed = subprocess.run(command, check=False)
    if missing:
        print(
            "status=partial run the missing node commands before making a "
            "research decision"
        )
    else:
        print("status=complete all four node summaries included")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

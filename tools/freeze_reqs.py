"""Freeze runtime dependencies to requirements.txt."""
from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    result = subprocess.run(
        ["uv", "pip", "freeze", "--exclude-editable"],
        check=True,
        capture_output=True,
        text=True,
    )
    Path("requirements.txt").write_text(result.stdout)


if __name__ == "__main__":
    main()

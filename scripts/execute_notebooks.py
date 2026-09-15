#!/usr/bin/env python3
"""Execute every published weekly notebook in an isolated temporary output folder."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    notebooks = sorted(repo_root.glob("week*/notebooks/*.ipynb"))
    notebooks = [path for path in notebooks if ".ipynb_checkpoints" not in path.parts]
    if not notebooks:
        print("No weekly notebooks were found.", file=sys.stderr)
        return 1

    executable_dir = Path(sys.executable).parent
    jupyter = executable_dir / ("jupyter.exe" if os.name == "nt" else "jupyter")
    if not jupyter.is_file():
        print(f"Jupyter executable not found beside Python: {jupyter}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="hi634-notebooks-") as output_dir:
        for notebook in notebooks:
            relative = notebook.relative_to(repo_root)
            print(f"Executing {relative}")
            subprocess.run(
                [
                    str(jupyter),
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    str(notebook),
                    "--output-dir",
                    output_dir,
                    "--ExecutePreprocessor.timeout=300",
                ],
                cwd=repo_root,
                check=True,
            )

    print(f"Executed {len(notebooks)} notebook(s) successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

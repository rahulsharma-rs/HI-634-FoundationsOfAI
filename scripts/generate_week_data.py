#!/usr/bin/env python3
"""Run every weekly deterministic teaching-data generator."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    generators = sorted(repo_root.glob("week*/scripts/generate_synthetic_data.py"))
    if not generators:
        print("No weekly synthetic-data generators were found.", file=sys.stderr)
        return 1

    for generator in generators:
        relative = generator.relative_to(repo_root)
        print(f"Running {relative}")
        subprocess.run([sys.executable, str(generator)], cwd=repo_root, check=True)

    print(f"Ran {len(generators)} synthetic-data generator(s) successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

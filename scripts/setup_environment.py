#!/usr/bin/env python3
"""Create and verify the repository's Python 3.14 teaching environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv


REQUIRED_PYTHON = (3, 14)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"\n> {printable}")
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create .venv, install the pinned HI 634 requirements, and verify "
            "the teaching environment."
        )
    )
    parser.add_argument(
        "--venv",
        default=".venv",
        help="virtual-environment path relative to the repository (default: .venv)",
    )
    parser.add_argument(
        "--no-pip-upgrade",
        action="store_true",
        help="skip upgrading pip before installing requirements",
    )
    return parser.parse_args()


def main() -> int:
    if sys.version_info[:2] != REQUIRED_PYTHON:
        found = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(
            f"ERROR: HI 634 requires Python 3.14.x; this command used Python {found}.\n"
            "Use `python3.14 scripts/setup_environment.py` on macOS/Linux or "
            "`py -3.14 scripts\\setup_environment.py` on Windows.",
            file=sys.stderr,
        )
        return 2

    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    requested = Path(args.venv).expanduser()
    venv_dir = requested if requested.is_absolute() else repo_root / requested
    requirements = repo_root / "requirements.txt"

    if not requirements.is_file():
        print(f"ERROR: requirements file not found: {requirements}", file=sys.stderr)
        return 2

    python = venv_python(venv_dir)
    if not python.exists():
        print(f"Creating virtual environment at {venv_dir}")
        venv.EnvBuilder(with_pip=True).create(venv_dir)
    else:
        print(f"Reusing virtual environment at {venv_dir}")

    version_check = subprocess.run(
        [str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not version_check.startswith("3.14."):
        print(
            f"ERROR: {venv_dir} uses Python {version_check}, not Python 3.14.x. "
            "Remove only that virtual-environment directory and rerun this helper.",
            file=sys.stderr,
        )
        return 2

    if not args.no_pip_upgrade:
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(python), "-m", "pip", "install", "-r", str(requirements)])
    run([str(python), str(repo_root / "scripts" / "verify_install.py")])

    jupyter = venv_dir / ("Scripts" if os.name == "nt" else "bin") / (
        "jupyter.exe" if os.name == "nt" else "jupyter"
    )
    print("\nSetup complete.")
    print(f"Start JupyterLab with:\n  {jupyter} lab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

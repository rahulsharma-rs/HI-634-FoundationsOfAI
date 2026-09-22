#!/usr/bin/env python3
"""Check the Python version, core imports, and current teaching files."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from pathlib import Path
import sys


PACKAGES = {
    "jupyterlab": "jupyterlab",
    "ipykernel": "ipykernel",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "nltk": "nltk",
    "spacy": "spacy",
}


def main() -> int:
    if sys.version_info[:2] != (3, 14):
        print(f"FAIL: expected Python 3.14.x, found {sys.version.split()[0]}")
        return 1

    failures: list[str] = []
    print(f"Python {sys.version.split()[0]}")
    for distribution, module in PACKAGES.items():
        try:
            import_module(module)
            print(f"  {distribution}=={version(distribution)}")
        except Exception as exc:  # report every missing or broken import together
            failures.append(f"{distribution}: {exc}")

    repo_root = Path(__file__).resolve().parents[1]
    required_files = [
        repo_root / "week4" / "notebooks" / "week4_machine_learning_health_services.ipynb",
        repo_root / "week4" / "data" / "breast_cancer_wisconsin_diagnostic.csv",
        repo_root / "week4" / "data" / "synthetic_length_of_stay.csv",
        repo_root / "week4" / "data" / "synthetic_claims.csv",
        repo_root / "week5" / "notebooks" / "week5_natural_language_processing_health_services.ipynb",
        repo_root / "week5" / "data" / "synthetic_care_management_notes.csv",
        repo_root / "week5" / "data" / "synthetic_social_needs_training.csv",
        repo_root / "week5" / "data" / "synthetic_language_challenge.csv",
    ]
    missing = [str(path.relative_to(repo_root)) for path in required_files if not path.is_file()]
    if missing:
        failures.append("missing teaching files: " + ", ".join(missing))

    if failures:
        print("\nEnvironment verification FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nEnvironment verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

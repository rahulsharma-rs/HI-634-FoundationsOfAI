# HI 634 Teaching Code

This repository contains reproducible weekly notebooks and data for **HI 634 Foundations of Artificial Intelligence in Health Services**. Each week is self-contained so learners can run the examples, inspect assumptions, and extend the exercises without changing another week's materials.

The Week 4 notebook is the first completed teaching module. It follows the course chapter on machine learning and data mining for health services and emphasizes decision framing, leakage-safe modeling, appropriate evaluation, cautious interpretation, Responsible AI, and Secure AI.

> Educational use only. The examples do not create clinically validated models and must not be used for diagnosis, treatment, eligibility, payment, or other patient-level decisions.

## Repository layout

```text
CodeBaseHI634/
├── .github/workflows/       # Automated notebook execution checks
├── .venv/                   # Local Python 3.14 environment, not committed
├── scripts/                 # Cross-platform setup and repository checks
├── INSTALL.md               # Detailed macOS, Linux, and Windows setup
├── requirements.txt         # Reproducible teaching dependencies
└── week4/
    ├── data/                # Public and clearly labeled synthetic datasets
    ├── notebooks/           # Classroom notebooks
    ├── scripts/             # Reproducible data and notebook builders
    └── source_notes/        # Chapter inventory and teaching map
```

Future modules should use the same pattern: `week5/data/`, `week5/notebooks/`, `week5/scripts/`, and `week5/source_notes/`.

## Student quick start

Install Python 3.14, clone the repository, and run the setup helper. Detailed macOS, Linux, Windows, troubleshooting, and future-week instructions are in [INSTALL.md](INSTALL.md).

### macOS or Linux

```bash
git clone https://github.com/rahulsharma-rs/HI-634-FoundationsOfAI.git
cd HI-634-FoundationsOfAI
python3.14 scripts/setup_environment.py
.venv/bin/jupyter lab
```

### Windows PowerShell

```powershell
git clone https://github.com/rahulsharma-rs/HI-634-FoundationsOfAI.git
cd HI-634-FoundationsOfAI
py -3.14 scripts\setup_environment.py
.venv\Scripts\jupyter.exe lab
```

Open `week4/notebooks/week4_machine_learning_health_services.ipynb` in JupyterLab.

## Verify the teaching materials

Verify installed packages:

```bash
.venv/bin/python scripts/verify_install.py
```

Execute every published weekly notebook in a temporary output directory:

```bash
.venv/bin/python scripts/execute_notebooks.py
```

Recreate the deterministic synthetic datasets:

```bash
.venv/bin/python week4/scripts/generate_synthetic_data.py
```

Rebuild the notebook source after editing its builder:

```bash
.venv/bin/python week4/scripts/build_week4_notebook.py
```

## Adding a new teaching week

1. Create `weekX/data`, `weekX/notebooks`, `weekX/scripts`, and `weekX/source_notes`.
2. Keep every dataset used by the notebook under that week's `data` directory.
3. Add a data README with provenance, license, retrieval date, and checksum, or a clear synthetic-data statement and generation method.
4. State measurable learning objectives and connect code outputs to health-service decisions.
5. Execute the notebook from top to bottom before committing it.
6. Never commit credentials, protected health information, identifiable patient records, or the local `.venv` directory.

## Dependency policy for future chapters

`requirements.txt` is the single pinned environment for all published course notebooks. When a chapter needs a new package, add its exact tested version there and rerun the setup and notebook checks. Keeping one environment prevents students from having to guess which requirements file applies to a lesson.

The automated notebook check discovers every `week*/notebooks/*.ipynb` file, so newly added weeks join continuous integration without editing the execution script.

## Publishing to GitHub

Before publishing changes, review data attribution and confirm that no restricted course readings, credentials, PHI, or student information have been added. The repository uses the license stored in `LICENSE`.

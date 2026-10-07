# HI 634 Teaching Code

This repository contains reproducible weekly notebooks and data for **HI 634 Foundations of Artificial Intelligence in Health Services**. Each week is self-contained so learners can run the examples, inspect assumptions, and extend the exercises without changing another week's materials.

The modules cover Week 4 machine learning, Week 5 natural language processing, and a Colab-based Week 7 lesson on building a tiny language model. The offline Week 4 and Week 5 materials emphasize decision framing, leakage-safe modeling, appropriate evaluation, cautious interpretation, Responsible AI, and Secure AI.

> Educational use only. The examples do not create clinically validated models and must not be used for diagnosis, treatment, eligibility, payment, or other patient-level decisions.

## Repository layout

```text
CodeBaseHI634/
├── .github/workflows/       # Automated notebook execution checks
├── .venv/                   # Local Python 3.14 environment, not committed
├── scripts/                 # Cross-platform setup and repository checks
├── INSTALL.md               # Detailed macOS, Linux, and Windows setup
├── requirements.txt         # Reproducible teaching dependencies
├── week4/
│   ├── data/                # Public and clearly labeled synthetic datasets
│   ├── notebooks/           # Classroom notebooks
│   ├── scripts/             # Reproducible data and notebook builders
│   └── source_notes/        # Chapter inventory and teaching map
├── week5/                   # NLP module using the same weekly structure
└── week7/                   # Colab-based tiny LLM lesson and earlier source version
```

Future offline modules should use the same pattern: `weekX/data/`, `weekX/notebooks/`, `weekX/scripts/`, and `weekX/source_notes/`.

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

Open a notebook under `week4/notebooks/` or `week5/notebooks/` in JupyterLab.

For Week 7, follow the [Google Colab instructions](week7/README.md) and open the current notebook from GitHub in Colab.

## Verify the teaching materials

Verify installed packages:

```bash
.venv/bin/python scripts/verify_install.py
```

Execute every published weekly notebook in a temporary output directory:

```bash
.venv/bin/python scripts/execute_notebooks.py
```

Recreate all deterministic weekly synthetic datasets:

```bash
.venv/bin/python scripts/generate_week_data.py
```

Rebuild notebook source after editing a week's builder:

```bash
.venv/bin/python week4/scripts/build_week4_notebook.py
.venv/bin/python week5/scripts/build_week5_notebook.py
```

## Adding a new teaching week

1. Create `weekX/data`, `weekX/notebooks`, `weekX/scripts`, and `weekX/source_notes`.
2. Keep every dataset used by the notebook under that week's `data` directory.
3. Add a data README with provenance, license, retrieval date, and checksum, or a clear synthetic-data statement and generation method.
4. State measurable learning objectives and connect code outputs to health-service decisions.
5. Execute the notebook from top to bottom before committing it.
6. Never commit credentials, protected health information, identifiable patient records, or the local `.venv` directory.

## Dependency policy for future chapters

`requirements.txt` is the single pinned environment for the offline course notebooks. When an offline chapter needs a new package, add its exact tested version there and rerun the setup and notebook checks. Week 7 uses Colab's managed runtime and follows the setup steps in its own README.

The automated notebook check discovers every `week*/notebooks/*.ipynb` file. Week 7 uses `week7/colab/` because it needs external downloads and longer training, so it is run interactively in Colab.

## Publishing to GitHub

Before publishing changes, review data attribution and confirm that no restricted course readings, credentials, PHI, or student information have been added. The repository uses the license stored in `LICENSE`.

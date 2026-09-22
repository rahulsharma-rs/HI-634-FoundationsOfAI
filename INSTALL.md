# Installing the HI 634 teaching environment

The course uses one repository-level Python environment so every published week can be opened from the same JupyterLab installation.

## 1. Install Python

Install a **Python 3.14.x** release from [python.org](https://www.python.org/downloads/) or through your institution's approved software manager. This repository was tested with Python 3.14.5.

Confirm the installation:

### macOS or Linux

```bash
python3.14 --version
```

### Windows PowerShell

```powershell
py -3.14 --version
```

If the command is not found, close and reopen the terminal after installing Python. On Windows, enable the Python launcher when the installer offers that option.

## 2. Clone and enter the repository

```bash
git clone https://github.com/rahulsharma-rs/HI-634-FoundationsOfAI.git
cd HI-634-FoundationsOfAI
```

## 3. Run the setup helper

The helper creates `.venv`, upgrades `pip`, installs the pinned course packages, and runs an import check. It does not alter another Python installation.

### macOS or Linux

```bash
python3.14 scripts/setup_environment.py
```

### Windows PowerShell

```powershell
py -3.14 scripts\setup_environment.py
```

Re-running the command is safe: it reuses `.venv` and brings it back to the versions in `requirements.txt`.

## 4. Start JupyterLab

### macOS or Linux

```bash
.venv/bin/jupyter lab
```

### Windows PowerShell

```powershell
.venv\Scripts\jupyter.exe lab
```

Open the notebook under the week's `notebooks` folder. The current lessons are:

```text
week4/notebooks/week4_machine_learning_health_services.ipynb
week5/notebooks/week5_natural_language_processing_health_services.ipynb
```

Because JupyterLab is launched from `.venv`, its default Python kernel already contains the course packages.

## Verify or repair an installation

Run the verification helper at any time:

### macOS or Linux

```bash
.venv/bin/python scripts/verify_install.py
```

### Windows PowerShell

```powershell
.venv\Scripts\python.exe scripts\verify_install.py
```

If an installation becomes inconsistent, close JupyterLab, remove only the repository's `.venv` directory, and rerun the setup helper. Do not delete the repository or any `weekX/data` directory.

## How future weeks are supported

- `requirements.txt` is the pinned, shared environment for every published chapter.
- When a future notebook imports a new package, its exact tested version is added to the root requirements file.
- `scripts/generate_week_data.py` automatically runs each available `weekX/scripts/generate_synthetic_data.py` generator.
- `scripts/execute_notebooks.py` automatically discovers notebooks under every `weekX/notebooks` directory, so continuous integration expands as weeks are added.
- Every week's data remains under `weekX/data` with a README describing its source, license, checksum, or synthetic-generation method.

This single-environment policy keeps setup predictable for students. If a future module genuinely requires a conflicting or unusually large dependency, document that exception in the week's README rather than silently changing the course environment.

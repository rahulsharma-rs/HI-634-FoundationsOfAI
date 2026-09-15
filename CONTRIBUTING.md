# Contributing to HI 634 Teaching Code

## Weekly module standard

Every weekly module should include:

- a notebook that runs from a clean kernel;
- measurable learning objectives and an explicit lesson sequence;
- short instructor prompts and learner exercises;
- data stored only in the week's `data` directory;
- provenance and licensing notes for public data;
- an explicit synthetic-data notice for generated examples;
- a fixed random seed when randomness is used;
- plain-language limitations and prohibited uses; and
- no credentials, PHI, or identifiable student information.

## Notebook quality check

Before committing a notebook:

1. Restart the kernel and run all cells.
2. Confirm that every plot has a title and labeled axes.
3. Confirm that model metrics are interpreted in the context of a decision, prevalence, capacity, and error consequences.
4. Keep preprocessing and feature selection inside model-development pipelines.
5. Do not tune against the final test set.
6. Record package changes in `requirements.txt`.
7. Update the week's data README when a dataset changes.

## Environment and dependency changes

The root `requirements.txt` is the shared environment for all published weeks. Pin every direct dependency to the version used for browser and clean-kernel testing. After changing it, run:

```bash
python3.14 scripts/setup_environment.py
.venv/bin/python scripts/execute_notebooks.py
```

Do not create a separate weekly environment unless a documented dependency conflict makes the shared environment impossible. If that happens, explain the exception and exact setup commands in both `INSTALL.md` and the week's README.

## Commit scope

Keep commits focused on one week or one teaching improvement. Do not commit `.venv`, notebook checkpoints, operating-system metadata, secrets, or generated caches.

Before staging files, use `git status --short --ignored` to confirm that local environments, caches, credentials, and private-data directories remain ignored.

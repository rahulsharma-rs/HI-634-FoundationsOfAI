# Week 5: Natural Language Processing in Health Services

This module turns Chapter 5 into a reproducible, offline Jupyter lesson. It follows text from raw strings through tokenization and numerical representations, then demonstrates information extraction, classification, retrieval, summarization, evaluation, privacy, and accountable workflow design.

## Contents

- `data/`: locally generated synthetic text only, with provenance and integrity hashes.
- `notebooks/`: the executed classroom notebook.
- `scripts/`: deterministic data and notebook builders.
- `source_notes/`: source inventory and chapter-to-lesson teaching map.

## Rebuild and test

From the repository root:

```bash
.venv/bin/python week5/scripts/generate_synthetic_data.py
.venv/bin/python week5/scripts/build_week5_notebook.py
.venv/bin/python scripts/execute_notebooks.py
```

The notebook uses no network calls, pretrained language models, real clinical notes, or protected health information. Its scores describe authored synthetic examples only.

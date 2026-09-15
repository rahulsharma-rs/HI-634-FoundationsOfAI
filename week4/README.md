# Week 4 Machine Learning and Data Mining for Health Services

This module turns Chapter 4 into a guided, executable lesson. The notebook is designed for approximately 120 minutes of instruction plus optional extension exercises.

## Learning sequence

1. Frame a health-service decision before choosing an algorithm.
2. Match supervised, unsupervised, semi-supervised, and self-supervised paradigms to appropriate tasks.
3. Inspect the Breast Cancer Wisconsin Diagnostic teaching dataset and define the positive class.
4. Create train, validation, and untouched test partitions.
5. Build a leakage-safe preprocessing and logistic-regression pipeline.
6. Compare a prevalence baseline with the model using multiple metrics.
7. Examine discrimination, calibration, thresholds, and capacity-constrained ranking.
8. Compare model families with identical cross-validation folds.
9. Interpret coefficients and permutation importance without causal overclaiming.
10. Extend the ideas to regression, clustering, anomaly detection, Responsible AI, and Secure AI.

## Files

- `notebooks/week4_machine_learning_health_services.ipynb`: student-facing executable lesson with instructor prompts and exercises.
- `data/breast_cancer_wisconsin_diagnostic.csv`: public UCI teaching dataset.
- `data/synthetic_length_of_stay.csv`: deterministic, fictional operational regression data.
- `data/synthetic_claims.csv`: deterministic, fictional anomaly-triage data.
- `scripts/generate_synthetic_data.py`: recreates the synthetic datasets.
- `scripts/build_week4_notebook.py`: reproducibly builds the notebook.
- `source_notes/source_inventory.md`: provenance and source assessment.
- `source_notes/chapter4_teaching_map.md`: chapter-to-notebook alignment.

## Instructor preparation

Run the notebook once before class, then restart the kernel. During class, pause at the decision-framing, leakage, threshold, and interpretation checkpoints before showing the code output. The notebook includes suggested responses in collapsible instructor notes.

The WDBC results are intentionally framed as historical, internal, and educational. Strong scores do not establish clinical validity, current transportability, or regulatory readiness.

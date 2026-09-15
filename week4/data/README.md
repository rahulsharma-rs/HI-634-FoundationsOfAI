# Week 4 Data

All data used by the Week 4 notebook are stored in this directory.

## Breast Cancer Wisconsin Diagnostic

- File: `breast_cancer_wisconsin_diagnostic.csv`
- Source: UCI Machine Learning Repository, dataset ID 17
- Record page: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Direct data URL: https://archive.ics.uci.edu/static/public/17/data.csv
- DOI: https://doi.org/10.24432/C5DW2B
- Creators: William Wolberg, Olvi Mangasarian, Nick Street, and W. Street
- Retrieved: 2026-09-15
- License shown by UCI: Creative Commons Attribution 4.0 International
- Shape: 569 records, 30 real-valued predictors, one diagnosis target, and one identifier
- SHA-256: `3487b754ff02c93cdc8a18afafe223961d9cfbfdb932e57483e52581042767d7`

The measurements were computed from digitized images of fine-needle aspirates of breast masses. The dataset is historical and lacks the populations, sites, demographics, workflow context, and prospective evidence required for clinical use. It is included only to teach classification mechanics and limitations.

## Synthetic length of stay

- File: `synthetic_length_of_stay.csv`
- Generator: `../scripts/generate_synthetic_data.py`
- Seed: 42
- Shape: 1,500 fictional encounter records
- SHA-256: `abfc593eafbe0ddb72a26cccf405c137c17bdcd4a05f944f173ffd0560812079`
- Purpose: compare a median baseline, regularized regression, and a nonlinear ensemble while retaining error units in days

## Synthetic claims

- File: `synthetic_claims.csv`
- Generator: `../scripts/generate_synthetic_data.py`
- Seed: 42
- Shape: 1,000 fictional claim records
- SHA-256: `6313349098e805c5bc76ba45c8bf69254636ecb44bfaa049750b81f38b1fe220`
- Purpose: demonstrate anomaly-based review prioritization and the effect of a fixed review capacity

The two synthetic files contain **no real patient or claims data**. Their intentionally clear patterns make them useful for instruction but unrealistic as estimates of operational performance. The `simulation_group` field records how examples were generated; it is not a fraud label and is excluded from model fitting.

"""Generate deterministic, fictional Week 4 teaching datasets.

The outputs contain no real patient or claims information. They intentionally use
simple data-generating processes so students can connect model behavior to known
assumptions. Run from any working directory with the Week 4 environment.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build_length_of_stay(rng: np.random.Generator, n: int = 1_500) -> pd.DataFrame:
    """Return a fictional admission-time length-of-stay dataset."""
    age = np.clip(rng.normal(62, 15, n), 18, 95)
    comorbidity_count = rng.poisson(2.3, n)
    emergency_admission = rng.binomial(1, 0.38, n)
    lab_severity_index = rng.normal(0, 1, n)
    length_of_stay_days = (
        1.2
        + 0.018 * (age - 50)
        + 0.55 * comorbidity_count
        + 2.10 * emergency_admission
        + 0.70 * np.maximum(lab_severity_index, 0)
        + rng.normal(0, 1.3, n)
    ).clip(min=0.25)

    return pd.DataFrame(
        {
            "synthetic_record_id": [f"LOS-{i:04d}" for i in range(1, n + 1)],
            "age": age.round(1),
            "comorbidity_count": comorbidity_count,
            "emergency_admission": emergency_admission,
            "lab_severity_index": lab_severity_index.round(4),
            "length_of_stay_days": length_of_stay_days.round(3),
        }
    )


def build_claims(rng: np.random.Generator) -> pd.DataFrame:
    """Return fictional claims-like records for anomaly-triage instruction."""
    normal_n = 980
    unusual_n = 20

    normal = pd.DataFrame(
        {
            "allowed_amount": rng.lognormal(np.log(900), 0.65, normal_n),
            "service_count": rng.poisson(3, normal_n) + 1,
            "days_since_last_claim": rng.exponential(45, normal_n),
            "simulation_group": "common_pattern",
        }
    )
    unusual = pd.DataFrame(
        {
            "allowed_amount": rng.lognormal(np.log(12_000), 0.25, unusual_n),
            "service_count": rng.integers(15, 40, unusual_n),
            "days_since_last_claim": rng.uniform(0, 3, unusual_n),
            "simulation_group": "injected_unusual_pattern",
        }
    )

    claims = pd.concat([normal, unusual], ignore_index=True)
    claims.insert(0, "synthetic_claim_id", [f"CLM-{i:04d}" for i in range(1, len(claims) + 1)])
    claims["allowed_amount"] = claims["allowed_amount"].round(2)
    claims["days_since_last_claim"] = claims["days_since_last_claim"].round(2)
    return claims.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    los = build_length_of_stay(rng)
    claims = build_claims(rng)

    los_path = DATA_DIR / "synthetic_length_of_stay.csv"
    claims_path = DATA_DIR / "synthetic_claims.csv"
    los.to_csv(los_path, index=False)
    claims.to_csv(claims_path, index=False)

    print(f"Wrote {len(los):,} fictional records to {los_path}")
    print(f"Wrote {len(claims):,} fictional records to {claims_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate deterministic synthetic text datasets for the Week 5 NLP lesson."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


CARE_MANAGEMENT_NOTES = [
    ("N001", "Patient reports difficulty paying rent.", 1, "affirmed", "housing"),
    ("N002", "Housing currently stable.", 0, "negated_or_stable", "housing"),
    ("N003", "Denies food insecurity.", 0, "negated_or_stable", "food"),
    ("N004", "Needs a ride to dialysis appointments.", 1, "affirmed", "transportation"),
    ("N005", "No transportation problems reported.", 0, "negated_or_stable", "transportation"),
    ("N006", "Pt unable to afford prescribed medication.", 1, "affirmed", "medication_cost"),
    ("N007", "Patient reports running out of food.", 1, "affirmed", "food"),
    ("N008", "Housing situation is unclear; assessment requested.", 0, "uncertain", "housing"),
    ("N009", "Social history not documented.", 0, "not_documented", "none"),
    ("N010", "Mother cannot pay rent; patient has stable housing.", 0, "other_experiencer", "housing"),
    ("N011", "Previously homeless; now has stable housing.", 0, "historical_resolved", "housing"),
    ("N012", "Patient reports chest pain.", 0, "unrelated", "none"),
]


POSITIVE_TRAINING = [
    ("Cannot pay for groceries.", "food"),
    ("Food runs out before payday.", "food"),
    ("Patient requests help obtaining meals.", "food"),
    ("The pantry is empty this week.", "food"),
    ("Requests assistance with rent.", "housing"),
    ("Eviction is expected this week.", "housing"),
    ("Lost housing and needs shelter.", "housing"),
    ("Cannot keep up with apartment costs.", "housing"),
    ("Cannot afford the medication copay.", "medication_cost"),
    ("Stopped medication because of cost.", "medication_cost"),
    ("Needs help paying for prescriptions.", "medication_cost"),
    ("Prescription was not filled because it was too expensive.", "medication_cost"),
    ("Has no ride to the clinic.", "transportation"),
    ("Bus fare is unaffordable.", "transportation"),
    ("Needs transportation for the follow-up visit.", "transportation"),
    ("Missed dialysis because a ride was unavailable.", "transportation"),
]


NEGATIVE_TRAINING = [
    ("Food supply is adequate.", "food"),
    ("Denies problems paying for groceries.", "food"),
    ("Food access has not been assessed.", "food"),
    ("A sibling reports running out of food.", "food"),
    ("Lives in stable housing.", "housing"),
    ("Rent is paid without difficulty.", "housing"),
    ("Housing needs are uncertain.", "housing"),
    ("Past housing difficulty has resolved.", "housing"),
    ("Medication costs are affordable.", "medication_cost"),
    ("No financial concerns about prescriptions are reported.", "medication_cost"),
    ("Medication affordability was not discussed.", "medication_cost"),
    ("The caregiver cannot afford a prescription.", "medication_cost"),
    ("Transportation to the clinic is available.", "transportation"),
    ("Has reliable transport.", "transportation"),
    ("Transportation status is unknown.", "transportation"),
    ("Previously lacked a ride but transportation is arranged now.", "transportation"),
]


LANGUAGE_CHALLENGE = [
    ("C001", "Patient has no ride to dialysis.", 1, "affirmed", "en"),
    ("C002", "Patient has no transportation problems.", 0, "negated_or_stable", "en"),
    ("C003", "Housing risk cannot be excluded.", 0, "uncertain", "en"),
    ("C004", "Hx housing difficulty, resolved.", 0, "historical_resolved", "en"),
    ("C005", "Food insecurity: [ ] Yes [ ] No", 0, "not_documented", "en"),
    ("C006", "No tiene transporte para la cita.", 1, "affirmed", "es"),
    ("C007", "Sibling cannot afford medication; patient reports no cost concern.", 0, "other_experiencer", "en"),
    ("C008", "Patient says the food supply is adequate.", 0, "negated_or_stable", "en"),
    ("C009", "Eviction notice received; shelter is needed tomorrow.", 1, "affirmed", "en"),
    ("C010", "Bus fare is too expensive for the follow-up appointment.", 1, "affirmed", "en"),
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    note_rows = [
        {
            "note_id": note_id,
            "text": text,
            "reference_current_patient_need": reference,
            "documentation_status": status,
            "need_type": need_type,
        }
        for note_id, text, reference, status, need_type in CARE_MANAGEMENT_NOTES
    ]
    write_csv(
        DATA_DIR / "synthetic_care_management_notes.csv",
        ["note_id", "text", "reference_current_patient_need", "documentation_status", "need_type"],
        note_rows,
    )

    training_rows: list[dict[str, object]] = []
    for index, (text, need_type) in enumerate(POSITIVE_TRAINING, start=1):
        training_rows.append(
            {"training_id": f"P{index:03d}", "text": text, "label": 1, "need_type": need_type}
        )
    for index, (text, need_type) in enumerate(NEGATIVE_TRAINING, start=1):
        training_rows.append(
            {"training_id": f"G{index:03d}", "text": text, "label": 0, "need_type": need_type}
        )
    write_csv(
        DATA_DIR / "synthetic_social_needs_training.csv",
        ["training_id", "text", "label", "need_type"],
        training_rows,
    )

    challenge_rows = [
        {
            "case_id": case_id,
            "text": text,
            "reference_current_patient_need": reference,
            "documentation_status": status,
            "language": language,
        }
        for case_id, text, reference, status, language in LANGUAGE_CHALLENGE
    ]
    write_csv(
        DATA_DIR / "synthetic_language_challenge.csv",
        ["case_id", "text", "reference_current_patient_need", "documentation_status", "language"],
        challenge_rows,
    )

    for path in sorted(DATA_DIR.glob("*.csv")):
        print(f"{path.name}: {sha256(path)}")


if __name__ == "__main__":
    main()

# Week 5 data

Every CSV in this directory is deterministic, synthetic educational data created by `week5/scripts/generate_synthetic_data.py`. No row describes a real person, patient, encounter, message, or claim. The files are suitable for an open classroom repository but not for clinical validation or deployment claims.

## Files

| File | Rows | Purpose | SHA-256 |
|---|---:|---|---|
| `synthetic_care_management_notes.csv` | 12 | Progressive social-needs case with authored reference labels and documentary states | `f17edcadd01231334e84306d64439bca4f0d553e9b49fd629cad7a2a453d9a0c` |
| `synthetic_social_needs_training.csv` | 32 | Separately worded positive and negative examples for a small classical classifier | `73454d7c90a04aeb6ecd186f85dab7bd31b49f4af6aea8834dd039d571b1939b` |
| `synthetic_language_challenge.csv` | 10 | Negation, uncertainty, history, experiencer, template, and language-coverage stress cases | `9d0696afaca45b05445af0e4cb664ea77179b5db010b85d3d4e7c432d2636d8f` |

## Reference-label definition

`reference_current_patient_need = 1` means the sentence explicitly documents a current social need of the patient under the teaching annotation policy. A value of `0` is not synonymous with “the person has no need”; it includes stable or negated statements, uncertainty, missing documentation, historical resolution, another experiencer, and unrelated text. The accompanying status column preserves those differences.

## Regeneration

From the repository root:

```bash
.venv/bin/python week5/scripts/generate_synthetic_data.py
```

The generator prints each file's SHA-256 digest. A changed digest requires reviewing whether the teaching labels, expected outputs, and notebook explanations still agree.

## Public corpus discussed but not distributed

Chapter 5 discusses the UCI Drug Reviews (Druglib.com) dataset, ID 461, DOI `10.24432/C55G6J`. It is not downloaded or redistributed in this repository because the chapter identifies terms that require clarification. The required Week 5 notebook runs entirely offline with the synthetic files above.

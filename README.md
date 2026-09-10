# CPRI Hackathon Screening Solution

This package contains a reproducible program for:

1. identifying Invalid test records;
2. predicting `Reference_Parameter` for every test record; and
3. generating the required automated summary.

## Run with the supplied workbook

Place `CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx` beside
`cpri_solution.py`, then run:

```bash
python cpri_solution.py \
  --input-workbook CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx \
  --team-name Team_OogaBooga \
  --output-dir outputs
```

The program creates:

- `outputs/Team_OogaBooga.csv` - required predictions and labels
- `outputs/summary.json` - automated Task 03 summary
- `outputs/model_validation.json` - audit, validation metrics, and integrity checks

## Run with separate CSV files

```bash
python cpri_solution.py \
  --train-csv training_data.csv \
  --test-csv test_data.csv \
  --sample-csv sample_submission.csv \
  --team-name Team_OogaBooga \
  --output-dir outputs
```

`--sample-csv` is optional. Without it, the program preserves the order in
`test_data.csv`.

## Method overview

The validity method combines three independently interpretable checks: exact
duplicate measurements, missing primary sensors, and residual disagreement
between S1-S3 and their expected response under the recorded operating
conditions. S4 is excluded from validity decisions because the historical data
shows that it is auxiliary and can be missing in Valid records.

The reference model is trained only on historical Valid records. It uses a
physics-guided polynomial representation with explicit load-current and
high-ambient regimes. Missing or clearly faulty primary sensors are repaired
from operating-condition response models before reference prediction.

## Renaming the submission

Change only `--team-name` to create the final filename required by the event.
Do not edit individual prediction rows.

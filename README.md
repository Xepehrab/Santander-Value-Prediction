# Santander Value Prediction Challenge

My solution for the [Santander Value Prediction Challenge](https://www.kaggle.com/competitions/santander-value-prediction-challenge)
on Kaggle.

The goal is to predict an anonymized transaction `target` value for each
customer. The data is "wide" (more columns than rows) and mostly zeros, and
the competition is scored with **RMSLE** (Root Mean Squared Logarithmic Error).

## How it works

1. **Load** `train.csv` and `test.csv`.
2. **Use `log(1 + target)`** instead of the raw target, because the metric is
   RMSLE (error measured in log space).
3. **Clean the columns** — drop columns that are always the same value, and
   drop exact-duplicate columns.
4. **Add row features** — since the real column meanings are hidden, build
   simple per-row statistics (sum, mean, std, min, max, zero counts, etc.).
5. **Train LightGBM** with 5-fold cross-validation and early stopping.
6. **Save** `submission.csv` in the format Kaggle expects.

## Files

| File | What it does |
|------|--------------|
| `prepare.py` | Loads and cleans the data, builds the row features. |
| `main.py` | Trains the model and writes the submission file. |
| `requirements.txt` | The Python packages needed to run the project. |
| `train.csv` / `test.csv` | The competition data (download from Kaggle). |
| `sample_submission.csv` | Example of the submission format. |
| `submission.csv` | The file produced by `main.py` (this is what you upload). |

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

Make sure `train.csv`, `test.csv`, and `sample_submission.csv` are in the same
folder (download them from the competition's **Data** page on Kaggle).

## Run

```bash
python main.py
```

`main.py` automatically uses `prepare.py` to get the clean data, trains the
model, prints the cross-validation score, and creates `submission.csv`.

## Results

Latest run (`python main.py`):

### Data preparation

| Stage | Train | Test |
|-------|-------|------|
| Raw shape | (4459, 4993) | (49342, 4992) |
| After cleaning + row features | (4459, 4741) | (49342, 4741) |

- Removed **256** constant columns and **5** duplicate columns.
- Added 11 row-level features per sample (sum, mean, std, min, max, zero/non-zero counts, and stats over non-zero values only).

You may see harmless NumPy warnings during row-feature creation (`Mean of empty slice`, `All-NaN slice encountered`). These come from rows that are all zeros; `prepare.py` fills the resulting NaNs with 0.

### Cross-validation (5-fold, LightGBM)

| Fold | RMSLE | Trees |
|------|-------|-------|
| 1 | 1.33997 | 253 |
| 2 | 1.39150 | 370 |
| 3 | 1.35996 | 304 |
| 4 | 1.35176 | 414 |
| 5 | 1.32297 | 426 |

**Overall CV RMSLE: 1.35343**

Training uses early stopping (patience 100 rounds, max 5000 trees). Predictions are averaged across folds, converted back from log space with `expm1`, clipped to non-negative values, and written to `submission.csv`.

## Submission

Upload the generated `submission.csv` to the competition page on Kaggle.

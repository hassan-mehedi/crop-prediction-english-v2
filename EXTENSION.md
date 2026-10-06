# 2026 Thesis Reassessment

**Status:** Work in progress

This is a later follow-up to the collaborative 2021 BSc thesis. It is separate from, and was not part of, the original submitted thesis.

## Research Framing

This reassessment defines the task as supervised 22-class crop suitability classification. The inputs are soil and environmental measurements (N, P, K, temperature, humidity, pH, and rainfall), and the target is a crop label from 22 classes. A model output is interpreted as a recommended or suitable crop class for the provided measurements.

The dataset contains no crop-yield or productivity measurements, so it does not support yield prediction or claims about the “most productive crop.” This precise framing is deliberate: it improves methodological clarity for the 2026 reassessment while leaving the collaborative 2021 thesis as historical work.

## Planned scope

- Dataset audit (implemented)
- Clean separation of training, model selection, and final testing (split implemented)
- Stratified cross-validation (training-only folds implemented)
- Random Forest baseline plus 1–2 comparison models
- Accuracy, macro F1, weighted F1, and confusion matrix
- Reproducible seeds, dependencies, commands, and saved results

Explainability, modern boosting models, top-3 recommendations, deployment improvements, and regional validation are optional later work.

## Dataset audit

Run from the repository root with `python extension/audit.py`.

Generated files in `results/metrics/`:

- `dataset_summary.json`
- `class_distribution.csv`
- `feature_summary.csv`

## Evaluation split

Run `python extension/split.py` from the repository root. This creates
`results/metrics/evaluation_split.json` with an 80% training / 20% final-test
split, stratified by crop label, using fixed random seed `42`. For the current
dataset this gives 1,760 training rows and 440 final-test rows (80/20 per crop).
The file saves zero-based dataset row positions, class counts, the dataset
SHA-256 fingerprint, and five stratified cross-validation validation folds.
CV positions refer to the saved training-row order, not the original CSV.

Later model-selection code should call `load_training_data()` from
`extension.split` and pass its returned `cv` folds to `GridSearchCV` or
`cross_validate`, using `scoring="f1_macro"`. It returns only training features,
training labels, and training-only folds. All models must reuse these folds.
Learned preprocessing such as scaling must be inside a scikit-learn `Pipeline`
so it is fitted separately on each fold's training subset.

Existing split files are checked against the dataset and configuration on every
load. A mismatch raises an error without overwriting the saved holdout. Do not
delete or regenerate the split to improve results.

Keep final-test rows out of fitting, preprocessing fitting, tuning, and model
selection. Checking their size and class distribution during split preparation
is allowed. Freeze model configurations before final evaluation; do not revise
settings or select a winner using final-test scores. Training and final-test
evaluation are still pending in `extension/train.py` and `extension/evaluate.py`.

Verify split isolation and repeatability with
`python -m unittest extension.test_split -v`.
This requires pandas, NumPy, and scikit-learn in the active Python environment.
The split was verified with pandas 3.0.6 and scikit-learn 1.9.1. For this
checkout's local verification dependencies, first run the following in PowerShell:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.tmp-evaluation-deps'
python extension/split.py
python -m unittest extension.test_split -v
```

The local dependency folder is ignored by Git; other environments should install
the declared dependencies before running these commands.

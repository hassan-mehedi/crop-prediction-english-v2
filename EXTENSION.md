# 2026 Thesis Reassessment

**Status:** Random Forest baseline complete and frozen. Additional research is optional.

This is a later follow-up to the collaborative 2021 BSc thesis. It is separate from, and was not part of, the original submitted thesis.

## Research Framing

This reassessment defines the task as supervised 22-class crop suitability classification. The inputs are soil and environmental measurements (N, P, K, temperature, humidity, pH, and rainfall), and the target is a crop label from 22 classes. A model output is interpreted as a recommended or suitable crop class for the provided measurements.

The dataset contains no crop-yield or productivity measurements, so it does not support yield prediction or claims about the “most productive crop.” This precise framing is deliberate: it improves methodological clarity for the 2026 reassessment while leaving the collaborative 2021 thesis as historical work.

## Completed baseline scope

- Dataset audit and saved summary reports
- Fixed, stratified training / final-test split and training-only cross-validation folds
- Random Forest hyperparameter selection using macro F1
- One recorded final-test evaluation, with accuracy, macro F1, weighted F1, classification report, and confusion matrix
- Saved seeds, split, selection metadata, and baseline results

Model comparisons, explainability, modern boosting models, top-3 recommendations, deployment improvements, and regional validation are optional future work. They are not required to complete or report this baseline.

## The process in plain language

Each dataset row is one example. The seven measurements (`N`, `P`, `K`,
temperature, humidity, pH, and rainfall) are the information given to the model.
The `label` column is the crop name the model should learn to predict. This is
classification: choosing one crop category from 22 possible categories. It is
not yield prediction, because the dataset has no yield measurements.

The workflow has two separate checks so the final score stays meaningful:

1. **Set aside a final test group.** The split contains 1,760 training rows and
   440 final-test rows. The model does not use the 440 rows while learning or
   while its settings are being chosen. Think of them as questions saved for a
   final exam.
2. **Compare settings using training rows only.** `extension.train` tries 12
   Random Forest setting combinations. For each combination, five-fold
   cross-validation takes turns using parts of the training rows to learn and
   another part to check performance. This is like practice exams used to
   choose which study approach works best.
3. **Save the selected model.** The setting with the best average macro F1 score
   is selected, then the model is trained again using all 1,760 training rows.
   The original training workflow saved this fitted model under
   `results/models/` so the evaluation step could load it.
4. **Give the model the final test rows once.** The original
   `extension.evaluate` workflow used the saved model to predict crop labels for
   the 440 rows it had not seen. It compared those predictions with the known
   labels and saved the results. These final scores are for reporting; they
   must not be used to change the model settings.

The fitted model is excluded from Git and is absent from a clean checkout. The
original one-time training/evaluation scripts therefore cannot reproduce the
whole workflow in this checkout; the isolated reproduction command described
below rebuilds the already-selected estimator without changing the saved
evaluation files.

In short: **learn from training rows, compare settings with training-only
cross-validation, then do one final check on held-back rows.** Keeping the final
test rows separate avoids giving the model an unfair preview of its exam.

### Reading the scores

- **Accuracy** is the share of all test rows where the predicted crop matches
  the known crop.
- **F1** combines how often the model finds examples of a crop with how often
  its predictions for that crop are correct. **Macro F1** gives each crop equal
  importance; **weighted F1** gives more influence to crops with more test rows.
- **Confusion matrix** is a count table: rows show the actual crop and columns
  show the predicted crop. Counts on the diagonal are correct predictions;
  counts outside it show which crops were confused.

These scores describe this dataset and its saved split. They do not prove that
the model will perform equally well in other regions, seasons, or farming
conditions.

## Environment

The reassessment uses Python 3.14.4, NumPy 2.5.3, pandas 3.0.6, and
scikit-learn 1.9.1, matching the recorded experiment metadata. All direct and
resolved runtime packages are pinned in `requirements-extension.txt`. Keep
this environment separate from the legacy Streamlit app, whose historical
`model.pkl` requires Python 3.9 and scikit-learn 0.24.1; its pinned packages are
in `requirements.txt`.

From the repository root in PowerShell:

```powershell
uv python install 3.14.4
uv venv --python 3.14.4 .venv-extension
uv pip install --python .venv-extension\Scripts\python.exe -r requirements-extension.txt
```

## Dataset audit

Run from the repository root. Use an isolated output folder to keep the saved
baseline audit reports under `results/metrics/` unchanged:

```powershell
& .\.venv-extension\Scripts\python.exe extension\audit.py --output-dir results\reproductions\audit
```

The audit command writes these files to `results/reproductions/audit/`:

- `dataset_summary.json`
- `class_distribution.csv`
- `feature_summary.csv`

## Evaluation split

Run the saved-split check from the repository root:

```powershell
& .\.venv-extension\Scripts\python.exe -m extension.split
```

The split is stored in
`results/metrics/evaluation_split.json` with an 80% training / 20% final-test
split, stratified by crop label, using fixed random seed `42`. For the current
dataset this gives 1,760 training rows and 440 final-test rows (80/20 per crop).
The file saves zero-based dataset row positions, class counts, the dataset
SHA-256 fingerprint, and five stratified cross-validation validation folds.
Dataset line endings are canonicalized before fingerprinting so LF and CRLF
checkouts verify against the same saved split.
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
settings or select a winner using final-test scores. The Random Forest baseline
workflow and its one-time final evaluation are implemented below.

Verify split isolation, repeatability, and line-ending independence with:

```powershell
& .\.venv-extension\Scripts\python.exe -m unittest extension.test_split -v
```

The extension dependencies include pandas, NumPy, and scikit-learn.

## Random Forest baseline and saved results

The original one-time workflow selected model settings using only the saved
training rows and five stratified training-only folds:

```powershell
& .\.venv-extension\Scripts\python.exe -m extension.train
```

The search compares 12 small parameter combinations (100 or 250 trees, maximum
depth of unlimited, 10, or 20, and minimum leaf size of 1 or 2), selecting by
mean macro F1 across the saved folds. Each candidate uses random seed `42` and
one worker for repeatable execution. The best pipeline is refit on all 1,760
training rows. Random Forest does not need feature scaling, so its pipeline
contains the classifier without a learned preprocessing step.

Training outputs are `results/models/random_forest_baseline.joblib`,
`results/metrics/random_forest_selection.json`, and
`results/metrics/random_forest_cv_results.csv`. The selection JSON records the
chosen settings, cross-validation score, dataset fingerprint, split, and
software versions. The training script never loads final-test rows.

Before the final report existed, the saved pipeline was evaluated once with:

```powershell
& .\.venv-extension\Scripts\python.exe -m extension.evaluate
```

This evaluates the saved pipeline on the 440 final-test rows and writes accuracy,
macro F1, weighted F1, a per-class classification report, and a confusion matrix
under `results/metrics/`. The final JSON report prevents another evaluation or
retraining after test results exist. In this checkout the model artifact is
missing and the final report already exists, so `extension.train` intentionally
refuses to retrain and `extension.evaluate` cannot load its model. Do not remove
the guard or change settings based on the saved final scores.

### Reproduce from a clean checkout

Use the saved selection, dataset fingerprint, and split to rebuild the already
selected estimator and compare its scores with the saved report:

```powershell
& .\.venv-extension\Scripts\python.exe -m extension.reproduce
```

This command does not tune or select a model. It fits the 250-tree Random Forest
on the 1,760 training rows and predicts the same 440-row holdout. The model,
reproduced metrics, classification report, confusion matrix, and score
comparison are written under
`results/reproductions/frozen_random_forest/`, not `results/metrics/`. The
output directory must not already exist; pass `--output-dir` to use a different
new directory. This is a reproduction check against the already observed
holdout, not an independent evaluation or new test result.

Baseline run on the current dataset: the selected settings were 250 trees, no
maximum depth, and minimum leaf size 1. Mean training cross-validation macro F1
was 0.9943. The single final-test run on 440 rows produced accuracy 0.9955,
macro F1 0.9955, and weighted F1 0.9955; 438 of 440 predictions were correct.
The measured results are saved in `results/metrics/random_forest_final_test.json`,
with per-class details in `random_forest_classification_report.csv` and counts
in `random_forest_confusion_matrix.csv`. Do not use this final-test score to
change the settings or select another model.

## Methods, results, and limitations

### Methods

The 2026 reassessment treats crop recommendation as supervised classification
across 22 crop labels, using seven soil and environmental measurements. The
dataset audit checks required columns, missing and non-numeric values, infinite
values, exact duplicate rows, class balance, feature ranges, and selected
indicators of obvious target leakage. A fixed-seed (`42`) stratified 80:20 split
reserves 440 of 2,200 rows for final testing and uses 1,760 rows for training.
Random Forest settings are selected from 12 candidates with five-fold
cross-validation on training rows only. The selected pipeline is refitted on all
training rows and evaluated once on the held-back test rows.

The original 2021 training script performs randomized hyperparameter search
with cross-validation on the full dataset, then fits the selected model on that
same dataset. It does not reserve an independent final test set. This is a
limitation of the original evaluation design; the 2026 reassessment is a
separate follow-up and does not alter the historical thesis.

### Results

The audit report records 2,200 rows, 22 crop classes, and 100 examples per
class. It reports no missing values, exact duplicate rows, non-numeric or
infinite feature values, or conflicting labels for identical measurements.
These checks are an initial screen, not proof that all data issues are absent.

Training-only cross-validation selected 250 trees, unlimited maximum depth,
and minimum leaf size 1, with mean macro F1 of 0.9943. On the 440-row final-test
set, 438 predictions were correct. Accuracy, macro F1, and weighted F1 were
each approximately 0.9955. Detailed results are in the saved JSON and CSV
reports under `results/metrics/`.

### Limitations

The audit cannot establish that measurements are accurate, representative of
real farms, or free from every possible data-quality or leakage issue. Scores
describe this dataset and its saved split only; they do not establish
performance in other regions, seasons, or farming conditions. The dataset has
crop labels but no yield or productivity measurements, so this work evaluates
crop classification rather than yield prediction or productivity. This
baseline does not include comparison models, explainability analysis, or
external validation. The final-test result has been observed and must not be
used to tune settings or select another model.

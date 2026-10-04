# 2026 Thesis Reassessment

**Status:** Work in progress

This is a later follow-up to the collaborative 2021 BSc thesis. It is separate from, and was not part of, the original submitted thesis.

## Research Framing

This reassessment defines the task as supervised 22-class crop suitability classification. The inputs are soil and environmental measurements (N, P, K, temperature, humidity, pH, and rainfall), and the target is a crop label from 22 classes. A model output is interpreted as a recommended or suitable crop class for the provided measurements.

The dataset contains no crop-yield or productivity measurements, so it does not support yield prediction or claims about the “most productive crop.” This precise framing is deliberate: it improves methodological clarity for the 2026 reassessment while leaving the collaborative 2021 thesis as historical work.

## Planned scope

- Dataset audit (implemented)
- Clean separation of training, model selection, and final testing
- Stratified cross-validation
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

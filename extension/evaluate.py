"""Evaluate the frozen Random Forest baseline once on the final test set."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from extension.split import FEATURES, SPLIT_PATH, prepare_split
from extension.train import FINAL_REPORT_PATH, MODEL_PATH, SELECTION_PATH


ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
CLASSIFICATION_REPORT_PATH = METRICS_DIR / "random_forest_classification_report.csv"
CONFUSION_MATRIX_PATH = METRICS_DIR / "random_forest_confusion_matrix.csv"


def main():
    if FINAL_REPORT_PATH.exists():
        raise FileExistsError(
            f"Final-test evaluation already exists at {FINAL_REPORT_PATH.relative_to(ROOT)}. "
            "It is intentionally a one-time evaluation."
        )
    for path in (MODEL_PATH, SELECTION_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required training artifact is missing: {path.relative_to(ROOT)}. "
                "Run python -m extension.train first."
            )

    data, split = prepare_split()
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("final_test_used_for_selection") is not False:
        raise ValueError("Selection metadata does not confirm training-only tuning.")
    if selection.get("split_manifest") != str(SPLIT_PATH.relative_to(ROOT)):
        raise ValueError("The model was selected against a different split manifest.")
    if selection.get("dataset_sha256") != split["dataset_sha256"]:
        raise ValueError("The model was selected against a different dataset version.")

    model = joblib.load(MODEL_PATH)
    test_rows = split["final_test_rows"]
    X_test = data.iloc[test_rows][FEATURES]
    y_test = data.iloc[test_rows]["label"]
    predictions = model.predict(X_test)
    labels = list(model.classes_)

    metrics = {
        "model": "RandomForestClassifier",
        "evaluation": "single evaluation on untouched final test set",
        "test_rows": len(test_rows),
        "class_count": len(labels),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "training_cv_mean_macro_f1": selection["best_mean_cv_macro_f1"],
        "selected_parameters": selection["best_parameters"],
        "random_seed": selection["random_seed"],
        "split_dataset_sha256": split["dataset_sha256"],
        "final_test_used_for_model_selection": False,
    }
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    per_class = pd.DataFrame(report).transpose()
    matrix = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=labels,
        columns=labels,
    )
    matrix.index.name = "actual_crop"
    matrix.columns.name = "predicted_crop"

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(CLASSIFICATION_REPORT_PATH, index_label="class_or_average")
    matrix.to_csv(CONFUSION_MATRIX_PATH)
    # Write the JSON result last; its presence locks this final evaluation.
    FINAL_REPORT_PATH.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Final-test rows: {len(test_rows)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Saved final metrics: {FINAL_REPORT_PATH.relative_to(ROOT)}")
    print(f"Saved classification report: {CLASSIFICATION_REPORT_PATH.relative_to(ROOT)}")
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

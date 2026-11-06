"""Rebuild the frozen baseline into isolated outputs and compare its holdout score."""

import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from extension.split import (
    DATASET_PATH,
    FEATURES,
    SEED,
    SPLIT_PATH,
    load_training_data,
    prepare_split,
)


ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
SELECTION_PATH = METRICS_DIR / "random_forest_selection.json"
FINAL_REPORT_PATH = METRICS_DIR / "random_forest_final_test.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "reproductions" / "frozen_random_forest"
FROZEN_PARAMETERS = {
    "classifier__max_depth": None,
    "classifier__min_samples_leaf": 1,
    "classifier__n_estimators": 250,
}
METRIC_NAMES = ("accuracy", "macro_f1", "weighted_f1")


def reproduce(output_dir=DEFAULT_OUTPUT_DIR):
    """Refit saved settings on training rows and compare on the same frozen holdout.

    Outputs are kept separate from results/metrics. This is a reproduction of
    the recorded evaluation, not a new independent evaluation or model search.
    """
    output_dir = Path(output_dir).resolve()
    reproduction_root = (ROOT / "results" / "reproductions").resolve()
    metrics_dir = METRICS_DIR.resolve()
    if output_dir != reproduction_root and reproduction_root not in output_dir.parents:
        raise ValueError("Reproduction output must be inside results/reproductions.")
    if output_dir.exists():
        raise FileExistsError(f"Choose a new reproduction output directory: {output_dir}")
    if output_dir == metrics_dir or metrics_dir in output_dir.parents:
        raise ValueError("Reproduction output must not be written under results/metrics.")

    for required in (DATASET_PATH, SPLIT_PATH, SELECTION_PATH, FINAL_REPORT_PATH):
        if not required.is_file():
            raise FileNotFoundError(f"Required frozen reference is missing: {required}")

    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    reference = json.loads(FINAL_REPORT_PATH.read_text(encoding="utf-8"))
    data, split = prepare_split()

    if selection.get("final_test_used_for_selection") is not False:
        raise ValueError("Saved selection does not confirm training-only model selection.")
    if selection.get("dataset_sha256") != split["dataset_sha256"]:
        raise ValueError("Saved selection does not match the current dataset and split.")
    if selection.get("best_parameters") != FROZEN_PARAMETERS:
        raise ValueError("Saved selected parameters differ from the frozen baseline.")
    if reference.get("split_dataset_sha256") != split["dataset_sha256"]:
        raise ValueError("Saved final metrics do not match the current dataset and split.")

    X_train, y_train, _ = load_training_data()
    classifier = RandomForestClassifier(
        random_state=SEED,
        n_jobs=1,
        max_features="sqrt",
        n_estimators=FROZEN_PARAMETERS["classifier__n_estimators"],
        max_depth=FROZEN_PARAMETERS["classifier__max_depth"],
        min_samples_leaf=FROZEN_PARAMETERS["classifier__min_samples_leaf"],
    )
    model = Pipeline([("classifier", classifier)])
    model.fit(X_train, y_train)

    test_rows = split["final_test_rows"]
    X_test = data.iloc[test_rows][FEATURES]
    y_test = data.iloc[test_rows]["label"]
    predictions = model.predict(X_test)
    labels = list(model.classes_)
    reproduced_metrics = {
        "model": "RandomForestClassifier",
        "evaluation": (
            "reproduction on the same frozen holdout; not a new independent evaluation"
        ),
        "test_rows": len(test_rows),
        "class_count": len(labels),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "training_cv_mean_macro_f1": selection["best_mean_cv_macro_f1"],
        "selected_parameters": selection["best_parameters"],
        "random_seed": selection["random_seed"],
        "split_dataset_sha256": split["dataset_sha256"],
        "final_test_used_for_model_selection": False,
    }
    report = pd.DataFrame(
        classification_report(
            y_test, predictions, labels=labels, output_dict=True, zero_division=0
        )
    ).transpose()
    matrix = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=labels,
        columns=labels,
    )
    matrix.index.name = "actual_crop"
    matrix.columns.name = "predicted_crop"

    comparison = {
        "purpose": "reproduction check against the saved result; not model selection",
        "reference_metrics": str(FINAL_REPORT_PATH.relative_to(ROOT)),
        "reproduced_metrics": "reproduced_final_test.json",
        "absolute_tolerance": 1e-9,
        "metrics": {},
    }
    for name in METRIC_NAMES:
        expected = float(reference[name])
        actual = reproduced_metrics[name]
        comparison["metrics"][name] = {
            "reference": expected,
            "reproduced": actual,
            "absolute_difference": abs(expected - actual),
            "matches_within_tolerance": math.isclose(
                expected, actual, rel_tol=0.0, abs_tol=1e-9
            ),
        }
    comparison["all_metrics_match"] = all(
        item["matches_within_tolerance"] for item in comparison["metrics"].values()
    )

    # Fail without modifying existing outputs. The caller can choose a fresh path.
    output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, output_dir / "random_forest_baseline.joblib")
    (output_dir / "reproduced_final_test.json").write_text(
        json.dumps(reproduced_metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report.to_csv(output_dir / "classification_report.csv", index_label="class_or_average")
    matrix.to_csv(output_dir / "confusion_matrix.csv")
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(f"Training rows: {len(X_train)}; final holdout rows: {len(test_rows)}")
    print(f"Reproduced accuracy: {reproduced_metrics['accuracy']:.4f}")
    print(f"Reproduced macro F1: {reproduced_metrics['macro_f1']:.4f}")
    print(f"Reference metrics match: {comparison['all_metrics_match']}")
    print(f"Isolated reproduction outputs: {output_dir}")
    if not comparison["all_metrics_match"]:
        raise RuntimeError(
            "Reproduced scores differ from the frozen metrics; inspect comparison.json."
        )
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="new, isolated directory for the regenerated model and comparison outputs",
    )
    args = parser.parse_args()
    reproduce(args.output_dir)


if __name__ == "__main__":
    main()

"""Tune and fit the 2026 Random Forest baseline using training data only."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from extension.split import N_FOLDS, SEED, SPLIT_PATH, load_training_data


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "results" / "models"
METRICS_DIR = ROOT / "results" / "metrics"
MODEL_PATH = MODEL_DIR / "random_forest_baseline.joblib"
SELECTION_PATH = METRICS_DIR / "random_forest_selection.json"
CV_RESULTS_PATH = METRICS_DIR / "random_forest_cv_results.csv"
FINAL_REPORT_PATH = METRICS_DIR / "random_forest_final_test.json"

PARAM_GRID = {
    "classifier__n_estimators": [100, 250],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_leaf": [1, 2],
}


def main():
    if FINAL_REPORT_PATH.exists():
        raise FileExistsError(
            "Final-test evaluation already exists. The selected model is frozen; "
            "do not retune after viewing final-test results."
        )

    X_train, y_train, cv = load_training_data()
    split_manifest = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    pipeline = Pipeline(
        [
            (
                "classifier",
                RandomForestClassifier(
                    random_state=SEED,
                    n_jobs=1,
                    max_features="sqrt",
                ),
            )
        ]
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        refit=True,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    # GridSearchCV refits the selected pipeline on all training rows. The final
    # test set is never loaded or passed to this script.
    joblib.dump(search.best_estimator_, MODEL_PATH)
    cv_results = pd.DataFrame(
        {
            "rank": search.cv_results_["rank_test_score"],
            "mean_macro_f1": search.cv_results_["mean_test_score"],
            "std_macro_f1": search.cv_results_["std_test_score"],
            "parameters": [
                json.dumps(params, sort_keys=True)
                for params in search.cv_results_["params"]
            ],
        }
    ).sort_values("rank", kind="stable")
    cv_results.to_csv(CV_RESULTS_PATH, index=False)

    selection = {
        "model": "RandomForestClassifier",
        "selection_metric": "f1_macro",
        "best_mean_cv_macro_f1": float(search.best_score_),
        "best_parameters": search.best_params_,
        "parameter_grid": PARAM_GRID,
        "cv_folds": N_FOLDS,
        "random_seed": SEED,
        "training_rows": len(X_train),
        "final_test_used_for_selection": False,
        "split_manifest": str(SPLIT_PATH.relative_to(ROOT)),
        "dataset_sha256": split_manifest["dataset_sha256"],
        "artifacts": {
            "fitted_model": str(MODEL_PATH.relative_to(ROOT)),
            "cv_results": str(CV_RESULTS_PATH.relative_to(ROOT)),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
        },
    }
    SELECTION_PATH.write_text(
        json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Training rows: {len(X_train)}")
    print(f"Candidates: {len(search.cv_results_['params'])}")
    print(f"Best mean CV macro F1: {search.best_score_:.4f}")
    print(f"Selected parameters: {search.best_params_}")
    print(f"Saved selection: {SELECTION_PATH.relative_to(ROOT)}")
    print(f"Saved model: {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

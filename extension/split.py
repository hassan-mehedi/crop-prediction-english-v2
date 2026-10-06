"""Persist the final holdout and expose training-only cross-validation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "crop_prediction_model_one.csv"
SPLIT_PATH = ROOT / "results" / "metrics" / "evaluation_split.json"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5


def prepare_split(dataset_path=DATASET_PATH, split_path=SPLIT_PATH):
    """Create once, or verify the saved split without overwriting it."""
    dataset_path, split_path = Path(dataset_path), Path(split_path)
    raw = dataset_path.read_bytes()
    data = pd.read_csv(dataset_path)
    required = FEATURES + ["label"]
    if not set(required).issubset(data.columns):
        raise ValueError("Dataset must contain the seven features and label.")
    if data[required].isna().any().any():
        raise ValueError("Resolve missing features or labels before splitting.")
    numeric = data[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Features must be finite numeric values.")

    train_rows, test_rows = train_test_split(
        np.arange(len(data)), test_size=TEST_SIZE, random_state=SEED,
        shuffle=True, stratify=data["label"],
    )
    training = data.iloc[train_rows].reset_index(drop=True)
    if training["label"].value_counts().min() < N_FOLDS:
        raise ValueError("Every training class needs at least five rows for CV.")

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    validation_folds = [
        validation.tolist()
        for _, validation in cv.split(training[FEATURES], training["label"])
    ]
    manifest = {
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(data),
        "random_seed": SEED,
        "final_test_fraction": TEST_SIZE,
        "cv_folds": N_FOLDS,
        "model_selection_metric": "f1_macro",
        "index_convention": "Zero-based CSV data-row positions; CV positions refer to train_rows order.",
        "train_rows": train_rows.tolist(),
        "final_test_rows": test_rows.tolist(),
        "cv_validation_positions": validation_folds,
        "training_class_counts": training["label"].value_counts().sort_index().to_dict(),
        "final_test_class_counts": data.iloc[test_rows]["label"].value_counts().sort_index().to_dict(),
    }
    if split_path.exists():
        saved = json.loads(split_path.read_text(encoding="utf-8"))
        if saved != manifest:
            raise ValueError(
                "Saved split differs from the dataset or split configuration. "
                "Investigate the change; the existing final holdout was not overwritten."
            )
    else:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return data, manifest


def load_training_data(dataset_path=DATASET_PATH, split_path=SPLIT_PATH):
    """Return only training X/y and reusable (fit, validation) CV positions.

    Pass cv to GridSearchCV or cross_validate. Fit learned preprocessing in
    an sklearn Pipeline so each fold learns it from its own fit rows.
    """
    data, manifest = prepare_split(dataset_path, split_path)
    training = data.iloc[manifest["train_rows"]].reset_index(drop=True)
    positions = np.arange(len(training))
    cv = []
    for validation in manifest["cv_validation_positions"]:
        validation = np.asarray(validation, dtype=int)
        fit = positions[~np.isin(positions, validation)]
        cv.append((fit, validation))
    return training[FEATURES], training["label"], cv


if __name__ == "__main__":
    _, saved = prepare_split()
    print(f"Training rows: {len(saved['train_rows'])}")
    print(f"Final-test rows: {len(saved['final_test_rows'])}")
    print(f"Training-only CV: {N_FOLDS} stratified folds; seed {SEED}")
    print(f"Saved split: {SPLIT_PATH.relative_to(ROOT)}")

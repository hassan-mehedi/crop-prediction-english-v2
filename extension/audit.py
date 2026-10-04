"""Run dataset-quality checks for the 2026 thesis reassessment."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPOSITORY_ROOT / "crop_prediction_model_one.csv"
METRICS_DIR = REPOSITORY_ROOT / "results" / "metrics"
FEATURE_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
TARGET_COLUMN = "label"
NOTICEABLE_IMBALANCE_RATIO = 0.10


def json_safe(value):
    """Convert pandas and NumPy values to standard JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def feature_encodes_target(data, feature):
    """Return true only for a complete one-to-one feature/class mapping."""
    pairs = data[[feature, TARGET_COLUMN]].dropna()
    class_count = pairs[TARGET_COLUMN].nunique()
    if class_count < 2 or pairs[feature].nunique() != class_count:
        return False

    class_by_feature = pairs.groupby(feature, dropna=False)[TARGET_COLUMN].nunique()
    feature_by_class = pairs.groupby(TARGET_COLUMN, dropna=False)[feature].nunique()
    return bool((class_by_feature == 1).all() and (feature_by_class == 1).all())


def main():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at repository path: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    row_count, column_count = data.shape
    missing_by_column = data.isna().sum()
    missing_rows = int(data.isna().any(axis=1).sum())
    duplicate_rows = int(data.duplicated().sum())
    duplicate_percent = duplicate_rows / row_count * 100 if row_count else 0.0

    class_counts = data[TARGET_COLUMN].value_counts(dropna=True)
    class_count = int(class_counts.size)
    class_names = class_counts.index.tolist()
    class_total = int(class_counts.sum())
    class_distribution = pd.DataFrame(
        {
            "label": class_counts.index,
            "count": class_counts.values,
            "percentage": (
                class_counts.values / class_total * 100 if class_total else []
            ),
        }
    )

    if class_count:
        min_class_size = int(class_counts.min())
        max_class_size = int(class_counts.max())
        perfectly_balanced = min_class_size == max_class_size
        relative_spread = (
            (max_class_size - min_class_size) / min_class_size
            if min_class_size
            else math.inf
        )
        noticeably_imbalanced = (
            not perfectly_balanced
            and relative_spread >= NOTICEABLE_IMBALANCE_RATIO
        )
        balance_status = (
            "perfectly balanced"
            if perfectly_balanced
            else "noticeably imbalanced"
            if noticeably_imbalanced
            else "minor variation"
        )
    else:
        min_class_size = None
        max_class_size = None
        perfectly_balanced = False
        relative_spread = None
        noticeably_imbalanced = False
        balance_status = "unavailable (no non-missing target labels)"

    feature_summary_rows = []
    non_numeric_by_feature = {}
    infinite_by_feature = {}
    for feature in FEATURE_COLUMNS:
        numeric = pd.to_numeric(data[feature], errors="coerce")
        non_numeric = data[feature].notna() & numeric.isna()
        infinite = np.isinf(numeric)
        non_numeric_by_feature[feature] = int(non_numeric.sum())
        infinite_by_feature[feature] = int(infinite.sum())
        feature_summary_rows.append(
            {
                "feature": feature,
                "min": numeric.min(),
                "max": numeric.max(),
                "mean": numeric.mean(),
                "median": numeric.median(),
                "standard_deviation": numeric.std(),
                "non_numeric_values": int(non_numeric.sum()),
                "infinite_values": int(infinite.sum()),
            }
        )

    feature_summary = pd.DataFrame(feature_summary_rows)
    constant_columns = [
        str(column)
        for column in data.columns
        if data[column].nunique(dropna=False) <= 1
    ]

    feature_checks = []
    for feature in FEATURE_COLUMNS:
        feature_checks.append(
            {
                "feature": feature,
                "identical_to_target": bool(data[feature].equals(data[TARGET_COLUMN])),
                "directly_encodes_target": feature_encodes_target(data, feature),
            }
        )

    feature_group_sizes = data.groupby(
        FEATURE_COLUMNS, dropna=False, sort=False
    ).size()
    labels_per_feature_row = data.groupby(
        FEATURE_COLUMNS, dropna=False, sort=False
    )[TARGET_COLUMN].nunique(dropna=False)
    conflicting_groups = labels_per_feature_row[labels_per_feature_row > 1]
    conflicting_feature_rows = int(feature_group_sizes.loc[conflicting_groups.index].sum())
    conflicting_group_count = int(conflicting_groups.size)
    obvious_leakage = any(
        check["identical_to_target"] or check["directly_encodes_target"]
        for check in feature_checks
    )

    summary = {
        "dataset": {
            "path": str(DATASET_PATH.relative_to(REPOSITORY_ROOT)),
            "rows": row_count,
            "columns": column_count,
            "shape": [row_count, column_count],
            "column_names": data.columns.tolist(),
            "data_types": {column: str(dtype) for column, dtype in data.dtypes.items()},
            "first_rows": data.head().to_dict(orient="records"),
        },
        "target": {
            "column": TARGET_COLUMN,
            "unique_class_count": class_count,
            "class_names": class_names,
        },
        "missing_data": {
            "total_missing_values": int(missing_by_column.sum()),
            "missing_values_per_column": missing_by_column.to_dict(),
            "rows_containing_missing_values": missing_rows,
        },
        "duplicates": {
            "exact_duplicate_rows": duplicate_rows,
            "percentage_duplicate_rows": duplicate_percent,
            "sample_rows": data.loc[data.duplicated(keep=False)].head().to_dict(
                orient="records"
            ),
        },
        "class_balance": {
            "counts": class_distribution.to_dict(orient="records"),
            "minimum_class_size": min_class_size,
            "maximum_class_size": max_class_size,
            "perfectly_balanced": perfectly_balanced,
            "noticeably_imbalanced": noticeably_imbalanced,
            "relative_spread_threshold": NOTICEABLE_IMBALANCE_RATIO,
            "relative_spread": relative_spread,
            "status": balance_status,
        },
        "feature_ranges": feature_summary.to_dict(orient="records"),
        "sanity_checks": {
            "expected_feature_columns": FEATURE_COLUMNS,
            "missing_expected_feature_columns": [],
            "target_column_present": True,
            "constant_columns": constant_columns,
            "non_numeric_values_per_feature": non_numeric_by_feature,
            "infinite_values_per_feature": infinite_by_feature,
            "domain_thresholds_applied": False,
            "domain_threshold_note": (
                "No domain-specific physical thresholds were assumed; non-numeric "
                "and infinite feature values are reported directly."
            ),
        },
        "leakage_risk": {
            "feature_checks": feature_checks,
            "conflicting_duplicate_feature_groups": conflicting_group_count,
            "conflicting_duplicate_feature_rows": conflicting_feature_rows,
            "obvious_leakage_warning_detected": obvious_leakage,
            "direct_encoding_rule": (
                "A feature is flagged only when it has a complete one-to-one mapping "
                "with the non-missing target classes."
            ),
        },
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with (METRICS_DIR / "dataset_summary.json").open("w", encoding="utf-8") as output:
        json.dump(json_safe(summary), output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    class_distribution.to_csv(METRICS_DIR / "class_distribution.csv", index=False)
    feature_summary.to_csv(METRICS_DIR / "feature_summary.csv", index=False)

    print("Dataset audit summary")
    print(f"Rows: {row_count}")
    print(f"Columns: {column_count}")
    print(f"Crop classes: {class_count}")
    print(f"Rows containing missing values: {missing_rows}")
    print(f"Exact duplicate rows: {duplicate_rows}")
    print(f"Conflicting duplicate-feature rows: {conflicting_feature_rows}")
    print(f"Class distribution: {balance_status}")
    print(f"Obvious leakage warning detected: {'yes' if obvious_leakage else 'no'}")
    print(f"Audit outputs: {METRICS_DIR.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()

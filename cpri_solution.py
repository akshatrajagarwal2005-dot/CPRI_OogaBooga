#!/usr/bin/env python3
"""End-to-end CPRI screening-round solution.

The program reads either the supplied Excel workbook or separate training/test
CSV files, identifies invalid tests, predicts Reference_Parameter for every
test row, and writes the required submission and automated summary.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


RANDOM_SEED = 2409
ID_COLUMN = "Test_ID"
TARGET_COLUMN = "Reference_Parameter"
LABEL_COLUMN = "Validity_Label"

OPERATING_COLUMNS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
]
PRIMARY_SENSORS = ["Sensor_S1", "Sensor_S2", "Sensor_S3"]
AUXILIARY_SENSOR = "Sensor_S4"
INPUT_COLUMNS = OPERATING_COLUMNS + PRIMARY_SENSORS + [AUXILIARY_SENSOR]

# Breakpoints and model structures selected with grouped/out-of-fold validation.
SENSOR_TARGET_CONFIGS = [
    (64.00, 46.30),
    (64.00, 45.90),
    (63.95, 46.30),
    (64.00, 45.00),
    (63.95, 45.00),
]
OPERATING_TARGET_CONFIGS = [(64.00, 45.00)]
PRODUCTION_STRATEGY = "invalid_operating_only"
RIDGE_ALPHA = 300.0


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load train/test records and optional sample submission order."""
    if args.train_csv or args.test_csv:
        if not (args.train_csv and args.test_csv):
            raise ValueError("Provide both --train-csv and --test-csv.")
        train = pd.read_csv(args.train_csv)
        test = pd.read_csv(args.test_csv)
        sample = pd.read_csv(args.sample_csv) if args.sample_csv else None
    else:
        workbook_path = Path(args.input_workbook)
        if not workbook_path.exists():
            raise FileNotFoundError(
                f"Input workbook not found: {workbook_path}. "
                "Use --input-workbook or provide --train-csv and --test-csv."
            )
        train = pd.read_excel(workbook_path, sheet_name=args.training_sheet)
        test = pd.read_excel(workbook_path, sheet_name=args.test_sheet)
        try:
            sample = pd.read_excel(workbook_path, sheet_name=args.sample_sheet)
        except ValueError:
            sample = None

    _require_columns(train, [ID_COLUMN] + INPUT_COLUMNS + [TARGET_COLUMN, LABEL_COLUMN], "Training data")
    _require_columns(test, [ID_COLUMN] + INPUT_COLUMNS, "Test data")

    if train[ID_COLUMN].isna().any() or test[ID_COLUMN].isna().any():
        raise ValueError("Test_ID cannot be missing.")
    if train[ID_COLUMN].duplicated().any() or test[ID_COLUMN].duplicated().any():
        raise ValueError("Test_ID values must be unique within each dataset.")
    if set(train[ID_COLUMN]) & set(test[ID_COLUMN]):
        raise ValueError("Training and test Test_ID values overlap.")
    labels = set(train[LABEL_COLUMN].dropna().astype(str))
    if labels != {"Valid", "Invalid"}:
        raise ValueError(f"Expected Validity_Label values Valid/Invalid; found {sorted(labels)}")
    if train[TARGET_COLUMN].isna().any():
        raise ValueError("Training Reference_Parameter contains missing values.")
    if train[OPERATING_COLUMNS].isna().any().any() or test[OPERATING_COLUMNS].isna().any().any():
        raise ValueError("Operating-condition columns cannot be missing for this method.")

    return train.copy(), test.copy(), sample


def duplicate_mask(frame: pd.DataFrame) -> np.ndarray:
    """Flag repeated operating conditions, including conflicting sensor records."""
    return frame.duplicated(OPERATING_COLUMNS, keep=False).to_numpy()


def duplicate_groups(frame: pd.DataFrame) -> np.ndarray:
    """Keep every repeated operating condition in one validation fold."""
    hashes = pd.util.hash_pandas_object(frame[OPERATING_COLUMNS], index=False).astype(str)
    counts = hashes.map(hashes.value_counts()).to_numpy()
    return np.where(counts > 1, "duplicate_" + hashes, "row_" + frame.index.astype(str))


def fit_sensor_response_models(frame: pd.DataFrame, fit_mask: np.ndarray) -> dict[str, LinearRegression]:
    models: dict[str, LinearRegression] = {}
    for sensor in PRIMARY_SENSORS:
        usable = fit_mask & frame[sensor].notna().to_numpy()
        if usable.sum() < len(OPERATING_COLUMNS) + 2:
            raise ValueError(f"Insufficient valid observations to model {sensor}.")
        models[sensor] = LinearRegression().fit(frame.loc[usable, OPERATING_COLUMNS], frame.loc[usable, sensor])
    return models


def expected_sensor_values(frame: pd.DataFrame, models: dict[str, LinearRegression]) -> pd.DataFrame:
    return pd.DataFrame(
        {sensor: model.predict(frame[OPERATING_COLUMNS]) for sensor, model in models.items()},
        index=frame.index,
    )


def sensor_residuals(
    frame: pd.DataFrame, models: dict[str, LinearRegression]
) -> tuple[pd.DataFrame, np.ndarray]:
    expected = expected_sensor_values(frame, models)
    residual = (frame[PRIMARY_SENSORS] - expected).abs()
    values = residual.to_numpy(dtype=float)
    available = np.where(np.isnan(values), -np.inf, values)
    score = available.max(axis=1)
    score[~np.isfinite(score)] = np.inf
    return residual, score


def calibrate_sensor_threshold(
    invalid: np.ndarray, structural_invalid: np.ndarray, scores: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Choose a threshold from labelled data, using a clean gap when one exists."""
    valid_scores = scores[(invalid == 0) & ~structural_invalid & np.isfinite(scores)]
    fault_scores = scores[(invalid == 1) & ~structural_invalid & np.isfinite(scores)]
    if not len(valid_scores) or not len(fault_scores):
        raise ValueError("Training data must contain valid rows and non-structural sensor faults.")

    max_valid = float(valid_scores.max())
    min_fault = float(fault_scores.min())
    if max_valid < min_fault:
        threshold = (max_valid + min_fault) / 2.0
    else:
        finite = np.unique(scores[np.isfinite(scores) & ~structural_invalid])
        candidates = np.r_[finite[0] - 1e-9, (finite[:-1] + finite[1:]) / 2, finite[-1] + 1e-9]
        scored = []
        for candidate in candidates:
            prediction = structural_invalid | (scores > candidate)
            scored.append(
                (
                    balanced_accuracy_score(invalid, prediction),
                    accuracy_score(invalid, prediction),
                    -candidate,
                    candidate,
                )
            )
        threshold = float(max(scored)[-1])
    return threshold, {"maximum_valid_score": max_valid, "minimum_fault_score": min_fault}


def classify_records(
    frame: pd.DataFrame,
    sensor_models: dict[str, LinearRegression],
    threshold: float,
) -> dict[str, np.ndarray | pd.DataFrame]:
    duplicated = duplicate_mask(frame)
    primary_missing_count = frame[PRIMARY_SENSORS].isna().sum(axis=1).to_numpy()
    primary_missing = primary_missing_count > 0
    residuals, scores = sensor_residuals(frame, sensor_models)
    sensor_fault = scores > threshold
    invalid = duplicated | primary_missing | sensor_fault
    return {
        "invalid": invalid,
        "duplicated": duplicated,
        "primary_missing": primary_missing,
        "primary_missing_count": primary_missing_count,
        "sensor_fault": sensor_fault,
        "sensor_score": scores,
        "sensor_residuals": residuals,
    }


def cross_validate_validity(train: pd.DataFrame, folds: int = 10) -> dict[str, object]:
    y = train[LABEL_COLUMN].eq("Invalid").astype(int).to_numpy()
    structural = duplicate_mask(train) | train[PRIMARY_SENSORS].isna().any(axis=1).to_numpy()
    groups = duplicate_groups(train)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    prediction = np.zeros(len(train), dtype=bool)

    for train_index, validation_index in splitter.split(train, y, groups):
        fit_mask = np.zeros(len(train), dtype=bool)
        fit_mask[train_index] = y[train_index] == 0
        models = fit_sensor_response_models(train, fit_mask)
        _, train_scores = sensor_residuals(train.iloc[train_index], models)
        threshold, _ = calibrate_sensor_threshold(
            y[train_index], structural[train_index], train_scores
        )
        _, validation_scores = sensor_residuals(train.iloc[validation_index], models)
        prediction[validation_index] = structural[validation_index] | (validation_scores > threshold)

    return {
        "folds": folds,
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision_invalid": float(precision_score(y, prediction)),
        "recall_invalid": float(recall_score(y, prediction)),
        "f1_invalid": float(f1_score(y, prediction)),
        "matthews_correlation": float(matthews_corrcoef(y, prediction)),
        "confusion_matrix_true_valid_invalid": confusion_matrix(y, prediction).tolist(),
    }


def regression_features(
    frame: pd.DataFrame,
    current_threshold: float,
    ambient_hinge: float,
    include_sensors: bool,
) -> pd.DataFrame:
    """Physics-guided polynomial and regime features selected by CV."""
    current = frame["Load_Current_A"].to_numpy(dtype=float)
    ambient = frame["Ambient_Temperature_C"].to_numpy(dtype=float)
    current_regime = (current >= current_threshold).astype(float)
    ambient_regime = (ambient >= ambient_hinge).astype(float)

    result = pd.DataFrame(
        {
            "current": current,
            "ambient": ambient,
            "current_regime": current_regime,
            "current_hinge": current_regime * (current - current_threshold),
            "ambient_regime": ambient_regime,
            "ambient_hinge": ambient_regime * (ambient - ambient_hinge),
            "voltage": frame["Applied_Voltage_kV"].to_numpy(dtype=float),
            "duration": frame["Test_Duration_min"].to_numpy(dtype=float),
        },
        index=frame.index,
    )
    if include_sensors:
        result["sensor_s1"] = frame["Sensor_S1"].to_numpy(dtype=float)
        result["sensor_s2"] = frame["Sensor_S2"].to_numpy(dtype=float)
        result["sensor_s3"] = frame["Sensor_S3"].to_numpy(dtype=float)

    scaled_current = current / 100.0
    scaled_ambient = ambient / 40.0
    for power in range(2, 7):
        result[f"current_power_{power}"] = scaled_current**power
    for power in range(2, 5):
        result[f"ambient_power_{power}"] = scaled_ambient**power

    result["joint_regime"] = current_regime * ambient_regime
    result["current_regime_x_ambient"] = current_regime * ambient
    result["ambient_regime_x_current"] = ambient_regime * current
    return result


def fit_target_ensemble(
    valid_train: pd.DataFrame,
    configs: list[tuple[float, float]],
    include_sensors: bool,
    alpha: float = RIDGE_ALPHA,
) -> list[tuple[float, float, Ridge]]:
    models = []
    for current_threshold, ambient_hinge in configs:
        design = regression_features(valid_train, current_threshold, ambient_hinge, include_sensors)
        model = Ridge(alpha=alpha).fit(design, valid_train[TARGET_COLUMN])
        models.append((current_threshold, ambient_hinge, model))
    return models


def predict_target_ensemble(
    models: list[tuple[float, float, LinearRegression | Ridge]],
    frame: pd.DataFrame,
    include_sensors: bool,
) -> tuple[np.ndarray, np.ndarray]:
    individual = []
    for current_threshold, ambient_hinge, model in models:
        design = regression_features(frame, current_threshold, ambient_hinge, include_sensors)
        individual.append(model.predict(design))
    matrix = np.column_stack(individual)
    return matrix.mean(axis=1), matrix.std(axis=1)


def repair_primary_sensors(
    frame: pd.DataFrame,
    sensor_models: dict[str, LinearRegression],
    residuals: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Replace only missing/clearly faulty sensor values with operating-model estimates."""
    repaired = frame.copy()
    expected = expected_sensor_values(frame, sensor_models)
    replacement_count = np.zeros(len(frame), dtype=int)
    for sensor in PRIMARY_SENSORS:
        replace = frame[sensor].isna().to_numpy() | (residuals[sensor].to_numpy() > threshold)
        repaired.loc[replace, sensor] = expected.loc[replace, sensor]
        replacement_count += replace.astype(int)
    return repaired, replacement_count


def cross_validate_regression(train: pd.DataFrame, folds: int = 10) -> dict[str, object]:
    valid = train.loc[train[LABEL_COLUMN].eq("Valid")].reset_index(drop=True)
    y = valid[TARGET_COLUMN].to_numpy(dtype=float)
    groups = duplicate_groups(valid)
    splitter = GroupKFold(n_splits=folds)
    prediction = np.zeros(len(valid), dtype=float)

    for train_index, validation_index in splitter.split(valid, groups=groups):
        fold_train = valid.iloc[train_index]
        fold_validation = valid.iloc[validation_index]
        models = fit_target_ensemble(fold_train, SENSOR_TARGET_CONFIGS, include_sensors=True)
        prediction[validation_index], _ = predict_target_ensemble(
            models, fold_validation, include_sensors=True
        )

    errors = y - prediction
    return {
        "population": "historical Valid records only",
        "records": int(len(valid)),
        "folds": folds,
        "mae_celsius": float(mean_absolute_error(y, prediction)),
        "rmse_celsius": float(math.sqrt(mean_squared_error(y, prediction))),
        "r_squared": float(r2_score(y, prediction)),
        "maximum_absolute_error_celsius": float(np.max(np.abs(errors))),
        "mean_error_celsius": float(np.mean(errors)),
    }


def _regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Return a consistent regression report for a non-empty evaluation population."""
    errors = y - prediction
    return {
        "mae_celsius": float(mean_absolute_error(y, prediction)),
        "rmse_celsius": float(math.sqrt(mean_squared_error(y, prediction))),
        "r_squared": float(r2_score(y, prediction)),
        "maximum_absolute_error_celsius": float(np.max(np.abs(errors))),
        "mean_error_celsius": float(np.mean(errors)),
    }


def cross_validate_all_record_regression(train: pd.DataFrame, folds: int = 10) -> dict[str, object]:
    """Evaluate the deployed repair-and-predict path on every labelled record.

    The original report scored only historical Valid records.  The competition
    requires a reference prediction for Invalid records too, so each fold fits
    fault detection and target models on its training partition, repairs held-out
    faulty sensors, and scores the exact production prediction path on all rows.
    """
    invalid = train[LABEL_COLUMN].eq("Invalid").to_numpy()
    groups = duplicate_groups(train)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    sensor_prediction = np.zeros(len(train), dtype=float)
    operating_prediction = np.zeros(len(train), dtype=float)
    predicted_invalid = np.zeros(len(train), dtype=bool)

    for train_index, validation_index in splitter.split(train, invalid, groups):
        fold_train = train.iloc[train_index].reset_index(drop=True)
        fold_validation = train.iloc[validation_index].reset_index(drop=True)
        fold_invalid = fold_train[LABEL_COLUMN].eq("Invalid").to_numpy()
        valid_fit_mask = ~fold_invalid
        sensor_models = fit_sensor_response_models(fold_train, valid_fit_mask)
        structural = duplicate_mask(fold_train) | fold_train[PRIMARY_SENSORS].isna().any(axis=1).to_numpy()
        _, train_scores = sensor_residuals(fold_train, sensor_models)
        threshold, _ = calibrate_sensor_threshold(fold_invalid, structural, train_scores)

        validation_classification = classify_records(fold_validation, sensor_models, threshold)
        repaired_validation, _ = repair_primary_sensors(
            fold_validation,
            sensor_models,
            validation_classification["sensor_residuals"],
            threshold,
        )
        valid_train = fold_train.loc[valid_fit_mask].reset_index(drop=True)
        sensor_models_target = fit_target_ensemble(valid_train, SENSOR_TARGET_CONFIGS, include_sensors=True)
        operating_models_target = fit_target_ensemble(valid_train, OPERATING_TARGET_CONFIGS, include_sensors=False)
        fold_sensor_prediction, _ = predict_target_ensemble(sensor_models_target, repaired_validation, include_sensors=True)
        fold_operating_prediction, _ = predict_target_ensemble(
            operating_models_target, fold_validation, include_sensors=False
        )
        sensor_prediction[validation_index] = fold_sensor_prediction
        operating_prediction[validation_index] = fold_operating_prediction
        predicted_invalid[validation_index] = np.asarray(validation_classification["invalid"], dtype=bool)

    target = train[TARGET_COLUMN].to_numpy(dtype=float)
    valid_mask = ~invalid
    candidates = {
        "repaired_sensor_model": sensor_prediction,
        "invalid_operating_blend_25_percent": np.where(predicted_invalid, 0.75 * sensor_prediction + 0.25 * operating_prediction, sensor_prediction),
        "invalid_operating_blend_50_percent": np.where(predicted_invalid, 0.50 * sensor_prediction + 0.50 * operating_prediction, sensor_prediction),
        "invalid_operating_only": np.where(predicted_invalid, operating_prediction, sensor_prediction),
    }
    candidate_metrics = {
        name: {
            "all_records": _regression_metrics(target, values),
            "valid_records": _regression_metrics(target[valid_mask], values[valid_mask]),
            "invalid_records": _regression_metrics(target[invalid], values[invalid]),
        }
        for name, values in candidates.items()
    }
    chosen_name = PRODUCTION_STRATEGY
    prediction = candidates[chosen_name]
    return {
        "population": "all historical records; fault detection and sensor repair refit within each fold",
        "records": int(len(train)),
        "folds": folds,
        "selected_production_strategy": chosen_name,
        "candidate_metrics": candidate_metrics,
        "all_records": _regression_metrics(target, prediction),
        "valid_records": {"records": int(valid_mask.sum()), **_regression_metrics(target[valid_mask], prediction[valid_mask])},
        "invalid_records": {"records": int(invalid.sum()), **_regression_metrics(target[invalid], prediction[invalid])},
    }


def order_submission(
    result: pd.DataFrame, sample: pd.DataFrame | None
) -> pd.DataFrame:
    if sample is None:
        return result
    _require_columns(sample, [ID_COLUMN], "Sample submission")
    if sample[ID_COLUMN].duplicated().any():
        raise ValueError("Sample submission contains duplicate Test_ID values.")
    if set(sample[ID_COLUMN]) != set(result[ID_COLUMN]):
        raise ValueError("Sample submission Test_ID set does not match Test_Data.")
    ordered = sample[[ID_COLUMN]].merge(result, on=ID_COLUMN, how="left", validate="one_to_one")
    return ordered


def safe_team_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not safe:
        raise ValueError("Team name must contain at least one letter or number.")
    return safe


def build_outputs(args: argparse.Namespace) -> dict[str, Path]:
    train, test, sample = load_data(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    invalid_train = train[LABEL_COLUMN].eq("Invalid").astype(int).to_numpy()
    valid_fit_mask = invalid_train == 0
    sensor_models = fit_sensor_response_models(train, valid_fit_mask)
    train_duplicate = duplicate_mask(train)
    train_missing_primary = train[PRIMARY_SENSORS].isna().any(axis=1).to_numpy()
    train_structural = train_duplicate | train_missing_primary
    _, train_sensor_scores = sensor_residuals(train, sensor_models)
    sensor_threshold, threshold_gap = calibrate_sensor_threshold(
        invalid_train, train_structural, train_sensor_scores
    )

    validity_cv = cross_validate_validity(train)
    regression_cv = cross_validate_regression(train)
    all_record_regression_cv = cross_validate_all_record_regression(train)
    test_classification = classify_records(test, sensor_models, sensor_threshold)

    repaired_test, replaced_sensor_count = repair_primary_sensors(
        test,
        sensor_models,
        test_classification["sensor_residuals"],
        sensor_threshold,
    )

    valid_train = train.loc[valid_fit_mask].reset_index(drop=True)
    sensor_target_models = fit_target_ensemble(
        valid_train, SENSOR_TARGET_CONFIGS, include_sensors=True
    )
    operating_target_models = fit_target_ensemble(
        valid_train, OPERATING_TARGET_CONFIGS, include_sensors=False
    )
    sensor_prediction, ensemble_spread = predict_target_ensemble(
        sensor_target_models, repaired_test, include_sensors=True
    )
    operating_prediction, _ = predict_target_ensemble(
        operating_target_models, test, include_sensors=False
    )

    invalid_test = np.asarray(test_classification["invalid"], dtype=bool)
    prediction = sensor_prediction.copy()
    selected_strategy = PRODUCTION_STRATEGY
    if selected_strategy == "invalid_operating_only":
        prediction = np.where(invalid_test, operating_prediction, prediction)
    elif selected_strategy == "invalid_operating_blend_50_percent":
        prediction = np.where(invalid_test, 0.50 * prediction + 0.50 * operating_prediction, prediction)
    elif selected_strategy == "invalid_operating_blend_25_percent":
        prediction = np.where(invalid_test, 0.75 * prediction + 0.25 * operating_prediction, prediction)
    elif selected_strategy != "repaired_sensor_model":
        raise AssertionError(f"Unknown production strategy: {selected_strategy}")
    labels = np.where(invalid_test, "Invalid", "Valid")
    result = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].astype(str),
            "Predicted_Reference_Parameter": np.round(prediction, 4),
            LABEL_COLUMN: labels,
        }
    )
    result = order_submission(result, sample)

    finite_sensor_score = np.asarray(test_classification["sensor_score"], dtype=float)
    finite_sensor_score = np.nan_to_num(
        finite_sensor_score,
        nan=sensor_threshold,
        posinf=sensor_threshold * 4,
        neginf=0.0,
    )
    attention_score = (
        invalid_test.astype(float) * 100.0
        + np.minimum(finite_sensor_score / sensor_threshold, 20.0) * 10.0
        + np.asarray(test_classification["primary_missing_count"], dtype=float) * 15.0
        + np.asarray(test_classification["duplicated"], dtype=float) * 5.0
        + replaced_sensor_count.astype(float) * 3.0
        + np.abs(sensor_prediction - operating_prediction) / max(regression_cv["rmse_celsius"], 1e-9)
        + ensemble_spread / max(regression_cv["rmse_celsius"], 1e-9)
    )
    attention_order = np.lexsort((test[ID_COLUMN].astype(str).to_numpy(), -attention_score))
    top_attention = test.iloc[attention_order[:3]][ID_COLUMN].astype(str).tolist()

    explanation = (
        "Historical labels were audited for missing primary sensors, repeated operating conditions "
        "and sensor inconsistency. Linear response models learned expected S1-S3 behaviour from "
        "Valid tests, separating true operating regimes from sensor faults. Reference predictions "
        "use a regime-aware polynomial ensemble trained only on Valid records, with explicit load "
        "and high-ambient transitions. Valid rows use repaired sensor predictions; Invalid rows use "
        "the operating-only fallback. Condition-grouped validation, deterministic settings and automated "
        "integrity checks make the workflow reproducible."
    )
    if len(explanation.split()) > 100:
        raise AssertionError("Automated explanation exceeds 100 words.")

    predicted_values = result["Predicted_Reference_Parameter"].astype(float)
    summary = {
        "records_analyzed": int(len(result)),
        "abnormal_invalid_records_identified": int((result[LABEL_COLUMN] == "Invalid").sum()),
        "predicted_reference_parameter": {
            "minimum": round(float(predicted_values.min()), 4),
            "maximum": round(float(predicted_values.max()), 4),
            "average": round(float(predicted_values.mean()), 4),
        },
        "three_test_ids_requiring_highest_attention": top_attention,
        "approach_explanation": explanation,
    }

    diagnostics = {
        "data_audit": {
            "training_records": int(len(train)),
            "test_records": int(len(test)),
            "training_valid": int((train[LABEL_COLUMN] == "Valid").sum()),
            "training_invalid": int((train[LABEL_COLUMN] == "Invalid").sum()),
            "training_repeated_condition_rows": int(train_duplicate.sum()),
            "test_repeated_condition_rows": int(np.asarray(test_classification["duplicated"]).sum()),
            "test_primary_sensor_missing_rows": int(
                np.asarray(test_classification["primary_missing"]).sum()
            ),
            "test_sensor_fault_only_rows": int(
                (
                    np.asarray(test_classification["sensor_fault"])
                    & ~np.asarray(test_classification["primary_missing"])
                    & ~np.asarray(test_classification["duplicated"])
                ).sum()
            ),
            "test_auxiliary_sensor_missing_rows": int(test[AUXILIARY_SENSOR].isna().sum()),
            "test_repaired_primary_sensor_values": int(replaced_sensor_count.sum()),
        },
        "validity_model": {
            "sensor_residual_threshold": float(sensor_threshold),
            "training_threshold_gap": threshold_gap,
            "cross_validation": validity_cv,
        },
        "reference_model": {
            "production_strategy": PRODUCTION_STRATEGY,
            "current_regime_threshold_amperes": 64.0,
            "regularization_alpha": RIDGE_ALPHA,
            "sensor_model_configurations": [
                {"current_threshold": x, "ambient_hinge": y}
                for x, y in SENSOR_TARGET_CONFIGS
            ],
            "cross_validation": regression_cv,
            "all_record_cross_validation": all_record_regression_cv,
        },
        "integrity_checks": {
            "submission_rows_match_test_rows": bool(len(result) == len(test)),
            "submission_ids_unique": bool(result[ID_COLUMN].is_unique),
            "submission_ids_match_test": bool(set(result[ID_COLUMN]) == set(test[ID_COLUMN])),
            "predictions_all_finite": bool(np.isfinite(predicted_values).all()),
            "labels_all_valid_values": bool(set(result[LABEL_COLUMN]) <= {"Valid", "Invalid"}),
            "summary_explanation_word_count": len(explanation.split()),
        },
    }

    if not all(diagnostics["integrity_checks"].values()):
        raise AssertionError(f"Final integrity checks failed: {diagnostics['integrity_checks']}")

    team_file = output_dir / f"{safe_team_name(args.team_name)}.csv"
    summary_file = output_dir / "summary.json"
    diagnostics_file = output_dir / "model_validation.json"
    result.to_csv(team_file, index=False, lineterminator="\n")
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    diagnostics_file.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "submission": team_file,
        "summary": summary_file,
        "diagnostics": diagnostics_file,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-workbook",
        default="CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx",
        help="Workbook containing Training_Data, Test_Data and Sample_Submission sheets.",
    )
    parser.add_argument("--training-sheet", default="Training_Data")
    parser.add_argument("--test-sheet", default="Test_Data")
    parser.add_argument("--sample-sheet", default="Sample_Submission")
    parser.add_argument("--train-csv", help="Optional training CSV; use together with --test-csv.")
    parser.add_argument("--test-csv", help="Optional test CSV; use together with --train-csv.")
    parser.add_argument("--sample-csv", help="Optional sample-submission CSV for row ordering.")
    parser.add_argument("--team-name", default="OogaBooga")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_outputs(args)
    for name, path in outputs.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()

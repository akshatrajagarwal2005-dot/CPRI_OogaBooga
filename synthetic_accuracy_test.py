#!/usr/bin/env python3
"""Run the CPRI pipeline against temporary synthetic data with known truth."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


SEED = 2409
OPERATING_COLUMNS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
]
PRIMARY_SENSORS = ["Sensor_S1", "Sensor_S2", "Sensor_S3"]
INPUT_COLUMNS = OPERATING_COLUMNS + PRIMARY_SENSORS + ["Sensor_S4"]


def make_clean_data(rng: np.random.Generator, rows: int, prefix: str) -> tuple[pd.DataFrame, np.ndarray]:
    voltage = rng.uniform(110.0, 250.0, rows)
    current = rng.uniform(25.0, 95.0, rows)
    ambient = rng.uniform(20.0, 60.0, rows)
    duration = rng.uniform(20.0, 180.0, rows)

    sensor_s1 = 0.20 * voltage + 0.16 * current + 0.35 * ambient + 0.04 * duration
    sensor_s2 = -0.10 * voltage + 0.18 * current + 0.20 * ambient + 0.05 * duration
    sensor_s3 = 0.08 * voltage + 0.14 * current + 0.28 * ambient - 0.02 * duration
    sensor_s1 += rng.normal(0.0, 0.25, rows)
    sensor_s2 += rng.normal(0.0, 0.25, rows)
    sensor_s3 += rng.normal(0.0, 0.25, rows)

    frame = pd.DataFrame(
        {
            "Test_ID": [f"{prefix}-{index:04d}" for index in range(rows)],
            "Applied_Voltage_kV": voltage,
            "Load_Current_A": current,
            "Ambient_Temperature_C": ambient,
            "Test_Duration_min": duration,
            "Sensor_S1": sensor_s1,
            "Sensor_S2": sensor_s2,
            "Sensor_S3": sensor_s3,
            "Sensor_S4": (
                0.06 * voltage + 0.11 * current + 0.09 * ambient + rng.normal(0.0, 0.08, rows)
            ),
        }
    )
    scaled_current = current / 100.0
    reference = (
        18.0
        + 0.13 * current
        + 0.21 * ambient
        + 0.025 * voltage
        + 0.012 * duration
        + 2.2 * scaled_current**2
        - 1.3 * scaled_current**3
        + 0.05 * sensor_s1
        + 0.03 * sensor_s2
        - 0.02 * sensor_s3
    )
    reference += rng.normal(0.0, 0.20, rows)
    return frame, reference


def inject_faults(
    rng: np.random.Generator,
    frame: pd.DataFrame,
    reference: np.ndarray,
    fault_rows: range,
    missing_rows: range,
    duplicate_rows: range,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    result = frame.copy()
    truth = reference.copy()
    labels = np.full(len(result), "Valid", dtype=object)
    fault_type = np.full(len(result), "clean", dtype=object)

    for offset, row in enumerate(fault_rows):
        sensor = PRIMARY_SENSORS[offset % len(PRIMARY_SENSORS)]
        result.loc[row, sensor] += rng.choice([-1.0, 1.0]) * rng.uniform(4.5, 8.0)
        labels[row] = "Invalid"
        fault_type[row] = "sensor_fault"

    for offset, row in enumerate(missing_rows):
        sensor = PRIMARY_SENSORS[offset % len(PRIMARY_SENSORS)]
        result.loc[row, sensor] = np.nan
        labels[row] = "Invalid"
        fault_type[row] = "primary_missing"

    duplicate_indices = list(duplicate_rows)
    if len(duplicate_indices) % 2:
        raise ValueError("Duplicate rows must contain an even number of indices.")
    for first, second in zip(duplicate_indices[::2], duplicate_indices[1::2]):
        result.loc[second, INPUT_COLUMNS] = result.loc[first, INPUT_COLUMNS].to_numpy()
        truth[second] = truth[first]
        labels[[first, second]] = "Invalid"
        fault_type[[first, second]] = "exact_duplicate"

    return result, truth, labels, fault_type


def build_synthetic_inputs(
    directory: Path,
) -> tuple[Path, Path, Path, Path, pd.DataFrame]:
    rng = np.random.default_rng(SEED)

    train, train_reference = make_clean_data(rng, 800, "TRN")
    train, train_reference, train_labels, _ = inject_faults(
        rng,
        train,
        train_reference,
        fault_rows=range(0, 50),
        missing_rows=range(50, 70),
        duplicate_rows=range(70, 110),
    )
    train["Reference_Parameter"] = train_reference
    train["Validity_Label"] = train_labels

    test, test_reference = make_clean_data(rng, 300, "TST")
    test, test_reference, test_labels, fault_types = inject_faults(
        rng,
        test,
        test_reference,
        fault_rows=range(0, 45),
        missing_rows=range(45, 65),
        duplicate_rows=range(65, 105),
    )
    test.loc[105:119, "Sensor_S4"] = np.nan

    truth = pd.DataFrame(
        {
            "Test_ID": test["Test_ID"],
            "True_Reference_Parameter": test_reference,
            "True_Validity_Label": test_labels,
            "Fault_Type": fault_types,
        }
    )
    sample = pd.DataFrame({"Test_ID": rng.permutation(test["Test_ID"].to_numpy())})

    train_path = directory / "training_data.csv"
    test_path = directory / "test_data.csv"
    sample_path = directory / "sample_submission.csv"
    workbook_path = directory / "synthetic_data.xlsx"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample.to_csv(sample_path, index=False)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        train.to_excel(writer, sheet_name="Training_Data", index=False)
        test.to_excel(writer, sheet_name="Test_Data", index=False)
        sample.to_excel(writer, sheet_name="Sample_Submission", index=False)
    return train_path, test_path, sample_path, workbook_path, truth


def score_predictions(predictions: pd.DataFrame, truth: pd.DataFrame) -> dict[str, object]:
    scored = truth.merge(predictions, on="Test_ID", how="left", validate="one_to_one")
    if scored.isna().any().any():
        raise AssertionError("Pipeline output is missing one or more synthetic test records.")

    y_true = scored["True_Validity_Label"]
    y_pred = scored["Validity_Label"]
    target_true = scored["True_Reference_Parameter"].to_numpy()
    target_pred = scored["Predicted_Reference_Parameter"].to_numpy()
    valid_mask = y_true.eq("Valid").to_numpy()

    classification = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_invalid": float(precision_score(y_true, y_pred, pos_label="Invalid")),
        "recall_invalid": float(recall_score(y_true, y_pred, pos_label="Invalid")),
        "f1_invalid": float(f1_score(y_true, y_pred, pos_label="Invalid")),
        "confusion_matrix_valid_invalid": confusion_matrix(
            y_true, y_pred, labels=["Valid", "Invalid"]
        ).tolist(),
    }
    regression = {
        "mae_all": float(mean_absolute_error(target_true, target_pred)),
        "rmse_all": float(np.sqrt(mean_squared_error(target_true, target_pred))),
        "r_squared_all": float(r2_score(target_true, target_pred)),
        "maximum_absolute_error_all": float(np.max(np.abs(target_true - target_pred))),
        "mae_true_valid": float(mean_absolute_error(target_true[valid_mask], target_pred[valid_mask])),
        "rmse_true_valid": float(
            np.sqrt(mean_squared_error(target_true[valid_mask], target_pred[valid_mask]))
        ),
    }
    by_fault_type = {}
    for fault_type, group in scored.groupby("Fault_Type"):
        by_fault_type[fault_type] = {
            "records": int(len(group)),
            "classification_accuracy": float(
                accuracy_score(group["True_Validity_Label"], group["Validity_Label"])
            ),
            "target_mae": float(
                mean_absolute_error(
                    group["True_Reference_Parameter"], group["Predicted_Reference_Parameter"]
                )
            ),
        }
    return {
        "classification": classification,
        "regression": regression,
        "by_fault_type": by_fault_type,
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    solution = project_dir / "cpri_solution.py"
    with tempfile.TemporaryDirectory(prefix="cpri-synthetic-") as temporary:
        temporary_dir = Path(temporary)
        train_path, test_path, sample_path, workbook_path, truth = build_synthetic_inputs(
            temporary_dir
        )
        csv_output_dir = temporary_dir / "csv_outputs"
        csv_run = subprocess.run(
            [
                sys.executable,
                str(solution),
                "--train-csv",
                str(train_path),
                "--test-csv",
                str(test_path),
                "--sample-csv",
                str(sample_path),
                "--team-name",
                "Synthetic_Evaluation",
                "--output-dir",
                str(csv_output_dir),
            ],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        workbook_output_dir = temporary_dir / "workbook_outputs"
        workbook_run = subprocess.run(
            [
                sys.executable,
                str(solution),
                "--input-workbook",
                str(workbook_path),
                "--team-name",
                "Synthetic_Evaluation",
                "--output-dir",
                str(workbook_output_dir),
            ],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )

        csv_predictions = pd.read_csv(csv_output_dir / "Synthetic_Evaluation.csv")
        workbook_predictions = pd.read_csv(workbook_output_dir / "Synthetic_Evaluation.csv")
        pd.testing.assert_frame_equal(csv_predictions, workbook_predictions)
        diagnostics = json.loads((csv_output_dir / "model_validation.json").read_text())
        scores = score_predictions(csv_predictions, truth)
        report = {
            "seed": SEED,
            "input_paths_tested": ["separate_csv_files", "excel_workbook"],
            "csv_and_workbook_predictions_identical": True,
            "synthetic_records": {
                "training": 800,
                "test": 300,
                "test_valid": int((truth["True_Validity_Label"] == "Valid").sum()),
                "test_invalid": int((truth["True_Validity_Label"] == "Invalid").sum()),
            },
            **scores,
            "pipeline_internal_cross_validation": {
                "validity": diagnostics["validity_model"]["cross_validation"],
                "reference": diagnostics["reference_model"]["cross_validation"],
            },
            "pipeline_stdout": {
                "csv": csv_run.stdout.strip().splitlines(),
                "workbook": workbook_run.stdout.strip().splitlines(),
            },
        }

    print(json.dumps(report, indent=2))
    if report["classification"]["balanced_accuracy"] < 0.98:
        raise SystemExit("Synthetic validity test failed: balanced accuracy is below 0.98")
    if report["regression"]["rmse_all"] > 0.35:
        raise SystemExit("Synthetic regression test failed: RMSE is above 0.35")


if __name__ == "__main__":
    main()

"""Train stage-specific classifiers, calibrate on validation data, and test once."""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from app.config import FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST
from app.services.features import STAGE_FEATURES

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "orders_processed.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
THRESHOLDS = np.linspace(0.05, 0.95, 19)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("train_model")


def chronological_split(frame: pd.DataFrame):
    frame = frame.sort_values("order_purchase_timestamp").reset_index(drop=True)
    if len(frame) < 100:
        raise ValueError(f"At least 100 eligible orders are required; found {len(frame)}.")
    train_end = int(len(frame) * 0.70)
    validation_end = int(len(frame) * 0.85)
    train, validation, test = frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:]
    if not (
        train["order_purchase_timestamp"].max() < validation["order_purchase_timestamp"].min()
        and validation["order_purchase_timestamp"].max() < test["order_purchase_timestamp"].min()
    ):
        raise ValueError("Chronological boundaries overlap; a distinct-time-period split could not be formed.")
    return train, validation, test


def make_pipeline(features, model_name):
    categorical = [name for name in features if name in ("customer_state", "payment_type")]
    numerical = [name for name in features if name not in categorical]
    transformers = [
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ]
    estimator = (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        if model_name == "Logistic Regression"
        else RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )
    return Pipeline([("preprocessor", ColumnTransformer(transformers)), ("classifier", estimator)])


def _metrics(labels, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return {
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(len(labels)),
    }


def _cost(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    false_positives = int(((predicted == 1) & (labels == 0)).sum())
    false_negatives = int(((predicted == 0) & (labels == 1)).sum())
    return {
        "threshold": float(threshold),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_cost": float(false_positives * FALSE_POSITIVE_COST + false_negatives * FALSE_NEGATIVE_COST),
    }


def train_stage(frame, stage):
    features = STAGE_FEATURES[stage]
    eligible = frame.loc[frame[features].notna().all(axis=1)].copy()
    if eligible["late_delivery"].nunique() != 2:
        raise ValueError(f"Stage {stage} needs both target classes among eligible orders.")
    train, validation, test = chronological_split(eligible)
    x_train, y_train = train[features], train["late_delivery"]
    x_validation, y_validation = validation[features], validation["late_delivery"]
    x_test, y_test = test[features], test["late_delivery"]

    candidates = []
    comparisons = []
    for model_name in ("Logistic Regression", "Random Forest"):
        model = make_pipeline(features, model_name)
        model.fit(x_train, y_train)
        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        score = float(average_precision_score(y_validation, validation_probabilities))
        candidates.append((score, model_name, model))
        comparisons.append({
            "stage": stage,
            "model": model_name,
            "validation_pr_auc": score,
            **_metrics(y_validation, validation_probabilities, 0.5),
        })
    _, selected_name, selected = max(candidates, key=lambda candidate: candidate[0])

    calibrated = CalibratedClassifierCV(FrozenEstimator(selected), method="sigmoid")
    calibrated.fit(x_validation, y_validation)
    validation_probabilities = calibrated.predict_proba(x_validation)[:, 1]
    cost_rows = [_cost(y_validation.to_numpy(), validation_probabilities, threshold) for threshold in THRESHOLDS]
    selected_threshold = min(cost_rows, key=lambda row: (row["total_cost"], -row["threshold"]))["threshold"]

    test_probabilities = calibrated.predict_proba(x_test)[:, 1]
    test_metrics = _metrics(y_test.to_numpy(), test_probabilities, selected_threshold)
    test_metrics.update({
        "stage": stage,
        "stage_name": {"A": "Order placement", "B": "Payment approval", "C": "Carrier handover"}[stage],
        "selected_model": selected_name,
        "threshold": selected_threshold,
        "validation_selected_cost": next(row for row in cost_rows if row["threshold"] == selected_threshold),
        "test_cost_at_selected_threshold": _cost(y_test.to_numpy(), test_probabilities, selected_threshold),
        "train_period": {
            "start": train["order_purchase_timestamp"].min().isoformat(),
            "end": train["order_purchase_timestamp"].max().isoformat(),
        },
        "validation_period": {
            "start": validation["order_purchase_timestamp"].min().isoformat(),
            "end": validation["order_purchase_timestamp"].max().isoformat(),
        },
        "test_period": {
            "start": test["order_purchase_timestamp"].min().isoformat(),
            "end": test["order_purchase_timestamp"].max().isoformat(),
        },
        "train_count": int(len(train)),
        "validation_count": int(len(validation)),
        "test_count": int(len(test)),
        "features": features,
    })
    observed_fraction, mean_predicted_probability = calibration_curve(
        y_test, test_probabilities, n_bins=10, strategy="quantile"
    )
    calibration_data = {
        "stage": stage,
        "observed_fraction": observed_fraction.tolist(),
        "mean_predicted_probability": mean_predicted_probability.tolist(),
        "binning": "10 quantile bins on the final test set, for evaluation only.",
    }
    calibrated_costs = [_cost(y_test.to_numpy(), test_probabilities, threshold) for threshold in THRESHOLDS]
    importance = permutation_importance(
        calibrated, x_validation, y_validation, scoring="average_precision",
        n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1,
    )
    importance_data = [
        {"feature": feature, "importance_mean": float(mean), "importance_std": float(std)}
        for feature, mean, std in sorted(
            zip(features, importance.importances_mean, importance.importances_std),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    predictions = pd.DataFrame({
        "order_id": test["order_id"].astype(str).to_numpy(),
        "stage": stage,
        "actual_late_delivery": y_test.to_numpy(),
        "delay_probability": test_probabilities,
        "predicted_late_delivery": (test_probabilities >= selected_threshold).astype(int),
        "threshold": selected_threshold,
    })
    bundle = {
        "stage": stage,
        "selected_model": selected_name,
        "features": features,
        "model": selected,
        "calibrated_model": calibrated,
        "threshold": selected_threshold,
    }
    return bundle, comparisons, test_metrics, calibration_data, cost_rows, calibrated_costs, importance_data, predictions


def main() -> int:
    if not DATA_PATH.is_file():
        LOGGER.error("Prepared dataset missing. Run scripts/inspect_dataset.py and scripts/prepare_dataset.py first.")
        return 1
    frame = pd.read_csv(DATA_PATH, parse_dates=["order_purchase_timestamp"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_comparisons, stage_results, calibration_results, cost_results, importance_results, predictions = [], [], [], [], [], []
    try:
        for stage in ("A", "B", "C"):
            bundle, comparisons, metrics, calibration, validation_costs, test_costs, importance, test_predictions = train_stage(frame, stage)
            joblib.dump(bundle, MODEL_DIR / f"stage_{stage}.joblib")
            all_comparisons.extend(comparisons)
            stage_results.append(metrics)
            calibration_results.append(calibration)
            cost_results.append({
                "stage": stage,
                "false_positive_cost": FALSE_POSITIVE_COST,
                "false_negative_cost": FALSE_NEGATIVE_COST,
                "validation_thresholds": validation_costs,
                "thresholds": test_costs,
            })
            importance_results.append({"stage": stage, "method": "Permutation importance, validation-period PR-AUC; importance is not causal.", "features": importance})
            predictions.append(test_predictions)
            LOGGER.info("Stage %s (%s) tested on %d orders; PR-AUC=%.4f, Brier=%.4f",
                        stage, bundle["selected_model"], metrics["test_count"], metrics["pr_auc"], metrics["brier_score"])
    except (ValueError, KeyError, ImportError) as exc:
        LOGGER.error("Training failed: %s", exc)
        return 1

    (REPORT_DIR / "model_comparison.json").write_text(json.dumps(all_comparisons, indent=2), encoding="utf-8")
    (REPORT_DIR / "stage_metrics.json").write_text(json.dumps(stage_results, indent=2), encoding="utf-8")
    (REPORT_DIR / "calibration.json").write_text(json.dumps(calibration_results, indent=2), encoding="utf-8")
    (REPORT_DIR / "cost_analysis.json").write_text(json.dumps(cost_results, indent=2), encoding="utf-8")
    (REPORT_DIR / "feature_importance.json").write_text(json.dumps(importance_results, indent=2), encoding="utf-8")
    pd.concat(predictions, ignore_index=True).to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    LOGGER.info("Training and final evaluation succeeded; reports saved to %s", REPORT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

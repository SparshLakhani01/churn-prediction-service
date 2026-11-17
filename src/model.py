"""Model training, calibration, evaluation, and local run logging.

Methodology (all splits stratified, seed 42):
  1. Split synthetic data into train / validation / test (60/20/20).
  2. Train two candidates on train: HistGradientBoosting and a
     logistic-regression baseline. Select by ROC-AUC on validation.
  3. Calibrate the winner with isotonic calibration via 3-fold
     cross-fitting on the training set, so calibration never touches
     validation or test data.
  4. Report final metrics once, on test.

Run logging is MLflow-style but local-only: each run appends one JSON
record under runs/ with parameters, metrics, and data provenance. No
MLflow server, no network calls.
"""

import json
import os

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def build_candidates(random_state=42):
    """Return {name: unfitted Pipeline} for the two model candidates."""
    return {
        "histgb": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.08,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "logreg": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, random_state=random_state),
                ),
            ]
        ),
    }


def calibration_table(y_true, y_prob, n_bins=10):
    """Return per-bin (mean predicted, fraction positive, count)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_low": round(float(lo), 2),
                "bin_high": round(float(hi), 2),
                "mean_predicted": round(float(y_prob[mask].mean()), 4),
                "fraction_positive": round(float(y_true[mask].mean()), 4),
                "count": int(mask.sum()),
            }
        )
    return rows


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE: count-weighted mean |mean predicted - fraction positive|."""
    table = calibration_table(y_true, y_prob, n_bins=n_bins)
    n = len(y_true)
    return float(
        sum(row["count"] / n * abs(row["mean_predicted"] - row["fraction_positive"])
            for row in table)
    )


def evaluate(y_true, y_prob):
    """Return the headline metric dict for one set of predictions."""
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "average_precision": round(
            float(average_precision_score(y_true, y_prob)), 4
        ),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "ece_10bin": round(expected_calibration_error(y_true, y_prob), 4),
        "n": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
    }


def calibrate_with_cv(model, X_train, y_train, method="isotonic", cv=3):
    """Fit cross-fitted calibration on the training set.

    CalibratedClassifierCV refits the base estimator on each fold and
    calibrates on the held-out fold, then averages the calibrated
    probabilities at predict time. Nothing from validation or test is
    used here. (sklearn 1.9 removed cv="prefit"; cross-fitting is the
    supported replacement.)
    """
    calibrator = CalibratedClassifierCV(model, method=method, cv=cv)
    calibrator.fit(X_train, y_train)
    return calibrator


def log_run(runs_dir, record):
    """Append one JSON run record; returns the record path."""
    os.makedirs(runs_dir, exist_ok=True)
    existing = [f for f in os.listdir(runs_dir) if f.startswith("run_")]
    run_id = f"run_{len(existing) + 1:03d}"
    record = dict(record)
    record["run_id"] = run_id
    path = os.path.join(runs_dir, f"{run_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return path

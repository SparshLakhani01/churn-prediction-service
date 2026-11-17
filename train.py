#!/usr/bin/env python3
"""Train the churn model on synthetic data, calibrate, evaluate, save.

Pipeline: generate synthetic customers -> stratified 60/20/20 split
-> train HistGB + logistic-regression candidates -> select by
validation ROC-AUC -> isotonic calibration via 3-fold cross-fitting
on the training set -> final evaluation on test (touched once) ->
save model artifact, metrics JSON, and a local run log.

Usage:
    python train.py [--n 5000] [--seed 42] [--quiet]
"""

import argparse
import json
import os
import sys

import joblib
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_gen import generate_customers
from src.features import ALL_FEATURES, TARGET
from src.model import (
    build_candidates,
    calibrate_with_cv,
    calibration_table,
    evaluate,
    log_run,
)

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument("--n", type=int, default=5000,
                        help="synthetic customers to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    df = generate_customers(n_customers=args.n, seed=args.seed)
    X, y = df[ALL_FEATURES], df[TARGET]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=args.seed
    )

    candidates = build_candidates(random_state=args.seed)
    val_scores = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        val_scores[name] = evaluate(
            y_val, model.predict_proba(X_val)[:, 1]
        )["roc_auc"]
    best_name = max(val_scores, key=val_scores.get)
    best_model = candidates[best_name]

    test_uncal = evaluate(y_test, best_model.predict_proba(X_test)[:, 1])
    calibrated = calibrate_with_cv(candidates[best_name], X_train, y_train)
    test_prob = calibrated.predict_proba(X_test)[:, 1]
    test_cal = evaluate(y_test, test_prob)

    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
    artifact = os.path.join(BASE, "models", "churn_model.joblib")
    joblib.dump(
        {"model": calibrated, "features": ALL_FEATURES, "seed": args.seed},
        artifact,
    )

    metrics = {
        "synthetic_data": True,
        "n_customers": args.n,
        "seed": args.seed,
        "splits": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "churn_rate_overall": round(float(y.mean()), 4),
        "validation_roc_auc": val_scores,
        "selected_model": best_name,
        "test_uncalibrated": test_uncal,
        "test_calibrated": test_cal,
        "calibration_table_test": calibration_table(y_test, test_prob),
    }
    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
    with open(os.path.join(BASE, "results", "metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    run_path = log_run(os.path.join(BASE, "runs"), metrics)

    if not args.quiet:
        print(f"Synthetic customers: {args.n} (seed {args.seed}), "
              f"churn rate {metrics['churn_rate_overall']:.3f}")
        print(f"Validation ROC-AUC: {val_scores} -> selected {best_name}")
        print(f"Test ROC-AUC (calibrated): {test_cal['roc_auc']}")
        print(f"Test average precision: {test_cal['average_precision']}")
        print(f"Test Brier: uncalibrated {test_uncal['brier_score']}, "
              f"calibrated {test_cal['brier_score']}")
        print(f"Test ECE (10-bin): {test_cal['ece_10bin']}")
        print(f"Model artifact: {artifact}")
        print(f"Run log: {run_path}")


if __name__ == "__main__":
    main()

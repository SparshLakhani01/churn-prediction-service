"""Unit tests for the churn prediction service (stdlib unittest)."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_gen import generate_customers
from src.features import ALL_FEATURES, build_preprocessor
from src.model import (
    build_candidates,
    calibration_table,
    evaluate,
    expected_calibration_error,
    log_run,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDataGen(unittest.TestCase):
    def test_shape_and_columns(self):
        df = generate_customers(n_customers=200, seed=7)
        self.assertEqual(len(df), 200)
        for col in ALL_FEATURES + ["churn", "customer_id"]:
            self.assertIn(col, df.columns)
        self.assertTrue(df["churn"].isin([0, 1]).all())

    def test_deterministic(self):
        a = generate_customers(n_customers=100, seed=3)
        b = generate_customers(n_customers=100, seed=3)
        pd.testing.assert_frame_equal(a, b)

    def test_churn_rate_reasonable(self):
        df = generate_customers(n_customers=2000, seed=42)
        rate = df["churn"].mean()
        self.assertGreater(rate, 0.05)
        self.assertLess(rate, 0.60)

    def test_customer_ids_marked_synthetic(self):
        df = generate_customers(n_customers=10, seed=1)
        self.assertTrue(df["customer_id"].str.startswith("SYN-").all())


class TestPreprocessing(unittest.TestCase):
    def test_handles_unseen_categories(self):
        df = generate_customers(n_customers=100, seed=5)
        prep = build_preprocessor()
        out = prep.fit_transform(df[ALL_FEATURES])
        self.assertEqual(out.shape[0], 100)
        row = df[ALL_FEATURES].iloc[[0]].copy()
        row["contract_type"] = "never-seen-contract"
        out2 = prep.transform(row)
        self.assertEqual(out2.shape[1], out.shape[1])

    def test_no_nan_after_transform(self):
        df = generate_customers(n_customers=100, seed=5)
        df.loc[0, "monthly_charges"] = np.nan
        prep = build_preprocessor()
        out = prep.fit_transform(df[ALL_FEATURES])
        self.assertFalse(np.isnan(np.asarray(out)).any())


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = generate_customers(n_customers=600, seed=11)
        cls.X, cls.y = df[ALL_FEATURES], df["churn"]

    def test_candidates_beat_chance(self):
        for name, model in build_candidates(random_state=11).items():
            model.fit(self.X, self.y)
            prob = model.predict_proba(self.X)[:, 1]
            metrics = evaluate(self.y, prob)
            self.assertGreater(metrics["roc_auc"], 0.5, name)
            self.assertGreaterEqual(metrics["brier_score"], 0.0)
            self.assertLessEqual(metrics["brier_score"], 1.0)

    def test_calibration_table_sums_to_n(self):
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, size=200)
        p = np.clip(rng.normal(0.3, 0.2, size=200), 0.01, 0.99)
        table = calibration_table(y, p)
        self.assertEqual(sum(r["count"] for r in table), 200)
        ece = expected_calibration_error(y, p)
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_log_run_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = log_run(tmp, {"metric": 1.0})
            with open(path, encoding="utf-8") as fh:
                record = json.load(fh)
            self.assertEqual(record["metric"], 1.0)
            self.assertIn("run_id", record)


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Train a small artifact in a temp location is complex; instead
        # test the API against the real artifact if train.py has run.
        # demo.sh always trains before testing, so this is reliable
        # in the demo flow. Skip gracefully otherwise.
        artifact = os.path.join(BASE, "models", "churn_model.joblib")
        if not os.path.exists(artifact):
            raise unittest.SkipTest("model artifact not present; run train.py")
        from fastapi.testclient import TestClient

        import api as api_module

        cls.client = TestClient(api_module.app)

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_predict_returns_probability(self):
        payload = {
            "tenure_months": 12,
            "monthly_charges": 80.0,
            "total_charges": 960.0,
            "contract_type": "month-to-month",
            "internet_service": "fiber",
            "payment_method": "electronic-check",
            "senior_citizen": 0,
            "dependents": 0,
            "paperless_billing": 1,
            "num_support_calls": 3,
            "avg_monthly_gb": 120.0,
            "late_payments": 1,
        }
        resp = self.client.post("/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertGreaterEqual(body["churn_probability"], 0.0)
        self.assertLessEqual(body["churn_probability"], 1.0)
        self.assertIsInstance(body["churn_prediction"], bool)

    def test_predict_rejects_bad_input(self):
        resp = self.client.post("/predict", json={"tenure_months": -5})
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()

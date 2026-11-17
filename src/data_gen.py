"""Synthetic telecom-style customer data generator.

Every row is invented by this module with a fixed seed. There are no
real customers, no real billing records, and no real churn outcomes.
The churn label is drawn from a logistic function of the features so
the data has learnable signal with realistic noise. The generator is
deterministic for a given (n_customers, seed).
"""

import numpy as np
import pandas as pd

CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
INTERNET_SERVICES = ["fiber", "dsl", "none"]
PAYMENT_METHODS = [
    "electronic-check", "mailed-check", "bank-transfer", "credit-card",
]


def generate_customers(n_customers=5000, seed=42):
    """Return a DataFrame of synthetic customers with a churn label."""
    rng = np.random.default_rng(seed)

    tenure = np.clip(
        rng.gamma(shape=2.0, scale=11.0, size=n_customers), 1, 72
    ).round().astype(int)
    senior = rng.binomial(1, 0.15, size=n_customers)
    dependents = rng.binomial(1, 0.30, size=n_customers)
    paperless = rng.binomial(1, 0.60, size=n_customers)

    internet = rng.choice(
        INTERNET_SERVICES, size=n_customers, p=[0.45, 0.35, 0.20]
    )
    # Longer-tenure customers skew toward longer contracts.
    contract = []
    for t in tenure:
        if t < 6:
            p = [0.80, 0.15, 0.05]
        elif t < 24:
            p = [0.55, 0.30, 0.15]
        else:
            p = [0.30, 0.35, 0.35]
        contract.append(rng.choice(CONTRACT_TYPES, p=p))
    contract = np.array(contract)
    payment = rng.choice(PAYMENT_METHODS, size=n_customers)

    base_charge = np.where(
        internet == "fiber", 85.0, np.where(internet == "dsl", 60.0, 30.0)
    )
    monthly_charges = np.clip(
        base_charge + rng.normal(0, 12, size=n_customers), 18, 125
    ).round(2)
    total_charges = (
        tenure * monthly_charges * rng.uniform(0.9, 1.1, size=n_customers)
    ).round(2)
    support_calls = np.clip(
        rng.poisson(1.2, size=n_customers), 0, 8
    )
    late_payments = np.clip(rng.poisson(0.5, size=n_customers), 0, 6)
    avg_gb = np.clip(
        rng.gamma(shape=3.0, scale=25.0, size=n_customers), 0, 400
    ).round(1)
    avg_gb = np.where(internet == "none", 0.0, avg_gb)

    # Logistic churn model: month-to-month, high charges, support
    # calls, and late payments raise churn; tenure and long contracts
    # lower it. Coefficients are invented for this synthetic data.
    logit = (
        -2.1
        + 0.9 * (contract == "month-to-month")
        - 0.55 * (contract == "two-year")
        - 0.028 * tenure
        + 0.014 * monthly_charges
        + 0.45 * support_calls
        + 0.40 * late_payments
        + 0.25 * (internet == "fiber")
        + 0.20 * paperless
        + rng.normal(0, 0.8, size=n_customers)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(size=n_customers) < prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": [f"SYN-{i:05d}" for i in range(n_customers)],
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "senior_citizen": senior,
            "dependents": dependents,
            "paperless_billing": paperless,
            "num_support_calls": support_calls,
            "avg_monthly_gb": avg_gb,
            "late_payments": late_payments,
            "churn": churn,
        }
    )

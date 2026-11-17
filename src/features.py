"""Feature preprocessing: one ColumnTransformer shared by train and serve.

Numeric features are standardized; categorical features are one-hot
encoded with unknown categories ignored, so the API never crashes on
a category value unseen in training. The transformer is fitted only
on training data, inside the sklearn Pipeline, so no information
leaks from validation or test splits.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "avg_monthly_gb",
    "late_payments",
    "senior_citizen",
    "dependents",
    "paperless_billing",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"


def build_preprocessor():
    """Return an unfitted ColumnTransformer for the churn features."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )

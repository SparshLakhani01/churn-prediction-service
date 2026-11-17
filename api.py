"""FastAPI prediction service for the churn model.

Loads the model artifact written by train.py (models/churn_model.joblib).
If the artifact is missing, the service fails fast with a clear error
instead of serving stale or dummy predictions. Run train.py first,
or use the Dockerfile, which trains before serving.

Endpoints:
  GET  /health     liveness + model-loaded status
  GET  /model_info model name, features, seed, and decision threshold
  POST /predict    churn probability for one synthetic-style customer
"""

import os

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(BASE, "models", "churn_model.joblib")
THRESHOLD = 0.5

app = FastAPI(title="Churn Prediction Service (synthetic demo)")


class CustomerFeatures(BaseModel):
    """One customer, same fields the model was trained on."""

    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=500)
    total_charges: float = Field(ge=0)
    contract_type: str
    internet_service: str
    payment_method: str
    senior_citizen: int = Field(ge=0, le=1)
    dependents: int = Field(ge=0, le=1)
    paperless_billing: int = Field(ge=0, le=1)
    num_support_calls: int = Field(ge=0, le=50)
    avg_monthly_gb: float = Field(ge=0, le=2000)
    late_payments: int = Field(ge=0, le=50)


def _load_artifact():
    if not os.path.exists(ARTIFACT):
        raise RuntimeError(
            "Model artifact not found. Run `python train.py` first."
        )
    return joblib.load(ARTIFACT)


# Loaded once at import; uvicorn workers each get their own copy.
_ARTIFACT = _load_artifact()
_MODEL = _ARTIFACT["model"]
_FEATURES = _ARTIFACT["features"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/model_info")
def model_info():
    return {
        "model": type(_MODEL).__name__,
        "features": _FEATURES,
        "seed": _ARTIFACT.get("seed"),
        "threshold": THRESHOLD,
        "synthetic_data": True,
    }


@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        import pandas as pd

        row = pd.DataFrame(
            [customer.model_dump()], columns=_FEATURES
        )
        prob = float(_MODEL.predict_proba(row)[0, 1])
    except Exception as exc:  # noqa: BLE001 - surface as 500, not crash
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": prob >= THRESHOLD,
        "threshold": THRESHOLD,
    }

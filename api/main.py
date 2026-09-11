"""FastAPI risk scoring service."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features.engineering import build_features
from src.models.rules import rules_score

app = FastAPI(title="Real-time Payment Fraud Detection", version="0.1.0")
MODEL_PATH = Path("artifacts/model.joblib")
try:
    import joblib
except ImportError:
    joblib = None
bundle = joblib.load(MODEL_PATH) if joblib is not None and MODEL_PATH.exists() else None


class Transaction(BaseModel):
    step: int = Field(ge=0)
    type: str
    amount: float = Field(ge=0)
    nameOrig: str
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    nameDest: str
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)
    isFlaggedFraud: int = Field(default=0, ge=0, le=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": bundle is not None}


@app.post("/predict")
def predict(transaction: Transaction) -> dict:
    started = time.perf_counter()
    row = pd.DataFrame([transaction.model_dump()])
    if bundle is None:
        score = float(rules_score(row).iloc[0])
        model_source = "rules"
    else:
        features = build_features(row)
        for column in bundle["categorical"]:
            features[column] = features[column].astype("category")
        score = float(bundle["model"].predict_proba(features[bundle["columns"]])[:, 1][0])
        model_source = "lightgbm"
    return {"risk_score": score, "decision": "review" if score >= 0.5 else "allow", "model": model_source, "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
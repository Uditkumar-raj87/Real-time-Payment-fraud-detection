"""Leakage-safe transaction and graph feature engineering."""

from __future__ import annotations

import pandas as pd
import numpy as np

TARGET = "isFraud"
IDENTITY = {"step", "nameOrig", "nameDest", TARGET, "isFlaggedFraud"}


def time_split(frame: pd.DataFrame, train_fraction: float = 0.6, validation_fraction: float = 0.2):
    """Split chronologically by PaySim's simulation step."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    ordered = frame.sort_values("step", kind="mergesort").reset_index(drop=True)
    steps = ordered["step"].drop_duplicates().tolist()
    train_end = max(1, int(len(steps) * train_fraction))
    validation_end = min(len(steps), train_end + max(1, int(len(steps) * validation_fraction)))
    train_steps = set(steps[:train_end])
    validation_steps = set(steps[train_end:validation_end])
    return (
        ordered[ordered.step.isin(train_steps)].copy(),
        ordered[ordered.step.isin(validation_steps)].copy(),
        ordered[~ordered.step.isin(train_steps | validation_steps)].copy(),
    )


def build_features(frame: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create point-in-time features; reference supplies prior graph statistics only."""
    data = frame.sort_values("step", kind="mergesort").copy()
    prior = reference if reference is not None else data.iloc[0:0]
    for column in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    data["amount_log"] = np.log1p(data["amount"])
    data["balance_delta_orig"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["balance_delta_dest"] = data["newbalanceDest"] - data["oldbalanceDest"]
    data["hour"] = data["step"] % 24
    data["is_transfer"] = data["type"].isin(["TRANSFER", "CASH_OUT"]).astype("int8")
    data["orig_tx_count"] = data.groupby("nameOrig").cumcount()
    data["dest_tx_count"] = data.groupby("nameDest").cumcount()
    data["orig_amount_rolling"] = data.groupby("nameOrig")["amount"].transform(
        lambda values: values.shift(1).rolling(10, min_periods=1).mean()
    ).fillna(0.0)
    prior_orig = prior.groupby("nameOrig").size()
    prior_dest = prior.groupby("nameDest").size()
    prior_device = prior.groupby("nameOrig")["nameDest"].nunique()
    data["orig_history_count"] = data["nameOrig"].map(prior_orig).fillna(0).astype("int64")
    data["dest_history_count"] = data["nameDest"].map(prior_dest).fillna(0).astype("int64")
    data["account_device_degree"] = data["nameOrig"].map(prior_device).fillna(0).astype("int64")
    shared_dest = prior.groupby("nameDest")["nameOrig"].nunique()
    data["shared_device_risk"] = data["nameDest"].map(shared_dest).fillna(0).astype("int64")
    fraud_rate = prior.groupby("nameDest")[TARGET].mean() if TARGET in prior else pd.Series(dtype=float)
    data["merchant_anomaly_score"] = data["nameDest"].map(fraud_rate).fillna(0.0)
    return data.drop(columns=[column for column in IDENTITY if column in data], errors="ignore")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model-ready numeric and one-hot-ready categorical feature names."""
    return [column for column in frame.columns if column not in {TARGET}]
"""Fast, dependency-light rules baseline for payment risk scoring."""

from __future__ import annotations

import pandas as pd


def rules_score(frame: pd.DataFrame) -> pd.Series:
    """Score suspicious balance depletion, flagged transfers, and transfer types."""
    return (
        (frame["amount"] > frame["oldbalanceOrg"] * 0.95).astype(float) * 0.45
        + frame["isFlaggedFraud"].astype(float) * 0.4
        + frame["type"].isin(["TRANSFER", "CASH_OUT"]).astype(float) * 0.15
    ).clip(0, 1)
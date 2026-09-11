import pandas as pd
import pytest

from src.data.ingest import clean_paysim


def test_clean_paysim_normalizes_types_and_columns():
    frame = pd.DataFrame({
        "step": [1], "type": [" payment "], "amount": [10], "nameOrig": ["C1"],
        "oldbalanceOrg": [20], "newbalanceOrig": [10], "nameDest": ["M1"],
        "oldbalanceDest": [0], "newbalanceDest": [10], "isFraud": [0],
        "isFlaggedFraud": [0], "extra": ["ignored"],
    })
    cleaned = clean_paysim(frame)
    assert list(cleaned.columns) == [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
    ]
    assert cleaned.loc[0, "type"] == "PAYMENT"
    assert cleaned["step"].dtype == "int64"


def test_clean_paysim_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        clean_paysim(pd.DataFrame({"step": [1]}))
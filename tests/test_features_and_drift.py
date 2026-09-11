import pandas as pd

from src.features.engineering import build_features, time_split
from src.monitoring.drift import psi


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "step": [1, 2, 3, 4, 5], "type": ["PAYMENT"] * 5, "amount": [10, 20, 30, 40, 50],
        "nameOrig": ["a", "a", "b", "b", "c"], "oldbalanceOrg": [20] * 5,
        "newbalanceOrig": [10] * 5, "nameDest": ["m"] * 5, "oldbalanceDest": [0] * 5,
        "newbalanceDest": [10] * 5, "isFraud": [0, 0, 1, 0, 0], "isFlaggedFraud": [0] * 5,
    })


def test_time_split_is_chronological():
    train, validation, test = time_split(sample_frame(), 0.6, 0.2)
    assert train.step.max() < validation.step.min() < test.step.min()


def test_features_use_prior_reference_and_psi_is_zero_for_same_data():
    frame = sample_frame()
    features = build_features(frame, frame.iloc[:2])
    assert features.loc[0, "orig_history_count"] == 2
    assert psi(frame.amount, frame.amount) == 0.0
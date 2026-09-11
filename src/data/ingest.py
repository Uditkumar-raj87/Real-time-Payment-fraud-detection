"""Download and clean the public synthetic PaySim dataset."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

DEFAULT_URL = (
    "https://raw.githubusercontent.com/EdgarLopezPhD/PaySim/master/"
    "PS_20174392719_1491204439457_log.csv"
)
REQUIRED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]
NUMERIC_COLUMNS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
    "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def clean_paysim(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized PaySim frame and fail fast on missing columns."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"PaySim data is missing required columns: {missing}")

    cleaned = frame[REQUIRED_COLUMNS].copy()
    cleaned["type"] = cleaned["type"].astype("string").str.strip().str.upper()
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
    cleaned["step"] = cleaned["step"].astype("int64")
    cleaned["isFraud"] = cleaned["isFraud"].astype("int64")
    cleaned["isFlaggedFraud"] = cleaned["isFlaggedFraud"].astype("int64")
    return cleaned


def download_paysim(url: str = DEFAULT_URL, output_path: str | Path = "data/raw/paysim.csv") -> Path:
    """Download, clean, and persist PaySim without embedding data in the repository."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        raw = pd.read_csv(response)
    cleaned = clean_paysim(raw)
    if destination.suffix.lower() == ".parquet":
        cleaned.to_parquet(destination, index=False)
    else:
        cleaned.to_csv(destination, index=False)
    return destination
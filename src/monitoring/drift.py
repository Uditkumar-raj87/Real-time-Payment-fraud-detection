"""Population Stability Index (PSI) utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Calculate PSI with quantile bins from the reference population."""
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_rate = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_rate = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return float(np.sum((actual_rate - expected_rate) * np.log(actual_rate / expected_rate)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    columns = sorted(set(reference.select_dtypes("number")) & set(current.select_dtypes("number")))
    return pd.DataFrame({"feature": columns, "psi": [psi(reference[c], current[c]) for c in columns]})
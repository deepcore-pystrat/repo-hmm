"""
Train / test splitting utilities.
"""

import numpy as np
import pandas as pd


def fixed_split(
    X: np.ndarray,
    idx: pd.DatetimeIndex,
    train_ratio: float = 0.80,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split features chronologically (no shuffle).

    Returns (X_train, X_test, idx_train, idx_test).
    """
    n = int(train_ratio * len(X))
    return X[:n], X[n:], idx[:n], idx[n:]

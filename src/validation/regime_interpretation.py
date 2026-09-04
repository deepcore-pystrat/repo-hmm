from __future__ import annotations

import pandas as pd
import numpy as np


def summarize_spot_states(
    states: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    sample: str = "sample",
) -> pd.DataFrame:
    """
    Build a state-level summary from spot-only HMM features.
    """
    df = pd.DataFrame(X, columns=feature_names).copy()
    df["state"] = states

    rows = []
    for s in sorted(df["state"].unique()):
        sub = df[df["state"] == s]

        rows.append({
            "sample": sample,
            "state": int(s),
            "count": len(sub),
            "freq": len(sub) / len(df),
            "mean_return": sub["returns_MID"].mean(),
            "std_return": sub["returns_MID"].std(),
            "mean_rvol": sub["rvol_20_MID"].mean(),
            "std_rvol": sub["rvol_20_MID"].std(),
            "mean_momentum": sub["momentum_10_MID"].mean(),
            "std_momentum": sub["momentum_10_MID"].std(),
            "mean_spread_z": sub["zscore_20_BA_SPREAD"].mean(),
        })

    return pd.DataFrame(rows)


def assign_spot_regime_labels(summary_df: pd.DataFrame) -> dict[int, str]:
    """
    Assign interpretable labels to HMM states for a spot-only model.

    Rules:
    - lowest volatility state -> CALM
    - highest volatility state -> CRISIS
    - among remaining states:
        highest momentum -> TREND_PLUS
        lowest momentum -> TREND_MINUS
    """
    df = summary_df.copy()
    label_map: dict[int, str] = {}

    calm_state = int(df.loc[df["mean_rvol"].idxmin(), "state"])
    crisis_state = int(df.loc[df["mean_rvol"].idxmax(), "state"])

    label_map[calm_state] = "CALM"
    label_map[crisis_state] = "CRISIS"

    remaining = [s for s in df["state"].tolist() if s not in label_map]

    if len(remaining) == 2:
        sub = df[df["state"].isin(remaining)].copy()
        trend_plus_state = int(sub.loc[sub["mean_momentum"].idxmax(), "state"])
        trend_minus_state = int(sub.loc[sub["mean_momentum"].idxmin(), "state"])

        label_map[trend_plus_state] = "TREND_PLUS"
        label_map[trend_minus_state] = "TREND_MINUS"
    else:
        # fallback in case K != 4
        for s in remaining:
            if s not in label_map:
                label_map[s] = "UNLABELED"

    return label_map


def apply_regime_labels(states: np.ndarray, label_map: dict[int, str]) -> pd.Series:
    """
    Map numerical HMM states to human-readable labels.
    """
    return pd.Series(states).map(label_map)
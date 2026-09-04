from __future__ import annotations

import numpy as np
import pandas as pd


def build_market_context_df(
    spot_df: pd.DataFrame,
    futures_df: pd.DataFrame,
    state_series: pd.Series,
) -> pd.DataFrame:
    """
    Merge spot, futures, and HMM states on dates, then build economic profiling variables.

    Required:
    - spot_df indexed by date with column MID
    - futures_df indexed by date with at least FUTURES_CLOSE
    - state_series indexed by date, containing HMM states
    """
    df = (
        spot_df[["MID"]]
        .join(futures_df[["FUTURES_CLOSE"]], how="inner")
        .join(state_series.rename("state"), how="inner")
        .sort_index()
        .copy()
    )

    df["MID"] = pd.to_numeric(df["MID"], errors="coerce")
    df["FUTURES_CLOSE"] = pd.to_numeric(df["FUTURES_CLOSE"], errors="coerce")
    df = df.dropna(subset=["MID", "FUTURES_CLOSE", "state"]).copy()

    df = df[(df["MID"] > 0) & (df["FUTURES_CLOSE"] > 0)].copy()

    df["log_spot"] = np.log(df["MID"])
    df["log_fut"] = np.log(df["FUTURES_CLOSE"])

    df["dlog_spot"] = df["log_spot"].diff()
    df["dlog_fut"] = df["log_fut"].diff()

    # basis simple: futures vs spot
    df["basis"] = df["log_fut"] - df["log_spot"]

    # rolling volatilities
    df["spot_vol_20"] = df["dlog_spot"].rolling(20).std() * np.sqrt(252)
    df["fut_vol_20"] = df["dlog_fut"].rolling(20).std() * np.sqrt(252)

    # trend proxies
    df["spot_momentum_20"] = df["MID"].pct_change(20)
    df["fut_momentum_20"] = df["FUTURES_CLOSE"].pct_change(20)

    return df


def summarize_economic_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute state-by-state economic summaries.
    """
    out = df.groupby("state").agg(
        n_obs=("state", "size"),
        freq=("state", lambda x: len(x) / len(df)),
        mean_dlog_spot=("dlog_spot", "mean"),
        std_dlog_spot=("dlog_spot", "std"),
        mean_dlog_fut=("dlog_fut", "mean"),
        std_dlog_fut=("dlog_fut", "std"),
        mean_basis=("basis", "mean"),
        std_basis=("basis", "std"),
        mean_spot_vol_20=("spot_vol_20", "mean"),
        mean_fut_vol_20=("fut_vol_20", "mean"),
        mean_spot_mom_20=("spot_momentum_20", "mean"),
        mean_fut_mom_20=("fut_momentum_20", "mean"),
    ).reset_index()

    return out


def assign_economic_labels(summary_df: pd.DataFrame) -> tuple[dict[int, str], pd.DataFrame]:
    """
    Rule-based economic labeling.

    Important:
    This does NOT redefine the HMM states.
    It only gives an ex-post interpretation.
    """
    label_map: dict[int, str] = {}
    rows = []

    for _, row in summary_df.iterrows():
        s = int(row["state"])

        mean_spot = row["mean_dlog_spot"]
        mean_fut = row["mean_dlog_fut"]
        vol = row["mean_spot_vol_20"]
        basis = row["mean_basis"]
        mom = row["mean_spot_mom_20"]

        # Heuristic labeling rules
        if mean_spot < 0 and mean_fut < 0 and vol >= summary_df["mean_spot_vol_20"].median():
            label = "CRASH_NORMALIZATION"
        elif mean_spot > 0 and mean_fut > 0 and vol >= summary_df["mean_spot_vol_20"].median():
            label = "BOOM_STRESS"
        elif mean_spot > 0 and mom > 0 and vol < summary_df["mean_spot_vol_20"].median():
            label = "RECOVERY_TIGHTENING"
        else:
            label = "DEPRESSION_OVERSUPPLY"

        label_map[s] = label

        row_out = row.to_dict()
        row_out["economic_label"] = label
        rows.append(row_out)

    labeled_summary = pd.DataFrame(rows)
    return label_map, labeled_summary
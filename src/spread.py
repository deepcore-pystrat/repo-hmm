"""
Main production-style pipeline for spot-derived scarcity signal validation.

Key convention
--------------
Quoted BID / OFFER / MID are not spot prices.
They are quotations around the futures price, with convention:

    SPOT_IMPLIED = FUTURES_CLOSE + MID / 100

Hence:
- MID is a quoted spread component
- SPOT_IMPLIED is the reconstructed spot-like price
- scarcity_signal = log(SPOT_IMPLIED / FUTURES_CLOSE)

Purpose
-------
1. reconstruct implied spot from quotations
2. build the spot-futures scarcity signal
3. evaluate predictive power on forward futures returns (5d, 20d)
4. compare futures-only vs futures+spot models
5. run a top-vs-bottom quantile test
6. backtest simple signal-based futures strategies
7. test a regime filter using hmmlearn as a robustness check
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import pearsonr, ttest_ind
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from simulations.backtester import backtest, print_metrics, plot_backtest
from data_loader.loader_data import DataConfig
from data_loader.laoding_services import get_bid_offer_data
from features.splitter import fixed_split


# ============================================================================
# Plot helpers
# ============================================================================

def plot_spot_vs_futures(df: pd.DataFrame) -> None:
    tmp = df[["SPOT_IMPLIED", "FUTURES_CLOSE"]].dropna().copy()

    plt.figure(figsize=(14, 6))
    plt.plot(tmp.index, tmp["SPOT_IMPLIED"], label="Spot implied", linewidth=1.5)
    plt.plot(tmp.index, tmp["FUTURES_CLOSE"], label="Futures", linewidth=1.5)
    plt.title("Spot implied vs Futures")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scarcity_signal(df: pd.DataFrame) -> None:
    tmp = df[["scarcity_signal"]].dropna().copy()

    plt.figure(figsize=(14, 5))
    plt.plot(tmp.index, tmp["scarcity_signal"], linewidth=1.5)
    plt.axhline(tmp["scarcity_signal"].mean(), linestyle="--", linewidth=1.0)
    plt.title("Scarcity signal over time")
    plt.xlabel("Date")
    plt.ylabel("Scarcity signal")
    plt.tight_layout()
    plt.show()


def plot_forward_return_by_quantile(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    target_col: str = "future_fut_return_20d",
    q: int = 5,
) -> None:
    tmp = df[[signal_col, target_col]].dropna().copy()
    tmp["bucket"] = pd.qcut(tmp[signal_col], q=q, labels=False, duplicates="drop")

    summary = tmp.groupby("bucket")[target_col].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(summary.index.astype(str), summary.values)
    plt.title(f"Average forward return by signal quantile ({target_col})")
    plt.xlabel("Signal quantile")
    plt.ylabel("Average forward return")
    plt.tight_layout()
    plt.show()


def plot_forward_return_by_quantile_three_panels(
    df_full: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    target_col: str = "future_fut_return_20d",
    q: int = 5,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, data, title in zip(
        axes,
        [df_full, df_train, df_test],
        ["Full sample", "Train", "Test"]
    ):
        tmp = data[[signal_col, target_col]].dropna().copy()
        tmp["bucket"] = pd.qcut(tmp[signal_col], q=q, labels=False, duplicates="drop")
        summary = tmp.groupby("bucket")[target_col].mean()

        ax.bar(summary.index.astype(str), summary.values)
        ax.set_title(title)
        ax.set_xlabel("Signal quantile")
        ax.set_ylabel("Average forward return")

    fig.suptitle(f"Average forward return by signal quantile ({target_col})")
    plt.tight_layout()
    plt.show()


def plot_top_bottom_bucket_cumulative(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    target_col: str = "future_fut_return_20d",
    q: int = 5,
) -> None:
    tmp = df[[signal_col, target_col]].dropna().copy()
    tmp["bucket"] = pd.qcut(tmp[signal_col], q=q, labels=False, duplicates="drop")

    low_bucket = tmp["bucket"].min()
    high_bucket = tmp["bucket"].max()

    top = tmp.loc[tmp["bucket"] == high_bucket, target_col].reset_index(drop=True)
    bottom = tmp.loc[tmp["bucket"] == low_bucket, target_col].reset_index(drop=True)

    top_cum = (1 + top).cumprod()
    bottom_cum = (1 + bottom).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(top_cum.index, top_cum.values, label="Top bucket")
    plt.plot(bottom_cum.index, bottom_cum.values, label="Bottom bucket")
    plt.title(f"Cumulative forward returns: top vs bottom signal bucket ({target_col})")
    plt.xlabel("Observation rank")
    plt.ylabel("Cumulative growth")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rolling_ic(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    target_col: str = "future_fut_return_20d",
    window: int = 60,
) -> None:
    tmp = df[[signal_col, target_col]].dropna().copy()
    ic = tmp[signal_col].rolling(window).corr(tmp[target_col])

    plt.figure(figsize=(14, 5))
    plt.plot(tmp.index, ic, label="Rolling IC", linewidth=1.5)
    plt.axhline(0, linestyle="--", linewidth=1.0)
    plt.title(f"Rolling Information Coefficient ({window} days)")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ic_by_regime(ic_df: pd.DataFrame, title: str) -> None:
    tmp = ic_df.dropna(subset=["corr"]).copy()

    plt.figure(figsize=(8, 5))
    plt.bar(tmp["state"].astype(str), tmp["corr"])
    plt.axhline(0, linestyle="--", linewidth=1.0)
    plt.title(title)
    plt.xlabel("State")
    plt.ylabel("Signal / future return correlation")
    plt.tight_layout()
    plt.show()


# ============================================================================
# Data loading
# ============================================================================

def load_spot_data() -> pd.DataFrame:
    cfg = DataConfig(
        data_path=r"D:\Downloads\deepcore--2026-04-07-12_42-0026205d-213a-4e96-bbe9-a7c977a1c3d4.xlsx",
        asset_name="AR",
    )
    df = get_bid_offer_data(cfg).copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def load_futures_data() -> pd.DataFrame:
    fut = pd.read_csv(
        r"D:\Downloads\Futures CBOT Corn.csv",
        sep=";",
    ).copy()

    fut["Date"] = pd.to_datetime(fut["Date"], errors="coerce")
    fut["Close"] = pd.to_numeric(fut["Close"], errors="coerce")

    fut = fut.dropna(subset=["Date", "Close"]).copy()
    fut = fut.sort_values("Date")
    fut = fut.groupby("Date", as_index=False).last()
    fut = fut.rename(columns={"Close": "FUTURES_CLOSE"})
    fut = fut.set_index("Date")
    return fut


# ============================================================================
# Signal construction
# ============================================================================

def build_spread_dataset() -> pd.DataFrame:
    spot_df = load_spot_data()
    fut_df = load_futures_data()

    spot_keep = ["MID", "BID", "OFFER", "BA_SPREAD"]
    missing_spot = [c for c in spot_keep if c not in spot_df.columns]
    if missing_spot:
        raise ValueError(f"Colonnes spot manquantes: {missing_spot}")

    df = spot_df[spot_keep].join(fut_df[["FUTURES_CLOSE"]], how="inner")
    df = df.sort_index()

    for col in ["MID", "FUTURES_CLOSE", "BID", "OFFER", "BA_SPREAD"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["MID", "FUTURES_CLOSE"]).copy()
    df = df[df["FUTURES_CLOSE"] > 0].copy()

    # Quotations -> implied spot
    df["SPOT_IMPLIED"] = df["FUTURES_CLOSE"] + df["MID"] / 100.0
    df["BID_IMPLIED"] = df["FUTURES_CLOSE"] + df["BID"] / 100.0
    df["OFFER_IMPLIED"] = df["FUTURES_CLOSE"] + df["OFFER"] / 100.0

    df = df[
        (df["SPOT_IMPLIED"] > 0) &
        (df["BID_IMPLIED"] > 0) &
        (df["OFFER_IMPLIED"] > 0)
    ].copy()

    df["log_spot"] = np.log(df["SPOT_IMPLIED"])
    df["log_fut"] = np.log(df["FUTURES_CLOSE"])

    df["logspread"] = df["log_spot"] - df["log_fut"]
    df["spread_level"] = df["SPOT_IMPLIED"] - df["FUTURES_CLOSE"]

    # Main signal
    df["scarcity_signal"] = df["logspread"]

    # Diagnostics
    df["basis_points_quote"] = df["MID"]
    df["basis_quote_level"] = df["MID"] / 100.0

    # Forward futures returns
    df["future_fut_return_1d"] = df["log_fut"].shift(-1) - df["log_fut"]
    df["future_fut_return_5d"] = df["log_fut"].shift(-5) - df["log_fut"]
    df["future_fut_return_20d"] = df["log_fut"].shift(-20) - df["log_fut"]

    # Futures-only controls
    df["past_fut_5d"] = df["log_fut"] - df["log_fut"].shift(5)
    df["past_fut_20d"] = df["log_fut"] - df["log_fut"].shift(20)

    # Rolling normalization / thresholds
    df["scarcity_mean_60"] = df["scarcity_signal"].rolling(60).mean()
    df["scarcity_std_60"] = df["scarcity_signal"].rolling(60).std()
    df["scarcity_z_60"] = (
        (df["scarcity_signal"] - df["scarcity_mean_60"]) /
        df["scarcity_std_60"]
    )
    df["scarcity_q20_60"] = df["scarcity_signal"].rolling(60).quantile(0.20)
    df["scarcity_q80_60"] = df["scarcity_signal"].rolling(60).quantile(0.80)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ============================================================================
# Split helper
# ============================================================================

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.80
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = df.index
    _, _, idx_train, idx_test = fixed_split(
        np.zeros((len(df), 1)),
        idx,
        train_ratio=train_ratio
    )
    train_df = df.loc[idx_train].copy()
    test_df = df.loc[idx_test].copy()
    return train_df, test_df


# ============================================================================
# Validation blocks
# ============================================================================

def predictive_corr_summary(
    df: pd.DataFrame,
    target_col: str,
    signal_col: str = "scarcity_signal",
) -> dict:
    tmp = df[[signal_col, target_col]].dropna().copy()
    corr, pval = pearsonr(tmp[signal_col], tmp[target_col])
    return {
        "n": int(len(tmp)),
        "corr": float(corr),
        "p_value": float(pval),
    }


def incremental_alpha_summary(
    df: pd.DataFrame,
    target_col: str,
    signal_col: str = "scarcity_signal",
) -> dict:
    cols = [signal_col, target_col, "past_fut_5d", "past_fut_20d"]
    tmp = df[cols].dropna().copy()

    y = tmp[target_col]
    X = tmp[[signal_col, "past_fut_5d", "past_fut_20d"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type="HC3")

    return {
        "n": int(len(tmp)),
        "r2": float(model.rsquared),
        "coef_signal": float(model.params[signal_col]),
        "t_signal": float(model.tvalues[signal_col]),
        "p_signal": float(model.pvalues[signal_col]),
    }


def compare_future_only_vs_spot_augmented_summary(
    df: pd.DataFrame,
    target_col: str,
    signal_col: str = "scarcity_signal",
) -> dict:
    cols = [signal_col, target_col, "past_fut_5d", "past_fut_20d"]
    tmp = df[cols].dropna().copy()

    y = tmp[target_col]

    X1 = sm.add_constant(tmp[["past_fut_5d", "past_fut_20d"]])
    m1 = sm.OLS(y, X1).fit(cov_type="HC3")

    X2 = sm.add_constant(tmp[[signal_col, "past_fut_5d", "past_fut_20d"]])
    m2 = sm.OLS(y, X2).fit(cov_type="HC3")

    return {
        "n": int(len(tmp)),
        "r2_future_only": float(m1.rsquared),
        "r2_future_plus_spot": float(m2.rsquared),
        "delta_r2": float(m2.rsquared - m1.rsquared),
        "coef_signal": float(m2.params[signal_col]),
        "t_signal": float(m2.tvalues[signal_col]),
        "p_signal": float(m2.pvalues[signal_col]),
        "aic_future_only": float(m1.aic),
        "aic_future_plus_spot": float(m2.aic),
    }


def quantile_spread_summary(
    df: pd.DataFrame,
    target_col: str,
    signal_col: str = "scarcity_signal",
    q: int = 5,
) -> dict:
    tmp = df[[signal_col, target_col]].dropna().copy()
    tmp["bucket"] = pd.qcut(tmp[signal_col], q=q, labels=False, duplicates="drop")

    low_bucket = tmp["bucket"].min()
    high_bucket = tmp["bucket"].max()

    low = tmp.loc[tmp["bucket"] == low_bucket, target_col]
    high = tmp.loc[tmp["bucket"] == high_bucket, target_col]

    stat, pval = ttest_ind(high, low, equal_var=False, nan_policy="omit")

    return {
        "n_top": int(len(high)),
        "n_bottom": int(len(low)),
        "top_mean": float(high.mean()),
        "bottom_mean": float(low.mean()),
        "top_bottom_spread": float(high.mean() - low.mean()),
        "t_stat": float(stat),
        "p_value": float(pval),
    }


# ============================================================================
# Strategy helpers
# ============================================================================

def build_quantile_directions(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    low_q_col: str = "scarcity_q20_60",
    high_q_col: str = "scarcity_q80_60",
) -> pd.Series:
    directions = pd.Series(0.0, index=df.index, name="direction")
    directions[df[signal_col] >= df[high_q_col]] = 1.0
    directions[df[signal_col] <= df[low_q_col]] = -1.0
    return directions


def build_quantile_directions_holding_period(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    low_q_col: str = "scarcity_q20_60",
    high_q_col: str = "scarcity_q80_60",
    holding_period: int = 5,
) -> pd.Series:
    out = df.copy()
    directions = pd.Series(0.0, index=out.index, name="direction")

    n = len(out)
    i = 0
    while i < n:
        sig = out.iloc[i][signal_col]
        low = out.iloc[i][low_q_col]
        high = out.iloc[i][high_q_col]

        direction = 0.0
        if pd.notna(sig) and pd.notna(low) and pd.notna(high):
            if sig >= high:
                direction = 1.0
            elif sig <= low:
                direction = -1.0

        end = min(i + holding_period, n)
        directions.iloc[i:end] = direction
        i += holding_period

    return directions


def build_overlapping_quantile_exposures(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    low_q_col: str = "scarcity_q20_60",
    high_q_col: str = "scarcity_q80_60",
    holding_period: int = 20,
    normalize: bool = True,
) -> pd.Series:
    out = df.copy()

    signal = pd.Series(0.0, index=out.index, name="daily_signal")
    signal[out[signal_col] >= out[high_q_col]] = 1.0
    signal[out[signal_col] <= out[low_q_col]] = -1.0

    exposure = signal.rolling(window=holding_period, min_periods=1).sum()

    if normalize:
        exposure = exposure / holding_period

    exposure.name = "overlapping_exposure"
    return exposure


def build_hmm_filtered_overlapping_exposures(
    df: pd.DataFrame,
    good_states: list[int],
    state_col: str = "state",
    signal_col: str = "scarcity_signal",
    low_q_col: str = "scarcity_q20_60",
    high_q_col: str = "scarcity_q80_60",
    holding_period: int = 20,
    normalize: bool = True,
) -> pd.Series:
    out = df.copy()

    signal = pd.Series(0.0, index=out.index, name="daily_signal")
    eligible = out[state_col].isin(good_states)

    signal[(out[signal_col] >= out[high_q_col]) & eligible] = 1.0
    signal[(out[signal_col] <= out[low_q_col]) & eligible] = -1.0

    exposure = signal.rolling(window=holding_period, min_periods=1).sum()

    if normalize:
        exposure = exposure / holding_period

    exposure.name = "hmm_filtered_exposure"
    return exposure


# ============================================================================
# Backtest helpers
# ============================================================================

def run_backtest_from_directions(
    df: pd.DataFrame,
    directions: pd.Series | np.ndarray,
    price_col: str = "FUTURES_CLOSE",
    vol_target: float = 0.15,
    vol_lookback: int = 20,
    max_leverage: float = 2.0,
    plot: bool = True,
):
    tmp = df.copy()

    if isinstance(directions, pd.Series):
        directions = directions.reindex(tmp.index).values
    else:
        directions = np.asarray(directions, dtype=float)

    states = np.zeros(len(tmp), dtype=int)

    valid_mask = tmp[price_col].notna()
    tmp = tmp.loc[valid_mask].copy()
    directions = directions[valid_mask.values]
    states = states[valid_mask.values]

    prices = tmp[price_col].astype(float).values
    dates = pd.DatetimeIndex(tmp.index)

    result = backtest(
        prices=prices,
        states=states,
        directions=directions,
        vol_target=vol_target,
        vol_lookback=vol_lookback,
        max_leverage=max_leverage,
    )

    print("\nBACKTEST RESULTS")
    print_metrics(result)

    if plot:
        plot_backtest(
            result=result,
            dates=dates,
            prices=prices,
            states=states,
            vol_target=vol_target,
            max_leverage=max_leverage,
        )

    return result, tmp


def run_backtest_from_exposures(
    df: pd.DataFrame,
    exposures: pd.Series | np.ndarray,
    price_col: str = "FUTURES_CLOSE",
    vol_target: float = 0.15,
    vol_lookback: int = 20,
    max_leverage: float = 2.0,
    plot: bool = True,
):
    tmp = df.copy()

    if isinstance(exposures, pd.Series):
        exposures = exposures.reindex(tmp.index).values
    else:
        exposures = np.asarray(exposures, dtype=float)

    states = np.zeros(len(tmp), dtype=int)

    valid_mask = tmp[price_col].notna()
    tmp = tmp.loc[valid_mask].copy()
    exposures = exposures[valid_mask.values]
    states = states[valid_mask.values]

    prices = tmp[price_col].astype(float).values
    dates = pd.DatetimeIndex(tmp.index)

    result = backtest(
        prices=prices,
        states=states,
        exposures=exposures,
        vol_target=vol_target,
        vol_lookback=vol_lookback,
        max_leverage=max_leverage,
    )

    print("\nBACKTEST RESULTS")
    print_metrics(result)

    if plot:
        plot_backtest(
            result=result,
            dates=dates,
            prices=prices,
            states=states,
            vol_target=vol_target,
            max_leverage=max_leverage,
        )

    return result, tmp


# ============================================================================
# hmmlearn benchmark
# ============================================================================

def build_hmm_features_for_hmmlearn(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["dlog_fut"] = df["log_fut"].diff()
    out["fut_vol_20"] = out["dlog_fut"].rolling(20).std()
    out["scarcity_signal"] = df["scarcity_signal"]
    out["scarcity_z_60"] = df["scarcity_z_60"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def fit_hmmlearn_regimes(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    n_states: int = 2,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42,
):
    feat_df = build_hmm_features_for_hmmlearn(df)

    split_idx = int(len(feat_df) * train_ratio)
    train_feat = feat_df.iloc[:split_idx].copy()
    test_feat = feat_df.iloc[split_idx:].copy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat.values)
    X_test = scaler.transform(test_feat.values)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )

    model.fit(X_train)

    states_train = model.predict(X_train)
    states_test = model.predict(X_test)

    proba_train = model.predict_proba(X_train)
    proba_test = model.predict_proba(X_test)

    train_states_df = pd.DataFrame(
        proba_train,
        index=train_feat.index,
        columns=[f"state_prob_{k}" for k in range(n_states)],
    )
    train_states_df["state"] = states_train

    test_states_df = pd.DataFrame(
        proba_test,
        index=test_feat.index,
        columns=[f"state_prob_{k}" for k in range(n_states)],
    )
    test_states_df["state"] = states_test

    return {
        "model": model,
        "feat_df": feat_df,
        "train_states_df": train_states_df,
        "test_states_df": test_states_df,
    }


def signal_ic_by_regime(
    df: pd.DataFrame,
    state_col: str = "state",
    signal_col: str = "scarcity_signal",
    target_col: str = "future_fut_return_20d",
) -> pd.DataFrame:
    rows = []
    tmp = df[[state_col, signal_col, target_col]].dropna().copy()

    for s in sorted(tmp[state_col].unique()):
        sub = tmp[tmp[state_col] == s].copy()

        if len(sub) < 20:
            rows.append({
                "state": s,
                "n": len(sub),
                "corr": np.nan,
                "p_value": np.nan,
            })
            continue

        corr, pval = pearsonr(sub[signal_col], sub[target_col])
        rows.append({
            "state": int(s),
            "n": int(len(sub)),
            "corr": float(corr),
            "p_value": float(pval),
        })

    return pd.DataFrame(rows)


# ============================================================================
# Reporting helpers
# ============================================================================

def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_dict_as_lines(d: dict) -> None:
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{k:>24s}: {v:.6f}")
        else:
            print(f"{k:>24s}: {v}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    df = build_spread_dataset()
    train_df, test_df = split_dataset(df, train_ratio=0.80)

    # ------------------------------------------------------------------
    # hmmlearn benchmark only
    # ------------------------------------------------------------------
    print_section("HMMLEARN | FIT AND STATE INFERENCE")
    hmmlearn_out = fit_hmmlearn_regimes(df, train_ratio=0.80, n_states=2)

    train_hmm_df = train_df.join(hmmlearn_out["train_states_df"], how="left")
    test_hmm_df = test_df.join(hmmlearn_out["test_states_df"], how="left")

    print_section("HMMLEARN | SIGNAL IC BY REGIME | TRAIN")
    ic_regime_train = signal_ic_by_regime(
        train_hmm_df,
        state_col="state",
        signal_col="scarcity_signal",
        target_col="future_fut_return_20d",
    )
    print(ic_regime_train)

    print_section("HMMLEARN | SIGNAL IC BY REGIME | TEST")
    ic_regime_test = signal_ic_by_regime(
        test_hmm_df,
        state_col="state",
        signal_col="scarcity_signal",
        target_col="future_fut_return_20d",
    )
    print(ic_regime_test)

    plot_ic_by_regime(ic_regime_train, "Signal IC by regime - TRAIN (hmmlearn)")
    plot_ic_by_regime(ic_regime_test, "Signal IC by regime - TEST (hmmlearn)")

    good_states = (
        ic_regime_train.loc[
            (ic_regime_train["corr"] > 0.30) &
            (ic_regime_train["p_value"] < 0.05),
            "state"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    print_section("HMMLEARN | GOOD STATES")
    print("Good states selected from TRAIN:", good_states)

    # ------------------------------------------------------------------
    # Basic data overview
    # ------------------------------------------------------------------
    print_section("DATA SNAPSHOT")
    print(f"Full sample  : {df.index.min()} -> {df.index.max()} | n={len(df)}")
    print(f"Train sample : {train_df.index.min()} -> {train_df.index.max()} | n={len(train_df)}")
    print(f"Test sample  : {test_df.index.min()} -> {test_df.index.max()} | n={len(test_df)}")

    print_section("RAW DATA CHECK")
    print_section("PLOTS | MARKET VIEW")
    plot_spot_vs_futures(df)
    plot_scarcity_signal(df)

    cols_check = [
        "MID", "BID", "OFFER", "FUTURES_CLOSE",
        "SPOT_IMPLIED", "spread_level", "scarcity_signal"
    ]
    print(df[cols_check].head(10))
    print("\nDescriptive stats:")
    print(df[cols_check].describe().round(6))

    # ------------------------------------------------------------------
    # Signal validation
    # ------------------------------------------------------------------
    for horizon in [5, 20]:
        target = f"future_fut_return_{horizon}d"

        print_section(f"PREDICTIVE CORRELATION | H={horizon}D | TRAIN")
        print_dict_as_lines(predictive_corr_summary(train_df, target_col=target))

        print_section(f"PREDICTIVE CORRELATION | H={horizon}D | TEST")
        print_dict_as_lines(predictive_corr_summary(test_df, target_col=target))

        print_section(f"INCREMENTAL ALPHA | H={horizon}D | TRAIN")
        print_dict_as_lines(incremental_alpha_summary(train_df, target_col=target))

        print_section(f"INCREMENTAL ALPHA | H={horizon}D | TEST")
        print_dict_as_lines(incremental_alpha_summary(test_df, target_col=target))

        print_section(f"MODEL COMPARISON | H={horizon}D | TRAIN")
        print_dict_as_lines(
            compare_future_only_vs_spot_augmented_summary(train_df, target_col=target)
        )

        print_section(f"MODEL COMPARISON | H={horizon}D | TEST")
        print_dict_as_lines(
            compare_future_only_vs_spot_augmented_summary(test_df, target_col=target)
        )

    print_section("QUANTILE TEST | H=5D | TRAIN")
    print_dict_as_lines(
        quantile_spread_summary(train_df, target_col="future_fut_return_5d", q=5)
    )

    print_section("QUANTILE TEST | H=5D | TEST")
    print_dict_as_lines(
        quantile_spread_summary(test_df, target_col="future_fut_return_5d", q=5)
    )

    print_section("DIRECTION SNAPSHOT | TEST")
    directions_test = build_quantile_directions(test_df)
    export_df = pd.DataFrame({
        "FUTURES_CLOSE": test_df["FUTURES_CLOSE"],
        "SPOT_IMPLIED": test_df["SPOT_IMPLIED"],
        "scarcity_signal": test_df["scarcity_signal"],
        "scarcity_q20_60": test_df["scarcity_q20_60"],
        "scarcity_q80_60": test_df["scarcity_q80_60"],
        "direction": directions_test,
    }).dropna()
    print(export_df.head(10))
    print(f"\nExportable rows for backtest: {len(export_df)}")

    # ------------------------------------------------------------------
    # Backtests without HMM
    # ------------------------------------------------------------------
    print_section("BACKTEST | TEST SAMPLE | DAILY QUANTILE STRATEGY")

    directions_daily = build_quantile_directions(
        test_df,
        signal_col="scarcity_signal",
        low_q_col="scarcity_q20_60",
        high_q_col="scarcity_q80_60",
    )

    bt_daily, bt_daily_df = run_backtest_from_directions(
        df=test_df,
        directions=directions_daily,
        price_col="FUTURES_CLOSE",
        vol_target=0.15,
        vol_lookback=20,
        max_leverage=2.0,
        plot=False,
    )

    print_section("BACKTEST | TEST SAMPLE | 5D HOLDING QUANTILE STRATEGY")

    directions_5d = build_quantile_directions_holding_period(
        test_df,
        signal_col="scarcity_signal",
        low_q_col="scarcity_q20_60",
        high_q_col="scarcity_q80_60",
        holding_period=5,
    )

    bt_5d, bt_5d_df = run_backtest_from_directions(
        df=test_df,
        directions=directions_5d,
        price_col="FUTURES_CLOSE",
        vol_target=0.15,
        vol_lookback=20,
        max_leverage=2.0,
        plot=False,
    )

    print_section("BACKTEST | TEST SAMPLE | 20D HOLDING QUANTILE STRATEGY")

    directions_20d = build_quantile_directions_holding_period(
        test_df,
        signal_col="scarcity_signal",
        low_q_col="scarcity_q20_60",
        high_q_col="scarcity_q80_60",
        holding_period=20,
    )

    bt_20d, bt_20d_df = run_backtest_from_directions(
        df=test_df,
        directions=directions_20d,
        price_col="FUTURES_CLOSE",
        vol_target=0.15,
        vol_lookback=20,
        max_leverage=2.0,
        plot=True,
    )

    print_section("BACKTEST | TEST SAMPLE | 20D OVERLAPPING QUANTILE EXPOSURE")

    exposures_20d = build_overlapping_quantile_exposures(
        test_df,
        signal_col="scarcity_signal",
        low_q_col="scarcity_q20_60",
        high_q_col="scarcity_q80_60",
        holding_period=20,
        normalize=True,
    )

    bt_overlap_20d, bt_overlap_20d_df = run_backtest_from_exposures(
        df=test_df,
        exposures=exposures_20d,
        price_col="FUTURES_CLOSE",
        vol_target=0.15,
        vol_lookback=20,
        max_leverage=2.0,
        plot=True,
    )

    # ------------------------------------------------------------------
    # Additional plots
    # ------------------------------------------------------------------
    print_section("PLOTS | SIGNAL VALIDATION")

    plot_forward_return_by_quantile(
        train_df,
        signal_col="scarcity_signal",
        target_col="future_fut_return_20d",
        q=5,
    )

    plot_forward_return_by_quantile(
        test_df,
        signal_col="scarcity_signal",
        target_col="future_fut_return_20d",
        q=5,
    )

    plot_forward_return_by_quantile_three_panels(
        df_full=df,
        df_train=train_df,
        df_test=test_df,
        signal_col="scarcity_signal",
        target_col="future_fut_return_20d",
        q=5,
    )

    plot_top_bottom_bucket_cumulative(
        test_df,
        signal_col="scarcity_signal",
        target_col="future_fut_return_20d",
        q=5,
    )

    print_section("PLOTS | SIGNAL STABILITY")

    plot_rolling_ic(
        train_df,
        signal_col="scarcity_signal",
        target_col="future_fut_return_20d",
        window=60,
    )

    plot_rolling_ic(
        test_df,
        signal_col="scarcity_signal",
        target_col="future_fut_return_20d",
        window=60,
    )

    # ------------------------------------------------------------------
    # hmmlearn filtered backtest
    # ------------------------------------------------------------------
    print_section("BACKTEST | TEST SAMPLE | 20D OVERLAPPING EXPOSURE WITH HMMLEARN FILTER")

    exposures_hmmlearn = build_hmm_filtered_overlapping_exposures(
        test_hmm_df,
        good_states=good_states,
        state_col="state",
        signal_col="scarcity_signal",
        low_q_col="scarcity_q20_60",
        high_q_col="scarcity_q80_60",
        holding_period=20,
        normalize=True,
    )

    bt_hmmlearn, bt_hmmlearn_df = run_backtest_from_exposures(
        df=test_hmm_df,
        exposures=exposures_hmmlearn,
        price_col="FUTURES_CLOSE",
        vol_target=0.15,
        vol_lookback=20,
        max_leverage=2.0,
        plot=True,
    )


if __name__ == "__main__":
    main()
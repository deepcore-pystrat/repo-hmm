"""
HMM regime trading strategy -- sugar markets.
Gaussian HMM variant.

Pipeline:  data -> features -> train/test -> fit HMM -> filter -> map regimes -> backtest
"""

import numpy as np
import pandas as pd

from data_loader.loader_data import DataConfig
from data_loader.laoding_services import get_sugar_data, load_futures_sugar
from features.registry import build_features, standardize
from features.splitter import fixed_split
from models import GaussianHMMConfig, create_hmm, map_regimes
from simulations.plotting import (
    plot_features,
    plot_hmm_dashboard,
    print_regime_summary,
)
from simulations.backtester import backtest, print_metrics, plot_backtest
from validation.diagnostics import print_diagonal_diagnostic


# ── Data loading ──────────────────────────────────────────────────────

def load_sugar_data() -> pd.DataFrame:
    cfg_vhp  = DataConfig(data_path="data/data_spot_VHP.xlsx",     asset_name="VHP")
    cfg_thp  = DataConfig(data_path="data/data_spot_THP.xlsx",     asset_name="THP")
    cfg_sb11 = DataConfig(data_path="data/data_futures_sb11.xlsx", asset_name="SB11")

    df = pd.concat(
        [get_sugar_data(cfg_vhp), get_sugar_data(cfg_thp), load_futures_sugar(cfg_sb11)],
        axis=1,
    ).dropna()

    df['SPREAD_VHP_THP'] = df['MID_VHP'] - df['MID_THP']

    # print(df)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # # plt.bar(df.index, df['Volume'])
    # plt.plot(df.index, df['MID_VHP'].pct_change().rolling(20).mean(), label='MID_VHP_PCT_CHANGE')
    # plt.plot(df.index, df['MID_VHP'].diff().rolling(20).mean(), label='MID_VHP_DIFF')

    # # plt.plot(df.index, _trend(df['MID_VHP'], 10, 100), label='MID_VHP_TREND')
    # # plt.plot(df.index, df['MID_VHP'].rolling(100).mean() - df['MID_THP'].rolling(100).mean(), label='MID_VHP')
    # # plt.plot(df.index, df['MID_THP'].rolling(100).mean(), label='MID_THP')

    # # plt.plot(df.index, df['MID_VHP'].diff(), label='MID_VHP')
    # # plt.plot(df.index, df['MID_THP'].diff(), label='MID_THP')
    # # plt.plot(df.index, df['MID_VHP'].diff() - df['MID_THP'].diff() - df['SPREAD_VHP_THP'], label='MID_VHP_MINUS_MID_THP')

    # # plt.plot(df.index, df['SPREAD_VHP_THP'].diff(), label='SPREAD_VHP_THP')

    # plt.legend()
    # plt.show()

    return df


# ── Pipeline ──────────────────────────────────────────────────────────

def main() -> None:



    nb_states=2

    # ── Configuration ─────────────────────────────────────────────────
    feature_specs = [
        # ── Close ─────────────────────────────────────────────
        # ("returns",    "Close"),
        ("id", "Volume"),
        # ("diff", "Volume"),

        # ("zscore",     "Volume"),

        # ("rvol",       "Close", {"window": 20}),
        # ("momentum",   "Close", {"period": 10}),
        # ("rsi",        "Close", {"period": 14}),
        # ("zscore",     "Close", {"window": 20}),
        # ("ma_ratio",   "Close", {"fast": 10, "slow": 50}),
        # ("rvol_ratio", "Close", {"fast": 5, "slow": 20}),
        # ("skew",       "Close", {"window": 42}),
        # ("kurt",       "Close", {"window": 42}),
        # ("atr",        "Close", {"window": 14}),
        # ("log_range",  "Close", {"window": 5}),

        # ── SPREAD_VHP_THP ────────────────────────────────────
        # ("delta", "SPREAD_VHP_THP"),
        # ("rvol",       "SPREAD_VHP_THP", {"window": 42}),
        # ("ma", "MID_VHP", {"window": 20}),
        # ("log_returns", "SPREAD_VHP_THP"),
        # ("zscore",     "SPREAD_VHP_THP", {"window": 21}),
        # ("skew",       "SPREAD_VHP_THP", {"window": 21}),
        # ("kurt",       "SPREAD_VHP_THP", {"window": 21}),
        # ── MID_VHP ──────────────────────────────────────────
        # ("returns",    "MID_VHP"),
        ("diff_ma",    "MID_VHP", {"period": 14, "window": 21}),

        # ("rvol",       "MID_VHP", {"window": 20}),
        # ("momentum", "MID_VHP", {"period": 10}),
        # ("skew", "MID_VHP", {"window": 42}),
        # ("kurt", "MID_VHP", {"window": 21}),

        # ── MID_THP ──────────────────────────────────────────
        # ("returns",    "MID_THP"),
        ("diff_ma", "MID_THP", {"period": 14, "window": 21}),

        # ("rvol",       "MID_THP", {"window": 20}),
        # ("momentum",   "MID_THP", {"period": 10}),
    ]

 

    # ── 1) Load data ──────────────────────────────────────────────────
    print("Loading data ...")
    data_df = load_sugar_data()
    print(f"  {data_df.shape}  |  {data_df.index[0]} -> {data_df.index[-1]}")

    # ── 2) Build features ─────────────────────────────────────────────
    print("\nBuilding features ...")
    feat_df = build_features(data_df, feature_specs, shift_cols={"Close": 1, 'Volume': 1})

    print(feat_df)

    X, idx = feat_df.values, feat_df.index
    names  = list(feat_df.columns)
    print(f"  {X.shape}  |  {idx[0]} -> {idx[-1]}")

    print(feat_df)



    # ── 3) Train / test split + standardization ───────────────────────
    print("\nSplitting train / test ...")
    X_train, X_test, idx_train, idx_test = fixed_split(X, idx, train_ratio=0.80)
    X_train_z, X_test_z, feat_mu, feat_sd = standardize(X_train, X_test)

    print(f"  Train: {X_train.shape}  |  {idx_train[0]} -> {idx_train[-1]}")
    print(f"  Test:  {X_test.shape}  |  {idx_test[0]} -> {idx_test[-1]}")
    print(f"  Train Z mean: {X_train_z.mean(axis=0).round(4)}")
    print(f"  Train Z std:  {X_train_z.std(axis=0, ddof=1).round(4)}")

    # ── 4) Fit HMM ───────────────────────────────────────────────────
    print("\nFitting Gaussian HMM ...")
    # hmm_cfg = GaussianHMMConfig(K=2, n_iter=100, seed=42)
    hmm_cfg = GaussianHMMConfig(K=nb_states, n_iter=100, seed=42, cov_type="full")

    model = create_hmm(hmm_cfg)
    result = model.fit(X_train_z)

    print(f"  Final LL: {result.ll_hist[-1]:.2f}  ({len(result.ll_hist)} iterations)")
    print(f"  pi:  {np.round(result.pi, 3)}")
    print(f"  A:\n{np.round(result.A, 3)}")

    print_regime_summary(result, feat_mu, feat_sd, names)


    # ── 5) Regime mapping on train ────────────────────────────────────
    print("\nMapping regimes on TRAIN (forward-only) ...")
    states_train, alpha_train, ll_train = model.filter(X_train_z)

    # Compute futures returns aligned with train dates
    p_train = data_df.loc[idx_train, "Close"].astype(float).values[:len(states_train)]
    r_train = np.zeros(len(p_train))
    r_train[1:] = p_train[1:] / p_train[:-1] - 1.0

    position_map = map_regimes(states_train, r_train, flat_threshold=0.20)

    # print_diagonal_diagnostic(X_train_z, states_train, names, threshold=0.30)

    _label = {1: "LONG", -1: "SHORT", 0: "FLAT"}
    for s, pos in sorted(position_map.items()):
        mask_s = states_train == s
        r_s = r_train[mask_s]
        mean_ann = np.nanmean(r_s) * 252
        std_ann = np.nanstd(r_s, ddof=1) * np.sqrt(252) if mask_s.sum() > 1 else 0
        sharpe = mean_ann / std_ann if std_ann > 0 else 0
        print(f"  State {s} -> {_label[pos]:>5s}  |  "
              f"Sharpe={sharpe:+.2f}  mean_ann={mean_ann:+.2%}  n={mask_s.sum()}")

    # ── 6) Forward-only filtering on test ─────────────────────────────
    print("\nFiltering on TEST (forward-only, causal) ...")
    states_test, alpha_test, ll_test = model.filter(X_test_z)
    print(f"  Test LL: {ll_test:.2f}  |  avg/step: {ll_test / len(X_test_z):.4f}")

    # Compute futures returns aligned with test dates
    p_test_full = data_df.loc[idx_test, "Close"].astype(float).values[:len(states_test)]
    r_test = np.zeros(len(p_test_full))
    r_test[1:] = p_test_full[1:] / p_test_full[:-1] - 1.0

    unique, counts = np.unique(states_test, return_counts=True)
    for s, c in zip(unique, counts):
        mask_s = states_test == s
        r_s = r_test[mask_s]
        mean_ann = np.nanmean(r_s) * 252
        std_ann = np.nanstd(r_s, ddof=1) * np.sqrt(252) if mask_s.sum() > 1 else 0
        sharpe = mean_ann / std_ann if std_ann > 0 else 0
        pos = position_map.get(s, 0)
        print(f"  State {s} -> {_label[pos]:>5s}  |  "
              f"Sharpe={sharpe:+.2f}  mean_ann={mean_ann:+.2%}  n={c} ({c / len(states_test):.0%})")

    # ── 7) Map states -> positions ─────────────────────────────────────
    directions = np.array([position_map.get(s, 0) for s in states_test])
    print(f"\n  Positions:  LONG={np.sum(directions == 1)}  "
          f"SHORT={np.sum(directions == -1)}  FLAT={np.sum(directions == 0)}")

    # Plots
    p_test = data_df.loc[idx_test, "Close"].astype(float).values
    vol_test = data_df.loc[idx_test, "Volume"].astype(float).values
    rvol_test = pd.Series(p_test).pct_change().rolling(20).std().values * np.sqrt(252)
    plot_hmm_dashboard(
        ll_hist=result.ll_hist,
        dates=idx_test,
        alpha=alpha_test,
        prices=p_test,
        states=states_test,
        volume=vol_test,
        volatility=rvol_test,
        title="Gaussian HMM — Dashboard",
    )

    # ── 8) Backtest ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BACKTEST — Gaussian HMM (no look-ahead)")
    print("=" * 60)

    bt = backtest(
        prices=p_test,
        states=states_test,
        position_map=position_map,
        vol_target=0.15,
        vol_lookback=20,
        max_leverage=2.0,
    )
    print_metrics(bt)
    plot_backtest(bt, idx_test, p_test, states_test)


if __name__ == "__main__":
    main()

"""
HMM regime analysis on spot-futures spread.

Pipeline:
- load spot data
- load futures data
- align by date
- build log spread
- build spread features
- split train/test
- standardize
- fit HMM
- infer regimes on test
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import pearsonr, ttest_ind

from data_loader.loader_data import DataConfig
from data_loader.laoding_services import get_bid_offer_data
from features.registry import standardize
from features.splitter import fixed_split
from models import StudentTHMMConfig, create_hmm


def correlation_by_regime(
    df: pd.DataFrame,
    signal_col: str = "storage_signal",
    ret_col: str = "dlog_spread",
) -> None:
    df = df.copy()
    df["future_return"] = df[ret_col].shift(-1)

    print("\nCORRELATION BY REGIME")

    df = df[[signal_col, "future_return", "state"]].dropna()

    corr, pval = pearsonr(df[signal_col], df["future_return"])
    print("\nGLOBAL:")
    print(f"  corr  = {corr:.4f}")
    print(f"  p-val = {pval:.4e}")
    print(f"  n     = {len(df)}")

    for s in sorted(df["state"].unique()):
        sub = df[df["state"] == s]

        if len(sub) < 20:
            print(f"\nState {s}: not enough data (n={len(sub)})")
            continue

        corr, pval = pearsonr(sub[signal_col], sub["future_return"])
        print(f"\nState {s}:")
        print(f"  corr  = {corr:.4f}")
        print(f"  p-val = {pval:.4e}")
        print(f"  n     = {len(sub)}")


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
        sep=";"
    ).copy()

    fut["Date"] = pd.to_datetime(fut["Date"], errors="coerce")
    fut["Close"] = pd.to_numeric(fut["Close"], errors="coerce")

    fut = fut.dropna(subset=["Date", "Close"]).copy()
    fut = fut.sort_values("Date")
    fut = fut.groupby("Date", as_index=False).last()
    fut = fut.rename(columns={"Close": "FUTURES_CLOSE"})
    fut = fut.set_index("Date")

    return fut


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
    df = df[(df["MID"] > 0) & (df["FUTURES_CLOSE"] > 0)].copy()

    df["log_spot"] = np.log(df["MID"])
    df["log_fut"] = np.log(df["FUTURES_CLOSE"])

    df["logspread"] = df["log_spot"] - df["log_fut"]
    df["spread_level"] = df["MID"] - df["FUTURES_CLOSE"]

    df["dlog_spread"] = df["logspread"].diff()
    df["dlog_spot"] = df["log_spot"].diff()
    df["dlog_fut"] = df["log_fut"].diff()

    # cibles business: rendements futurs du future
    df["future_fut_return_1d"] = df["dlog_fut"].shift(-1)
    df["future_fut_return_5d"] = df["log_fut"].shift(-5) - df["log_fut"]
    df["future_fut_return_20d"] = df["log_fut"].shift(-20) - df["log_fut"]

    return df


def print_basic_diagnostics(df: pd.DataFrame) -> None:
    print("\nDATASET")
    print(f"Shape: {df.shape}")
    print(f"Start: {df.index.min()}")
    print(f"End:   {df.index.max()}")

    cols = ["MID", "FUTURES_CLOSE", "logspread", "spread_level", "dlog_spread"]
    print("\nHEAD")
    print(df[cols].head())

    print("\nDESCRIPTIVE STATS")
    print(df[cols].describe().round(6))

    print("\nMISSING VALUES")
    print(df[cols].isna().sum())


def plot_spread(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(df.index, df["logspread"], color="black", linewidth=1.2)
    axes[0].set_title("Log Spread = log(MID) - log(FUTURES_CLOSE)")
    axes[0].set_ylabel("logspread")

    axes[1].plot(df.index, df["spread_level"], color="tab:blue", linewidth=1.0)
    axes[1].set_title("Level Spread = MID - FUTURES_CLOSE")
    axes[1].set_ylabel("spread_level")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()


def build_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["dlog_spread"] = out["logspread"].diff()
    out["spread_vol_20"] = out["dlog_spread"].rolling(20).std()
    out["spread_momentum_20"] = out["logspread"] - out["logspread"].shift(20)

    out["spread_zscore_20"] = (
        (out["logspread"] - out["logspread"].rolling(20).mean()) /
        out["logspread"].rolling(20).std()
    )

    out["spread_return"] = out["logspread"].diff()
    out["spread_slope"] = out["logspread"] - out["logspread"].shift(1)

    # Signaux économiques
    out["scarcity_signal"] = out["logspread"]      # élevé = spot riche vs future
    out["storage_signal"] = -out["logspread"]      # élevé = abondance/stockage

    out["scarcity_change"] = out["scarcity_signal"].diff()
    out["storage_change"] = out["storage_signal"].diff()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().copy()
    return out


def predictive_signal_test(df: pd.DataFrame) -> None:
    df = df.copy()
    df["future_return"] = df["dlog_spread"].shift(-1)

    print("\nSIGNAL QUALITY TEST")

    signal = "storage_signal"
    corr = df[[signal, "future_return"]].corr().iloc[0, 1]
    print(f"Global corr(signal, future_return): {corr:.4f}")

    for s in sorted(df["state"].unique()):
        sub = df[df["state"] == s]
        corr = sub[[signal, "future_return"]].corr().iloc[0, 1]
        print(f"State {s}: corr = {corr:.4f}")


def predictive_futures_test(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    target_col: str = "future_fut_return_1d",
) -> None:
    tmp = df[[signal_col, target_col, "state"]].dropna().copy()

    print("\nPREDICTIVE TEST ON FUTURES RETURNS")
    print(f"Signal: {signal_col}")
    print(f"Target: {target_col}")

    corr, pval = pearsonr(tmp[signal_col], tmp[target_col])
    print("\nGLOBAL:")
    print(f"  corr  = {corr:.4f}")
    print(f"  p-val = {pval:.4e}")
    print(f"  n     = {len(tmp)}")

    for s in sorted(tmp["state"].unique()):
        sub = tmp[tmp["state"] == s]
        if len(sub) < 20:
            print(f"\nState {s}: not enough data (n={len(sub)})")
            continue

        corr, pval = pearsonr(sub[signal_col], sub[target_col])
        print(f"\nState {s}:")
        print(f"  corr  = {corr:.4f}")
        print(f"  p-val = {pval:.4e}")
        print(f"  n     = {len(sub)}")


def incremental_alpha_test(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    target_col: str = "future_fut_return_1d",
) -> None:
    tmp = df.copy()

    tmp["past_fut_5d"] = tmp["log_fut"] - tmp["log_fut"].shift(5)
    tmp["past_fut_20d"] = tmp["log_fut"] - tmp["log_fut"].shift(20)

    cols = [signal_col, target_col, "past_fut_5d", "past_fut_20d"]
    tmp = tmp[cols].dropna().copy()

    y = tmp[target_col]
    X = tmp[[signal_col, "past_fut_5d", "past_fut_20d"]]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HC3")

    print("\nINCREMENTAL ALPHA TEST")
    print(f"Target: {target_col}")
    print(model.summary())


def quantile_forward_return_test(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    target_col: str = "future_fut_return_5d",
    q: int = 5,
) -> None:
    tmp = df[[signal_col, target_col]].dropna().copy()

    tmp["bucket"] = pd.qcut(tmp[signal_col], q=q, labels=False, duplicates="drop")

    summary = tmp.groupby("bucket")[target_col].agg(
        mean="mean",
        std="std",
        count="count",
        median="median"
    )

    print("\nQUANTILE FORWARD RETURN TEST")
    print(f"Signal: {signal_col}")
    print(f"Target: {target_col}")
    print(summary.round(6))

    low_bucket = tmp["bucket"].min()
    high_bucket = tmp["bucket"].max()

    high_mean = tmp.loc[tmp["bucket"] == high_bucket, target_col].mean()
    low_mean = tmp.loc[tmp["bucket"] == low_bucket, target_col].mean()

    print(f"\nTop bucket mean    : {high_mean:.6f}")
    print(f"Bottom bucket mean : {low_mean:.6f}")
    print(f"Spread (Top-Bottom): {high_mean - low_mean:.6f}")


def quantile_spread_test(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    target_col: str = "future_fut_return_5d",
    q: int = 5,
) -> None:
    tmp = df[[signal_col, target_col]].dropna().copy()
    tmp["bucket"] = pd.qcut(tmp[signal_col], q=q, labels=False, duplicates="drop")

    low_bucket = tmp["bucket"].min()
    high_bucket = tmp["bucket"].max()

    low = tmp.loc[tmp["bucket"] == low_bucket, target_col]
    high = tmp.loc[tmp["bucket"] == high_bucket, target_col]

    stat, pval = ttest_ind(high, low, equal_var=False, nan_policy="omit")

    print("\nQUANTILE SPREAD TEST")
    print(f"Signal: {signal_col}")
    print(f"Target: {target_col}")
    print(f"Top mean         : {high.mean():.6f}")
    print(f"Bottom mean      : {low.mean():.6f}")
    print(f"Top-Bottom       : {high.mean() - low.mean():.6f}")
    print(f"t-stat           : {stat:.4f}")
    print(f"p-value          : {pval:.4e}")
    print(f"n_top / n_bottom : {len(high)} / {len(low)}")


def quantile_forward_return_test_by_regime(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    target_col: str = "future_fut_return_5d",
    state_col: str = "state",
    q: int = 5,
) -> None:
    print("\nQUANTILE FORWARD RETURN TEST BY REGIME")

    for s in sorted(df[state_col].dropna().unique()):
        sub = df[df[state_col] == s].copy()
        sub = sub[[signal_col, target_col]].dropna()

        if len(sub) < max(30, q * 10):
            print(f"\nState {s}: not enough data (n={len(sub)})")
            continue

        sub["bucket"] = pd.qcut(sub[signal_col], q=q, labels=False, duplicates="drop")

        summary = sub.groupby("bucket")[target_col].agg(
            mean="mean",
            std="std",
            count="count"
        )

        print(f"\nState {s}")
        print(summary.round(6))

        low_bucket = sub["bucket"].min()
        high_bucket = sub["bucket"].max()

        high_mean = sub.loc[sub["bucket"] == high_bucket, target_col].mean()
        low_mean = sub.loc[sub["bucket"] == low_bucket, target_col].mean()

        print(f"Top-Bottom spread: {high_mean - low_mean:.6f}")


def compare_future_only_vs_spot_augmented(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    target_col: str = "future_fut_return_5d",
) -> None:
    tmp = df.copy()

    tmp["past_fut_5d"] = tmp["log_fut"] - tmp["log_fut"].shift(5)
    tmp["past_fut_20d"] = tmp["log_fut"] - tmp["log_fut"].shift(20)

    cols = [signal_col, target_col, "past_fut_5d", "past_fut_20d"]
    tmp = tmp[cols].dropna().copy()

    y = tmp[target_col]

    X1 = tmp[["past_fut_5d", "past_fut_20d"]]
    X1 = sm.add_constant(X1)
    m1 = sm.OLS(y, X1).fit(cov_type="HC3")

    X2 = tmp[[signal_col, "past_fut_5d", "past_fut_20d"]]
    X2 = sm.add_constant(X2)
    m2 = sm.OLS(y, X2).fit(cov_type="HC3")

    print("\nMODEL COMPARISON: FUTURE-ONLY VS FUTURE+SPOT")
    print(f"Target: {target_col}")

    print("\n[Future-only]")
    print(f"R²       : {m1.rsquared:.6f}")
    print(f"Adj R²   : {m1.rsquared_adj:.6f}")
    print(f"AIC      : {m1.aic:.2f}")
    print(f"BIC      : {m1.bic:.2f}")

    print("\n[Future + spot]")
    print(f"R²       : {m2.rsquared:.6f}")
    print(f"Adj R²   : {m2.rsquared_adj:.6f}")
    print(f"AIC      : {m2.aic:.2f}")
    print(f"BIC      : {m2.bic:.2f}")

    print("\n[Incremental contribution of spot]")
    print(f"Delta R² : {m2.rsquared - m1.rsquared:.6f}")

    if signal_col in m2.params.index:
        print(f"{signal_col} coef   : {m2.params[signal_col]:.6f}")
        print(f"{signal_col} t-stat : {m2.tvalues[signal_col]:.4f}")
        print(f"{signal_col} p-val  : {m2.pvalues[signal_col]:.4e}")


def compare_future_only_vs_spot_augmented_by_regime(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    target_col: str = "future_fut_return_5d",
    state_col: str = "state",
) -> None:
    print("\nMODEL COMPARISON BY REGIME")

    for s in sorted(df[state_col].dropna().unique()):
        sub = df[df[state_col] == s].copy()

        if len(sub) < 50:
            print(f"\nState {s}: not enough data (n={len(sub)})")
            continue

        print(f"\n===== State {s} =====")
        compare_future_only_vs_spot_augmented(
            sub,
            signal_col=signal_col,
            target_col=target_col,
        )


def simple_strategy(
    df: pd.DataFrame,
    signal_col: str = "scarcity_signal",
    ret_col: str = "future_fut_return_1d",
    state_col: str = "state",
    target_state: int = 1,
) -> pd.DataFrame:
    out = df.copy()

    # rendement futur de la cible choisie
    out["future_return"] = out[ret_col].shift(-1)

    out["position"] = 0.0

    mask = out[state_col] == target_state
    out.loc[mask, "position"] = np.sign(out.loc[mask, signal_col])

    out["pnl"] = out["position"] * out["future_return"]

    out = out.dropna(subset=["future_return", "position", "pnl"]).copy()

    mean_pnl = out["pnl"].mean()
    std_pnl = out["pnl"].std(ddof=1)
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else np.nan
    hit_ratio = (out["pnl"] > 0).mean()
    active_rate = (out["position"] != 0).mean()

    print("\nSIMPLE STRATEGY RESULTS")
    print(f"Target state   : {target_state}")
    print(f"N obs          : {len(out)}")
    print(f"Active rate    : {active_rate:.4f}")
    print(f"Mean PnL       : {mean_pnl:.6f}")
    print(f"Std PnL        : {std_pnl:.6f}")
    print(f"Sharpe         : {sharpe:.4f}")
    print(f"Hit ratio      : {hit_ratio:.4f}")

    return out


def main() -> None:
    nb_states = 2

    print("Building spot-futures spread dataset ...")
    df = build_spread_dataset()

    print_basic_diagnostics(df)
    plot_spread(df)

    print("\nBuilding spread features ...")
    df_feat = build_spread_features(df)

    feature_cols = [
        "logspread",
        "dlog_spread",
        "spread_vol_20",
        "spread_zscore_20",
    ]

    print(df_feat[feature_cols].head())
    print(f"Features shape: {df_feat[feature_cols].shape}")

    X = df_feat[feature_cols].values
    idx = df_feat.index

    print("\nSplitting train / test ...")
    X_train, X_test, idx_train, idx_test = fixed_split(X, idx, train_ratio=0.80)
    X_train_z, X_test_z, mu, sd = standardize(X_train, X_test)

    print(f"Train: {X_train.shape} | {idx_train[0]} -> {idx_train[-1]}")
    print(f"Test:  {X_test.shape} | {idx_test[0]} -> {idx_test[-1]}")
    print(f"Train Z mean: {X_train_z.mean(axis=0).round(4)}")
    print(f"Train Z std:  {X_train_z.std(axis=0, ddof=1).round(4)}")

    print("\nFitting Student HMM ...")
    hmm_cfg = StudentTHMMConfig(K=nb_states, n_iter=100, seed=42)
    model = create_hmm(hmm_cfg)

    result = model.fit_multi_start(
        X_train_z,
        seeds=[0, 1, 2, 42, 123],
        verbose=True,
    )

    states_train, alpha_train, ll_train = model.filter(X_train_z)

    print("Train state counts:", np.bincount(states_train, minlength=nb_states))
    train_summary = df_feat.loc[idx_train].copy()
    train_summary = train_summary.iloc[:len(states_train)].copy()
    train_summary["state"] = states_train

    # ===============================
    # TRAIN
    # ===============================
    train_summary["spread_return"] = train_summary["dlog_spread"]

    perf = train_summary.groupby("state")["spread_return"].agg(
        mean="mean",
        std="std",
        median="median",
        count="count"
    )
    perf["sharpe"] = perf["mean"] / perf["std"] * np.sqrt(252)

    print("\nREGIME PERFORMANCE (TRAIN)")
    print(perf.round(6))

    correlation_by_regime(train_summary)
    predictive_signal_test(train_summary)

    predictive_futures_test(
        train_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_1d",
    )
    predictive_futures_test(
        train_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
    )
    predictive_futures_test(
        train_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_20d",
    )

    incremental_alpha_test(
        train_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_1d",
    )
    incremental_alpha_test(
        train_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
    )

    compare_future_only_vs_spot_augmented(
        train_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
    )
    compare_future_only_vs_spot_augmented(
        train_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_20d",
    )

    quantile_forward_return_test(
        train_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
        q=5,
    )
    quantile_spread_test(
        train_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
        q=5,
    )

    quantile_forward_return_test_by_regime(
        train_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
        state_col="state",
        q=5,
    )

    compare_future_only_vs_spot_augmented_by_regime(
        train_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
        state_col="state",
    )

    print("\nSTORAGE SIGNAL BY REGIME (TRAIN)")
    print(
        train_summary.groupby("state")[[
            "storage_signal",
            "storage_change"
        ]].agg(["mean", "std", "median", "count"]).round(4)
    )

    print("\nTRAIN SPREAD REGIME SUMMARY")
    print(
        train_summary.groupby("state")[[
            "logspread",
            "dlog_spread",
            "spread_vol_20",
            "spread_momentum_20",
            "spread_zscore_20",
        ]].agg(["mean", "std", "median", "count"]).round(4)
    )

    train_summary["future_dlog_spread"] = train_summary["dlog_spread"].shift(-1)

    print("\nMEAN REVERSION TEST (TRAIN)")
    for s in sorted(train_summary["state"].unique()):
        mask = train_summary["state"] == s
        future_mean = train_summary.loc[mask, "future_dlog_spread"].mean()
        current_z = train_summary.loc[mask, "spread_zscore_20"].mean()

        print(f"State {s}:")
        print(f"  avg zscore        = {current_z:+.3f}")
        print(f"  next spread move  = {future_mean:+.5f}")

        if current_z > 0 and future_mean < 0:
            print("  → mean reversion detected (rich → down)")
        elif current_z < 0 and future_mean > 0:
            print("  → mean reversion detected (cheap → up)")
        else:
            print("  → NO mean reversion (momentum or noise)")

    # ===============================
    # TEST
    # ===============================
    print("\nFiltering on TEST ...")
    states_test, alpha_test, ll_test = model.filter(X_test_z)
    states_smooth, gamma_test, _ = model.smooth(X_test_z)
    states_vit = model.viterbi(X_test_z)

    test_summary = df_feat.loc[idx_test].copy()
    test_summary = test_summary.iloc[:len(states_test)].copy()
    test_summary["state"] = states_test
    test_summary["spread_return"] = test_summary["dlog_spread"]

    perf_test = test_summary.groupby("state")["spread_return"].agg(
        mean="mean",
        std="std",
        median="median",
        count="count"
    )
    perf_test["sharpe"] = perf_test["mean"] / perf_test["std"] * np.sqrt(252)

    print("\nREGIME PERFORMANCE (TEST)")
    print(perf_test.round(6))

    correlation_by_regime(test_summary)
    predictive_signal_test(test_summary)

    predictive_futures_test(
        test_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_1d",
    )
    predictive_futures_test(
        test_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
    )
    predictive_futures_test(
        test_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_20d",
    )

    incremental_alpha_test(
        test_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_1d",
    )
    incremental_alpha_test(
        test_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
    )

    compare_future_only_vs_spot_augmented(
        test_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
    )
    compare_future_only_vs_spot_augmented(
        test_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_20d",
    )

    quantile_forward_return_test(
        test_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
        q=5,
    )
    quantile_spread_test(
        test_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
        q=5,
    )

    quantile_forward_return_test_by_regime(
        test_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
        state_col="state",
        q=5,
    )

    compare_future_only_vs_spot_augmented_by_regime(
        test_summary,
        signal_col="scarcity_signal",
        target_col="future_fut_return_5d",
        state_col="state",
    )

    print("\nTEST SPREAD REGIME SUMMARY")
    print(
        test_summary.groupby("state")[[
            "logspread",
            "dlog_spread",
            "spread_vol_20",
            "spread_momentum_20",
            "spread_zscore_20",
        ]].agg(["mean", "std", "median", "count"]).round(4)
    )

    print("Filter state counts:", np.bincount(states_test, minlength=nb_states))
    print("Smooth state counts:", np.bincount(states_smooth, minlength=nb_states))
    print("Viterbi state counts:", np.bincount(states_vit, minlength=nb_states))

    print(f"\nFinal train LL: {result.ll_hist[-1]:.2f}")
    print(f"pi: {np.round(result.pi, 3)}")
    print(f"A:\n{np.round(result.A, 3)}")
    print(f"Test LL: {ll_test:.2f} | avg/step: {ll_test / len(X_test_z):.4f}")

    print("\nFirst 20 inferred states on TEST")
    for i in range(min(20, len(states_test))):
        print(
            idx_test[i],
            "| filt:", states_test[i],
            "| smooth:", states_smooth[i],
            "| vit:", states_vit[i],
            "| alpha:", np.round(alpha_test[i], 3),
            "| gamma:", np.round(gamma_test[i], 3),
        )

    simple_strategy(
        test_summary,
        signal_col="scarcity_signal",
        ret_col="future_fut_return_1d",
        state_col="state",
        target_state=1,
    )


if __name__ == "__main__":
    main()
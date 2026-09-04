import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from arch import arch_model
from features.registry import standardize
from models import StudentTHMMConfig, GaussianHMMConfig, create_hmm
import validation.regime_causality_analysis as rc

from scipy import stats
from scipy.stats import binomtest
from itertools import combinations


# ============================================================
# USER CONFIG
# ============================================================

SPOT_PATH = r"D:\Downloads\deepcore--2026-04-07-12_42-0026205d-213a-4e96-bbe9-a7c977a1c3d4.xlsx"
FUTURES_PATH = r"D:\Downloads\Futures CBOT Corn.csv"

NB_STATES = 4

HMM_TYPE = "student"  # "student" or "gaussian"
COV_TYPE = "diag"     # "full" or "diag"

N_ITER = 100
SEED = 42
MULTI_START_SEEDS = [0, 1, 2, 42, 123]

TRAIN_RATIO = 0.7

HMM_WINDOW = 20
RSI_WINDOW = 14


# ============================================================
# PREDICTIVE ANALYSIS: E[r_{t+1} | state_t]
# ============================================================

def analyze_future_return_by_state(
    df: pd.DataFrame,
    state_col: str = "state",
    return_col: str = "future_return",
    annualization: int = 252,
):
    """
    Analyse si le régime courant state_t contient de l'information
    sur le rendement futur stocké dans return_col.
    """

    x = df[[state_col, return_col]].dropna().copy()
    x[state_col] = x[state_col].astype(int)

    summary = (
        x.groupby(state_col)[return_col]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    q25 = x.groupby(state_col)[return_col].quantile(0.25).rename("q25")
    q75 = x.groupby(state_col)[return_col].quantile(0.75).rename("q75")
    hit = (
        x.groupby(state_col)[return_col]
        .apply(lambda s: (s > 0).mean())
        .rename("hit_ratio_pos")
    )

    summary = summary.merge(q25.reset_index(), on=state_col, how="left")
    summary = summary.merge(q75.reset_index(), on=state_col, how="left")
    summary = summary.merge(hit.reset_index(), on=state_col, how="left")

    summary["sharpe_daily"] = summary["mean"] / summary["std"].replace(0, np.nan)
    summary["sharpe_annualized"] = np.sqrt(annualization) * summary["sharpe_daily"]

    # Tests globaux
    groups = [g[return_col].values for _, g in x.groupby(state_col)]

    anova_res = None
    kruskal_res = None

    groups_valid = [g for g in groups if len(g) > 1]

    states = sorted(x[state_col].unique())
    pairwise_rows = []


    if len(groups_valid) >= 2:
        f_stat, f_p = stats.f_oneway(*groups_valid)
        h_stat, h_p = stats.kruskal(*groups_valid)

        anova_res = {
            "F_stat"  : float(f_stat),
            "p_value" : float(f_p),
            "n_groups": len(groups_valid),
            "note"    : f"{len(groups) - len(groups_valid)} groupe(s) ignoré(s) (n<2)",
        }
        kruskal_res = {
            "H_stat"  : float(h_stat),
            "p_value" : float(h_p),
            "n_groups": len(groups_valid),
        }
        # Tests pairwise
        

    for s1, s2 in combinations(states, 2):
        r1 = x.loc[x[state_col] == s1, return_col].values
        r2 = x.loc[x[state_col] == s2, return_col].values

        if len(r1) < 2 or len(r2) < 2:
            continue

        t_stat, t_p = stats.ttest_ind(
            r1,
            r2,
            equal_var=False,
            nan_policy="omit",
        )

        try:
            u_stat, u_p = stats.mannwhitneyu(
                r1,
                r2,
                alternative="two-sided",
            )
        except Exception:
            u_stat, u_p = np.nan, np.nan

        pairwise_rows.append({
            "state_1": s1,
            "state_2": s2,
            "mean_1": np.mean(r1),
            "mean_2": np.mean(r2),
            "mean_diff": np.mean(r1) - np.mean(r2),
            "t_stat": float(t_stat),
            "t_pvalue": float(t_p),
            "mw_u_stat": float(u_stat) if pd.notna(u_stat) else np.nan,
            "mw_pvalue": float(u_p) if pd.notna(u_p) else np.nan,
            "n1": len(r1),
            "n2": len(r2),
        })

    pairwise_df = pd.DataFrame(pairwise_rows)

    if not pairwise_df.empty:
        pairwise_df = pairwise_df.sort_values("t_pvalue", na_position="last")

    return {
        "summary": summary.sort_values("mean").reset_index(drop=True),
        "anova": anova_res,
        "kruskal": kruskal_res,
        "pairwise": pairwise_df,
    }


def print_future_return_analysis(results: dict, title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print("\nSummary by regime:")
    print(results["summary"].to_string(index=False))

    print("\nGlobal tests:")
    print("ANOVA :", results["anova"])
    print("Kruskal :", results["kruskal"])

    print("\nPairwise tests:")
    if results["pairwise"].empty:
        print("No pairwise results.")
    else:
        print(results["pairwise"].to_string(index=False))


def build_predictive_df(
    data_df: pd.DataFrame,
    states: np.ndarray,
    target_col: str = "FUTURES_CLOSE",
    state_col: str = "state",
    use_log_return: bool = True,
    horizon: int = 1,
) -> pd.DataFrame:

    out = data_df.copy()
    out[state_col] = states

    x = pd.to_numeric(out[target_col], errors="coerce")

    if use_log_return:
        out["future_return"] = np.log(x.shift(-horizon) / x)
    else:
        out["future_return"] = x.shift(-horizon) - x

    out["horizon"] = horizon

    return out

# ============================================================
# VOLATILITY / TECHNICAL FEATURES
# ============================================================

def compute_garch_vol(price: pd.Series) -> pd.Series:
    """
    Compute conditional volatility from a GARCH(1,1) model on the provided series only.
    """

    p = pd.to_numeric(price, errors="coerce")
    p = p.replace([np.inf, -np.inf], np.nan).dropna()

    if len(p) < 60:
        return pd.Series(index=price.index, dtype=float, name="garch_vol")

    r = np.log(p).diff().dropna() * 100.0

    if len(r) < 50 or float(np.nanstd(r)) < 1e-10:
        return pd.Series(index=price.index, dtype=float, name="garch_vol")

    try:
        am = arch_model(
            r,
            vol="Garch",
            p=1,
            q=1,
            mean="Zero",
            dist="normal",
        )
        res = am.fit(disp="off")

        garch_vol = pd.Series(
            res.conditional_volatility,
            index=r.index,
            name="garch_vol",
        )

        return garch_vol.reindex(price.index)

    except Exception:
        return pd.Series(index=price.index, dtype=float, name="garch_vol")


def compute_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def cross_count_feature(price: pd.Series, window: int = 20) -> pd.Series:
    """
    CrossCount_t:
    Number of times price crosses its rolling mean over last window days.
    """

    p = pd.to_numeric(price, errors="coerce")

    m = p.rolling(window).mean()
    diff = p - m

    sign = np.sign(diff)
    sign = sign.replace(0, np.nan).ffill()

    cross = (sign * sign.shift(1) < 0).astype(float)
    cross_count = cross.rolling(window).sum() / window

    return cross_count.rename(f"cross_count_{window}")


# ============================================================
# PLOTTING
# ============================================================

def plot_hmm_regimes_background(
    df: pd.DataFrame,
    state_col: str,
    price_col: str = "MID",
    title: str = "HMM Regimes",
    figsize=(18, 7),
    colors=None,
    alpha=0.35,
):
    plot_df = df.copy()

    if "Date" in plot_df.columns:
        plot_df["Date"] = pd.to_datetime(plot_df["Date"])
        plot_df = plot_df.sort_values("Date").set_index("Date")
    else:
        plot_df = plot_df.sort_index()

    plot_df = plot_df[[price_col, state_col]].dropna().copy()
    plot_df[state_col] = plot_df[state_col].astype(int)

    if colors is None:
        colors = {
            0: "lightcoral",
            1: "darkseagreen",
            2: "slateblue",
        }

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        plot_df.index,
        plot_df[price_col],
        color="black",
        linewidth=2,
        label="Price",
    )

    block_id = (plot_df[state_col] != plot_df[state_col].shift()).cumsum()

    for _, block in plot_df.groupby(block_id):
        regime = int(block[state_col].iloc[0])
        start = block.index[0]
        end = block.index[-1]

        ax.axvspan(
            start,
            end,
            color=colors.get(regime, "gray"),
            alpha=alpha,
        )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel(price_col, fontsize=13)

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_regimes(
    df: pd.DataFrame,
    states: np.ndarray,
    title: str,
    price_col: str = "MID",
):
    plot_df = df.copy()
    plot_df["state"] = states

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(
        plot_df.index,
        plot_df[price_col],
        linewidth=1.2,
        label=price_col,
    )

    for s in sorted(plot_df["state"].unique()):
        mask = plot_df["state"] == s

        ax.scatter(
            plot_df.index[mask],
            plot_df.loc[mask, price_col],
            s=14,
            alpha=0.8,
            label=f"Regime {s}",
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(price_col)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_state_probabilities(
    index,
    probas: np.ndarray,
    title: str,
):
    fig, ax = plt.subplots(figsize=(15, 6))

    for s in range(probas.shape[1]):
        ax.plot(
            index,
            probas[:, s],
            linewidth=1.2,
            label=f"P(state={s})",
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_rsi_by_regime(df: pd.DataFrame, title: str = "RSI by regime"):
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    ax[0].plot(df.index, df["MID"], color="black", linewidth=1.5)
    ax[0].set_title(title + " - MID")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(df.index, df["RSI_14"], color="blue", linewidth=1.2)

    ax[1].axhline(70, color="red", linestyle="--", alpha=0.8)
    ax[1].axhline(30, color="green", linestyle="--", alpha=0.8)

    ax[1].set_title(title + " - RSI(14)")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ============================================================
# DATA LOADING
# ============================================================

def load_spot_data() -> pd.DataFrame:
    spot = pd.read_excel(SPOT_PATH, skiprows=1)
    spot.columns = spot.columns.str.strip()

    if "DATE" not in spot.columns or "BID" not in spot.columns or "OFFER" not in spot.columns:
        raise ValueError("Spot file must contain DATE, BID, OFFER columns.")

    spot["DATE"] = pd.to_datetime(spot["DATE"], errors="coerce")
    spot["BID"] = pd.to_numeric(spot["BID"], errors="coerce")
    spot["OFFER"] = pd.to_numeric(spot["OFFER"], errors="coerce")

    spot["MID"] = (spot["BID"] + spot["OFFER"]) / 2
    spot["MID"] = spot["MID"].ffill()
    spot["BA_SPREAD"] = spot["OFFER"] - spot["BID"]

    spot_df = spot[["DATE", "BID", "OFFER", "MID", "BA_SPREAD"]].copy()
    spot_df = spot_df.dropna(subset=["DATE", "MID"])
    spot_df = spot_df.sort_values("DATE")
    spot_df = spot_df.rename(columns={"DATE": "Date"})
    spot_df = spot_df.groupby("Date", as_index=False).last()

    return spot_df


def load_futures_data() -> pd.DataFrame:
    fut = pd.read_csv(FUTURES_PATH, sep=";")
    fut.columns = fut.columns.str.strip()

    if "Date" not in fut.columns or "Close" not in fut.columns:
        raise ValueError("Futures file must contain Date and Close columns.")

    fut["Date"] = pd.to_datetime(fut["Date"], errors="coerce")
    fut["Close"] = pd.to_numeric(fut["Close"], errors="coerce")

    fut_df = fut[["Date", "Close"]].copy()
    fut_df = fut_df.dropna(subset=["Date", "Close"])
    fut_df = fut_df.sort_values("Date")
    fut_df = fut_df.groupby("Date", as_index=False).last()
    fut_df = fut_df.rename(columns={"Close": "FUTURES_CLOSE"})
    fut_df = fut_df[fut_df["FUTURES_CLOSE"] > 0].copy()

    return fut_df


def build_full_dataset(shift_spot_by_1d: bool = False) -> pd.DataFrame:
    spot_df = load_spot_data().copy()
    fut_df = load_futures_data().copy()

    print("Spot rows before merge :", len(spot_df))
    print("Futures rows before merge:", len(fut_df))

    data = pd.merge(
        spot_df,
        fut_df,
        on="Date",
        how="inner",
        validate="one_to_one",
    )

    print("Rows after merge :", len(data))

    data = data.sort_values("Date").copy()

    data["MID"] = pd.to_numeric(data["MID"], errors="coerce")
    data["BA_SPREAD"] = pd.to_numeric(data["BA_SPREAD"], errors="coerce")
    data["FUTURES_CLOSE"] = pd.to_numeric(data["FUTURES_CLOSE"], errors="coerce")

    data = data.dropna(subset=["Date", "MID", "BA_SPREAD", "FUTURES_CLOSE"])

    data = data[
        (data["MID"] > 0)
        & (data["FUTURES_CLOSE"] > 0)
        & (data["BA_SPREAD"] >= 0)
    ].copy()

    if shift_spot_by_1d:
        spot_cols = ["BID", "OFFER", "MID", "BA_SPREAD"]
        data[spot_cols] = data[spot_cols].shift(1)

        data = data.dropna(
            subset=["BID", "OFFER", "MID", "BA_SPREAD", "FUTURES_CLOSE"]
        ).copy()

    data = data.set_index("Date")

    return data


# ============================================================
# HMM FEATURES
# ============================================================

def rolling_slope(series: pd.Series, window: int = 20) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    x = np.arange(window)

    def _slope(y):
        if np.isnan(y).any():
            return np.nan
        return np.polyfit(x, y, 1)[0]

    return vals.rolling(window).apply(_slope, raw=True)


def build_hmm_features_for_subset(
    df: pd.DataFrame,
    window: int = 20,
    trend_window: int = 50,
    rsi_window: int = 14,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Features HMM spot enrichies :
    - price action
    - spread / liquidité
    - dynamique bid-offer
    - momentum / mean-reversion proxy
    """

    out = pd.DataFrame(index=df.index)

    bid = pd.to_numeric(df["BID"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    ask = pd.to_numeric(df["OFFER"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    mid = (bid + ask) / 2.0
    mid = mid.replace([np.inf, -np.inf], np.nan)

    log_mid = np.log(mid)

    spread_abs = ask - bid
    spread_rel = spread_abs / (mid + eps)

    # =====================================================
    # 1) Price dynamics
    # =====================================================
    out["r_mid"] = log_mid.diff()
    out["abs_r_mid"] = out["r_mid"].abs()

    out["rolling_vol"] = out["r_mid"].rolling(window).std()
    out["downside_vol"] = out["r_mid"].clip(upper=0).rolling(window).std()

    # =====================================================
    # 2) Liquidity / spread
    # =====================================================
    out["spread_rel"] = spread_rel
    out["spread_change"] = spread_rel.diff()
    out["spread_vol"] = spread_rel.rolling(window).std()
    out["spread_zscore"] = (
        spread_rel - spread_rel.rolling(window).mean()
    ) / (spread_rel.rolling(window).std() + eps)

    # =====================================================
    # 3) Bid / offer pressure
    # =====================================================
    out["bid_return"] = np.log(bid / bid.shift(1))
    out["ask_return"] = np.log(ask / ask.shift(1))

    out["bid_ask_return_gap"] = out["bid_return"] - out["ask_return"]

    out["bid_momentum"] = np.log(bid / bid.shift(window))
    out["ask_momentum"] = np.log(ask / ask.shift(window))

    out["quote_pressure"] = out["bid_momentum"] - out["ask_momentum"]

    # =====================================================
    # 4) Trend / momentum
    # =====================================================
    out["momentum"] = log_mid - log_mid.shift(window)

    rolling_max = mid.rolling(window).max()
    out["drawdown"] = (mid / (rolling_max + eps)) - 1.0

    rolling_mean = mid.rolling(window).mean()
    rolling_std = mid.rolling(window).std()
    out["zscore"] = (mid - rolling_mean) / (rolling_std + eps)

    out["trend_strength"] = rolling_slope(mid, trend_window)

    # =====================================================
    # 5) RSI
    # =====================================================
    out["rsi"] = compute_rsi(mid, window=rsi_window)
        # =====================================================
    # 6) Local covariance eigenstructure
    # =====================================================

    cov_features = pd.concat(
        [
            out["r_mid"],
            out["spread_change"],
            out["quote_pressure"],
            out["momentum"],
        ],
        axis=1,
    )

    cov_features.columns = [
        "r_mid",
        "spread_change",
        "quote_pressure",
        "momentum",
    ]

    eig1 = []

    for i in range(len(cov_features)):
        if i < window:
            eig1.append(np.nan)
            continue

        X = cov_features.iloc[i - window:i].dropna()

        if len(X) < window // 2:
            eig1.append(np.nan)
            continue

        cov_matrix = X.cov().values

        try:
            eigenvals = np.linalg.eigvalsh(cov_matrix)
            eig1.append(np.max(eigenvals))
        except:
            eig1.append(np.nan)

    out["cov_eig1"] = eig1
        # =====================================================
    # 2bis) Jump component (realized jumps)
    # =====================================================

    r = out["r_mid"]

    # Realized variance (rolling)
    rv = (r ** 2).rolling(window).sum()

    # Bipower variation approximation
    abs_r = r.abs()

    bv = (
        abs_r.rolling(window).mean()
        * abs_r.shift(1).rolling(window).mean()
    ) * (window * np.pi / 2)

    out["realized_var"] = rv
    out["bipower_var"] = bv

    out["jump_component"] = rv - bv

    # optional normalization (important for HMM stability)
    out["jump_ratio"] = (rv - bv) / (rv + eps)
    out["jump_intensity"] = out["r_mid"].abs() / (out["rolling_vol"] + eps)
    from sklearn.decomposition import PCA

    # =====================================================
    # 7) Term structure PCA (Level / Slope / Curvature)
    # =====================================================

    # détecter colonnes futures (adapter selon ton dataset)
    curve_cols = [
    c for c in df.columns
    if c.startswith("F")
]

    

        # =====================================================
    # 8) Order Flow Imbalance (proxy from quotes)
    # =====================================================

    bid_change = bid.diff()
    ask_change = ask.diff()

    # normalize
    bid_sign = np.sign(bid_change)
    ask_sign = np.sign(ask_change)

    # volume proxy (price sensitivity proxy)
    bid_vol = bid_change.abs()
    ask_vol = ask_change.abs()

    # OFI proxy
    ofi = (bid_sign * bid_vol) - (ask_sign * ask_vol)

    out["ofi"] = ofi
        # =====================================================
    # 9) VPIN proxy (liquidity toxicity)
    # =====================================================

    delta_mid = out["r_mid"]

    sign = np.sign(delta_mid)

    abs_flow = delta_mid.abs()

    vpin_proxy = sign * abs_flow

    out["vpin"] = vpin_proxy.rolling(window).mean()

    out["vpin_abs"] = out["vpin"].abs()
    out["vpin_zscore"] = (
        out["vpin"]
        - out["vpin"].rolling(window).mean()
    ) / (out["vpin"].rolling(window).std() + eps)
    out["liquidity_stress"] = out["vpin"] * out["spread_rel"]
        # =====================================================
    # 10) Correlation dispersion / fragmentation
    # =====================================================

    corr_features = pd.concat(
        [
            out["r_mid"],
            out["spread_rel"],
            out["ofi"],
            out["vpin"],
            out["jump_component"],
        ],
        axis=1,
    )

    corr_features.columns = [
        "r_mid",
        "spread_rel",
        "ofi",
        "vpin",
        "jump_component",
    ]

    corr_dispersion = []

    for i in range(len(corr_features)):

        if i < window:
            corr_dispersion.append(np.nan)
            continue

        X = corr_features.iloc[i-window:i].dropna()

        if len(X) < window // 2:
            corr_dispersion.append(np.nan)
            continue

        try:
            corr_matrix = X.corr().values

            # upper triangle only
            upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

            corr_dispersion.append(np.std(upper))

        except:
            corr_dispersion.append(np.nan)

    out["corr_dispersion"] = corr_dispersion
    avg_corr = []

    for i in range(len(corr_features)):

        if i < window:
            avg_corr.append(np.nan)
            continue

        X = corr_features.iloc[i-window:i].dropna()

        if len(X) < window // 2:
            avg_corr.append(np.nan)
            continue

        try:
            corr_matrix = X.corr().values

            upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

            avg_corr.append(np.mean(upper))

        except:
            avg_corr.append(np.nan)

    out["avg_corr"] = avg_corr
    # =====================================================
    # Clean
    # =====================================================
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().copy()

    return out

# ============================================================
# HMM FIT
# ============================================================

def fit_hmm(X_z: np.ndarray):

    if HMM_TYPE.lower() == "student":
        hmm_cfg = StudentTHMMConfig(
            K=NB_STATES,
            n_iter=N_ITER,
            seed=SEED,
            cov_type=COV_TYPE,
        )

    elif HMM_TYPE.lower() == "gaussian":
        hmm_cfg = GaussianHMMConfig(
            K=NB_STATES,
            n_iter=N_ITER,
            seed=SEED,
            cov_type=COV_TYPE,
        )

    else:
        raise ValueError("HMM_TYPE must be 'student' or 'gaussian'.")

    model = create_hmm(hmm_cfg)

    if hasattr(model, "fit_multi_start"):
        result = model.fit_multi_start(
            X_z,
            seeds=MULTI_START_SEEDS,
            verbose=True,
        )
    else:
        result = model.fit(X_z)

    return model, result


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def chronological_split_raw(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
):
    n = len(df)
    split_idx = int(n * train_ratio)

    data_train = df.iloc[:split_idx].copy()
    data_test = df.iloc[split_idx:].copy()

    return data_train, data_test


# ============================================================
# REGIME ORDERING
# ============================================================

def reorder_states_by_mean_return(states, probas, ref_series):
    """
    Réordonne les états HMM pour avoir:
    state 0 = worst regime
    state K-1 = best regime

    basé sur la moyenne de ref_series (typiquement r_mid)
    """

    tmp = pd.DataFrame({
        "state": states,
        "ref": ref_series
    }).dropna()

    means = tmp.groupby("state")["ref"].mean().sort_values()
    ordered_states = means.index.tolist()

    mapping = {old: new for new, old in enumerate(ordered_states)}

    states_new = np.array([mapping[s] for s in states], dtype=int)

    if probas is not None:
        probas_new = np.zeros_like(probas)
        for old, new in mapping.items():
            probas_new[:, new] = probas[:, old]
    else:
        probas_new = None

    return states_new, probas_new, mapping, means


# ============================================================
# SEGMENTATION HMM (BOOK STYLE)
# ============================================================


# ============================================================
# REGIME DESCRIPTION
# ============================================================



# ============================================================
# FUTURES RSI / REVERSION ANALYSIS
# ============================================================

def build_futures_rsi_reversion_df(
    df: pd.DataFrame,
    spot_states: np.ndarray,
    futures_col: str = "FUTURES_CLOSE",
    rsi_window: int = 14,
    horizons: list = [1, 2, 3, 5, 10, 15, 20, 30, 40, 60],
):
    out = df.copy()

    out["spot_state_available"] = spot_states
    out["FUT_RSI_14"] = compute_rsi(out[futures_col], window=rsi_window)

    out["rsi_dev"] = out["FUT_RSI_14"] - 50.0
    out["abs_rsi_dev"] = out["rsi_dev"].abs()

    for h in horizons:
        out[f"rsi_dev_h{h}"] = out["rsi_dev"].shift(-h)
        out[f"abs_rsi_dev_h{h}"] = out["abs_rsi_dev"].shift(-h)

        out[f"reversion_gain_h{h}"] = (
            out["abs_rsi_dev"] - out[f"abs_rsi_dev_h{h}"]
        )

        out[f"reversion_score_h{h}"] = -out["rsi_dev"] * (
            out[f"rsi_dev_h{h}"] - out["rsi_dev"]
        )

    return out


def analyze_futures_rsi_reversion_by_spot_regime(
    df: pd.DataFrame,
    state_col: str = "spot_state_lag",
    horizons: list = [1, 2, 3, 5, 10, 15],
):
    results = []

    for h in horizons:
        gain_col = f"reversion_gain_h{h}"
        score_col = f"reversion_score_h{h}"

        x = df[[state_col, "rsi_dev", "abs_rsi_dev", gain_col, score_col]]
        x = x.dropna().copy()

        for s in sorted(x[state_col].unique()):
            sub = x[x[state_col] == s]

            if len(sub) < 15:
                continue

            mean_gain = sub[gain_col].mean()
            mean_score = sub[score_col].mean()
            hit_revert = (sub[gain_col] > 0).mean()

            t_stat, p_val = stats.ttest_1samp(
                sub[gain_col],
                popmean=0.0,
                nan_policy="omit",
            )

            results.append({
                "spot_state": int(s),
                "horizon": h,
                "n_obs": len(sub),
                "mean_rsi_dev_t": sub["rsi_dev"].mean(),
                "mean_abs_rsi_dev_t": sub["abs_rsi_dev"].mean(),
                "mean_reversion_gain": mean_gain,
                "mean_reversion_score": mean_score,
                "hit_ratio_revert": hit_revert,
                "t_stat_gain_gt_0": float(t_stat) if pd.notna(t_stat) else np.nan,
                "p_value_gain_gt_0": float(p_val) if pd.notna(p_val) else np.nan,
            })

    return pd.DataFrame(results)


def print_futures_rsi_reversion_results(res: pd.DataFrame, title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if res.empty:
        print("No results.")
        return

    print(res.round(4).to_string(index=False))


def futures_rsi_conditioned_on_available_state(
    df: pd.DataFrame,
    states: np.ndarray,
    futures_col: str = "FUTURES_CLOSE",
    rsi_window: int = 14,
):
    out = df.copy()

    out["state_available"] = states
    out["FUT_RSI_14"] = compute_rsi(out[futures_col], window=rsi_window)

    results = []

    for k in sorted(out["state_available"].dropna().unique()):
        sub = out[out["state_available"] == k]
        rsi = sub["FUT_RSI_14"].dropna()

        if len(rsi) < 10:
            continue

        results.append({
            "state_available": int(k),
            "count": len(rsi),
            "mean_rsi": rsi.mean(),
            "std_rsi": rsi.std(),
            "median_rsi": rsi.median(),
            "pct_overbought (>70)": (rsi > 70).mean(),
            "pct_oversold (<30)": (rsi < 30).mean(),
        })

    return pd.DataFrame(results)
def test_forward_return_hac(
    df,
    state_col="state",
    horizons=(1, 3, 5, 10, 15, 20),
):
    rows = []

    for h in horizons:
        col = f"fwd_ret_{h}"
        x = df[[state_col, col]].dropna().copy()

        for s in sorted(x[state_col].unique()):
            x[f"is_state_{s}"] = (x[state_col] == s).astype(int)

            y = x[col]
            X = sm.add_constant(x[f"is_state_{s}"])

            model = sm.OLS(y, X).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": h},
            )

            beta = model.params[f"is_state_{s}"]
            tval = model.tvalues[f"is_state_{s}"]
            pval = model.pvalues[f"is_state_{s}"]

            rows.append({
                "state": int(s),
                "horizon": h,
                "beta_state": beta,
                "t_hac": tval,
                "p_hac": pval,
                "n": int(model.nobs),
            })

    return pd.DataFrame(rows)

# ============================================================
# FUTURES RSI IMPACT TESTS
# ============================================================
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
def prove_spot_regime_shifts_futures_rsi(
    df: pd.DataFrame,
    states: np.ndarray,
    futures_col: str = "FUTURES_CLOSE",
    rsi_window: int = 14,
    high_thr: float = 55.0,
    low_thr: float = 45.0,
):
    out = df.copy()

    out["spot_state_available"] = states
    out["FUT_RSI_14"] = compute_rsi(out[futures_col], window=rsi_window)

    state_col = "spot_state_available"

    x = out[[state_col, "FUT_RSI_14"]].dropna().copy()
    x[state_col] = x[state_col].astype(int)

    summary = (
        x.groupby(state_col)["FUT_RSI_14"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            median="median",
        )
        .reset_index()
    )

    groups = [g["FUT_RSI_14"].values for _, g in x.groupby(state_col)]

    anova_stat, anova_p = stats.f_oneway(*groups)
    kruskal_stat, kruskal_p = stats.kruskal(*groups)

    ordered_pairs = [(0, 1), (0, 2), (1, 2)]
    pair_rows = []

    for s1, s2 in ordered_pairs:
        r1 = x.loc[x[state_col] == s1, "FUT_RSI_14"].values
        r2 = x.loc[x[state_col] == s2, "FUT_RSI_14"].values

        t_stat_2s, p_2s = stats.ttest_ind(
            r1,
            r2,
            equal_var=False,
            nan_policy="omit",
        )

        p_1s = p_2s / 2 if t_stat_2s > 0 else 1 - p_2s / 2

        u_stat, u_p = stats.mannwhitneyu(
            r1,
            r2,
            alternative="greater",
        )

        pooled = np.sqrt(
            (
                (len(r1) - 1) * np.var(r1, ddof=1)
                + (len(r2) - 1) * np.var(r2, ddof=1)
            )
            / (len(r1) + len(r2) - 2)
        )

        d = (np.mean(r1) - np.mean(r2)) / pooled

        pair_rows.append({
            "state_1": s1,
            "state_2": s2,
            "mean_1": np.mean(r1),
            "mean_2": np.mean(r2),
            "mean_diff": np.mean(r1) - np.mean(r2),
            "welch_t_stat": t_stat_2s,
            "welch_p_one_sided": p_1s,
            "mw_u_stat": u_stat,
            "mw_p_one_sided": u_p,
            "cohen_d": d,
            "n1": len(r1),
            "n2": len(r2),
        })

    pairwise = pd.DataFrame(pair_rows)

    if not pairwise.empty:
        pairwise["welch_p_holm"] = multipletests(
            pairwise["welch_p_one_sided"],
            method="holm",
        )[1]

        pairwise["mw_p_holm"] = multipletests(
            pairwise["mw_p_one_sided"],
            method="holm",
        )[1]

    reg_df = x.copy()

    X = sm.add_constant(reg_df[state_col])
    y = reg_df["FUT_RSI_14"]

    ols = sm.OLS(y, X).fit()

    beta = ols.params[state_col]
    t_beta = ols.tvalues[state_col]
    p_beta_two_sided = ols.pvalues[state_col]

    p_beta_one_sided = (
        p_beta_two_sided / 2
        if beta < 0
        else 1 - p_beta_two_sided / 2
    )

    spearman_rho, spearman_p_two = stats.spearmanr(
        reg_df[state_col],
        reg_df["FUT_RSI_14"],
    )

    spearman_p_one = (
        spearman_p_two / 2
        if spearman_rho < 0
        else 1 - spearman_p_two / 2
    )

    monotonic = {
        "ols_beta": beta,
        "ols_t_stat": t_beta,
        "ols_p_one_sided_beta_lt_0": p_beta_one_sided,
        "ols_r2": ols.rsquared,
        "spearman_rho": spearman_rho,
        "spearman_p_one_sided_rho_lt_0": spearman_p_one,
    }

    x["is_high"] = (x["FUT_RSI_14"] > high_thr).astype(int)
    x["is_low"] = (x["FUT_RSI_14"] < low_thr).astype(int)

    event_summary = (
        x.groupby(state_col)[["is_high", "is_low"]]
        .mean()
        .reset_index()
    )

    sub02 = x[x[state_col].isin([0, 2])].copy()

    count_high = np.array([
        sub02.loc[sub02[state_col] == 0, "is_high"].sum(),
        sub02.loc[sub02[state_col] == 2, "is_high"].sum(),
    ])

    nobs_high = np.array([
        (sub02[state_col] == 0).sum(),
        (sub02[state_col] == 2).sum(),
    ])

    z_high, p_high = proportions_ztest(
        count_high,
        nobs_high,
        alternative="larger",
    )

    count_low = np.array([
        sub02.loc[sub02[state_col] == 2, "is_low"].sum(),
        sub02.loc[sub02[state_col] == 0, "is_low"].sum(),
    ])

    nobs_low = np.array([
        (sub02[state_col] == 2).sum(),
        (sub02[state_col] == 0).sum(),
    ])

    z_low, p_low = proportions_ztest(
        count_low,
        nobs_low,
        alternative="larger",
    )

    events = {
        "threshold_high": high_thr,
        "threshold_low": low_thr,
        "event_summary": event_summary,
        "test_high_rsi_state0_gt_state2": {
            "z_stat": float(z_high),
            "p_value": float(p_high),
        },
        "test_low_rsi_state2_gt_state0": {
            "z_stat": float(z_low),
            "p_value": float(p_low),
        },
    }

    return {
        "summary": summary,
        "global_tests": {
            "anova": {
                "stat": float(anova_stat),
                "p_value": float(anova_p),
            },
            "kruskal": {
                "stat": float(kruskal_stat),
                "p_value": float(kruskal_p),
            },
        },
        "pairwise_ordered_tests": pairwise,
        "monotonic_tests": monotonic,
        "event_tests": events,
    }


# ============================================================
# OU / MEAN REVERSION TESTS
# ============================================================

def test_forward_ou_by_available_spot_state(
    df_test: pd.DataFrame,
    states_test: np.ndarray,
    futures_col: str = "FUTURES_CLOSE",
    horizon: int = 5,
    hac_lags: int = 5,
):
    data = df_test.copy()

    data["state_available"] = states_test
    data["logF_t"] = np.log(pd.to_numeric(data[futures_col], errors="coerce"))
    data["logF_t_plus_h"] = data["logF_t"].shift(-horizon)

    data = data.dropna(
        subset=["logF_t", "logF_t_plus_h", "state_available"]
    ).copy()

    data["state_available"] = data["state_available"].astype(int)

    X = pd.DataFrame(index=data.index)

    for s in sorted(data["state_available"].unique()):
        I = (data["state_available"] == s).astype(float)
        X[f"c_state_{s}"] = I
        X[f"phi_state_{s}"] = I * data["logF_t"]

    y = data["logF_t_plus_h"]

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )

    rows = []

    for s in sorted(data["state_available"].unique()):
        c = model.params[f"c_state_{s}"]
        phi = model.params[f"phi_state_{s}"]
        se_phi = model.bse[f"phi_state_{s}"]

        t_phi_lt_1 = (phi - 1.0) / se_phi
        p_phi_lt_1 = stats.norm.cdf(t_phi_lt_1)

        if 0 < phi < 1:
            kappa = -np.log(phi)
            half_life = np.log(2) / kappa
            mu = c / (1 - phi)
        else:
            kappa = np.nan
            half_life = np.nan
            mu = np.nan

        rows.append({
            "state_available": s,
            "horizon": horizon,
            "n_obs": int((data["state_available"] == s).sum()),
            "c": c,
            "phi": phi,
            "se_phi": se_phi,
            "p_value_phi_lt_1": p_phi_lt_1,
            "mu_long_run_log": mu,
            "kappa": kappa,
            "half_life_periods": half_life,
            "significant_MR_5pct": bool(
                (phi < 1) and (p_phi_lt_1 < 0.05)
            ),
        })

    return model, pd.DataFrame(rows)


def test_ou_futures_by_spot_state_on_test(
    df_test: pd.DataFrame,
    states_test: np.ndarray,
    futures_col: str = "FUTURES_CLOSE",
    hac_lags: int = 5,
):
    """
    Test OU conditionnel aux états HMM spot sur la partie TEST.

    Modèle :
        log(F_t) = c_state + phi_state * log(F_{t-1}) + eps_t

    Ici state_t vient du HMM spot appliqué au test.
    Comme le spot est déjà shifté au début, state_t correspond bien
    à l'information spot disponible avant / au moment de prédire F_t.
    """

    data = df_test.copy()

    data["state"] = states_test
    data["logF"] = np.log(pd.to_numeric(data[futures_col], errors="coerce"))
    data["logF_lag"] = data["logF"].shift(1)

    data = data.dropna(subset=["logF", "logF_lag", "state"]).copy()
    data["state"] = data["state"].astype(int)

    X = pd.DataFrame(index=data.index)

    for s in sorted(data["state"].unique()):
        I = (data["state"] == s).astype(float)
        X[f"c_state_{s}"] = I
        X[f"phi_state_{s}"] = I * data["logF_lag"]

    y = data["logF"]

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )

    rows = []

    for s in sorted(data["state"].unique()):
        c_name = f"c_state_{s}"
        phi_name = f"phi_state_{s}"

        c = model.params[c_name]
        phi = model.params[phi_name]
        se_phi = model.bse[phi_name]

        t_phi_lt_1 = (phi - 1.0) / se_phi
        p_phi_lt_1 = stats.norm.cdf(t_phi_lt_1)

        if 0 < phi < 1:
            kappa = -np.log(phi)
            half_life = np.log(2) / kappa
            mu = c / (1 - phi)
            conclusion = "Mean-reverting OU"
        elif phi >= 1:
            kappa = np.nan
            half_life = np.nan
            mu = np.nan
            conclusion = "Non mean-reverting / random walk"
        else:
            kappa = np.nan
            half_life = np.nan
            mu = c / (1 - phi)
            conclusion = "Oscillatory / unstable interpretation"

        rows.append({
            "state": s,
            "n_obs": int((data["state"] == s).sum()),
            "c": c,
            "phi": phi,
            "se_phi": se_phi,
            "t_phi_lt_1": t_phi_lt_1,
            "p_value_phi_lt_1": p_phi_lt_1,
            "mu_long_run_log": mu,
            "kappa": kappa,
            "half_life_periods": half_life,
            "conclusion": conclusion,
        })

    res = pd.DataFrame(rows)

    return model, res


def test_ou_futures_by_lagged_spot_state_on_test(
    df_test,
    states_test,
    futures_col: str = "FUTURES_CLOSE",
    state_lag: int = 15,
    hac_lags: int = 5,
):
    data = df_test.copy()

    data["state"] = states_test
    data["state_lag"] = data["state"].shift(state_lag)

    data["logF"] = np.log(pd.to_numeric(data[futures_col], errors="coerce"))
    data["logF_lag"] = data["logF"].shift(1)

    data = data.dropna(subset=["logF", "logF_lag", "state_lag"]).copy()
    data["state_lag"] = data["state_lag"].astype(int)

    X = pd.DataFrame(index=data.index)

    for s in sorted(data["state_lag"].unique()):
        I = (data["state_lag"] == s).astype(float)
        X[f"c_state_lag{s}"] = I
        X[f"phi_state_lag{s}"] = I * data["logF_lag"]

    y = data["logF"]

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )

    rows = []

    for s in sorted(data["state_lag"].unique()):
        c = model.params[f"c_state_lag{s}"]
        phi = model.params[f"phi_state_lag{s}"]
        se_phi = model.bse[f"phi_state_lag{s}"]

        t_phi_lt_1 = (phi - 1.0) / se_phi
        p_phi_lt_1 = stats.norm.cdf(t_phi_lt_1)

        if 0 < phi < 1:
            kappa = -np.log(phi)
            half_life = np.log(2) / kappa
            mu = c / (1 - phi)
        else:
            kappa = np.nan
            half_life = np.nan
            mu = np.nan

        rows.append({
            "state_lag": s,
            "lag_days": state_lag,
            "n_obs": int((data["state_lag"] == s).sum()),
            "c": c,
            "phi": phi,
            "se_phi": se_phi,
            "t_phi_lt_1": t_phi_lt_1,
            "p_value_phi_lt_1": p_phi_lt_1,
            "mu_long_run_log": mu,
            "kappa": kappa,
            "half_life_periods": half_life,
            "significant_MR_5pct": bool(
                (phi < 1) and (p_phi_lt_1 < 0.05)
            ),
        })

    return model, pd.DataFrame(rows)


# ============================================================
# FORWARD RETURNS / EVENT STUDIES
# ============================================================

def build_forward_returns(
    df,
    price_col="FUTURES_CLOSE",
    horizons=(1, 3, 5, 10, 15),
):
    out = df.copy()
    p = pd.to_numeric(out[price_col], errors="coerce")

    for h in horizons:
        out[f"fwd_ret_{h}"] = np.log(p.shift(-h) / p)

    return out


def plot_forward_returns_by_state(
    df,
    states,
    horizon=15,
    col="FUTURES_CLOSE",
):
    out = df.copy()
    out["state"] = states

    price = out[col]

    for s in sorted(out["state"].unique()):
        sub = out[out["state"] == s].copy()

        paths = []

        for t in sub.index:
            if t not in price.index:
                continue

            idx = price.index.get_loc(t)

            if idx + horizon >= len(price):
                continue

            path = price.iloc[idx:idx + horizon + 1].values
            path = path / path[0]

            paths.append(path)

        if len(paths) == 0:
            continue

        paths = np.array(paths)
        mean_path = paths.mean(axis=0)

        plt.plot(mean_path, label=f"State {s}")

    plt.axhline(1.0, linestyle="--", color="black")
    plt.title(f"Forward returns (normalized) up to {horizon} days")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_forward_returns_by_state_enhanced(
    df,
    states,
    horizon=15,
    col="FUTURES_CLOSE",
):
    out = df.copy()
    out["state"] = states

    price = out[col]

    plt.figure(figsize=(14, 6))

    for s in sorted(out["state"].unique()):
        sub = out[out["state"] == s]

        paths = []

        for t in sub.index:
            idx = price.index.get_loc(t)

            if idx + horizon >= len(price):
                continue

            path = price.iloc[idx:idx + horizon + 1].values
            path = path / path[0]

            paths.append(path)

        if len(paths) < 10:
            continue

        paths = np.array(paths)

        mean_path = paths.mean(axis=0)
        q10 = np.quantile(paths, 0.1, axis=0)
        q90 = np.quantile(paths, 0.9, axis=0)

        plt.plot(mean_path, label=f"State {s}", linewidth=2)
        plt.fill_between(range(len(mean_path)), q10, q90, alpha=0.15)

    plt.axhline(1.0, linestyle="--", color="black", linewidth=1.5)

    plt.title(
        "Forward Futures Performance by Spot Regime (TEST)",
        fontsize=14,
    )
    plt.xlabel("Days ahead")
    plt.ylabel("Normalized price (F_t+h / F_t)")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_event_study_price_paths(
    df,
    states,
    price_col="FUTURES_CLOSE",
    event_state=2,
    horizon=15,
):
    data = df.copy()
    data["state"] = states

    price = pd.to_numeric(data[price_col], errors="coerce")

    event_indices = data.index[data["state"] == event_state]

    paths = []

    for t in event_indices:
        loc = data.index.get_loc(t)

        if loc + horizon >= len(data):
            continue

        p0 = price.iloc[loc]

        if p0 <= 0 or np.isnan(p0):
            continue

        future_prices = price.iloc[loc:loc + horizon + 1]
        path = future_prices / p0

        paths.append(path.values)

    paths = np.array(paths)

    if len(paths) == 0:
        print("No valid paths.")
        return

    mean_path = np.mean(paths, axis=0)
    lower = np.min(paths, axis=0)
    upper = np.max(paths, axis=0)

    x = np.arange(horizon + 1)

    plt.figure(figsize=(10, 6))

    plt.fill_between(x, lower, upper, alpha=0.15, label="Min-Max band")
    plt.plot(x, mean_path, linewidth=2, label="Mean path")

    plt.axhline(1.0, linestyle="--", color="black")

    plt.title(f"Event Study: Futures Price after State {event_state}")
    plt.xlabel("Days after event")
    plt.ylabel("Normalized price (F_t+h / F_t)")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================
# STATISTICAL TESTS ON FORWARD RETURNS
# ============================================================

def test_directional_hypothesis(
# Version K=4 — hypothèses correctement assignées
    df,
    state_col="state",
    horizons=(3, 5, 10, 15, 20),
    bearish_states=(1,),    # states attendus bearish
    bullish_states=(3,),    # states attendus bullish
):
    results = []

    for h in horizons:
        col = f"fwd_ret_{h}"
        x = df[[state_col, col]].dropna()

        for s in sorted(x[state_col].unique()):
            sub = x[x[state_col] == s][col]

            if len(sub) < 2:
                continue

            mean  = sub.mean()
            std   = sub.std()
            n     = len(sub)
            t_stat, p_two = stats.ttest_1samp(sub, 0.0, nan_policy="omit")

            if s in bearish_states:
                p_one   = p_two / 2 if t_stat < 0 else 1 - p_two / 2
                direction = "mean < 0 (bearish)"
            elif s in bullish_states:
                p_one   = p_two / 2 if t_stat > 0 else 1 - p_two / 2
                direction = "mean > 0 (bullish)"
            else:
                p_one   = np.nan
                direction = "no hypothesis"

            results.append({
                "state"            : int(s),
                "horizon"          : h,
                "n"                : n,
                "mean_return"      : mean,
                "std"              : std,
                "hit_ratio"        : (sub > 0).mean(),
                "t_stat"           : t_stat,
                "p_value_one_sided": p_one,
                "hypothesis"       : direction,
            })

    return pd.DataFrame(results)
# ============================================================
# SIMPLE STATE-BASED BACKTEST
# ============================================================

def backtest_state2_strategy(
    df: pd.DataFrame,
    states: np.ndarray,
    price_col: str = "FUTURES_CLOSE",
    holding_period: int = 15,
):
    data = df.copy()
    data["state"] = states

    price = pd.to_numeric(data[price_col], errors="coerce")

    data["ret"] = np.log(price / price.shift(1))
    data = data.dropna().copy()

    n = len(data)
    position = np.zeros(n)

    # Signal → long pendant holding_period jours
    for i in range(n):
        if data["state"].iloc[i] == 2:
            for h in range(holding_period):
                if i + h < n:
                    position[i + h] = 1

    data["position"] = position
    data["strategy_ret"] = data["position"].shift(1) * data["ret"]

    data["cum_market"] = np.exp(data["ret"].cumsum())
    data["cum_strategy"] = np.exp(data["strategy_ret"].cumsum())

    return data


def print_performance(data):
    r = data["strategy_ret"].dropna()

    sharpe = np.sqrt(252) * r.mean() / r.std()

    cum = np.exp(r.cumsum())
    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    print("\n=== PERFORMANCE ===")
    print(f"Total return : {cum.iloc[-1] - 1:.2%}")
    print(f"Sharpe : {sharpe:.2f}")
    print(f"Max DD : {max_dd:.2%}")


def plot_pnl(data):
    plt.figure(figsize=(12, 6))

    plt.plot(
        data.index,
        data["cum_market"],
        label="Buy & Hold",
        linewidth=2,
    )

    plt.plot(
        data.index,
        data["cum_strategy"],
        label="State 2 Strategy",
        linewidth=2,
    )

    plt.title("Strategy Performance vs Market", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show(block=False)


# ============================================================
# HEDGE-FUND STYLE FORWARD IMPACT PLOT
# ============================================================

def plot_hedge_fund_forward_impact(
    df,
    states,
    price_col="FUTURES_CLOSE",
    horizons=(0, 1, 3, 5, 10, 15, 20, 30),
    title="Spot Regime Signal → Futures Forward Impact",
):
    data = df.copy()
    data["state"] = states

    price = pd.to_numeric(data[price_col], errors="coerce")

    rows = []

    for s in sorted(data["state"].dropna().unique()):
        for h in horizons:
            if h == 0:
                fwd_ret = pd.Series(0.0, index=data.index)
            else:
                fwd_ret = np.log(price.shift(-h) / price)

            sub = fwd_ret[data["state"] == s].dropna()

            if len(sub) < 10:
                continue

            mean = sub.mean()
            se = sub.std(ddof=1) / np.sqrt(len(sub))

            rows.append({
                "state": int(s),
                "horizon": h,
                "mean_return": mean,
                "lower_95": mean - 1.96 * se,
                "upper_95": mean + 1.96 * se,
                "n": len(sub),
                "hit_ratio": (sub > 0).mean(),
            })

    res = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 7))

    labels = {
        0: "State 0: Stress / Oversold",
        1: "State 1: Neutral / Range",
        2: "State 2: Trend / Momentum",
    }

    colors = {
        0: "#D62728",
        1: "#7F7F7F",
        2: "#1F77B4",
    }

    for s in sorted(res["state"].unique()):
        sub = res[res["state"] == s].sort_values("horizon")

        ax.plot(
            sub["horizon"],
            100 * sub["mean_return"],
            marker="o",
            linewidth=3,
            markersize=8,
            label=labels.get(s, f"State {s}"),
            color=colors.get(s, None),
        )

        ax.fill_between(
            sub["horizon"],
            100 * sub["lower_95"],
            100 * sub["upper_95"],
            alpha=0.15,
            color=colors.get(s, None),
        )

    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(15, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(30, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.text(
        15,
        ax.get_ylim()[1] * 0.85,
        "J+15\nMomentum peak",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85),
    )

    ax.text(
        30,
        ax.get_ylim()[1] * 0.85,
        "J+30\nMean reversion zone",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85),
    )

    state2 = res[res["state"] == 2].set_index("horizon")

    if 15 in state2.index and 30 in state2.index:
        y15 = 100 * state2.loc[15, "mean_return"]
        y30 = 100 * state2.loc[30, "mean_return"]

        ax.annotate(
            "State 2: continuation\nthen reversal risk",
            xy=(15, y15),
            xytext=(18, y15 + 1.5),
            arrowprops=dict(arrowstyle="->", linewidth=2),
            fontsize=12,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.9),
        )

        ax.annotate(
            "",
            xy=(30, y30),
            xytext=(15, y15),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=2.5,
                color=colors[2],
            ),
        )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    ax.set_xlabel("Forward horizon after available spot regime signal", fontsize=13)
    ax.set_ylabel("Average futures forward return (%)", fontsize=13)

    subtitle = (
        "Regime signal built only from shifted spot data: "
        "state at date t is economically available from spot information at t-1"
    )

    fig.text(
        0.5,
        0.01,
        subtitle,
        ha="center",
        fontsize=10,
        alpha=0.75,
    )

    ax.legend(fontsize=11, loc="best", frameon=True)
    ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show(block=False)

    return res

# ============================================================
# REAL-TIME REGIME FILTERING
# ============================================================
def build_exposure_k4(
    probas,
    bullish_state=3,      # state 3 : fort momentum positif
    bearish_state=1,      # state 1 : bearish persistant
    entry_threshold=0.65,
    exit_threshold=0.45,
    max_holding=15,       # aligné sur le pic momentum state 3 à J+15
    min_confirm=2,
):
    """
    Stratégie adaptée à K=4 :
    - State 3 (bullish, half-life=7j) → LONG avec holding court
    - State 1 (bearish, persistant)   → SHORT avec holding plus long
    """
    n = len(probas)
    exposure = np.zeros(n)

    in_position = False
    direction   = 0
    entry_day   = None
    confirm_l   = 0
    confirm_s   = 0

    for t in range(n):
        pb = probas[t, bullish_state]
        ps = probas[t, bearish_state]

        confirm_l = confirm_l + 1 if pb > entry_threshold else 0
        confirm_s = confirm_s + 1 if ps > entry_threshold else 0

        if not in_position:
            if confirm_l >= min_confirm:
                in_position = True
                direction   = 1
                entry_day   = t
                exposure[t] = 1.0

            elif confirm_s >= min_confirm:
                in_position = True
                direction   = -1
                entry_day   = t
                exposure[t] = -1.0

        else:
            holding = t - entry_day

            if direction == 1:
                if pb < exit_threshold or holding >= max_holding:
                    in_position = False
                    direction   = 0
                    entry_day   = None
                    exposure[t] = 0.0
                else:
                    exposure[t] = 1.0

            elif direction == -1:
                if ps < exit_threshold or holding >= max_holding * 2:
                    in_position = False
                    direction   = 0
                    entry_day   = None
                    exposure[t] = 0.0
                else:
                    exposure[t] = -1.0

    return exposure
def online_confidence_regime_filter(
    probas: np.ndarray,
    threshold: float = 0.70,
    min_confirm: int = 3,
):
    """
    Segmentation temps réel robuste :

    - On ne change de régime que si:
        (1) la probabilité est élevée
        (2) le nouveau régime est confirmé plusieurs jours

    => évite le bruit HMM (très important en trading réel)
    """

    n = len(probas)

    raw_states = np.argmax(probas, axis=1)
    pmax = np.max(probas, axis=1)

    online_states = np.zeros(n, dtype=int)

    current_state = raw_states[0]
    online_states[0] = current_state

    candidate_state = None
    candidate_count = 0

    for t in range(1, n):
        proposed_state = raw_states[t]
        confidence = pmax[t]

        if proposed_state == current_state:
            candidate_state = None
            candidate_count = 0
            online_states[t] = current_state
            continue

        if confidence < threshold:
            online_states[t] = current_state
            continue

        if candidate_state != proposed_state:
            candidate_state = proposed_state
            candidate_count = 1
        else:
            candidate_count += 1

        if candidate_count >= min_confirm:
            current_state = proposed_state
            candidate_state = None
            candidate_count = 0

        online_states[t] = current_state

    return online_states


# ============================================================
# PROBABILITY-BASED TRADING STRATEGIES
# ============================================================

def build_exposure_from_proba(
    probas,
    entry_state=2,
    entry_threshold=0.70,
    exit_threshold=0.50,
    max_holding=20,
):
    n = len(probas)

    exposure = np.zeros(n)

    in_position = False
    entry_day = None

    for t in range(n):
        p_state = probas[t, entry_state]
        dominant_state = np.argmax(probas[t])

        if not in_position:
            if p_state > entry_threshold:
                in_position = True
                entry_day = t
                exposure[t] = 1.0

        else:
            holding_days = t - entry_day

            if (
                p_state < exit_threshold
                or dominant_state != entry_state
                or holding_days >= max_holding
            ):
                in_position = False
                entry_day = None
                exposure[t] = 0.0
            else:
                exposure[t] = 1.0

    return exposure


def build_exposure_from_proba_multi(
    probas,
    entry_threshold=0.70,
    exit_threshold=0.50,
    reverse_threshold=0.70,
    max_holding=20,
    min_confirm=2,
):
    """
    Long / Short dynamique :

    - State 2 → LONG
    - State 0 → SHORT
    """

    n = len(probas)
    exposure = np.zeros(n)

    in_position = False
    direction = 0
    entry_day = None

    confirm_long = 0
    confirm_short = 0

    for t in range(n):
        p0 = probas[t, 0]
        p2 = probas[t, 2]

        # confirmation logic
        confirm_long = confirm_long + 1 if p2 > entry_threshold else 0
        confirm_short = confirm_short + 1 if p0 > entry_threshold else 0

        if not in_position:
            if confirm_long >= min_confirm:
                in_position = True
                direction = 1
                entry_day = t
                exposure[t] = 1.0

            elif confirm_short >= min_confirm:
                in_position = True
                direction = -1
                entry_day = t
                exposure[t] = -1.0

        else:
            holding = t - entry_day

            if direction == 1:
                if (
                    p0 > reverse_threshold and confirm_short >= min_confirm
                ):
                    direction = -1
                    entry_day = t
                    exposure[t] = -1.0

                elif (
                    p2 < exit_threshold
                    or holding >= max_holding
                ):
                    in_position = False
                    direction = 0
                    entry_day = None
                    exposure[t] = 0.0
                else:
                    exposure[t] = 1.0

            elif direction == -1:
                if (
                    p2 > reverse_threshold and confirm_long >= min_confirm
                ):
                    direction = 1
                    entry_day = t
                    exposure[t] = 1.0

                elif (
                    p0 < exit_threshold
                    or holding >= max_holding
                ):
                    in_position = False
                    direction = 0
                    entry_day = None
                    exposure[t] = 0.0
                else:
                    exposure[t] = -1.0

    return exposure


# ============================================================
# REGIME AGE (VERY IMPORTANT FEATURE)
# ============================================================

def compute_regime_age(states):
    age = np.zeros(len(states), dtype=int)
    age[0] = 1

    for t in range(1, len(states)):
        if states[t] == states[t - 1]:
            age[t] = age[t - 1] + 1
        else:
            age[t] = 1

    return age


def build_exposure_with_regime_age(
    probas,
    states,
    entry_threshold=0.85,
    exit_threshold=0.60,
    max_holding=20,
    max_age_state0=20,
    max_age_state2=15,
):
    """
    Intuition hedge fund:

    - momentum meurt avec le temps
    - stress aussi

    => on trade seulement les régimes "jeunes"
    """

    n = len(probas)
    exposure = np.zeros(n)

    age = compute_regime_age(states)

    in_position = False
    direction = 0
    entry_day = None

    for t in range(n):
        p0 = probas[t, 0]
        p2 = probas[t, 2]

        s = states[t]
        a = age[t]

        if not in_position:
            if p2 > entry_threshold and s == 2 and a <= max_age_state2:
                in_position = True
                direction = 1
                entry_day = t
                exposure[t] = 1.0

            elif p0 > entry_threshold and s == 0 and a <= max_age_state0:
                in_position = True
                direction = -1
                entry_day = t
                exposure[t] = -1.0

        else:
            holding = t - entry_day

            if direction == 1:
                if (
                    p2 < exit_threshold
                    or s != 2
                    or holding >= max_holding
                    or a > max_age_state2
                ):
                    in_position = False
                    direction = 0
                    entry_day = None
                    exposure[t] = 0.0
                else:
                    exposure[t] = 1.0

            elif direction == -1:
                if (
                    p0 < exit_threshold
                    or s != 0
                    or holding >= max_holding
                    or a > max_age_state0
                ):
                    in_position = False
                    direction = 0
                    entry_day = None
                    exposure[t] = 0.0
                else:
                    exposure[t] = -1.0

    return exposure


# ============================================================
# BAYESIAN CORRECTION (RSI + HMM)
# ============================================================

def fit_rsi_regime_likelihood(rsi: pd.Series, states: np.ndarray):
    tmp = pd.DataFrame({
        "rsi": rsi.values,
        "state": states
    }).dropna()

    params = {}

    for k in sorted(tmp["state"].unique()):
        vals = tmp[tmp["state"] == k]["rsi"].values

        params[k] = {
            "mu": float(np.mean(vals)),
            "sd": float(max(np.std(vals, ddof=1), 5.0)),
            "n": len(vals),
        }

    return params

def segment_hmm_book_style(df, probas, info_df=None):
    seg = df.copy()

    if info_df is not None:
        for col in ["r_mid", "rolling_vol"]:
            if col not in seg.columns and col in info_df.columns:
                seg[col] = info_df.loc[seg.index, col]

    seg["state"] = np.argmax(probas, axis=1)
    seg["pmax"] = np.max(probas, axis=1)
    seg["block_id"] = (seg["state"] != seg["state"].shift()).cumsum()

    agg_dict = {
        "state": ("state", "first"),
        "start": ("state", lambda x: x.index[0]),
        "end": ("state", lambda x: x.index[-1]),
        "duration": ("state", "size"),
        "mean_pmax": ("pmax", "mean"),
        "min_pmax": ("pmax", "min"),
    }

    if "r_mid" in seg.columns:
        agg_dict["mean_r_mid"] = ("r_mid", "mean")
        agg_dict["vol_r_mid"] = ("r_mid", "std")

    if "rolling_vol" in seg.columns:
        agg_dict["mean_rolling_vol"] = ("rolling_vol", "mean")

    blocks = (
        seg.groupby("block_id")
        .agg(**agg_dict)
        .reset_index(drop=True)
    )

    return seg, blocks
def apply_rsi_bayesian_correction(
    hmm_probas,
    rsi,
    rsi_params,
    weight=1.0,
):
    """
    Combinaison HMM + RSI (Bayesian update)
    """

    rsi_vals = rsi.values.astype(float)
    n, K = hmm_probas.shape

    likelihood = np.ones((n, K))

    for k in range(K):
        mu = rsi_params[k]["mu"]
        sd = rsi_params[k]["sd"]

        likelihood[:, k] = stats.norm.pdf(rsi_vals, loc=mu, scale=sd)

    likelihood = np.maximum(likelihood, 1e-12)

    scores = hmm_probas * (likelihood ** weight)
    scores = scores / scores.sum(axis=1, keepdims=True)

    states_corrected = np.argmax(scores, axis=1)

    return states_corrected, scores
# ============================================================
# ROLLING WINDOW RETRAIN
# ============================================================
def map_states_by_train_mean_return(
    states_train: np.ndarray,
    states_test: np.ndarray,
    alpha_test: np.ndarray,
    ref_series: np.ndarray,
):
    tmp = pd.DataFrame({
        "state": states_train,
        "ref": ref_series,
    }).dropna()

    means = tmp.groupby("state")["ref"].mean().sort_values()
    ordered_states = means.index.tolist()

    mapping = {old: new for new, old in enumerate(ordered_states)}

    states_test_ord = np.array(
        [mapping.get(s, s) for s in states_test],
        dtype=int,
    )

    alpha_test_ord = np.zeros_like(alpha_test)
    for old, new in mapping.items():
        alpha_test_ord[:, new] = alpha_test[:, old]

    return states_test_ord, alpha_test_ord, mapping
def rolling_hmm_pipeline(
    data_df, selected_features, mapping_ref,
    train_window=504, retrain_every=63, hmm_window=20, clip_val=3.5,
):
    all_dates = data_df.index
    n = len(all_dates)
    results = []
    start_idx = train_window

    retrain_dates = []
    idx = start_idx
    while idx < n:
        retrain_dates.append(idx)
        idx += retrain_every

    print(f"Rolling retrains prévus : {len(retrain_dates)}")

    for i, train_end_idx in enumerate(retrain_dates):

        train_start_idx = max(0, train_end_idx - train_window)
        test_start_idx  = train_end_idx
        test_end_idx    = min(n, train_end_idx + retrain_every)

        if test_start_idx >= n:
            break

        data_train = data_df.iloc[train_start_idx:train_end_idx].copy()
        data_test  = data_df.iloc[test_start_idx:test_end_idx].copy()

        print(f"\n[Retrain {i+1}/{len(retrain_dates)}] "
              f"Train: {data_train.index[0].date()} → {data_train.index[-1].date()} "
              f"({len(data_train)}j) | "
              f"OOS: {data_test.index[0].date()} → {data_test.index[-1].date()} "
              f"({len(data_test)}j)")

        warmup_size = 60
        df_warmup = pd.concat([data_train.iloc[-warmup_size:], data_test])

        try:
            feat_train    = build_hmm_features_for_subset(data_train, window=hmm_window)
            feat_test_raw = build_hmm_features_for_subset(df_warmup,  window=hmm_window)
            feat_test     = feat_test_raw.loc[
                feat_test_raw.index.isin(data_test.index)
            ].copy()
        except Exception as e:
            print(f"  Feature error: {e} — skip")
            continue

        missing = [c for c in selected_features if c not in feat_train.columns]
        if missing:
            print(f"  Features manquantes: {missing} — skip")
            continue

        feat_train_full = feat_train.copy()

        feat_train = feat_train[selected_features]
        feat_test  = feat_test[selected_features]

        if len(feat_train) < 100 or len(feat_test) == 0:
            print(f"  Pas assez de data — skip")
            continue

        X_train, X_test, _, _ = standardize(feat_train.values, feat_test.values)
        X_train = np.clip(X_train, -clip_val, clip_val)
        X_test  = np.clip(X_test,  -clip_val, clip_val)

        try:
            model, result = fit_hmm(X_train)
        except Exception as e:
            print(f"  HMM fit error: {e} — skip")
            continue

        # ================================================================
        # VALIDATION DU FIT — AJOUT ICI, JUSTE APRÈS fit_hmm
        # ================================================================
        final_ll = result.ll_hist[-1]
        nu_vals  = np.array(result.nu) if (
            hasattr(result, "nu") and result.nu is not None
        ) else np.array([])

        ll_ok  = final_ll > 0
        nu_ok  = len(nu_vals) == 0 or np.all(nu_vals < 50)

        states_check, _, _ = model.filter(X_train)
        counts = np.bincount(states_check, minlength=NB_STATES) / len(states_check)
        pop_ok = np.all(counts >= 0.03)

        if not (ll_ok and nu_ok and pop_ok):
            nu_max = float(nu_vals.max()) if len(nu_vals) > 0 else float("nan")
            print(
                f"  ⚠ Retrain {i+1} skippé — "
                f"LL={final_ll:.1f}  nu_max={nu_max:.1f}  "
                f"counts={np.round(counts, 2)}"
            )
            continue
        # ================================================================
        # FIN VALIDATION
        # ================================================================

        try:
            states_filt, alpha, _ = model.filter(X_test)
        except Exception as e:
            print(f"  Filter error: {e} — skip")
            continue

        states_train, alpha_train, _ = model.filter(X_train)
        states_test, alpha_test, _ = model.filter(X_test)

        states_ord, alpha_ord, local_mapping = map_states_by_train_mean_return(
            states_train=states_train,
            states_test=states_test,
            alpha_test=alpha_test,
            ref_series=feat_train_full["r_mid"].values,
        )

        test_dates = feat_test.index
        for t_idx, date in enumerate(test_dates):
            row = {
                "date": date,
                "state": int(states_ord[t_idx]),
                "pmax": float(np.max(alpha_ord[t_idx])),
                "retrain_id": i,
            }
            for k in range(alpha_ord.shape[1]):
                row[f"p{k}"] = float(alpha_ord[t_idx, k])
            results.append(row)

    results_df = pd.DataFrame(results).set_index("date")
    results_df = results_df[~results_df.index.duplicated(keep="last")]
    results_df = results_df.sort_index()

    return results_df

def _apply_reference_mapping(
    states: np.ndarray,
    alpha: np.ndarray,
    feat_train: pd.DataFrame,
    selected_features: list,
    mapping_ref: dict,
) -> np.ndarray:
    """
    Ancrage par volatilité croissante : state 0 = calme, state K-1 = volatile.
    Plus stable que r_mid entre retrains car la vol est toujours positive
    et monotone par rapport à l'intensité du régime.
    """
    # Priorité : rolling_vol, sinon downside_vol, sinon première feature
    vol_candidates = ["rolling_vol", "downside_vol", "spread_vol", "realized_var"]
    
    ref_col = None
    for c in vol_candidates:
        if c in feat_train.columns:
            ref_col = c
            break
    
    if ref_col is None:
        ref_col = feat_train.columns[0]

    # Train states pour calculer la moyenne de vol par état
    n_train = len(states)
    ref_series = feat_train[ref_col].values[:n_train]

    tmp = pd.DataFrame({
        "state": states,
        "ref": ref_series
    }).dropna()

    if tmp.empty or tmp["state"].nunique() < 2:
        return states

    # Ordonner par vol croissante : state 0 = plus calme
    means = tmp.groupby("state")["ref"].mean().sort_values(ascending=True)
    ordered = means.index.tolist()
    local_mapping = {old: new for new, old in enumerate(ordered)}

    return np.array([local_mapping.get(s, s) for s in states], dtype=int)

def _reorder_probas(
    alpha: np.ndarray,
    states_ord: np.ndarray,
    states_orig: np.ndarray,
) -> np.ndarray:
    """Réordonne les colonnes de alpha selon le mapping états."""
    K = alpha.shape[1]
    alpha_ord = np.zeros_like(alpha)

    # Reconstruire le mapping inverse
    unique_orig = np.unique(states_orig)
    unique_ord  = np.unique(states_ord)

    if len(unique_orig) != len(unique_ord):
        return alpha  # fallback

    for i, o in enumerate(unique_orig):
        n = states_ord[states_orig == o]
        if len(n) == 0:
            continue
        new_state = int(np.bincount(n).argmax())
        if new_state < K and o < K:
            alpha_ord[:, new_state] = alpha[:, o]

    return alpha_ord

# Version K=4 — hypothèses correctement assignées
def test_directional_hypothesis_k4(
    df,
    state_col="state",
    horizons=(3, 5, 10, 15, 20),
    bearish_states=(1,),    # states attendus bearish
    bullish_states=(3,),    # states attendus bullish
):
    results = []

    for h in horizons:
        col = f"fwd_ret_{h}"
        x = df[[state_col, col]].dropna()

        for s in sorted(x[state_col].unique()):
            sub = x[x[state_col] == s][col]

            if len(sub) < 2:
                continue

            mean  = sub.mean()
            std   = sub.std()
            n     = len(sub)
            t_stat, p_two = stats.ttest_1samp(sub, 0.0, nan_policy="omit")

            if s in bearish_states:
                p_one   = p_two / 2 if t_stat < 0 else 1 - p_two / 2
                direction = "mean < 0 (bearish)"
            elif s in bullish_states:
                p_one   = p_two / 2 if t_stat > 0 else 1 - p_two / 2
                direction = "mean > 0 (bullish)"
            else:
                p_one   = np.nan
                direction = "no hypothesis"

            results.append({
                "state"            : int(s),
                "horizon"          : h,
                "n"                : n,
                "mean_return"      : mean,
                "std"              : std,
                "hit_ratio"        : (sub > 0).mean(),
                "t_stat"           : t_stat,
                "p_value_one_sided": p_one,
                "hypothesis"       : direction,
            })

    return pd.DataFrame(results)
# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    print("=" * 80)
    print("TRAIN / TEST HMM REGIME ANALYSIS")
    print("=" * 80)

    # ================================================================
    # [1] CHARGEMENT
    # ================================================================
    SHIFT_SPOT_BY_1D = True
    CLIP = 3.5

    data_df = build_full_dataset(shift_spot_by_1d=SHIFT_SPOT_BY_1D)

    print(f"Data shape: {data_df.shape}")
    print(f"Date range: {data_df.index.min()} -> {data_df.index.max()}")

    data_train_raw, data_test_raw = chronological_split_raw(
        data_df, train_ratio=TRAIN_RATIO
    )

    print(f"Raw train: {len(data_train_raw)} | {data_train_raw.index.min().date()} -> {data_train_raw.index.max().date()}")
    print(f"Raw test : {len(data_test_raw)}  | {data_test_raw.index.min().date()} -> {data_test_raw.index.max().date()}")

    # ================================================================
    # [2] FEATURES AVEC WARMUP
    # ================================================================
    print("\n[2] Building features...")

    feat_train = build_hmm_features_for_subset(data_train_raw, window=HMM_WINDOW)

    warmup_size = 80
    df_warmup = pd.concat([data_train_raw.iloc[-warmup_size:], data_test_raw])

    feat_test_raw = build_hmm_features_for_subset(df_warmup, window=HMM_WINDOW)
    feat_test = feat_test_raw.loc[
        feat_test_raw.index.isin(data_test_raw.index)
    ].copy()

    feat_train_full = feat_train.copy()

    data_train = data_train_raw.loc[feat_train.index].copy()
    data_test = data_test_raw.loc[feat_test.index].copy()

    print(f"Jours perdus test (warmup fix): {len(data_test_raw) - len(feat_test)}")
    print(f"Train features shape: {feat_train.shape}")
    print(f"Test  features shape: {feat_test.shape}")

    X_train_full = feat_train.values
    X_test_full = feat_test.values

    X_train_full_z, X_test_full_z, _, _ = standardize(X_train_full, X_test_full)
    X_train_full_z = np.clip(X_train_full_z, -CLIP, CLIP)
    X_test_full_z = np.clip(X_test_full_z, -CLIP, CLIP)

    # ================================================================
    # [3] STANDARDISATION + CLIP (sur toutes les features, avant sélection)
    # ================================================================
    X_train_full = feat_train.values
    X_test_full  = feat_test.values

    X_train_full_z, X_test_full_z, _, _ = standardize(X_train_full, X_test_full)
    X_train_full_z = np.clip(X_train_full_z, -CLIP, CLIP)
    X_test_full_z  = np.clip(X_test_full_z,  -CLIP, CLIP)

    # ================================================================
    # [4] FEATURE SELECTION — une seule fois offline sur le train fixe
    # ================================================================
    print("\n[4] Feature saliency selection...")

    from feature_selection.fshmm_diag import FeatureSaliencyHMMDiag

    fs = FeatureSaliencyHMMDiag(
        n_states      = NB_STATES,
        n_iter        = 100,
        tol           = 1e-4,
        rho_prior_k   = len(X_train_full_z) / 4,
        use_map       = True,
        random_state  = SEED,
        verbose       = True,
    )

    fs.fit(X_train_full_z, feature_names=feat_train.columns)

    ranking = fs.transform_ranking()
    print(ranking.to_string(index=False))

    selected_cols = fs.selected_features(threshold=0.50)
    print(f"\nSelected {len(selected_cols)} features: {selected_cols}")

    # ================================================================
    # [5] RE-STANDARDISATION sur les features sélectionnées
    # ================================================================
    feat_train_sel = feat_train[selected_cols].copy()
    feat_test_sel  = feat_test[selected_cols].copy()

    X_train_z, X_test_z, feat_mu, feat_sd = standardize(
        feat_train_sel.values,
        feat_test_sel.values,
    )
    X_train_z = np.clip(X_train_z, -CLIP, CLIP)
    X_test_z  = np.clip(X_test_z,  -CLIP, CLIP)

    print(f"\nTrain Z shape: {X_train_z.shape} | Test Z shape: {X_test_z.shape}")

    # ================================================================
    # [6] FIT HMM SUR TRAIN FIXE
    # ================================================================
    # ================================================================
# [6] FIT HMM SUR TRAIN FIXE
# ================================================================
    print("\n[6] Fitting HMM on fixed train...")

    model, result = fit_hmm(X_train_z)

    # --- Validation qualité fit ---
    final_ll = result.ll_hist[-1]
    nu_vals  = np.array(result.nu) if hasattr(result, "nu") and result.nu is not None else np.array([])

    ll_ok  = final_ll > 0
    nu_ok  = len(nu_vals) == 0 or np.all(nu_vals < 50)

    # Vérification états sous-peuplés
    states_check, _, _ = model.filter(X_train_z)   # <-- CORRIGÉ
    counts = np.bincount(states_check, minlength=NB_STATES) / len(states_check)
    pop_ok = np.all(counts >= 0.03)

    if not (ll_ok and nu_ok and pop_ok):
        print(
            f"  ⚠ Fit dégénéré "
            f"(LL={final_ll:.1f}, nu_max={nu_vals.max() if len(nu_vals)>0 else 'N/A'}, "
            f"counts={np.round(counts,2)})"
        )
        return   # <-- CORRIGÉ

    # --- Validation du fit ---
    final_ll = result.ll_hist[-1]
    nu_vals  = result.nu if hasattr(result, "nu") and result.nu is not None else np.array([])

    print(f"Final train LL : {final_ll:.4f}")
    print(f"pi             : {np.round(result.pi, 4)}")
    print(f"diag(A)        : {np.round(np.diag(result.A), 4)}")

    if len(nu_vals) > 0:
        print(f"nu             : {np.round(nu_vals, 3)}")

    # Warnings sur qualité du fit
    if final_ll < 0:
        print("⚠ AVERTISSEMENT : LL négatif sur le train fixe — modèle potentiellement dégénéré")

    if len(nu_vals) > 0 and np.any(np.array(nu_vals) > 50):
        print(f"⚠ AVERTISSEMENT : nu dégénéré {np.round(nu_vals, 1)} — vérifier le modèle")
    # ================================================================
    # [7] INFERENCE TRAIN
    # ================================================================
    print("\n[7] Inference on TRAIN...")

    train_states_filt,   train_alpha, _ = model.filter(X_train_z)
    train_states_smooth, train_gamma, _ = model.smooth(X_train_z)

    # Vérification état sous-peuplé
    state_counts_train = np.bincount(train_states_filt, minlength=NB_STATES) / len(train_states_filt)
    if np.any(state_counts_train < 0.03):
        print(f"⚠ AVERTISSEMENT : état sous-peuplé train {np.round(state_counts_train, 3)}")

    # Réordonnancement par mean r_mid (croissant = bearish → bullish)
    train_states_smooth_ord, train_gamma_ord, mapping, train_means = (
        reorder_states_by_mean_return(
            train_states_smooth,
            train_gamma,
            feat_train_full.loc[feat_train_sel.index, "r_mid"].values,
        )
    )

    train_states_filt_ord = np.array([mapping[s] for s in train_states_filt], dtype=int)
    train_alpha_ord = np.zeros_like(train_alpha)
    for old, new in mapping.items():
        train_alpha_ord[:, new] = train_alpha[:, old]

    print("Mapping états (old → new) :", mapping)
    print("Mean r_mid par état (après réordonnancement) :")
    print(train_means)

    # Profil des régimes train
    regime_profile = feat_train_sel.copy()
    regime_profile["state"] = train_states_filt_ord

    profile_cols = ["rolling_vol", "downside_vol", "spread_vol", "momentum"]
    profile_cols = [c for c in profile_cols if c in regime_profile.columns]

    if profile_cols:
        profile = regime_profile.groupby("state")[profile_cols].mean().round(4)
        print("\n=== REGIME PROFILE — TRAIN ===")
        print(profile)

    # ================================================================
    # [8] SEGMENTATION TRAIN
    # ================================================================
    print("\n[8] Segmentation TRAIN...")

    train_seg_book, blocks_train_book = segment_hmm_book_style(
        df     = feat_train_sel,
        probas = train_alpha_ord,
    )

    print(blocks_train_book.to_string(index=False))
    print("\nRésumé blocs par régime :")
    print(
        blocks_train_book.groupby("state")["duration"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )

    plot_hmm_regimes_background(
        df        = train_seg_book.join(data_train[["MID"]], how="left"),
        state_col = "state",
        price_col = "MID",
        title     = "TRAIN — HMM Regimes",
    )

    # ================================================================
    # [9] INFERENCE TEST
    # ================================================================
    print("\n[9] Inference on TEST...")

    test_states_filt,   test_alpha, _ = model.filter(X_test_z)
    test_states_smooth, test_gamma, _ = model.smooth(X_test_z)

    test_states_filt_ord   = np.array([mapping[s] for s in test_states_filt],   dtype=int)
    test_states_smooth_ord = np.array([mapping[s] for s in test_states_smooth], dtype=int)

    test_alpha_ord = np.zeros_like(test_alpha)
    for old, new in mapping.items():
        test_alpha_ord[:, new] = test_alpha[:, old]

    # Profil régimes test
    regime_profile_test = feat_test_sel.copy()
    regime_profile_test["state"] = test_states_filt_ord

    if profile_cols:
        profile_test = regime_profile_test.groupby("state")[profile_cols].mean().round(4)
        print("\n=== REGIME PROFILE — TEST ===")
        print(profile_test)

    # ================================================================
    # [10] SEGMENTATION TEST
    # ================================================================
    print("\n[10] Segmentation TEST...")

    test_seg_book, blocks_test_book = segment_hmm_book_style(
        df     = feat_test_sel,
        probas = test_alpha_ord,
    )

    print(blocks_test_book.to_string(index=False))
    print("\nRésumé blocs par régime :")
    print(
        blocks_test_book.groupby("state")["duration"]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
    )

    plot_hmm_regimes_background(
        df        = test_seg_book.join(data_test[["MID"]], how="left"),
        state_col = "state",
        price_col = "MID",
        title     = "TEST — HMM Regimes",
    )

    # ================================================================
    # [11] BACKTEST TEST FIXE
    # ================================================================
    print("\n[11] Backtest TEST fixe...")

    from simulations.backtester import backtest, print_metrics, plot_backtest

    exposure_test = build_exposure_from_proba_multi(
        probas            = test_alpha_ord,
        entry_threshold   = 0.70,
        exit_threshold    = 0.50,
        reverse_threshold = 0.70,
        max_holding       = 20,
        min_confirm       = 2,
    )

    bt_test = backtest(
        prices       = pd.to_numeric(data_test["FUTURES_CLOSE"], errors="coerce").values,
        states       = test_states_filt_ord,
        exposures    = exposure_test,
        vol_target   = 0.15,
        vol_lookback = 20,
        max_leverage = 2.0,
    )

    print_metrics(bt_test)

    plot_backtest(
        result       = bt_test,
        dates        = data_test.index,
        prices       = pd.to_numeric(data_test["FUTURES_CLOSE"], errors="coerce").values,
        states       = test_states_filt_ord,
        vol_target   = 0.15,
        max_leverage = 2.0,
    )

    # ================================================================
    # [12] PHASE 2 — ROLLING RETRAIN OOS
    # ================================================================
    print("\n" + "=" * 80)
    print("PHASE 2 — ROLLING RETRAIN OOS")
    print("=" * 80)

    TRAIN_WINDOW  = 400
    RETRAIN_EVERY = 42

    rolling_results = rolling_hmm_pipeline(
        data_df           = data_df,
        selected_features = selected_cols,   # figé depuis phase 1
        mapping_ref       = mapping,          # mapping de référence figé
        train_window      = TRAIN_WINDOW,
        retrain_every     = RETRAIN_EVERY,
        hmm_window        = HMM_WINDOW,
        clip_val          = CLIP,
    )

    if rolling_results.empty:
        print("⚠ Aucun résultat rolling — vérifier les paramètres")
        return

    print(f"\nRolling OOS shape    : {rolling_results.shape}")
    print(f"Période OOS          : {rolling_results.index.min().date()} → {rolling_results.index.max().date()}")
    print(f"Jours OOS totaux     : {len(rolling_results)}")
    print("\nDistribution états OOS :")
    print(rolling_results["state"].value_counts().sort_index())
    # Diagnostic retrains utilisés
    print("\nDiagnostic retrains OOS :")
    print(rolling_results.groupby("retrain_id").size().rename("n_obs_oos"))

    n_retrains_used = rolling_results["retrain_id"].nunique()
    print(f"Retrains valides utilisés : {n_retrains_used} / 11")
    rolling_full = rolling_results.join(data_df[["FUTURES_CLOSE", "MID"]], how="left")
    rolling_states = rolling_results["state"].values
    if n_retrains_used == 11:
        print("⚠ ATTENTION : tous les retrains inclus — vérifier le skip dégénéré")

    rolling_full = rolling_results.join(
        data_df[["FUTURES_CLOSE", "MID"]],
        how="left"
    )

    rolling_states = rolling_results["state"].values
    # ================================================================
    # [13] E[r|state] — PREUVE PRINCIPALE
    # ================================================================
    print("\n[13] Predictive analysis E[r|state]...")

    pred_df = build_predictive_df(
        data_df        = rolling_full,
        states         = rolling_states,
        target_col     = "FUTURES_CLOSE",
        use_log_return = True,
        horizon        = 1,
    )

    results_pred = analyze_future_return_by_state(
        df         = pred_df,
        state_col  = "state",
        return_col = "future_return",
    )

    print_future_return_analysis(
        results_pred,
        title = "E[futures_return_t+1 | spot_state_t] — ROLLING OOS",
    )

    # ================================================================
    # [14] TESTS OU — MEAN REVERSION PAR RÉGIME
    # ================================================================
    print("\n[14] OU mean reversion tests...")

    ou_model, ou_results = test_ou_futures_by_spot_state_on_test(
        df_test     = rolling_full,
        states_test = rolling_states,
        futures_col = "FUTURES_CLOSE",
    )

    print("\n=== OU MEAN REVERSION — ROLLING OOS ===")
    print(ou_results[[
        "state", "n_obs", "phi",
        "p_value_phi_lt_1", "half_life_periods", "conclusion"
    ]].to_string(index=False))

    # ================================================================
    # [15] FORWARD RETURNS MULTI-HORIZON
    # ================================================================
    # ================================================================
    # [15] FORWARD RETURNS & DIRECTIONAL TESTS — K=4
    # ================================================================
    print("\n[15] Forward returns & directional tests...")

    fwd_df = build_forward_returns(
        rolling_full,
        price_col = "FUTURES_CLOSE",
        horizons  = (1, 3, 5, 10, 15, 20, 30),
    )
    fwd_df["state"] = rolling_states

    # Version K=4 avec hypothèses correctement assignées
    hac_forward = test_forward_return_hac(
    fwd_df,
    state_col="state",
    horizons=(1, 3, 5, 10, 15, 20, 30),
)   
    directional = test_directional_hypothesis_k4(
    fwd_df,
    state_col="state",
    horizons=(3, 5, 10, 15, 20),
    bearish_states=(1,),
    bullish_states=(3,),
)

    print("\n=== FORWARD RETURNS — HAC TESTS ===")
    print(hac_forward.to_string(index=False))

    print("\n=== DIRECTIONAL HYPOTHESIS TESTS — ROLLING OOS ===")
    print(directional.to_string(index=False))

    # Correction Holm sur les tests ayant une hypothèse directionnelle
    from statsmodels.stats.multitest import multipletests

    dir_valid = directional.dropna(subset=["p_value_one_sided"]).copy()

    if len(dir_valid) >= 2:
        _, p_holm, _, _ = multipletests(
            dir_valid["p_value_one_sided"].values,
            method="holm",
        )
        dir_valid = dir_valid.copy()
        dir_valid["p_holm"] = p_holm

        print("\n=== DIRECTIONAL TESTS — APRÈS CORRECTION HOLM ===")
        print(dir_valid[[
            "state", "horizon", "n",
            "mean_return", "t_stat",
            "p_value_one_sided", "p_holm", "hypothesis",
        ]].to_string(index=False))

        sig = dir_valid[dir_valid["p_holm"] < 0.05]
        print(f"\nTests significatifs après Holm (p<0.05) : {len(sig)}")
        if not sig.empty:
            print(sig[[
                "state", "horizon", "mean_return", "t_stat", "p_holm"
            ]].to_string(index=False))

    plot_hedge_fund_forward_impact(
        df        = rolling_full,
        states    = rolling_states,
        price_col = "FUTURES_CLOSE",
        horizons  = (0, 1, 3, 5, 10, 15, 20, 30),
        title     = "Spot Regime → Futures Forward Impact (Rolling OOS)",
    )
    # [16] BACKTEST ROLLING OOS — adapté K=4
    print("\n[16] Backtest rolling OOS...")

    # ================================================================
    # [16] BACKTEST ROLLING OOS + IC SHARPE
    # ================================================================
    print("\n[16] Backtest rolling OOS...")

    from simulations.backtester import backtest, print_metrics, plot_backtest

    K      = NB_STATES
    p_cols = [f"p{k}" for k in range(K) if f"p{k}" in rolling_results.columns]

    if len(p_cols) != K:
        print(f"⚠ Probas incomplètes ({len(p_cols)}/{K})")
        ic_low, ic_high, sr_ann = np.nan, np.nan, np.nan
    else:
        rolling_probas = rolling_results[p_cols].values
        rolling_prices = pd.to_numeric(
            rolling_full["FUTURES_CLOSE"], errors="coerce"
        ).values

        BULLISH_STATE = 3
        BEARISH_STATE = 1

        exposure_rolling = build_exposure_k4(
            probas          = rolling_probas,
            bullish_state   = BULLISH_STATE,
            bearish_state   = BEARISH_STATE,
            entry_threshold = 0.65,
            exit_threshold  = 0.45,
            max_holding     = 15,
            min_confirm     = 2,
        )

        print(f"  Long sur state {BULLISH_STATE} | Short sur state {BEARISH_STATE}")
        print(f"  Jours investis : {(exposure_rolling != 0).sum()} / {len(exposure_rolling)}")

        bt_rolling = backtest(
            prices       = rolling_prices,
            states       = rolling_states,
            exposures    = exposure_rolling,
            vol_target   = 0.15,
            vol_lookback = 20,
            max_leverage = 2.0,
        )

        print("\n=== BACKTEST ROLLING OOS — K=4 ===")
        print_metrics(bt_rolling)

        plot_backtest(
            result       = bt_rolling,
            dates        = rolling_results.index,
            prices       = rolling_prices,
            states       = rolling_states,
            vol_target   = 0.15,
            max_leverage = 2.0,
        )

        # --- IC Sharpe ---
        try:
            price_s    = pd.Series(rolling_prices)
            raw_rets   = np.log(price_s / price_s.shift(1)).dropna()
            exp_s      = pd.Series(exposure_rolling[:len(raw_rets)])
            strat_rets = pd.Series(
                (exp_s.values * raw_rets.values)
            ).replace(0, np.nan).dropna()

            n_obs  = len(strat_rets)
            sr_daily = strat_rets.mean() / strat_rets.std()
            sr_ann = np.sqrt(252) * sr_daily

            se_sr_daily = np.sqrt((1 + 0.5 * sr_daily**2) / n_obs)
            se_sr_ann = np.sqrt(252) * se_sr_daily

            ic_low  = sr_ann - 1.96 * se_sr_ann
            ic_high = sr_ann + 1.96 * se_sr_ann

            print(f"\n=== SHARPE ROLLING — IC 95% ===")
            print(f"  Sharpe annualisé   : {sr_ann:.3f}")
            
            print(f"  IC 95%             : [{ic_low:.3f}, {ic_high:.3f}]")
            print(f"  N observations     : {n_obs}")

            if ic_low > 0:
                print("  ✓ IC entièrement positif — robuste")
            elif ic_high > 0:
                print("  ~ IC inclut 0 — signal présent mais incertain")
            else:
                print("  ✗ IC négatif")

        except Exception as e:
            print(f"⚠ IC Sharpe non calculé : {e}")
            ic_low, ic_high, sr_ann = np.nan, np.nan, np.nan
        # ================================================================
# [17] TABLE DE SYNTHÈSE — MEMO HEDGE FUND
# ================================================================
# ================================================================
    # [17] SYNTHÈSE MEMO HEDGE FUND
    # ================================================================
    print("\n" + "=" * 80)
    print("SYNTHÈSE — VALEUR INFORMATIONNELLE DATA SPOT (ROLLING OOS)")
    print("=" * 80)

    # Tests directionnels des états clés
    summary_rows = []
    for _, row in directional.iterrows():
        if row["state"] in [1, 3] and pd.notna(row["p_value_one_sided"]):
            summary_rows.append({
                "Régime"        : f"State {int(row['state'])}",
                "Horizon"       : f"J+{int(row['horizon'])}",
                "N obs OOS"     : int(row["n"]),
                "Rendement moy" : f"{row['mean_return']*100:.2f}%",
                "Hit ratio"     : f"{row['hit_ratio']*100:.0f}%",
                "t-stat"        : f"{row['t_stat']:.2f}",
                "p-value"       : f"{row['p_value_one_sided']:.4f}",
            })

    if summary_rows:
        print("\nPrédictibilité futures CBOT Corn par régime spot :")
        print(pd.DataFrame(summary_rows).to_string(index=False))

    # Mean reversion
    print("\nDynamique mean-reversion futures par régime spot :")
    print(ou_results[[
        "state", "n_obs", "phi",
        "p_value_phi_lt_1", "half_life_periods", "conclusion"
    ]].to_string(index=False))

    # Performance
    ic_str = (
        f"[{ic_low:.2f}, {ic_high:.2f}]"
        if not np.isnan(ic_low) else "N/A"
    )

    
if __name__ == "__main__":
    main()
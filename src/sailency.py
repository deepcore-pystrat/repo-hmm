# ============================================================
# FSHMM FEATURE SELECTION ONLY
# Spot-only features → Feature Saliency HMM → relevant features
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from feature_selection.saliency_selector import FeatureSaliencySelector


# ============================================================
# CONFIG
# ============================================================

SPOT_PATH = r"D:\Downloads\data_spot_corn_brz.xlsx"

TRAIN_RATIO = 0.65
NB_STATES = 3
SEED = 42
CLIP = 3.5
EPS = 1e-10


# ============================================================
# DATA
# ============================================================

def load_spot():
    df = pd.read_excel(SPOT_PATH, header=2)
    df.columns = df.columns.str.strip()

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["BID"] = pd.to_numeric(df["BID"], errors="coerce")
    df["OFFER"] = pd.to_numeric(df["OFFER"], errors="coerce")

    df["BID"] = df["BID"].ffill(limit=3)
    df["OFFER"] = df["OFFER"].ffill(limit=3)

    df["MID"] = (df["BID"] + df["OFFER"]) / 2.0
    df["BA_SPREAD"] = df["OFFER"] - df["BID"]

    df = (
        df[["DATE", "FUTURES", "BID", "OFFER", "MID", "BA_SPREAD"]]
        .dropna(subset=["DATE", "FUTURES", "MID", "BID", "OFFER"])
        .sort_values("DATE")
        .rename(columns={"DATE": "Date"})
        .groupby("Date", as_index=False)
        .last()
    )

    df = df[
        (df["MID"] > 0)
        & (df["BID"] > 0)
        & (df["OFFER"] > 0)
        & (df["BA_SPREAD"] >= 0)
    ]

    return df.set_index("Date")


def build_dataset(shift_spot=True):
    data = load_spot()

    if shift_spot:
        for c in ["BID", "OFFER", "MID", "BA_SPREAD"]:
            data[c] = data[c].shift(1)

    data = data.dropna(subset=["MID"])
    data = data[
    (data["MID"] > 0) &
    (data["BA_SPREAD"] >= 0)
]

    return data


# ============================================================
# UTILS
# ============================================================

def _variance_ratio(x, lag=5):
    s = pd.Series(x).dropna()
    if len(s) < lag + 2:
        return np.nan

    r1 = s.diff().dropna()
    rq = s.diff(lag).dropna()

    v1 = r1.var()
    if v1 < EPS:
        return np.nan

    return float(rq.var() / (lag * v1))


def _ar1_beta(x):
    s = pd.Series(x).dropna()
    if len(s) < 5:
        return np.nan

    y = s.diff().dropna()
    x_lag = s.shift(1).dropna().reindex(y.index)

    if x_lag.std() < EPS:
        return np.nan

    return float(np.cov(y.values, x_lag.values)[0, 1] / (x_lag.var() + EPS))


def _sign_entropy(x):
    s = pd.Series(x).dropna()
    if len(s) < 5:
        return np.nan

    signs = np.sign(s.values)
    p = np.array([
        (signs > 0).mean(),
        (signs < 0).mean(),
        (signs == 0).mean(),
    ])

    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _mad_zscore(x, window):
    med = x.rolling(window).median()

    mad = (
        (x - med)
        .abs()
        .rolling(window)
        .median()
    )

    # garde-fou robuste
    mad = mad.clip(lower=0.1)

    z = (x - med) / mad

    # winsorisation
    z = z.clip(-10, 10)

    return z


# ============================================================
# FULL FEATURE PORTFOLIO
# ============================================================

def build_features(df, eps=1e-10):
    """
    Robust spot/basis feature engineering for latent regime detection.

    Philosophy:
    - MID is treated as a spot basis / OTC quote level, not necessarily as a classical asset price.
    - Use absolute changes instead of log-returns.
    - Spread/liquidity features are central.
    - All features are lagged to avoid look-ahead.
    """

    bid = pd.to_numeric(df["BID"], errors="coerce")
    ask = pd.to_numeric(df["OFFER"], errors="coerce")
    mid = pd.to_numeric(df["MID"], errors="coerce")
    spread = pd.to_numeric(df["BA_SPREAD"], errors="coerce")

    out = pd.DataFrame(index=df.index)

    # ============================================================
    # Lagged raw series
    # ============================================================
    mid_lag = mid.shift(1)
    bid_lag = bid.shift(1)
    ask_lag = ask.shift(1)
    spread_lag = spread.shift(1)

    d_mid = mid.diff().shift(1)
    d_bid = bid.diff().shift(1)
    d_ask = ask.diff().shift(1)
    d_spread = spread.diff().shift(1)

    # ============================================================
    # 1. Basis / spot level dynamics
    # ============================================================
    out["chg_1"] = d_mid
    out["chg_3"] = mid_lag - mid.shift(4)
    out["chg_5"] = mid_lag - mid.shift(6)
    out["chg_10"] = mid_lag - mid.shift(11)
    out["chg_20"] = mid_lag - mid.shift(21)
    out["chg_60"] = mid_lag - mid.shift(61)

    out["abs_chg_5"] = out["chg_5"].abs()
    out["abs_chg_20"] = out["chg_20"].abs()
    out["abs_chg_60"] = out["chg_60"].abs()

    # ============================================================
    # 2. Volatility / stress of basis changes
    # ============================================================
    vol_5 = d_mid.rolling(5).std()
    vol_10 = d_mid.rolling(10).std()
    vol_20 = d_mid.rolling(20).std()
    vol_60 = d_mid.rolling(60).std()

    out["vol_chg_5"] = vol_5
    out["vol_chg_10"] = vol_10
    out["vol_chg_20"] = vol_20
    out["vol_chg_60"] = vol_60

    out["vol_ratio_5_20"] = vol_5 / (vol_20 + eps)
    out["vol_ratio_20_60"] = vol_20 / (vol_60 + eps)

    d_down = d_mid.where(d_mid < 0, 0.0)
    d_up = d_mid.where(d_mid > 0, 0.0)

    out["downside_chg_vol_20"] = d_down.rolling(20).std()
    out["upside_chg_vol_20"] = d_up.rolling(20).std()
    out["vol_skew_diff_20"] = (
        out["downside_chg_vol_20"] - out["upside_chg_vol_20"]
    )

    out["chg_skew_20"] = d_mid.rolling(20).skew()
    out["chg_kurt_20"] = d_mid.rolling(20).kurt()

    # ============================================================
    # 3. Path structure: trend, chop, mean reversion
    # ============================================================
    path_20 = d_mid.abs().rolling(20).sum()
    path_60 = d_mid.abs().rolling(60).sum()

    out["efficiency_20"] = out["chg_20"].abs() / (path_20 + eps)
    out["efficiency_60"] = out["chg_60"].abs() / (path_60 + eps)

    out["signchg_20"] = d_mid.rolling(20).apply(
        lambda x: np.mean(np.sign(pd.Series(x).dropna()).diff().abs() > 0)
        if len(pd.Series(x).dropna()) > 3 else np.nan,
        raw=False,
    )

    out["signchg_60"] = d_mid.rolling(60).apply(
        lambda x: np.mean(np.sign(pd.Series(x).dropna()).diff().abs() > 0)
        if len(pd.Series(x).dropna()) > 5 else np.nan,
        raw=False,
    )

    out["autocorr_1_20"] = d_mid.rolling(20).apply(
        lambda x: pd.Series(x).dropna().autocorr(lag=1)
        if len(pd.Series(x).dropna()) > 5 else np.nan,
        raw=False,
    )

    out["autocorr_1_60"] = d_mid.rolling(60).apply(
        lambda x: pd.Series(x).dropna().autocorr(lag=1)
        if len(pd.Series(x).dropna()) > 10 else np.nan,
        raw=False,
    )

    out["vr_5_20"] = d_mid.rolling(20).apply(
        lambda x: _variance_ratio(x, lag=5),
        raw=False,
    )

    out["vr_5_60"] = d_mid.rolling(60).apply(
        lambda x: _variance_ratio(x, lag=5),
        raw=False,
    )

    # ============================================================
    # 4. Range / drawdown in absolute basis space
    # ============================================================
    roll_max_20 = mid_lag.rolling(20).max()
    roll_min_20 = mid_lag.rolling(20).min()
    roll_max_60 = mid_lag.rolling(60).max()
    roll_min_60 = mid_lag.rolling(60).min()

    out["range_abs_20"] = roll_max_20 - roll_min_20
    out["range_abs_60"] = roll_max_60 - roll_min_60
    out["range_ratio_20_60"] = out["range_abs_20"] / (out["range_abs_60"] + eps)

    out["dd_abs_20"] = mid_lag - roll_max_20
    out["dd_abs_60"] = mid_lag - roll_max_60
    out["abs_dd_20"] = out["dd_abs_20"].abs()
    out["abs_dd_60"] = out["dd_abs_60"].abs()

    # ============================================================
    # 5. Spread / liquidity
    # ============================================================
    out["spread"] = spread_lag
    out["spread_20"] = spread_lag.rolling(20).mean()
    out["spread_60"] = spread_lag.rolling(60).mean()

    out["spread_vol_20"] = spread_lag.rolling(20).std()
    out["spread_vol_60"] = spread_lag.rolling(60).std()

    out["spread_chg_1"] = d_spread
    out["spread_chg_5"] = spread_lag - spread.shift(6)
    out["spread_chg_20"] = spread_lag - spread.shift(21)

    spread_med_60 = spread_lag.rolling(60).median()
    spread_mad_60 = spread_lag.rolling(60).apply(
        lambda x: np.median(np.abs(x - np.median(x))),
        raw=True,
    )

    out["spread_mz_60"] = (spread_lag - spread_med_60) / (spread_mad_60 + eps)

    out["spread_stress_20"] = (
        out["spread_mz_60"] > 1.5
    ).rolling(20).sum()

    # Relative spread kept, but not dominant
    rel_spread = spread_lag / (mid_lag.abs() + eps)

    out["rel_spread"] = rel_spread
    out["rel_spread_mz_60"] = _mad_zscore(rel_spread, 60)
    out["rel_spread_vol_20"] = rel_spread.rolling(20).std()

    # Robust spread shock, less explosive than old spread_shock_mz
    out["spread_shock_5"] = d_spread.abs().rolling(5).mean()
    out["spread_shock_20"] = d_spread.abs().rolling(20).mean()

    # ============================================================
    # 6. Bid / ask asymmetry
    # ============================================================
    out["bid_chg_1"] = d_bid
    out["ask_chg_1"] = d_ask

    out["bid_vol_20"] = d_bid.rolling(20).std()
    out["ask_vol_20"] = d_ask.rolling(20).std()

    out["ba_vol_gap_20"] = out["ask_vol_20"] - out["bid_vol_20"]
    out["quote_slope"] = d_ask - d_bid

    out["ba_corr_20"] = d_bid.rolling(20).corr(d_ask)
    from scipy.stats import linregress

    def rolling_r2(x):
        y = np.asarray(x)
        t = np.arange(len(y))

        if np.isnan(y).any():
            return np.nan

        r = linregress(t, y)
        return r.rvalue ** 2

    out["trend_r2_20"] = mid_lag.rolling(20).apply(
        rolling_r2,
        raw=False
    )

    out["trend_r2_60"] = mid_lag.rolling(60).apply(
        rolling_r2,
        raw=False
    )
    def rolling_slope(x):
        y = np.asarray(x)
        t = np.arange(len(y))

        if np.isnan(y).any():
            return np.nan

        return np.polyfit(t, y, 1)[0]

    out["trend_slope_20"] = mid_lag.rolling(20).apply(
        rolling_slope,
        raw=False
    )

    out["trend_slope_60"] = mid_lag.rolling(60).apply(
        rolling_slope,
        raw=False
    )
    out["vol_compression"] = (
    vol_20 / (vol_60 + eps)
)
    ma60 = mid_lag.rolling(60).mean()
    std60 = mid_lag.rolling(60).std()

    out["zscore_ma60"] = (
        (mid_lag - ma60)
        / (std60 + eps)
    )
    roll_max = mid_lag.rolling(60).max()
    roll_min = mid_lag.rolling(60).min()

    out["breakout_up"] = (
        mid_lag / (roll_max + eps)
    )

    out["breakout_down"] = (
        mid_lag / (roll_min + eps)
    )
    # ============================================================
    # 7. Liquidity / price impact proxies
    # ============================================================
    out["impact_proxy_20"] = d_mid.abs().rolling(20).mean() / (
        spread_lag.rolling(20).mean() + eps
    )

    out["impact_proxy_60"] = d_mid.abs().rolling(60).mean() / (
        spread_lag.rolling(60).mean() + eps
    )

    out["spread_to_vol_20"] = spread_lag.rolling(20).mean() / (vol_20 + eps)
    # ============================================================
# 8. Mean reversion / trending structure (Hurst-like)
# ============================================================

# Hurst exponent approximé par R/S rolling
    def _hurst_rs(x):
        x = pd.Series(x).dropna()
        if len(x) < 10:
            return np.nan
        mean = x.mean()
        deviations = (x - mean).cumsum()
        R = deviations.max() - deviations.min()
        S = x.std()
        if S < eps:
            return np.nan
        return np.log(R / S) / np.log(len(x))

    out["hurst_20"] = d_mid.rolling(20).apply(_hurst_rs, raw=False)
    out["hurst_60"] = d_mid.rolling(60).apply(_hurst_rs, raw=False)

    # Ornstein-Uhlenbeck half-life (vitesse de mean-reversion)
    # half_life = -log(2) / log(beta) où beta = coeff AR(1)
    def _ou_halflife(x):
        x = pd.Series(x).dropna()
        if len(x) < 5:
            return np.nan
        y = x.diff().dropna()
        x_lag = x.shift(1).dropna().reindex(y.index)
        if x_lag.std() < eps:
            return np.nan
        beta = np.cov(y.values, x_lag.values)[0, 1] / (x_lag.var() + eps)
        if beta >= 0 or beta <= -1:
            return np.nan
        return -np.log(2) / np.log(1 + beta)

    out["ou_halflife_20"] = d_mid.rolling(20).apply(_ou_halflife, raw=False)
    out["ou_halflife_60"] = d_mid.rolling(60).apply(_ou_halflife, raw=False)

    # Price position dans son range (0 = bas du range, 1 = haut)
    out["price_position_20"] = (mid_lag - roll_min_20) / (out["range_abs_20"] + eps)
    out["price_position_60"] = (mid_lag - roll_min_60) / (out["range_abs_60"] + eps)
    # ============================================================
    # Final cleaning
    # ============================================================
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=1, how="all")

    return out

SALIENCY_FEATURE_POOL = [
    # Direction / momentum
    "chg_20",
    "chg_60",
    "trend_slope_20",
    "trend_slope_60",

    # Qualité de tendance / choppiness
    "trend_r2_20",
    "trend_r2_60",
    "efficiency_20",
    "efficiency_60",
    "signchg_20",

    # Volatilité / stress
    "vol_chg_20",
    "vol_chg_60",
    "vol_ratio_20_60",
    "downside_chg_vol_20",
    "chg_kurt_20",

    # Mean-reversion / persistence
    "autocorr_1_20",
    "vr_5_20",
    "hurst_20",
    "hurst_60",
    "ou_halflife_20",

    # Position dans le range
    "price_position_20",
    "price_position_60",
    "zscore_ma60",
    "abs_dd_60",
    "range_ratio_20_60",

    # Liquidity / spread
    "spread_20",
    "spread_vol_20",
    "spread_mz_60",
    "spread_stress_20",
    "rel_spread_mz_60",
    "spread_shock_20",

    # Impact / microstructure
    "ba_vol_gap_20",
    "quote_slope",
    "impact_proxy_20",
    "spread_to_vol_20",
]

# ============================================================
# FSHMM FEATURE SELECTION
# ============================================================

def run_fshmm_feature_selection():
    print("=" * 80)
    print("FSHMM FEATURE SELECTION ONLY")
    print("=" * 80)

    data = build_dataset(shift_spot=True)

    n = len(data)
    n_train = int(n * TRAIN_RATIO)
    data_train = data.iloc[:n_train].copy()

    feat_train = build_features(data_train)
    missing_rate = feat_train.isna().mean().sort_values(ascending=False)

    print("\nTop missing-rate features:")
    print(missing_rate.head(20).round(3).to_string())
    BURN_IN = 60                                    # ← ajouter
    feat_train = feat_train.iloc[BURN_IN:].copy()   # ← ajouter
    candidate_cols = [
        c for c in SALIENCY_FEATURE_POOL
        if c in feat_train.columns
    ]

    if len(candidate_cols) < 3:
        raise ValueError("Pas assez de features candidates.")

    X_df = feat_train[candidate_cols].copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    # Supprimer colonnes totalement vides
    X_df = X_df.dropna(axis=1, how="all")

    # Forward-fill court pour trous isolés
    X_df = X_df.ffill(limit=3)

    # Puis médiane train-only pour le reste
    X_df = X_df.fillna(X_df.median())

    # Sécurité finale
    X_df = X_df.dropna(axis=0, how="any")

    nunique = X_df.nunique(dropna=True)
    X_df = X_df.loc[:, nunique > 3]

    std = X_df.std()
    X_df = X_df.loc[:, std > 1e-10]

    final_candidates = list(X_df.columns)

    print(f"\nNombre de features candidates propres : {len(final_candidates)}")

    scaler = RobustScaler(
    with_centering=True,
    with_scaling=True,
    quantile_range=(25.0, 75.0)
)

    X = scaler.fit_transform(X_df[final_candidates])
    X = np.clip(X, -CLIP, CLIP)

    selector = FeatureSaliencySelector(
        n_states=NB_STATES,
        threshold=0.70,
        n_init=10,
        n_iter=150,
        rho_prior_k=2.0,
        random_state=SEED,
        verbose=True,
    )

    selector.fit(X, feature_names=final_candidates)

    ranking = selector.get_ranking().sort_values("rho", ascending=False)
    selected = selector.get_selected_features()

    print("\n" + "=" * 80)
    print("FEATURE SALIENCY RANKING")
    print("=" * 80)
    print(ranking.to_string(index=False))

    print("\n" + "=" * 80)
    print("SELECTED FEATURES")
    print("=" * 80)
    print(selected)

    ranking.to_csv("fshmm_feature_saliency_ranking.csv", index=False)

    pd.DataFrame({
        "selected_features": selected
    }).to_csv("fshmm_selected_features.csv", index=False)

    print("\nSaved:")
    print(" - fshmm_feature_saliency_ranking.csv")
    print(" - fshmm_selected_features.csv")

    return ranking, selected


if __name__ == "__main__":
    # ── Sanity check synthétique ──────────────────────────────
    rng = np.random.default_rng(0)
    T = 800
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        states[t] = 1 if (states[t-1]==0 and rng.random()<0.05) else \
                    0 if (states[t-1]==1 and rng.random()<0.05) else states[t-1]

    X_syn = np.zeros((T, 5))
    X_syn[:,0] = np.where(states==0, rng.normal(-1,0.5,T), rng.normal(+1,0.5,T))
    X_syn[:,1] = np.where(states==0, rng.normal(0,0.3,T),  rng.normal(0,1.5,T))
    X_syn[:,2] = rng.normal(0,1,T)
    X_syn[:,3] = rng.normal(0,1,T)
    X_syn[:,4] = rng.normal(0,1,T)

    sel = FeatureSaliencySelector(n_states=2, threshold=0.5, n_init=5, verbose=False)
    sel.fit(X_syn, feature_names=["f0","f1","f2","f3","f4"])
    print("SANITY CHECK:", sel.get_ranking().to_string(index=False))
    # Attendu : rho(f0) et rho(f1) >> rho(f2,f3,f4)
    assert sel.get_ranking().iloc[0]["feature"] in ["f0","f1"], "SANITY FAILED"
    assert sel.get_ranking().iloc[1]["feature"] in ["f0","f1"], "SANITY FAILED"
    print("SANITY OK\n")

    ranking, selected_features = run_fshmm_feature_selection()
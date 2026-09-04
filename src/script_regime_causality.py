# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

from features.registry import build_features, standardize
from features.splitter import fixed_split
from models import GaussianHMMConfig, create_hmm


# ============================================================
# 1) LOAD DATA (TA VERSION)
# ============================================================

spot_AR = pd.read_excel(
    r"D:\Downloads\deepcore--2026-04-07-12_42-0026205d-213a-4e96-bbe9-a7c977a1c3d4.xlsx",
    skiprows=1
)

futures = pd.read_csv(
    r"D:\Downloads\Futures CBOT Corn.csv",
    sep=';'
)


# ============================================================
# 2) CLEAN SPOT
# ============================================================

spot_AR["DATE"] = pd.to_datetime(spot_AR["DATE"], errors="coerce")

spot_AR["BID"] = pd.to_numeric(spot_AR["BID"], errors="coerce")
spot_AR["OFFER"] = pd.to_numeric(spot_AR["OFFER"], errors="coerce")

spot_AR["MID"] = (spot_AR["BID"] + spot_AR["OFFER"]) / 2
spot_AR["MID"] = spot_AR["MID"].ffill()

spot_AR["BA_SPREAD"] = spot_AR["OFFER"] - spot_AR["BID"]

spot_AR = spot_AR[["DATE", "MID", "BA_SPREAD"]].copy()
spot_AR = spot_AR.dropna()
spot_AR = spot_AR.sort_values("DATE")
spot_AR = spot_AR.groupby("DATE", as_index=False).last()

spot_AR = spot_AR.rename(columns={"DATE": "Date"})
spot_AR = spot_AR.set_index("Date")


# ============================================================
# 3) CLEAN FUTURES
# ============================================================

futures["Date"] = pd.to_datetime(futures["Date"], errors="coerce")
futures["Close"] = pd.to_numeric(futures["Close"], errors="coerce")

fut = futures[["Date", "Close"]].copy()
fut = fut.dropna()
fut = fut.sort_values("Date")
fut = fut.groupby("Date", as_index=False).last()

fut = fut.rename(columns={"Close": "FUTURES_CLOSE"})
fut = fut.set_index("Date")


# ============================================================
# 4) MERGE
# ============================================================

full_df = spot_AR.join(fut, how="inner")

full_df = full_df[(full_df["MID"] > 0) & (full_df["FUTURES_CLOSE"] > 0)].copy()

print("Data merged:", full_df.shape)


# ============================================================
# 5) BUILD HMM FEATURES
# ============================================================

feature_specs = [
    ("returns", "MID"),
    ("rvol", "MID", {"window": 20}),
    ("momentum", "MID", {"period": 10}),
    ("zscore", "BA_SPREAD", {"window": 20}),
]

feat_df = build_features(
    full_df,
    feature_specs,
    shift_cols={"MID": 1, "BA_SPREAD": 1},
)

feat_df = feat_df.dropna().copy()

X = feat_df.values
idx = feat_df.index


# ============================================================
# 6) SPLIT + STANDARDIZE
# ============================================================

X_train, X_test, idx_train, idx_test = fixed_split(X, idx, train_ratio=0.80)
X_train_z, X_test_z, _, _ = standardize(X_train, X_test)


# ============================================================
# 7) HMM
# ============================================================

hmm_cfg = GaussianHMMConfig(K=3, n_iter=100, seed=42, cov_type="diag")

model = create_hmm(hmm_cfg)
model.fit(X_train_z)

states_train, alpha_train, _ = model.filter(X_train_z)
states_test, alpha_test, _ = model.filter(X_test_z)


# ============================================================
# 8) BUILD REGIME DF (SANS SMOOTH)
# ============================================================

def make_regime_df(idx, states, alpha):
    df = pd.DataFrame(index=idx)
    df.index.name = "Date"

    df["state"] = states

    for k in range(alpha.shape[1]):
        df[f"proba_state_{k}"] = alpha[:, k]

    return df


regime_train = make_regime_df(idx_train, states_train, alpha_train)
regime_test = make_regime_df(idx_test, states_test, alpha_test)


# ============================================================
# 9) BUILD CAUSALITY DATASET
# ============================================================

def prepare_causality_df(df):
    tmp = df.reset_index()

    tmp["log_spot"] = np.log(tmp["MID"])
    tmp["log_fut"] = np.log(tmp["FUTURES_CLOSE"])

    tmp["dlog_spot"] = tmp["log_spot"].diff()
    tmp["dlog_fut"] = tmp["log_fut"].diff()

    tmp = tmp.dropna()

    return tmp


full_train = full_df.loc[idx_train]
full_test = full_df.loc[idx_test]

causality_train = prepare_causality_df(full_train)
causality_test = prepare_causality_df(full_test)


# ============================================================
# 10) MERGE WITH REGIMES (IMPORTANT)
# ============================================================

causality_train = pd.merge(
    causality_train,
    regime_train.reset_index(),
    on="Date",
    how="inner"
)

causality_test = pd.merge(
    causality_test,
    regime_test.reset_index(),
    on="Date",
    how="inner"
)


# ============================================================
# 11) LAG SELECTION
# ============================================================

def choose_var_lag(df, maxlags=8):
    model = VAR(df)
    sel = model.select_order(maxlags=maxlags)

    lag = sel.selected_orders.get("bic", 1)
    if lag is None or lag < 1:
        lag = 1

    return lag


lag = choose_var_lag(causality_train[["dlog_fut", "dlog_spot"]])
print("Lag:", lag)


# ============================================================
# 12) WEIGHTED CAUSALITY
# ============================================================

def weighted_test(df, regime, lag):
    tmp = df.copy()

    for i in range(1, lag + 1):
        tmp[f"fut_lag_{i}"] = tmp["dlog_fut"].shift(i)
        tmp[f"spot_lag_{i}"] = tmp["dlog_spot"].shift(i)

    tmp = tmp.dropna()

    weights = tmp[f"proba_state_{regime}"]

    y = tmp["dlog_fut"]

    X = tmp[
        [f"fut_lag_{i}" for i in range(1, lag + 1)] +
        [f"spot_lag_{i}" for i in range(1, lag + 1)]
    ]

    X = sm.add_constant(X)

    model = sm.WLS(y, X, weights=weights).fit(cov_type="HAC", cov_kwds={"maxlags": lag})

    restriction = " = 0, ".join([f"spot_lag_{i}" for i in range(1, lag + 1)]) + " = 0"
    pval = float(model.wald_test(restriction).pvalue)

    return pval


print("\nTRAIN RESULTS")
for k in range(3):
    pval = weighted_test(causality_train, k, lag)
    print(f"Regime {k} -> p-value = {pval:.4f}")


print("\nTEST RESULTS")
for k in range(3):
    pval = weighted_test(causality_test, k, lag)
    print(f"Regime {k} -> p-value = {pval:.4f}")
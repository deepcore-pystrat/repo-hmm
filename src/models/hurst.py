import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# =========================================================
# 1) Charger les données
# =========================================================
spot_AR = pd.read_excel(
    r"D:\Downloads\deepcore--2026-04-07-12_42-0026205d-213a-4e96-bbe9-a7c977a1c3d4.xlsx",
    skiprows=1
)

# =========================================================
# 2) Fonctions utilitaires
# =========================================================
def adf_report(series, name="series", regression="c"):
    s = pd.Series(series).dropna()
    res = adfuller(s, regression=regression, autolag="AIC")
    out = {
        "name": name,
        "adf_stat": res[0],
        "pvalue": res[1],
        "used_lag": res[2],
        "nobs": res[3],
        "critical_values": res[4]
    }
    return out

def print_adf_result(res):
    print(f"\nADF test on {res['name']}")
    print(f"ADF statistic : {res['adf_stat']:.6f}")
    print(f"p-value       : {res['pvalue']:.6g}")
    print(f"used lag      : {res['used_lag']}")
    print(f"nobs          : {res['nobs']}")
    print("critical values:")
    for k, v in res["critical_values"].items():
        print(f"  {k}: {v:.6f}")

def estimate_ar1_half_life(series):
    """
    Estime un AR(1):
        X_t = alpha + phi X_{t-1} + eps_t

    et aussi la régression en différences:
        ΔX_t = a + b X_{t-1} + eps_t

    Mean reversion:
    - AR(1): phi < 1
    - Delta form: b < 0

    Half-life:
    - AR(1): ln(2)/(-ln(phi)) si 0 < phi < 1
    - Delta approx: ln(2)/(-b) si b < 0
    """
    x = pd.Series(series).dropna().copy()

    # ---------- AR(1) ----------
    x_lag = x.shift(1)
    df_ar = pd.DataFrame({"x": x, "x_lag": x_lag}).dropna()

    X_ar = sm.add_constant(df_ar["x_lag"])
    ar1_model = sm.OLS(df_ar["x"], X_ar).fit()

    alpha = ar1_model.params["const"]
    phi = ar1_model.params["x_lag"]

    if 0 < phi < 1:
        half_life_ar1 = np.log(2) / (-np.log(phi))
    else:
        half_life_ar1 = np.nan

    # ---------- Delta regression ----------
    dx = x.diff()
    x_lag2 = x.shift(1)
    df_delta = pd.DataFrame({"dx": dx, "x_lag": x_lag2}).dropna()

    X_delta = sm.add_constant(df_delta["x_lag"])
    delta_model = sm.OLS(df_delta["dx"], X_delta).fit()

    a = delta_model.params["const"]
    b = delta_model.params["x_lag"]

    if b < 0:
        half_life_delta = np.log(2) / (-b)
    else:
        half_life_delta = np.nan

    return {
        "ar1_model": ar1_model,
        "delta_model": delta_model,
        "alpha": alpha,
        "phi": phi,
        "half_life_ar1_steps": half_life_ar1,
        "a_delta": a,
        "b_delta": b,
        "half_life_delta_steps_approx": half_life_delta
    }

# =========================================================
# 3) Nettoyage des colonnes
# =========================================================
spot_AR["DATE"] = pd.to_datetime(spot_AR["DATE"], errors="coerce")
spot_AR["BID"] = pd.to_numeric(spot_AR["BID"], errors="coerce")
spot_AR["OFFER"] = pd.to_numeric(spot_AR["OFFER"], errors="coerce")

# Garde seulement les lignes valides
quotes = spot_AR.dropna(subset=["DATE", "BID", "OFFER"]).copy()

# Contraintes naturelles
quotes = quotes[(quotes["BID"] > 0) & (quotes["OFFER"] > 0)].copy()
quotes = quotes[quotes["OFFER"] >= quotes["BID"]].copy()

# =========================================================
# 4) Construire le mid-quote spot
# =========================================================
quotes["mid_quote"] = (quotes["BID"] + quotes["OFFER"]) / 2
quotes["ba_spread"] = quotes["OFFER"] - quotes["BID"]
quotes["rel_spread"] = quotes["ba_spread"] / quotes["mid_quote"]

import numpy as np
import numpy as np
import nolds

log_price = np.log(quotes["mid_quote"]).dropna().values
returns = np.diff(log_price)

print("=== log_price ===")
print("Hurst R/S :", nolds.hurst_rs(log_price))
print("DFA       :", nolds.dfa(log_price))

print("\n=== returns ===")
print("Hurst R/S :", nolds.hurst_rs(returns))
print("DFA       :", nolds.dfa(returns))

print("\n=== abs(returns) ===")
print("Hurst R/S :", nolds.hurst_rs(np.abs(returns)))
print("DFA       :", nolds.dfa(np.abs(returns)))
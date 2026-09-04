import numpy as np
import pandas as pd
import statsmodels.api as sm

from dataclasses import dataclass
from typing import Tuple
from statsmodels.tsa.api import VAR


@dataclass
class RegimeCausalityResult:
    regime: int
    direction: str
    n_obs: int
    lag_used: int
    mean_weight: float
    test_pvalue: float
    significant: bool
    coef_sum: float
    coef_sign: str
    r2: float
    comment: str


def prepare_spot_futures_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build clean log-return dataframe for regime-dependent causality tests.
    """
    out = df.copy()

    required = ["MID", "FUTURES_CLOSE"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out["MID"] = pd.to_numeric(out["MID"], errors="coerce")
    out["FUTURES_CLOSE"] = pd.to_numeric(out["FUTURES_CLOSE"], errors="coerce")

    out = out[(out["MID"] > 0) & (out["FUTURES_CLOSE"] > 0)].copy()

    out["dlog_spot"] = np.log(out["MID"]).diff()
    out["dlog_fut"] = np.log(out["FUTURES_CLOSE"]).diff()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["dlog_spot", "dlog_fut"]).copy()
    return out


def choose_var_lag_safe(
    df_ret: pd.DataFrame,
    maxlags: int = 5,
    min_lag: int = 1,
    fallback_lag: int = 1,
) -> int:
    """
    Safe lag selection using VAR information criteria.
    Falls back to `fallback_lag` if selection fails.
    """
    tmp = df_ret[["dlog_spot", "dlog_fut"]].dropna().copy()

    if len(tmp) < 50:
        return fallback_lag

    maxlags_eff = min(maxlags, max(1, len(tmp) // 10))
    if maxlags_eff < 1:
        return fallback_lag

    try:
        sel = VAR(tmp).select_order(maxlags=maxlags_eff)

        lag = sel.bic
        if lag is None:
            lag = sel.aic

        if lag is None or not np.isfinite(lag):
            return fallback_lag

        return int(max(min_lag, lag))

    except Exception:
        return fallback_lag


def attach_regime_probabilities(
    df: pd.DataFrame,
    probs: np.ndarray,
    prefix: str = "proba_state_",
) -> pd.DataFrame:
    """
    Attach regime probabilities to a dataframe with strict length checking.
    """
    out = df.copy()

    if probs.ndim != 2:
        raise ValueError("`probs` must be a 2D array of shape (n_obs, n_states).")

    if len(out) != probs.shape[0]:
        raise ValueError(
            f"Length mismatch: dataframe has {len(out)} rows, probabilities have {probs.shape[0]} rows."
        )

    probs = np.asarray(probs, dtype=float)
    probs = np.where(np.isfinite(probs), probs, np.nan)

    row_sum = np.nansum(probs, axis=1, keepdims=True)
    valid = row_sum.squeeze() > 0

    probs_clean = np.full_like(probs, np.nan)
    probs_clean[valid] = probs[valid] / row_sum[valid]

    n_states = probs.shape[1]
    for s in range(n_states):
        out[f"{prefix}{s}"] = probs_clean[:, s]

    return out


def _weighted_directional_granger_by_regime(
    df: pd.DataFrame,
    regime_value: int,
    y_col: str,
    own_col: str,
    cause_col: str,
    prob_col: str,
    lag: int = 1,
    alpha: float = 0.05,
    min_obs: int = 50,
    min_weight_sum: float = 10.0,
) -> RegimeCausalityResult:
    """
    Weighted dynamic regression:
        y_t on own lags + cause lags
    using regime probabilities as weights.
    """
    cols = list(dict.fromkeys([y_col, own_col, cause_col, prob_col]))
    tmp = df[cols].copy()
    tmp = tmp.sort_index().copy()

    for i in range(1, lag + 1):
        tmp[f"own_lag_{i}"] = tmp[own_col].shift(i)
        tmp[f"cause_lag_{i}"] = tmp[cause_col].shift(i)

    tmp = tmp.dropna().copy()
    direction = f"{cause_col} -> {y_col}"

    if len(tmp) < min_obs:
        return RegimeCausalityResult(
            regime=regime_value,
            direction=direction,
            n_obs=len(tmp),
            lag_used=lag,
            mean_weight=np.nan,
            test_pvalue=np.nan,
            significant=False,
            coef_sum=np.nan,
            coef_sign="na",
            r2=np.nan,
            comment="Too few observations after lagging",
        )

    weights = pd.to_numeric(tmp[prob_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    if weights.sum() < min_weight_sum:
        return RegimeCausalityResult(
            regime=regime_value,
            direction=direction,
            n_obs=len(tmp),
            lag_used=lag,
            mean_weight=float(weights.mean()),
            test_pvalue=np.nan,
            significant=False,
            coef_sum=np.nan,
            coef_sign="na",
            r2=np.nan,
            comment="Weights too small for this regime",
        )

    y = tmp[y_col]
    x_cols = [f"own_lag_{i}" for i in range(1, lag + 1)] + [f"cause_lag_{i}" for i in range(1, lag + 1)]
    X = sm.add_constant(tmp[x_cols], has_constant="add")

    try:
        fit = sm.WLS(y, X, weights=weights).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": lag},
        )

        restriction = " = 0, ".join([f"cause_lag_{i}" for i in range(1, lag + 1)]) + " = 0"
        wald = fit.wald_test(restriction)
        pval = float(np.asarray(wald.pvalue).squeeze())

        cause_coeffs = np.array(
            [fit.params.get(f"cause_lag_{i}", np.nan) for i in range(1, lag + 1)],
            dtype=float
        )
        coef_sum = float(np.nansum(cause_coeffs))

        if coef_sum > 0:
            coef_sign = "positive"
        elif coef_sum < 0:
            coef_sign = "negative"
        else:
            coef_sign = "neutral"

        significant = bool(pval < alpha)

        comment = (
            f"Evidence of regime-dependent predictive effect ({coef_sign})"
            if significant
            else "No clear evidence of regime-dependent predictive effect"
        )

        return RegimeCausalityResult(
            regime=regime_value,
            direction=direction,
            n_obs=len(tmp),
            lag_used=lag,
            mean_weight=float(weights.mean()),
            test_pvalue=pval,
            significant=significant,
            coef_sum=coef_sum,
            coef_sign=coef_sign,
            r2=float(getattr(fit, "rsquared", np.nan)),
            comment=comment,
        )

    except Exception as e:
        return RegimeCausalityResult(
            regime=regime_value,
            direction=direction,
            n_obs=len(tmp),
            lag_used=lag,
            mean_weight=float(weights.mean()),
            test_pvalue=np.nan,
            significant=False,
            coef_sum=np.nan,
            coef_sign="na",
            r2=np.nan,
            comment=f"Estimation error: {e}",
        )


def weighted_spot_to_futures_by_regime(
    df: pd.DataFrame,
    regime_value: int,
    lag: int = 1,
    alpha: float = 0.05,
) -> RegimeCausalityResult:
    return _weighted_directional_granger_by_regime(
        df=df,
        regime_value=regime_value,
        y_col="dlog_fut",
        own_col="dlog_fut",
        cause_col="dlog_spot",
        prob_col=f"proba_state_{regime_value}",
        lag=lag,
        alpha=alpha,
    )


def weighted_futures_to_spot_by_regime(
    df: pd.DataFrame,
    regime_value: int,
    lag: int = 1,
    alpha: float = 0.05,
) -> RegimeCausalityResult:
    return _weighted_directional_granger_by_regime(
        df=df,
        regime_value=regime_value,
        y_col="dlog_spot",
        own_col="dlog_spot",
        cause_col="dlog_fut",
        prob_col=f"proba_state_{regime_value}",
        lag=lag,
        alpha=alpha,
    )


def run_regime_causality_all_regimes(
    df: pd.DataFrame,
    n_states: int,
    lag: int = 1,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spot_to_fut_rows = []
    fut_to_spot_rows = []

    for s in range(n_states):
        r1 = weighted_spot_to_futures_by_regime(df, regime_value=s, lag=lag, alpha=alpha)
        r2 = weighted_futures_to_spot_by_regime(df, regime_value=s, lag=lag, alpha=alpha)

        spot_to_fut_rows.append(vars(r1))
        fut_to_spot_rows.append(vars(r2))

    return pd.DataFrame(spot_to_fut_rows), pd.DataFrame(fut_to_spot_rows)
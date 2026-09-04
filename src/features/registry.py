from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# ── Registry ──────────────────────────────────────────────────────────

# Each registered function has signature: (s: pd.Series, **kwargs) -> pd.Series
_REGISTRY: dict[str, Callable[..., pd.Series]] = {}


def register(name: str):
    """Decorator -- registers a feature function under *name*."""
    def _wrap(fn: Callable[..., pd.Series]):
        _REGISTRY[name] = fn
        return fn
    return _wrap


def available_features() -> list[str]:
    return list(_REGISTRY.keys())


# ── Built-in features ────────────────────────────────────────────────

@register("returns")
def _returns(s: pd.Series, **kw) -> pd.Series:
    r = s.pct_change().replace([np.inf, -np.inf], 0)
    return r

@register("diff")
def _diff(s: pd.Series, d:int = 1, **kw) -> pd.Series:
    return s.diff(d)

@register("id")
def _id(s: pd.Series, **kw) -> pd.Series:
    return s

@register("log_returns")
def _log_returns(s: pd.Series, **kw) -> pd.Series:
    return np.log(s / s.shift(1)).replace([np.inf, -np.inf], 0)


@register("delta")
def _delta(s: pd.Series, period: int = 1, **kw) -> pd.Series:
    return s.diff(period)


@register("rvol")
def _rvol(s: pd.Series, window: int = 20, **kw) -> pd.Series:
    """Rolling realised volatility (std of daily changes)."""
    return s.diff().rolling(window).std(ddof=1)


@register("momentum")
def _momentum(s: pd.Series, period: int = 10, **kw) -> pd.Series:
    return s.diff(period)


@register("rsi")
def _rsi(s: pd.Series, period: int = 14, **kw) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


@register("zscore")
def _zscore(s: pd.Series, window: int = 20, **kw) -> pd.Series:
    """Rolling z-score: (x - mean) / std over *window*."""
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=1)
    return (s - mu) / sd.replace(0, np.nan)


@register("skew")
def _skew(s: pd.Series, window: int = 20, **kw) -> pd.Series:
    """Rolling skewness of returns."""
    r = s.pct_change()
    return r.rolling(window).skew()


@register("kurt")
def _kurt(s: pd.Series, window: int = 20, **kw) -> pd.Series:
    """Rolling excess kurtosis of returns (useful for Student-t)."""
    r = s.pct_change()
    return r.rolling(window).kurt()

@register("ma")
def _ma(s: pd.Series, window: int = 20, **kw) -> pd.Series:
    """Rolling moving average."""
    return s.rolling(window).mean()

@register("diff_ma")
def _diff_ma(s: pd.Series, period: int,  window: int = 20, **kw) -> pd.Series:
    """Difference of rolling moving average."""
    return s.diff(period).rolling(window).mean()

@register("ma_ratio")
def _ma_ratio(s: pd.Series, fast: int = 10, slow: int = 50, **kw) -> pd.Series:
    """Ratio of fast / slow moving average − 1 (mean-reversion / trend signal)."""
    return s.rolling(fast).mean() / s.rolling(slow).mean() - 1


@register("atr")
def _atr(s: pd.Series, window: int = 14, **kw) -> pd.Series:
    """Average True Range proxy (uses abs daily change when no H/L available)."""
    return s.diff().abs().rolling(window).mean()


@register("rvol_ratio")
def _rvol_ratio(s: pd.Series, fast: int = 5, slow: int = 20, **kw) -> pd.Series:
    """Short-term vol / long-term vol — captures vol regime shifts."""
    r = s.diff()
    vol_fast = r.rolling(fast).std(ddof=1)
    vol_slow = r.rolling(slow).std(ddof=1)
    return (vol_fast / vol_slow.replace(0, np.nan))


@register("drawdown")
def _drawdown(s: pd.Series, window: int = 63, **kw) -> pd.Series:
    """Rolling drawdown: (price - rolling_max) / rolling_max.

    Always <= 0.  Deep negative = price well below recent high = bearish.
    """
    rolling_max = s.rolling(window, min_periods=1).max()
    return (s - rolling_max) / rolling_max.replace(0, np.nan)


@register("log_range")
def _log_range(s: pd.Series, window: int = 5, **kw) -> pd.Series:
    """Log of rolling max/min range — a vol proxy (Parkinson-like)."""
    hi = s.rolling(window).max()
    lo = s.rolling(window).min()
    return np.log((hi / lo).replace(0, np.nan))

@register("binary_returns")
def _binary_returns(s: pd.Series, **kw) -> pd.Series:
    prices_diff = s.diff()
    obs_pos = (prices_diff > 0).astype(np.int64)
    obs_neg = (prices_diff < 0).astype(np.int64)
    return obs_pos - obs_neg


@register("trend")
def _trend(s: pd.Series, short: int = 10, long: int = 50, **kw) -> pd.Series:

    ema_short = s.ewm(span=short, min_periods=short).mean()
    ema_long = s.ewm(span=long, min_periods=long).mean()
    return (ema_short - ema_long) / ema_long.replace(0, np.nan)


# ── Builder ───────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    specs: list[tuple[str, str] | tuple[str, str, dict]],
    shift_cols: dict[str, int] | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build a feature DataFrame from specs.

    Each spec is either:
      - ``("feature_name", "column")``                — no extra params
      - ``("feature_name", "column", {"window": 30})``  — with params

    Parameters
    ----------
    shift_cols : dict mapping column name -> number of periods to shift
        **before** computing features. Use ``{"Close": 1}`` to avoid
        look-ahead on futures close prices (feature at date *t* will
        only use close prices up to *t-1*).
    """
    shift_cols = shift_cols or {}

    # Pre-shift requested columns
    df_work = df.copy()
    for col_name, periods in shift_cols.items():
        if col_name in df_work.columns:
            df_work[col_name] = df_work[col_name].shift(periods)

    cols: dict[str, pd.Series] = {}
    for spec in specs:
        if len(spec) == 3:
            feat, col, params = spec
        else:
            feat, col = spec
            params = {}

        if feat not in _REGISTRY:
            raise KeyError(f"Unknown feature '{feat}'. Available: {available_features()}")

        # Build column name including non-default params for clarity
        suffix = "_".join(f"{v}" for v in params.values()) if params else ""
        name = f"{feat}_{col}" if not suffix else f"{feat}_{suffix}_{col}"
        cols[name] = _REGISTRY[feat](df_work[col].astype(float), **params)

    out = pd.DataFrame(cols, index=df.index)
    return out.dropna() if dropna else out


# ── Standardization ──────────────────────────────────────────────────

def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score fitted on train, applied to both splits."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=1)
    sd = np.where(sd < eps, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd

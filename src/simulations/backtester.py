"""
Micro-backtester for HMM regime strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Result container ──────────────────────────────────────────────────

@dataclass
class BacktestResult:
    equity: np.ndarray
    equity_bh: np.ndarray
    positions: np.ndarray          # sized positions (shifted by 1 day)
    exposures: np.ndarray          # raw exposure weights before sizing (shifted)
    trades: list[tuple[int, float, str]]
    metrics_strategy: dict
    metrics_bh: dict


# ── Core backtest logic ──────────────────────────────────────────────

def backtest(
    prices: np.ndarray,
    states: np.ndarray,
    exposures: np.ndarray | None = None,
    position_map: dict[int, int] | None = None,
    directions: np.ndarray | None = None,
    vol_target: float = 0.15,
    vol_lookback: int = 20,
    max_leverage: float = 2.0,
) -> BacktestResult:
    """Run a backtest — supports two modes:

    **Exposure mode** (regime-sizing on buy & hold):
        Pass *exposures* array of floats in [0, 1].
    **Direction mode** (legacy — LONG / SHORT / FLAT):
        Pass *directions* or *position_map*.

    Both use vol-targeting and 1-day lag.
    """
    T = len(prices)
    if exposures is not None:
        T = min(T, len(states), len(exposures))
    else:
        T = min(T, len(states))

    p = prices[:T].astype(float)
    st = states[:T].astype(int)

    # Simple returns
    r = np.zeros(T)
    r[1:] = p[1:] / p[:-1] - 1.0

    # Resolve signal: exposure mode vs direction mode
    if exposures is not None:
        expo = np.asarray(exposures[:T], dtype=float)
        exposure_mode = True
    elif directions is not None:
        expo = np.asarray(directions[:T], dtype=float)
        exposure_mode = False
    elif position_map is not None:
        expo = np.array([float(position_map.get(s, 0)) for s in st])
        exposure_mode = False
    else:
        raise ValueError("Provide exposures, directions, or position_map.")

    # Vol-targeting sizing (base leverage)
    realized_vol = np.full(T, np.nan)
    for t in range(vol_lookback, T):
        realized_vol[t] = np.std(r[t - vol_lookback + 1: t + 1], ddof=1) * np.sqrt(252)

    sizing = np.ones(T)
    valid = np.isfinite(realized_vol) & (realized_vol > 1e-6)
    sizing[valid] = np.clip(vol_target / realized_vol[valid], 0.0, max_leverage)

    positions_sized = expo * sizing

    # Shift by 1 day (no look-ahead)
    pos_shifted = np.zeros(T)
    pos_shifted[1:] = positions_sized[:-1]

    expo_shifted = np.zeros(T)
    expo_shifted[1:] = expo[:-1]

    # P&L
    pnl = pos_shifted * r
    equity = np.cumprod(1.0 + pnl)
    equity_bh = np.cumprod(1.0 + r)

    # Trade list
    trades: list[tuple[int, float, str]] = []
    for t in range(1, T):
        prev = expo_shifted[t - 1] if t > 1 else 0.0
        curr = expo_shifted[t]
        if abs(curr - prev) > 1e-8:
            if exposure_mode:
                label = "INCREASE" if curr > prev else "DECREASE"
            else:
                if curr > 0:
                    label = "BUY"
                elif curr < 0:
                    label = "SELL"
                else:
                    label = "CLOSE"
            trades.append((t, p[t], label))

    m_strat = _compute_metrics(equity, pos_shifted)
    m_bh = _compute_metrics(equity_bh)

    return BacktestResult(
        equity=equity,
        equity_bh=equity_bh,
        positions=pos_shifted,
        exposures=expo_shifted,
        trades=trades,
        metrics_strategy=m_strat,
        metrics_bh=m_bh,
    )


# ── Metrics ───────────────────────────────────────────────────────────

def _compute_metrics(
    equity: np.ndarray,
    positions: np.ndarray | None = None,
) -> dict:
    rets = np.diff(equity) / equity[:-1]
    total = equity[-1] / equity[0] - 1
    n = len(equity)
    ann_ret = (1 + total) ** (252 / max(n, 1)) - 1
    ann_vol = np.nanstd(rets, ddof=1) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    m = {
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }
    if positions is not None:
        m["pct_invested"] = np.mean(positions != 0) * 100
        active = positions[positions != 0]
        m["avg_leverage"] = np.mean(np.abs(active)) if len(active) > 0 else 0.0

    return m


def print_metrics(result: BacktestResult) -> None:
    """Print backtest metrics to stdout."""
    def _fmt(label: str, m: dict) -> None:
        print(f"\n  [{label}]")
        print(f"    Return total:    {m['total_return']:+.2%}")
        print(f"    Return ann.:     {m['ann_return']:+.2%}")
        print(f"    Vol ann.:        {m['ann_vol']:.2%}")
        print(f"    Sharpe:          {m['sharpe']:+.2f}")
        print(f"    Max Drawdown:    {m['max_drawdown']:.2%}")
        if "pct_invested" in m:
            print(f"    % invested:      {m['pct_invested']:.0f}%")
            print(f"    Avg leverage:    {m['avg_leverage']:.2f}x")

    # Detect mode from trade labels
    labels = {t[2] for t in result.trades}
    if labels & {"INCREASE", "DECREASE"}:
        n_inc = sum(1 for t in result.trades if t[2] == "INCREASE")
        n_dec = sum(1 for t in result.trades if t[2] == "DECREASE")
        print(f"\n  Regime transitions: {len(result.trades)}  "
              f"(INCREASE={n_inc}  DECREASE={n_dec})")
        _fmt("HMM Regime Sizing", result.metrics_strategy)
    else:
        n_buy = sum(1 for t in result.trades if t[2] == "BUY")
        n_sell = sum(1 for t in result.trades if t[2] == "SELL")
        n_close = sum(1 for t in result.trades if t[2] == "CLOSE")
        print(f"\n  Trades: {len(result.trades)}  (BUY={n_buy}  SELL={n_sell}  CLOSE={n_close})")
        _fmt("HMM Strategy", result.metrics_strategy)

    _fmt("Buy & Hold", result.metrics_bh)


# ── Backtest plot ─────────────────────────────────────────────────────

def plot_backtest(
    result: BacktestResult,
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    states: np.ndarray,
    vol_target: float = 0.15,
    max_leverage: float = 2.0,
) -> None:
    T = len(result.equity)
    dates = dates[:T]
    prices = prices[:T]
    states = states[:T]

    # Detect mode from trade labels
    labels = {t[2] for t in result.trades}
    exposure_mode = bool(labels & {"INCREASE", "DECREASE"})

    title_1 = (f"Backtest: HMM Regime Sizing (vol-target {vol_target:.0%}) vs Buy & Hold"
               if exposure_mode else
               f"Backtest: HMM (vol-target {vol_target:.0%}) vs Buy & Hold")
    title_4 = "Regime exposure weight" if exposure_mode else "Filtered state (forward-only)"

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.30, 0.25, 0.15, 0.15, 0.15],
        vertical_spacing=0.03,
        subplot_titles=[
            title_1,
            "Price + regime transitions" if exposure_mode else "Price + trade signals",
            f"Position (max {max_leverage:.0f}x)",
            title_4,
            "Drawdown",
        ],
    )

    # 1) Equity curves
    strat_name = "HMM Regime Sizing" if exposure_mode else "HMM Strategy"
    fig.add_trace(go.Scatter(x=dates, y=result.equity, mode="lines",
                             name=strat_name, line=dict(width=1.5),
                             hovertemplate="Date: %{x|%Y-%m-%d}<br>Equity: %{y:.4f}<extra>Strategy</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=result.equity_bh, mode="lines",
                             name="Buy & Hold", line=dict(width=1.5), opacity=0.7,
                             hovertemplate="Date: %{x|%Y-%m-%d}<br>Equity: %{y:.4f}<extra>B&H</extra>"), row=1, col=1)

    # 2) Price + trade/transition markers
    fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines",
                             name="Price", line=dict(width=0.8, color="gray"), opacity=0.8,
                             hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>"),
                  row=2, col=1)

    if exposure_mode:
        increases = [e for e in result.trades if e[2] == "INCREASE"]
        decreases = [e for e in result.trades if e[2] == "DECREASE"]
        if increases:
            fig.add_trace(go.Scatter(
                x=[dates[e[0]] for e in increases], y=[e[1] for e in increases],
                mode="markers", name="INCREASE expo",
                marker=dict(symbol="triangle-up", color="green", size=10),
                hovertemplate="INCREASE<br>Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            ), row=2, col=1)
        if decreases:
            fig.add_trace(go.Scatter(
                x=[dates[e[0]] for e in decreases], y=[e[1] for e in decreases],
                mode="markers", name="DECREASE expo",
                marker=dict(symbol="triangle-down", color="red", size=10),
                hovertemplate="DECREASE<br>Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            ), row=2, col=1)
    else:
        buys = [e for e in result.trades if e[2] == "BUY"]
        sells = [e for e in result.trades if e[2] == "SELL"]
        closes = [e for e in result.trades if e[2] == "CLOSE"]
        if buys:
            fig.add_trace(go.Scatter(
                x=[dates[e[0]] for e in buys], y=[e[1] for e in buys],
                mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", color="green", size=10),
                hovertemplate="BUY<br>Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            ), row=2, col=1)
        if sells:
            fig.add_trace(go.Scatter(
                x=[dates[e[0]] for e in sells], y=[e[1] for e in sells],
                mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", color="red", size=10),
                hovertemplate="SELL<br>Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            ), row=2, col=1)
        if closes:
            fig.add_trace(go.Scatter(
                x=[dates[e[0]] for e in closes], y=[e[1] for e in closes],
                mode="markers", name="CLOSE",
                marker=dict(symbol="x", color="orange", size=8),
                hovertemplate="CLOSE<br>Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            ), row=2, col=1)

    # 3) Position
    fig.add_trace(go.Scatter(x=dates, y=result.positions, mode="lines",
                             name="Position", line=dict(width=1, color="purple", shape="hv"),
                             hovertemplate="Date: %{x|%Y-%m-%d}<br>Position: %{y:.2f}x<extra></extra>"),
                  row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.5, row=3, col=1)

    # 4) Exposure weight or state
    if exposure_mode:
        fig.add_trace(go.Scatter(x=dates, y=result.exposures, mode="lines",
                                 name="Exposure", line=dict(width=1.5, color="teal", shape="hv"),
                                 hovertemplate="Date: %{x|%Y-%m-%d}<br>Exposure: %{y:.2f}<extra></extra>"),
                      row=4, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="green", line_width=0.5,
                      annotation_text="full", row=4, col=1)
        fig.update_yaxes(title_text="Exposure", row=4, col=1)
    else:
        fig.add_trace(go.Scatter(x=dates, y=states, mode="lines",
                                 name="State", line=dict(width=1, color="teal", shape="hv"),
                                 hovertemplate="Date: %{x|%Y-%m-%d}<br>State: %{y}<extra></extra>"),
                      row=4, col=1)
        fig.update_yaxes(title_text="State", row=4, col=1)

    # 5) Drawdown
    peak_s = np.maximum.accumulate(result.equity)
    dd_s = (result.equity - peak_s) / peak_s
    peak_bh = np.maximum.accumulate(result.equity_bh)
    dd_bh = (result.equity_bh - peak_bh) / peak_bh
    fig.add_trace(go.Scatter(x=dates, y=dd_s, fill="tozeroy", mode="lines",
                             name="Strategy DD", opacity=0.5, line=dict(width=0.5),
                             hovertemplate="Date: %{x|%Y-%m-%d}<br>DD: %{y:.2%}<extra>Strategy</extra>"),
                  row=5, col=1)
    fig.add_trace(go.Scatter(x=dates, y=dd_bh, fill="tozeroy", mode="lines",
                             name="Buy & Hold DD", opacity=0.4, line=dict(width=0.5),
                             hovertemplate="Date: %{x|%Y-%m-%d}<br>DD: %{y:.2%}<extra>B&H</extra>"),
                  row=5, col=1)

    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Price",  row=2, col=1)
    fig.update_yaxes(title_text="Pos",    row=3, col=1)
    fig.update_yaxes(title_text="DD",     row=5, col=1)

    fig.update_layout(
        template="plotly_white",
        height=1100,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.03),
        margin=dict(t=40, b=40),
    )
    fig.show()

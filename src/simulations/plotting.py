"""
Plotting utilities for the HMM pipeline (Plotly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Colour palette for states (vivid) ────────────────────────────────

_STATE_COLORS = [
    "#FF0040",   # vivid red-pink
    "#00C9FF",   # electric blue
    "#00E676",   # neon green
    "#AA00FF",   # vivid purple
    "#FF9100",   # vivid orange
    "#FFEA00",   # vivid yellow
    "#FF4081",   # hot pink
    "#00BFA5",   # teal
]


def _sc(k: int) -> str:
    return _STATE_COLORS[k % len(_STATE_COLORS)]


# ── Log-likelihood convergence ────────────────────────────────────────

def plot_log_likelihood(ll_hist: np.ndarray, title: str = "EM log-likelihood") -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=ll_hist, mode="lines", name="LL",
        line=dict(width=2),
        hovertemplate="Iter %{x}<br>LL = %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, xaxis_title="Iteration", yaxis_title="Log-Likelihood",
        template="plotly_white", height=300, hovermode="x unified",
    )
    fig.show()


# ── Feature subplots ──────────────────────────────────────────────────

def plot_features(X: np.ndarray, names: list[str], title: str = "Features") -> None:
    n = len(names)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=names,
                        vertical_spacing=0.04)
    for i, name in enumerate(names):
        fig.add_trace(
            go.Scatter(
                y=X[:, i], mode="lines", name=name, line=dict(width=1),
                hovertemplate=f"{name}: " + "%{y:.4f}<extra></extra>",
            ),
            row=i + 1, col=1,
        )
    fig.update_layout(
        title=title, template="plotly_white",
        height=250 * n, showlegend=False, hovermode="x unified",
    )
    fig.show()


# ── Forward filtering probabilities ──────────────────────────────────

def plot_filtering_probs(
    dates: pd.DatetimeIndex,
    alpha: np.ndarray,
    title: str = "Filtering probabilities (forward-only)",
) -> None:
    K = alpha.shape[1]
    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Scatter(
            x=dates, y=alpha[:, k], mode="lines",
            name=f"P(state {k} | x₁..xₜ)",
            line=dict(width=1.5, color=_sc(k)),
            hovertemplate=f"State {k}: " + "%{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title=title, yaxis=dict(range=[-0.05, 1.05]),
        template="plotly_white", height=350,
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
    )
    fig.show()


# ── Regimes on price chart ───────────────────────────────────────────

def plot_regimes_on_price(
    dates: pd.DatetimeIndex,
    prices: np.ndarray,
    states: np.ndarray,
    title: str = "Price coloured by regime",
) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode="lines",
        line=dict(width=1, color="lightgray"), name="Price",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
    ))
    for s in np.unique(states):
        mask = states == s
        fig.add_trace(go.Scatter(
            x=dates[mask], y=prices[mask], mode="markers",
            marker=dict(size=8, color=_sc(s), opacity=0.85,
                        line=dict(width=0.5, color="white")),
            name=f"state {s}",
            hovertemplate=f"State {s}<br>" + "Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        title=title, template="plotly_white", height=450,
        legend=dict(orientation="h", y=-0.12),
        hovermode="closest",
    )
    fig.show()


# ── Combined dashboard: LL + Filtering + Regimes on one page ─────────

def plot_hmm_dashboard(
    ll_hist: np.ndarray,
    dates: pd.DatetimeIndex,
    alpha: np.ndarray,
    prices: np.ndarray,
    states: np.ndarray,
    volume: np.ndarray | None = None,
    volatility: np.ndarray | None = None,
    exposures: np.ndarray | None = None,
    title: str = "HMM Dashboard",
) -> None:
    """Log-likelihood, filtering probabilities, price regimes (with volume overlay),
    volatility, and optionally price coloured by exposure level."""
    K = alpha.shape[1]
    has_exposures = exposures is not None

    n_rows = 5 if has_exposures else 4
    specs = [
        [{"secondary_y": False}],
        [{"secondary_y": False}],
        [{"secondary_y": True}],
        [{"secondary_y": False}],
    ]
    heights = [0.12, 0.22, 0.30, 0.18]
    titles = [
        "EM log-likelihood",
        "Filtering probabilities (forward-only)",
        "Price coloured by regime",
        "Futures realised volatility",
    ]
    if has_exposures:
        specs.append([{"secondary_y": False}])
        heights.append(0.18)
        titles.append("Price coloured by regime (BEARISH / NEUTRAL / BULLISH)")

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=False,
        row_heights=heights,
        vertical_spacing=0.04,
        subplot_titles=titles,
        specs=specs,
    )

    # Row 1 — LL convergence
    fig.add_trace(
        go.Scatter(
            y=ll_hist, mode="lines", name="Log-Likelihood",
            line=dict(width=2, color="#636EFA"),
            hovertemplate="Iter %{x}<br>LL = %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Row 2 — Filtering probabilities
    for k in range(K):
        fig.add_trace(
            go.Scatter(
                x=dates, y=alpha[:, k], mode="lines",
                name=f"P(state {k})",
                line=dict(width=1.5, color=_sc(k)),
                hovertemplate=f"State {k}: " + "%{y:.3f}<extra></extra>",
            ),
            row=2, col=1,
        )

    # Row 3 — Volume as bar on secondary y (drawn first so it stays behind)
    if volume is not None:
        fig.add_trace(
            go.Bar(
                x=dates, y=volume, name="Volume",
                marker_color="rgba(100, 149, 237, 0.30)",
                width=2 * 86_400_000,          # 2 days in ms → thicker bars
                hovertemplate="Volume: %{y:,.0f}<extra></extra>",
                showlegend=True,
            ),
            row=3, col=1, secondary_y=True,
        )

    # Row 3 — Price line + regime dots on primary y
    fig.add_trace(
        go.Scatter(
            x=dates, y=prices, mode="lines",
            line=dict(width=1, color="lightgray"), name="Price",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=3, col=1, secondary_y=False,
    )
    for s in np.unique(states):
        mask = states == s
        fig.add_trace(
            go.Scatter(
                x=dates[mask], y=prices[mask], mode="markers",
                marker=dict(size=8, color=_sc(s), opacity=0.85,
                            line=dict(width=0.5, color="white")),
                name=f"state {s}",
                hovertemplate=f"State {s}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
            ),
            row=3, col=1, secondary_y=False,
        )

    # Stretch volume axis so bars appear shorter (max volume × 3)
    if volume is not None:
        vol_max = float(np.nanmax(volume))
        fig.update_yaxes(
            title_text="Volume", range=[0, vol_max * 3],
            row=3, col=1, secondary_y=True,
        )

    # Row 4 — Realised volatility (dates aligned with row 3)
    if volatility is not None:
        fig.add_trace(
            go.Scatter(
                x=dates, y=volatility, mode="lines",
                name="Realised Vol",
                line=dict(width=1.5, color="#FF6D00"),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Vol: %{y:.4f}<extra></extra>",
                showlegend=True,
            ),
            row=4, col=1,
        )

    # Row 5 — Price coloured by exposure level (walk-forward only)
    if has_exposures:
        fig.add_trace(
            go.Scatter(
                x=dates, y=prices, mode="lines",
                line=dict(width=1, color="lightgray"), name="Price",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=n_rows, col=1,
        )
        # 3-level colour scheme:  BEARISH (0) / NEUTRAL (1) / BULLISH (>1)
        def _expo_color(e: float) -> tuple[str, str]:
            if e < 1e-6:
                return "#FF1744", "BEARISH"    # red
            elif e > 1.0 + 1e-6:
                return "#00C853", "BULLISH"    # green
            else:
                return "#FFD600", "NEUTRAL"    # yellow

        unique_expos = sorted(set(np.round(exposures, 4)))
        for e_val in unique_expos:
            mask = np.abs(np.round(exposures, 4) - e_val) < 1e-6
            if not mask.any():
                continue
            color, label = _expo_color(e_val)
            fig.add_trace(
                go.Scatter(
                    x=dates[mask], y=prices[mask], mode="markers",
                    marker=dict(size=7, color=color, opacity=0.85,
                                line=dict(width=0.5, color="white")),
                    name=f"{label} (expo={e_val:.1f})",
                    hovertemplate=f"{label} expo={e_val:.2f}<br>Date: %{{x|%Y-%m-%d}}<br>Price: %{{y:.2f}}<extra></extra>",
                ),
                row=n_rows, col=1,
            )
        fig.update_yaxes(title_text="Price", row=n_rows, col=1)

    # Align x-axes of rows 3, 4, and optionally 5
    date_min, date_max = dates.min(), dates.max()
    fig.update_xaxes(range=[date_min, date_max], row=3, col=1)
    fig.update_xaxes(range=[date_min, date_max], row=4, col=1)
    if has_exposures:
        fig.update_xaxes(range=[date_min, date_max], row=n_rows, col=1)

    fig.update_yaxes(title_text="LL", row=1, col=1)
    fig.update_yaxes(title_text="P(state)", range=[-0.05, 1.05], row=2, col=1)
    fig.update_yaxes(title_text="Price", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Vol", row=4, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1350 if has_exposures else 1150,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.04),
        margin=dict(t=50, b=50),
    )
    fig.show()


# ── Regime summary (text — unchanged) ────────────────────────────────

def print_regime_summary(
    result,
    feat_mu: np.ndarray,
    feat_sd: np.ndarray,
    feature_names: list[str],
) -> None:
    """Print interpretable regime parameters (original scale)."""
    K = result.means.shape[0]
    has_nu = result.nus is not None
    has_full_cov = result.covs_ is not None

    print("=" * 60)
    cov_tag = "full covariance" if has_full_cov else "diagonal"
    print(f"REGIME SUMMARY (original scale, {cov_tag})")
    print("=" * 60)

    for k in range(K):
        nu_str = f"  nu={result.nus[k]:.1f}" if has_nu else ""
        print(f"\n--- State {k}{nu_str} ---")
        for j, name in enumerate(feature_names):
            orig_mean = result.means[k, j] * feat_sd[j] + feat_mu[j]
            orig_std = np.sqrt(result.vars_[k, j]) * feat_sd[j]
            print(f"  {name:>25s}: mean={orig_mean:+.4f}  std={orig_std:.4f}")

        # Print correlation matrix when full covariance
        if has_full_cov:
            d = result.covs_.shape[1]
            cov_k = result.covs_[k]
            std_k = np.sqrt(np.diag(cov_k))
            corr_k = cov_k / (np.outer(std_k, std_k) + 1e-300)
            np.fill_diagonal(corr_k, 1.0)
            print(f"  Correlation matrix:")
            header = "        " + "".join(f"{n[:8]:>9s}" for n in feature_names)
            print(header)
            for i, name_i in enumerate(feature_names):
                row = "".join(f"{corr_k[i, j]:+8.3f} " for j in range(d))
                print(f"  {name_i[:8]:>8s}{row}")

    print(f"\nTransition matrix A:\n{np.round(result.A, 3)}")
    diag = np.diag(result.A)
    print(f"Persistence (diag A):  {np.round(diag, 3)}")
    print(f"Avg stay (days):       {np.round(1 / (1 - diag + 1e-12), 1)}")

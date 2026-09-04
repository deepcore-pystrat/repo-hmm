"""
Walk-forward engine for HMM regime-sizing strategies.

Slices the data into rolling windows (train_bars / test_bars).
On each window the full pipeline runs:

    standardise → select K (BIC) → fit HMM → filter train → map exposures → filter test

The out-of-sample fragments are then stitched together for a single
consolidated backtest over the whole period covered by the test windows.

Strategy: always LONG (buy & hold), HMM regimes control exposure level.
"""
from __future__ import annotations

from copy import copy

import numpy as np
import pandas as pd

from features.registry import standardize
from models import create_hmm, map_regime_exposures
from models.basis.basis_data import HMMConfig
from models.student_hmm.student_data import StudentTHMMConfig
from simulations.backtester import backtest, print_metrics, plot_backtest
from simulations.plotting import plot_hmm_dashboard


# ── BIC-based K selection ─────────────────────────────────────────────

def _count_params(K: int, d: int, cov_type: str, has_nu: bool) -> int:
    """Count free parameters of an HMM.

    pi: K-1, A: K(K-1), means: Kd,
    cov: Kd (diag) or K*d*(d+1)/2 (full), [nus: K]
    """
    n = (K - 1) + K * (K - 1) + K * d
    if cov_type == "full":
        n += K * d * (d + 1) // 2
    else:
        n += K * d
    if has_nu:
        n += K
    return n


def select_K(
    X_train_z: np.ndarray,
    hmm_cfg: HMMConfig,
    K_range: list[int],
) -> tuple[int, dict]:
    """Select the best K by BIC on the training set.

    Returns (best_K, info_dict) where info_dict maps K → {bic, ll, model, result}.
    """
    d = X_train_z.shape[1]
    T = X_train_z.shape[0]
    has_nu = isinstance(hmm_cfg, StudentTHMMConfig)
    cov_type = getattr(hmm_cfg, "cov_type", "diag")

    info: dict[int, dict] = {}

    for K in K_range:
        cfg_k = copy(hmm_cfg)
        cfg_k.K = K
        model = create_hmm(cfg_k)
        result = model.fit(X_train_z)
        ll = result.ll_hist[-1]
        n_params = _count_params(K, d, cov_type, has_nu)
        bic = -2 * ll + n_params * np.log(T)
        info[K] = {"bic": bic, "ll": ll, "model": model, "result": result}

    best_K = min(info, key=lambda k: info[k]["bic"])
    return best_K, info


# ── Walk-forward engine ──────────────────────────────────────────────

def walk_forward(
    data_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    hmm_cfg: HMMConfig,
    train_bars: int = 1008,
    test_bars: int = 252,
    step_bars: int | None = None,
    bear_threshold: float = -0.5,
    bull_threshold: float = 0.5,
    max_exposure: float = 1.5,
    vol_target: float = 0.15,
    vol_lookback: int = 20,
    max_leverage: float = 2.0,
    K_range: list[int] | None = None,
):
    """Run the walk-forward loop (regime-sizing on buy & hold).

    Parameters
    ----------
    data_df : raw DataFrame (needs 'Close', 'Volume' columns).
    feat_df : feature DataFrame (same index subset of data_df).
    hmm_cfg : HMM configuration (Gaussian or Student-t).
    train_bars : number of bars in each training window.
    test_bars  : number of bars in each test window.
    step_bars  : how many bars to slide forward between windows
                 (default = test_bars, i.e. non-overlapping test windows).
    bear_threshold : Sharpe below which a state is BEARISH (expo=0).
    bull_threshold : Sharpe above which a state is BULLISH (expo=max_exposure).
    max_exposure   : exposure multiplier for bullish regimes (>1).
    K_range : list of K values to try at each fold (BIC selection).
              If None, use hmm_cfg.K as-is (no selection).
    """
    step_bars = step_bars or test_bars

    X_all = feat_df.values
    idx_all = feat_df.index
    T = len(X_all)

    # Collectors for stitched OOS results
    oos_dates: list[pd.DatetimeIndex] = []
    oos_prices: list[np.ndarray] = []
    oos_states: list[np.ndarray] = []
    oos_alphas: list[np.ndarray] = []
    oos_exposure_maps: list[dict] = []

    fold = 0
    start = 0
    while start + train_bars < T:
        tr_end = start + train_bars
        te_end = min(start + train_bars + test_bars, T)

        # Skip if the test window is too small to be meaningful
        n_test = te_end - tr_end
        if n_test < 1:
            break

        tr_slice = slice(start, tr_end)
        te_slice = slice(tr_end, te_end)

        X_train = X_all[tr_slice]
        X_test  = X_all[te_slice]
        idx_train = idx_all[tr_slice]
        idx_test  = idx_all[te_slice]

        # ── Standardise on train ──────────────────────────────────
        X_train_z, X_test_z, feat_mu, feat_sd = standardize(X_train, X_test)

        # ── Select K (BIC) or use fixed K ─────────────────────────
        if K_range is not None:
            best_K, k_info = select_K(X_train_z, hmm_cfg, K_range)
            model = k_info[best_K]["model"]
            result = k_info[best_K]["result"]
            bic_summary = "  ".join(
                f"K={k}: BIC={v['bic']:.0f}" for k, v in sorted(k_info.items())
            )
        else:
            best_K = hmm_cfg.K
            model = create_hmm(hmm_cfg)
            result = model.fit(X_train_z)
            bic_summary = None

        # ── Map regime exposures on train ─────────────────────────
        states_train, _, _ = model.filter(X_train_z)
        p_train = data_df.loc[idx_train, "Close"].astype(float).values[:len(states_train)]
        r_train = np.zeros(len(p_train))
        r_train[1:] = p_train[1:] / p_train[:-1] - 1.0
        exposure_map, label_map = map_regime_exposures(
            states_train, r_train,
            bear_threshold=bear_threshold,
            bull_threshold=bull_threshold,
            max_exposure=max_exposure,
        )

        # ── Filter on test ────────────────────────────────────────
        states_test, alpha_test, ll_test = model.filter(X_test_z)

        p_test = data_df.loc[idx_test, "Close"].astype(float).values[:len(states_test)]
        r_test = np.zeros(len(p_test))
        r_test[1:] = p_test[1:] / p_test[:-1] - 1.0

        # ── Print fold summary ────────────────────────────────────
        partial_tag = " (PARTIAL)" if n_test < test_bars else ""
        print(f"\n{'='*60}")
        print(f"FOLD {fold}{partial_tag}  |  K={best_K}  |  "
              f"train {idx_train[0].date()}→{idx_train[-1].date()}  "
              f"test {idx_test[0].date()}→{idx_test[-1].date()}  ({n_test} bars)")
        if bic_summary:
            print(f"  K selection (BIC):  {bic_summary}")
        print(f"  LL final: {result.ll_hist[-1]:.1f}  |  test LL/step: {ll_test/len(X_test_z):.4f}")

        # Train stats
        print("  [TRAIN]")
        for s in sorted(exposure_map):
            mask_s = states_train == s
            r_s = r_train[mask_s]
            n_s = mask_s.sum()
            vol_ann = np.nanstd(r_s, ddof=1) * np.sqrt(252) if n_s > 1 else 0
            mean_ann = np.nanmean(r_s) * 252
            sharpe = mean_ann / vol_ann if vol_ann > 0 else 0.0
            lbl = label_map.get(s, "?")
            print(f"    State {s} → {lbl:>7s} (expo={exposure_map[s]:.2f})  "
                  f"sharpe={sharpe:+.2f}  vol={vol_ann:.1%}  "
                  f"mean={mean_ann:+.1%}  n={n_s}")

        # Test stats
        print("  [TEST]")
        for s in np.unique(states_test):
            mask_s = states_test == s
            r_s = r_test[mask_s]
            c = mask_s.sum()
            vol_ann = np.nanstd(r_s, ddof=1) * np.sqrt(252) if c > 1 else 0
            mean_ann = np.nanmean(r_s) * 252
            sharpe = mean_ann / vol_ann if vol_ann > 0 else 0.0
            expo = exposure_map.get(s, 1.0)
            lbl = label_map.get(s, "?")
            print(f"    State {s} → {lbl:>7s} (expo={expo:.2f})  "
                  f"sharpe={sharpe:+.2f}  vol={vol_ann:.1%}  "
                  f"mean={mean_ann:+.1%}  n={c}")

        # ── Store OOS fragment ────────────────────────────────────
        oos_dates.append(idx_test[:len(states_test)])
        oos_prices.append(p_test)
        oos_states.append(states_test)
        oos_alphas.append(alpha_test)
        oos_exposure_maps.append(exposure_map)

        fold += 1
        start += step_bars

    if fold == 0:
        print("ERROR: not enough data for even one window "
              f"(need {train_bars}+1, have {T}).")
        return

    # ── Stitch all OOS fragments ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"STITCHED OOS  |  {fold} folds")
    print(f"{'='*60}")

    all_dates   = np.concatenate(oos_dates)
    all_prices  = np.concatenate(oos_prices)
    all_states  = np.concatenate(oos_states)

    # Pad alpha arrays to max K so they can be concatenated
    max_K = max(a.shape[1] for a in oos_alphas)
    padded_alphas = []
    for a in oos_alphas:
        if a.shape[1] < max_K:
            pad = np.zeros((a.shape[0], max_K - a.shape[1]))
            a = np.hstack([a, pad])
        padded_alphas.append(a)
    all_alpha = np.concatenate(padded_alphas, axis=0)

    # Build the unified exposure vector from per-fold exposure maps
    all_exposures = np.ones(len(all_states), dtype=float)
    offset = 0
    for emap, st in zip(oos_exposure_maps, oos_states):
        n = len(st)
        all_exposures[offset:offset + n] = [emap.get(s, 1.0) for s in st]
        offset += n

    # ── Run backtest with stitched exposures ──────────────────────
    bt = backtest(
        prices=all_prices,
        states=all_states,
        exposures=all_exposures,
        vol_target=vol_target,
        vol_lookback=vol_lookback,
        max_leverage=max_leverage,
    )

    print_metrics(bt)

    # ── Plots ─────────────────────────────────────────────────────────
    T_bt = len(bt.equity)
    dates_idx = pd.DatetimeIndex(all_dates[:T_bt])

    vol_oos = data_df.loc[dates_idx, "Volume"].astype(float).values if "Volume" in data_df.columns else None
    rvol_oos = pd.Series(all_prices[:T_bt]).pct_change().rolling(20).std().values * np.sqrt(252)

    plot_hmm_dashboard(
        ll_hist=np.array([0.0]),   # no single LL curve for walk-forward
        dates=dates_idx,
        alpha=all_alpha[:T_bt],
        prices=all_prices[:T_bt],
        states=all_states[:T_bt],
        volume=vol_oos,
        volatility=rvol_oos,
        exposures=all_exposures[:T_bt],
        title=f"Walk-Forward HMM Regime Sizing — {fold} folds (OOS stitched)",
    )
    plot_backtest(bt, dates_idx, all_prices[:T_bt], all_states[:T_bt],
                  vol_target=vol_target, max_leverage=max_leverage)

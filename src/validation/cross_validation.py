"""
Walk-forward cross-validation for HMM feature / hyper-parameter selection.

Usage
-----
    results = walk_forward_cv(
        X_train, hmm_cfg,
        n_folds=5, val_ratio=0.20,
    )
    print_cv_report(results, feature_names)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from models import create_hmm
from models.basis.basis_data import HMMConfig


# ── Result container ──────────────────────────────────────────────────

@dataclass
class CVFoldResult:
    """Metrics from one train/val fold."""
    fold: int
    train_ll: float
    val_ll: float
    val_ll_per_step: float
    n_train: int
    n_val: int
    n_params: int
    bic: float
    aic: float
    persistence: np.ndarray        # diag(A) → avg stay
    n_iters: int


@dataclass
class CVResult:
    """Aggregate cross-validation results for one configuration."""
    label: str
    folds: list[CVFoldResult]
    feature_names: list[str]

    # ── Convenience aggregates ────────────────────────────────────
    @property
    def mean_val_ll(self) -> float:
        return float(np.mean([f.val_ll_per_step for f in self.folds]))

    @property
    def std_val_ll(self) -> float:
        return float(np.std([f.val_ll_per_step for f in self.folds], ddof=1))

    @property
    def mean_bic(self) -> float:
        return float(np.mean([f.bic for f in self.folds]))

    @property
    def mean_aic(self) -> float:
        return float(np.mean([f.aic for f in self.folds]))

    @property
    def mean_persistence(self) -> np.ndarray:
        return np.mean([f.persistence for f in self.folds], axis=0)


# ── Helpers ───────────────────────────────────────────────────────────

def _count_params(K: int, d: int, has_nu: bool = False) -> int:
    """Count free parameters in a diagonal-covariance HMM.

    pi: K-1, A: K(K-1), means: Kd, vars: Kd, [nus: K]
    """
    n = (K - 1) + K * (K - 1) + K * d + K * d
    if has_nu:
        n += K
    return n


def _make_folds(
    T: int, n_folds: int, val_ratio: float, min_train: int = 200,
) -> list[tuple[int, int, int]]:
    """Generate expanding-window fold indices.

    Returns list of ``(train_start, train_end, val_end)`` where
    ``train_end`` is the first index of the validation set.
    """
    val_size = max(int(T * val_ratio / n_folds), 50)
    folds = []
    for i in range(n_folds):
        val_end = T - (n_folds - 1 - i) * val_size
        train_end = val_end - val_size
        if train_end < min_train:
            continue
        folds.append((0, train_end, val_end))
    return folds


# ── Main CV function ──────────────────────────────────────────────────

def walk_forward_cv(
    X: np.ndarray,
    hmm_cfg: HMMConfig,
    n_folds: int = 5,
    val_ratio: float = 0.20,
    label: str = "",
    feature_names: list[str] | None = None,
    verbose: bool = True,
) -> CVResult:
    """Run walk-forward CV on standardised feature matrix *X*.

    Parameters
    ----------
    X : (T, d) array, already standardised on the full train set.
    hmm_cfg : HMM config (StudentTHMMConfig / GaussianHMMConfig …).
    n_folds : number of temporal folds.
    val_ratio : fraction of data used for validation across all folds.
    label : descriptive label for this configuration.
    feature_names : optional list of feature names for reporting.

    Returns
    -------
    CVResult with per-fold metrics.
    """
    from models.student_hmm.student_data import StudentTHMMConfig  # local to avoid circular

    X = np.asarray(X, dtype=float)
    T, d = X.shape
    K = hmm_cfg.K
    has_nu = isinstance(hmm_cfg, StudentTHMMConfig)
    n_params = _count_params(K, d, has_nu)

    folds_idx = _make_folds(T, n_folds, val_ratio)
    if not folds_idx:
        raise ValueError("Not enough data for the requested fold configuration.")

    fold_results: list[CVFoldResult] = []

    for i, (t_start, t_end, v_end) in enumerate(folds_idx):
        X_tr = X[t_start:t_end]
        X_va = X[t_end:v_end]

        # Re-standardise each fold independently (train stats only)
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0, ddof=1)
        sd = np.where(sd < 1e-12, 1.0, sd)
        X_tr_z = (X_tr - mu) / sd
        X_va_z = (X_va - mu) / sd

        model = create_hmm(hmm_cfg)
        try:
            result = model.fit(X_tr_z)
        except Exception as e:
            if verbose:
                print(f"  Fold {i}: FAILED — {e}")
            continue

        # Validation LL (forward-only)
        _, _, val_ll = model.filter(X_va_z)
        train_ll = result.ll_hist[-1]

        n_va = len(X_va_z)
        bic = -2 * val_ll + n_params * np.log(n_va)
        aic = -2 * val_ll + 2 * n_params

        diag_A = np.diag(result.A)
        persistence = 1.0 / (1.0 - diag_A + 1e-12)

        fr = CVFoldResult(
            fold=i,
            train_ll=train_ll,
            val_ll=val_ll,
            val_ll_per_step=val_ll / n_va,
            n_train=len(X_tr_z),
            n_val=n_va,
            n_params=n_params,
            bic=bic,
            aic=aic,
            persistence=persistence,
            n_iters=len(result.ll_hist),
        )
        fold_results.append(fr)

        if verbose:
            print(f"  Fold {i}: train_LL={train_ll:+.1f}  "
                  f"val_LL/step={fr.val_ll_per_step:+.4f}  "
                  f"BIC={bic:.0f}  "
                  f"persist={np.round(persistence, 1)}")

    return CVResult(
        label=label or f"K={K}_d={d}",
        folds=fold_results,
        feature_names=feature_names or [f"feat_{j}" for j in range(d)],
    )


# ── Multi-config comparison ──────────────────────────────────────────

def compare_configs(
    configs: list[dict],
    data_df,
    hmm_cfg: HMMConfig,
    shift_cols: dict | None = None,
    train_ratio: float = 0.80,
    n_folds: int = 5,
    val_ratio: float = 0.20,
    verbose: bool = True,
) -> list[CVResult]:
    """Compare multiple feature configurations via walk-forward CV.

    Parameters
    ----------
    configs : list of dicts, each with keys:
        - "label": str
        - "specs": list of feature spec tuples
    data_df : raw DataFrame (before features).
    hmm_cfg : HMM config to use for all comparisons.
    shift_cols : passed to build_features.
    train_ratio : train/test split ratio (CV only uses train portion).

    Returns
    -------
    List of CVResult, one per config, sorted by mean_val_ll descending.
    """
    from features.registry import build_features, standardize
    from features.splitter import fixed_split

    results: list[CVResult] = []

    for cfg in configs:
        label = cfg["label"]
        specs = cfg["specs"]

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"CONFIG: {label}")
            print(f"  Features: {[s[0] for s in specs]}")
            print(f"{'=' * 60}")

        try:
            feat_df = build_features(data_df, specs, shift_cols=shift_cols)
        except Exception as e:
            if verbose:
                print(f"  SKIPPED — feature build failed: {e}")
            continue

        X, idx = feat_df.values, feat_df.index
        names = list(feat_df.columns)
        X_train, _, _, _ = fixed_split(X, idx, train_ratio=train_ratio)
        # Standardise full train for CV
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0, ddof=1)
        sd = np.where(sd < 1e-12, 1.0, sd)
        X_train_z = (X_train - mu) / sd

        cv = walk_forward_cv(
            X_train_z, hmm_cfg,
            n_folds=n_folds,
            val_ratio=val_ratio,
            label=label,
            feature_names=names,
            verbose=verbose,
        )
        results.append(cv)

    # Sort by best mean val LL per step (higher is better)
    results.sort(key=lambda r: r.mean_val_ll, reverse=True)
    return results


# ── Pretty report ─────────────────────────────────────────────────────

def print_cv_report(results: list[CVResult]) -> None:
    """Print a comparison table across configs."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD CV — COMPARISON REPORT")
    print("=" * 80)
    print(f"{'Rank':<5} {'Config':<35} {'LL/step':>10} {'±std':>8} "
          f"{'BIC':>10} {'AIC':>10} {'Persist':>10}")
    print("-" * 80)

    for rank, cv in enumerate(results, 1):
        persist_str = "/".join(f"{p:.0f}" for p in cv.mean_persistence)
        print(f"{rank:<5} {cv.label:<35} {cv.mean_val_ll:>+10.4f} "
              f"{cv.std_val_ll:>8.4f} {cv.mean_bic:>10.0f} "
              f"{cv.mean_aic:>10.0f} {persist_str:>10}")

    print("-" * 80)
    best = results[0]
    print(f"\n  BEST: {best.label}")
    print(f"  Features: {best.feature_names}")
    print(f"  Mean val LL/step: {best.mean_val_ll:+.4f} ± {best.std_val_ll:.4f}")
    print(f"  Mean BIC: {best.mean_bic:.0f}   Mean AIC: {best.mean_aic:.0f}")
    print(f"  Mean persistence (days): {np.round(best.mean_persistence, 1)}")

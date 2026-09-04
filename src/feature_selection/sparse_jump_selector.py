"""
sparse_jump_selector.py
======================

Paper-aligned Sparse Jump feature selector.

Core implementation follows the official jump-models package logic:
  1) initialize feature weights w_j = 1/sqrt(P)
  2) fit a Jump Model on X weighted by sqrt(w)
  3) compute BCSS_j on the original standardized X
  4) update w by solving the Lasso constraint
         ||w||_2 <= 1, ||w||_1 <= sqrt(max_feats), w_j >= 0
     implemented by soft-thresholding BCSS / max(BCSS)
  5) iterate until weights converge

Use this AFTER FSHMM saliency and BEFORE the final Student-t HMM.
X must be cleaned/scaled on TRAIN only before calling fit().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ArrayLike = np.ndarray | pd.DataFrame


@dataclass
class SparseJumpSelectorConfig:
    n_states: int = 2

    # Paper parameters
    max_feats: float = 6.0          # official repo parameter; kappa = sqrt(max_feats)
    jump_penalty: float = 0.1       # lambda; internally scaled by 1/sqrt(P), like official repo

    # Optimization
    max_outer_iter: int = 30
    max_inner_iter: int = 100
    tol_weights: float = 1e-4
    tol_jump: float = 1e-8
    n_init: int = 20
    random_state: int = 42

    # Final reporting/selection
    weight_tol: float = 1e-8
    min_features: int = 1          # safeguard only; paper selection is via max_feats

    verbose: bool = False


class SparseJumpSelector:
    def __init__(self, config: SparseJumpSelectorConfig | None = None) -> None:
        self.cfg = config or SparseJumpSelectorConfig()
        self.feature_names_: list[str] | None = None
        self.weights_: np.ndarray | None = None             # paper w, L2-normalized
        self.feat_weights_: np.ndarray | None = None        # sqrt(w), used to weight X
        self.states_: np.ndarray | None = None
        self.centers_: np.ndarray | None = None             # unweighted centers
        self.selected_indices_: list[int] | None = None
        self.selected_features_: list[str] | None = None
        self.report_: pd.DataFrame | None = None
        self.objective_: float | None = None
        self.n_iter_: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: ArrayLike, feature_names: Iterable[str] | None = None) -> "SparseJumpSelector":
        X_arr, names = self._as_array_and_names(X, feature_names)
        T, P = X_arr.shape

        if T < 5 * self.cfg.n_states:
            raise ValueError("Not enough observations for the requested number of states.")
        if not (1.0 <= self.cfg.max_feats <= P):
            raise ValueError(f"max_feats must be in [1, P]. Got max_feats={self.cfg.max_feats}, P={P}.")

        rng = np.random.default_rng(self.cfg.random_state)
        # Official repo scaling: SparseJumpModel.init_jm divides lambda by sqrt(n_features)
        lam_scaled = float(self.cfg.jump_penalty) / np.sqrt(P)
        norm_ub = np.sqrt(float(self.cfg.max_feats))  # = sqrt(6) = 2.449


        # Official initialization: w = ones(P) / sqrt(P)
        w = np.ones(P, dtype=float) / np.sqrt(P)
        w_old = np.ones(P, dtype=float) * 2.0  # only to enter first iteration

        centers_unweighted = None
        states = None
        obj = np.inf

        for outer in range(1, self.cfg.max_outer_iter + 1):
            rel_change = np.linalg.norm(w - w_old, ord=1) / (np.linalg.norm(w_old, ord=1) + 1e-12)
            if outer > 1 and rel_change <= self.cfg.tol_weights:
                break

            w_old = w.copy()
            feat_weights = np.sqrt(np.maximum(w, 0.0))
            Z = X_arr * feat_weights[None, :]

            # Fit discrete Jump Model on weighted data.
            init_centers = None
            if centers_unweighted is not None:
                init_centers = centers_unweighted * feat_weights[None, :]

            states, centers_weighted, obj = self._fit_jump_model(
                Z=Z,
                lam=lam_scaled,
                rng=rng,
                init_centers=init_centers,
            )

            # Compute unweighted centers and BCSS on original X, like official repo.
            centers_unweighted = self._centers_from_states(X_arr, states)
            bcss = self._compute_bcss(X_arr, states, centers_unweighted)

            if np.all(bcss <= 0):
                if self.cfg.verbose:
                    print(f"[SparseJump] outer={outer} all BCSS <= 0; stopping.")
                break

            # Official repo: w = solve_lasso(BCSS / BCSS.max(), sqrt(max_feats))
            w = self._solve_lasso(bcss / np.max(bcss), norm_ub=norm_ub)

            if self.cfg.verbose:
                nnz = int(np.sum(w > self.cfg.weight_tol))
                print(
                    f"[SparseJump] outer={outer:02d} "
                    f"rel_w_change={rel_change:.3e} "
                    f"nnz={nnz} obj={obj:.6f}"
                )

        if states is None or centers_unweighted is None:
            raise RuntimeError("SparseJumpSelector failed to fit.")

        selected_idx = np.flatnonzero(w > self.cfg.weight_tol).tolist()
        if len(selected_idx) < self.cfg.min_features:
            selected_idx = np.argsort(-w)[: self.cfg.min_features].tolist()
        selected_idx = sorted(selected_idx, key=lambda i: -w[i])

        self.feature_names_ = names
        self.weights_ = w
        self.feat_weights_ = np.sqrt(np.maximum(w, 0.0))
        self.states_ = states.astype(int)
        self.centers_ = centers_unweighted
        self.selected_indices_ = selected_idx
        self.selected_features_ = [names[i] for i in selected_idx]
        self.objective_ = float(obj)
        self.n_iter_ = outer

        self.report_ = pd.DataFrame({
            "feature": names,
            "sparse_jump_weight": w,
            "feature_weight_sqrt_w": self.feat_weights_,
            "selected": [i in selected_idx for i in range(P)],
        }).sort_values("sparse_jump_weight", ascending=False).reset_index(drop=True)

        if self.cfg.verbose:
            print("\n=== SparseJumpSelector paper-aligned selected features ===")
            print(self.report_.to_string(index=False))
            print(f"jump_penalty={self.cfg.jump_penalty}, lambda_scaled={lam_scaled}, max_feats={self.cfg.max_feats}")

        return self

    def transform(self, X: ArrayLike) -> np.ndarray | pd.DataFrame:
        self._check_fitted()
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.selected_features_ if c not in X.columns]
            if missing:
                raise ValueError(f"Selected features missing from X: {missing}")
            return X[self.selected_features_].copy()
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, self.selected_indices_]

    def fit_transform(self, X: ArrayLike, feature_names: Iterable[str] | None = None):
        self.fit(X, feature_names=feature_names)
        return self.transform(X)

    def get_selected_features(self) -> list[str]:
        self._check_fitted()
        return list(self.selected_features_)

    def get_report(self) -> pd.DataFrame:
        self._check_fitted()
        return self.report_.copy()

    # Compatibility with previous code; no grid in paper-aligned version.
    def get_grid_results(self) -> pd.DataFrame:
        self._check_fitted()
        return pd.DataFrame([{
            "jump_penalty": self.cfg.jump_penalty,
            "max_feats": self.cfg.max_feats,
            "objective": self.objective_,
            "n_selected": len(self.selected_features_),
            "n_iter": self.n_iter_,
        }])

    # ------------------------------------------------------------------
    # Jump Model core
    # ------------------------------------------------------------------
    def _fit_jump_model(
        self,
        Z: np.ndarray,
        lam: float,
        rng: np.random.Generator,
        init_centers: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        T, P = Z.shape
        K = self.cfg.n_states
        best_states = None
        best_centers = None
        best_obj = np.inf

        center_inits: list[np.ndarray] = []
        if init_centers is not None and np.isfinite(init_centers).all():
            center_inits.append(init_centers.copy())

        for _ in range(self.cfg.n_init):
            seed = int(rng.integers(0, 2**31 - 1))
            km = KMeans(n_clusters=K, init="k-means++", n_init=1, random_state=seed)
            km.fit(Z)
            center_inits.append(km.cluster_centers_.astype(float))

        for centers in center_inits:
            states = None
            prev_obj = np.inf

            for _ in range(self.cfg.max_inner_iter):
                losses = self._squared_losses(Z, centers)  # includes 0.5 factor
                new_states, obj = self._dynamic_programming_states(losses, lam)
                new_centers = self._centers_from_states(Z, new_states)

                if states is not None and np.array_equal(new_states, states):
                    states = new_states
                    centers = new_centers
                    break
                if abs(prev_obj - obj) <= self.cfg.tol_jump * (abs(prev_obj) + 1e-12):
                    states = new_states
                    centers = new_centers
                    break

                states = new_states
                centers = new_centers
                prev_obj = obj

            losses = self._squared_losses(Z, centers)
            states, obj = self._dynamic_programming_states(losses, lam)
            centers = self._centers_from_states(Z, states)

            if obj < best_obj:
                best_obj = float(obj)
                best_states = states.copy()
                best_centers = centers.copy()

        if best_states is None:
            raise RuntimeError("All Jump Model initializations failed.")
        return best_states, best_centers, best_obj

    @staticmethod
    def _squared_losses(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # Official repo uses 0.5 * squared Euclidean distance.
        diff = X[:, None, :] - centers[None, :, :]
        return 0.5 * np.sum(diff * diff, axis=2)

    def _dynamic_programming_states(self, losses: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
        T, K = losses.shape
        dp = np.empty((T, K), dtype=float)
        back = np.zeros((T, K), dtype=int)
        dp[0] = losses[0]

        eye = np.eye(K, dtype=bool)
        penalty = lam * (~eye)

        for t in range(1, T):
            trans = dp[t - 1][:, None] + penalty
            back[t] = np.argmin(trans, axis=0)
            dp[t] = losses[t] + np.min(trans, axis=0)

        states = np.empty(T, dtype=int)
        states[-1] = int(np.argmin(dp[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = back[t + 1, states[t + 1]]

        return states, float(np.min(dp[-1]))

    def _centers_from_states(self, X: np.ndarray, states: np.ndarray) -> np.ndarray:
        K = self.cfg.n_states
        P = X.shape[1]
        centers = np.empty((K, P), dtype=float)
        global_mean = X.mean(axis=0)
        for k in range(K):
            mask = states == k
            centers[k] = X[mask].mean(axis=0) if np.any(mask) else global_mean
        return centers

    def _compute_bcss(self, X: np.ndarray, states: np.ndarray, centers: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        K = self.cfg.n_states
        global_mean = X.mean(axis=0)
        bcss = np.zeros(X.shape[1], dtype=float)
        for k in range(K):
            nk = np.sum(states == k)
            if nk > 0:
                bcss += nk * (centers[k] - global_mean) ** 2
        bcss[np.abs(bcss) < tol] = 0.0
        return np.maximum(bcss, 0.0)

    # ------------------------------------------------------------------
    # Official-style Lasso solver
    # ------------------------------------------------------------------
    @staticmethod
    def _soft_threshold_l2_normalized(x: np.ndarray, threshold: float) -> np.ndarray:
        y = np.maximum(0.0, x - threshold)
        n = np.linalg.norm(y)
        if n <= 1e-14:
            out = np.zeros_like(x)
            out[int(np.argmax(x))] = 1.0
            return out
        return y / n

    def _solve_lasso(self, a: np.ndarray, norm_ub: float, tol: float = 1e-8) -> np.ndarray:
        a = np.maximum(np.asarray(a, dtype=float), 0.0)
        if not np.isfinite(a).all() or a.sum() <= 1e-14:
            return np.ones_like(a) / np.sqrt(len(a))

        norm_ub = float(norm_ub)
        if norm_ub < 1.0:
            raise ValueError("norm_ub must be >= 1")

        w0 = self._soft_threshold_l2_normalized(a, 0.0)
        if np.sum(np.abs(w0)) <= norm_ub:
            return w0

        unique = np.unique(a)
        right = unique[-2] if len(unique) >= 2 else unique[-1]
        if right < tol:
            return w0

        left = 0.0
        for _ in range(100):
            mid = 0.5 * (left + right)
            w = self._soft_threshold_l2_normalized(a, mid)
            l1 = np.sum(np.abs(w))
            if abs(l1 - norm_ub) <= tol:
                return w
            if l1 > norm_ub:
                left = mid
            else:
                right = mid
        return self._soft_threshold_l2_normalized(a, 0.5 * (left + right))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_array_and_names(X: ArrayLike, feature_names: Iterable[str] | None) -> tuple[np.ndarray, list[str]]:
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
            names = list(X.columns)
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim != 2:
                raise ValueError("X must be a 2D array.")
            names = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(X_arr.shape[1])]

        if X_arr.shape[1] != len(names):
            raise ValueError("feature_names length does not match X columns.")
        if not np.isfinite(X_arr).all():
            raise ValueError("X contains NaN or Inf. Clean/scale X before SparseJumpSelector.")
        return X_arr, names

    def _check_fitted(self) -> None:
        if self.selected_features_ is None or self.weights_ is None:
            raise RuntimeError("SparseJumpSelector is not fitted.")

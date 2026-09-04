"""
BaseHMM -- abstract class with all shared HMM logic.

Subclasses only implement:
  - _compute_log_emissions(X) -> (T, K)
  - _m_step_emissions(X, gamma)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from models.basis.basis_data import HMMConfig, HMMResult

from sklearn.cluster import KMeans
# ── Private helpers ───────────────────────────────────────────────────

def _random_stochastic_vector(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(np.ones(n))


def _random_stochastic_matrix(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(np.ones(cols), size=rows)


# ── Base HMM ──────────────────────────────────────────────────────────

class BaseHMM(ABC):
    """Abstract base for HMMs (diagonal or full covariance)."""

    _label: str = "HMM"
    def _load_result(self, result: HMMResult) -> None:
        """Load fitted parameters from an HMMResult into the current model."""
        self.pi = result.pi.copy()
        self.A = result.A.copy()
        self.means = result.means.copy()
        self.vars_ = result.vars_.copy()
        self.covs_ = result.covs_.copy() if result.covs_ is not None else None

        # Student-t specific parameter
        if result.nus is not None and hasattr(self, "nus"):
            self.nus = result.nus.copy()
        if result.garch_params is not None and hasattr(self, "garch_params"):
            self.garch_params = {
                key: np.array(value, copy=True)
                for key, value in result.garch_params.items()
            }
    def fit_multi_start(
        self,
        X: np.ndarray,
        seeds: list[int] | None = None,
        n_starts: int = 5,
        verbose: bool = True,
    ) -> HMMResult:
        """
        Run several EM fits with different seeds and keep the best result.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        seeds : list[int] | None
            Explicit list of seeds to try. If None, uses
            [cfg.seed, cfg.seed+1, ..., cfg.seed+n_starts-1].
        n_starts : int
            Number of starts if seeds is None.
        verbose : bool
            Whether to print diagnostics.

        Returns
        -------
        HMMResult
            Best result found across all starts.
        """
        X = np.asarray(X, dtype=float)

        if seeds is None:
            base_seed = 0 if self.cfg.seed is None else int(self.cfg.seed)
            seeds = list(range(base_seed, base_seed + n_starts))

        best_result: HMMResult | None = None
        best_ll = -np.inf
        best_seed: int | None = None

        original_seed = self.cfg.seed

        if verbose:
            print("\n" + "=" * 60)
            print("MULTI-START EM")
            print("=" * 60)

        for seed in seeds:
            self.cfg.seed = seed

            if verbose:
                print(f"\n[EM start] seed={seed}")

            result = self.fit(X)
            final_ll = float(result.ll_hist.max())


            if verbose:
                print(f"Final LL = {final_ll:.6f}")
                print(f"pi = {np.round(result.pi, 3)}")
                print(f"diag(A) = {np.round(np.diag(result.A), 3)}")
                if result.nus is not None:
                    print(f"nu = {np.round(result.nus, 3)}")

            if final_ll > best_ll:
                best_ll = final_ll
                best_result = result
                best_seed = seed

        self.cfg.seed = original_seed

        if best_result is None:
            raise RuntimeError("fit_multi_start failed: no valid result obtained.")

        # Load best parameters back into current model
        self._load_result(best_result)

        if verbose:
            print("\n" + "-" * 60)
            print(f"Best seed: {best_seed}")
            print(f"Best final LL: {best_ll:.6f}")
            print("-" * 60)

        return best_result
    def filter_with_pi(
        self,
        X: np.ndarray,
        pi_override: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Forward-only causal filtering with a custom initial distribution.
        Returns (states, alpha, loglik).
        """
        X = np.asarray(X, dtype=float)
        pi_override = np.asarray(pi_override, dtype=float)
        pi_override = pi_override / pi_override.sum()

        log_B = self._compute_log_emissions(X)
        alpha, _c, ll = self._forward_scaled(pi_override, self.A, log_B)
        states = np.argmax(alpha, axis=1)
        return states, alpha, ll
    def __init__(self, config: HMMConfig) -> None:
        self.cfg = config
        self.pi: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.vars_: np.ndarray | None = None
        self.covs_: np.ndarray | None = None  # (K, d, d) when cov_type="full"
    def _apply_transition_mask(self, A: np.ndarray) -> np.ndarray:
        mask = getattr(self.cfg, "transition_mask", None)

        A = np.asarray(A, dtype=float)

        if mask is None:
            row_sums = A.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-300)
            return A / row_sums

        mask = np.asarray(mask, dtype=bool)

        if mask.shape != A.shape:
            raise ValueError(
                f"transition_mask shape {mask.shape} != A shape {A.shape}"
            )

        A = np.where(mask, A, 0.0)

        for i in range(A.shape[0]):
            if A[i].sum() <= 0:
                allowed = mask[i]
                if not np.any(allowed):
                    raise ValueError(f"transition_mask row {i} has no allowed transition")
                A[i, allowed] = 1.0 / allowed.sum()

        A = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-300)
        return A
    # ── Initialisation ────────────────────────────────────────────

    def _init_params(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.cfg.seed)
        T, d = X.shape
        K = self.cfg.K

        kmeans = KMeans(
    n_clusters=K,
    random_state=self.cfg.seed,
    n_init=20,
)
        labels = kmeans.fit_predict(X)

        cluster_counts = np.bincount(labels, minlength=K).astype(float)
        self.pi = np.full(K, 1.0 / K)

        A = np.zeros((K, K))
        for t in range(T - 1):
            i = labels[t]
            j = labels[t + 1]
            A[i, j] += 1

        row_sums = A.sum(axis=1, keepdims=True)
        zero_rows = (row_sums.squeeze() == 0)
        if np.any(zero_rows):
            A[zero_rows] = 1.0 / K
            row_sums = A.sum(axis=1, keepdims=True)

        A = A + 1e-2 + self.cfg.sticky * np.eye(K)
    
        self.A = self._apply_transition_mask(A)

        self.means = np.asarray(kmeans.cluster_centers_, dtype=float)
        if self.means.ndim == 1:
            self.means = self.means.reshape((K, d))

        if self.cfg.cov_type == "full":
            sample_cov = np.cov(X, rowvar=False)
            sample_cov = sample_cov + self.cfg.min_var * np.eye(d)
            self.covs_ = np.tile(sample_cov, (K, 1, 1))
            self.vars_ = np.diagonal(self.covs_, axis1=1, axis2=2).copy()
        else:
            self.vars_ = np.tile(
            np.maximum(X.var(axis=0), self.cfg.min_var),
            (K, 1)
        )
            self.covs_ = None

    def _init_emission_params(self, X: np.ndarray) -> None:
        """Hook for subclass-specific init (e.g. nu for Student-t)."""

    # ── Abstract (emission-specific) ──────────────────────────────

    @abstractmethod
    def _compute_log_emissions(self, X: np.ndarray) -> np.ndarray:
        """Return log B[t, k] for every t, k.  Shape (T, K)."""

    @abstractmethod
    def _m_step_emissions(self, X: np.ndarray, gamma: np.ndarray) -> None:
        """Update emission parameters in-place given responsibilities."""

    # ── Result builder ────────────────────────────────────────────

    def _build_result(self, ll_hist: np.ndarray) -> HMMResult:
        nus = None
        if hasattr(self, "nus"):
            if self.nus is not None:
                nus = np.array(self.nus, copy=True)

        covs = None
        if self.covs_ is not None:
            covs = np.array(self.covs_, copy=True)

        return HMMResult(
            pi=np.array(self.pi, copy=True),
            A=np.array(self.A, copy=True),
            means=np.array(self.means, copy=True),
            vars_=np.array(self.vars_, copy=True),
            ll_hist=np.array(ll_hist, copy=True),
            nus=nus,
            covs_=covs,
        )

    # ── Forward / backward (common) ───────────────────────────────

    @staticmethod
    def _forward_scaled(
        pi: np.ndarray, A: np.ndarray, log_B: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Scaled forward pass.  Returns (alpha, c, loglik)."""
        T, K = log_B.shape
        alpha = np.zeros((T, K))
        c = np.zeros(T)

        row_max = log_B.max(axis=1)
        B = np.exp(log_B - row_max[:, None])

        alpha[0] = pi * B[0]
        c[0] = alpha[0].sum()
        if c[0] == 0:
            raise ValueError("c[0]=0 — bad init or underflow.")
        alpha[0] /= c[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            c[t] = alpha[t].sum()
            if c[t] == 0:
                raise ValueError(f"c[{t}]=0 — numerical issue.")
            alpha[t] /= c[t]

        loglik = float(np.sum(np.log(c)) + np.sum(row_max))
        return alpha, c, loglik

    @staticmethod
    def _backward_scaled(
        A: np.ndarray, log_B: np.ndarray, c: np.ndarray,
    ) -> np.ndarray:
        """Scaled backward pass.  Returns beta (T, K)."""
        T, K = log_B.shape
        beta = np.zeros((T, K))
        beta[T - 1] = 1.0

        row_max = log_B.max(axis=1)
        B = np.exp(log_B - row_max[:, None])

        for t in range(T - 2, -1, -1):
            beta[t] = A @ (B[t + 1] * beta[t + 1])
            beta[t] /= c[t + 1]

        return beta

    # ── EM loop ───────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> HMMResult:
        X = np.asarray(X, dtype=float)
        T = X.shape[0]
        K = self.cfg.K

        self._init_params(X)
        self._init_emission_params(X)

        ll_hist: list[float] = []

        # ── [FIX] snapshot du meilleur état ──────────────────────────
        best_ll_seen = -np.inf
        best_snapshot: HMMResult | None = None
        # ─────────────────────────────────────────────────────────────

        for it in tqdm(range(self.cfg.n_iter), desc=self._label):
            log_B = self._compute_log_emissions(X)
            alpha, c, ll = self._forward_scaled(self.pi, self.A, log_B)
            beta = self._backward_scaled(self.A, log_B, c)
            ll_hist.append(ll)

            # ── [FIX] snapshot si meilleur LL ────────────────────────
            if ll > best_ll_seen:
                best_ll_seen = ll
                best_snapshot = self._build_result(np.array(ll_hist))
            # ─────────────────────────────────────────────────────────

            if len(ll_hist) > 1 and ll_hist[-1] < ll_hist[-2] - 1e-6:
                print(
                    f"Warning: LL decreased at iteration {it}: "
                    f"{ll_hist[-2]:.6f} -> {ll_hist[-1]:.6f}"
                )

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            row_max = log_B.max(axis=1)
            B = np.exp(log_B - row_max[:, None])

            xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                numer = (alpha[t][:, None] * self.A) * (B[t + 1] * beta[t + 1])[None, :]
                denom = numer.sum()
                if denom == 0:
                    raise ValueError(f"xi denominator zero at t={t}")
                xi[t] = numer / denom

            pi_prior = 1e-2
            self.pi = gamma[0] + pi_prior
            self.pi = self.pi / self.pi.sum()

            A_counts = xi.sum(axis=0)
            transition_prior = 1e-2
            sticky_prior = self.cfg.sticky * np.eye(self.cfg.K)
            A_new = A_counts + transition_prior + sticky_prior
            A_new = np.clip(A_new, 1e-12, None)
            A_new = self._apply_transition_mask(A_new)
            self.A = A_new
            self._m_step_emissions(X, gamma)

            if len(ll_hist) > 1 and abs(ll_hist[-1] - ll_hist[-2]) < self.cfg.tol:
                print(f"Converged at iteration {it} (DLL={ll_hist[-1] - ll_hist[-2]:.2e})")
                break

        # ── [FIX] restaurer le meilleur état avant de retourner ──────
        if best_snapshot is not None:
            self._load_result(best_snapshot)
            # reconstruire avec l'historique complet mais LL du meilleur
            return self._build_result(np.array(ll_hist))
        # ─────────────────────────────────────────────────────────────

        return self._build_result(np.array(ll_hist))
    def fit_batch(self, Xs: list[np.ndarray]) -> HMMResult:
        Xs = [np.asarray(X, dtype=float) for X in Xs]

        X_all = np.vstack(Xs)
        self._init_params(X_all)
        self._init_emission_params(X_all)

        ll_hist = []

        for it in tqdm(range(self.cfg.n_iter), desc=self._label + " batch"):
            gammas = []
            xi_sum = np.zeros((self.cfg.K, self.cfg.K))
            pi_sum = np.zeros(self.cfg.K)
            total_ll = 0.0

            for X in Xs:
                log_B = self._compute_log_emissions(X)
                alpha, c, ll = self._forward_scaled(self.pi, self.A, log_B)
                beta = self._backward_scaled(self.A, log_B, c)

                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

                row_max = log_B.max(axis=1)
                B = np.exp(log_B - row_max[:, None])

                T = X.shape[0]
                xi = np.zeros((T - 1, self.cfg.K, self.cfg.K))

                for t in range(T - 1):
                    numer = (
                        alpha[t][:, None]
                        * self.A
                        * (B[t + 1] * beta[t + 1])[None, :]
                    )
                    denom = numer.sum()
                    xi[t] = numer / max(denom, 1e-300)

                gammas.append(gamma)
                xi_sum += xi.sum(axis=0)
                pi_sum += gamma[0]
                total_ll += ll

            ll_hist.append(total_ll)

            self.pi = pi_sum + 1e-2
            self.pi /= self.pi.sum()

            transition_prior = 1e-2
            sticky_prior = self.cfg.sticky * np.eye(self.cfg.K)

            A_new = xi_sum + transition_prior + sticky_prior
            A_new = self._apply_transition_mask(A_new)
            self.A = A_new

            gamma_all = np.vstack(gammas)
            self._m_step_emissions(X_all, gamma_all)

            if len(ll_hist) > 1 and abs(ll_hist[-1] - ll_hist[-2]) < self.cfg.tol:
                print(f"Converged at iteration {it}")
                break

        return self._build_result(np.array(ll_hist))
    # ── Inference (post-fit) ──────────────────────────────────────
    def smooth(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Forward-backward smoothing.

        Returns
        -------
        states_smooth : np.ndarray
            Argmax of smoothed marginals gamma[t]
        gamma : np.ndarray
            Smoothed state probabilities p(X_t | Y_{1:T})
        loglik : float
            Log-likelihood of the sequence
        """
        X = np.asarray(X, dtype=float)
        log_B = self._compute_log_emissions(X)

        alpha, c, ll = self._forward_scaled(self.pi, self.A, log_B)
        beta = self._backward_scaled(self.A, log_B, c)

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        states_smooth = np.argmax(gamma, axis=1)
        return states_smooth, gamma, ll
    def filter(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Forward-only causal filtering.

        Returns (states, alpha, loglik).
        """
        X = np.asarray(X, dtype=float)
        log_B = self._compute_log_emissions(X)
        alpha, _c, ll = self._forward_scaled(self.pi, self.A, log_B)
        states = np.argmax(alpha, axis=1)
        return states, alpha, ll

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding (uses full sequence — not causal)."""
        X = np.asarray(X, dtype=float)
        log_B = self._compute_log_emissions(X)
        T, K = log_B.shape

        eps = 1e-300
        log_pi = np.log(self.pi + eps)
        log_A = np.log(self.A + eps)

        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = log_pi + log_B[0]

        for t in range(1, T):
            for j in range(K):
                scores = delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(scores)
                delta[t, j] = scores[psi[t, j]] + log_B[t, j]

        states = np.zeros(T, dtype=int)
        states[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states


# ── Regime mapping ────────────────────────────────────────────────────

def map_regimes(
    states: np.ndarray,
    returns: np.ndarray,
    flat_threshold: float = 0.20,
) -> dict[int, int]:
    """Derive a state -> position mapping from train-set statistics.

    For each state, compute the annualised Sharpe of *returns* in that
    state.  If |Sharpe| < *flat_threshold* the position is 0 (FLAT),
    otherwise +1 (LONG) or -1 (SHORT).
    """
    position_map: dict[int, int] = {}
    for s in np.unique(states):
        mask = states == s
        r_s = returns[mask]
        n = mask.sum()
        mean_ann = np.nanmean(r_s) * 252
        std_ann = np.nanstd(r_s, ddof=1) * np.sqrt(252) if n > 1 else np.nan
        sharpe = mean_ann / std_ann if (std_ann and std_ann > 0) else 0.0

        if abs(sharpe) < flat_threshold:
            position_map[s] = 0
        elif mean_ann > 0:
            position_map[s] = 1
        else:
            position_map[s] = -1

    return position_map


def map_regime_exposures(
    states: np.ndarray,
    returns: np.ndarray,
    bear_threshold: float = -0.5,
    bull_threshold: float = 0.5,
    max_exposure: float = 1.5,
) -> tuple[dict[int, float], dict[int, str]]:
    """Map each state to an exposure weight based on train-set Sharpe.

    Classification per state (from train returns):

    - **BEARISH** : Sharpe < *bear_threshold* AND mean < 0  → expo = 0
    - **BULLISH** : Sharpe > *bull_threshold* AND mean > 0  → expo = *max_exposure*
    - **NEUTRAL** : everything else                         → expo = 1.0

    Returns (exposure_map, label_map) where label_map[s] ∈ {BEARISH, NEUTRAL, BULLISH}.
    """
    exposure_map: dict[int, float] = {}
    label_map: dict[int, str] = {}

    for s in np.unique(states):
        mask = states == s
        r_s = returns[mask]
        n = mask.sum()
        mean_ann = np.nanmean(r_s) * 252
        vol_ann = np.nanstd(r_s, ddof=1) * np.sqrt(252) if n > 1 else 0.0
        sharpe = mean_ann / vol_ann if vol_ann > 0 else 0.0

        if sharpe < bear_threshold and mean_ann < 0:
            exposure_map[s] = 0.0
            label_map[s] = "BEARISH"
        elif sharpe > bull_threshold and mean_ann > 0:
            exposure_map[s] = max_exposure
            label_map[s] = "BULLISH"
        else:
            exposure_map[s] = 1.0
            label_map[s] = "NEUTRAL"

    return exposure_map, label_map

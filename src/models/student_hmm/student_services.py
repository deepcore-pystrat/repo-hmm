"""
StudentTHMM -- diagonal or full-covariance multivariate Student-t emissions.
"""

from __future__ import annotations

from matplotlib.pyplot import step
import numpy as np
from scipy.special import gammaln, digamma

from models.basis.basis_data import HMMResult
from models.basis.basis_services import BaseHMM
from models.student_hmm.student_data import StudentTHMMConfig


# ── Private helpers (Student-t specific) ──────────────────────────────
def _symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def _make_spd(
    cov: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Retourne une covariance symétrique positive définie.

    La fonction est idempotente :
    une matrice déjà suffisamment positive définie
    n'est pas modifiée à chaque appel.
    """
    cov = np.asarray(cov, dtype=float)
    cov = _symmetrize(cov)

    eigvals = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eigvals))

    # Ajouter une correction uniquement lorsque nécessaire.
    if min_eig < eps:
        correction = eps - min_eig
        cov = cov + correction * np.eye(
            cov.shape[0],
            dtype=float,
        )

    return _symmetrize(cov)
def _trigamma(x: float) -> float:
    """Trigamma approximation via central difference of digamma."""
    h = 1e-5
    return (digamma(x + h) - digamma(x - h)) / (2.0 * h)


def _estimate_nu_fixedpoint(
    gamma_k: np.ndarray,
    u_k: np.ndarray,
    log_u_k: np.ndarray,
    nu_old: float,
    n_fp_iter: int = 20,
) -> float:
    """Newton fixed-point update for nu (Peel & McLachlan 2000)."""
    w_sum = gamma_k.sum()
    if w_sum < 1e-12:
        return nu_old

    mean_log_u = np.sum(gamma_k * log_u_k) / w_sum
    mean_u = np.sum(gamma_k * u_k) / w_sum

    nu = nu_old
    for _ in range(n_fp_iter):
        half = nu / 2.0
        f = np.log(half) - digamma(half) + 1.0 + mean_log_u - mean_u
        f_prime = 1.0 / nu - 0.5 * _trigamma(half)
        if abs(f_prime) < 1e-14:
            break
        step = f / f_prime
        nu_new = np.clip(nu - 0.5 * step, 2.5, 200.0)
        if abs(nu_new - nu) < 1e-6:
            nu = nu_new
            break
        nu = nu_new

    return float(nu)


# ── Student-t HMM ────────────────────────────────────────────────────

class StudentTHMM(BaseHMM):
    """HMM with multivariate Student-t emissions (diagonal or full cov)."""

    _label = "Baum-Welch Student-t"

    def __init__(self, config: StudentTHMMConfig) -> None:
        super().__init__(config)
        self.nus: np.ndarray | None = None

    def _init_emission_params(self, X: np.ndarray) -> None:
        if getattr(self.cfg, "estimate_nu", False):
            rng = np.random.default_rng(self.cfg.seed)
            self.nus = rng.uniform(5.0, 15.0, size=self.cfg.K)
        else:
            fixed_nu = getattr(self.cfg, "fixed_nu", 6.0)
            self.nus = np.full(self.cfg.K, fixed_nu)

    # ── log-emission ──────────────────────────────────────────────

    def _compute_log_emissions(self, X: np.ndarray) -> np.ndarray:
        if self.cfg.cov_type == "full":
            return self._log_emissions_full(X)
        return self._log_emissions_diag(X)

    def _log_emissions_diag(self, X: np.ndarray) -> np.ndarray:
        T, d = X.shape
        K = self.cfg.K
        log_B = np.zeros((T, K))

        for k in range(K):
            nu = max(float(self.nus[k]), 2.5)
            var_k = np.maximum(self.vars_[k], self.cfg.min_var)

            delta = np.sum(((X - self.means[k]) ** 2) / var_k, axis=1)
            delta = np.clip(delta, 1e-12, 1e12)

            log_B[:, k] = (
                gammaln((nu + d) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * d * np.log(nu * np.pi)
                - 0.5 * np.sum(np.log(var_k))
                - ((nu + d) / 2.0) * np.log1p(delta / nu)
            )

        if not np.all(np.isfinite(log_B)):
            raise ValueError("Student-t diag log emissions contain nan/inf")

        return log_B

    def _log_emissions_full(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Log-densités Student-t multivariées avec covariance complète.

        Cette fonction est pure :
        elle ne modifie aucun paramètre du modèle.
        """
        X = np.asarray(X, dtype=float)

        T, d = X.shape
        K = self.cfg.K

        if self.covs_ is None:
            raise RuntimeError(
                "Full covariance emissions require self.covs_."
            )

        if self.means is None or self.nus is None:
            raise RuntimeError(
                "Student-t emission parameters are not initialized."
            )

        log_B = np.empty(
            (T, K),
            dtype=float,
        )

        eps = max(
            float(self.cfg.min_var),
            1e-8,
        )

        for k in range(K):
            nu = max(
                float(self.nus[k]),
                2.5,
            )

            # Copie locale : aucune modification de self.covs_.
            cov_k = _make_spd(
                np.array(
                    self.covs_[k],
                    dtype=float,
                    copy=True,
                ),
                eps=eps,
            )

            try:
                L = np.linalg.cholesky(cov_k)

            except np.linalg.LinAlgError:
                # Sécurité numérique locale uniquement.
                jitter = eps
                identity = np.eye(d)

                for _ in range(8):
                    try:
                        cov_local = (
                            cov_k
                            + jitter * identity
                        )

                        L = np.linalg.cholesky(
                            cov_local
                        )

                        cov_k = cov_local
                        break

                    except np.linalg.LinAlgError:
                        jitter *= 10.0

                else:
                    raise RuntimeError(
                        f"Unable to compute Cholesky "
                        f"decomposition for state {k}."
                    )

            log_det = 2.0 * np.sum(
                np.log(np.diag(L))
            )

            diff = X - self.means[k]

            solved = np.linalg.solve(
                L,
                diff.T,
            )

            mahalanobis_sq = np.sum(
                solved ** 2,
                axis=0,
            )

            mahalanobis_sq = np.clip(
                mahalanobis_sq,
                0.0,
                1e12,
            )

            log_B[:, k] = (
                gammaln((nu + d) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * d * np.log(nu * np.pi)
                - 0.5 * log_det
                - ((nu + d) / 2.0)
                * np.log1p(
                    mahalanobis_sq / nu
                )
            )

        if not np.all(np.isfinite(log_B)):
            bad_rows = np.where(
                ~np.all(
                    np.isfinite(log_B),
                    axis=1,
                )
            )[0]

            raise ValueError(
                "Student-t full log emissions contain "
                f"NaN or infinity. Bad rows: "
                f"{bad_rows[:10].tolist()}"
            )

        return log_B

    # ── M-step ────────────────────────────────────────────────────

    def _m_step_emissions(self, X: np.ndarray, gamma: np.ndarray) -> None:
        if self.cfg.cov_type == "full":
            self._m_step_full(X, gamma)
        else:
            self._m_step_diag(X, gamma)

    def _m_step_diag(self, X: np.ndarray, gamma: np.ndarray) -> None:
        old_means = self.means.copy()
        old_vars = self.vars_.copy()
        old_nus = self.nus.copy()
        T, d = X.shape
        for k in range(self.cfg.K):
            gk = gamma[:, k]
            old_var_k = np.maximum(old_vars[k], self.cfg.min_var)
            delta_k = ((X - old_means[k]) ** 2) / old_var_k
            delta_k = np.sum(delta_k, axis=1)
            delta_k = np.clip(delta_k, 1e-8, 1e8)

            nu = old_nus[k]

            # Latent weights
            uk = (nu + d) / (nu + delta_k)
            log_uk = (
                digamma((nu + d) / 2.0)
                - np.log((nu + delta_k) / 2.0)
            )

            w = gk * uk
            w_sum = w.sum()
            gk_sum = gk.sum()

            if w_sum <= 1e-12 or gk_sum <= 1e-12:
                continue

            # Location (weighted by u)
            mk = (w[:, None] * X).sum(axis=0) / w_sum
            self.means[k] = mk

            # Scale
            vk = (w[:, None] * (X - mk) ** 2).sum(axis=0) / gk_sum
            self.vars_[k] = np.maximum(vk, self.cfg.min_var)
            

            # Degrees of freedom
            if self.cfg.estimate_nu:
                nu_new = _estimate_nu_fixedpoint(gk, uk, log_uk, nu)
                self.nus[k] = max(nu_new, 2.5)

    def _m_step_full(self, X: np.ndarray, gamma: np.ndarray) -> None:
        old_means = self.means.copy()
        old_covs = self.covs_.copy()
        old_nus = self.nus.copy()
        T, d = X.shape

        global_cov = np.cov(X, rowvar=False)
        global_cov = _make_spd(global_cov, eps=max(self.cfg.min_var, 1e-6))
        global_diag = np.diag(np.diag(global_cov))

        for k in range(self.cfg.K):
            gk = gamma[:, k]

            cov_prev = _make_spd(old_covs[k], eps=max(self.cfg.min_var, 1e-6))
            L = np.linalg.cholesky(cov_prev)

            diff = X - old_means[k]
            solve = np.linalg.solve(L, diff.T)

            delta_k = np.sum(solve ** 2, axis=0)
            delta_k = np.clip(delta_k, 1e-8, 1e8)

            nu = old_nus[k]

            uk = (nu + d) / (nu + delta_k)
            log_uk = (
                digamma((nu + d) / 2.0)
                - np.log((nu + delta_k) / 2.0)
            )

            w = gk * uk
            w_sum = w.sum()
            gk_sum = gk.sum()

            if w_sum <= 1e-12 or gk_sum <= 1e-12:
                self.covs_[k] = global_diag.copy()
                self.vars_[k] = np.diag(self.covs_[k])
                continue

            mk = (w[:, None] * X).sum(axis=0) / w_sum
            self.means[k] = mk

            diff2 = X - mk
            cov_k = (diff2 * w[:, None]).T @ diff2 / max(gk_sum, 1e-12)

            cov_k = _symmetrize(cov_k)
            # shrinkage plus fort vers covariance diagonale globale
            alpha = min(0.3, 1.0 / np.sqrt(gk_sum + 1e-8))
            cov_k = (1.0 - alpha) * cov_k + alpha * global_diag

            # variance minimale relative
            min_diag = np.maximum(self.cfg.min_var, 0.01 * np.diag(global_cov))
            diag = np.diag(cov_k).copy()
            diag = np.maximum(diag, min_diag)
            cov_k[np.diag_indices_from(cov_k)] = diag

            cov_k = _make_spd(cov_k, eps=max(self.cfg.min_var, 1e-6))

            

            self.covs_[k] = cov_k
            self.vars_[k] = np.diag(cov_k)

            if self.cfg.estimate_nu:
                nu_new = _estimate_nu_fixedpoint(gk, uk, log_uk, nu)
                self.nus[k] = max(nu_new, 2.5)
    # ── Result builder ────────────────────────────────────────────

    def _build_result(self, ll_hist: np.ndarray) -> HMMResult:
        return HMMResult(
            pi=np.array(self.pi, copy=True),
            A=np.array(self.A, copy=True),
            means=np.array(self.means, copy=True),
            vars_=np.array(self.vars_, copy=True),
            ll_hist=np.array(ll_hist, copy=True),
            nus=np.array(self.nus, copy=True) if self.nus is not None else None,
            covs_=np.array(self.covs_, copy=True) if self.covs_ is not None else None,
        )
    def volatility_order(self) -> np.ndarray:
        if self.cfg.cov_type == "full":
            vols = np.array([
                np.sqrt(np.trace(self.covs_[k]))
                for k in range(self.cfg.K)
            ])
        else:
            vols = np.sqrt(self.vars_.mean(axis=1))

        return np.argsort(vols)


    def high_low_states(self) -> tuple[int, int]:
        order = self.volatility_order()
        return int(order[0]), int(order[-1])
    

    def predict_vol_regime(
        self,
        X: np.ndarray,
        method: str = "smooth",
        threshold: float = 0.5,
    ) -> dict:
        """
        Predict low/high volatility regimes.

        Parameters
        ----------
        X : np.ndarray
            Input observations, shape (T, d).
        method : {"smooth", "filter", "viterbi"}
            smooth  = uses full sequence, best for analysis/backtest labeling.
            filter  = causal, best for live usage.
            viterbi = most likely state path, non-causal.
        threshold : float
            Probability threshold for HIGH_VOL when method is smooth/filter.

        Returns
        -------
        dict with states, probabilities, low/high states and labels.
        """
        X = np.asarray(X, dtype=float)

        low_state, high_state = self.high_low_states()

        if method == "smooth":
            states, proba, ll = self.smooth(X)
            high_vol_proba = proba[:, high_state]
            low_vol_proba = proba[:, low_state]
            regime = np.where(high_vol_proba >= threshold, "HIGH_VOL", "LOW_VOL")

        elif method == "filter":
            states, proba, ll = self.filter(X)
            high_vol_proba = proba[:, high_state]
            low_vol_proba = proba[:, low_state]
            regime = np.where(high_vol_proba >= threshold, "HIGH_VOL", "LOW_VOL")

        elif method == "viterbi":
            states = self.viterbi(X)
            ll = np.nan
            high_vol_proba = (states == high_state).astype(float)
            low_vol_proba = (states == low_state).astype(float)
            regime = np.where(states == high_state, "HIGH_VOL", "LOW_VOL")

        else:
            raise ValueError("method must be one of: 'smooth', 'filter', 'viterbi'")

        return {
            "states": states,
            "low_state": low_state,
            "high_state": high_state,
            "low_vol_proba": low_vol_proba,
            "high_vol_proba": high_vol_proba,
            "regime": regime,
            "loglik": ll,
        }
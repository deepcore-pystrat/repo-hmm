"""
GaussianHMM -- diagonal or full-covariance Gaussian emissions.
"""

from __future__ import annotations

import numpy as np

from models.basis.basis_services import BaseHMM
from models.gaussian_hmm.gaussian_data import GaussianHMMConfig


class GaussianHMM(BaseHMM):
    """HMM with Gaussian emissions (diagonal or full covariance)."""

    _label = "Baum-Welch Gaussian"

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
            diff = X - self.means[k]
            log_B[:, k] = (
                -0.5 * d * np.log(2.0 * np.pi)
                - 0.5 * np.sum(np.log(self.vars_[k]))
                - 0.5 * np.sum(diff ** 2 / self.vars_[k], axis=1)
            )
        return log_B

    def _log_emissions_full(self, X: np.ndarray) -> np.ndarray:
        T, d = X.shape
        K = self.cfg.K
        log_B = np.zeros((T, K))
        const = -0.5 * d * np.log(2.0 * np.pi)
        for k in range(K):
            L = np.linalg.cholesky(self.covs_[k])        # (d, d)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            diff = X - self.means[k]                       # (T, d)
            solve = np.linalg.solve(L, diff.T)             # (d, T)
            maha = np.sum(solve ** 2, axis=0)              # (T,)
            log_B[:, k] = const - 0.5 * log_det - 0.5 * maha
        return log_B

    # ── M-step ────────────────────────────────────────────────────

    def _m_step_emissions(self, X: np.ndarray, gamma: np.ndarray) -> None:
        if self.cfg.cov_type == "full":
            self._m_step_full(X, gamma)
        else:
            self._m_step_diag(X, gamma)

    def _m_step_diag(self, X: np.ndarray, gamma: np.ndarray) -> None:
        for k in range(self.cfg.K):
            w = gamma[:, k]
            w_sum = w.sum()
            if w_sum <= 1e-12:
                continue
            mk = (w[:, None] * X).sum(axis=0) / w_sum
            self.means[k] = mk
            vk = (w[:, None] * (X - mk) ** 2).sum(axis=0) / w_sum
            self.vars_[k] = np.maximum(vk, self.cfg.min_var)

    def _m_step_full(self, X: np.ndarray, gamma: np.ndarray) -> None:
        d = X.shape[1]
        for k in range(self.cfg.K):
            w = gamma[:, k]
            w_sum = w.sum()
            if w_sum <= 1e-12:
                continue
            mk = (w[:, None] * X).sum(axis=0) / w_sum
            self.means[k] = mk
            diff = X - mk                                   # (T, d)
            cov_k = (diff * w[:, None]).T @ diff / w_sum     # (d, d)
            # Regularise: floor diagonal at min_var
            cov_k[np.diag_indices(d)] = np.maximum(
                cov_k[np.diag_indices(d)], self.cfg.min_var,
            )
            self.covs_[k] = cov_k
            self.vars_[k] = np.diag(cov_k)

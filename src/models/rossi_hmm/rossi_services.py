from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

from models.rossi_hmm.rossi_data import RossiHMMConfig, RossiHMMResult


class RossiHMM:
    """
    Exact Rossi & Gallo (2006) HMM structure:

    r_t = mu_t + sqrt(sigma2(z_t)) u_t

    mu_t = mu + sum_k c_k r_{t-k}

    sigma2_i = exp(a + d g_i)

    phi_t = Phi(alpha + beta |r_t|)

    M_t = M+ if r_t > 0
          M- if r_t <= 0

    Transition matrix is tridiagonal as in equation (7).
    """

    def __init__(self, config: RossiHMMConfig):
        if config.K < 2:
            raise ValueError("K must be >= 2")
        if config.p < 0:
            raise ValueError("p must be >= 0")

        self.cfg = config
        self.params_ = None
        self.sigma2_ = None
        self.filtered_proba_ = None
        self.states_ = None

    def _g_grid(self) -> np.ndarray:
        K = self.cfg.K
        i = np.arange(1, K + 1)
        return (2 * i - (K + 1)) / (K - 1)

    def _sigma2(self, a: float, d: float) -> np.ndarray:
        return np.maximum(np.exp(a + d * self._g_grid()), self.cfg.min_var)

    def _mean(self, r: np.ndarray, mu: float, c: np.ndarray) -> np.ndarray:
        T = len(r)
        p = self.cfg.p
        m = np.full(T, mu)

        for k in range(1, p + 1):
            m[k:] += c[k - 1] * r[:-k]

        return m

    def _transition_matrix(
        self,
        prev_return: float,
        alpha: float,
        beta: float,
        omega: float,
    ) -> np.ndarray:
        """
        Paper equation (7).

        IMPORTANT:
        matrix[j, i] = P(z_{t+1}=i | z_t=j, r_t)

        j = current state
        i = next state
        """

        K = self.cfg.K
        g = self._g_grid()

        phi = norm.cdf(alpha + beta * abs(prev_return))
        phi = float(np.clip(phi, 1e-12, 1.0 - 1e-12))

        omega = max(float(omega), 1e-12)

        M = np.zeros((K, K))

        for j in range(K):
            gj = g[j]

            if prev_return <= 0:
                # M-
                p_down = 0.5 * phi * (1.0 + gj)      # i = j - 1
                p_up = 0.5 * phi * (1.0 - gj)        # i = j + 1
                p_stay = 1.0 - phi
            else:
                # M+
                p_down = 0.5 * phi * omega * (1.0 + gj)
                p_up = 0.5 * (phi / omega) * (1.0 - gj)
                p_stay = 1.0 - p_down - p_up

            if j == 0:
                p_down = 0.0

            if j == K - 1:
                p_up = 0.0

            p_down = max(0.0, p_down)
            p_up = max(0.0, p_up)
            p_stay = max(0.0, p_stay)

            total = p_down + p_stay + p_up

            if total <= 0 or not np.isfinite(total):
                M[j, j] = 1.0
            else:
                p_down /= total
                p_stay /= total
                p_up /= total

                if j > 0:
                    M[j, j - 1] = p_down

                M[j, j] = p_stay

                if j < K - 1:
                    M[j, j + 1] = p_up

        return M

    def _student_logpdf(
        self,
        eps: np.ndarray,
        sigma2: np.ndarray,
        nu: float,
    ) -> np.ndarray:
        """
        Student-t with unit-variance innovation u_t.
        Therefore Var(eps | state i) = sigma2_i.

        nu must be > 2.
        """

        nu = max(float(nu), 2.0001)
        sigma2 = np.maximum(sigma2, self.cfg.min_var)

        log_const = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log((nu - 2.0) * np.pi * sigma2)
        )

        z = eps[:, None] ** 2 / ((nu - 2.0) * sigma2[None, :])

        log_kernel = -0.5 * (nu + 1.0) * np.log1p(z)

        return log_const[None, :] + log_kernel

    def _hamilton_filter(
        self,
        r: np.ndarray,
        mu: float,
        c: np.ndarray,
        a: float,
        d: float,
        alpha: float,
        beta: float,
        omega: float,
        nu: float,
    ):
        T = len(r)
        K = self.cfg.K

        sigma2 = self._sigma2(a, d)
        mean = self._mean(r, mu, c)
        eps = r - mean

        log_B = self._student_logpdf(eps, sigma2, nu)

        filt = np.zeros((T, K))
        pred = np.zeros((T, K))
        loglik_terms = np.zeros(T)

        pi0 = np.full(K, 1.0 / K)

        max_b = np.max(log_B[0])
        B = np.exp(log_B[0] - max_b)

        alpha_t = pi0 * B
        scale = alpha_t.sum()

        if scale <= 0 or not np.isfinite(scale):
            return -np.inf, None, None, None

        filt[0] = alpha_t / scale
        pred[0] = pi0
        loglik_terms[0] = np.log(scale) + max_b

        for t in range(1, T):
            M_prev = self._transition_matrix(
                prev_return=r[t - 1],
                alpha=alpha,
                beta=beta,
                omega=omega,
            )

            pred[t] = filt[t - 1] @ M_prev

            max_b = np.max(log_B[t])
            B = np.exp(log_B[t] - max_b)

            alpha_t = pred[t] * B
            scale = alpha_t.sum()

            if scale <= 0 or not np.isfinite(scale):
                return -np.inf, None, None, None

            filt[t] = alpha_t / scale
            loglik_terms[t] = np.log(scale) + max_b

        return float(np.sum(loglik_terms)), filt, pred, sigma2

    def _unpack(self, theta: np.ndarray):
        p = self.cfg.p

        mu = theta[0]
        c = theta[1:1 + p]

        idx = 1 + p

        a = theta[idx]
        d = np.exp(theta[idx + 1])

        alpha = theta[idx + 2]
        beta = np.exp(theta[idx + 3])
        omega = np.exp(theta[idx + 4])

        nu = 2.0001 + np.exp(theta[idx + 5])

        return mu, c, a, d, alpha, beta, omega, nu

    def _neg_loglik(self, theta: np.ndarray, r: np.ndarray) -> float:
        mu, c, a, d, alpha, beta, omega, nu = self._unpack(theta)

        ll, _, _, _ = self._hamilton_filter(
            r=r,
            mu=mu,
            c=c,
            a=a,
            d=d,
            alpha=alpha,
            beta=beta,
            omega=omega,
            nu=nu,
        )

        if not np.isfinite(ll):
            return 1e12

        return -ll

    def fit(self, returns: np.ndarray) -> RossiHMMResult:
        r = np.asarray(returns, dtype=float).reshape(-1)
        r = r[np.isfinite(r)]

        if len(r) < 100:
            raise ValueError("Need at least 100 observations.")

        rng = np.random.default_rng(self.cfg.seed)

        sample_var = np.var(r)
        mu0 = np.mean(r)
        c0 = np.zeros(self.cfg.p)

        base_theta = np.r_[
            mu0,
            c0,
            np.log(sample_var + self.cfg.min_var),
            np.log(1.0),
            -2.0,
            np.log(0.5),
            np.log(1.5),
            np.log(8.0 - 2.0001),
        ]

        best = None

        for start in range(self.cfg.n_starts):
            theta0 = base_theta.copy()

            if start > 0:
                theta0 += rng.normal(0.0, 0.3, size=theta0.shape)
                theta0[0] = mu0 + rng.normal(0.0, np.std(r) * 0.1)
                theta0[1:1 + self.cfg.p] = rng.normal(0.0, 0.05, self.cfg.p)

            opt = minimize(
                fun=lambda th: self._neg_loglik(th, r),
                x0=theta0,
                method="Nelder-Mead",
                options={
                    "maxiter": self.cfg.n_iter,
                    "xatol": 1e-5,
                    "fatol": 1e-5,
                    "disp": True,
                },
            )

            if best is None or opt.fun < best.fun:
                best = opt

        mu, c, a, d, alpha, beta, omega, nu = self._unpack(best.x)

        ll, filt, pred, sigma2 = self._hamilton_filter(
            r=r,
            mu=mu,
            c=c,
            a=a,
            d=d,
            alpha=alpha,
            beta=beta,
            omega=omega,
            nu=nu,
        )

        states = np.argmax(filt, axis=1)
        labels = self._labels(states)

        self.params_ = {
            "mu": mu,
            "c": c,
            "a": a,
            "d": d,
            "alpha": alpha,
            "beta": beta,
            "omega": omega,
            "nu": nu,
            "success": best.success,
            "message": best.message,
        }

        self.sigma2_ = sigma2
        self.filtered_proba_ = filt
        self.states_ = states

        return RossiHMMResult(
            params=self.params_,
            sigma2=sigma2,
            filtered_proba=filt,
            states=states,
            labels=labels,
            loglik=ll,
        )

    def _labels(self, states: np.ndarray) -> np.ndarray:
        K = self.cfg.K
        out = []

        for s in states:
            if s == 0:
                out.append("LOW_VOL")
            elif s == K - 1:
                out.append("HIGH_VOL")
            else:
                out.append(f"VOL_REGIME_{s}")

        return np.array(out)

    def filter(self, returns: np.ndarray):
        if self.params_ is None:
            raise RuntimeError("Fit the model first.")

        r = np.asarray(returns, dtype=float).reshape(-1)

        ll, filt, pred, sigma2 = self._hamilton_filter(
            r=r,
            mu=self.params_["mu"],
            c=self.params_["c"],
            a=self.params_["a"],
            d=self.params_["d"],
            alpha=self.params_["alpha"],
            beta=self.params_["beta"],
            omega=self.params_["omega"],
            nu=self.params_["nu"],
        )

        states = np.argmax(filt, axis=1)
        labels = self._labels(states)

        return states, labels, filt, ll
"""
MS-GARCH HMM — Haas, Mittnik & Paolella (2004) approximation.

Chaque état k a son propre GARCH(1,1) indépendant.
Le filtre de Hamilton propage les probabilités d'état.
Le M-step optimise un GARCH pondéré par les probabilités lissées (gamma).

Références
----------
Haas M., Mittnik S., Paolella M.S. (2004).
    "A New Approach to Markov-Switching GARCH Models."
    Journal of Financial Econometrics, 2(4), 493–530.

Gray S.F. (1996).
    "Modeling the Conditional Distribution of Interest Rates
    as a Regime-Switching Process."
    Journal of Financial Economics, 42(1), 27–62.
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class GarchHMMConfig:
    K: int = 3
    dist: Literal["gaussian", "student"] = "student"
    fixed_nu: float = 8.0

    min_omega: float = 1e-8
    max_persistence: float = 0.9995
    min_var: float = 1e-10

    n_iter: int = 100
    tol: float = 1e-6
    n_init: int = 5
    seed: int = 42

    m_step_maxiter: int = 200
    m_step_ftol: float = 1e-9

    log_zero_floor: float = -1e300
    exp_clip: float = 35.0
    def __post_init__(self):
        assert self.K >= 2, "K doit être >= 2"
        assert self.fixed_nu > 2.0, "nu doit être > 2 pour variance finie"
        assert 0.0 < self.max_persistence < 1.0


# ============================================================
# RÉSULTATS
# ============================================================

@dataclass
class GarchHMMResult:
    """Résultats du modèle ajusté."""

    # Paramètres HMM
    pi: np.ndarray           # (K,)   distribution initiale
    A: np.ndarray            # (K, K) matrice de transition

    # Paramètres GARCH par état
    mu: np.ndarray           # (K,)
    omega: np.ndarray        # (K,)
    alpha: np.ndarray        # (K,)
    beta: np.ndarray         # (K,)

    # Historique vraisemblance
    ll_hist: np.ndarray      # (n_iter,)
    ll_final: float

    # Config utilisée
    config: GarchHMMConfig

    
    @property
    def persistence(self) -> np.ndarray:
        return self.alpha + self.beta

    @property
    def unconditional_var(self) -> np.ndarray:
        pers = np.clip(self.persistence, 0, 0.9999)
        return self.omega / np.maximum(1.0 - pers, 1e-8)

    def summary(self) -> str:
        lines = ["=" * 55, "MS-GARCH Summary", "=" * 55]
        for k in range(self.config.K):
            lines.append(f"\n  État {k}")
            lines.append(f"    mu      = {self.mu[k]:.6f}")
            lines.append(f"    omega   = {self.omega[k]:.2e}")
            lines.append(f"    alpha   = {self.alpha[k]:.4f}")
            lines.append(f"    beta    = {self.beta[k]:.4f}")
            lines.append(f"    α+β     = {self.persistence[k]:.4f}")
            lines.append(f"    vol unc = {np.sqrt(self.unconditional_var[k]):.4f}")
        lines.append(f"\n  Log-lik final : {self.ll_final:.4f}")
        lines.append("=" * 55)
        return "\n".join(lines)
    

# ============================================================
# HELPERS PARAMÉTRISATION
# ============================================================

def _softmax_ab(a_raw: float, b_raw: float, cap: float):
    """
    (a_raw, b_raw) ∈ ℝ²  →  alpha, beta > 0  avec  alpha + beta < cap.
    Transformation bijective, différentiable partout.
    """
    ea = np.exp(np.clip(a_raw, -35.0, 35.0))
    eb = np.exp(np.clip(b_raw, -35.0, 35.0))
    denom = 1.0 + ea + eb
    return cap * ea / denom, cap * eb / denom


def _inv_softmax_ab(alpha: float, beta: float, cap: float):
    """Inverse de _softmax_ab."""
    eps = 1e-8
    alpha = np.clip(alpha, eps, cap - eps)
    beta  = np.clip(beta,  eps, cap - alpha - eps)
    rest  = cap - alpha - beta
    rest  = max(rest, eps)
    return np.log(alpha / rest), np.log(beta / rest)


def _pack(mu, omega, alpha, beta, cap, min_omega):
    a_raw, b_raw = _inv_softmax_ab(alpha, beta, cap)
    # omega stocké en log(omega - min_omega) pour garantir omega > min_omega
    log_omega_excess = np.log(max(omega - min_omega, 1e-12))
    return np.array([mu, log_omega_excess, a_raw, b_raw])


def _unpack(theta, cap, min_omega):
    mu    = theta[0]
    omega = np.exp(theta[1]) + min_omega
    alpha, beta = _softmax_ab(theta[2], theta[3], cap)
    return mu, omega, alpha, beta


# ============================================================
# GARCH FILTER UNIVARIÉ
# ============================================================

def _garch_variances(y, mu, omega, alpha, beta, min_var):
    """
    Filtre GARCH(1,1) sur la série complète.
    h[0] initialisé à la variance inconditionnelle.
    """
    T   = len(y)
    h   = np.empty(T)
    eps = y - mu

    persistence = alpha + beta
    if persistence < 0.9999:
        h0 = omega / max(1.0 - persistence, 1e-10)
    else:
        h0 = float(np.var(y))

    h[0] = max(h0, min_var)

    for t in range(1, T):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
        if not np.isfinite(h[t]) or h[t] < min_var:
            h[t] = min_var

    return h


# ============================================================
# LOG-DENSITÉS
# ============================================================

def _log_density_gaussian(eps2, h):
    """Log-densité N(0, h) — vectorisé."""
    return -0.5 * (np.log(2.0 * np.pi) + np.log(h) + eps2 / h)


def _log_density_student(eps2, h, nu):
    """Log-densité Student-t(0, h, nu) — vectorisé."""
    return (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * np.log((nu - 2.0) * np.pi * h)
        - ((nu + 1.0) / 2.0) * np.log1p(eps2 / ((nu - 2.0) * h))
    )


# ============================================================
# WEIGHTED GARCH NLL (M-STEP)
# ============================================================

def _weighted_nll(theta, y, weights, cfg: GarchHMMConfig):
    """
    NLL pondérée pour un état k.
    Minimisée pendant le M-step avec les poids gamma[:, k].
    """
    mu, omega, alpha, beta = _unpack(
        theta, cfg.max_persistence, cfg.min_omega
    )

    h    = _garch_variances(y, mu, omega, alpha, beta, cfg.min_var)
    h    = np.maximum(h, cfg.min_var)
    eps2 = (y - mu) ** 2

    if cfg.dist == "student":
        ll = _log_density_student(eps2, h, cfg.fixed_nu)
    else:
        ll = _log_density_gaussian(eps2, h)

    obj = -np.dot(weights, ll)
    return obj if np.isfinite(obj) else 1e300


# ============================================================
# LOG-ÉMISSIONS  (T × K)
# ============================================================

def _compute_log_B(y, mu, omega, alpha, beta, cfg: GarchHMMConfig):
    """Matrice log B[t, k] = log p(y_t | S_t=k, params_k)."""
    T, K = len(y), cfg.K
    log_B = np.empty((T, K))

    for k in range(K):
        h    = _garch_variances(y, mu[k], omega[k], alpha[k], beta[k], cfg.min_var)
        h    = np.maximum(h, cfg.min_var)
        eps2 = (y - mu[k]) ** 2

        if cfg.dist == "student":
            log_B[:, k] = _log_density_student(eps2, h, cfg.fixed_nu)
        else:
            log_B[:, k] = _log_density_gaussian(eps2, h)

    # remplace -inf / nan par un plancher fini
    log_B = np.where(np.isfinite(log_B), log_B, cfg.log_zero_floor)
    return log_B


# ============================================================
# FORWARD-BACKWARD (LOG-ESPACE)
# ============================================================

def _forward(log_pi, log_A, log_B):
    """
    Forward en log-espace avec log-sum-exp.
    Retourne log_alpha (T, K) et log-vraisemblance totale.
    """
    T, K = log_B.shape
    log_alpha = np.empty((T, K))

    log_alpha[0] = log_pi + log_B[0]

    for t in range(1, T):
        # log_alpha[t-1, j] + log_A[j, k] → max sur j puis log-sum-exp
        acc = log_alpha[t - 1, :, None] + log_A   # (K, K)
        m   = acc.max(axis=0)                       # (K,)
        log_alpha[t] = m + np.log(np.exp(acc - m).sum(axis=0)) + log_B[t]

    ll = np.logaddexp.reduce(log_alpha[-1])
    return log_alpha, ll


def _backward(log_A, log_B):
    """Backward en log-espace."""
    T, K = log_B.shape
    log_beta = np.zeros((T, K))

    for t in range(T - 2, -1, -1):
        acc = log_A + log_B[t + 1] + log_beta[t + 1]  # (K, K)
        m   = acc.max(axis=1)
        log_beta[t] = m + np.log(np.exp(acc - m[:, None]).sum(axis=1))

    return log_beta


def _e_step(log_pi, log_A, log_B):
    """
    E-step complet.
    Retourne gamma (T, K), xi (T-1, K, K), ll scalaire.
    """
    T, K = log_B.shape

    log_alpha, ll = _forward(log_pi, log_A, log_B)
    log_beta      = _backward(log_A, log_B)

    # gamma
    log_gamma = log_alpha + log_beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    # xi
    log_xi = np.empty((T - 1, K, K))
    for t in range(T - 1):
        log_xi[t] = (
            log_alpha[t, :, None]
            + log_A
            + log_B[t + 1]
            + log_beta[t + 1]
        )
        log_xi[t] -= np.logaddexp.reduce(
            log_xi[t].ravel()
        )

    xi = np.exp(log_xi)

    return gamma, xi, ll


# ============================================================
# M-STEP HMM (pi, A)
# ============================================================

def _m_step_hmm(gamma, xi, eps=1e-8):
    """Met à jour pi et A à partir des statistiques suffisantes."""
    pi = gamma[0] + eps
    pi /= pi.sum()

    A = xi.sum(axis=0) + eps          # (K, K)
    A /= A.sum(axis=1, keepdims=True)

    return np.log(pi), np.log(A)


# ============================================================
# M-STEP GARCH
# ============================================================

def _m_step_garch(y, gamma, mu, omega, alpha, beta, cfg: GarchHMMConfig):
    """
    Optimise les paramètres GARCH de chaque état k
    en minimisant la NLL pondérée par gamma[:, k].
    """
    K = cfg.K

    for k in range(K):
        w = gamma[:, k]
        w_sum = w.sum()

        if w_sum < 1e-6:
            logger.debug("État %d : poids trop faibles, skip", k)
            continue

        w = w / w_sum   # normalise pour stabilité numérique

        theta0 = _pack(
            mu[k], omega[k], alpha[k], beta[k],
            cfg.max_persistence, cfg.min_omega,
        )

        res = minimize(
            fun=_weighted_nll,
            x0=theta0,
            args=(y, w, cfg),
            method="L-BFGS-B",
            options={
                "maxiter": cfg.m_step_maxiter,
                "ftol":    cfg.m_step_ftol,
                "gtol":    1e-7,
            },
        )

        if np.all(np.isfinite(res.x)):
            mu_k, omega_k, alpha_k, beta_k = _unpack(
                res.x, cfg.max_persistence, cfg.min_omega
            )
            mu[k], omega[k], alpha[k], beta[k] = (
                mu_k, omega_k, alpha_k, beta_k
            )
            if not res.success:
                logger.debug(
                    "État %d : L-BFGS-B non convergé (%s) — "
                    "paramètres partiels utilisés.",
                    k, res.message,
                )
        else:
            logger.warning(
                "État %d : paramètres non finis après optimisation — "
                "paramètres précédents conservés.",
                k,
            )

    return mu, omega, alpha, beta


# ============================================================
# INITIALISATION
# ============================================================

def _init_params(y, cfg: GarchHMMConfig, rng: np.random.Generator):
    """
    Initialisation avec brisure de symétrie entre états.
    Chaque restart utilise un rng différent.
    """
    K   = cfg.K
    T   = len(y)

    # ---- pi uniforme ----
    log_pi = np.full(K, -np.log(K))

    # ---- A diagonale dominante ----
    A = np.full((K, K), 0.05 / (K - 1))
    np.fill_diagonal(A, 0.95)
    A /= A.sum(axis=1, keepdims=True)
    log_A = np.log(A)

    # ---- GARCH : persistence différente par état ----
    global_var = max(float(np.var(y)), cfg.min_var)

    # persistences croissantes : low vol → high vol
    persistences = np.linspace(0.80, 0.97, K)
    alphas       = np.linspace(0.04, 0.15, K)
    betas        = persistences - alphas

    # omegas calibrés sur la variance globale et la persistence
    omegas = global_var * (1.0 - persistences)
    omegas = np.maximum(omegas, cfg.min_omega * 10)

    # mu : petite perturbation aléatoire autour de la moyenne globale
    global_mu = float(np.mean(y))
    mu_noise  = rng.standard_normal(K) * np.std(y) * 0.1
    mu        = global_mu + mu_noise

    # optionnel : shuffle l'ordre pour diversifier les restarts
    perm   = rng.permutation(K)
    alphas = alphas[perm]
    betas  = betas[perm]
    omegas = omegas[perm]

    return log_pi, log_A, mu, omegas, alphas, betas


# ============================================================
# UNE PASSE EM
# ============================================================

def _run_em(y, log_pi, log_A, mu, omega, alpha, beta, cfg: GarchHMMConfig):
    """
    Lance l'algorithme EM jusqu'à convergence ou n_iter.
    Retourne les paramètres finaux et l'historique de ll.
    """
    ll_prev = -np.inf
    ll_hist = []

    for iteration in range(cfg.n_iter):

        # ------ E-step ------
        log_B = _compute_log_B(y, mu, omega, alpha, beta, cfg)
        gamma, xi, ll = _e_step(log_pi, log_A, log_B)

        ll_hist.append(ll)

        if not np.isfinite(ll):
            logger.warning("ll non fini à l'itération %d — arrêt.", iteration)
            break

        # ------ Convergence ------
        delta = ll - ll_prev
        if iteration > 0:
            if delta < 0:
                logger.debug(
                    "ll a décru de %.4e à l'itération %d (ΔLL=%.2e) — "
                    "instabilité numérique.",
                    ll, iteration, delta,
                )
            if abs(delta) < cfg.tol:
                logger.info("Convergé à l'itération %d (ΔLL=%.2e).", iteration, delta)
                break

        ll_prev = ll

        # ------ M-step HMM ------
        log_pi, log_A = _m_step_hmm(gamma, xi)

        # ------ M-step GARCH ------
        mu, omega, alpha, beta = _m_step_garch(
            y, gamma, mu, omega, alpha, beta, cfg
        )

    return log_pi, log_A, mu, omega, alpha, beta, np.array(ll_hist)


# ============================================================
# MODÈLE PRINCIPAL
# ============================================================

class GarchHMM:
    """
    Markov-Switching GARCH(1,1) — approximation Haas et al. (2004).

    Usage
    -----
    >>> cfg   = GarchHMMConfig(K=3, dist="student", fixed_nu=8.0)
    >>> model = GarchHMM(cfg)
    >>> result = model.fit(returns)          # returns : array (T,)
    >>> states = model.predict_states(returns)
    >>> vol    = model.conditional_volatility(returns)
    """

    def __init__(self, cfg: GarchHMMConfig | None = None):
        self.cfg    = cfg or GarchHMMConfig()
        self.result : GarchHMMResult | None = None

        # stockés après fit
        self._log_pi  = None
        self._log_A   = None
        self._mu      = None
        self._omega   = None
        self._alpha   = None
        self._beta    = None
    @property
    def A(self):
        self._require_fit()
        return self.result.A

    def filter(self, X):
        """
        Compatible avec ton pipeline HMM existant.
        Retourne: states, alpha, ll
        """
        self._require_fit()

        y = np.asarray(X, dtype=float).ravel()

        alpha, ll = self._forward_probs(y)
        states = np.argmax(alpha, axis=1)

        return states, alpha, ll

    def fit_multi_start(self, X, seeds=None, verbose=True):
        """
        Compatible avec GaussianHMM / StudentTHMM.
        Les restarts sont déjà gérés par cfg.n_init.
        """
        seed = self.cfg.seed if hasattr(self.cfg, "seed") else 42
        return self.fit(X, seed=seed)
    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    def fit(self, y, seed: int = 42) -> GarchHMMResult:
        """
        Ajuste le modèle par EM avec plusieurs restarts.

        Parameters
        ----------
        y    : array-like (T,)  — rendements bruts (pas les log-rendements %)
        seed : graine aléatoire pour reproductibilité
        """
        y   = np.asarray(y, dtype=float).ravel()
        cfg = self.cfg

        if len(y) < 50:
            warnings.warn("Série trop courte (< 50 obs). Résultats peu fiables.")

        best_ll   = -np.inf
        best_params = None

        rng_master = np.random.default_rng(seed)

        for restart in range(cfg.n_init):

            rng = np.random.default_rng(rng_master.integers(0, 2**32))
            log_pi, log_A, mu, omega, alpha, beta = _init_params(y, cfg, rng)

            try:
                log_pi_r, log_A_r, mu_r, omega_r, alpha_r, beta_r, ll_hist = (
                    _run_em(y, log_pi, log_A, mu, omega, alpha, beta, cfg)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Restart %d échoué : %s", restart, exc)
                continue

            ll_final = ll_hist[-1] if len(ll_hist) > 0 else -np.inf

            logger.info(
                "Restart %d/%d : ll_final = %.4f",
                restart + 1, cfg.n_init, ll_final,
            )

            if np.isfinite(ll_final) and ll_final > best_ll:
                best_ll = ll_final
                best_params = (log_pi_r, log_A_r, mu_r, omega_r, alpha_r, beta_r, ll_hist)

        if best_params is None:
            raise RuntimeError(
                "Tous les restarts ont échoué. "
                "Vérifie tes données (NaN, variance nulle ?)."
            )

        log_pi, log_A, mu, omega, alpha, beta, ll_hist = best_params

        self._log_pi = log_pi
        self._log_A  = log_A
        self._mu     = mu
        self._omega  = omega
        self._alpha  = alpha
        self._beta   = beta

        self.result = GarchHMMResult(
            pi      = np.exp(log_pi),
            A       = np.exp(log_A),
            mu      = mu.copy(),
            omega   = omega.copy(),
            alpha   = alpha.copy(),
            beta    = beta.copy(),
            ll_hist = ll_hist,
            ll_final= float(ll_hist[-1]),
            config  = cfg,
        )

        return self.result

    # --------------------------------------------------------
    # HELPERS INTERNES (post-fit)
    # --------------------------------------------------------

    def _require_fit(self):
        if self.result is None:
            raise RuntimeError("Appelle fit() avant de prédire.")

    def _log_B(self, y):
        return _compute_log_B(
            y,
            self._mu, self._omega, self._alpha, self._beta,
            self.cfg,
        )

    def _forward_probs(self, y):
        """Retourne filtered probs alpha (T, K) et ll."""
        log_alpha, ll = _forward(self._log_pi, self._log_A, self._log_B(y))
        alpha = np.exp(log_alpha - np.logaddexp.reduce(log_alpha, axis=1, keepdims=True))
        return alpha, ll

    def _smoothed_probs(self, y):
        """Retourne smoothed probs gamma (T, K)."""
        log_B = self._log_B(y)
        gamma, _, ll = _e_step(self._log_pi, self._log_A, log_B)
        return gamma, ll

    # --------------------------------------------------------
    # API PUBLIQUE
    # --------------------------------------------------------

    def predict_states(self, y, smoothed: bool = True) -> np.ndarray:
        """
        Retourne l'état le plus probable à chaque instant.

        Parameters
        ----------
        smoothed : True  → probabilités lissées (recommandé en post-analyse)
                   False → probabilités filtrées (temps réel / live trading)
        """
        self._require_fit()
        y = np.asarray(y, dtype=float).ravel()

        if smoothed:
            probs, _ = self._smoothed_probs(y)
        else:
            probs, _ = self._forward_probs(y)

        return np.argmax(probs, axis=1)

    def state_probabilities(self, y, smoothed: bool = True) -> np.ndarray:
        """Retourne P(S_t = k | data) pour chaque t et k. Shape (T, K)."""
        self._require_fit()
        y = np.asarray(y, dtype=float).ravel()

        if smoothed:
            probs, _ = self._smoothed_probs(y)
        else:
            probs, _ = self._forward_probs(y)

        return probs

    def conditional_volatility(self, y, smoothed: bool = True) -> np.ndarray:
        """
        Vol conditionnelle agrégée : sqrt( Σ_k P(S_t=k) * h_k(t) ).
        Shape (T,).
        """
        self._require_fit()
        y = np.asarray(y, dtype=float).ravel()

        probs = self.state_probabilities(y, smoothed=smoothed)    # (T, K)
        H     = self._all_state_variances(y)                       # (T, K)

        exp_var = np.sum(probs * H, axis=1)
        return np.sqrt(np.maximum(exp_var, self.cfg.min_var))

    def state_volatility(self, y) -> np.ndarray:
        """
        Vol conditionnelle de chaque état indépendamment.
        Shape (T, K). Utile pour le diagnostic.
        """
        self._require_fit()
        y = np.asarray(y, dtype=float).ravel()
        H = self._all_state_variances(y)
        return np.sqrt(np.maximum(H, self.cfg.min_var))

    def regime_label(self, y, smoothed: bool = True) -> np.ndarray:
        """
        Retourne 'low', 'mid' ou 'high' selon l'état trié par vol inconditionnelle.
        Shape (T,) dtype object.
        """
        self._require_fit()
        cfg = self.cfg

        unc_vol = np.sqrt(self.result.unconditional_var)
        order   = np.argsort(unc_vol)          # état → rang par vol croissante

        labels_map = {
            0: "low",
            cfg.K - 1: "high",
        }

        state_to_label = {}
        for rank, state in enumerate(order):
            if rank == 0:
                state_to_label[state] = "low"
            elif rank == cfg.K - 1:
                state_to_label[state] = "high"
            else:
                state_to_label[state] = "mid"

        states  = self.predict_states(y, smoothed=smoothed)
        regimes = np.array([state_to_label[s] for s in states], dtype=object)
        return regimes

    # --------------------------------------------------------
    # INTERNE
    # --------------------------------------------------------

    def _all_state_variances(self, y) -> np.ndarray:
        """Retourne H (T, K) — variance GARCH de chaque état."""
        T, K = len(y), self.cfg.K
        H = np.empty((T, K))
        for k in range(K):
            H[:, k] = _garch_variances(
                y,
                self._mu[k], self._omega[k],
                self._alpha[k], self._beta[k],
                self.cfg.min_var,
            )
        return H
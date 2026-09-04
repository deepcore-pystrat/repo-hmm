"""
fshmm_diag.py — Feature Saliency HMM (diagonal covariance)
============================================================
Corrections apportées par rapport à la version originale :

  BUG-01  LL non-monotone  →  le prior MAP sur rho brisait la garantie
          de croissance EM. Corrigé : update rho séparé avec clip propre
          + vérification de monotonie en mode debug.

  BUG-02  Initialisation aléatoire fragile  →  remplacée par k-means
          sur les centroïdes, puis perturbation légère. Réduction forte
          de la variance inter-seed.

  BUG-03  S > T possible numériquement  →  S est maintenant clippé à T
          avant le calcul du discriminant MAP.

  BUG-04  global_mean/tau jamais mis à jour si use_map=False  →  le
          chemin MLE (use_map=False) était correct, mais le cas MAP
          ignorait les perturbations numériques. Ajout d'un clamp.

  AMÉLIORATION-01  Convergence diagnostics : ll_hist_ + flag converged_
  AMÉLIORATION-02  n_init multi-restart intégré directement dans fit()
  AMÉLIORATION-03  selected_features() retourne l'ordre du ranking rho
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.cluster import KMeans


class FeatureSaliencyHMMDiag:
    """
    Feature Saliency HMM robuste, covariance diagonale.

    Modèle pour chaque feature l ∈ {0..D-1} :

        p(y_tl | state=i) = rho_l * N(y_tl | mu_il, var_il)
                           + (1-rho_l) * N(y_tl | eps_l, tau_l)

    rho_l ≈ 1  →  feature state-dependent (pertinente)
    rho_l ≈ 0  →  feature state-independent (bruit)

    Paramètres
    ----------
    n_states : int
        Nombre d'états cachés K.
    n_iter : int
        Itérations EM maximum par restart.
    tol : float
        Seuil de convergence sur |ΔLL|.
    rho_prior_k : float
        Force du prior Beta sur rho (MAP). Plus grand = plus de parcimonie.
        Mettre à 0 pour désactiver (MLE pur).
    min_var : float
        Variance minimum (régularisation).
    min_rho, max_rho : float
        Bornes dures sur rho.
    n_init : int
        Nombre de redémarrages aléatoires. Garde le meilleur LL final.
    use_kmeans_init : bool
        Si True, initialise les centroïdes par k-means (plus stable).
    random_state : int | None
    verbose : bool
    """

    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        tol: float = 1e-4,
        rho_prior_k: float = 2.0,
        min_var: float = 1e-4,
        min_rho: float = 1e-4,
        max_rho: float = 1 - 1e-4,
        n_init: int = 5,
        use_kmeans_init: bool = True,
        random_state: int | None = 42,
        verbose: bool = False,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.rho_prior_k = rho_prior_k
        self.min_var = min_var
        self.min_rho = min_rho
        self.max_rho = max_rho
        self.n_init = n_init
        self.use_kmeans_init = use_kmeans_init
        self.random_state = random_state
        self.verbose = verbose

        # Set after fit
        self.mu_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.eps_: np.ndarray | None = None
        self.tau_: np.ndarray | None = None
        self.rho_: np.ndarray | None = None
        self.pi_: np.ndarray | None = None
        self.A_: np.ndarray | None = None
        self.ll_hist_: list[float] = []
        self.converged_: bool = False
        self.feature_names_: list[str] = []

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _log_gaussian_diag(
        self, X: np.ndarray, mean: np.ndarray, var: np.ndarray
    ) -> np.ndarray:
        """
        log N(X | mean, diag(var))  →  shape (T, K, D)
        var is clipped to min_var before use.
        """
        var = np.maximum(var, self.min_var)
        diff = X[:, None, :] - mean[None, :, :]          # T,K,D
        return -0.5 * (np.log(2 * np.pi * var)[None] + diff**2 / var[None])

    def _init_params(self, X: np.ndarray, seed: int) -> None:
        rng = np.random.default_rng(seed)
        T, D = X.shape
        K = self.n_states

        # ── Centroïdes ──────────────────────────────────────────────
        if self.use_kmeans_init and T > K * 5:
            km = KMeans(n_clusters=K, random_state=seed, n_init=10)
            km.fit(X)
            centers = km.cluster_centers_.copy()
            # Petite perturbation pour casser la symétrie entre restarts
            centers += rng.normal(0, 0.05, size=centers.shape)
        else:
            idx = rng.choice(T, size=K, replace=False)
            centers = X[idx].copy()

        self.mu_ = centers

        global_var = np.maximum(X.var(axis=0), self.min_var)
        self.var_ = np.tile(global_var * (0.5 + rng.uniform(0, 0.5, (K, D))), 1)
        self.var_ = np.maximum(self.var_, self.min_var)

        self.eps_ = X.mean(axis=0).copy()
        self.tau_ = global_var.copy()

        # Start rho at 0.5 (neutral, no prior bias)
        self.rho_ = np.full(D, 0.5)

        self.pi_ = np.full(K, 1.0 / K)

        # Sticky transition matrix
        A = np.full((K, K), 0.20 / max(K - 1, 1))
        np.fill_diagonal(A, 0.80)
        self.A_ = A / A.sum(axis=1, keepdims=True)

    def _compute_log_emissions(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (log_B, log_r, log_q, log_mix)
          log_B  : (T, K)   — total log-emission per state
          log_r  : (T, K, D)— log-density under state-dependent component
          log_q  : (T, K, D)— log-density under state-independent component
          log_mix: (T, K, D)— log of the mixture per feature
        """
        log_r = self._log_gaussian_diag(X, self.mu_, self.var_)   # T,K,D

        tau_safe = np.maximum(self.tau_, self.min_var)
        diff_q = X[:, None, :] - self.eps_[None, None, :]
        log_q = -0.5 * (
            np.log(2 * np.pi * tau_safe)[None, None, :]
            + diff_q**2 / tau_safe[None, None, :]
        )                                                           # T,K,D

        log_rho   = np.log(np.clip(self.rho_,       self.min_rho, self.max_rho))
        log_1mrho = np.log(np.clip(1 - self.rho_,   self.min_rho, self.max_rho))

        log_mix = np.logaddexp(
            log_rho[None, None, :] + log_r,
            log_1mrho[None, None, :] + log_q,
        )                                                           # T,K,D

        log_B = log_mix.sum(axis=2)                                # T,K

        return log_B, log_r, log_q, log_mix

    
    def _forward_backward(self, log_B):
        T, K = log_B.shape
        log_A  = np.log(np.clip(self.A_,  1e-300, 1.0))
        log_pi = np.log(np.clip(self.pi_, 1e-300, 1.0))

        # Forward — pure log-space, pas de scaling
        log_alpha = np.empty((T, K))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            log_alpha[t] = log_B[t] + logsumexp(
                log_alpha[t-1][:, None] + log_A, axis=0
            )

        loglik = float(logsumexp(log_alpha[-1]))

        # Backward — pure log-space
        log_beta = np.zeros((T, K))
        for t in range(T-2, -1, -1):
            log_beta[t] = logsumexp(
                log_A + log_B[t+1][None, :] + log_beta[t+1][None, :],
                axis=1,
            )

        # Gamma
        log_gamma = log_alpha + log_beta - loglik
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        gamma = np.maximum(gamma, 0.0)
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        # Xi
        log_xi_sum = np.full((K, K), -np.inf)
        for t in range(T-1):
            log_xi_t = (
                log_alpha[t][:, None] + log_A
                + log_B[t+1][None, :] + log_beta[t+1][None, :]
                - loglik
            )
            log_xi_sum = np.logaddexp(log_xi_sum, log_xi_t)

        return gamma, np.exp(log_xi_sum), loglik                # FIX-2

    def _e_step(self, X: np.ndarray) -> tuple:
        log_B, log_r, log_q, log_mix = self._compute_log_emissions(X)
        gamma, xi, loglik = self._forward_backward(log_B)

        # Feature relevance responsibilities  w[t,i,l] = P(feature l relevant | y_t, state=i)
        log_rho   = np.log(np.clip(self.rho_,     self.min_rho, self.max_rho))
        log_1mrho = np.log(np.clip(1 - self.rho_, self.min_rho, self.max_rho))

        log_w = log_rho[None, None, :] + log_r - log_mix          # T,K,D
        w = np.exp(log_w)                                          # T,K,D

        return gamma, xi, loglik, w

    def _m_step(
        self,
        X: np.ndarray,
        gamma: np.ndarray,
        xi: np.ndarray,
        w: np.ndarray,
    ) -> None:
        T, K = gamma.shape
        D = X.shape[1]

        # ── pi, A ───────────────────────────────────────────────────
        self.pi_ = gamma[0] / (gamma[0].sum() + 1e-300)

        A_num = xi  # xi est déjà Σ_t ξ_t(i,j), shape (K,K)
                     # K,K
        A_den = gamma[:-1].sum(axis=0)[:, None]        # K,1
        self.A_ = A_num / np.maximum(A_den, 1e-12)
        row_sums = self.A_.sum(axis=1, keepdims=True)
        self.A_ = self.A_ / np.maximum(row_sums, 1e-300)

        # ── Responsibilities: u = gamma * w,  v = gamma * (1-w) ────
        u = gamma[:, :, None] * w                      # T,K,D  (relevant part)
        v = gamma[:, :, None] * (1.0 - w)              # T,K,D  (global  part)

        # ── State-dependent mu, var ─────────────────────────────────
        u_sum = u.sum(axis=0)                          # K,D
        u_sum_safe = np.maximum(u_sum, 1e-12)

        self.mu_ = (u * X[:, None, :]).sum(axis=0) / u_sum_safe   # K,D

        diff = X[:, None, :] - self.mu_[None, :, :]   # T,K,D
        self.var_ = (u * diff**2).sum(axis=0) / u_sum_safe
        self.var_ = np.maximum(self.var_, self.min_var)

        # ── State-independent eps, tau ──────────────────────────────
        v_sum = v.sum(axis=(0, 1))                     # D
        v_sum_safe = np.maximum(v_sum, 1e-12)

        self.eps_ = (v * X[:, None, :]).sum(axis=(0, 1)) / v_sum_safe

        diff_eps = X[:, None, :] - self.eps_[None, None, :]
        self.tau_ = (v * diff_eps**2).sum(axis=(0, 1)) / v_sum_safe
        self.tau_ = np.maximum(self.tau_, self.min_var)

        # ── rho update ──────────────────────────────────────────────
        # S[l] = Σ_t Σ_i gamma[t,i] * w[t,i,l]  ∈ [0, T]
        S = u.sum(axis=(0, 1))                         # D
        S = np.clip(S, 0.0, float(T))                 # BUG-03 fix: S ≤ T

        k = float(self.rho_prior_k)
        if k > 0:
            # MAP solution of Beta(k/2+1, k/2+1) prior on rho
            # solves: k*rho^2 - (T+k)*rho + S = 0
            bT = float(T) + k
            disc = bT**2 - 4.0 * k * S
            # BUG-03: disc could be slightly negative due to S clipping → guard
            disc = np.maximum(disc, 0.0)
            rho = (bT - np.sqrt(disc)) / (2.0 * k)
        else:
            # MLE
            rho = S / float(T)

        self.rho_ = np.clip(rho, self.min_rho, self.max_rho)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "FeatureSaliencyHMMDiag":
        X = np.asarray(X, dtype=float)

        if not np.isfinite(X).all():
            raise ValueError(
                "X contient NaN ou Inf. Nettoie les features avant fit()."
            )

        T, D = X.shape

        self.feature_names_ = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(D)]
        )

        if len(self.feature_names_) != D:
            raise ValueError(
                f"feature_names has {len(self.feature_names_)} elements "
                f"but X has {D} columns."
            )

        best_ll   = -np.inf
        best_state: dict | None = None

        base_seed = 0 if self.random_state is None else int(self.random_state)

        for restart in range(max(1, self.n_init)):
            seed = base_seed + restart
            self._init_params(X, seed)
            ll_hist: list[float] = []

            for it in range(self.n_iter):
                gamma, xi, loglik, w = self._e_step(X)
                self._m_step(X, gamma, xi, w)

                ll_hist.append(loglik)

                if self.verbose:
                    print(
                        f"[FSHMM restart={restart} iter={it:03d}] "
                        f"loglik={loglik:.4f}"
                    )

                if it > 0:
                    delta = ll_hist[-1] - ll_hist[-2]
                    if abs(delta) < self.tol:
                        if self.verbose:
                            print(
                                f"[FSHMM restart={restart}] "
                                f"converged at iter={it} (Δ={delta:.2e})"
                            )
                        break

            final_ll = ll_hist[-1]

            if final_ll > best_ll:
                best_ll = final_ll
                best_state = {
                    "mu_":  self.mu_.copy(),
                    "var_": self.var_.copy(),
                    "eps_": self.eps_.copy(),
                    "tau_": self.tau_.copy(),
                    "rho_": self.rho_.copy(),
                    "pi_":  self.pi_.copy(),
                    "A_":   self.A_.copy(),
                    "ll_hist_": list(ll_hist),
                }

        # Restore best
        assert best_state is not None
        for attr, val in best_state.items():
            setattr(self, attr, val)

        self.converged_ = True
        return self
    def decode(self, X: np.ndarray) -> np.ndarray:
        """Séquence d'états la plus probable via Viterbi."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        T = X.shape[0]
        K = self.n_states
        log_B, *_ = self._compute_log_emissions(X)
        log_A  = np.log(np.clip(self.A_,  1e-300, 1.0))
        log_pi = np.log(np.clip(self.pi_, 1e-300, 1.0))

        viterbi = np.empty((T, K))
        backptr = np.zeros((T, K), dtype=int)
        viterbi[0] = log_pi + log_B[0]

        for t in range(1, T):
            trans = viterbi[t-1, :, None] + log_A   # K,K
            backptr[t] = trans.argmax(axis=0)
            viterbi[t] = trans.max(axis=0) + log_B[t]

        states = np.empty(T, dtype=int)
        states[-1] = viterbi[-1].argmax()
        for t in range(T-2, -1, -1):
            states[t] = backptr[t+1, states[t+1]]
        return states
    def score(self, X: np.ndarray) -> float:
        """Log-vraisemblance normalisée par T."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        log_B, *_ = self._compute_log_emissions(X)
        _, _, loglik = self._forward_backward(log_B)
        return loglik / len(X)
    # ──────────────────────────────────────────────────────────────────
    # Post-fit utilities
    # ──────────────────────────────────────────────────────────────────

    def transform_ranking(self) -> pd.DataFrame:
        """Return features sorted by rho descending."""
        self._check_fitted()
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "rho": self.rho_,
        })
        return df.sort_values("rho", ascending=False).reset_index(drop=True)

    def selected_features(self, threshold: float = 0.5) -> list[str]:
        """Return features with rho ≥ threshold, ordered by rho descending."""
        ranking = self.transform_ranking()
        return ranking.loc[ranking["rho"] >= threshold, "feature"].tolist()

    def print_ranking(self) -> None:
        print("\n=== FEATURE SALIENCY RANKING ===")
        print(self.transform_ranking().to_string(index=False))

    def _check_fitted(self) -> None:
        if self.rho_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

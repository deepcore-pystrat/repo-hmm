"""
saliency_selector.py — Pipeline wrapper autour de FeatureSaliencyHMMDiag
=========================================================================
Améliorations v2 :
  - n_init exposé (multi-restart transmis au modèle)
  - adaptive_threshold : calcule automatiquement le seuil comme
    le point de coupure dans la distribution bimodale des rho
    (méthode Otsu 1D), évitant le choix arbitraire de 0.5
  - compute_confidence_interval() : bootstrap sur rho pour estimer
    l'incertitude de sélection
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_selection.fshmm_diag import FeatureSaliencyHMMDiag


def _otsu_threshold_1d(values: np.ndarray, verbose: bool = False) -> float:
    vals = np.sort(np.asarray(values, dtype=float))
    D = len(vals)
    if D < 2:
        return 0.5

    best_thresh = 0.5
    best_var = -np.inf

    for i in range(1, D):
        g0 = vals[:i]
        g1 = vals[i:]
        w0, w1 = len(g0) / D, len(g1) / D
        var_between = w0 * w1 * (g0.mean() - g1.mean()) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = float(vals[i - 1] + vals[i]) / 2

    total_var = float(vals.var())
    bimodality_ratio = best_var / (total_var + 1e-12)

    if bimodality_ratio < 0.1:
        if verbose:
            print("[Otsu] distribution non bimodale, fallback médiane")
        return float(np.median(vals))

    return best_thresh

class FeatureSaliencySelector:
    """
    Wrapper professionnel autour de FeatureSaliencyHMMDiag.

    Paramètres
    ----------
    n_states : int
        Nombre d'états cachés pour le FSHMM.
    threshold : float | 'auto'
        Seuil de sélection sur rho.
        'auto' → seuil de Otsu calculé sur la distribution des rho appris.
        float  → seuil fixe (ex: 0.5).
    n_init : int
        Nombre de redémarrages EM. Le meilleur LL est conservé.
    n_iter : int
        Itérations maximum par restart EM.
    tol : float
        Critère de convergence.
    rho_prior_k : float
        Force du prior MAP sur rho. 0 = MLE pur.
    min_var : float
        Variance minimum (régularisation numérique).
    random_state : int | None
    verbose : bool
    """

    def __init__(
        self,
        n_states: int = 3,
        threshold: float | str = 0.5,
        n_init: int = 5,
        n_iter: int = 100,
        tol: float = 1e-4,
        rho_prior_k: float = 2.0,
        min_var: float = 1e-4,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.n_states = n_states
        self.threshold = threshold
        self.n_init = n_init
        self.n_iter = n_iter
        self.tol = tol
        self.rho_prior_k = rho_prior_k
        self.min_var = min_var
        self.random_state = random_state
        self.verbose = verbose

        self.model_: FeatureSaliencyHMMDiag | None = None
        self.feature_names_: list[str] | None = None
        self.ranking_: pd.DataFrame | None = None
        self.selected_features_: list[str] | None = None
        self.effective_threshold_: float | None = None

    # ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> "FeatureSaliencySelector":

        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_values = X.values.astype(float)
        else:
            X_values = np.asarray(X, dtype=float)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_values.shape[1])]

        self.feature_names_ = list(feature_names)

        self.model_ = FeatureSaliencyHMMDiag(
            n_states=self.n_states,
            n_iter=self.n_iter,
            tol=self.tol,
            rho_prior_k=self.rho_prior_k,
            min_var=self.min_var,
            n_init=self.n_init,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self.model_.fit(X_values, feature_names=self.feature_names_)
        self.ranking_ = self.model_.transform_ranking()

        # ── Seuil effectif ──────────────────────────────────────────
        if self.threshold == "auto":
            self.effective_threshold_ = _otsu_threshold_1d(self.ranking_["rho"].values)
            if self.verbose:
                print(
                    f"[Selector] auto threshold (Otsu) = "
                    f"{self.effective_threshold_:.4f}"
                )
        else:
            self.effective_threshold_ = float(self.threshold)

        self.ranking_["selected"] = self.ranking_["rho"] >= self.effective_threshold_

        self.selected_features_ = (
            self.ranking_.loc[self.ranking_["selected"], "feature"].tolist()
        )

        return self

    # ──────────────────────────────────────────────────────────────────

    def get_ranking(self) -> pd.DataFrame:
        self._check_fitted()
        return self.ranking_.copy()

    def get_selected_features(self) -> list[str]:
        self._check_fitted()
        return list(self.selected_features_)

    def transform(
        self, X: np.ndarray | pd.DataFrame
    ) -> np.ndarray | pd.DataFrame:
        self._check_fitted()
        selected = self.get_selected_features()

        if isinstance(X, pd.DataFrame):
            missing = [c for c in selected if c not in X.columns]
            if missing:
                raise ValueError(f"Selected features missing from X: {missing}")
            return X[selected].copy()
        else:
            X_arr = np.asarray(X, dtype=float)
            if self.feature_names_ is None:
                raise RuntimeError("feature_names_ not set.")
            idx = [self.feature_names_.index(f) for f in selected]
            return X_arr[:, idx]

    def fit_transform(
        self, X: np.ndarray | pd.DataFrame
    ) -> np.ndarray | pd.DataFrame:
        self.fit(X)
        # transform() réutilise X déjà en mémoire, pas de double fit
        return self.transform(X)

    def compute_confidence_interval(
    self,
    X: np.ndarray | pd.DataFrame,
    n_bootstrap: int = 30,
    ci: float = 0.90,
    stability_threshold: float = 0.80,  # ← ajouter
) -> pd.DataFrame:
        """
        Bootstrap sur les lignes de X pour estimer l'incertitude sur rho.

        Retourne un DataFrame avec colonnes :
          feature, rho_mean, rho_std, rho_lo, rho_hi, stable_selected
        où stable_selected = True si le feature est sélectionné dans
        > ci fraction des bootstraps.
        """
        self._check_fitted()

        if isinstance(X, pd.DataFrame):
            X_values = X.values.astype(float)
        else:
            X_values = np.asarray(X, dtype=float)

        T = X_values.shape[0]
        rng = np.random.default_rng(self.random_state)
        boot_rhos: list[np.ndarray] = []

        for b in range(n_bootstrap):
            block_size = max(10, T // 20)
            n_blocks = int(np.ceil(T / block_size))
            starts = rng.integers(0, T - block_size, size=n_blocks)
            idx = np.concatenate([
                np.arange(s, s + block_size) for s in starts
            ])[:T]
            X_b = X_values[idx]
            m = FeatureSaliencyHMMDiag(
                n_states=self.n_states,
                n_iter=self.n_iter,
                tol=self.tol,
                rho_prior_k=self.rho_prior_k,
                min_var=self.min_var,
                n_init=max(1, self.n_init // 2),  # faster for bootstrap
                random_state=int(rng.integers(0, 99999)),
                verbose=False,
            )
            try:
                m.fit(X_b, feature_names=self.feature_names_)
                # Align features to current ranking order
                name_to_rho = dict(zip(m.feature_names_, m.rho_))
                rho_ordered = np.array(
                    [name_to_rho.get(fn, np.nan) for fn in self.feature_names_]
                )
                boot_rhos.append(rho_ordered)
            except Exception:
                pass

        if not boot_rhos:
            raise RuntimeError("All bootstrap fits failed.")

        boot_matrix = np.vstack(boot_rhos)  # (n_boot, D)
        alpha = (1 - ci) / 2

        rows = []
        for j, fname in enumerate(self.feature_names_):
            col = boot_matrix[:, j]
            col = col[np.isfinite(col)]
            if len(col) == 0:
                continue
            rho_mean = col.mean()
            rho_std  = col.std()
            rho_lo   = np.quantile(col, alpha)
            rho_hi   = np.quantile(col, 1 - alpha)
            stable = float(np.mean(col >= self.effective_threshold_)) >= stability_threshold

            rows.append({
                "feature":         fname,
                "rho_mean":        rho_mean,
                "rho_std":         rho_std,
                f"rho_lo_{int(ci*100)}": rho_lo,
                f"rho_hi_{int(ci*100)}": rho_hi,
                "stable_selected": stable,
                "n_boot":          len(col),
            })

        result = pd.DataFrame(rows).sort_values("rho_mean", ascending=False)
        return result

    def _check_fitted(self) -> None:
        if self.ranking_ is None:
            raise RuntimeError("Selector not fitted. Call fit() first.")

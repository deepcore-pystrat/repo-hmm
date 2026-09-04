"""
saliency_walk_forward.py — Walk-forward saliency avec stabilité temporelle
===========================================================================
Améliorations v2 :
  - run_saliency_walk_forward() : retourne aussi les métriques de stabilité
    agrégées (compute_selection_stability automatique)
  - get_consensus_features() : features stables sur toutes les fenêtres
  - Ajout de méta-données par fenêtre (n_selected, threshold_used, converged)

Corrections v3 :
  - BUG-01  .date() plante si index non-DatetimeIndex → guard hasattr
  - BUG-02  mean_jacc sur tableau vide (1 seule fenêtre) → guard len > 0
  - FIX-01  Validation step_size / train_size / test_size en entrée
  - FIX-02  X_test commenté explicitement (no-leakage intentionnel)
  - FIX-03  converged_ noté comme toujours True (bug fshmm_diag non corrigé)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_selection.saliency_selector import FeatureSaliencySelector
from feature_selection.saliency_stability import (
    compute_selection_stability,
    compute_jaccard_matrix,
    get_stable_features,
)


def _fmt_index(val) -> str:
    """Formate un index pour l'affichage — supporte DatetimeIndex et RangeIndex."""
    if hasattr(val, "date"):
        return str(val.date())
    return str(val)


def run_saliency_walk_forward(
    X: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int,
    selector_params: dict | None = None,
    min_stability_freq: float = 0.75,
    min_stability_rho: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Walk-forward saliency selection.

    Pour chaque fenêtre [train_start : train_end], fitte un FSHMM sur X_train
    uniquement et enregistre le ranking rho. Ne touche PAS à X_test
    (pas de leakage par construction).

    Paramètres
    ----------
    X : pd.DataFrame
        Features pré-calculées (standardisées, clippées).
        Index DatetimeIndex ou RangeIndex.
    train_size : int
        Taille de la fenêtre d'entraînement (en lignes).
    test_size : int
        Taille de la fenêtre de test (en lignes).
    step_size : int
        Pas entre deux fenêtres consécutives. Doit être > 0.
    selector_params : dict | None
        Kwargs passés à FeatureSaliencySelector.
    min_stability_freq : float
        Fréquence de sélection minimale pour consensus.
    min_stability_rho : float
        mean_rho minimal pour consensus.
    verbose : bool

    Retourne
    --------
    dict avec clés :
        rankings            pd.DataFrame  tous rankings empilés
        selected_by_window  pd.DataFrame  résumé par fenêtre
        stability           pd.DataFrame  métriques de stabilité par feature
        jaccard_matrix      pd.DataFrame  similarité Jaccard inter-fenêtres
        consensus_features  list[str]     features stables cross-fenêtres
    """
    # ── Validation des paramètres ────────────────────────────────────
    if step_size <= 0:
        raise ValueError(f"step_size doit être > 0, reçu : {step_size}")
    if train_size <= 0 or test_size <= 0:
        raise ValueError(
            f"train_size et test_size doivent être > 0, "
            f"reçu : train_size={train_size}, test_size={test_size}"
        )
    if train_size + test_size > len(X):
        raise ValueError(
            f"train_size + test_size ({train_size + test_size}) "
            f"> len(X) ({len(X)}) — pas assez de données."
        )

    selector_params = selector_params or {}

    rankings: list[pd.DataFrame] = []
    selected_by_window: list[dict] = []

    start = 0
    window_id = 0

    while start + train_size + test_size <= len(X):
        train_end = start + train_size
        test_end  = train_end + test_size

        X_train = X.iloc[start:train_end].copy()

        # X_test non utilisé dans le fit — conservé uniquement pour les
        # métadonnées de fenêtre (no-leakage intentionnel)
        X_test = X.iloc[train_end:test_end]

        if verbose:
            print(
                f"\n[WF window={window_id}] "
                f"train={_fmt_index(X_train.index[0])}→"
                f"{_fmt_index(X_train.index[-1])}  "
                f"test={_fmt_index(X_test.index[0])}→"
                f"{_fmt_index(X_test.index[-1])}"
            )

        selector = FeatureSaliencySelector(**selector_params)

        try:
            selector.fit(X_train)
            ranking = selector.get_ranking()
            ranking["window"]      = window_id
            ranking["train_start"] = X_train.index[0]
            ranking["train_end"]   = X_train.index[-1]
            ranking["test_start"]  = X_test.index[0]
            ranking["test_end"]    = X_test.index[-1]

            selected   = selector.get_selected_features()
            n_selected = len(selected)
            thresh_used = selector.effective_threshold_

            # NOTE : converged_ est toujours True dans fshmm_diag v2
            # (bug non corrigé) — cette colonne n'est pas encore fiable
            converged = getattr(selector.model_, "converged_", None)

            if verbose:
                print(
                    f"  → {n_selected} features selected "
                    f"(threshold={thresh_used:.3f})"
                )
                if selected:
                    print(
                        f"     {selected[:10]}"
                        f"{'...' if len(selected) > 10 else ''}"
                    )

        except Exception as e:
            if verbose:
                print(f"  ⚠ window {window_id} failed: {e}")
            start += step_size
            window_id += 1
            continue

        rankings.append(ranking)

        selected_by_window.append({
            "window":            window_id,
            "train_start":       X_train.index[0],
            "train_end":         X_train.index[-1],
            "test_start":        X_test.index[0],
            "test_end":          X_test.index[-1],
            "selected_features": selected,
            "n_selected":        n_selected,
            "threshold_used":    thresh_used,
            "converged":         converged,
        })

        start += step_size
        window_id += 1

    if not rankings:
        raise RuntimeError("Aucune fenêtre walk-forward n'a abouti.")

    all_rankings = pd.concat(rankings, ignore_index=True)
    selected_df  = pd.DataFrame(selected_by_window)

    # ── Stabilité ────────────────────────────────────────────────────
    stability = compute_selection_stability(rankings)
    jaccard   = compute_jaccard_matrix(rankings)
    consensus = get_stable_features(
        stability,
        min_frequency=min_stability_freq,
        min_mean_rho=min_stability_rho,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("WALK-FORWARD STABILITY SUMMARY")
        print("=" * 60)
        print(stability.round(4).to_string(index=False))
        print(f"\nConsensus features ({len(consensus)}): {consensus}")

        # Guard : tableau vide si une seule fenêtre
        triu_vals = jaccard.values[np.triu_indices(len(jaccard), k=1)]
        mean_jacc = float(triu_vals.mean()) if len(triu_vals) > 0 else float("nan")
        print(f"Mean inter-window Jaccard: {mean_jacc:.3f}")

    return {
        "rankings":           all_rankings,
        "selected_by_window": selected_df,
        "stability":          stability,
        "jaccard_matrix":     jaccard,
        "consensus_features": consensus,
    }


def get_consensus_features(
    wf_result: dict,
    min_frequency: float = 0.75,
    min_mean_rho: float = 0.5,
) -> list[str]:
    """
    Extraction rapide des features consensus depuis un résultat walk-forward.
    Permet d'ajuster les critères après coup sans refaire le walk-forward.
    """
    stability = wf_result["stability"]
    return get_stable_features(stability, min_frequency, min_mean_rho)
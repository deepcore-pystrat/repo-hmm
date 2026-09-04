"""
saliency_stability.py — Analyse de stabilité temporelle des saliency scores
=============================================================================
Améliorations v2 :
  - compute_selection_stability() : agrégation multi-fenêtres avec
    fréquence de sélection, CV de rho, et score de stabilité composite
  - compute_jaccard_matrix() : similarité Jaccard entre toutes paires de fenêtres
  - plot_stability_heatmap() : visualisation inline (matplotlib)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_selection_stability(
    rankings_by_window: list[pd.DataFrame],
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Agrège les rankings issus de plusieurs fenêtres walk-forward et calcule
    des métriques de stabilité par feature.

    Paramètres
    ----------
    rankings_by_window : list[pd.DataFrame]
        Chaque DataFrame doit avoir les colonnes ['feature', 'rho', 'selected'].
    threshold : float | None
        Si fourni, recalcule 'selected' avec ce seuil fixe au lieu de
        celui stocké dans chaque ranking.

    Retourne
    --------
    pd.DataFrame avec colonnes :
        feature, mean_rho, std_rho, cv_rho, selection_frequency,
        stability_score, n_windows
    """
    required = {"feature", "rho", "selected"}
    for i, r in enumerate(rankings_by_window):
        missing = required - set(r.columns)
        if missing:
            raise ValueError(
                f"rankings_by_window[{i}] manque les colonnes : {missing}"
            )

    if not rankings_by_window:
        raise ValueError("rankings_by_window est vide.")

    all_rankings = pd.concat(rankings_by_window, ignore_index=True)

    stability = (
        all_rankings
        .groupby("feature")
        .agg(
            mean_rho=("rho", "mean"),
            std_rho=("rho", "std"),
            selection_frequency=("selected", "mean"),
            n_windows=("window", "nunique"),
        )
        .reset_index()
        
    )
    stability["std_rho"] = stability["std_rho"].fillna(0.0)


    # Coefficient of variation (robustesse du score)
    stability["cv_rho"] = stability["std_rho"] / (stability["mean_rho"].abs() + 1e-8)

    # Score composite : haute fréquence × haut mean_rho × faible CV
    stability["stability_score"] = (
        stability["selection_frequency"]
        * stability["mean_rho"]
        / (1.0 + stability["cv_rho"])
    )

    return stability.sort_values(
        ["stability_score", "mean_rho"], ascending=False
    ).reset_index(drop=True)

def plot_stability_heatmap(
    jaccard_matrix: pd.DataFrame,
    title: str = "Jaccard Stability — Walk-Forward Windows",
) -> None:
    """
    Affiche la heatmap de la matrice Jaccard entre fenêtres.
    Requiert matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib requis : pip install matplotlib")

    fig, ax = plt.subplots(figsize=(max(6, len(jaccard_matrix)), 
                                    max(5, len(jaccard_matrix) - 1)))
    im = ax.imshow(jaccard_matrix.values, vmin=0, vmax=1, cmap="YlGnBu")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(jaccard_matrix.columns)))
    ax.set_yticks(range(len(jaccard_matrix.index)))
    ax.set_xticklabels(jaccard_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(jaccard_matrix.index)

    for i in range(len(jaccard_matrix)):
        for j in range(len(jaccard_matrix.columns)):
            ax.text(j, i, f"{jaccard_matrix.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if jaccard_matrix.values[i, j] < 0.7 else "white")

    ax.set_title(title)
    plt.tight_layout()
    plt.show()
def compute_jaccard_matrix(
    rankings_by_window: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Calcule la matrice de similarité Jaccard entre toutes les paires de fenêtres.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Retourne une DataFrame carrée (n_windows × n_windows).
    """
    n = len(rankings_by_window)
    sets = []
    for r in rankings_by_window:
        sel = r.loc[r["selected"].astype(bool), "feature"].tolist()

        sets.append(set(sel))

    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            matrix[i, j] = inter / union if union > 0 else 0.0


    labels = [f"w{i}" for i in range(n)]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def get_stable_features(
    stability_df: pd.DataFrame,
    min_frequency: float = 0.75,
    min_mean_rho: float = 0.5,
) -> list[str]:
    """
    Retourne les features stables selon deux critères :
      - sélectionnées dans au moins min_frequency fraction des fenêtres
      - mean_rho ≥ min_mean_rho

    Paramètres typiques recommandés :
      min_frequency=0.75, min_mean_rho=0.5  (conservative)
      min_frequency=0.60, min_mean_rho=0.4  (permissive)
    """
    mask = (
        (stability_df["selection_frequency"] >= min_frequency)
        & (stability_df["mean_rho"] >= min_mean_rho)
    )
    return stability_df.loc[mask, "feature"].tolist()

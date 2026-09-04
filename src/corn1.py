"""Spot HMM to latent market state feed, and futures conditional dynamics validation.

Builds the spot-price HMM used to derive the corn market regime, produces the
train/test state feeds consumed downstream (regime, sub-state, confidence and
quantile diagnostics), and runs the validation battery that checks whether
each regime and sub-state is a genuine, reproducible structure in the futures
market rather than a statistical artefact.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

from models import StudentTHMMConfig, GaussianHMMConfig, create_hmm
import sys

sys.path.insert(0, r"C:/Python/Python311/Lib/site-packages")
from jumpmodels.sparse_jump import SparseJumpModel

# ============================================================
# CONFIG
# ============================================================
from typing import Any, Dict, Tuple
from pathlib import Path

from models.optimal_switching import (
    StochasticProcessConfig,
    fit_stochastic_process,
    enrich_feed_with_stochastic_process,
    stochastic_process_report,
)

# ============================================================
# REGIME 2 — STUDENT-t SUB-HMM À 3 ÉTATS
# Chargement des résultats déjà estimés par regime2_subhmm_3states.py
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBHMM3_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "regime2_subhmm3_convergence"

SUBHMM3_COLUMNS = [
    # Variables futures déjà utilisées par l'ancienne sortie,
    # désormais produites par le pipeline Student-t sous-HMM.
    "fut_return_20",
    "fut_efficiency_20",
    "fut_volatility_20",
    # Nouvelles sorties du Student-t sous-HMM à trois états.
    "regime2_substate_3",
    "p_regime2_substate_0",
    "p_regime2_substate_1",
    "p_regime2_substate_2",
    "regime2_substate_confidence",
    "regime2_substate_changed",
    "regime2_substate_block_id",
    "regime2_substate_age",
]


def _normalize_feed_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise l'index sans changer les colonnes ni l'ordre logique."""

    out = frame.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()

    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    if out.index.has_duplicates:
        duplicates = out.index[out.index.duplicated()].unique()[:20]
        raise ValueError("Dates dupliquées dans le feed : " f"{duplicates.tolist()}")

    return out


def add_regime2_subhmm3_columns(
    feed: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    """
    Ajoute uniquement les nouvelles colonnes du Student-t sous-HMM.

    Garanties
    ----------
    - aucune colonne existante n'est supprimée ;
    - aucune colonne existante n'est écrasée ;
    - aucune propagation temporelle des sous-états ;
    - les colonnes sont jointes uniquement par date exacte ;
    - le GMM historique n'est ni ajusté ni appliqué.
    """

    split = str(split).strip().lower()

    if split not in {"train", "test"}:
        raise ValueError(
            "split doit être 'train' ou 'test'. " f"Valeur reçue : {split!r}"
        )

    source_path = SUBHMM3_OUTPUT_DIR / f"regime2_subhmm3_{split}.csv"

    if not source_path.exists():
        raise FileNotFoundError(
            "Résultat du Student-t sous-HMM introuvable : " f"{source_path}"
        )

    base = _normalize_feed_index(feed)

    if "spot_state" not in base.columns:
        raise ValueError("La colonne spot_state est absente du feed principal.")

    collisions = [column for column in SUBHMM3_COLUMNS if column in base.columns]

    if collisions:
        raise ValueError(
            "Les nouvelles colonnes existent déjà dans le feed. "
            "Aucune colonne existante ne sera écrasée : "
            f"{collisions}"
        )

    subhmm = pd.read_csv(
        source_path,
        index_col=0,
        parse_dates=True,
    )

    subhmm = _normalize_feed_index(subhmm)

    required = [
        "spot_state",
        *SUBHMM3_COLUMNS,
    ]

    missing = [column for column in required if column not in subhmm.columns]

    if missing:
        raise ValueError(
            "Colonnes absentes du résultat Student-t sous-HMM : " f"{missing}"
        )

    common_index = base.index.intersection(subhmm.index)

    if common_index.empty:
        raise ValueError(
            "Aucune date commune entre le feed principal et le "
            f"résultat sous-HMM {split}."
        )

    # Jointure simple sur les dates exactes.
    # Les colonnes existantes du feed principal restent inchangées.
    # Le spot_state enregistré dans le fichier du sous-HMM n'est pas utilisé.
    enriched = base.join(
        subhmm[SUBHMM3_COLUMNS],
        how="left",
        validate="one_to_one",
    )

    integer_columns = [
        "regime2_substate_3",
        "regime2_substate_changed",
        "regime2_substate_block_id",
        "regime2_substate_age",
    ]

    for column in integer_columns:
        enriched[column] = pd.to_numeric(
            enriched[column],
            errors="coerce",
        ).astype("Int64")

    covered = enriched.index.isin(common_index)

    print("\n" + "=" * 90)
    print(f"STUDENT-t SUB-HMM 3 ÉTATS AJOUTÉ — {split.upper()}")
    print("=" * 90)
    print("Source :", source_path)
    print("Dates du feed :", len(enriched))
    print("Dates communes :", len(common_index))
    print(
        pd.crosstab(
            enriched.loc[covered, "spot_state"],
            enriched.loc[covered, "regime2_substate_3"],
            margins=True,
            dropna=False,
        )
    )

    return enriched


SPOT_PATH = r"D:\Downloads\data_spot_corn_brz.xlsx"
FUTURES_PATH = r"D:\Downloads\data_futures_corn (2).xlsx"

NB_STATES = 3
HMM_TYPE = "student"  # "student" or "gaussian"
TRAIN_RATIO = 0.65
SEED = 42
CLIP = 3.5
HORIZON = 10

FSHMM_CANDIDATE_COLS = [
    "zscore_ma60",
    "spread_to_vol_20",
    "vol_chg_20",
    "spread_20",
    "chg_20",
    "ba_vol_gap_20",
    "trend_slope_20",
    "chg_60",
    "abs_dd_60",
    "signchg_20",
    "impact_proxy_20",
    "downside_chg_vol_20",
    "spread_mz_60",
    "trend_slope_60",
    "spread_vol_20",
    "hurst_20",
]
# ============================================================
# EXTRA REGIME STATS — PROJECTED ON FUTURES
# ============================================================
from typing import Tuple

import statsmodels.api as sm


def paper_style_state_characterization(
    *,
    data: pd.DataFrame,
    state_feed: pd.DataFrame,
    sample_name: str,
    price_col: str = "FUTURES_CLOSE",
    state_col: str = "spot_state",
    min_block_len: int = 5,
    hac_maxlags: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Caractérise les états HMM existants selon la logique du paper
    bear / sidewalk / bull.

    Important
    ---------
    - Le HMM n'est pas réentraîné.
    - Le nombre d'états n'est pas modifié.
    - Les rendements sont ceux du futures CORN.
    - Le premier rendement de chaque bloc est exclu pour éviter qu'un
      rendement traversant deux régimes soit attribué au nouveau régime.

    Sorties
    -------
    daily_summary:
        Statistiques quotidiennes conditionnelles à chaque état.

    block_summary:
        Statistiques calculées bloc par bloc, puis résumées par état.

    block_details:
        Une ligne par bloc continu de régime.
    """

    if price_col not in data.columns:
        raise ValueError(f"Colonne prix absente de data : {price_col}")

    if state_col not in state_feed.columns:
        raise ValueError(f"Colonne état absente de state_feed : {state_col}")

    idx = data.index.intersection(state_feed.index).sort_values()

    if len(idx) == 0:
        raise ValueError("Aucune date commune entre data et state_feed.")

    df = pd.DataFrame(index=idx)

    df["price"] = pd.to_numeric(
        data.loc[idx, price_col],
        errors="coerce",
    )

    df["state"] = pd.to_numeric(
        state_feed.loc[idx, state_col],
        errors="coerce",
    )

    df = df.dropna(subset=["price", "state"]).loc[lambda x: x["price"] > 0].sort_index()

    df["state"] = df["state"].astype(int)

    # Bloc continu de même état.
    df["block_id"] = df["state"].ne(df["state"].shift()).cumsum()

    log_price = np.log(df["price"])

    # Rendement quotidien en pourcentage.
    df["return_pct"] = 100.0 * log_price.diff()

    # Le premier rendement d'un bloc traverse la frontière entre deux états.
    # On l'exclut de la caractérisation intra-état.
    first_row_of_block = df["block_id"].ne(df["block_id"].shift())

    df.loc[
        first_row_of_block,
        "return_pct",
    ] = np.nan

    # ============================================================
    # 1. TABLE QUOTIDIENNE PAR ÉTAT
    # ============================================================

    daily_rows = []

    for state, group in df.groupby(
        "state",
        sort=True,
    ):
        returns = group["return_pct"].dropna().astype(float)

        if len(returns) < 3:
            continue

        # Test classique du paper : H0 moyenne = 0.
        t_stat, t_pvalue = stats.ttest_1samp(
            returns,
            popmean=0.0,
            nan_policy="omit",
        )

        # Version robuste à l'autocorrélation / hétéroscédasticité.
        X = np.ones(
            (len(returns), 1),
            dtype=float,
        )

        hac_fit = sm.OLS(
            returns.to_numpy(),
            X,
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": int(hac_maxlags),
            },
        )

        mean_return = float(returns.mean())

        std_return = float(returns.std(ddof=1))

        daily_rows.append(
            {
                "sample": sample_name,
                "state": int(state),
                "n_daily_returns": int(len(returns)),
                "n_blocks": int(group["block_id"].nunique()),
                "mean_return_pct": mean_return,
                "median_return_pct": float(returns.median()),
                "std_return_pct": std_return,
                "skewness": float(
                    stats.skew(
                        returns,
                        bias=False,
                    )
                ),
                "kurtosis_pearson": float(
                    stats.kurtosis(
                        returns,
                        fisher=False,
                        bias=False,
                    )
                ),
                "positive_rate": float((returns > 0).mean()),
                "negative_rate": float((returns < 0).mean()),
                "zero_rate": float((returns == 0).mean()),
                "ttest_stat_mean_zero": float(t_stat),
                "ttest_pvalue_mean_zero": float(t_pvalue),
                "hac_tstat_mean_zero": float(hac_fit.tvalues[0]),
                "hac_pvalue_mean_zero": float(hac_fit.pvalues[0]),
                "mean_significant_5pct_hac": bool(hac_fit.pvalues[0] < 0.05),
                "direction_from_mean": (
                    "positive"
                    if mean_return > 0
                    else "negative" if mean_return < 0 else "zero"
                ),
                "positive_balance_distance": float(abs((returns > 0).mean() - 0.5)),
                "standardized_abs_mean": (
                    float(abs(mean_return) / std_return) if std_return > 0 else np.nan
                ),
            }
        )

    daily_summary = pd.DataFrame(daily_rows).sort_values("state")

    # ============================================================
    # 2. TABLE PAR BLOC
    # ============================================================

    block_rows = []

    for block_id, group in df.groupby(
        "block_id",
        sort=True,
    ):
        state = int(group["state"].iloc[0])

        returns = group["return_pct"].dropna().astype(float)

        if len(group) < min_block_len:
            continue

        if len(returns) < 3:
            continue

        total_return = float(returns.sum())

        path_length = float(returns.abs().sum())

        signs = np.sign(returns.to_numpy())

        signs = signs[signs != 0]

        sign_change_rate = (
            float(np.mean(signs[1:] != signs[:-1])) if len(signs) > 1 else np.nan
        )

        block_rows.append(
            {
                "sample": sample_name,
                "block_id": int(block_id),
                "state": state,
                "start": group.index.min(),
                "end": group.index.max(),
                "length_days": int(len(group)),
                "n_returns": int(len(returns)),
                "block_return_pct": total_return,
                "block_abs_return_pct": float(abs(total_return)),
                "block_path_pct": path_length,
                "block_efficiency": (
                    float(abs(total_return) / path_length)
                    if path_length > 0
                    else np.nan
                ),
                "block_positive_rate": float((returns > 0).mean()),
                "block_negative_rate": float((returns < 0).mean()),
                "block_sign_change_rate": (sign_change_rate),
                "block_return_autocorr_1": (
                    float(returns.autocorr(lag=1)) if len(returns) >= 5 else np.nan
                ),
                "block_volatility_pct": float(returns.std(ddof=1)),
            }
        )

    block_details = pd.DataFrame(block_rows)

    if block_details.empty:
        block_summary = pd.DataFrame()
    else:
        block_summary = (
            block_details.groupby(
                ["sample", "state"],
                as_index=False,
            )
            .agg(
                n_blocks=(
                    "block_id",
                    "count",
                ),
                median_block_length=(
                    "length_days",
                    "median",
                ),
                median_block_return_pct=(
                    "block_return_pct",
                    "median",
                ),
                mean_block_return_pct=(
                    "block_return_pct",
                    "mean",
                ),
                share_positive_blocks=(
                    "block_return_pct",
                    lambda x: float((x > 0).mean()),
                ),
                share_negative_blocks=(
                    "block_return_pct",
                    lambda x: float((x < 0).mean()),
                ),
                median_block_efficiency=(
                    "block_efficiency",
                    "median",
                ),
                median_block_sign_change_rate=(
                    "block_sign_change_rate",
                    "median",
                ),
                median_block_autocorr_1=(
                    "block_return_autocorr_1",
                    "median",
                ),
                median_block_volatility_pct=(
                    "block_volatility_pct",
                    "median",
                ),
            )
            .sort_values("state")
        )

    # ============================================================
    # 3. DIAGNOSTIC OBJECTIF DES CANDIDATS
    # ============================================================

    if not daily_summary.empty:
        sidewalk_by_abs_mean = int(
            daily_summary.loc[
                daily_summary["mean_return_pct"].abs().idxmin(),
                "state",
            ]
        )

        sidewalk_by_balance = int(
            daily_summary.loc[
                daily_summary["positive_balance_distance"].idxmin(),
                "state",
            ]
        )

        bull_candidate = int(
            daily_summary.loc[
                daily_summary["mean_return_pct"].idxmax(),
                "state",
            ]
        )

        bear_candidate = int(
            daily_summary.loc[
                daily_summary["mean_return_pct"].idxmin(),
                "state",
            ]
        )

        print("\n" + "=" * 100)
        print(f"PAPER-STYLE STATE CHARACTERIZATION — {sample_name}")
        print("=" * 100)

        print(daily_summary.round(6).to_string(index=False))

        print("\nCandidats descriptifs, pas encore des labels définitifs :")
        print(
            "  Bear par moyenne minimale      :",
            bear_candidate,
        )
        print(
            "  Bull par moyenne maximale      :",
            bull_candidate,
        )
        print(
            "  Sidewalk par |moyenne| minimale:",
            sidewalk_by_abs_mean,
        )
        print(
            "  Sidewalk par équilibre 50/50   :",
            sidewalk_by_balance,
        )

        if sidewalk_by_abs_mean == sidewalk_by_balance:
            print("  Accord sidewalk               : OUI")
        else:
            print("  Accord sidewalk               : NON")

    if not block_summary.empty:
        print("\n" + "=" * 100)
        print(f"BLOCK-LEVEL VALIDATION — {sample_name}")
        print("=" * 100)
        print(block_summary.round(6).to_string(index=False))

    return (
        daily_summary,
        block_summary,
        block_details,
    )


def add_hmm_diagnostics_to_feed(feed: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le state feed HMM avec des diagnostics probabilistes
    directement exploitables par le repo de backtest.

    Colonnes ajoutées :
        - hmm_confidence
        - hmm_entropy
        - hmm_max_prob
        - hmm_p_margin
        - hmm_prob_jump
        - hmm_confidence_change
        - hmm_state_changed
        - hmm_prev_state
        - hmm_transition
        - hmm_block_id
        - hmm_regime_age_days
    """

    import numpy as np
    import pandas as pd

    out = feed.copy()
    out = out.sort_index()

    if "spot_state" not in out.columns:
        raise ValueError("Missing column: spot_state")

    p_cols = [c for c in out.columns if c.startswith("p_state_")]

    p_cols = sorted(p_cols, key=lambda x: int(x.replace("p_state_", "")))

    if len(p_cols) == 0:
        raise ValueError(
            "No p_state_* columns found. "
            "Cannot compute HMM entropy / probability diagnostics."
        )

    p = out[p_cols].astype(float).clip(lower=1e-12, upper=1.0)

    # Probabilité maximale / confiance
    out["hmm_max_prob"] = p.max(axis=1)

    if "spot_state_confidence" in out.columns:
        out["hmm_confidence"] = out["spot_state_confidence"].astype(float)
    else:
        out["hmm_confidence"] = out["hmm_max_prob"]

    # Entropie HMM
    # Plus c'est élevé, plus le HMM hésite entre plusieurs états.
    out["hmm_entropy"] = -(p * np.log(p)).sum(axis=1)

    # Marge entre la proba du meilleur état et celle du deuxième meilleur
    sorted_p = np.sort(p.values, axis=1)

    out["hmm_p_margin"] = sorted_p[:, -1] - sorted_p[:, -2]

    # Jump de probabilité : instabilité entre t-1 et t
    out["hmm_prob_jump"] = p.diff().abs().max(axis=1).fillna(0.0)

    out["hmm_confidence_change"] = out["hmm_confidence"].diff().abs().fillna(0.0)

    # Transitions de régime
    out["hmm_prev_state"] = out["spot_state"].shift(1)

    out["hmm_state_changed"] = (out["spot_state"] != out["spot_state"].shift(1)).astype(
        int
    )

    out["hmm_transition"] = (
        out["hmm_prev_state"].astype("Int64").astype(str)
        + "->"
        + out["spot_state"].astype("Int64").astype(str)
    )

    out.loc[out["hmm_prev_state"].isna(), "hmm_transition"] = "START"

    # Bloc continu de même régime
    out["hmm_block_id"] = (out["spot_state"] != out["spot_state"].shift(1)).cumsum()

    out["hmm_regime_age_days"] = out.groupby("hmm_block_id").cumcount() + 1

    return out


def futures_tail_stats_by_state(data, state_feed, horizons=(1, 5, 10, 20)):
    rows = []

    for h in horizons:
        m = build_futures_metrics(data.loc[state_feed.index].copy(), horizon=h)

        m["state"] = state_feed["spot_state"].astype(int)

        col = f"retF_fwd_{h}"

        for s, g in m.groupby("state"):
            r = g[col].dropna()

            if len(r) < 20:
                continue

            var_5 = r.quantile(0.05)
            cvar_5 = r[r <= var_5].mean()

            rows.append(
                {
                    "state": int(s),
                    "horizon": h,
                    "n": len(r),
                    "mean": r.mean(),
                    "median": r.median(),
                    "std": r.std(),
                    "skew": r.skew(),
                    "kurtosis": r.kurtosis(),
                    "hit_rate_pos": (r > 0).mean(),
                    "VaR_5": var_5,
                    "CVaR_5": cvar_5,
                    "q25": r.quantile(0.25),
                    "q75": r.quantile(0.75),
                }
            )

    out = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("FUTURES FORWARD TAIL STATS BY PROJECTED SPOT STATE")
    print("=" * 80)
    print(out.round(6).to_string(index=False))

    return out


def futures_transition_stats(data, state_feed, horizon=10):
    df = data.loc[state_feed.index].copy()
    df["state"] = state_feed["spot_state"].astype(int)

    logp = np.log(df["FUTURES_CLOSE"].astype(float))
    df[f"retF_fwd_{horizon}"] = logp.shift(-horizon) - logp

    df["next_state"] = df["state"].shift(-1)
    df["transition"] = (
        df["state"].astype(str) + " -> " + df["next_state"].astype("Int64").astype(str)
    )

    clean = df.dropna(subset=[f"retF_fwd_{horizon}", "next_state"])

    out = (
        clean.groupby("transition")[f"retF_fwd_{horizon}"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            hit_rate_pos=lambda x: (x > 0).mean(),
            q05=lambda x: x.quantile(0.05),
            q95=lambda x: x.quantile(0.95),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    print("\n" + "=" * 80)
    print(f"FUTURES FORWARD RETURN BY SPOT REGIME TRANSITION — {horizon}D")
    print("=" * 80)
    print(out.round(6).to_string(index=False))

    return out


def regime_survival_stats(state_feed):
    feed = state_feed.copy()
    feed["state"] = feed["spot_state"].astype(int)
    feed["block_id"] = (feed["state"] != feed["state"].shift()).cumsum()

    blocks = (
        feed.groupby("block_id")
        .agg(
            state=("state", "first"),
            start=("state", lambda x: x.index[0]),
            end=("state", lambda x: x.index[-1]),
            length=("state", "size"),
            confidence_mean=("spot_state_confidence", "mean"),
        )
        .reset_index(drop=True)
    )

    summary = (
        blocks.groupby("state")["length"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            q75=lambda x: x.quantile(0.75),
            q90=lambda x: x.quantile(0.90),
            max="max",
        )
        .reset_index()
    )

    print("\n" + "=" * 80)
    print("REGIME SURVIVAL / DURATION STATS")
    print("=" * 80)
    print(summary.round(3).to_string(index=False))

    return blocks, summary


def plot_futures_volatility_by_spot_regime(
    data, state_feed, window=20, save_path="03_futures_vol_by_regime.png"
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = data.loc[state_feed.index].copy()
    df["state"] = state_feed["spot_state"]

    logp = np.log(df["FUTURES_CLOSE"].astype(float))
    df["fut_vol_20"] = logp.diff().rolling(window).std() * (252**0.5)

    plot_df = df.dropna(subset=["fut_vol_20", "state"])

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.boxplot(column="fut_vol_20", by="state", ax=ax, showfliers=False)

    ax.set_title("Futures volatility by spot regime", fontweight="bold")
    ax.set_xlabel("Spot HMM regime")
    ax.set_ylabel("Annualized futures volatility")
    ax.grid(alpha=0.3)
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.show()


# ============================================================
# DATA
# ============================================================
def extract_blocks_from_feed(state_feed, min_len=5):
    df = state_feed.copy()
    df["block_id"] = (df["spot_state"] != df["spot_state"].shift()).cumsum()

    rows = []

    for block_id, g in df.groupby("block_id"):
        if len(g) < min_len:
            continue

        rows.append(
            {
                "block_id": int(block_id),
                "state": int(g["spot_state"].iloc[0]),
                "start": g.index[0],
                "end": g.index[-1],
                "length": len(g),
                "confidence_mean": g["spot_state_confidence"].mean(),
            }
        )

    return pd.DataFrame(rows)


def prepare_futures():
    """
    Futures nettoyés sur leur propre calendrier.
    Pas de reindex sur le calendrier spot.
    """
    fut = load_futures().set_index("Date")
    fut = fut.sort_index()

    # load_futures() fait déjà ffill(limit=3) sur les trous internes
    fut = fut[fut["FUTURES_CLOSE"].notna() & (fut["FUTURES_CLOSE"] > 0)]

    return fut


def load_spot():
    df = pd.read_excel(SPOT_PATH, header=2)
    df.columns = df.columns.str.strip()

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["BID"] = pd.to_numeric(df["BID"], errors="coerce")
    df["OFFER"] = pd.to_numeric(df["OFFER"], errors="coerce")

    df["BID"] = df["BID"].ffill(limit=3)
    df["OFFER"] = df["OFFER"].ffill(limit=3)
    df["MID"] = (df["BID"] + df["OFFER"]) / 2.0
    df["BA_SPREAD"] = df["OFFER"] - df["BID"]

    df = (
        df[["DATE", "FUTURES", "BID", "OFFER", "MID", "BA_SPREAD"]]
        .dropna(subset=["DATE", "FUTURES", "MID", "BID", "OFFER"])
        .sort_values("DATE")
        .rename(columns={"DATE": "Date"})
        .groupby("Date", as_index=False)
        .last()
    )

    return df


def load_futures():
    df = pd.read_excel(FUTURES_PATH)
    df.columns = df.columns.str.strip()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df = df.sort_values("Date")

    df["Volume"] = df["Volume"].ffill(limit=3)

    df = (
        df[["Date", "Close", "Volume"]]
        .dropna(subset=["Date", "Close", "Volume"])
        .groupby("Date", as_index=False)
        .last()
        .rename(columns={"Close": "FUTURES_CLOSE"})
    )

    df = df[df["FUTURES_CLOSE"] > 0].copy()
    return df


# ============================================================
# FEATURES
# ============================================================


def variance_ratio(x, lag=5):
    x = pd.Series(x).dropna()

    if len(x) < lag + 2:
        return np.nan

    r1 = x.diff().dropna()
    rq = x.diff(lag).dropna()

    v1 = r1.var()
    vq = rq.var()

    if v1 <= 1e-12:
        return np.nan

    return vq / (lag * v1)


def _mad_zscore(x, window):
    med = x.rolling(window).median()

    mad = (x - med).abs().rolling(window).median()

    # garde-fou robuste
    mad = mad.clip(lower=0.1)

    z = (x - med) / mad

    # winsorisation
    z = z.clip(-10, 10)

    return z


def build_features(df, eps=1e-8):

    out = pd.DataFrame(index=df.index)

    # ============================================================
    # RAW SERIES
    # ============================================================

    mid = pd.to_numeric(df["MID"], errors="coerce")
    bid = pd.to_numeric(df["BID"], errors="coerce")
    ask = pd.to_numeric(df["OFFER"], errors="coerce")
    spread = pd.to_numeric(df["BA_SPREAD"], errors="coerce")

    # ============================================================
    # LAGGED SERIES
    # ============================================================

    mid_lag = mid.shift(1)
    bid_lag = bid.shift(1)
    ask_lag = ask.shift(1)
    spread_lag = spread.shift(1)

    d_mid = mid.diff().shift(1)
    d_bid = bid.diff().shift(1)
    d_ask = ask.diff().shift(1)
    d_spread = spread.diff().shift(1)

    # ============================================================
    # BASIS / PRICE DYNAMICS
    # ============================================================

    out["chg_1"] = d_mid

    out["chg_5"] = mid.shift(1) - mid.shift(6)

    out["chg_20"] = mid.shift(1) - mid.shift(21)

    out["chg_60"] = mid.shift(1) - mid.shift(61)

    out["abs_chg_20"] = out["chg_20"].abs()

    # ============================================================
    # VOLATILITY / STRESS
    # ============================================================

    out["vol_chg_20"] = d_mid.rolling(20).std()

    out["vol_chg_60"] = d_mid.rolling(60).std()

    d_down = d_mid.where(d_mid < 0, 0.0)
    d_up = d_mid.where(d_mid > 0, 0.0)

    out["downside_chg_vol_20"] = d_down.rolling(20).std()

    out["upside_chg_vol_20"] = d_up.rolling(20).std()

    out["vol_skew_diff_20"] = out["downside_chg_vol_20"] - out["upside_chg_vol_20"]

    out["chg_kurt_20"] = d_mid.rolling(20).kurt()

    # ============================================================
    # PATH STRUCTURE / REGIME
    # ============================================================

    path_20 = d_mid.abs().rolling(20).sum()

    path_60 = d_mid.abs().rolling(60).sum()

    out["efficiency_20"] = out["chg_20"].abs() / (path_20 + eps)

    out["efficiency_60"] = out["chg_60"].abs() / (path_60 + eps)

    out["signchg_20"] = d_mid.rolling(20).apply(
        lambda x: (
            np.mean(np.sign(pd.Series(x).dropna()).diff().abs() > 0)
            if len(pd.Series(x).dropna()) > 3
            else np.nan
        ),
        raw=False,
    )

    out["autocorr_1_20"] = d_mid.rolling(20).apply(
        lambda x: (
            pd.Series(x).dropna().autocorr(lag=1)
            if len(pd.Series(x).dropna()) > 5
            else np.nan
        ),
        raw=False,
    )

    out["vr_5_20"] = d_mid.rolling(20).apply(
        lambda x: variance_ratio(x, lag=5),
        raw=False,
    )

    # ============================================================
    # RANGE / DRAWDOWN
    # ============================================================

    roll_max_20 = mid_lag.rolling(20).max()
    roll_min_20 = mid_lag.rolling(20).min()

    roll_max_60 = mid_lag.rolling(60).max()

    out["range_abs_20"] = roll_max_20 - roll_min_20

    out["abs_dd_60"] = (mid_lag - roll_max_60).abs()

    # ============================================================
    # SPREAD / LIQUIDITY
    # ============================================================

    out["spread"] = spread_lag

    out["spread_20"] = spread_lag.rolling(20).mean()

    out["spread_vol_20"] = spread_lag.rolling(20).std()

    out["spread_chg_5"] = spread_lag - spread.shift(6)

    out["spread_mz_60"] = _mad_zscore(
        spread_lag,
        60,
    )

    out["spread_stress_20"] = (out["spread_mz_60"] > 1.5).rolling(20).sum()

    # relative spread
    rel_spread = spread_lag / (mid_lag.abs() + eps)

    out["rel_spread"] = rel_spread

    out["rel_spread_vol_20"] = rel_spread.rolling(20).std()

    # ============================================================
    # BID / ASK ASYMMETRY
    # ============================================================

    out["bid_vol_20"] = d_bid.rolling(20).std()

    out["ask_vol_20"] = d_ask.rolling(20).std()

    out["ba_vol_gap_20"] = out["ask_vol_20"] - out["bid_vol_20"]

    out["quote_slope"] = d_ask - d_bid

    # ============================================================
    # LIQUIDITY / IMPACT
    # ============================================================

    out["impact_proxy_20"] = d_mid.abs().rolling(20).mean() / (
        spread_lag.rolling(20).mean() + eps
    )

    out["spread_to_vol_20"] = spread_lag.rolling(20).mean() / (out["vol_chg_20"] + eps)

    # ============================================================
    # CLEANING
    # ============================================================

    # ============================================================
    # TREND STRUCTURE
    # ============================================================

    def _rolling_slope(x):
        x = np.asarray(x)

        if np.isnan(x).any():
            return np.nan

        t = np.arange(len(x))

        return np.polyfit(t, x, 1)[0]

    out["trend_slope_20"] = mid_lag.rolling(20).apply(_rolling_slope, raw=False)

    out["trend_slope_60"] = mid_lag.rolling(60).apply(_rolling_slope, raw=False)
    ma60 = mid_lag.rolling(60).mean()

    std60 = mid_lag.rolling(60).std()

    out["zscore_ma60"] = (mid_lag - ma60) / (std60 + eps)
    out["hurst_20"] = d_mid.rolling(20).apply(_hurst_rs, raw=False)

    out = out.replace([np.inf, -np.inf], np.nan)

    out = out.dropna(axis=1, how="all")

    return out


# ==========================================================
# HMM
# ============================================================
def _hurst_rs(x):

    x = pd.Series(x).dropna()

    if len(x) < 10:
        return np.nan

    mean = x.mean()

    deviations = (x - mean).cumsum()

    R = deviations.max() - deviations.min()

    S = x.std()

    if S < 1e-12:
        return np.nan

    return np.log(R / S) / np.log(len(x))


def fit_hmm(X, K):
    if HMM_TYPE == "student":
        cfg = StudentTHMMConfig(
            K=K,
            n_iter=150,
            seed=SEED,
            cov_type="diag",
            min_var=1e-3,
            sticky=0.5,
            estimate_nu=True,
        )
    else:
        cfg = GaussianHMMConfig(
            K=K,
            n_iter=150,
            seed=SEED,
            cov_type="diag",
        )

    model = create_hmm(cfg)
    result = model.fit(X)

    return model, result


def summarize_spot_states(feat, alpha, feature_cols):
    states = np.argmax(alpha, axis=1)

    df = feat.iloc[: len(states)].copy()
    df["state"] = states

    summary = df.groupby("state")[feature_cols].agg(["mean", "median", "std", "count"])

    print("\n" + "=" * 80)
    print("SPOT LATENT STATE CHARACTERIZATION")
    print("=" * 80)
    print(summary.round(4).to_string())

    return summary


def transition_diagnostics(A):
    print("\n" + "=" * 80)
    print("HMM TRANSITION DIAGNOSTICS")
    print("=" * 80)

    print("Transition matrix:")
    print(np.round(A, 4))

    expected_duration = 1.0 / (1.0 - np.diag(A))

    for k, d in enumerate(expected_duration):
        print(f"State {k}: expected duration ≈ {d:.2f} days")


# ============================================================
# FUTURES CONDITIONAL DYNAMICS
# ============================================================


def forward_sum(x, horizon, func=None):
    out = pd.Series(0.0, index=x.index)

    for i in range(1, horizon + 1):
        v = x.shift(-i)

        if func is not None:
            v = func(v)

        out = out + v

    return out


def build_futures_metrics(data, horizon=10, eps=1e-8):
    df = data.copy()

    price = df["FUTURES_CLOSE"].astype(float)
    logp = np.log(price)
    r = logp.diff()

    df["rF_t"] = r

    df[f"retF_fwd_{horizon}"] = forward_sum(r, horizon)
    df[f"absF_fwd_{horizon}"] = forward_sum(r, horizon, func=lambda x: np.abs(x))
    df[f"rvF_fwd_{horizon}"] = forward_sum(r, horizon, func=lambda x: x**2)

    rows = []

    for i in range(len(df)):
        if i + horizon >= len(df):
            rows.append(
                {
                    f"rangeF_fwd_{horizon}": np.nan,
                    f"effF_fwd_{horizon}": np.nan,
                    f"signchgF_fwd_{horizon}": np.nan,
                }
            )
            continue

        p_win = logp.iloc[i : i + horizon + 1]
        r_win = r.iloc[i + 1 : i + horizon + 1].dropna()

        if len(r_win) < 3:
            rows.append(
                {
                    f"rangeF_fwd_{horizon}": np.nan,
                    f"effF_fwd_{horizon}": np.nan,
                    f"signchgF_fwd_{horizon}": np.nan,
                }
            )
            continue

        total_ret = p_win.iloc[-1] - p_win.iloc[0]
        path_len = r_win.abs().sum()

        signs = np.sign(r_win.values)
        signs = signs[signs != 0]

        sign_change = np.mean(signs[1:] != signs[:-1]) if len(signs) > 1 else np.nan

        rows.append(
            {
                f"rangeF_fwd_{horizon}": p_win.max() - p_win.min(),
                f"effF_fwd_{horizon}": abs(total_ret) / (path_len + eps),
                f"signchgF_fwd_{horizon}": sign_change,
            }
        )

    df = df.join(pd.DataFrame(rows, index=df.index))
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def conditional_futures_dynamics(data, alpha, index, horizon=10):
    metrics = build_futures_metrics(data.loc[index].copy(), horizon=horizon)

    states = np.argmax(alpha, axis=1)

    metrics["spot_state"] = pd.Series(states, index=index)

    cols = [
        f"retF_fwd_{horizon}",
        f"absF_fwd_{horizon}",
        f"rvF_fwd_{horizon}",
        f"rangeF_fwd_{horizon}",
        f"effF_fwd_{horizon}",
        f"signchgF_fwd_{horizon}",
    ]

    clean = metrics.dropna(subset=["spot_state"] + cols)

    summary = clean.groupby("spot_state")[cols].agg(["count", "mean", "median", "std"])

    print("\n" + "=" * 80)
    print("FUTURES CONDITIONAL DYNAMICS BY SPOT STATE")
    print("=" * 80)
    print(summary.round(6).to_string())

    print("\n" + "=" * 80)
    print("KRUSKAL TESTS ACROSS SPOT STATES")
    print("=" * 80)

    test_rows = []

    for col in cols:
        groups = [
            clean.loc[clean["spot_state"] == s, col].dropna().values
            for s in sorted(clean["spot_state"].unique())
        ]

        groups = [g for g in groups if len(g) > 5]

        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
        else:
            h, p = np.nan, np.nan

        test_rows.append(
            {
                "metric": col,
                "kruskal_H": h,
                "p_value": p,
            }
        )

    tests = pd.DataFrame(test_rows)
    print(tests.round(6).to_string(index=False))

    return clean, summary, tests


# ============================================================
# EXPORTABLE DATA FEED
# ============================================================


def build_state_feed(index, alpha):
    states = np.argmax(alpha, axis=1)

    feed = pd.DataFrame(index=index)

    feed["spot_state"] = states
    feed["spot_state_confidence"] = alpha.max(axis=1)

    for k in range(alpha.shape[1]):
        feed[f"p_state_{k}"] = alpha[:, k]

    return feed


# ============================================================
# PLOTS
# ============================================================


from collections import deque


def add_causal_regime_quantiles_to_test_feed(
    *,
    train_feed: pd.DataFrame,
    test_feed: pd.DataFrame,
    features: list[str],
    regime_col: str = "spot_state",
    quantiles: tuple[float, ...] = (
        0.75,
        0.80,
        0.85,
        0.90,
    ),
    min_history: int = 60,
    max_history: int | None = None,
) -> pd.DataFrame:
    """
    Ajoute des quantiles empiriques causaux conditionnels au régime.

    Pour chaque date test t et chaque variable X :

        1. Le régime courant S_t est observé.
        2. Le seuil est calculé avec les valeurs historiques
           de X appartenant au même régime.
        3. La valeur X_t n'est pas utilisée dans son propre seuil.
        4. X_t est ajoutée à l'historique seulement après
           le calcul des seuils de la date t.

    L'historique initial est constitué des observations du train.

    Paramètres
    ----------
    train_feed:
        Feed HMM du train contenant les régimes et les variables.

    test_feed:
        Feed HMM test à enrichir.

    features:
        Variables pour lesquelles calculer les quantiles.

    regime_col:
        Colonne contenant le régime HMM.

    quantiles:
        Niveaux de quantile à retourner.

    min_history:
        Nombre minimal d'observations historiques du même régime
        avant de produire un seuil.

    max_history:
        None :
            historique expanding : tout le train et tout le passé test.

        Entier :
            historique rolling utilisant uniquement les dernières
            observations du même régime.

    Returns
    -------
    pd.DataFrame
        Copie enrichie de test_feed.
    """

    if min_history < 1:
        raise ValueError("min_history doit être supérieur ou égal à 1.")

    if max_history is not None:
        if max_history < min_history:
            raise ValueError(
                "max_history doit être supérieur ou égal " "à min_history."
            )

    quantiles = tuple(sorted(set(float(q) for q in quantiles)))

    if any(q <= 0.0 or q >= 1.0 for q in quantiles):
        raise ValueError("Les quantiles doivent appartenir à ]0, 1[.")

    train = train_feed.copy().sort_index()
    test = test_feed.copy().sort_index()

    required_columns = [
        regime_col,
        *features,
    ]

    missing_train = [col for col in required_columns if col not in train.columns]

    missing_test = [col for col in required_columns if col not in test.columns]

    if missing_train:
        raise ValueError(f"Colonnes absentes de train_feed : {missing_train}")

    if missing_test:
        raise ValueError(f"Colonnes absentes de test_feed : {missing_test}")

    if train.index.has_duplicates:
        raise ValueError("train_feed contient des dates dupliquées.")

    if test.index.has_duplicates:
        raise ValueError("test_feed contient des dates dupliquées.")

    overlapping_dates = train.index.intersection(test.index)

    if len(overlapping_dates) > 0:
        raise ValueError(
            "train_feed et test_feed contiennent des dates "
            "communes. Le découpage train/test doit être strict."
        )

    train_states = train[regime_col].dropna().astype(int).unique().tolist()

    test_states = test[regime_col].dropna().astype(int).unique().tolist()

    all_states = sorted(set(train_states).union(test_states))

    # history[feature][state] contient uniquement les valeurs
    # disponibles avant la date test traitée.
    history: dict[str, dict[int, deque]] = {}

    for feature in features:
        history[feature] = {}

        for state in all_states:
            train_values = pd.to_numeric(
                train.loc[
                    train[regime_col].astype("Int64") == state,
                    feature,
                ],
                errors="coerce",
            ).dropna()

            values = train_values.to_numpy(dtype=float)

            if max_history is not None:
                values = values[-max_history:]

            history[feature][state] = deque(
                values.tolist(),
                maxlen=max_history,
            )

    # Création préalable des colonnes
    for feature in features:
        test[f"{feature}_regime_history_n"] = pd.Series(
            0,
            index=test.index,
            dtype="Int64",
        )

        test[f"{feature}_regime_percentile"] = np.nan

        test[f"{feature}_regime_quantile_ready"] = 0

        for q in quantiles:
            q_label = int(round(q * 100))

            test[f"{feature}_regime_q{q_label:02d}"] = np.nan

    # ============================================================
    # BOUCLE CHRONOLOGIQUE STRICTEMENT CAUSALE
    # ============================================================

    for date in test.index:
        regime_value = test.at[
            date,
            regime_col,
        ]

        if pd.isna(regime_value):
            continue

        state = int(regime_value)

        for feature in features:
            current_value = pd.to_numeric(
                pd.Series([test.at[date, feature]]),
                errors="coerce",
            ).iloc[0]

            state_history = history[feature].setdefault(
                state,
                deque(maxlen=max_history),
            )

            history_n = len(state_history)

            history_n_col = f"{feature}_regime_history_n"

            ready_col = f"{feature}_regime_quantile_ready"

            percentile_col = f"{feature}_regime_percentile"

            test.at[
                date,
                history_n_col,
            ] = history_n

            # Le seuil n'est disponible que si l'historique
            # du régime est suffisamment long.
            if history_n >= min_history:
                historical_values = np.fromiter(
                    state_history,
                    dtype=float,
                    count=history_n,
                )

                quantile_values = np.quantile(
                    historical_values,
                    quantiles,
                    method="linear",
                )

                for q, threshold in zip(
                    quantiles,
                    quantile_values,
                ):
                    q_label = int(round(q * 100))

                    test.at[
                        date,
                        f"{feature}_regime_q{q_label:02d}",
                    ] = float(threshold)

                test.at[
                    date,
                    ready_col,
                ] = 1

                # Rang percentile causal de la valeur actuelle.
                # La correction 0.5 sur les égalités produit
                # un rang médian en cas d'ex aequo.
                if pd.notna(current_value):
                    lower_count = np.sum(historical_values < float(current_value))

                    equal_count = np.sum(historical_values == float(current_value))

                    percentile = (lower_count + 0.5 * equal_count) / history_n

                    test.at[
                        date,
                        percentile_col,
                    ] = float(percentile)

            # IMPORTANT :
            # on ajoute X_t seulement après avoir calculé
            # le seuil et le percentile de la date t.
            if pd.notna(current_value):
                state_history.append(float(current_value))

    # Types propres pour le CSV
    for feature in features:
        test[f"{feature}_regime_quantile_ready"] = test[
            f"{feature}_regime_quantile_ready"
        ].astype("Int64")

    return test


def check_label_stability(
    feat_train,
    feat_test,
    train_alpha,
    test_alpha,
    feature_cols,
    train_index,
    test_index,
    max_dist_threshold=2.0,
):
    from itertools import permutations

    def state_means(feat, alpha, index):
        states = np.argmax(alpha, axis=1)
        df = feat.loc[index].copy()
        df["state"] = states
        return df.groupby("state")[feature_cols].mean()

    train_means = state_means(feat_train, train_alpha, train_index)
    test_means = state_means(feat_test, test_alpha, test_index)

    print("\n" + "=" * 80)
    print("LABEL SWITCHING CHECK — moyennes par état")
    print("=" * 80)
    print("\nTRAIN:")
    print(train_means.round(4))
    print("\nTEST:")
    print(test_means.round(4))

    common_train = set(train_means.index)
    common_test = set(test_means.index)
    absent_in_test = common_train - common_test

    if absent_in_test:
        print(
            f"\n⚠️  États absents en test : {sorted(absent_in_test)} "
            f"— ignorés dans la recherche de permutation"
        )

    test_states_present = sorted(common_test)
    K_test = len(test_states_present)

    # ── Retour sûr si rien à comparer ────────────────────────────────────────
    if K_test == 0:
        print("Aucun état présent en test — pas de remap possible.")
        identity = {}
        return identity, False, False  # ← toujours un tuple

    # ── Permutation sur les états TEST présents uniquement ───────────────────
    # On restreint aussi les candidats TRAIN aux états présents en train
    train_candidates = sorted(common_train)

    best_perm = test_states_present  # identité par défaut
    best_dist = np.inf

    # permutations(train_candidates, K_test) : chaque état train utilisé au plus une fois
    for perm in permutations(train_candidates, K_test):
        dist = sum(
            np.linalg.norm(
                train_means.loc[train_s].values - test_means.loc[test_s].values
            )
            for test_s, train_s in zip(test_states_present, perm)
            if train_s in train_means.index
        )
        if dist < best_dist:
            best_dist = dist
            best_perm = list(perm)

    remap = {test_s: train_s for test_s, train_s in zip(test_states_present, best_perm)}

    switched = any(remap[k] != k for k in remap)

    # ── Vérification de cohérence géométrique ────────────────────────────────
    per_state_dists = {
        test_s: np.linalg.norm(
            train_means.loc[train_s].values - test_means.loc[test_s].values
        )
        for test_s, train_s in remap.items()
        if train_s in train_means.index
    }

    coherent = all(d < max_dist_threshold for d in per_state_dists.values())

    print(f"\nRemapping : {remap}")
    print(
        f"Distance L2 par état : { {k: round(v, 3) for k, v in per_state_dists.items()} }"
    )

    if switched and coherent:
        print("Label switching : ⚠️  OUI — remapping appliqué (cohérent)")
    elif switched and not coherent:
        print(
            "Label switching : 🚫  Remapping incohérent (distances trop grandes) "
            "— identité conservée"
        )
        remap = {k: k for k in test_states_present}
        switched = False
    else:
        print("Label switching : ✅ NON — états stables")

    return remap, switched, coherent  # ← toujours un tuple
    # On cherche la permutation sur les états TEST présents vers les états TRAIN
    # qui mini


# ============================================================
# MAIN
# ============================================================


def conditional_futures_ac1_by_state(data, state_feed, horizons=(5, 10, 20, 40, 60)):
    df = data.loc[state_feed.index].copy()
    df["state"] = state_feed["spot_state"]
    df["block_id"] = (df["state"] != df["state"].shift()).cumsum()

    logp = np.log(df["FUTURES_CLOSE"].astype(float))
    df["rF"] = logp.diff()

    rows = []

    for state in sorted(df["state"].dropna().unique()):
        state_df = df[df["state"] == state].copy()

        for k in horizons:
            pairs = []

            for _, block in state_df.groupby("block_id"):
                r = block["rF"].dropna()

                if len(r) < 2 * k:
                    continue

                past_k = r.rolling(k).sum()

                future_k = pd.Series(0.0, index=r.index)
                for i in range(1, k + 1):
                    future_k = future_k + r.shift(-i)

                tmp = pd.DataFrame(
                    {
                        "past_k": past_k,
                        "future_k": future_k,
                    }
                ).dropna()

                pairs.append(tmp)

            if len(pairs) == 0:
                ac1 = np.nan
                pval = np.nan
                nobs = 0
            else:
                tmp_all = pd.concat(pairs)

                nobs = len(tmp_all)

                if nobs < 20:
                    ac1 = np.nan
                    pval = np.nan
                else:
                    ac1, pval = stats.pearsonr(tmp_all["past_k"], tmp_all["future_k"])

            rows.append(
                {
                    "state": int(state),
                    "horizon_k": k,
                    "AC1_k": ac1,
                    "p_value": pval,
                    "nobs": nobs,
                }
            )

    return pd.DataFrame(rows)


def build_block_regime_stats(data, state_feed, horizon=10, min_block_len=8):
    """
    Une ligne = un bloc continu d'un même état.
    Pas de concaténation entre blocs.
    """
    df = data.loc[state_feed.index].copy()
    df["spot_state"] = state_feed["spot_state"].astype(int)
    df["block_id"] = (df["spot_state"] != df["spot_state"].shift()).cumsum()

    logp = np.log(df["FUTURES_CLOSE"].astype(float))
    df["rF"] = logp.diff()

    rows = []

    for block_id, g in df.groupby("block_id"):
        state = int(g["spot_state"].iloc[0])
        r = g["rF"].dropna()
        p = logp.loc[g.index].dropna()

        if len(g) < min_block_len or len(r) < 4:
            continue

        total_ret = p.iloc[-1] - p.iloc[0]
        path = r.abs().sum()

        signs = np.sign(r.values)
        signs = signs[signs != 0]

        signchg = np.mean(signs[1:] != signs[:-1]) if len(signs) > 1 else np.nan

        ac1_1d = r.autocorr(lag=1) if len(r) >= 5 else np.nan

        # block-level mean reversion: past k return vs future k return INSIDE block
        ac1_k = np.nan
        if len(r) >= 2 * horizon:
            past_k = r.rolling(horizon).sum()

            future_k = pd.Series(0.0, index=r.index)
            for i in range(1, horizon + 1):
                future_k += r.shift(-i)

            tmp = pd.DataFrame(
                {
                    "past_k": past_k,
                    "future_k": future_k,
                }
            ).dropna()

            if len(tmp) >= 5:
                ac1_k = tmp["past_k"].corr(tmp["future_k"])

        rows.append(
            {
                "block_id": int(block_id),
                "state": state,
                "start": g.index[0],
                "end": g.index[-1],
                "length": len(g),
                "block_return": total_ret,
                "block_abs_return": abs(total_ret),
                "block_path": path,
                "block_efficiency": abs(total_ret) / (path + 1e-8),
                "block_rv": np.sum(r**2),
                "block_vol": r.std(),
                "block_range": p.max() - p.min(),
                "block_signchg": signchg,
                "block_ac1_1d": ac1_1d,
                f"block_ac1_{horizon}d": ac1_k,
            }
        )

    return pd.DataFrame(rows)


def block_level_tests(blocks, horizon=10, sample_name="Test"):
    from scipy import stats
    import numpy as np
    import pandas as pd

    metrics = [
        "length",
        "block_return",
        "block_abs_return",
        "block_rv",
        "block_vol",
        "block_range",
        "block_efficiency",
        "block_signchg",
        "block_ac1_1d",
        f"block_ac1_{horizon}d",
        "block_skew",
        "block_kurt",
    ]

    metrics = [m for m in metrics if m in blocks.columns]
    states = sorted(blocks["state"].dropna().unique())

    rows = []

    for metric in metrics:
        groups = [
            blocks.loc[blocks["state"] == s, metric].dropna().values for s in states
        ]
        groups = [g for g in groups if len(g) >= 3]

        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
        else:
            h, p = np.nan, np.nan

        row = {
            "sample": sample_name,
            "metric": metric,
            "kruskal_H": h,
            "p_value": p,
        }

        if len(states) == 2:
            x = blocks.loc[blocks["state"] == states[0], metric].dropna()
            y = blocks.loc[blocks["state"] == states[1], metric].dropna()

            if len(x) >= 3 and len(y) >= 3:
                U, p_u = stats.mannwhitneyu(x, y, alternative="two-sided")
                row["mannwhitney_U"] = U
                row["mannwhitney_p"] = p_u
                row[f"median_state_{int(states[0])}"] = x.median()
                row[f"median_state_{int(states[1])}"] = y.median()
                row["median_diff_second_minus_first"] = y.median() - x.median()

        rows.append(row)

    return pd.DataFrame(rows).round(6)


def plot_futures_with_spot_hmm_blocks(futures, state_feed, title):
    import matplotlib.pyplot as plt
    import pandas as pd

    fut = futures.copy()

    fut["Close"] = pd.to_numeric(fut["Close"], errors="coerce")
    fut = fut.dropna(subset=["Close"])
    fut = fut[fut["Close"] > 0]

    state_feed = state_feed.copy()
    state_feed.index = pd.to_datetime(state_feed.index)

    start_hmm = state_feed.index.min()
    end_hmm = state_feed.index.max()

    # garder seulement la période des régimes HMM
    fut = fut.loc[start_hmm:end_hmm]

    blocks = state_feed.copy()
    blocks["block_id"] = (blocks["spot_state"] != blocks["spot_state"].shift()).cumsum()

    colors = {
        0: "red",
        1: "green",
        2: "blue",
    }

    plt.figure(figsize=(16, 6))
    ax = plt.gca()

    # COURBE FUTURES
    ax.plot(
        fut.index, fut["Close"], color="black", linewidth=1.2, label="Futures observé"
    )

    used = set()

    for _, block in blocks.groupby("block_id"):
        state = int(block["spot_state"].iloc[0])
        start = block.index.min()
        end = block.index.max()

        label = f"State {state}" if state not in used else None
        used.add(state)

        ax.axvspan(start, end, color=colors.get(state, "gray"), alpha=0.18, label=label)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Futures Close")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def feature_sanity_report(X_df):
    report = pd.DataFrame(
        {
            "mean": X_df.mean(),
            "std": X_df.std(),
            "min": X_df.min(),
            "p01": X_df.quantile(0.01),
            "p50": X_df.quantile(0.50),
            "p99": X_df.quantile(0.99),
            "max": X_df.max(),
            "missing": X_df.isna().mean(),
        }
    )

    print("\nFEATURE SANITY REPORT")
    print(report.round(6).to_string())

    bad = report[
        (report["std"] <= 1e-10)
        | (report["max"].abs() > 1e6)
        | (report["min"].abs() > 1e6)
    ]

    if len(bad) > 0:
        raise ValueError("Unstable features detected:\n" + bad.to_string())


# ============================================================
# VOLATILITY CLUSTERING DURING STRESS REGIMES
# ============================================================


def regime_duration_table(blocks):
    return (
        blocks.groupby("state")["length"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .round(3)
    )


def empirical_survival_by_state(blocks):
    """
    S(d) = P(duration >= d | state)
    """
    rows = []

    for s, g in blocks.groupby("state"):
        lengths = g["length"].dropna().astype(int)
        for d in range(1, lengths.max() + 1):
            rows.append(
                {
                    "state": s,
                    "duration_day": d,
                    "survival_prob": (lengths >= d).mean(),
                    "n_blocks": len(lengths),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 6) TRANSITION EVENT STUDY
# 0->1 et 1->0, sans mélanger les blocs
# ============================================================


def transition_event_study(
    data, state_feed, horizons=(1, 5, 10, 20), price_col="FUTURES_CLOSE"
):
    df = data.loc[state_feed.index].copy()
    df["state"] = state_feed["spot_state"].astype(int)
    df["prev_state"] = df["state"].shift(1)

    logp = np.log(df[price_col].astype(float))

    events = df[df["state"] != df["prev_state"]].copy()
    events = events.dropna(subset=["prev_state"])

    rows = []

    for date, row in events.iterrows():
        i = logp.index.get_loc(date)

        for h in horizons:
            if i + h >= len(logp):
                continue

            rows.append(
                {
                    "date": date,
                    "from_state": int(row["prev_state"]),
                    "to_state": int(row["state"]),
                    "transition": f"{int(row['prev_state'])}->{int(row['state'])}",
                    "horizon": h,
                    "fwd_return": logp.iloc[i + h] - logp.iloc[i],
                }
            )

    res = pd.DataFrame(rows)

    summary = (
        res.groupby(["transition", "horizon"])["fwd_return"]
        .agg(["count", "mean", "median", "std"])
        .round(6)
    )

    return res, summary


# ============================================================
# 7) CRASH / EXTREME RISK PAR BLOC
# ============================================================


def block_tail_risk(blocks):
    """
    Compare la probabilité qu'un bloc appartienne aux pires épisodes.
    """
    x = blocks["block_return"].dropna()
    q05 = x.quantile(0.05)
    q10 = x.quantile(0.10)

    out = blocks.copy()
    out["crash_5pct_block"] = out["block_return"] <= q05
    out["crash_10pct_block"] = out["block_return"] <= q10

    return (
        out.groupby("state")[["crash_5pct_block", "crash_10pct_block"]].mean().round(4)
    )


# ============================================================
# INTERNAL REGIME MIXTURE TEST
# Est-ce qu'un régime HMM est homogène ou mélange de sous-régimes ?
# ============================================================


# ============================================================
# ECONOMIC INTERPRETATION TEST — REGIME ENTRY EVENT STUDY
# ============================================================


def regime_entry_event_study(
    feat,
    state_feed,
    features=None,
    pre_window=10,
    post_window=10,
    min_events=3,
):
    """
    Test économique d'entrée en régime.

    Objectif :
    identifier ce qui change autour de l'entrée dans chaque régime :
    - spread / liquidité
    - impact de marché
    - volatilité baissière
    - drawdown
    - tendance

    Ce test ne sert pas à dire bullish/bearish.
    Il sert à donner un label économique :
    - Liquidity deterioration
    - Market impact regime
    - Downside volatility shock
    - Drawdown / dislocation regime
    - Price discovery / transition regime
    """

    if features is None:
        features = [
            "spread_20",
            "spread_vol_20",
            "impact_proxy_20",
            "downside_chg_vol_20",
            "vol_chg_20",
            "abs_dd_60",
            "trend_slope_20",
            "trend_slope_60",
            "spread_to_vol_20",
        ]

    features = [c for c in features if c in feat.columns]

    idx = feat.index.intersection(state_feed.index)

    df = feat.loc[idx, features].copy()
    df["state"] = state_feed.loc[idx, "spot_state"].astype(int)

    df = df.sort_index()

    # Entrée en régime : state_t != state_{t-1}
    df["prev_state"] = df["state"].shift(1)
    entry_dates = df.index[(df["state"] != df["prev_state"]) & df["prev_state"].notna()]

    rows = []

    for entry_date in entry_dates:
        pos = df.index.get_loc(entry_date)

        if pos < pre_window:
            continue

        if pos + post_window >= len(df):
            continue

        target_state = int(df.loc[entry_date, "state"])
        previous_state = int(df.loc[entry_date, "prev_state"])

        window = df.iloc[pos - pre_window : pos + post_window + 1].copy()
        window["tau"] = np.arange(-pre_window, post_window + 1)
        window["entry_date"] = entry_date
        window["target_state"] = target_state
        window["previous_state"] = previous_state

        rows.append(window.reset_index().rename(columns={"index": "Date"}))

    if len(rows) == 0:
        print("Pas assez d'entrées de régime pour faire l'event-study.")
        return None, None, None

    events = pd.concat(rows, ignore_index=True)

    # ============================================================
    # 1) Profil moyen autour de l'entrée
    # ============================================================

    event_profile = (
        events.groupby(["target_state", "tau"])[features].median().reset_index()
    )

    # ============================================================
    # 2) Test pré-régime vs début de régime
    # ============================================================

    test_rows = []

    for state in sorted(events["target_state"].unique()):
        sub = events[events["target_state"] == state]

        n_events = sub["entry_date"].nunique()

        if n_events < min_events:
            continue

        for col in features:
            pre_values = sub.loc[sub["tau"].between(-pre_window, -1), col].dropna()

            early_values = sub.loc[
                sub["tau"].between(0, min(5, post_window)), col
            ].dropna()

            post_values = sub.loc[sub["tau"].between(0, post_window), col].dropna()

            if len(pre_values) < 10 or len(early_values) < 10:
                continue

            # Test robuste non-paramétrique
            h_early, p_early = stats.kruskal(pre_values, early_values)

            h_post, p_post = stats.kruskal(pre_values, post_values)

            pre_med = pre_values.median()
            early_med = early_values.median()
            post_med = post_values.median()

            eps = 1e-8

            test_rows.append(
                {
                    "target_state": state,
                    "n_events": n_events,
                    "feature": col,
                    "pre_median": pre_med,
                    "early_median": early_med,
                    "post_median": post_med,
                    "early_minus_pre": early_med - pre_med,
                    "post_minus_pre": post_med - pre_med,
                    "early_ratio": early_med / (abs(pre_med) + eps),
                    "post_ratio": post_med / (abs(pre_med) + eps),
                    "kruskal_H_early": h_early,
                    "p_value_early": p_early,
                    "kruskal_H_post": h_post,
                    "p_value_post": p_post,
                }
            )

    tests = pd.DataFrame(test_rows)

    if tests.empty:
        print("Pas assez de données pour les tests pré/post.")
        return events, event_profile, tests

    # ============================================================
    # 3) Score économique par régime
    # ============================================================

    economic_map = {
        "spread_20": "Liquidity cost / spread widening",
        "spread_vol_20": "Liquidity instability",
        "impact_proxy_20": "Market impact / execution pressure",
        "downside_chg_vol_20": "Downside volatility shock",
        "vol_chg_20": "General volatility shock",
        "abs_dd_60": "Drawdown / market dislocation",
        "trend_slope_20": "Short-term price discovery",
        "trend_slope_60": "Medium-term price discovery",
        "spread_to_vol_20": "Relative liquidity condition",
    }

    label_rows = []

    for state, g in tests.groupby("target_state"):

        g_sig = g[g["p_value_early"] < 0.05].copy()

        if g_sig.empty:
            g_sig = g.copy()

        # On prend la variable qui change le plus au début du régime
        g_sig["abs_effect"] = g_sig["early_minus_pre"].abs()

        dominant = g_sig.sort_values("abs_effect", ascending=False).iloc[0]

        feature = dominant["feature"]
        mechanism = economic_map.get(feature, "Unknown mechanism")

        if feature in ["spread_20", "spread_vol_20", "spread_to_vol_20"]:
            label = "Liquidity deterioration / liquidity regime"

        elif feature == "impact_proxy_20":
            label = "Market impact regime"

        elif feature in ["downside_chg_vol_20", "vol_chg_20"]:
            label = "Volatility shock regime"

        elif feature == "abs_dd_60":
            label = "Market dislocation / drawdown regime"

        elif feature in ["trend_slope_20", "trend_slope_60"]:
            label = "Price discovery / repricing regime"

        else:
            label = "Mixed economic regime"

        label_rows.append(
            {
                "state": state,
                "dominant_feature": feature,
                "economic_mechanism": mechanism,
                "suggested_label": label,
                "early_minus_pre": dominant["early_minus_pre"],
                "p_value_early": dominant["p_value_early"],
                "n_events": dominant["n_events"],
            }
        )

    labels = pd.DataFrame(label_rows)

    print("\n" + "=" * 100)
    print("REGIME ENTRY EVENT STUDY — ECONOMIC MECHANISM")
    print("=" * 100)

    print("\nTESTS PRE-ENTRY VS EARLY REGIME")
    print(
        tests.sort_values(["target_state", "p_value_early"])
        .round(6)
        .to_string(index=False)
    )

    print("\nECONOMIC LABEL SUGGESTION")
    print(labels.round(6).to_string(index=False))

    return events, event_profile, labels


def add_global_futures_abs_dd_60(
    feed: pd.DataFrame,
    futures: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Ajoute le drawdown absolu futures sur 60 séances à toutes
    les dates du feed, indépendamment du régime HMM.

    fut_abs_dd_60 = distance relative entre le prix actuel
    et le maximum observé sur les 60 dernières séances.

    La fenêtre est uniquement passée et courante :
    aucune information future n'est utilisée.
    """

    out = _normalize_feed_index(feed)

    fut = futures.copy()

    fut.index = pd.to_datetime(
        fut.index,
        errors="coerce",
    )

    fut = fut.loc[fut.index.notna()].sort_index()

    if fut.index.tz is not None:
        fut.index = fut.index.tz_localize(None)

    if "FUTURES_CLOSE" not in fut.columns:
        raise KeyError(
            "La colonne FUTURES_CLOSE est absente des futures. "
            f"Colonnes disponibles : {fut.columns.tolist()}"
        )

    close = pd.to_numeric(
        fut["FUTURES_CLOSE"],
        errors="coerce",
    )

    close = close.where(close > 0)

    rolling_peak_60 = close.rolling(
        window=window,
        min_periods=window,
    ).max()

    futures_abs_dd_60 = 1.0 - close / rolling_peak_60

    # Protection numérique.
    futures_abs_dd_60 = futures_abs_dd_60.clip(lower=0.0).rename("fut_abs_dd_60")

    # Jointure par date exacte, sans propagation de régime.
    out["fut_abs_dd_60"] = futures_abs_dd_60.reindex(out.index)

    return out


def analyze_state1_hmm_certainty(
    *,
    data: pd.DataFrame,
    state_feed: pd.DataFrame,
    sample_name: str,
    state_value: int = 1,
    price_col: str = "FUTURES_CLOSE",
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Diagnostic structurel du régime HMM 1.

    Objectif :
        Vérifier si la certitude du HMM permet de distinguer
        des observations State 1 stables/persistantes de zones
        transitoires/incertaines.

    IMPORTANT :
        - aucun seuil n'est choisi pour maximiser un backtest ;
        - les groupes sont définis par terciles internes à l'échantillon ;
        - les mêmes statistiques sont produites sur TRAIN et TEST ;
        - aucune stratégie de trading n'est utilisée.

    Analyse :
        1. p_state_1
        2. hmm_confidence
        3. hmm_p_margin
        4. hmm_entropy
        5. hmm_prob_jump

    Pour chaque bucket :
        - nombre d'observations
        - nombre de blocs
        - âge médian du régime
        - probabilité de quitter State 1 dans 1/3/5/10 jours
        - rendement futur moyen / médian
        - hit rate positif
        - volatilité des forward returns
    """

    required_feed_cols = [
        "spot_state",
        "p_state_1",
        "hmm_confidence",
        "hmm_entropy",
        "hmm_p_margin",
        "hmm_prob_jump",
        "hmm_block_id",
        "hmm_regime_age_days",
    ]

    missing = [col for col in required_feed_cols if col not in state_feed.columns]

    if missing:
        raise ValueError(f"Colonnes HMM manquantes : {missing}")

    if price_col not in data.columns:
        raise ValueError(f"Colonne prix absente : {price_col}")

    idx = data.index.intersection(state_feed.index).sort_values()

    df = pd.DataFrame(index=idx)

    df["price"] = pd.to_numeric(
        data.loc[idx, price_col],
        errors="coerce",
    )

    for col in required_feed_cols:
        df[col] = state_feed.loc[idx, col]

    df = df.dropna(subset=["price", "spot_state"]).sort_index()

    df["spot_state"] = pd.to_numeric(
        df["spot_state"],
        errors="coerce",
    ).astype("Int64")

    # ============================================================
    # FORWARD RETURNS
    # ============================================================

    log_price = np.log(df["price"].astype(float))

    for h in horizons:
        df[f"ret_fwd_{h}"] = log_price.shift(-h) - log_price

        # Sortie du State 1 à horizon h :
        # vrai si au moins une observation future dans les h jours
        # n'appartient plus au State 1.
        future_states = pd.concat(
            [df["spot_state"].shift(-i) for i in range(1, h + 1)],
            axis=1,
        )

        df[f"exit_within_{h}d"] = future_states.ne(state_value).any(axis=1)

    # ============================================================
    # UNIQUEMENT STATE 1
    # ============================================================

    s1 = df[df["spot_state"] == state_value].copy()

    if s1.empty:
        raise ValueError(f"Aucune observation pour State {state_value}.")

    # ============================================================
    # VARIABLES DE CERTITUDE
    # ============================================================

    certainty_variables = {
        # Plus haut = plus certain
        "p_state_1": "higher_is_more_certain",
        # Plus haut = plus certain
        "hmm_confidence": "higher_is_more_certain",
        # Plus haut = meilleure séparation entre le 1er et 2e état
        "hmm_p_margin": "higher_is_more_certain",
        # Plus haut = plus incertain
        "hmm_entropy": "higher_is_less_certain",
        # Plus haut = changement brutal de probabilité
        "hmm_prob_jump": "higher_is_less_stable",
    }

    summary_rows = []

    bucket_details = []

    for variable, interpretation in certainty_variables.items():

        values = pd.to_numeric(
            s1[variable],
            errors="coerce",
        )

        valid = values.notna()

        tmp = s1.loc[valid].copy()

        if len(tmp) < 15:
            continue

        # --------------------------------------------------------
        # TERCILES
        #
        # Pas de seuil 0.8 / 0.9 choisi après observation.
        # On regarde simplement low / middle / high.
        # --------------------------------------------------------

        try:
            tmp["bucket"] = pd.qcut(
                pd.to_numeric(
                    tmp[variable],
                    errors="coerce",
                ),
                q=3,
                labels=[
                    "LOW",
                    "MID",
                    "HIGH",
                ],
                duplicates="drop",
            )

        except ValueError:
            continue

        for bucket, g in tmp.groupby(
            "bucket",
            observed=True,
        ):

            row = {
                "sample": sample_name,
                "state": state_value,
                "variable": variable,
                "interpretation": interpretation,
                "bucket": str(bucket),
                "n_obs": len(g),
                "n_blocks": int(g["hmm_block_id"].nunique()),
                "variable_mean": float(
                    pd.to_numeric(
                        g[variable],
                        errors="coerce",
                    ).mean()
                ),
                "variable_median": float(
                    pd.to_numeric(
                        g[variable],
                        errors="coerce",
                    ).median()
                ),
                "regime_age_mean": float(
                    pd.to_numeric(
                        g["hmm_regime_age_days"],
                        errors="coerce",
                    ).mean()
                ),
                "regime_age_median": float(
                    pd.to_numeric(
                        g["hmm_regime_age_days"],
                        errors="coerce",
                    ).median()
                ),
            }

            for h in horizons:

                ret = pd.to_numeric(
                    g[f"ret_fwd_{h}"],
                    errors="coerce",
                ).dropna()

                exit_col = f"exit_within_{h}d"

                exits = g[exit_col].dropna().astype(bool)

                row[f"exit_rate_{h}d"] = (
                    float(exits.mean()) if len(exits) > 0 else np.nan
                )

                row[f"ret_mean_{h}d"] = float(ret.mean()) if len(ret) > 0 else np.nan

                row[f"ret_median_{h}d"] = (
                    float(ret.median()) if len(ret) > 0 else np.nan
                )

                row[f"ret_std_{h}d"] = (
                    float(ret.std(ddof=1)) if len(ret) > 1 else np.nan
                )

                row[f"positive_rate_{h}d"] = (
                    float((ret > 0).mean()) if len(ret) > 0 else np.nan
                )

            summary_rows.append(row)

        bucket_details.append(
            tmp[
                [
                    "spot_state",
                    "hmm_block_id",
                    "hmm_regime_age_days",
                    variable,
                    "bucket",
                    *[f"ret_fwd_{h}" for h in horizons],
                    *[f"exit_within_{h}d" for h in horizons],
                ]
            ].assign(
                sample=sample_name,
                variable_name=variable,
            )
        )

    summary = pd.DataFrame(summary_rows)

    details = (
        pd.concat(
            bucket_details,
            axis=0,
            ignore_index=False,
        )
        if bucket_details
        else pd.DataFrame()
    )

    # ============================================================
    # PRINT
    # ============================================================

    print("\n" + "=" * 120)

    print(f"STATE 1 — HMM CERTAINTY / " f"PERSISTENCE ANALYSIS — {sample_name}")

    print("=" * 120)

    if summary.empty:

        print("Aucun résultat exploitable.")

    else:

        display_cols = [
            "variable",
            "bucket",
            "n_obs",
            "n_blocks",
            "variable_median",
            "regime_age_median",
            "exit_rate_1d",
            "exit_rate_3d",
            "exit_rate_5d",
            "exit_rate_10d",
            "ret_mean_1d",
            "ret_mean_3d",
            "ret_mean_5d",
            "ret_mean_10d",
        ]

        display_cols = [col for col in display_cols if col in summary.columns]

        print(summary[display_cols].round(5).to_string(index=False))

    return summary, details


def analyze_regime2_substate_stability(
    *,
    state_feed: pd.DataFrame,
    sample_name: str,
    horizons=(1, 3, 5, 10, 20),
):
    """
    Diagnostic TRAIN / TEST des sous-états du régime 2.

    Objectif :
        Déterminer si la confiance du sous-HMM permet d'identifier
        des périodes particulièrement persistantes.

    IMPORTANT :
        Ce test n'utilise PAS la stratégie.
        Il étudie uniquement la structure du sous-HMM.

    Pour chaque observation appartenant au régime 2 :
        - sous-état courant
        - probabilité du sous-état courant
        - confiance max
        - marge entre les deux meilleures probabilités
        - entropie
        - jump des probabilités
        - âge du sous-état
        - probabilité de rester dans le même sous-état à h jours
        - probabilité d'avoir quitté le sous-état à h jours
    """

    df = state_feed.copy().sort_index()

    required = [
        "spot_state",
        "regime2_substate_3",
        "p_regime2_substate_0",
        "p_regime2_substate_1",
        "p_regime2_substate_2",
        "regime2_substate_confidence",
        "regime2_substate_age",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    # ============================================================
    # GARDER UNIQUEMENT LE RÉGIME PRINCIPAL 2
    # ============================================================

    df = df[
        pd.to_numeric(
            df["spot_state"],
            errors="coerce",
        )
        == 2
    ].copy()

    df["substate"] = pd.to_numeric(
        df["regime2_substate_3"],
        errors="coerce",
    )

    df = df.dropna(subset=["substate"])

    df["substate"] = df["substate"].astype(int)
    # Garder uniquement les 3 sous-états valides du Student-t sub-HMM.
    # -1 signifie : aucun sous-état valide / hors couverture.
    df = df[df["substate"].isin([0, 1, 2])].copy()
    p_cols = [
        "p_regime2_substate_0",
        "p_regime2_substate_1",
        "p_regime2_substate_2",
    ]

    p = df[p_cols].apply(pd.to_numeric, errors="coerce").clip(lower=1e-12, upper=1.0)

    # ============================================================
    # DIAGNOSTICS PROBABILISTES DU SOUS-HMM
    # ============================================================

    df["substate_max_prob"] = p.max(axis=1)

    df["substate_entropy"] = -(p * np.log(p)).sum(axis=1)

    sorted_p = np.sort(
        p.to_numpy(dtype=float),
        axis=1,
    )

    df["substate_p_margin"] = sorted_p[:, -1] - sorted_p[:, -2]

    df["substate_prob_jump"] = p.diff().abs().max(axis=1).fillna(0.0)

    # Probabilité attribuée au sous-état effectivement sélectionné
    current_prob = []

    for idx, row in df.iterrows():

        s = int(row["substate"])

        col = f"p_regime2_substate_{s}"

        current_prob.append(float(row[col]) if pd.notna(row[col]) else np.nan)

    df["current_substate_prob"] = current_prob

    # ============================================================
    # PERSISTANCE FUTURE
    # ============================================================

    # ATTENTION :
    # shift(-h) est utilisé uniquement pour le diagnostic,
    # jamais comme feature de trading.

    for h in horizons:

        future_substate = df["substate"].shift(-h)

        # On exige également que l'observation future appartienne
        # encore au régime principal 2.
        future_main_state = (
            pd.to_numeric(
                state_feed["spot_state"],
                errors="coerce",
            )
            .reindex(df.index)
            .shift(-h)
        )

        valid_future = future_substate.notna() & future_main_state.notna()

        same = (future_substate == df["substate"]) & (future_main_state == 2)

        df[f"same_substate_{h}d"] = np.where(
            valid_future,
            same.astype(float),
            np.nan,
        )

        df[f"exit_substate_{h}d"] = np.where(
            valid_future,
            1.0 - same.astype(float),
            np.nan,
        )

    # ============================================================
    # BUCKETS
    # ============================================================

    def safe_qcut(series):

        valid = series.dropna()

        if valid.nunique() < 4:
            return pd.Series(
                np.nan,
                index=series.index,
                dtype="object",
            )

        try:
            return pd.qcut(
                series,
                q=4,
                labels=[
                    "Q1_low",
                    "Q2",
                    "Q3",
                    "Q4_high",
                ],
                duplicates="drop",
            )

        except ValueError:
            return pd.Series(
                np.nan,
                index=series.index,
                dtype="object",
            )

    df["confidence_bucket"] = safe_qcut(df["current_substate_prob"])

    df["entropy_bucket"] = safe_qcut(df["substate_entropy"])

    df["margin_bucket"] = safe_qcut(df["substate_p_margin"])

    df["jump_bucket"] = safe_qcut(df["substate_prob_jump"])

    # ============================================================
    # RÉSUMÉ PAR SOUS-ÉTAT
    # ============================================================

    summary_rows = []

    for substate in sorted(df["substate"].unique()):

        g = df[df["substate"] == substate]

        base = {
            "sample": sample_name,
            "substate": int(substate),
            "n": int(len(g)),
            "mean_confidence": g["current_substate_prob"].mean(),
            "median_confidence": g["current_substate_prob"].median(),
            "mean_entropy": g["substate_entropy"].mean(),
            "median_entropy": g["substate_entropy"].median(),
            "mean_margin": g["substate_p_margin"].mean(),
            "median_margin": g["substate_p_margin"].median(),
            "mean_prob_jump": g["substate_prob_jump"].mean(),
            "median_prob_jump": g["substate_prob_jump"].median(),
            "median_age": pd.to_numeric(
                g["regime2_substate_age"],
                errors="coerce",
            ).median(),
        }

        for h in horizons:

            base[f"stay_rate_{h}d"] = g[f"same_substate_{h}d"].mean()

            base[f"exit_rate_{h}d"] = g[f"exit_substate_{h}d"].mean()

        summary_rows.append(base)

    summary = pd.DataFrame(summary_rows)

    # ============================================================
    # CONDITIONNEMENT PAR CONFIANCE / ENTROPIE / MARGE / JUMP
    # ============================================================

    conditional_rows = []

    diagnostics = [
        "confidence_bucket",
        "entropy_bucket",
        "margin_bucket",
        "jump_bucket",
    ]

    for substate in sorted(df["substate"].unique()):

        sub = df[df["substate"] == substate]

        for diagnostic in diagnostics:

            for bucket, g in sub.groupby(
                diagnostic,
                observed=True,
            ):

                if len(g) < 10:
                    continue

                row = {
                    "sample": sample_name,
                    "substate": int(substate),
                    "diagnostic": diagnostic,
                    "bucket": str(bucket),
                    "n": int(len(g)),
                }

                for h in horizons:

                    row[f"stay_rate_{h}d"] = g[f"same_substate_{h}d"].mean()

                    row[f"exit_rate_{h}d"] = g[f"exit_substate_{h}d"].mean()

                conditional_rows.append(row)

    conditional = pd.DataFrame(conditional_rows)

    # ============================================================
    # PRINT
    # ============================================================

    print("\n" + "=" * 110)
    print(f"REGIME 2 SUB-HMM STABILITY — {sample_name}")
    print("=" * 110)

    print("\n--- GLOBAL BY SUBSTATE ---")

    if not summary.empty:
        print(summary.round(4).to_string(index=False))

    print("\n--- CONDITIONAL STABILITY ---")

    if not conditional.empty:
        print(conditional.round(4).to_string(index=False))

    return summary, conditional, df


def analyze_regime2_substate0_economic_validation(
    *,
    data: pd.DataFrame,
    state_feed: pd.DataFrame,
    sample_name: str,
    price_col: str = "FUTURES_CLOSE",
    horizons=(1, 3, 5, 10),
):
    """
    Validation économique du couple :

        régime principal = 2
        substate = 0

    Objectif :
        vérifier si la certitude du sous-HMM est associée
        à un comportement futures différent/stable.

    IMPORTANT :
        - aucun filtre de trading ici ;
        - aucune position de stratégie utilisée ;
        - aucune optimisation de seuil ;
        - les quartiles servent uniquement au diagnostic TRAIN/TEST.
    """

    required = [
        "spot_state",
        "regime2_substate_3",
        "p_regime2_substate_0",
        "p_regime2_substate_1",
        "p_regime2_substate_2",
        "regime2_substate_confidence",
    ]

    missing = [c for c in required if c not in state_feed.columns]

    if missing:
        raise ValueError(f"Colonnes sous-HMM manquantes : {missing}")

    if price_col not in data.columns:
        raise ValueError(f"Colonne prix absente : {price_col}")

    # ============================================================
    # PRIX FUTURES SUR SON CALENDRIER COMPLET
    # ============================================================

    market = data.copy().sort_index()

    price = pd.to_numeric(
        market[price_col],
        errors="coerce",
    )

    market = market.loc[price.notna() & (price > 0)].copy()

    log_price = np.log(
        pd.to_numeric(
            market[price_col],
            errors="coerce",
        )
    )

    daily_ret = log_price.diff()

    # Forward returns calculés AVANT de réduire aux dates du HMM.
    for h in horizons:

        market[f"ret_fwd_{h}d"] = log_price.shift(-h) - log_price

        # Volatilité réalisée future sur les h prochaines observations.
        future_returns = pd.concat(
            [daily_ret.shift(-i) for i in range(1, h + 1)],
            axis=1,
        )

        market[f"future_vol_{h}d"] = future_returns.std(
            axis=1,
            ddof=1,
        )

    # ============================================================
    # ALIGNEMENT EXACT AVEC LE FEED HMM
    # ============================================================

    idx = market.index.intersection(state_feed.index).sort_values()

    df = pd.DataFrame(index=idx)

    for col in required:
        df[col] = state_feed.loc[
            idx,
            col,
        ]

    for h in horizons:

        df[f"ret_fwd_{h}d"] = market.loc[
            idx,
            f"ret_fwd_{h}d",
        ]

        df[f"future_vol_{h}d"] = market.loc[
            idx,
            f"future_vol_{h}d",
        ]

    # ============================================================
    # GARDER UNIQUEMENT REGIME 2 / SUBSTATE 0
    # ============================================================

    df["spot_state"] = pd.to_numeric(
        df["spot_state"],
        errors="coerce",
    )

    df["substate"] = pd.to_numeric(
        df["regime2_substate_3"],
        errors="coerce",
    )

    s0 = df[(df["spot_state"] == 2) & (df["substate"] == 0)].copy()

    if len(s0) < 20:
        raise ValueError(f"Trop peu d'observations substate 0 : {len(s0)}")

    # ============================================================
    # DIAGNOSTICS DU SOUS-HMM
    # ============================================================

    p_cols = [
        "p_regime2_substate_0",
        "p_regime2_substate_1",
        "p_regime2_substate_2",
    ]

    p = (
        s0[p_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .clip(
            lower=1e-12,
            upper=1.0,
        )
    )

    s0["p_substate0"] = p["p_regime2_substate_0"]

    s0["sub_confidence"] = pd.to_numeric(
        s0["regime2_substate_confidence"],
        errors="coerce",
    )

    s0["sub_entropy"] = -(p * np.log(p)).sum(axis=1)

    sorted_p = np.sort(
        p.to_numpy(dtype=float),
        axis=1,
    )

    s0["sub_margin"] = sorted_p[:, -1] - sorted_p[:, -2]

    # ============================================================
    # QUARTILES DESCRIPTIFS
    # ============================================================

    def _quartiles(series):

        out = pd.Series(
            np.nan,
            index=series.index,
            dtype="object",
        )

        valid = series.dropna()

        if valid.nunique() < 4:
            return out

        try:
            out.loc[valid.index] = pd.qcut(
                valid,
                q=4,
                labels=[
                    "Q1_low",
                    "Q2",
                    "Q3",
                    "Q4_high",
                ],
                duplicates="drop",
            ).astype(str)

        except ValueError:
            pass

        return out

    s0["confidence_bucket"] = _quartiles(s0["sub_confidence"])

    s0["entropy_bucket"] = _quartiles(s0["sub_entropy"])

    s0["p0_bucket"] = _quartiles(s0["p_substate0"])

    s0["margin_bucket"] = _quartiles(s0["sub_margin"])

    # ============================================================
    # RÉSUMÉS
    # ============================================================

    rows = []

    diagnostics = [
        "confidence_bucket",
        "entropy_bucket",
        "p0_bucket",
        "margin_bucket",
    ]

    for diagnostic in diagnostics:

        for bucket, g in s0.groupby(
            diagnostic,
            observed=True,
        ):

            if len(g) < 8:
                continue

            row = {
                "sample": sample_name,
                "diagnostic": diagnostic,
                "bucket": str(bucket),
                "n": int(len(g)),
            }

            for h in horizons:

                r = pd.to_numeric(
                    g[f"ret_fwd_{h}d"],
                    errors="coerce",
                ).dropna()

                vol = pd.to_numeric(
                    g[f"future_vol_{h}d"],
                    errors="coerce",
                ).dropna()

                row[f"ret_mean_{h}d"] = float(r.mean()) if len(r) else np.nan

                row[f"ret_median_{h}d"] = float(r.median()) if len(r) else np.nan

                row[f"positive_rate_{h}d"] = float((r > 0).mean()) if len(r) else np.nan

                row[f"negative_rate_{h}d"] = float((r < 0).mean()) if len(r) else np.nan

                row[f"future_vol_mean_{h}d"] = float(vol.mean()) if len(vol) else np.nan

                if len(r) >= 10:

                    q05 = r.quantile(0.05)

                    row[f"var05_{h}d"] = float(q05)

                    row[f"cvar05_{h}d"] = float(r[r <= q05].mean())

                else:

                    row[f"var05_{h}d"] = np.nan
                    row[f"cvar05_{h}d"] = np.nan

            rows.append(row)

    summary = pd.DataFrame(rows)

    # ============================================================
    # PRINT
    # ============================================================

    print("\n" + "=" * 120)

    print(f"REGIME 2 / SUBSTATE 0 — " f"ECONOMIC VALIDATION — {sample_name}")

    print("=" * 120)

    print(f"Observations regime2/substate0 : {len(s0)}")

    if not summary.empty:

        print(summary.round(5).to_string(index=False))

    return summary, s0


def analyze_regime2_substate_market_structure(
    *,
    feat: pd.DataFrame,
    state_feed: pd.DataFrame,
    futures: pd.DataFrame,
    sample_name: str,
    price_col: str = "FUTURES_CLOSE",
    horizons=(1, 3, 5, 10, 20),
):
    """
    Analyse STRUCTURELLE du régime 2 / sous-HMM.

    Aucun trade.
    Aucune position.
    Aucun PnL de stratégie.
    Aucun seuil optimisé.

    Objectif
    --------
    Déterminer si les sous-états 0/1/2 correspondent à des
    structures de marché distinctes et reproductibles TRAIN -> TEST.

    Variables courantes, toutes causales :
        - vol_chg_20
        - downside_chg_vol_20
        - vol_skew_diff_20
        - efficiency_20
        - signchg_20
        - autocorr_1_20
        - vr_5_20
        - spread_vol_20
        - spread_to_vol_20
        - impact_proxy_20
        - abs_dd_60
        - trend_slope_20

    Variables FUTURES diagnostiques :
        - rendement futur
        - mouvement absolu futur
        - variance réalisée future
        - range futur
        - efficiency future
        - fréquence future de changement de signe

    Les variables futures servent UNIQUEMENT au diagnostic.
    Elles ne doivent jamais entrer dans un filtre.
    """

    # ============================================================
    # FEATURES CAUSALES QUE NOUS VOULONS ÉTUDIER
    # ============================================================

    candidate_features = [
        # Vol / stress
        "vol_chg_20",
        "downside_chg_vol_20",
        "upside_chg_vol_20",
        "vol_skew_diff_20",
        # Choppiness / structure du chemin
        "efficiency_20",
        "signchg_20",
        "autocorr_1_20",
        "vr_5_20",
        # Liquidité / microstructure
        "spread_vol_20",
        "spread_to_vol_20",
        "impact_proxy_20",
        # Dislocation / tendance
        "abs_dd_60",
        "trend_slope_20",
        "trend_slope_60",
    ]

    features = [c for c in candidate_features if c in feat.columns]

    if not features:
        raise ValueError("Aucune feature de structure de marché disponible.")

    required_state_cols = [
        "spot_state",
        "regime2_substate_3",
        "regime2_substate_confidence",
        "regime2_substate_age",
    ]

    missing = [c for c in required_state_cols if c not in state_feed.columns]

    if missing:
        raise ValueError(f"Colonnes sous-HMM absentes : {missing}")

    if price_col not in futures.columns:
        raise ValueError(f"Colonne futures absente : {price_col}")

    # ============================================================
    # NORMALISATION
    # ============================================================

    feat_local = feat.copy().sort_index()
    states = state_feed.copy().sort_index()
    market = futures.copy().sort_index()

    for frame in (
        feat_local,
        states,
        market,
    ):
        frame.index = pd.to_datetime(frame.index)

        if frame.index.tz is not None:
            frame.index = frame.index.tz_localize(None)

    # ============================================================
    # FUTURES : MÉTRIQUES SUR CALENDRIER COMPLET
    # ============================================================

    price = pd.to_numeric(
        market[price_col],
        errors="coerce",
    )

    market = market.loc[price.notna() & (price > 0)].copy()

    log_price = np.log(
        pd.to_numeric(
            market[price_col],
            errors="coerce",
        )
    )

    one_day_ret = log_price.diff()

    # ------------------------------------------------------------
    # On calcule d'abord TOUT sur le calendrier futures complet.
    # Ensuite seulement on aligne sur les dates HMM.
    # ------------------------------------------------------------

    for h in horizons:

        # Rendement total futur
        market[f"future_return_{h}d"] = log_price.shift(-h) - log_price

        # Somme des mouvements absolus futurs
        future_ret_matrix = pd.concat(
            [one_day_ret.shift(-i) for i in range(1, h + 1)],
            axis=1,
        )

        market[f"future_abs_path_{h}d"] = future_ret_matrix.abs().sum(axis=1)

        market[f"future_rv_{h}d"] = future_ret_matrix.pow(2).sum(axis=1)

        # Efficiency :
        # déplacement net / chemin total
        market[f"future_efficiency_{h}d"] = market[f"future_return_{h}d"].abs() / (
            market[f"future_abs_path_{h}d"] + 1e-12
        )

        # --------------------------------------------------------
        # Sign-change rate FUTUR
        # --------------------------------------------------------

        sign_values = np.sign(future_ret_matrix.to_numpy(dtype=float))

        signchg = []

        for row in sign_values:

            valid = row[np.isfinite(row)]

            valid = valid[valid != 0]

            if len(valid) <= 1:
                signchg.append(np.nan)
                continue

            signchg.append(float(np.mean(valid[1:] != valid[:-1])))

        market[f"future_signchg_{h}d"] = signchg

        # --------------------------------------------------------
        # Future range
        # --------------------------------------------------------

        future_price_matrix = pd.concat(
            [log_price.shift(-i) for i in range(0, h + 1)],
            axis=1,
        )

        market[f"future_range_{h}d"] = future_price_matrix.max(
            axis=1
        ) - future_price_matrix.min(axis=1)

    # ============================================================
    # ALIGNEMENT EXACT :
    # FEATURE SPOT × HMM × FUTURES
    # ============================================================

    idx = (
        feat_local.index.intersection(states.index)
        .intersection(market.index)
        .sort_values()
    )

    df = pd.DataFrame(index=idx)

    # Features CAUSALES
    for col in features:

        df[col] = pd.to_numeric(
            feat_local.loc[
                idx,
                col,
            ],
            errors="coerce",
        )

    # HMM
    for col in required_state_cols:

        df[col] = pd.to_numeric(
            states.loc[
                idx,
                col,
            ],
            errors="coerce",
        )

    # Futures uniquement diagnostic
    future_cols = []

    for h in horizons:

        cols_h = [
            f"future_return_{h}d",
            f"future_abs_path_{h}d",
            f"future_rv_{h}d",
            f"future_range_{h}d",
            f"future_efficiency_{h}d",
            f"future_signchg_{h}d",
        ]

        future_cols.extend(cols_h)

        for col in cols_h:

            df[col] = market.loc[
                idx,
                col,
            ]

    # ============================================================
    # UNIQUEMENT REGIME PRINCIPAL 2
    # ============================================================

    df = df[df["spot_state"] == 2].copy()

    df["substate"] = pd.to_numeric(
        df["regime2_substate_3"],
        errors="coerce",
    )

    df = df[df["substate"].isin([0, 1, 2])].copy()

    df["substate"] = df["substate"].astype(int)

    if df.empty:
        raise ValueError(f"{sample_name}: aucune observation R2 valide.")

    # ============================================================
    # 1. PROFIL ACTUEL DES 3 SUBSTATES
    # ============================================================

    current_rows = []

    for s in (0, 1, 2):

        g = df[df["substate"] == s]

        if g.empty:
            continue

        for feature in features:

            x = pd.to_numeric(
                g[feature],
                errors="coerce",
            ).dropna()

            if len(x) < 5:
                continue

            current_rows.append(
                {
                    "sample": sample_name,
                    "substate": s,
                    "feature": feature,
                    "n": len(x),
                    "mean": float(x.mean()),
                    "median": float(x.median()),
                    "std": float(x.std(ddof=1)),
                    "q25": float(x.quantile(0.25)),
                    "q75": float(x.quantile(0.75)),
                }
            )

    current_summary = pd.DataFrame(current_rows)

    # ============================================================
    # 2. PROFIL FUTUR DES 3 SUBSTATES
    # ============================================================

    future_rows = []

    for s in (0, 1, 2):

        g = df[df["substate"] == s]

        if g.empty:
            continue

        for col in future_cols:

            x = pd.to_numeric(
                g[col],
                errors="coerce",
            ).dropna()

            if len(x) < 5:
                continue

            row = {
                "sample": sample_name,
                "substate": s,
                "metric": col,
                "n": len(x),
                "mean": float(x.mean()),
                "median": float(x.median()),
                "std": float(x.std(ddof=1)),
                "q25": float(x.quantile(0.25)),
                "q75": float(x.quantile(0.75)),
            }

            # Pour les rendements uniquement
            if col.startswith("future_return_"):

                row["positive_rate"] = float((x > 0).mean())

                row["negative_rate"] = float((x < 0).mean())

                q05 = x.quantile(0.05)

                row["VaR_05"] = float(q05)

                row["CVaR_05"] = float(x[x <= q05].mean())

            else:

                row["positive_rate"] = np.nan
                row["negative_rate"] = np.nan
                row["VaR_05"] = np.nan
                row["CVaR_05"] = np.nan

            future_rows.append(row)

    future_summary = pd.DataFrame(future_rows)

    # ============================================================
    # 3. KRUSKAL ENTRE SUBSTATES
    #
    # Pas de seuil optimisé.
    # On teste simplement si les distributions diffèrent.
    # ============================================================

    test_rows = []

    all_test_cols = features + future_cols

    for col in all_test_cols:

        groups = []

        sizes = []

        for s in (0, 1, 2):

            values = pd.to_numeric(
                df.loc[
                    df["substate"] == s,
                    col,
                ],
                errors="coerce",
            ).dropna()

            if len(values) >= 10:

                groups.append(values.to_numpy())

                sizes.append(len(values))

        if len(groups) < 2:
            continue

        try:

            H, p_value = stats.kruskal(*groups)

        except ValueError:

            H = np.nan
            p_value = np.nan

        test_rows.append(
            {
                "sample": sample_name,
                "metric": col,
                "n_groups": len(groups),
                "n_total": int(sum(sizes)),
                "kruskal_H": H,
                "p_value": p_value,
            }
        )

    tests = pd.DataFrame(test_rows)

    # ============================================================
    # 4. EFFET STANDARDISÉ SIMPLE
    #
    # Pour faciliter comparaison TRAIN / TEST :
    # distance max des médianes / dispersion globale.
    # ============================================================

    effect_rows = []

    for col in all_test_cols:

        valid = pd.to_numeric(
            df[col],
            errors="coerce",
        ).dropna()

        if len(valid) < 20:
            continue

        medians = {}

        for s in (0, 1, 2):

            values = pd.to_numeric(
                df.loc[
                    df["substate"] == s,
                    col,
                ],
                errors="coerce",
            ).dropna()

            if len(values) >= 5:

                medians[s] = float(values.median())

        if len(medians) < 2:
            continue

        # Robust scale = IQR
        iqr = valid.quantile(0.75) - valid.quantile(0.25)

        spread = max(medians.values()) - min(medians.values())

        robust_effect = spread / iqr if abs(iqr) > 1e-12 else np.nan

        ordered = sorted(
            medians.items(),
            key=lambda item: item[1],
        )

        ranking = " < ".join([f"S{s}" for s, _ in ordered])

        effect_rows.append(
            {
                "sample": sample_name,
                "metric": col,
                "median_s0": medians.get(
                    0,
                    np.nan,
                ),
                "median_s1": medians.get(
                    1,
                    np.nan,
                ),
                "median_s2": medians.get(
                    2,
                    np.nan,
                ),
                "ranking_low_to_high": ranking,
                "median_spread": spread,
                "global_iqr": iqr,
                "robust_effect": robust_effect,
            }
        )

    effects = pd.DataFrame(effect_rows)

    # ============================================================
    # PRINT
    # ============================================================

    print("\n" + "=" * 120)

    print(f"REGIME 2 — SUBSTATE MARKET STRUCTURE — {sample_name}")

    print("=" * 120)

    print("\nObservations :")

    print(df["substate"].value_counts().sort_index().to_string())

    print("\n--- CURRENT MARKET STRUCTURE ---")

    if not current_summary.empty:

        pivot_current = current_summary.pivot(
            index="feature",
            columns="substate",
            values="median",
        )

        print(pivot_current.round(5).to_string())

    print("\n--- FUTURE MARKET BEHAVIOUR ---")

    if not future_summary.empty:

        pivot_future = future_summary.pivot(
            index="metric",
            columns="substate",
            values="median",
        )

        print(pivot_future.round(5).to_string())

    print("\n--- STRONGEST CROSS-SUBSTATE DIFFERENCES ---")

    if not tests.empty:

        print(tests.sort_values("p_value").head(40).round(6).to_string(index=False))

    print("\n--- ROBUST MEDIAN EFFECTS ---")

    if not effects.empty:

        print(
            effects.sort_values(
                "robust_effect",
                ascending=False,
            )
            .head(40)
            .round(5)
            .to_string(index=False)
        )

    return {
        "details": df,
        "current_summary": current_summary,
        "future_summary": future_summary,
        "tests": tests,
        "effects": effects,
    }


def compare_regime2_market_structure_train_test(
    *,
    train_results: Dict[str, pd.DataFrame],
    test_results: Dict[str, pd.DataFrame],
):
    """
    Compare automatiquement les hiérarchies des substates
    entre TRAIN et TEST.

    On ne recherche PAS la meilleure performance.

    Une métrique est intéressante lorsque :
        - son ordre S0/S1/S2 est identique TRAIN et TEST ;
        - son effet robuste conserve une amplitude raisonnable ;
        - idéalement Kruskal montre une séparation dans les deux samples.
    """

    train_eff = train_results["effects"].copy()

    test_eff = test_results["effects"].copy()

    train_tests = train_results["tests"].copy()

    test_tests = test_results["tests"].copy()

    train_eff = train_eff.rename(
        columns={
            "ranking_low_to_high": "ranking_train",
            "robust_effect": "effect_train",
            "median_s0": "train_s0",
            "median_s1": "train_s1",
            "median_s2": "train_s2",
        }
    )

    test_eff = test_eff.rename(
        columns={
            "ranking_low_to_high": "ranking_test",
            "robust_effect": "effect_test",
            "median_s0": "test_s0",
            "median_s1": "test_s1",
            "median_s2": "test_s2",
        }
    )

    keep_eff = [
        "metric",
        "ranking_train",
        "effect_train",
        "train_s0",
        "train_s1",
        "train_s2",
    ]

    keep_test_eff = [
        "metric",
        "ranking_test",
        "effect_test",
        "test_s0",
        "test_s1",
        "test_s2",
    ]

    comparison = train_eff[keep_eff].merge(
        test_eff[keep_test_eff],
        on="metric",
        how="inner",
    )

    comparison["same_ranking"] = (
        comparison["ranking_train"] == comparison["ranking_test"]
    )

    # Ratio d'amplitude OOS / TRAIN.
    # On ne veut pas forcément 1.0,
    # seulement éviter un effondrement complet.
    comparison["effect_ratio_test_train"] = comparison["effect_test"] / (
        comparison["effect_train"].abs() + 1e-12
    )

    train_p = train_tests[
        [
            "metric",
            "p_value",
        ]
    ].rename(columns={"p_value": "p_train"})

    test_p = test_tests[
        [
            "metric",
            "p_value",
        ]
    ].rename(columns={"p_value": "p_test"})

    comparison = comparison.merge(
        train_p,
        on="metric",
        how="left",
    ).merge(
        test_p,
        on="metric",
        how="left",
    )

    # Ceci n'est PAS un score de trading.
    # C'est seulement un filtre de robustesse descriptive.
    comparison["replicates_direction"] = comparison["same_ranking"]

    comparison = comparison.sort_values(
        [
            "replicates_direction",
            "effect_test",
        ],
        ascending=[
            False,
            False,
        ],
    )

    print("\n" + "=" * 130)

    print("REGIME 2 — TRAIN / TEST STRUCTURAL REPLICATION")

    print("=" * 130)

    print(
        comparison[
            [
                "metric",
                "ranking_train",
                "ranking_test",
                "same_ranking",
                "effect_train",
                "effect_test",
                "effect_ratio_test_train",
                "p_train",
                "p_test",
                "train_s0",
                "train_s1",
                "train_s2",
                "test_s0",
                "test_s1",
                "test_s2",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    return comparison


from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from scipy.stats import spearmanr


def analyze_regime2_substate_incremental_oos_value(
    *,
    train_details: pd.DataFrame,
    test_details: pd.DataFrame,
    targets=None,
    feature_cols=None,
    ridge_alphas=None,
    n_splits: int = 4,
):
    """
    Test incrémental OOS du sous-HMM du régime 2.

    QUESTION
    --------
    Est-ce que le label substate 0/1/2 contient une information
    prédictive supplémentaire AU-DELA des variables observables
    de marché déjà disponibles ?

    MODELE A
    --------
        target_future
            ~ variables observables causales

    MODELE B
    --------
        target_future
            ~ mêmes variables observables causales
            + one-hot(substate)

    IMPORTANT
    ---------
    - TRAIN utilisé pour choisir alpha.
    - TEST jamais utilisé pour choisir hyperparamètre.
    - scaler ajusté uniquement sur TRAIN.
    - médianes d'imputation calculées uniquement sur TRAIN.
    - substate utilisé à date t uniquement.
    - aucune position de stratégie.
    - aucun PnL.
    - aucune optimisation de filtre.
    - aucune variable future utilisée comme feature.

    Le but n'est PAS de maximiser un backtest.

    On cherche uniquement à savoir si le sous-HMM apporte
    une information incrémentale OOS.
    """

    train = train_details.copy().sort_index()
    test = test_details.copy().sort_index()

    # ============================================================
    # TARGETS FUTURES
    # ============================================================

    if targets is None:
        targets = [
            "future_abs_path_5d",
            "future_abs_path_10d",
            "future_abs_path_20d",
            "future_range_5d",
            "future_range_10d",
            "future_range_20d",
            "future_rv_5d",
            "future_rv_10d",
            "future_rv_20d",
        ]

    targets = [c for c in targets if (c in train.columns and c in test.columns)]

    if not targets:
        raise ValueError(
            "Aucune target future disponible dans " "train_details / test_details."
        )

    # ============================================================
    # VARIABLES OBSERVABLES CAUSALES
    #
    # Aucun target futur ici.
    # ============================================================

    if feature_cols is None:
        feature_cols = [
            # Volatilité / asymétrie
            "vol_chg_20",
            "downside_chg_vol_20",
            "upside_chg_vol_20",
            "vol_skew_diff_20",
            # Structure / choppiness
            "efficiency_20",
            "signchg_20",
            "autocorr_1_20",
            "vr_5_20",
            # Liquidité
            "spread_vol_20",
            "spread_to_vol_20",
            "impact_proxy_20",
            # Dislocation / trend
            "abs_dd_60",
            "trend_slope_20",
            "trend_slope_60",
        ]

    feature_cols = [
        c for c in feature_cols if (c in train.columns and c in test.columns)
    ]

    if not feature_cols:
        raise ValueError("Aucune feature observable commune TRAIN / TEST.")

    if "substate" not in train.columns:
        raise ValueError("Colonne substate absente de train_details.")

    if "substate" not in test.columns:
        raise ValueError("Colonne substate absente de test_details.")

    # ============================================================
    # RIDGE GRID
    #
    # Choix exclusivement sur TRAIN.
    # ============================================================

    if ridge_alphas is None:
        ridge_alphas = np.array(
            [
                0.01,
                0.03,
                0.10,
                0.30,
                1.00,
                3.00,
                10.0,
                30.0,
                100.0,
            ],
            dtype=float,
        )

    # ============================================================
    # ONE-HOT SUBSTATE
    #
    # Toujours 3 colonnes fixes :
    # S0, S1, S2
    #
    # drop S0 pour éviter colinéarité parfaite.
    # ============================================================

    def _make_substate_dummies(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        substate = pd.to_numeric(
            df["substate"],
            errors="coerce",
        )

        out = pd.DataFrame(index=df.index)

        out["substate_1"] = (substate == 1).astype(float)

        out["substate_2"] = (substate == 2).astype(float)

        # S0 = catégorie de référence.

        return out

    train_substate = _make_substate_dummies(train)

    test_substate = _make_substate_dummies(test)

    # ============================================================
    # FEATURES BRUTES
    # ============================================================

    X_train_base = (
        train[feature_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    X_test_base = (
        test[feature_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # ============================================================
    # IMPUTATION
    #
    # Très important :
    # médiane TRAIN seulement.
    # ============================================================

    train_medians = X_train_base.median()

    X_train_base = X_train_base.fillna(train_medians)

    X_test_base = X_test_base.fillna(train_medians)

    # Colonnes complètement invalides éventuelles.
    valid_cols = [
        c
        for c in X_train_base.columns
        if (X_train_base[c].notna().all() and X_test_base[c].notna().all())
    ]

    X_train_base = X_train_base[valid_cols]

    X_test_base = X_test_base[valid_cols]

    if not valid_cols:
        raise ValueError("Toutes les features sont invalides après nettoyage.")

    # ============================================================
    # CONSTRUCTION A / B
    # ============================================================

    X_train_A_raw = X_train_base.copy()

    X_test_A_raw = X_test_base.copy()

    X_train_B_raw = pd.concat(
        [
            X_train_base,
            train_substate,
        ],
        axis=1,
    )

    X_test_B_raw = pd.concat(
        [
            X_test_base,
            test_substate,
        ],
        axis=1,
    )

    # ============================================================
    # HELPERS
    # ============================================================

    def _rmse(
        y_true,
        y_pred,
    ):
        return float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        )

    def _spearman(
        y_true,
        y_pred,
    ):
        if len(y_true) < 3:
            return np.nan

        rho, p = spearmanr(
            y_true,
            y_pred,
            nan_policy="omit",
        )

        return (
            float(rho),
            float(p),
        )

    # ============================================================
    # CHOIX CAUSAL DE ALPHA SUR TRAIN
    #
    # TimeSeriesSplit :
    # jamais de random shuffle.
    # ============================================================

    def _choose_alpha(
        X_raw: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[float, pd.DataFrame]:

        common = X_raw.index.intersection(y.dropna().index).sort_values()

        X = X_raw.loc[common].copy()

        y_local = y.loc[common].astype(float)

        if len(X) < 40:
            raise ValueError(
                "Pas assez d'observations TRAIN " f"pour TimeSeriesSplit : {len(X)}"
            )

        actual_splits = min(
            int(n_splits),
            max(
                2,
                len(X) // 30,
            ),
        )

        tscv = TimeSeriesSplit(n_splits=actual_splits)

        alpha_rows = []

        for alpha in ridge_alphas:

            fold_errors = []

            for (
                train_idx,
                val_idx,
            ) in tscv.split(X):

                X_fit = X.iloc[train_idx]

                X_val = X.iloc[val_idx]

                y_fit = y_local.iloc[train_idx]

                y_val = y_local.iloc[val_idx]

                # -----------------------------------------------
                # Scaling fit uniquement sur passé du fold.
                # -----------------------------------------------

                scaler = StandardScaler()

                X_fit_scaled = scaler.fit_transform(X_fit)

                X_val_scaled = scaler.transform(X_val)

                model = Ridge(
                    alpha=float(alpha),
                    fit_intercept=True,
                )

                model.fit(
                    X_fit_scaled,
                    y_fit,
                )

                pred = model.predict(X_val_scaled)

                fold_mae = mean_absolute_error(
                    y_val,
                    pred,
                )

                fold_errors.append(float(fold_mae))

            alpha_rows.append(
                {
                    "alpha": float(alpha),
                    "cv_mae_mean": float(np.mean(fold_errors)),
                    "cv_mae_std": float(np.std(fold_errors)),
                    "n_folds": len(fold_errors),
                }
            )

        alpha_table = (
            pd.DataFrame(alpha_rows)
            .sort_values(
                [
                    "cv_mae_mean",
                    "alpha",
                ]
            )
            .reset_index(drop=True)
        )

        best_alpha = float(alpha_table.iloc[0]["alpha"])

        return (
            best_alpha,
            alpha_table,
        )

    # ============================================================
    # FIT TRAIN COMPLET + TEST OOS
    # ============================================================

    def _fit_and_test(
        *,
        X_train_raw: pd.DataFrame,
        X_test_raw: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        alpha: float,
    ):

        train_idx = X_train_raw.index.intersection(y_train.dropna().index).sort_values()

        test_idx = X_test_raw.index.intersection(y_test.dropna().index).sort_values()

        Xtr = X_train_raw.loc[train_idx]

        ytr = y_train.loc[train_idx].astype(float)

        Xte = X_test_raw.loc[test_idx]

        yte = y_test.loc[test_idx].astype(float)

        scaler = StandardScaler()

        Xtr_scaled = scaler.fit_transform(Xtr)

        Xte_scaled = scaler.transform(Xte)

        model = Ridge(
            alpha=float(alpha),
            fit_intercept=True,
        )

        model.fit(
            Xtr_scaled,
            ytr,
        )

        pred_train = model.predict(Xtr_scaled)

        pred_test = model.predict(Xte_scaled)

        test_rho, test_rho_p = _spearman(
            yte,
            pred_test,
        )

        result = {
            "n_train": len(ytr),
            "n_test": len(yte),
            "alpha": float(alpha),
            # TRAIN, juste diagnostic
            "train_mae": float(
                mean_absolute_error(
                    ytr,
                    pred_train,
                )
            ),
            "train_rmse": _rmse(
                ytr,
                pred_train,
            ),
            "train_r2": float(
                r2_score(
                    ytr,
                    pred_train,
                )
            ),
            # TEST = résultat qui compte
            "test_mae": float(
                mean_absolute_error(
                    yte,
                    pred_test,
                )
            ),
            "test_rmse": _rmse(
                yte,
                pred_test,
            ),
            "test_r2": float(
                r2_score(
                    yte,
                    pred_test,
                )
            ),
            "test_spearman": test_rho,
            "test_spearman_p": test_rho_p,
        }

        coef = pd.DataFrame(
            {
                "feature": X_train_raw.columns,
                "coefficient": model.coef_,
            }
        )

        coef["abs_coefficient"] = coef["coefficient"].abs()

        coef = coef.sort_values(
            "abs_coefficient",
            ascending=False,
        )

        predictions = pd.DataFrame(
            {
                "actual": yte,
                "prediction": pred_test,
            },
            index=test_idx,
        )

        return (
            result,
            coef,
            predictions,
        )

    # ============================================================
    # LOOP TARGETS
    # ============================================================

    result_rows = []

    cv_tables = {}

    coefficient_tables = {}

    prediction_tables = {}

    for target in targets:

        print("\n" + "=" * 120)

        print(f"INCREMENTAL OOS TEST — {target}")

        print("=" * 120)

        y_train = pd.to_numeric(
            train[target],
            errors="coerce",
        )

        y_test = pd.to_numeric(
            test[target],
            errors="coerce",
        )

        # ========================================================
        # MODEL A
        # ========================================================

        alpha_A, cv_A = _choose_alpha(
            X_train_A_raw,
            y_train,
        )

        (
            result_A,
            coef_A,
            pred_A,
        ) = _fit_and_test(
            X_train_raw=X_train_A_raw,
            X_test_raw=X_test_A_raw,
            y_train=y_train,
            y_test=y_test,
            alpha=alpha_A,
        )

        # ========================================================
        # MODEL B
        # ========================================================

        alpha_B, cv_B = _choose_alpha(
            X_train_B_raw,
            y_train,
        )

        (
            result_B,
            coef_B,
            pred_B,
        ) = _fit_and_test(
            X_train_raw=X_train_B_raw,
            X_test_raw=X_test_B_raw,
            y_train=y_train,
            y_test=y_test,
            alpha=alpha_B,
        )

        # ========================================================
        # INCREMENTAL VALUE
        # ========================================================

        mae_improvement = result_A["test_mae"] - result_B["test_mae"]

        rmse_improvement = result_A["test_rmse"] - result_B["test_rmse"]

        r2_improvement = result_B["test_r2"] - result_A["test_r2"]

        spearman_improvement = result_B["test_spearman"] - result_A["test_spearman"]

        mae_improvement_pct = mae_improvement / (result_A["test_mae"] + 1e-12)

        rmse_improvement_pct = rmse_improvement / (result_A["test_rmse"] + 1e-12)

        result_rows.append(
            {
                "target": target,
                "n_train": result_A["n_train"],
                "n_test": result_A["n_test"],
                "alpha_A": alpha_A,
                "alpha_B": alpha_B,
                # ------------------------------
                # MODEL A
                # ------------------------------
                "A_test_mae": result_A["test_mae"],
                "A_test_rmse": result_A["test_rmse"],
                "A_test_r2": result_A["test_r2"],
                "A_test_spearman": result_A["test_spearman"],
                # ------------------------------
                # MODEL B
                # ------------------------------
                "B_test_mae": result_B["test_mae"],
                "B_test_rmse": result_B["test_rmse"],
                "B_test_r2": result_B["test_r2"],
                "B_test_spearman": result_B["test_spearman"],
                # ------------------------------
                # INCREMENTAL
                # ------------------------------
                "mae_improvement": mae_improvement,
                "mae_improvement_pct": mae_improvement_pct,
                "rmse_improvement": rmse_improvement,
                "rmse_improvement_pct": rmse_improvement_pct,
                "r2_improvement": r2_improvement,
                "spearman_improvement": spearman_improvement,
                # B meilleur si erreur OOS baisse.
                "substate_improves_mae": bool(mae_improvement > 0),
                "substate_improves_rmse": bool(rmse_improvement > 0),
            }
        )

        # ========================================================
        # STOCKAGE DEBUG
        # ========================================================

        cv_tables[f"{target}_A"] = cv_A

        cv_tables[f"{target}_B"] = cv_B

        coefficient_tables[f"{target}_A"] = coef_A

        coefficient_tables[f"{target}_B"] = coef_B

        joined_predictions = (
            pred_A.rename(columns={"prediction": "prediction_A"})
            .drop(columns=["actual"])
            .join(pred_B.rename(columns={"prediction": "prediction_B"}))
        )

        prediction_tables[target] = joined_predictions

        # ========================================================
        # PRINT TARGET
        # ========================================================

        print("\nMODEL A — observable market variables")

        print(
            "alpha       :",
            round(
                alpha_A,
                6,
            ),
        )

        print(
            "TEST MAE    :",
            round(
                result_A["test_mae"],
                8,
            ),
        )

        print(
            "TEST RMSE   :",
            round(
                result_A["test_rmse"],
                8,
            ),
        )

        print(
            "TEST R²     :",
            round(
                result_A["test_r2"],
                6,
            ),
        )

        print(
            "TEST Spearman:",
            round(
                result_A["test_spearman"],
                6,
            ),
        )

        print("\nMODEL B — market variables + SUBSTATE")

        print(
            "alpha       :",
            round(
                alpha_B,
                6,
            ),
        )

        print(
            "TEST MAE    :",
            round(
                result_B["test_mae"],
                8,
            ),
        )

        print(
            "TEST RMSE   :",
            round(
                result_B["test_rmse"],
                8,
            ),
        )

        print(
            "TEST R²     :",
            round(
                result_B["test_r2"],
                6,
            ),
        )

        print(
            "TEST Spearman:",
            round(
                result_B["test_spearman"],
                6,
            ),
        )

        print("\nINCREMENTAL VALUE OF SUBSTATE")

        print(
            "MAE improvement % :",
            round(
                100.0 * mae_improvement_pct,
                3,
            ),
            "%",
        )

        print(
            "RMSE improvement %:",
            round(
                100.0 * rmse_improvement_pct,
                3,
            ),
            "%",
        )

        print(
            "R² improvement    :",
            round(
                r2_improvement,
                6,
            ),
        )

        print(
            "Spearman change   :",
            round(
                spearman_improvement,
                6,
            ),
        )

        print("\nSUBSTATE COEFFICIENTS MODEL B")

        substate_coef = coef_B[
            coef_B["feature"].isin(
                [
                    "substate_1",
                    "substate_2",
                ]
            )
        ]

        print(substate_coef.round(8).to_string(index=False))

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    summary = pd.DataFrame(result_rows)

    summary = summary.sort_values(
        "mae_improvement_pct",
        ascending=False,
    ).reset_index(drop=True)

    print("\n" + "=" * 140)

    print("REGIME 2 SUBSTATE — INCREMENTAL OOS VALUE SUMMARY")

    print("=" * 140)

    display_cols = [
        "target",
        "A_test_mae",
        "B_test_mae",
        "mae_improvement_pct",
        "A_test_rmse",
        "B_test_rmse",
        "rmse_improvement_pct",
        "A_test_r2",
        "B_test_r2",
        "r2_improvement",
        "A_test_spearman",
        "B_test_spearman",
        "spearman_improvement",
        "substate_improves_mae",
        "substate_improves_rmse",
    ]

    print(summary[display_cols].round(6).to_string(index=False))

    # ============================================================
    # SIMPLE CONSISTENCY SCORE
    #
    # Attention :
    # descriptif seulement.
    # Ce n'est PAS un seuil de trading.
    # ============================================================

    summary["improves_both_errors"] = (
        summary["substate_improves_mae"] & summary["substate_improves_rmse"]
    )

    n_targets = len(summary)

    n_better = int(summary["improves_both_errors"].sum())

    print(
        "\nTargets où le substate améliore " "MAE ET RMSE OOS :",
        f"{n_better}/{n_targets}",
    )

    if n_targets > 0:

        print(
            "Fraction :",
            round(
                n_better / n_targets,
                3,
            ),
        )

    return {
        "summary": summary,
        "cv_tables": cv_tables,
        "coefficient_tables": coefficient_tables,
        "prediction_tables": prediction_tables,
    }


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    brier_score_loss,
)


def main():
    print("=" * 80)
    print("SPOT HMM LATENT STATE FEED — DATA PRODUCT VERSION")
    print("=" * 80)

    # ── Spot seul pour le HMM ────────────────────────────────────────
    spot = load_spot()
    spot = spot.sort_values("Date")

    spot["Date"] = pd.to_datetime(spot["Date"], errors="coerce")
    spot = spot[spot["Date"] >= pd.Timestamp("2011-12-01")].copy()

    spot = spot.set_index("Date")

    spot = spot.dropna(subset=["MID"])
    spot = spot[(spot["MID"] > 0) & (spot["BA_SPREAD"] >= 0)]

    n = len(spot)
    n_train = int(n * TRAIN_RATIO)

    spot_train = spot.iloc[:n_train].copy()
    spot_test = spot.iloc[n_train:].copy()

    print(f"Train observations: {len(spot_train)}")
    print(f"Test observations : {len(spot_test)}")

    feat_train = build_features(spot_train)
    feat_all_raw = build_features(pd.concat([spot_train, spot_test]))
    feat_test = feat_all_raw.loc[feat_all_raw.index.isin(spot_test.index)].copy()

    candidate_cols = [
        c
        for c in FSHMM_CANDIDATE_COLS
        if c in feat_train.columns and c in feat_test.columns
    ]
    # ============================================================
    # SAME CLEANING AS SALIENCY
    # ============================================================

    X_train_df = feat_train[candidate_cols].copy()
    X_test_df = feat_test[candidate_cols].copy()

    # Replace inf
    X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan)
    X_test_df = X_test_df.replace([np.inf, -np.inf], np.nan)

    # Remove empty columns
    X_train_df = X_train_df.dropna(axis=1, how="all")
    X_test_df = X_test_df[X_train_df.columns]

    # Forward fill short gaps
    X_train_df = X_train_df.ffill(limit=3)
    X_test_df = X_test_df.ffill(limit=3)

    # Median fill
    train_medians = X_train_df.median()

    X_train_df = X_train_df.fillna(train_medians)
    X_test_df = X_test_df.fillna(train_medians)

    # Remove low-information columns
    nunique = X_train_df.nunique(dropna=True)
    keep = nunique[nunique > 3].index

    X_train_df = X_train_df[keep]
    X_test_df = X_test_df[keep]

    # Remove near-zero variance
    std = X_train_df.std()
    keep = std[std > 1e-10].index

    X_train_df = X_train_df[keep]
    X_test_df = X_test_df[keep]

    # Final feature list
    candidate_cols = list(X_train_df.columns)
    if len(candidate_cols) != len(FSHMM_CANDIDATE_COLS):
        missing = set(FSHMM_CANDIDATE_COLS) - set(candidate_cols)
        raise ValueError(f"Missing candidate features: {missing}")

    from sklearn.preprocessing import RobustScaler

    feature_sanity_report(X_train_df)
    scaler = RobustScaler()

    X_train_candidates = scaler.fit_transform(X_train_df)

    X_test_candidates = scaler.transform(X_test_df)

    sjm = SparseJumpModel(
        n_components=NB_STATES,
        jump_penalty=30.0,
        max_feats=6,
        cont=False,
        random_state=SEED,
    )

    X_train_df_scaled = pd.DataFrame(
        X_train_candidates,
        columns=candidate_cols,
    )
    sjm.fit(X_train_df_scaled)

    weights_series = sjm.feat_weights
    common_cols = weights_series[weights_series > 1e-6].index.tolist()
    selected_weights = weights_series[common_cols].values
    selected_idx = [candidate_cols.index(c) for c in common_cols]

    print("\n" + "=" * 80)
    print("SPARSE JUMP FINAL FEATURES")
    print("=" * 80)
    print(common_cols)
    print("\n" + "=" * 80)
    print("SPARSE WEIGHTS USED BY FINAL HMM")
    print("=" * 80)
    print(weights_series.sort_values(ascending=False).round(6))

    weights_series.to_csv("sparse_jump_feature_report.csv", header=True)

    X_train = X_train_candidates[:, selected_idx] * selected_weights[None, :]
    X_test = X_test_candidates[:, selected_idx] * selected_weights[None, :]
    if not np.isfinite(X_train).all():
        raise ValueError("X_train contains NaN or Inf")

    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains NaN or Inf")

    # select_k(X_train, k_range=(2, 3, 4))

    model, result = fit_hmm(X_train, NB_STATES)

    train_states, train_alpha, train_ll = model.filter(X_train)

    pi_test = train_alpha[-1] @ model.A
    test_states, test_alpha, test_ll = model.filter_with_pi(X_test, pi_test)

    print("\n=== HMM FIT ===")
    print("LL train:", train_ll)
    print("LL test :", test_ll)

    transition_diagnostics(model.A)

    train_index = feat_train.index[: len(train_alpha)]
    test_index = feat_test.index[: len(test_alpha)]

    summarize_spot_states(
        feat=feat_train.loc[train_index],
        alpha=train_alpha,
        feature_cols=common_cols,
    )

    remap, switched, coherent = check_label_stability(
        feat_train,
        feat_test,
        train_alpha,
        test_alpha,
        feature_cols=common_cols,
        train_index=train_index,
        test_index=test_index,
        max_dist_threshold=2.0,
    )

    if switched:
        test_alpha_reordered = np.zeros_like(test_alpha)
        for test_s, train_s in remap.items():
            if test_s < test_alpha.shape[1] and train_s < test_alpha.shape[1]:
                test_alpha_reordered[:, train_s] = test_alpha[:, test_s]
        test_alpha = test_alpha_reordered
        print("test_alpha reordonné selon remap.")

    train_feed = build_state_feed(
        train_index,
        train_alpha,
    )

    test_feed = build_state_feed(
        test_index,
        test_alpha,
    )

    # Ajout des diagnostics HMM
    train_feed = add_hmm_diagnostics_to_feed(train_feed)

    test_feed = add_hmm_diagnostics_to_feed(test_feed)
    # ============================================================
    # OPTIMAL SWITCHING
    # STEP 1 - STOCHASTIC PROCESS ESTIMATION
    # ============================================================

    # ============================================================
    # OPTIMAL SWITCHING
    # STEP 1 - STOCHASTIC PROCESS ESTIMATION
    # ============================================================
    # ============================================================
    # FUTURES MARKET DATA FOR OPTIMAL SWITCHING
    # ============================================================

    futures_market = prepare_futures()

    # Futures correspondant aux périodes HMM train/test
    switching_market_train = futures_market.loc[
        futures_market.index.intersection(train_feed.index)
    ].copy()

    switching_market_test = futures_market.loc[
        futures_market.index.intersection(test_feed.index)
    ].copy()

    if switching_market_train.empty:
        raise ValueError("No common dates between TRAIN HMM feed and futures.")

    if switching_market_test.empty:
        raise ValueError("No common dates between TEST HMM feed and futures.")

    print(
        "[Optimal Switching] TRAIN futures observations:",
        len(switching_market_train),
    )

    print(
        "[Optimal Switching] TEST futures observations:",
        len(switching_market_test),
    )
    # ============================================================
    # OPTIMAL SWITCHING — STOCHASTIC PROCESS ON FUTURES
    # ============================================================

    switching_process_cfg = StochasticProcessConfig(
        # Actif réellement tradé : futures CORN
        price_col="FUTURES_CLOSE",
        # Régime latent fourni par le HMM entraîné sur le spot
        state_col="spot_state",
        min_obs_per_regime=30,
        # Le rendement t-1 -> t est associé au régime connu à t-1
        use_lagged_state=True,
        # Pas journalier
        dt=1.0,
        generator_method="first_order",
    )

    # ============================================================
    # FIT ON TRAIN ONLY
    # ============================================================

    switching_process = fit_stochastic_process(
        market_train=data_train_fut,
        train_feed=train_feed_val,
        transition_matrix=model.A,
        config=switching_process_cfg,
    )

    stochastic_process_report(switching_process)

    # ============================================================
    # ENRICH TRAIN
    # ============================================================

    train_feed_val = enrich_feed_with_stochastic_process(
        feed=train_feed_val,
        market_data=data_train_fut,
        estimate=switching_process,
    )

    # ============================================================
    # ENRICH TEST
    # ============================================================

    test_feed_val = enrich_feed_with_stochastic_process(
        feed=test_feed_val,
        market_data=data_test,
        estimate=switching_process,
    )
    # ============================================================
    # VARIABLE FUTURES GLOBALE — TOUS LES RÉGIMES
    # ============================================================

    futures_context = prepare_futures()

    train_feed = add_global_futures_abs_dd_60(
        feed=train_feed,
        futures=futures_context,
        window=60,
    )

    test_feed = add_global_futures_abs_dd_60(
        feed=test_feed,
        futures=futures_context,
        window=60,
    )
    # ============================================================
    # AJOUT DES VARIABLES ECONOMIQUES AU CSV HMM
    # ============================================================

    TRADE_CONTEXT_FEATURES = [
        "impact_proxy_20",
        "spread_to_vol_20",
        "downside_chg_vol_20",
        "vol_chg_20",
        # Autres variables utiles pour test_jobs.py
        "upside_chg_vol_20",
        "vol_skew_diff_20",
        "vr_5_20",
        "spread_vol_20",
        "range_abs_20",
        "trend_slope_20",
        "abs_dd_60",
        "efficiency_20",
        "signchg_20",
    ]

    # Vérifier que les variables existent
    missing_train_features = [
        col for col in TRADE_CONTEXT_FEATURES if col not in feat_train.columns
    ]

    missing_test_features = [
        col for col in TRADE_CONTEXT_FEATURES if col not in feat_test.columns
    ]

    if missing_train_features:
        raise ValueError(
            "Variables absentes de feat_train : " f"{missing_train_features}"
        )

    if missing_test_features:
        raise ValueError(
            "Variables absentes de feat_test : " f"{missing_test_features}"
        )

    # Ajouter les variables au feed HMM
    train_feed = train_feed.join(
        feat_train[TRADE_CONTEXT_FEATURES],
        how="left",
    )

    test_feed = test_feed.join(
        feat_test[TRADE_CONTEXT_FEATURES],
        how="left",
    )
    # ============================================================
    # QUANTILES DYNAMIQUES CAUSAUX PAR REGIME
    # ============================================================

    QUANTILE_FEATURES = [
        "impact_proxy_20",
        "spread_to_vol_20",
        "downside_chg_vol_20",
        "vol_chg_20",
        "upside_chg_vol_20",
        "vol_skew_diff_20",
        "vr_5_20",
        "spread_vol_20",
        "range_abs_20",
        "trend_slope_20",
        "abs_dd_60",
        "hmm_prob_jump",
        "p_state_1",
        "hmm_entropy",
    ]

    test_feed = add_causal_regime_quantiles_to_test_feed(
        train_feed=train_feed,
        test_feed=test_feed,
        features=QUANTILE_FEATURES,
        regime_col="spot_state",
        quantiles=(
            0.75,
            0.80,
            0.85,
            0.90,
        ),
        min_history=60,
        # None = expanding historique.
        # Aucun choix arbitraire de fenêtre mobile pour le moment.
        max_history=None,
    )
    print("\n[CHECK] train_feed columns after enrichment:")
    print(train_feed.columns.tolist())

    print("\n[CHECK] test_feed columns after enrichment:")
    print(test_feed.columns.tolist())

    print("\n[CHECK] test_feed columns after diagnostics:")
    print(test_feed.columns.tolist())

    # ── MODIFICATION : merge test spot ∩ futures uniquement ──────────
    fut_test = prepare_futures()

    # Validation uniquement sur dates communes spot ∩ futures
    data_test = spot_test.join(fut_test, how="inner")

    data_test = data_test[
        data_test["FUTURES_CLOSE"].notna() & (data_test["FUTURES_CLOSE"] > 0)
    ].copy()

    print(
        f"\ndata_test : {len(data_test)} lignes | "
        f"{data_test.index[0].date()} → {data_test.index[-1].date()}"
    )

    # ── MODIFICATION : filtrer test_feed et reconstruire test_alpha ──
    test_feed_val = test_feed.loc[test_feed.index.intersection(data_test.index)].copy()
    alpha_cols = [c for c in test_feed_val.columns if c.startswith("p_state_")]
    test_alpha_val = test_feed_val[alpha_cols].values

    print(f"test_feed_val : {len(test_feed_val)} observations")

    # ── MODIFICATION : diagnostic dates manquantes ───────────────────
    print("Dates dans test_feed_val non présentes dans data_test :")
    missing_futures_dates = test_feed.index.difference(data_test.index)

    print("Dates HMM spot exclues car futures non observé :")
    print(f"  → {len(missing_futures_dates)} dates exclues")

    # ── MODIFICATION : uniquement test pour la validation futures ────
    test_ac1 = conditional_futures_ac1_by_state(data_test, test_feed_val)
    # ============================================================
    # TRAIN FUTURES VALIDATION
    # Projection des régimes spot train sur la période futures disponible
    # de début futures jusqu'à fin train spot
    # ============================================================

    fut_all = prepare_futures()

    # Garder le calendrier quotidien complet du future
    data_train_fut = fut_all.loc[train_feed.index.min() : train_feed.index.max()].copy()

    # Projeter causalement le dernier régime spot connu
    # sur chaque date future
    train_feed_val = train_feed.reindex(
        data_train_fut.index,
        method="ffill",
    )

    # Supprimer uniquement les dates avant le premier régime disponible
    valid_mask = train_feed_val["spot_state"].notna()

    data_train_fut = data_train_fut.loc[valid_mask].copy()

    train_feed_val = train_feed_val.loc[valid_mask].copy()
    alpha_cols_train = [c for c in train_feed_val.columns if c.startswith("p_state_")]

    train_alpha_val = train_feed_val[alpha_cols_train].values
    print(
        f"\ndata_train_fut : {len(data_train_fut)} lignes | "
        f"{data_train_fut.index[0].date()} → {data_train_fut.index[-1].date()}"
    )

    print(f"train_feed_val : {len(train_feed_val)} observations")

    missing_train_futures_dates = train_feed.index.difference(data_train_fut.index)
    # ============================================================
    # PAPER — CARACTÉRISATION BEAR / SIDEWALK / BULL
    # SUR LES TROIS ÉTATS HMM EXISTANTS
    # ============================================================

    (
        paper_train_daily,
        paper_train_blocks,
        paper_train_block_details,
    ) = paper_style_state_characterization(
        data=data_train_fut,
        state_feed=train_feed_val,
        sample_name="TRAIN",
        price_col="FUTURES_CLOSE",
        state_col="spot_state",
        # On garde seulement les blocs ayant au moins 5 observations.
        min_block_len=5,
        # Correction robuste pour le test de moyenne.
        hac_maxlags=5,
    )

    paper_train_daily.to_csv(
        "paper_train_state_daily_characterization.csv",
        index=False,
    )

    paper_train_blocks.to_csv(
        "paper_train_state_block_characterization.csv",
        index=False,
    )

    paper_train_block_details.to_csv(
        "paper_train_block_details.csv",
        index=False,
    )

    print("\nSaved:")
    print("paper_train_state_daily_characterization.csv")
    print("paper_train_state_block_characterization.csv")
    print("paper_train_block_details.csv")

    print("Dates HMM train exclues car futures non observé :")
    print(f"  → {len(missing_train_futures_dates)} dates exclues")
    # ============================================================
    # TRAIN FUTURES CONDITIONAL AC1
    # ============================================================

    train_ac1 = conditional_futures_ac1_by_state(data_train_fut, train_feed_val)

    print("\n" + "=" * 80)
    print("TRAIN FUTURES CONDITIONAL AC1(k) BY SPOT STATE")
    print("=" * 80)
    print(train_ac1.round(4).to_string(index=False))

    # ============================================================
    # TRAIN FUTURES CONDITIONAL DYNAMICS
    # ============================================================

    print("\n" + "=" * 80)
    print("TRAIN CONDITIONAL FUTURES DYNAMICS")
    print("=" * 80)

    train_dyn, train_summary, train_tests = conditional_futures_dynamics(
        data=data_train_fut,
        alpha=train_alpha_val,
        index=train_feed_val.index,
        horizon=HORIZON,
    )
    # ============================================================
    # TRAIN BLOCK-LEVEL REGIME STATS
    # ============================================================

    train_blocks = build_block_regime_stats(
        data=data_train_fut,
        state_feed=train_feed_val,
        horizon=HORIZON,
        min_block_len=8,
    )

    print("\n" + "=" * 80)
    print("TRAIN BLOCK-LEVEL REGIME STATS")
    print("=" * 80)

    print(
        train_blocks.groupby("state")[
            [
                "length",
                "block_rv",
                "block_abs_return",
                "block_return",
                "block_range",
                "block_efficiency",
                f"block_ac1_{HORIZON}d",
            ]
        ]
        .agg(["count", "mean", "median", "std"])
        .round(4)
    )

    train_block_tests = block_level_tests(
        train_blocks, horizon=HORIZON, sample_name="Train"
    )

    print("\nTRAIN BLOCK-LEVEL TESTS")
    print(train_block_tests.round(6).to_string(index=False))

    print("\n" + "=" * 80)
    print("TEST FUTURES CONDITIONAL AC1(k) BY SPOT STATE")
    print("=" * 80)
    print(test_ac1.round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print("TEST CONDITIONAL FUTURES DYNAMICS")
    print("=" * 80)

    test_dyn, test_summary, test_tests = conditional_futures_dynamics(
        data=data_test,
        alpha=test_alpha_val,
        index=test_feed_val.index,
        horizon=HORIZON,
    )

    test_feed_val = test_feed_val.copy()
    test_feed_val["state_label"] = "State " + test_feed_val["spot_state"].astype(str)
    print("\n" + "=" * 80)
    print("STATE FEED SAMPLE")
    print("=" * 80)
    print(test_feed_val.tail(10).round(4).to_string())

    fut_raw = pd.read_excel(FUTURES_PATH)
    fut_raw.columns = fut_raw.columns.str.strip()
    fut_raw["Date"] = pd.to_datetime(fut_raw["Date"], errors="coerce")

    fut_raw = fut_raw.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    plot_futures_with_spot_hmm_blocks(
        futures=fut_raw,
        state_feed=test_feed_val,
        title="Futures observé avec blocs HMM spot projetés",
    )
    # ============================================================
    # ============================================================
    # AJOUT ADDITIF DES 3 SOUS-ÉTATS DU RÉGIME 2
    # Aucun GMM. Aucune colonne existante n'est modifiée.
    # ============================================================

    train_feed = add_regime2_subhmm3_columns(
        train_feed,
        split="train",
    )

    test_feed_val = add_regime2_subhmm3_columns(
        test_feed_val,
        split="test",
    )

    # ============================================================
    # REGIME 2 — MARKET STRUCTURE TEST
    # AUCUNE STRATEGIE / AUCUN TRADE
    # ============================================================

    r2_structure_train = analyze_regime2_substate_market_structure(
        feat=feat_train,
        state_feed=train_feed,
        futures=futures_context,
        sample_name="TRAIN",
        price_col="FUTURES_CLOSE",
        horizons=(1, 3, 5, 10, 20),
    )

    r2_structure_test = analyze_regime2_substate_market_structure(
        feat=feat_test,
        state_feed=test_feed_val,
        futures=futures_context,
        sample_name="TEST",
        price_col="FUTURES_CLOSE",
        horizons=(1, 3, 5, 10, 20),
    )

    r2_structure_comparison = compare_regime2_market_structure_train_test(
        train_results=r2_structure_train,
        test_results=r2_structure_test,
    )
    # ============================================================
    # REGIME 2 — SUBSTATE INCREMENTAL OOS VALUE
    #
    # A = MARKET FEATURES
    # B = MARKET FEATURES + SUBSTATE
    #
    # AUCUN TRADE / AUCUN PNL
    # ============================================================

    r2_incremental_oos = analyze_regime2_substate_incremental_oos_value(
        train_details=r2_structure_train["details"],
        test_details=r2_structure_test["details"],
        targets=[
            "future_abs_path_5d",
            "future_abs_path_10d",
            "future_abs_path_20d",
            "future_range_5d",
            "future_range_10d",
            "future_range_20d",
            "future_rv_5d",
            "future_rv_10d",
            "future_rv_20d",
        ],
        n_splits=4,
    )
    # ============================================================
    # REGIME 2 — SUBSTATE STABILITY / PERSISTENCE
    # ============================================================

    (
        r2_train_summary,
        r2_train_conditional,
        r2_train_details,
    ) = analyze_regime2_substate_stability(
        state_feed=train_feed,
        sample_name="TRAIN",
        horizons=(1, 3, 5, 10, 20),
    )

    (
        r2_test_summary,
        r2_test_conditional,
        r2_test_details,
    ) = analyze_regime2_substate_stability(
        state_feed=test_feed_val,
        sample_name="TEST",
        horizons=(1, 3, 5, 10, 20),
    )
    # ============================================================
    # REGIME 2 / SUBSTATE 0 — VALIDATION ECONOMIQUE
    # ============================================================

    (
        r2_s0_econ_train,
        r2_s0_econ_train_details,
    ) = analyze_regime2_substate0_economic_validation(
        data=data_train_fut,
        state_feed=train_feed,
        sample_name="TRAIN",
        price_col="FUTURES_CLOSE",
        horizons=(1, 3, 5, 10),
    )

    (
        r2_s0_econ_test,
        r2_s0_econ_test_details,
    ) = analyze_regime2_substate0_economic_validation(
        data=data_test,
        state_feed=test_feed_val,
        sample_name="TEST",
        price_col="FUTURES_CLOSE",
        horizons=(1, 3, 5, 10),
    )

    r2_s0_econ_train.to_csv(
        "regime2_substate0_economic_train.csv",
        index=False,
    )

    r2_s0_econ_test.to_csv(
        "regime2_substate0_economic_test.csv",
        index=False,
    )

    r2_train_summary.to_csv(
        "regime2_substate_stability_train_summary.csv",
        index=False,
    )

    r2_test_summary.to_csv(
        "regime2_substate_stability_test_summary.csv",
        index=False,
    )

    r2_train_conditional.to_csv(
        "regime2_substate_stability_train_conditional.csv",
        index=False,
    )

    r2_test_conditional.to_csv(
        "regime2_substate_stability_test_conditional.csv",
        index=False,
    )

    r2_train_details.to_csv(
        "regime2_substate_stability_train_details.csv",
    )

    r2_test_details.to_csv(
        "regime2_substate_stability_test_details.csv",
    )
    train_feed.to_csv(PROJECT_ROOT / "spot_hmm_state_feed_train.csv")

    test_feed_val.to_csv(PROJECT_ROOT / "spot_hmm_state_feed_test.csv")
    switching_process.regime_params.to_csv(
        "optimal_switching_regime_diffusion_params.csv"
    )

    switching_process.generator_frame().to_csv("optimal_switching_generator_Q.csv")

    switching_process.transition_frame().to_csv("optimal_switching_transition_P.csv")
    test_ac1.to_csv("test_futures_conditional_ac1.csv", index=False)
    test_dyn.to_csv("test_futures_conditional_dynamics.csv", index=False)

    print("\nSaved:")
    print(PROJECT_ROOT / "spot_hmm_state_feed_train.csv")
    print(PROJECT_ROOT / "spot_hmm_state_feed_test.csv")
    print("test_futures_conditional_ac1.csv")
    print("test_futures_conditional_dynamics.csv")

    test_blocks = build_block_regime_stats(
        data=data_test,
        state_feed=test_feed_val,
        horizon=HORIZON,
        min_block_len=8,
    )

    print("\n" + "=" * 80)
    print("TEST BLOCK-LEVEL REGIME STATS")
    print("=" * 80)
    print(
        test_blocks.groupby("state")[
            [
                "length",
                "block_rv",
                "block_abs_return",
                "block_range",
                "block_efficiency",
                f"block_ac1_{HORIZON}d",
            ]
        ]
        .agg(["count", "mean", "median", "std"])
        .round(4)
    )

    test_block_tests = block_level_tests(
        test_blocks, horizon=HORIZON, sample_name="Test"
    )

    print("\nBLOCK-LEVEL TESTS")
    print(test_block_tests.round(6).to_string(index=False))

    test_blocks.to_csv("test_block_regime_stats.csv", index=False)

    fut_raw = pd.read_excel(FUTURES_PATH)
    fut_raw.columns = fut_raw.columns.str.strip()
    fut_raw["Date"] = pd.to_datetime(fut_raw["Date"], errors="coerce")
    fut_raw = fut_raw.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    plot_futures_with_spot_hmm_blocks(
        futures=fut_raw,
        state_feed=test_feed_val,
        title="Futures observé avec blocs HMM détectés sur le spot",
    )

    plot_futures_volatility_by_spot_regime(data_test, test_feed_val)

    blocks = extract_blocks_from_feed(test_feed_val)

    blocks.to_csv("hmm_regime_blocks.csv", index=False)
    print("\n" + "=" * 80)
    print("REGIME DURATION ANALYSIS")
    print("=" * 80)

    duration_stats = regime_duration_table(test_blocks)

    print(duration_stats)
    survival = empirical_survival_by_state(test_blocks)

    survival.to_csv("regime_survival_curve.csv", index=False)

    print("\nSaved: regime_survival_curve.csv")
    print("\n" + "=" * 80)
    print("BLOCK SUMMARY")
    print("=" * 80)

    tail_risk = block_tail_risk(test_blocks)

    print(tail_risk)
    transition_rows, transition_summary = transition_event_study(
        data=data_test,
        state_feed=test_feed_val,
        horizons=(1, 5, 10, 20),
        price_col="FUTURES_CLOSE",
    )

    print("\n" + "=" * 80)
    print("TRANSITION EVENT STUDY")
    print("=" * 80)

    print(transition_summary)
    # ============================================================
    # EXTRA REGIME VALIDATION — FUTURES PROJECTED
    # ============================================================

    test_tail_stats = futures_tail_stats_by_state(
        data=data_test,
        state_feed=test_feed_val,
        horizons=(1, 5, 10, 20),
    )

    test_transition_stats = futures_transition_stats(
        data=data_test,
        state_feed=test_feed_val,
        horizon=10,
    )

    test_regime_blocks, test_survival_stats = regime_survival_stats(test_feed_val)

    test_tail_stats.to_csv("test_futures_tail_stats_by_state.csv", index=False)
    test_transition_stats.to_csv("test_futures_transition_stats.csv", index=False)
    test_regime_blocks.to_csv("test_regime_blocks.csv", index=False)
    test_survival_stats.to_csv("test_regime_survival_stats.csv", index=False)
    # ==========================================================
    #
    # ==========================================================
    # ==========================================================
    # HMM UNCERTAINTY / TRANSITION ANALYSIS
    # ==========================================================

    eps = 1e-12

    # Entropie
    p_cols = [c for c in test_feed_val.columns if c.startswith("p_state_")]

    test_feed_val["hmm_entropy"] = -(
        test_feed_val[p_cols] * np.log(test_feed_val[p_cols] + eps)
    ).sum(axis=1)

    test_feed_val["max_prob"] = test_feed_val[p_cols].max(axis=1)

    p_change_cols = []

    for c in p_cols:
        change_col = c + "_change"
        test_feed_val[change_col] = test_feed_val[c].diff().abs()
        p_change_cols.append(change_col)

    test_feed_val["prob_jump"] = test_feed_val[p_change_cols].max(axis=1)

    # Variation journalière des probabilités
    test_feed_val["p0_change"] = test_feed_val["p_state_0"].diff().abs()

    test_feed_val["p1_change"] = test_feed_val["p_state_1"].diff().abs()

    test_feed_val["prob_jump"] = test_feed_val[["p0_change", "p1_change"]].max(axis=1)

    print("\n" + "=" * 100)
    print("HMM CONFIDENCE DIAGNOSTICS")
    print("=" * 100)

    print("\nMAX PROBABILITY")
    print(test_feed_val["max_prob"].describe().round(6))

    print("\nENTROPY")
    print(test_feed_val["hmm_entropy"].describe().round(6))

    print("\nPROBABILITY JUMP")
    print(test_feed_val["prob_jump"].describe().round(6))
    # ==========================================================
    # FUTURES RETURNS VS HMM UNCERTAINTY
    # ==========================================================

    tmp = test_feed_val.copy()

    log_fut = np.log(data_test["FUTURES_CLOSE"])

    tmp["retF_fwd_20"] = log_fut.shift(-20) - log_fut

    tmp = tmp.dropna(subset=["retF_fwd_20"])

    tmp["entropy_bucket"] = pd.qcut(
        tmp["hmm_entropy"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]
    )

    print("\n" + "=" * 100)
    print("FORWARD RETURNS BY HMM ENTROPY")
    print("=" * 100)

    print(
        tmp.groupby("entropy_bucket")["retF_fwd_20"]
        .agg(["count", "mean", "median", "std"])
        .round(6)
    )
    tmp["state_entropy"] = (
        tmp["spot_state"].astype(str) + "_" + tmp["entropy_bucket"].astype(str)
    )

    print(tmp.groupby("state_entropy")["retF_fwd_20"].agg(["count", "mean", "median"]))
    entry_events, entry_profile, economic_labels = regime_entry_event_study(
        feat=feat_test,
        state_feed=test_feed_val,
        pre_window=10,
        post_window=10,
        min_events=3,
    )

    entry_events.to_csv("regime_entry_event_study_events.csv", index=False)
    entry_profile.to_csv("regime_entry_event_study_profile.csv", index=False)
    economic_labels.to_csv("regime_economic_labels.csv", index=False)
    # ============================================================
    # STATE 1 — HMM CERTAINTY / PERSISTENCE TEST
    # ============================================================

    state1_train_summary, state1_train_details = analyze_state1_hmm_certainty(
        data=data_train_fut,
        state_feed=train_feed_val,
        sample_name="TRAIN",
        state_value=1,
        price_col="FUTURES_CLOSE",
        horizons=(1, 3, 5, 10),
    )

    state1_test_summary, state1_test_details = analyze_state1_hmm_certainty(
        data=data_test,
        state_feed=test_feed_val,
        sample_name="TEST",
        state_value=1,
        price_col="FUTURES_CLOSE",
        horizons=(1, 3, 5, 10),
    )

    state1_train_summary.to_csv(
        "state1_hmm_certainty_train_summary.csv",
        index=False,
    )

    state1_test_summary.to_csv(
        "state1_hmm_certainty_test_summary.csv",
        index=False,
    )

    state1_train_details.to_csv(
        "state1_hmm_certainty_train_details.csv",
    )

    state1_test_details.to_csv(
        "state1_hmm_certainty_test_details.csv",
    )

    print("\nSaved:")
    print("state1_hmm_certainty_train_summary.csv")
    print("state1_hmm_certainty_test_summary.csv")


if __name__ == "__main__":
    main()
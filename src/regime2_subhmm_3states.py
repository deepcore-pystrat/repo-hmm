from __future__ import annotations

import argparse
import copy
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


FEATURE_COLUMNS: Tuple[str, ...] = (
    "fut_return_20",
    "fut_efficiency_20",
    "fut_abs_dd_60",
    "fut_volatility_20",
)


@dataclass(frozen=True)
class SubHMMConfig:
    n_states: int = 3
    target_regime: int = 2
    n_iter: int = 200
    n_starts: int = 10
    base_seed: int = 42
    cov_type: str = "diag"
    min_var: float = 1e-3
    sticky: float = 1.0
    estimate_nu: bool = True
    min_train_sequence_length: int = 5
    min_state_fraction: float = 0.03
    min_state_observations: int = 15
    price_column: str = "FUTURES_CLOSE"
    state_column: str = "spot_state"
    main_block_column: str = "hmm_block_id"


def normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        date_candidates = ["Date", "date", "Datetime", "datetime", "timestamp"]
        date_column = next((c for c in date_candidates if c in out.columns), None)

        if date_column is None:
            if len(out.columns) == 0:
                raise ValueError("Le tableau ne contient aucune colonne.")
            candidate = out.columns[0]
            parsed = pd.to_datetime(out[candidate], errors="coerce")
            if parsed.notna().mean() < 0.80:
                raise ValueError(
                    "Impossible d'identifier la colonne de date. "
                    f"Colonnes : {list(out.columns)}"
                )
            date_column = candidate

        out.index = pd.to_datetime(out.pop(date_column), errors="coerce")
    else:
        out.index = pd.to_datetime(out.index, errors="coerce")

    out = out.loc[out.index.notna()].sort_index()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.loc[~out.index.duplicated(keep="last")]


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        raw = pd.read_excel(path)
    else:
        raise ValueError(f"Format non supporté : {path.suffix}")

    raw.columns = [str(c).strip() for c in raw.columns]
    return normalize_datetime_index(raw)


def resolve_price_column(futures: pd.DataFrame, requested: str) -> str:
    if requested in futures.columns:
        return requested

    by_lower = {str(c).strip().lower(): str(c) for c in futures.columns}
    for candidate in [
        requested,
        "FUTURES_CLOSE",
        "Close",
        "Settlement",
        "Settle",
        "Price",
    ]:
        key = candidate.strip().lower()
        if key in by_lower:
            return by_lower[key]

    raise KeyError(
        "Colonne de prix introuvable. "
        f"Demandée={requested!r}; disponibles={list(futures.columns)}"
    )


def build_causal_futures_features(
    futures: pd.DataFrame,
    *,
    price_column: str,
) -> pd.DataFrame:
    """Features calculées sur tout le future, sans donnée future."""

    futures = normalize_datetime_index(futures)
    if price_column not in futures.columns:
        raise KeyError(f"Colonne absente : {price_column}")

    price = pd.to_numeric(futures[price_column], errors="coerce")
    price = price.where(price.gt(0))

    out = pd.DataFrame(index=futures.index)
    out[price_column] = price

    log_price = np.log(price)
    ret_1 = log_price.diff()

    out["fut_return_20"] = log_price - log_price.shift(20)

    path_20 = ret_1.abs().rolling(20, min_periods=15).sum()
    out["fut_efficiency_20"] = (
        out["fut_return_20"].abs() / (path_20 + 1e-12)
    ).clip(0.0, 1.0)

    rolling_high_60 = log_price.rolling(60, min_periods=40).max()
    out["fut_abs_dd_60"] = (rolling_high_60 - log_price).clip(lower=0.0)

    out["fut_volatility_20"] = (
        ret_1.rolling(20, min_periods=15).std(ddof=1) * np.sqrt(252.0)
    )

    return out.replace([np.inf, -np.inf], np.nan)


def project_main_feed_to_futures(
    *,
    market_features: pd.DataFrame,
    state_feed: pd.DataFrame,
    state_column: str,
    main_block_column: str,
) -> pd.DataFrame:
    """Projection causale du HMM principal sur le calendrier futures."""

    features = normalize_datetime_index(market_features)
    feed = normalize_datetime_index(state_feed)

    if state_column not in feed.columns:
        raise KeyError(
            f"Colonne absente : {state_column}. "
            f"Disponibles : {list(feed.columns)}"
        )

    projected = feed.reindex(features.index, method="ffill")
    outside = (projected.index < feed.index.min()) | (projected.index > feed.index.max())
    projected.loc[outside, :] = np.nan

    out = features.copy()
    out[state_column] = pd.to_numeric(projected[state_column], errors="coerce")

    inferred_block = out[state_column].ne(out[state_column].shift()).cumsum()

    if main_block_column in projected.columns:
        block = projected[main_block_column].astype("object")
    else:
        block = pd.Series(np.nan, index=out.index, dtype="object")

    missing = block.isna()
    block.loc[missing] = inferred_block.loc[missing].map(lambda x: f"AUTO_{int(x)}")
    out["main_regime_block_id"] = block

    return out.loc[
        (out.index >= feed.index.min()) & (out.index <= feed.index.max())
    ].copy()


def build_regime2_sequences(
    frame: pd.DataFrame,
    *,
    config: SubHMMConfig,
    training: bool,
) -> Tuple[List[np.ndarray], List[pd.Index], pd.DataFrame]:
    """Construit des séquences séparées bloc par bloc, sans concaténation."""

    work = frame.copy()
    work["_row_number"] = np.arange(len(work), dtype=int)

    valid = (
        work[config.state_column].eq(config.target_regime)
        & work[list(FEATURE_COLUMNS)].notna().all(axis=1)
    )
    regime2 = work.loc[valid].copy()

    if regime2.empty:
        raise ValueError("Aucune observation exploitable dans le régime 2.")

    new_main_block = regime2["main_regime_block_id"].ne(
        regime2["main_regime_block_id"].shift()
    )
    missing_observation_gap = regime2["_row_number"].diff().fillna(1).gt(1)
    regime2["sequence_id"] = (new_main_block | missing_observation_gap).cumsum().astype(int)

    arrays: List[np.ndarray] = []
    indexes: List[pd.Index] = []
    catalog: List[Dict[str, Any]] = []

    for sequence_id, group in regime2.groupby("sequence_id", sort=True):
        group = group.sort_index()
        used = (not training) or len(group) >= config.min_train_sequence_length

        catalog.append(
            {
                "sequence_id": int(sequence_id),
                "main_regime_block_id": str(group["main_regime_block_id"].iloc[0]),
                "start": group.index.min(),
                "end": group.index.max(),
                "length": int(len(group)),
                "used": bool(used),
            }
        )

        if not used:
            continue

        x = group[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError(f"NaN/inf dans la séquence {sequence_id}.")

        arrays.append(x)
        indexes.append(group.index)

    if not arrays:
        raise ValueError(
            "Aucune séquence suffisamment longue. "
            f"min_train_sequence_length={config.min_train_sequence_length}"
        )

    return arrays, indexes, pd.DataFrame(catalog)


def new_student_model(config: SubHMMConfig, seed: int):
    """Utilise directement le StudentTHMM déjà présent dans le repo."""

    from models import StudentTHMMConfig, create_hmm

    model_config = StudentTHMMConfig(
        K=config.n_states,
        n_iter=config.n_iter,
        seed=seed,
        cov_type=config.cov_type,
        min_var=config.min_var,
        sticky=config.sticky,
        estimate_nu=config.estimate_nu,
    )
    return create_hmm(model_config)


def capture_parameters(model) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for name in ["pi", "A", "means", "vars_", "covs_", "nus"]:
        if not hasattr(model, name):
            continue
        value = getattr(model, name)
        snapshot[name] = (
            None if value is None else np.array(value, copy=True)
            if isinstance(value, np.ndarray)
            else copy.deepcopy(value)
        )
    return snapshot


def restore_parameters(model, snapshot: Dict[str, Any]) -> None:
    for name, value in snapshot.items():
        setattr(
            model,
            name,
            np.array(value, copy=True) if isinstance(value, np.ndarray) else copy.deepcopy(value),
        )


def initialize_transitions_without_boundaries(model, sequences: Sequence[np.ndarray]) -> None:
    """Initialise pi/A sans fausse transition entre deux blocs distincts."""

    means = np.asarray(model.means, dtype=float)
    k = int(model.cfg.K)
    initial_counts = np.zeros(k)
    transition_counts = np.zeros((k, k))

    for x in sequences:
        distance = ((x[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distance, axis=1)
        initial_counts[int(labels[0])] += 1.0
        for left, right in zip(labels[:-1], labels[1:]):
            transition_counts[int(left), int(right)] += 1.0

    model.pi = initial_counts + 1e-2
    model.pi /= model.pi.sum()
    model.A = model._apply_transition_mask(
        transition_counts + 1e-2 + float(model.cfg.sticky) * np.eye(k)
    )


def fit_one_multisequence_start(
    model,
    sequences: Sequence[np.ndarray],
) -> Tuple[Any, float, int, bool, np.ndarray]:
    """Baum-Welch multi-séquences avec réinitialisation à chaque bloc."""

    sequences = [np.asarray(x, dtype=float) for x in sequences]
    x_all = np.vstack(sequences)

    model._init_params(x_all)
    model._init_emission_params(x_all)
    initialize_transitions_without_boundaries(model, sequences)

    ll_history: List[float] = []
    best_ll = -np.inf
    best_iteration = -1
    best_snapshot: Optional[Dict[str, Any]] = None
    converged = False

    for iteration in range(int(model.cfg.n_iter)):
        gammas: List[np.ndarray] = []
        pi_sum = np.zeros(model.cfg.K)
        xi_sum = np.zeros((model.cfg.K, model.cfg.K))
        total_ll = 0.0

        for x in sequences:
            log_b = model._compute_log_emissions(x)
            alpha, scaling, ll = model._forward_scaled(model.pi, model.A, log_b)
            beta = model._backward_scaled(model.A, log_b, scaling)

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
            gammas.append(gamma)
            pi_sum += gamma[0]
            total_ll += float(ll)

            if len(x) > 1:
                row_max = log_b.max(axis=1)
                b = np.exp(log_b - row_max[:, None])
                for t in range(len(x) - 1):
                    numer = (
                        alpha[t][:, None]
                        * model.A
                        * (b[t + 1] * beta[t + 1])[None, :]
                    )
                    denom = numer.sum()
                    if denom <= 0:
                        raise ValueError("Dénominateur xi nul.")
                    xi_sum += numer / denom

        ll_history.append(total_ll)

        if total_ll > best_ll:
            best_ll = float(total_ll)
            best_iteration = int(iteration)
            best_snapshot = capture_parameters(model)

        if len(ll_history) > 1 and abs(ll_history[-1] - ll_history[-2]) < model.cfg.tol:
            converged = True
            break

        model.pi = pi_sum + 1e-2
        model.pi /= model.pi.sum()
        model.A = model._apply_transition_mask(
            xi_sum + 1e-2 + float(model.cfg.sticky) * np.eye(model.cfg.K)
        )
        model._m_step_emissions(x_all, np.vstack(gammas))

    if best_snapshot is None:
        raise RuntimeError("Aucun snapshot EM valide.")

    restore_parameters(model, best_snapshot)
    return model, best_ll, best_iteration, converged, np.asarray(ll_history)


def occupancy(model, sequences: Sequence[np.ndarray]) -> Tuple[List[int], List[float]]:
    hard: List[np.ndarray] = []
    soft: List[np.ndarray] = []

    for x in sequences:
        states, gamma, _ = model.smooth(x)
        hard.append(states)
        soft.append(gamma)

    hard_all = np.concatenate(hard)
    soft_all = np.vstack(soft)

    counts = np.bincount(hard_all, minlength=model.cfg.K).astype(int)
    fractions = soft_all.sum(axis=0) / soft_all.shape[0]
    return counts.tolist(), fractions.astype(float).tolist()


def fit_multistart(
    scaled_sequences: Sequence[np.ndarray],
    config: SubHMMConfig,
) -> Tuple[Any, pd.DataFrame]:
    candidates: List[Tuple[Any, Dict[str, Any]]] = []

    for offset in range(config.n_starts):
        seed = config.base_seed + offset
        print(f"\n[SUB-HMM] start {offset + 1}/{config.n_starts}, seed={seed}")

        model = new_student_model(config, seed)
        model, best_ll, best_it, converged, ll_history = fit_one_multisequence_start(
            model,
            scaled_sequences,
        )
        counts, fractions = occupancy(model, scaled_sequences)

        nondegenerate = all(
            count >= config.min_state_observations for count in counts
        ) and all(
            fraction >= config.min_state_fraction for fraction in fractions
        )

        diagnostics = {
            "seed": seed,
            "best_loglik": best_ll,
            "best_iteration": best_it,
            "final_loglik": float(ll_history[-1]),
            "n_iterations": int(len(ll_history)),
            "converged": converged,
            "nondegenerate": nondegenerate,
            "hard_state_counts": counts,
            "soft_state_fractions": fractions,
        }
        candidates.append((model, diagnostics))

        print("  best LL       :", round(best_ll, 6))
        print("  hard counts   :", counts)
        print("  soft fractions:", np.round(fractions, 4).tolist())
        print("  nondegenerate :", nondegenerate)

    valid = [candidate for candidate in candidates if candidate[1]["nondegenerate"]]
    pool = valid if valid else candidates

    if not valid:
        print(
            "\nATTENTION : aucune initialisation ne passe tous les seuils "
            "d'occupation. Le meilleur LL est retenu, mais K=3 est fragile."
        )

    best_model, best_info = max(pool, key=lambda item: item[1]["best_loglik"])
    print(
        f"\n[SUB-HMM] modèle retenu : seed={best_info['seed']}, "
        f"LL={best_info['best_loglik']:.6f}"
    )

    table = pd.DataFrame([info for _, info in candidates])
    table = table.sort_values(
        ["nondegenerate", "best_loglik"],
        ascending=[False, False],
    )
    return best_model, table


def decode_split(
    frame: pd.DataFrame,
    *,
    model,
    scaler: RobustScaler,
    config: SubHMMConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Décodage forward causal, réinitialisé au début de chaque bloc."""

    out = frame.copy()
    out["regime2_substate_3"] = -1
    for state in range(config.n_states):
        out[f"p_regime2_substate_{state}"] = np.nan
    out["regime2_substate_confidence"] = np.nan
    out["regime2_substate_changed"] = 0
    out["regime2_substate_block_id"] = -1
    out["regime2_substate_age"] = np.nan

    sequences, indexes, catalog = build_regime2_sequences(
        frame,
        config=config,
        training=False,
    )

    global_block = 0
    summary_rows: List[Dict[str, Any]] = []

    for sequence_number, (x_raw, index) in enumerate(zip(sequences, indexes), start=1):
        x = scaler.transform(x_raw)
        states, probabilities, ll = model.filter(x)

        out.loc[index, "regime2_substate_3"] = states.astype(int)
        for state in range(config.n_states):
            out.loc[index, f"p_regime2_substate_{state}"] = probabilities[:, state]

        confidence = probabilities.max(axis=1)
        out.loc[index, "regime2_substate_confidence"] = confidence

        state_series = pd.Series(states, index=index, dtype=int)
        changed = state_series.ne(state_series.shift())
        changed.iloc[0] = False
        out.loc[index, "regime2_substate_changed"] = changed.astype(int)

        local_blocks = state_series.ne(state_series.shift()).cumsum()
        mapped: Dict[int, int] = {}
        for local_id in local_blocks.unique():
            global_block += 1
            mapped[int(local_id)] = global_block
        blocks = local_blocks.map(mapped).astype(int)
        out.loc[index, "regime2_substate_block_id"] = blocks
        out.loc[index, "regime2_substate_age"] = blocks.groupby(blocks).cumcount() + 1

        summary_rows.append(
            {
                "sequence_number": sequence_number,
                "start": index.min(),
                "end": index.max(),
                "length": len(index),
                "loglik": float(ll),
                "mean_confidence": float(confidence.mean()),
                "substate_changes": int(changed.sum()),
            }
        )

    out["regime2_substate_3"] = out["regime2_substate_3"].astype(int)
    out["regime2_substate_changed"] = out["regime2_substate_changed"].astype(int)
    out["regime2_substate_block_id"] = out["regime2_substate_block_id"].astype(int)

    return out, pd.DataFrame(summary_rows)


def state_profile(decoded: pd.DataFrame) -> pd.DataFrame:
    valid = decoded.loc[decoded["regime2_substate_3"].ge(0)].copy()
    if valid.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    total = len(valid)

    for state, group in valid.groupby("regime2_substate_3", sort=True):
        row: Dict[str, Any] = {
            "substate": int(state),
            "count": int(len(group)),
            "share": float(len(group) / total),
            "mean_confidence": float(group["regime2_substate_confidence"].mean()),
        }
        for feature in FEATURE_COLUMNS:
            row[f"mean_{feature}"] = float(group[feature].mean())
            row[f"median_{feature}"] = float(group[feature].median())
            row[f"std_{feature}"] = float(group[feature].std(ddof=1))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("substate")


def save_bundle(
    output_dir: Path,
    *,
    model,
    scaler: RobustScaler,
    config: SubHMMConfig,
    price_column: str,
) -> None:
    bundle = {
        "model": model,
        "scaler": scaler,
        "config": asdict(config),
        "feature_columns": list(FEATURE_COLUMNS),
        "price_column": price_column,
    }
    with (output_dir / "regime2_subhmm3_bundle.pkl").open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    parameters = {
        "pi": np.asarray(model.pi),
        "A": np.asarray(model.A),
        "means": np.asarray(model.means),
        "vars_": np.asarray(model.vars_),
    }
    if getattr(model, "covs_", None) is not None:
        parameters["covs_"] = np.asarray(model.covs_)
    if getattr(model, "nus", None) is not None:
        parameters["nus"] = np.asarray(model.nus)

    np.savez_compressed(output_dir / "regime2_subhmm3_parameters.npz", **parameters)

    metadata = {
        "config": asdict(config),
        "feature_columns": list(FEATURE_COLUMNS),
        "price_column": price_column,
    }
    (output_dir / "regime2_subhmm3_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_pipeline(
    *,
    futures_path: str | Path,
    train_feed_path: str | Path,
    test_feed_path: str | Path,
    output_dir: str | Path,
    config: SubHMMConfig,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    futures = load_table(futures_path)
    train_feed = load_table(train_feed_path)
    test_feed = load_table(test_feed_path)

    price_column = resolve_price_column(futures, config.price_column)
    features = build_causal_futures_features(futures, price_column=price_column)

    train_frame = project_main_feed_to_futures(
        market_features=features,
        state_feed=train_feed,
        state_column=config.state_column,
        main_block_column=config.main_block_column,
    )
    test_frame = project_main_feed_to_futures(
        market_features=features,
        state_feed=test_feed,
        state_column=config.state_column,
        main_block_column=config.main_block_column,
    )

    raw_train_sequences, _, training_catalog = build_regime2_sequences(
        train_frame,
        config=config,
        training=True,
    )

    scaler = RobustScaler().fit(np.vstack(raw_train_sequences))
    scaled_train_sequences = [scaler.transform(x) for x in raw_train_sequences]

    model, multistart_table = fit_multistart(scaled_train_sequences, config)

    train_decoded, train_sequence_summary = decode_split(
        train_frame,
        model=model,
        scaler=scaler,
        config=config,
    )
    test_decoded, test_sequence_summary = decode_split(
        test_frame,
        model=model,
        scaler=scaler,
        config=config,
    )

    train_profile = state_profile(train_decoded)
    test_profile = state_profile(test_decoded)

    train_decoded.to_csv(output_dir / "regime2_subhmm3_train.csv")
    test_decoded.to_csv(output_dir / "regime2_subhmm3_test.csv")
    train_profile.to_csv(output_dir / "regime2_subhmm3_train_profile.csv", index=False)
    test_profile.to_csv(output_dir / "regime2_subhmm3_test_profile.csv", index=False)
    training_catalog.to_csv(
        output_dir / "regime2_subhmm3_training_sequences.csv",
        index=False,
    )
    train_sequence_summary.to_csv(
        output_dir / "regime2_subhmm3_train_decode_sequences.csv",
        index=False,
    )
    test_sequence_summary.to_csv(
        output_dir / "regime2_subhmm3_test_decode_sequences.csv",
        index=False,
    )
    multistart_table.to_csv(
        output_dir / "regime2_subhmm3_multistart.csv",
        index=False,
    )

    save_bundle(
        output_dir,
        model=model,
        scaler=scaler,
        config=config,
        price_column=price_column,
    )

    print("\n" + "=" * 100)
    print("RÉGIME 2 — STUDENT-t SUB-HMM, 3 ÉTATS ANONYMES")
    print("=" * 100)
    print("Features :", list(FEATURE_COLUMNS))
    print("Séquences TRAIN :", len(raw_train_sequences))
    print("Observations TRAIN :", sum(len(x) for x in raw_train_sequences))
    print("\nProfil TRAIN :")
    print(train_profile.round(6).to_string(index=False))
    print("\nProfil TEST :")
    print(test_profile.round(6).to_string(index=False))
    print("\nMatrice de transition apprise :")
    print(pd.DataFrame(model.A).round(6).to_string(index=False, header=False))
    print("\nSorties :", output_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Student-t HMM à trois sous-états anonymes, entraîné uniquement "
            "dans les blocs du régime principal 2."
        )
    )
    parser.add_argument(
        "--futures",
        default=r"D:\Downloads\data_futures_corn (2).xlsx",
    )
    parser.add_argument("--train-feed", default="spot_hmm_state_feed_train.csv")
    parser.add_argument("--test-feed", default="spot_hmm_state_feed_test.csv")
    parser.add_argument("--output-dir", default="outputs/regime2_subhmm3")
    parser.add_argument("--price-column", default="FUTURES_CLOSE")
    parser.add_argument("--state-column", default="spot_state")
    parser.add_argument("--main-block-column", default="hmm_block_id")
    parser.add_argument("--n-starts", type=int, default=10)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sticky", type=float, default=1.0)
    parser.add_argument("--min-train-sequence-length", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SubHMMConfig(
        n_starts=args.n_starts,
        n_iter=args.n_iter,
        base_seed=args.seed,
        sticky=args.sticky,
        min_train_sequence_length=args.min_train_sequence_length,
        price_column=args.price_column,
        state_column=args.state_column,
        main_block_column=args.main_block_column,
    )
    run_pipeline(
        futures_path=args.futures,
        train_feed_path=args.train_feed,
        test_feed_path=args.test_feed,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
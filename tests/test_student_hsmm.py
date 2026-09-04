from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


# ============================================================
# IMPORT DU DOSSIER src/
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from models.student_hsmm import (  # noqa: E402
    StudentTHSMM,
    StudentTHSMMConfig,
)


# ============================================================
# HELPERS
# ============================================================

def make_config(
    n_iter: int = 6,
    max_duration: int = 40,
) -> StudentTHSMMConfig:
    return StudentTHSMMConfig(
        K=3,
        n_iter=n_iter,
        seed=42,
        min_var=1e-4,
        tol=1e-4,
        cov_type="diag",
        sticky=0.5,
        estimate_nu=True,
        fixed_nu=6.0,
        min_duration=1,
        max_duration=max_duration,
        min_blocks_per_state=2,
        duration_state_path="filter",
    )


def make_repeated_regime_data() -> np.ndarray:
    """
    Données synthétiques avec plusieurs passages répétés
    dans chacun des trois régimes.
    """

    rng = np.random.default_rng(42)

    chunks: list[np.ndarray] = []

    for _ in range(5):
        chunks.append(
            rng.normal(
                loc=(-4.0, -4.0),
                scale=0.30,
                size=(12, 2),
            )
        )

        chunks.append(
            rng.normal(
                loc=(0.0, 0.0),
                scale=0.45,
                size=(10, 2),
            )
        )

        chunks.append(
            rng.normal(
                loc=(4.0, 4.0),
                scale=0.35,
                size=(14, 2),
            )
        )

    return np.vstack(chunks)


@pytest.fixture(scope="module")
def fitted_model_and_data() -> tuple[
    StudentTHSMM,
    np.ndarray,
]:
    X = make_repeated_regime_data()

    model = StudentTHSMM(
        make_config()
    )

    model.fit(X)

    return model, X


# ============================================================
# TEST 1 — VALIDATION DE LA CONFIGURATION
# ============================================================

def test_invalid_student_hsmm_config_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="max_duration",
    ):
        StudentTHSMMConfig(
            max_duration=1,
        )

    with pytest.raises(
        ValueError,
        match="duration_state_path",
    ):
        StudentTHSMMConfig(
            duration_state_path="invalid",
        )

    with pytest.raises(
        ValueError,
        match="min_blocks_per_state",
    ):
        StudentTHSMMConfig(
            min_blocks_per_state=0,
        )


# ============================================================
# TEST 2 — EXTRACTION DES BLOCS
# ============================================================

def test_extract_state_durations() -> None:
    states = np.array(
        [
            0, 0,
            1, 1, 1,
            0, 0,
            2, 2, 2, 2,
            1, 1,
        ],
        dtype=int,
    )

    (
        complete_durations,
        all_durations,
        truncation_counts,
    ) = StudentTHSMM._extract_state_durations(
        states=states,
        n_states=3,
        max_duration=10,
    )

    # Tous les blocs, y compris les deux frontières.
    assert all_durations == {
        0: [2, 2],
        1: [3, 2],
        2: [4],
    }

    # Le premier bloc de l'état 0 et le dernier bloc
    # de l'état 1 sont considérés comme censurés.
    assert complete_durations == {
        0: [2],
        1: [3],
        2: [4],
    }

    assert truncation_counts == {
        0: 0,
        1: 0,
        2: 0,
    }


# ============================================================
# TEST 3 — TRONCATURE À MAX_DURATION
# ============================================================

def test_duration_extraction_applies_max_duration() -> None:
    states = np.array(
        [0] * 12
        + [1] * 4
        + [2] * 3,
        dtype=int,
    )

    (
        _complete_durations,
        all_durations,
        truncation_counts,
    ) = StudentTHSMM._extract_state_durations(
        states=states,
        n_states=3,
        max_duration=5,
    )

    assert all_durations[0] == [5]
    assert truncation_counts[0] == 1

    assert all_durations[1] == [4]
    assert truncation_counts[1] == 0

    assert all_durations[2] == [3]
    assert truncation_counts[2] == 0


# ============================================================
# TEST 4 — ESTIMATION EMPIRIQUE DES DURÉES
# ============================================================

def test_initialize_duration_layer_from_repeated_blocks() -> None:
    config = StudentTHSMMConfig(
        K=3,
        n_iter=2,
        seed=42,
        max_duration=20,
        min_blocks_per_state=2,
    )

    model = StudentTHSMM(config)

    # Paramètres HMM nécessaires pour initialiser
    # la couche HSMM.
    model.pi = np.array(
        [0.4, 0.35, 0.25],
        dtype=float,
    )

    model.A = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.15, 0.80],
        ],
        dtype=float,
    )

    blocks = [
        (0, 2),
        (1, 2),
        (2, 3),
        (0, 2),
        (1, 12),
        (2, 9),
        (0, 10),
        (1, 3),
        (2, 2),
        (0, 3),
        (1, 8),
        (2, 12),
    ]

    states = np.concatenate(
        [
            np.full(
                duration,
                state,
                dtype=int,
            )
            for state, duration in blocks
        ]
    )

    model._initialize_hsmm_layer(
        states
    )

    assert model.duration_models is not None
    assert len(model.duration_models) == 3

    assert model.hsmm_engine is not None

    assert (
        model.segment_transition_matrix
        is not None
    )

    np.testing.assert_allclose(
        np.diag(
            model.segment_transition_matrix
        ),
        np.zeros(3),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        model.segment_transition_matrix.sum(
            axis=1
        ),
        np.ones(3),
        atol=1e-12,
    )

    summary = model.get_duration_summary()

    assert len(summary) == 3

    for row in summary:
        assert (
            row["estimation_source"]
            == "complete_internal_blocks"
        )

        assert row["n_selected_blocks"] >= 2

        assert (
            1.0
            <= row["expected_duration"]
            <= config.max_duration
        )

        assert row["shape"] > 0
        assert 0 < row["probability"] <= 1


# ============================================================
# TEST 5 — ENTRAÎNEMENT COMPLET
# ============================================================

def test_fit_initializes_student_hsmm(
    fitted_model_and_data: tuple[
        StudentTHSMM,
        np.ndarray,
    ],
) -> None:
    model, X = fitted_model_and_data

    assert model.means is not None
    assert model.vars_ is not None
    assert model.nus is not None
    assert model.pi is not None
    assert model.A is not None

    assert model.duration_models is not None
    assert len(model.duration_models) == 3

    assert model.hsmm_engine is not None

    assert (
        model.segment_transition_matrix
        is not None
    )

    assert model.means.shape == (
        3,
        X.shape[1],
    )

    assert model.vars_.shape == (
        3,
        X.shape[1],
    )

    assert model.nus.shape == (3,)


# ============================================================
# TEST 6 — FILTRAGE CAUSAL
# ============================================================

def test_filter_full_returns_valid_result(
    fitted_model_and_data: tuple[
        StudentTHSMM,
        np.ndarray,
    ],
) -> None:
    model, X = fitted_model_and_data

    result = model.filter_full(X)

    number_of_observations = X.shape[0]

    assert result.states.shape == (
        number_of_observations,
    )

    assert result.state_probabilities.shape == (
        number_of_observations,
        3,
    )

    assert result.age_state_probabilities.shape == (
        number_of_observations,
        3,
        model.cfg.max_duration,
    )

    np.testing.assert_allclose(
        result.state_probabilities.sum(
            axis=1
        ),
        np.ones(number_of_observations),
        atol=1e-10,
    )

    np.testing.assert_allclose(
        result.age_state_probabilities.sum(
            axis=(1, 2)
        ),
        np.ones(number_of_observations),
        atol=1e-10,
    )

    np.testing.assert_allclose(
        result.last_context
        .age_state_posterior
        .sum(),
        1.0,
        atol=1e-12,
    )

    assert np.all(
        result.expected_age >= 1.0
    )

    assert np.all(
        result.expected_age
        <= model.cfg.max_duration
    )

    assert np.all(
        result.exit_probability >= 0.0
    )

    assert np.all(
        result.exit_probability <= 1.0
    )


# ============================================================
# TEST 7 — CONTINUITÉ EXACTE ENTRE DEUX BLOCS
# ============================================================

def test_context_split_matches_full_filtering(
    fitted_model_and_data: tuple[
        StudentTHSMM,
        np.ndarray,
    ],
) -> None:
    model, X = fitted_model_and_data

    split_index = X.shape[0] // 2

    full_result = model.filter_full(
        X
    )

    first_result = model.filter_full(
        X[:split_index]
    )

    second_result = model.filter_full(
        X[split_index:],
        initial_context=(
            first_result.last_context
        ),
    )

    combined_state_probabilities = (
        np.vstack(
            [
                first_result.state_probabilities,
                second_result.state_probabilities,
            ]
        )
    )

    combined_age_probabilities = (
        np.concatenate(
            [
                first_result.age_state_probabilities,
                second_result.age_state_probabilities,
            ],
            axis=0,
        )
    )

    combined_expected_age = np.concatenate(
        [
            first_result.expected_age,
            second_result.expected_age,
        ]
    )

    combined_exit_probability = np.concatenate(
        [
            first_result.exit_probability,
            second_result.exit_probability,
        ]
    )

    np.testing.assert_allclose(
        combined_state_probabilities,
        full_result.state_probabilities,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        combined_age_probabilities,
        full_result.age_state_probabilities,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        combined_expected_age,
        full_result.expected_age,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        combined_exit_probability,
        full_result.exit_probability,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        (
            first_result.log_likelihood
            + second_result.log_likelihood
        ),
        full_result.log_likelihood,
        atol=1e-9,
    )


# ============================================================
# TEST 8 — MODÈLE NON ENTRAÎNÉ
# ============================================================

def test_filter_before_fit_is_rejected() -> None:
    model = StudentTHSMM(
        make_config()
    )

    X = np.zeros(
        (10, 2),
        dtype=float,
    )

    with pytest.raises(
        RuntimeError,
        match="fit",
    ):
        model.filter_full(X)
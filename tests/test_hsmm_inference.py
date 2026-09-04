from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pytest


# ============================================================
# IMPORTS DEPUIS src/
# ============================================================

SRC_DIR = Path(__file__).resolve().parents[1] / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from models.hsmm_duration import (  # noqa: E402
    NegativeBinomialDuration,
    NegativeBinomialDurationConfig,
)
from models.hsmm_inference import (  # noqa: E402
    ExplicitDurationHSMMFilter,
    HSMMFilterContext,
    embedded_transition_matrix_from_hmm,
)


# ============================================================
# HELPERS
# ============================================================

def make_duration_models(
    max_duration: int = 12,
) -> list[NegativeBinomialDuration]:
    """
    Construit trois distributions de durée différentes.
    """

    config = NegativeBinomialDurationConfig(
        min_duration=1,
        max_duration=max_duration,
    )

    return [
        NegativeBinomialDuration(
            shape=3.0,
            probability=0.35,
            config=config,
        ),
        NegativeBinomialDuration(
            shape=5.0,
            probability=0.40,
            config=config,
        ),
        NegativeBinomialDuration(
            shape=2.5,
            probability=0.25,
            config=config,
        ),
    ]


def make_embedded_transition_matrix() -> np.ndarray:
    """
    Matrice de transition entre segments.

    La diagonale est nulle car la persistance est contrôlée
    par les distributions de durée.
    """

    return np.array(
        [
            [0.0, 0.60, 0.40],
            [0.45, 0.0, 0.55],
            [0.30, 0.70, 0.0],
        ],
        dtype=float,
    )


def make_filter(
    max_duration: int = 12,
) -> ExplicitDurationHSMMFilter:
    return ExplicitDurationHSMMFilter(
        duration_models=make_duration_models(
            max_duration=max_duration,
        ),
        transition_matrix=make_embedded_transition_matrix(),
        initial_state_probabilities=np.array(
            [0.50, 0.30, 0.20],
            dtype=float,
        ),
    )


# ============================================================
# TEST 1 — MATRICE HMM VERS MATRICE HSMM
# ============================================================

def test_embedded_transition_matrix_from_hmm() -> None:
    hmm_matrix = np.array(
        [
            [0.80, 0.10, 0.10],
            [0.20, 0.70, 0.10],
            [0.25, 0.25, 0.50],
        ],
        dtype=float,
    )

    embedded = embedded_transition_matrix_from_hmm(
        hmm_matrix
    )

    expected = np.array(
        [
            [0.0, 0.50, 0.50],
            [2.0 / 3.0, 0.0, 1.0 / 3.0],
            [0.50, 0.50, 0.0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        embedded,
        expected,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        embedded.sum(axis=1),
        np.ones(3),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.diag(embedded),
        np.zeros(3),
        atol=1e-12,
    )


# ============================================================
# TEST 2 — NORMALISATION DU CONTEXTE
# ============================================================

def test_filter_context_is_normalized() -> None:
    posterior = np.zeros(
        (3, 10),
        dtype=float,
    )

    posterior[0, 0] = 2.0
    posterior[1, 3] = 3.0
    posterior[2, 5] = 5.0

    context = HSMMFilterContext(
        age_state_posterior=posterior
    )

    np.testing.assert_allclose(
        context.age_state_posterior.sum(),
        1.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        context.age_state_posterior[0, 0],
        0.20,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        context.age_state_posterior[1, 3],
        0.30,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        context.age_state_posterior[2, 5],
        0.50,
        atol=1e-12,
    )


# ============================================================
# TEST 3 — PROBABILITÉS VALIDES
# ============================================================

def test_filter_returns_valid_probabilities() -> None:
    hsmm_filter = make_filter()

    emission_probabilities = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.75, 0.20, 0.05],
            [0.65, 0.25, 0.10],
            [0.15, 0.75, 0.10],
            [0.10, 0.80, 0.10],
            [0.10, 0.65, 0.25],
            [0.10, 0.20, 0.70],
            [0.05, 0.15, 0.80],
        ],
        dtype=float,
    )

    emission_log_likelihood = np.log(
        emission_probabilities
    )

    result = hsmm_filter.filter(
        emission_log_likelihood
    )

    assert result.states.shape == (8,)
    assert result.state_probabilities.shape == (8, 3)
    assert result.age_state_probabilities.shape == (
        8,
        3,
        12,
    )

    assert np.all(
        np.isfinite(result.state_probabilities)
    )

    assert np.all(
        result.state_probabilities >= 0
    )

    np.testing.assert_allclose(
        result.state_probabilities.sum(axis=1),
        np.ones(8),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.age_state_probabilities.sum(
            axis=(1, 2)
        ),
        np.ones(8),
        atol=1e-12,
    )

    expected_states = np.argmax(
        result.state_probabilities,
        axis=1,
    )

    np.testing.assert_array_equal(
        result.states,
        expected_states,
    )

    assert np.isfinite(result.log_likelihood)

    np.testing.assert_allclose(
        result.last_context.age_state_posterior.sum(),
        1.0,
        atol=1e-12,
    )


# ============================================================
# TEST 4 — CONTINUATION ET SORTIE D'UN ÉTAT
# ============================================================

def test_prediction_from_concentrated_age_context() -> None:
    hsmm_filter = make_filter(
        max_duration=10
    )

    context_posterior = np.zeros(
        (3, 10),
        dtype=float,
    )

    # État 0, âge 1 avec probabilité 1.
    context_posterior[0, 0] = 1.0

    context = HSMMFilterContext(
        age_state_posterior=context_posterior
    )

    # Même vraisemblance dans les trois états :
    # le posterior doit être égal à la prédiction temporelle.
    neutral_emissions = np.zeros(
        (1, 3),
        dtype=float,
    )

    result = hsmm_filter.filter(
        neutral_emissions,
        initial_context=context,
    )

    posterior = result.age_state_probabilities[0]

    duration_state_0 = hsmm_filter.duration_models[0]

    exit_probability = (
        duration_state_0.exit_probability(age=1)
    )

    continuation_probability = (
        1.0 - exit_probability
    )

    transition_matrix = (
        hsmm_filter.transition_matrix
    )

    # Continuation de l'état 0 :
    # âge 1 -> âge 2
    np.testing.assert_allclose(
        posterior[0, 1],
        continuation_probability,
        atol=1e-12,
    )

    # Sortie de l'état 0 vers l'état 1 à l'âge 1.
    np.testing.assert_allclose(
        posterior[1, 0],
        (
            exit_probability
            * transition_matrix[0, 1]
        ),
        atol=1e-12,
    )

    # Sortie de l'état 0 vers l'état 2 à l'âge 1.
    np.testing.assert_allclose(
        posterior[2, 0],
        (
            exit_probability
            * transition_matrix[0, 2]
        ),
        atol=1e-12,
    )

    # Aucune auto-transition de segment vers l'âge 1.
    np.testing.assert_allclose(
        posterior[0, 0],
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        posterior.sum(),
        1.0,
        atol=1e-12,
    )


# ============================================================
# TEST 5 — SORTIE OBLIGATOIRE À LA DURÉE MAXIMALE
# ============================================================

def test_regime_must_exit_at_maximum_duration() -> None:
    max_duration = 8

    hsmm_filter = make_filter(
        max_duration=max_duration
    )

    context_posterior = np.zeros(
        (3, max_duration),
        dtype=float,
    )

    # L'état 1 a atteint la durée maximale.
    context_posterior[
        1,
        max_duration - 1,
    ] = 1.0

    context = HSMMFilterContext(
        age_state_posterior=context_posterior
    )

    result = hsmm_filter.filter(
        emission_log_likelihood=np.zeros(
            (1, 3),
            dtype=float,
        ),
        initial_context=context,
    )

    posterior = result.age_state_probabilities[0]

    transition_matrix = (
        hsmm_filter.transition_matrix
    )

    # L'état 1 ne peut pas continuer.
    np.testing.assert_allclose(
        posterior[1].sum(),
        0.0,
        atol=1e-12,
    )

    # La masse repart vers les autres états à l'âge 1.
    np.testing.assert_allclose(
        posterior[0, 0],
        transition_matrix[1, 0],
        atol=1e-12,
    )

    np.testing.assert_allclose(
        posterior[2, 0],
        transition_matrix[1, 2],
        atol=1e-12,
    )

    np.testing.assert_allclose(
        posterior.sum(),
        1.0,
        atol=1e-12,
    )


# ============================================================
# TEST 6 — ABSENCE DE LOOK-AHEAD
# ============================================================

def test_filter_is_causal_without_lookahead() -> None:
    hsmm_filter = make_filter()

    rng = np.random.default_rng(42)

    emissions = rng.normal(
        loc=-1.0,
        scale=0.5,
        size=(15, 3),
    )

    full_result = hsmm_filter.filter(
        emissions
    )

    prefix_result = hsmm_filter.filter(
        emissions[:7]
    )

    # Les résultats des sept premières dates doivent être
    # strictement identiques avec ou sans observations futures.
    np.testing.assert_allclose(
        full_result.state_probabilities[:7],
        prefix_result.state_probabilities,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        full_result.age_state_probabilities[:7],
        prefix_result.age_state_probabilities,
        atol=1e-12,
    )

    np.testing.assert_array_equal(
        full_result.states[:7],
        prefix_result.states,
    )


# ============================================================
# TEST 7 — CONTINUITÉ TRAIN / TEST
# ============================================================

def test_train_test_context_matches_full_filtering() -> None:
    hsmm_filter = make_filter()

    rng = np.random.default_rng(123)

    emissions = rng.normal(
        loc=-1.5,
        scale=0.7,
        size=(20, 3),
    )

    split_index = 11

    full_result = hsmm_filter.filter(
        emissions
    )

    train_result = hsmm_filter.filter(
        emissions[:split_index]
    )

    test_result = hsmm_filter.filter(
        emissions[split_index:],
        initial_context=train_result.last_context,
    )

    np.testing.assert_allclose(
        test_result.state_probabilities,
        full_result.state_probabilities[
            split_index:
        ],
        atol=1e-12,
    )

    np.testing.assert_allclose(
        test_result.age_state_probabilities,
        full_result.age_state_probabilities[
            split_index:
        ],
        atol=1e-12,
    )

    np.testing.assert_allclose(
        test_result.expected_age,
        full_result.expected_age[
            split_index:
        ],
        atol=1e-12,
    )

    np.testing.assert_allclose(
        test_result.exit_probability,
        full_result.exit_probability[
            split_index:
        ],
        atol=1e-12,
    )

    np.testing.assert_allclose(
        (
            train_result.log_likelihood
            + test_result.log_likelihood
        ),
        full_result.log_likelihood,
        atol=1e-12,
    )


# ============================================================
# TEST 8 — DIAGNOSTICS DE DURÉE
# ============================================================

def test_duration_diagnostics_are_valid() -> None:
    max_duration = 15

    hsmm_filter = make_filter(
        max_duration=max_duration
    )

    emissions = np.array(
        [
            [-0.2, -2.0, -3.0],
            [-0.3, -1.8, -2.5],
            [-0.4, -1.5, -2.0],
            [-1.5, -0.3, -2.0],
            [-2.0, -0.2, -1.5],
            [-2.5, -1.0, -0.2],
        ],
        dtype=float,
    )

    result = hsmm_filter.filter(
        emissions
    )

    assert np.all(
        np.isfinite(result.expected_age)
    )

    assert np.all(
        result.expected_age >= 1.0
    )

    assert np.all(
        result.expected_age <= max_duration
    )

    assert np.all(
        np.isfinite(result.exit_probability)
    )

    assert np.all(
        result.exit_probability >= 0.0
    )

    assert np.all(
        result.exit_probability <= 1.0
    )

    assert np.all(
        np.isfinite(
            result.expected_remaining_duration
        )
    )

    assert np.all(
        result.expected_remaining_duration >= 0.0
    )

    assert np.all(
        result.expected_remaining_duration
        <= max_duration - 1
    )


# ============================================================
# TEST 9 — MATRICE HSMM INVALIDE
# ============================================================

def test_transition_matrix_with_diagonal_is_rejected() -> None:
    invalid_matrix = np.array(
        [
            [0.20, 0.50, 0.30],
            [0.40, 0.10, 0.50],
            [0.30, 0.60, 0.10],
        ],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="diagonale",
    ):
        ExplicitDurationHSMMFilter(
            duration_models=make_duration_models(),
            transition_matrix=invalid_matrix,
        )


# ============================================================
# TEST 10 — ÉMISSIONS INVALIDES
# ============================================================

def test_nan_emission_is_rejected() -> None:
    hsmm_filter = make_filter()

    emissions = np.array(
        [
            [-0.2, -1.0, -2.0],
            [-0.3, np.nan, -1.5],
        ],
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="NaN",
    ):
        hsmm_filter.filter(
            emissions
        )
from pathlib import Path
import sys

import numpy as np
import pytest


# Permet aux tests d'importer src/models
SRC_DIR = Path(__file__).resolve().parents[1] / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from models.hsmm_duration import (  # noqa: E402
    NegativeBinomialDuration,
    NegativeBinomialDurationConfig,
)


def test_default_configuration_is_valid() -> None:
    config = NegativeBinomialDurationConfig()

    assert config.min_duration == 1
    assert config.max_duration == 252
    assert config.min_shape > 0
    assert config.max_shape > config.min_shape
    assert 0 < config.min_probability < config.max_probability < 1


def test_support_is_correct() -> None:
    config = NegativeBinomialDurationConfig(
        min_duration=2,
        max_duration=10,
    )

    model = NegativeBinomialDuration(
        shape=3.0,
        probability=0.25,
        config=config,
    )

    expected_support = np.arange(2, 11)

    np.testing.assert_array_equal(
        model.support,
        expected_support,
    )


def test_pmf_is_a_valid_probability_distribution() -> None:
    model = NegativeBinomialDuration(
        shape=4.0,
        probability=0.30,
    )

    pmf = model.pmf()

    assert pmf.ndim == 1
    assert len(pmf) == 252
    assert np.all(np.isfinite(pmf))
    assert np.all(pmf >= 0)

    np.testing.assert_allclose(
        pmf.sum(),
        1.0,
        atol=1e-12,
    )


def test_survival_function_is_non_increasing() -> None:
    model = NegativeBinomialDuration(
        shape=5.0,
        probability=0.20,
    )

    survival = model.survival()

    assert np.all(np.isfinite(survival))
    assert np.all(survival > 0)
    assert np.all(survival <= 1)

    # S(min_duration) doit être égal à 1.
    np.testing.assert_allclose(
        survival[0],
        1.0,
        atol=1e-12,
    )

    # Une fonction de survie ne peut pas augmenter avec la durée.
    differences = np.diff(survival)

    assert np.all(differences <= 1e-12)


def test_hazard_is_consistent_with_pmf_and_survival() -> None:
    model = NegativeBinomialDuration(
        shape=2.5,
        probability=0.15,
    )

    pmf = model.pmf()
    survival = model.survival()
    hazard = model.hazard()

    assert np.all(np.isfinite(hazard))
    assert np.all(hazard >= 0)
    assert np.all(hazard <= 1)

    np.testing.assert_allclose(
        hazard[:-1],
        pmf[:-1] / survival[:-1],
        atol=1e-12,
    )

    # À la durée maximale, le régime doit obligatoirement sortir.
    np.testing.assert_allclose(
        hazard[-1],
        1.0,
        atol=1e-12,
    )


def test_expected_duration_is_inside_support() -> None:
    config = NegativeBinomialDurationConfig(
        min_duration=1,
        max_duration=120,
    )

    model = NegativeBinomialDuration(
        shape=3.0,
        probability=0.20,
        config=config,
    )

    expected_duration = model.expected_duration()
    variance_duration = model.variance_duration()

    assert np.isfinite(expected_duration)
    assert config.min_duration <= expected_duration <= config.max_duration

    assert np.isfinite(variance_duration)
    assert variance_duration >= 0


def test_expected_remaining_duration_is_valid() -> None:
    config = NegativeBinomialDurationConfig(
        min_duration=1,
        max_duration=100,
    )

    model = NegativeBinomialDuration(
        shape=4.0,
        probability=0.25,
        config=config,
    )

    remaining_at_1 = model.expected_remaining_duration(age=1)
    remaining_at_50 = model.expected_remaining_duration(age=50)
    remaining_at_100 = model.expected_remaining_duration(age=100)

    assert np.isfinite(remaining_at_1)
    assert np.isfinite(remaining_at_50)
    assert np.isfinite(remaining_at_100)

    assert remaining_at_1 >= 0
    assert remaining_at_50 >= 0

    np.testing.assert_allclose(
        remaining_at_100,
        0.0,
        atol=1e-12,
    )


def test_fit_from_empirical_durations() -> None:
    durations = np.array(
        [
            5,
            7,
            8,
            10,
            12,
            15,
            18,
            20,
            25,
            30,
            40,
            60,
        ],
        dtype=float,
    )

    config = NegativeBinomialDurationConfig(
        min_duration=1,
        max_duration=120,
    )

    model = NegativeBinomialDuration.fit_from_durations(
        durations=durations,
        config=config,
    )

    summary = model.summary()

    assert np.isfinite(model.shape)
    assert np.isfinite(model.probability)

    assert config.min_shape <= model.shape <= config.max_shape

    assert (
        config.min_probability
        <= model.probability
        <= config.max_probability
    )

    assert (
        config.min_duration
        <= summary["expected_duration"]
        <= config.max_duration
    )

    np.testing.assert_allclose(
        model.pmf().sum(),
        1.0,
        atol=1e-12,
    )


def test_invalid_empirical_durations_raise_error() -> None:
    config = NegativeBinomialDurationConfig(
        min_duration=1,
        max_duration=100,
    )

    durations = np.array(
        [
            np.nan,
            np.inf,
            -5,
            0,
            150,
        ]
    )

    with pytest.raises(
        ValueError,
        match="Aucune durée valide",
    ):
        NegativeBinomialDuration.fit_from_durations(
            durations=durations,
            config=config,
        )


def test_duration_outside_support_raises_error() -> None:
    config = NegativeBinomialDurationConfig(
        min_duration=2,
        max_duration=20,
    )

    model = NegativeBinomialDuration(
        shape=3.0,
        probability=0.30,
        config=config,
    )

    with pytest.raises(ValueError):
        model.exit_probability(age=1)

    with pytest.raises(ValueError):
        model.exit_probability(age=21)
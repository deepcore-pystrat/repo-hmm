"""
Data structures and configuration for the optimal-switching layer.

This module contains no estimation logic.
It only defines the objects exchanged by the stochastic-process
estimator and, later, by the optimal-switching solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StochasticProcessConfig:
    """
    Configuration of the regime-switching diffusion estimator.

    We model the traded futures asset

        X_t = log(FUTURES_CLOSE_t)

    while the discrete regime alpha_t is provided by the
    spot-market HMM through spot_state.

    and approximate, conditionally on regime p,

        dX_t = mu_p dt + sigma_p dB_t.

    For the first implementation, one observation interval is treated
    as one unit of time (typically one trading day).

    Parameters
    ----------
    price_col:
        Spot price column used to construct X_t.

    state_col:
        HMM regime column.

    min_obs_per_regime:
        Minimum number of increments required to estimate a regime.

    use_lagged_state:
        If True, the price increment from t-1 to t is attributed to
        the regime observed at t-1.

        This is the recommended causal convention for trading:
        the regime used to describe the next increment was already
        known at the beginning of the interval.

    dt:
        Time step between two observations.
        dt=1.0 means parameters are expressed per observation/day.

    generator_method:
        Method used to map the discrete HMM transition matrix P=model.A
        to a continuous-time generator Q.

        "first_order":
            Q = (P - I) / dt

        This is a first-order approximation but always preserves
        non-negative off-diagonal intensities if P is a valid
        transition matrix.
    """

    price_col: str = "FUTURES_CLOSE"
    state_col: str = "spot_state"

    min_obs_per_regime: int = 30

    use_lagged_state: bool = True

    dt: float = 1.0

    generator_method: str = "first_order"

    eps: float = 1e-12


@dataclass
class StochasticProcessEstimate:
    """
    Fitted stochastic-process parameters.

    Attributes
    ----------
    regime_params:
        DataFrame indexed by HMM state.

        Expected columns:
            mu
            sigma
            variance
            n_obs
            mean_dx
            std_dx

    generator:
        Continuous-time generator approximation Q.

    transition_matrix:
        Original discrete HMM transition matrix P=model.A.

    n_states:
        Number of HMM regimes.

    dt:
        Observation time step.
    """

    regime_params: pd.DataFrame

    generator: np.ndarray

    transition_matrix: np.ndarray

    n_states: int

    dt: float

    price_col: str = "MID"

    state_col: str = "spot_state"

    def validate(self) -> None:
        """Validate the fitted object."""

        if self.regime_params.empty:
            raise ValueError(
                "regime_params is empty."
            )

        required = {
            "mu",
            "sigma",
            "variance",
            "n_obs",
        }

        missing = required.difference(
            self.regime_params.columns
        )

        if missing:
            raise ValueError(
                f"Missing regime parameter columns: {sorted(missing)}"
            )

        if self.generator.shape != (
            self.n_states,
            self.n_states,
        ):
            raise ValueError(
                "Invalid generator shape: "
                f"{self.generator.shape}. "
                f"Expected {(self.n_states, self.n_states)}."
            )

        if self.transition_matrix.shape != (
            self.n_states,
            self.n_states,
        ):
            raise ValueError(
                "Invalid transition matrix shape."
            )

        if not np.isfinite(
            self.generator
        ).all():
            raise ValueError(
                "Generator contains NaN or Inf."
            )

        if not np.allclose(
            self.generator.sum(axis=1),
            0.0,
            atol=1e-8,
        ):
            raise ValueError(
                "Rows of generator Q do not sum to zero."
            )

    def generator_frame(self) -> pd.DataFrame:
        """Return Q as a labeled DataFrame."""

        labels = [
            f"state_{i}"
            for i in range(self.n_states)
        ]

        return pd.DataFrame(
            self.generator,
            index=labels,
            columns=labels,
        )

    def transition_frame(self) -> pd.DataFrame:
        """Return HMM transition matrix as a labeled DataFrame."""

        labels = [
            f"state_{i}"
            for i in range(self.n_states)
        ]

        return pd.DataFrame(
            self.transition_matrix,
            index=labels,
            columns=labels,
        )
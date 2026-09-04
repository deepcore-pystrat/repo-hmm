"""
Regime-switching stochastic-process estimation.

Paper connection
----------------

The paper considers

    dX_t = b(alpha_t, X_t) dt
           + sigma(alpha_t, X_t) dB_t

where alpha_t is a finite-state Markov chain.

For the first implementation in this repository, we use the simpler
regime-dependent constant-coefficient approximation

    X_t = log(MID_t)

    dX_t = mu_{alpha_t} dt
           + sigma_{alpha_t} dB_t.

The HMM provides the regime alpha_t through spot_state.

IMPORTANT
---------
All stochastic parameters must be fitted on TRAIN only.
The TEST feed is enriched using frozen TRAIN parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.optimal_switching.switching_data import (
    StochasticProcessConfig,
    StochasticProcessEstimate,
)


def validate_transition_matrix(
    transition_matrix: np.ndarray,
    atol: float = 1e-8,
) -> np.ndarray:
    """
    Validate a discrete HMM transition matrix.

    Parameters
    ----------
    transition_matrix:
        HMM transition matrix, normally model.A.

    Returns
    -------
    np.ndarray
        Validated copy of the matrix.
    """

    P = np.asarray(
        transition_matrix,
        dtype=float,
    ).copy()

    if P.ndim != 2:
        raise ValueError(
            "transition_matrix must be 2-dimensional."
        )

    if P.shape[0] != P.shape[1]:
        raise ValueError(
            "transition_matrix must be square."
        )

    if not np.isfinite(P).all():
        raise ValueError(
            "transition_matrix contains NaN or Inf."
        )

    if (P < -atol).any():
        raise ValueError(
            "transition_matrix contains negative probabilities."
        )

    P = np.clip(
        P,
        0.0,
        None,
    )

    row_sums = P.sum(axis=1)

    if np.any(row_sums <= 0):
        raise ValueError(
            "transition_matrix contains an empty row."
        )

    # Small numerical correction only.
    P = P / row_sums[:, None]

    if not np.allclose(
        P.sum(axis=1),
        1.0,
        atol=atol,
    ):
        raise ValueError(
            "Rows of transition_matrix do not sum to 1."
        )

    return P


def transition_matrix_to_generator(
    transition_matrix: np.ndarray,
    dt: float = 1.0,
    method: str = "first_order",
) -> np.ndarray:
    """
    Convert the discrete HMM transition matrix P into an approximation
    of the continuous-time Markov generator Q used in the paper.

    First-order approximation:

        P ~= I + Q dt

    therefore

        Q ~= (P - I) / dt.

    Notes
    -----
    This is intentionally used instead of blindly applying matrix log.

    log(P)/dt is mathematically attractive, but a generic estimated HMM
    transition matrix is not necessarily embeddable as a continuous-time
    Markov chain. In that case log(P) may produce negative off-diagonal
    intensities.

    The first-order approximation is safer for the first implementation.
    """

    if dt <= 0:
        raise ValueError(
            "dt must be strictly positive."
        )

    P = validate_transition_matrix(
        transition_matrix
    )

    if method != "first_order":
        raise ValueError(
            "Unsupported generator method: "
            f"{method}. "
            "Currently supported: 'first_order'."
        )

    n_states = P.shape[0]

    Q = (
        P - np.eye(n_states)
    ) / dt

    # Numerical cleanup
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                Q[i, j] = max(
                    0.0,
                    Q[i, j],
                )

    # Force exact generator row constraint
    for i in range(n_states):
        Q[i, i] = -(
            np.sum(Q[i, :])
            - Q[i, i]
        )

    if not np.allclose(
        Q.sum(axis=1),
        0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Failed to construct a valid generator Q."
        )

    return Q


def build_regime_increment_sample(
    market_data: pd.DataFrame,
    state_feed: pd.DataFrame,
    config: StochasticProcessConfig,
) -> pd.DataFrame:
    """
    Construct the TRAIN sample used to estimate mu_p and sigma_p.

    We define

        X_t = log(MID_t)

    and

        dX_t = X_t - X_{t-1}.

    By default the increment t-1 -> t is associated with the regime
    known at t-1. This avoids conditioning a realized return on a state
    that is only observed at the end of that same interval.

    Returns
    -------
    DataFrame with:
        price
        x
        dx
        state
    """

    if config.price_col not in market_data.columns:
        raise ValueError(
            f"Missing price column "
            f"'{config.price_col}' in market_data."
        )

    if config.state_col not in state_feed.columns:
        raise ValueError(
            f"Missing state column "
            f"'{config.state_col}' in state_feed."
        )

    spot = market_data[
        [config.price_col]
    ].copy()

    spot = spot.sort_index()

    price = pd.to_numeric(
        spot[config.price_col],
        errors="coerce",
    )

    price = price.where(
        price > 0
    )

    # Continuous state used by the switching diffusion.
    x = np.log(price)

    # Increment of X_t = log(MID_t)
    dx = x.diff()

    state = (
        state_feed[config.state_col]
        .sort_index()
        .astype("Int64")
    )

    if config.use_lagged_state:
        state_for_increment = state.shift(1)
    else:
        state_for_increment = state

    sample = pd.concat(
        [
            price.rename("price"),
            x.rename("x"),
            dx.rename("dx"),
            state_for_increment.rename("state"),
        ],
        axis=1,
        join="inner",
    )

    sample = sample.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    sample = sample.dropna(
        subset=[
            "price",
            "x",
            "dx",
            "state",
        ]
    )

    sample["state"] = (
        sample["state"]
        .astype(int)
    )

    return sample


def estimate_regime_diffusion(
    market_data: pd.DataFrame,
    state_feed: pd.DataFrame,
    n_states: int,
    config: StochasticProcessConfig | None = None,
) -> pd.DataFrame:
    """
    Estimate mu_p and sigma_p for every HMM regime.

    Model
    -----

        dX_t = mu_p dt + sigma_p dB_t

    Conditional moments imply approximately

        E[dX | p]   = mu_p * dt

        Var[dX | p] = sigma_p^2 * dt

    Therefore

        mu_p    = mean(dX | p) / dt

        sigma_p = std(dX | p) / sqrt(dt).

    Only TRAIN data must be passed here.
    """

    if config is None:
        config = StochasticProcessConfig()

    if n_states <= 0:
        raise ValueError(
            "n_states must be positive."
        )

    if config.dt <= 0:
        raise ValueError(
            "config.dt must be positive."
        )

    sample = build_regime_increment_sample(
        market_data=market_data,
        state_feed=state_feed,
        config=config,
    )

    rows = []

    for state in range(n_states):

        group = sample.loc[
            sample["state"] == state,
            "dx",
        ].dropna()

        n_obs = int(len(group))

        if n_obs < config.min_obs_per_regime:
            raise ValueError(
                f"State {state} has only {n_obs} usable "
                f"observations. Minimum required: "
                f"{config.min_obs_per_regime}."
            )

        mean_dx = float(
            group.mean()
        )

        std_dx = float(
            group.std(ddof=1)
        )

        if not np.isfinite(std_dx):
            raise ValueError(
                f"Invalid volatility estimate for state {state}."
            )

        if std_dx <= config.eps:
            raise ValueError(
                f"Near-zero volatility for state {state}."
            )

        mu = (
            mean_dx
            / config.dt
        )

        sigma = (
            std_dx
            / np.sqrt(config.dt)
        )

        variance = sigma ** 2

        rows.append(
            {
                "state": state,
                "mu": mu,
                "sigma": sigma,
                "variance": variance,
                "n_obs": n_obs,
                "mean_dx": mean_dx,
                "std_dx": std_dx,
            }
        )

    params = (
        pd.DataFrame(rows)
        .set_index("state")
        .sort_index()
    )

    return params


def fit_stochastic_process(
    market_train: pd.DataFrame,
    train_feed: pd.DataFrame,
    transition_matrix: np.ndarray,
    config: StochasticProcessConfig | None = None,
) -> StochasticProcessEstimate:
    """
    Fit the regime-switching diffusion using TRAIN data only.

    Parameters
    ----------
    spot_train:
        TRAIN spot DataFrame.
        In corn1.py this is the existing `spot_train`.

    train_feed:
        TRAIN HMM state feed produced by build_state_feed().

    transition_matrix:
        model.A from the fitted HMM.

    Returns
    -------
    StochasticProcessEstimate
    """

    if config is None:
        config = StochasticProcessConfig()

    P = validate_transition_matrix(
        transition_matrix
    )

    n_states = P.shape[0]

    regime_params = estimate_regime_diffusion(
    market_data=market_train,
    state_feed=train_feed,
    n_states=n_states,
    config=config,
)

    Q = transition_matrix_to_generator(
        transition_matrix=P,
        dt=config.dt,
        method=config.generator_method,
    )

    result = StochasticProcessEstimate(
        regime_params=regime_params,
        generator=Q,
        transition_matrix=P,
        n_states=n_states,
        dt=config.dt,
        price_col=config.price_col,
        state_col=config.state_col,
    )

    result.validate()

    return result


def enrich_feed_with_stochastic_process(
    feed: pd.DataFrame,
    market_data: pd.DataFrame,
    estimate: StochasticProcessEstimate,
) -> pd.DataFrame:
    """
    Add stochastic-process variables to a state feed.

    No parameter is re-estimated here.

    This means the same TRAIN estimates can safely be projected onto
    both TRAIN and TEST feeds.

    Added columns
    -------------
    switch_x
        Continuous state X_t = log(MID_t).

    switch_mu
        Drift mu_p corresponding to current HMM regime.

    switch_sigma
        Diffusion volatility sigma_p corresponding to current regime.

    switch_variance
        sigma_p^2.

    switch_exit_intensity
        -Q_pp: total instantaneous intensity of leaving current state.

    switch_q_to_0, switch_q_to_1, ...
        Current row of generator Q.
    """

    out = feed.copy()
    out = out.sort_index()

    state_col = estimate.state_col
    price_col = estimate.price_col

    if state_col not in out.columns:
        raise ValueError(
            f"Missing state column '{state_col}' in feed."
        )

    if price_col not in market_data.columns:
        raise ValueError(
            f"Missing price column '{price_col}' in market_data."
        )

    price = (
        pd.to_numeric(
            market_data[price_col],
            errors="coerce",
        )
        .reindex(out.index)
    )

    if price.isna().any():
        n_missing = int(
            price.isna().sum()
        )

        raise ValueError(
            f"{n_missing} feed dates do not have a valid "
            f"{price_col} observation."
        )

    if (price <= 0).any():
        raise ValueError(
            f"{price_col} must be strictly positive."
        )

    state = (
        out[state_col]
        .astype(int)
    )

    params = estimate.regime_params

    unknown_states = (
        set(state.unique())
        - set(params.index.astype(int))
    )

    if unknown_states:
        raise ValueError(
            "Feed contains regimes with no fitted parameters: "
            f"{sorted(unknown_states)}"
        )

    # Continuous diffusion state
    out["switch_x"] = np.log(
        price.astype(float)
    )

    # Parameters of current regime
    out["switch_mu"] = state.map(
        params["mu"]
    )

    out["switch_sigma"] = state.map(
        params["sigma"]
    )

    out["switch_variance"] = state.map(
        params["variance"]
    )

    # Generator information
    Q = estimate.generator

    out["switch_exit_intensity"] = [
        -Q[s, s]
        for s in state
    ]

    for target_state in range(
        estimate.n_states
    ):

        out[
            f"switch_q_to_{target_state}"
        ] = [
            Q[s, target_state]
            for s in state
        ]

    # Basic safety check
    stochastic_cols = [
        "switch_x",
        "switch_mu",
        "switch_sigma",
        "switch_variance",
        "switch_exit_intensity",
    ]

    if not np.isfinite(
        out[stochastic_cols].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "NaN or Inf found in stochastic-process feed columns."
        )

    return out


def stochastic_process_report(
    estimate: StochasticProcessEstimate,
) -> None:
    """
    Human-readable diagnostics printed by corn1.py.
    """

    print("\n" + "=" * 80)
    print("OPTIMAL SWITCHING - REGIME DIFFUSION PARAMETERS")
    print("=" * 80)

    print(
        estimate.regime_params
        .round(8)
        .to_string()
    )

    print("\n" + "=" * 80)
    print("HMM DISCRETE TRANSITION MATRIX P")
    print("=" * 80)

    print(
        estimate.transition_frame()
        .round(6)
        .to_string()
    )

    print("\n" + "=" * 80)
    print("CONTINUOUS-TIME GENERATOR APPROXIMATION Q")
    print("=" * 80)

    print(
        estimate.generator_frame()
        .round(6)
        .to_string()
    )
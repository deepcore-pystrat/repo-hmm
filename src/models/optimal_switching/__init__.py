"""
Optimal switching model.

Current implementation
----------------------
Step 1:
    Regime-dependent stochastic-process estimation.

Next steps:
    - running reward f(i, p, x)
    - switching costs g_ij
    - value functions v(i, p, x)
    - continuation/switching regions
    - KEEP / SWITCH trading signal
"""

from models.optimal_switching.switching_data import (
    StochasticProcessConfig,
    StochasticProcessEstimate,
)

from models.optimal_switching.stochastic_process import (
    build_regime_increment_sample,
    enrich_feed_with_stochastic_process,
    estimate_regime_diffusion,
    fit_stochastic_process,
    stochastic_process_report,
    transition_matrix_to_generator,
    validate_transition_matrix,
)


__all__ = [
    "StochasticProcessConfig",
    "StochasticProcessEstimate",
    "build_regime_increment_sample",
    "estimate_regime_diffusion",
    "validate_transition_matrix",
    "transition_matrix_to_generator",
    "fit_stochastic_process",
    "enrich_feed_with_stochastic_process",
    "stochastic_process_report",
]
"""
models package -- re-exports for convenience.
"""

from models.basis import (
    HMMConfig,
    HMMResult,
    BaseHMM,
    map_regimes,
    map_regime_exposures,
)

from models.gaussian_hmm import (
    GaussianHMMConfig,
    GaussianHMM,
)

from models.student_hmm import (
    StudentTHMMConfig,
    StudentTHMM,
)

from models.student_hsmm import (
    StudentTHSMMConfig,
    StudentTHSMM,
)

from models.garch_hmm import (
    GarchHMMConfig,
    GarchHMM,
)


def create_hmm(config):
    """
    Construit le modèle correspondant à la configuration fournie.

    Important
    ---------
    StudentTHSMMConfig hérite de StudentTHMMConfig.

    Le test StudentTHSMMConfig doit donc être placé avant
    le test StudentTHMMConfig.
    """

    if isinstance(config, GaussianHMMConfig):
        return GaussianHMM(config)

    # Doit obligatoirement être avant StudentTHMMConfig.
    if isinstance(config, StudentTHSMMConfig):
        return StudentTHSMM(config)

    if isinstance(config, StudentTHMMConfig):
        return StudentTHMM(config)

    if isinstance(config, GarchHMMConfig):
        return GarchHMM(config)

    raise TypeError(
        f"No HMM class registered for "
        f"{type(config).__name__}"
    )


__all__ = [
    "HMMConfig",
    "HMMResult",
    "BaseHMM",
    "map_regimes",
    "map_regime_exposures",
    "GaussianHMMConfig",
    "GaussianHMM",
    "StudentTHMMConfig",
    "StudentTHMM",
    "StudentTHSMMConfig",
    "StudentTHSMM",
    "GarchHMMConfig",
    "GarchHMM",
    "create_hmm",
]
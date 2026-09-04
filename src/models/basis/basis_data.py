"""
Base HMM configuration and result data containers.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class HMMConfig:
    K: int = 3
    n_iter: int = 300
    seed: int = 42
    min_var: float = 1e-6
    tol: float = 1e-4
    cov_type: str = "diag"
    sticky: float = 1.0

    transition_mask: np.ndarray | None = None


@dataclass
class HMMResult:
    pi: np.ndarray
    A: np.ndarray
    means: np.ndarray
    vars_: np.ndarray
    ll_hist: np.ndarray

    gamma: np.ndarray | None = None
    filtered_proba: np.ndarray | None = None
    states: np.ndarray | None = None

    final_ll: float | None = None
    bic: float | None = None
    aic: float | None = None
    state_durations: np.ndarray | None = None

    nus: np.ndarray | None = None
    covs_: np.ndarray | None = None
    estimate_nu: bool = False
    fixed_nu: float = 6.0

    garch_params: dict | None = None
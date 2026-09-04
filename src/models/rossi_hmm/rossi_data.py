from dataclasses import dataclass
import numpy as np


@dataclass
class RossiHMMConfig:
    K: int = 7
    p: int = 1
    n_iter: int = 3000
    n_starts: int = 10
    seed: int = 42
    min_var: float = 1e-10


@dataclass
class RossiHMMResult:
    params: dict
    sigma2: np.ndarray
    filtered_proba: np.ndarray
    states: np.ndarray
    labels: np.ndarray
    loglik: float
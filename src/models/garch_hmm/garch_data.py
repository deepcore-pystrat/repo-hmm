from dataclasses import dataclass
from models.basis.basis_data import HMMConfig


@dataclass
class GarchHMMConfig(HMMConfig):
    dist: str = "student"
    fixed_nu: float = 8.0

    min_omega: float = 1e-10
    max_persistence: float = 0.995
    m_step_maxiter: int = 200
    m_step_ftol: float = 1e-9

    n_init: int = 10

    log_zero_floor: float = -1e300
    exp_clip: float = 35.0
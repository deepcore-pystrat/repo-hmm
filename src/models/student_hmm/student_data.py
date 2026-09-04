from dataclasses import dataclass
from models.basis.basis_data import HMMConfig


@dataclass
class StudentTHMMConfig(HMMConfig):
    estimate_nu: bool = True
    fixed_nu: float = 6.0
"""
Gaussian HMM configuration.
"""

from dataclasses import dataclass

from models.basis.basis_data import HMMConfig


@dataclass
class GaussianHMMConfig(HMMConfig):
    """Gaussian (diagonal covariance) HMM -- no extra params."""
    pass

"""
Privacy module for the Federated Functional Privacy (FFP) framework.
Implements the RKHS-calibrated Matrix-Normal noise injection, strict L2 sensitivity
bounds (Theorem 1), and the Rényi Differential Privacy (RDP) accountant for the
two-phase, multi-round pipeline.
"""

from .sensitivity import SensitivityCalculator
from .rkhs_functional_dp import RKHSFunctionalDP
from .rdp_accountant import RDPAccountant

__all__ = [
    "SensitivityCalculator",
    "RKHSFunctionalDP",
    "RDPAccountant",
]
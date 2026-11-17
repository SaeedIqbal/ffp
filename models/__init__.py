"""
Models module for the Federated Functional Privacy (FFP) framework.
Contains the core Takagi-Sugeno fuzzy model and its parameter estimation algorithms.
"""

from .ts_fuzzy import TSFuzzyModel
from .fcm import FuzzyCMeans
from .wls import WeightedLeastSquares

__all__ = [
    "TSFuzzyModel",
    "FuzzyCMeans",
    "WeightedLeastSquares"
]
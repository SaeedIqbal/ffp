"""
Data module for the Federated Functional Privacy (FFP) framework.
Handles data loading, preprocessing, and the Differentially Private Coarse Histogram mechanism.
"""

from .preprocessing import DataPreprocessor
from .dataloader import FederatedDataLoader
from .dp_histogram import DPHistogramMechanism

__all__ = [
    "DataPreprocessor",
    "FederatedDataLoader",
    "DPHistogramMechanism"
]
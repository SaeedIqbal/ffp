import numpy as np
from typing import Tuple

class TSFuzzyModel:
    """
    Core Takagi-Sugeno (TS) Fuzzy Model.
    Implements the Gaussian membership functions, normalized firing strengths, 
    and the final rule-based output aggregation.
    """

    def __init__(self, num_rules: int, input_dim: int):
        """
        :param num_rules: Number of fuzzy rules (R).
        :param input_dim: Dimensionality of the input space (d).
        """
        self.R = num_rules
        self.d = input_dim
        self.epsilon = 1e-8  # Numerical stability constant

    def compute_unnormalized_firing_strength(
        self, 
        X: np.ndarray, 
        centers: np.ndarray, 
        widths: np.ndarray
    ) -> np.ndarray:
        """
        Computes the unnormalized Gaussian firing strengths.
        Equation: \tilde{\phi}_r(x) = \exp\left(-\frac{1}{2}(x - \mathbf{c}_r)^\top \boldsymbol{\Sigma}_r^{-1} (x - \mathbf{c}_r)\right)
        Assuming diagonal covariance \boldsymbol{\Sigma}_r = \text{diag}(\sigma_{r,1}^2, \dots, \sigma_{r,d}^2).
        
        :param X: Input data of shape (N, d).
        :param centers: Antecedent centers of shape (R, d).
        :param widths: Antecedent widths (standard deviations) of shape (R, d).
        :return: Unnormalized firing strengths of shape (N, R).
        """
        # X: (N, 1, d), centers: (1, R, d) -> diff: (N, R, d)
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        variance = widths[np.newaxis, :, :] ** 2 + self.epsilon
        
        # Mahalanobis distance squared for diagonal covariance
        mahalanobis_sq = np.sum((diff ** 2) / variance, axis=2)
        
        return np.exp(-0.5 * mahalanobis_sq)

    def compute_normalized_firing_strength(
        self, 
        X: np.ndarray, 
        centers: np.ndarray, 
        widths: np.ndarray
    ) -> np.ndarray:
        """
        Computes the normalized firing strengths satisfying the partition of unity.
        Equation: \phi_r(x) = \frac{\tilde{\phi}_r(x)}{\sum_{k=1}^R \tilde{\phi}_k(x)}
        
        :return: Normalized firing strengths of shape (N, R).
        """
        unnorm_phi = self.compute_unnormalized_firing_strength(X, centers, widths)
        sum_phi = np.sum(unnorm_phi, axis=1, keepdims=True) + self.epsilon
        return unnorm_phi / sum_phi

    def predict(
        self, 
        X: np.ndarray, 
        centers: np.ndarray, 
        widths: np.ndarray, 
        betas: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates the global TS fuzzy model output.
        Equation: f(x) = \sum_{r=1}^R \phi_r(x) \langle \boldsymbol{\beta}_r, \tilde{x} \rangle
        
        :param betas: Consequent parameters of shape (R, d + 1).
        :return: Predicted outputs of shape (N,).
        """
        N = X.shape[0]
        phi = self.compute_normalized_firing_strength(X, centers, widths)  # (N, R)
        
        # Augment X with a column of ones for the bias term: \tilde{x} = [1, x^\top]^\top
        X_aug = np.hstack([np.ones((N, 1)), X])  # (N, d + 1)
        
        # Compute consequent outputs for all rules: (N, R)
        rule_outputs = X_aug @ betas.T 
        
        # Weighted sum aggregation
        return np.sum(phi * rule_outputs, axis=1)
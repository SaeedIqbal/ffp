import numpy as np
from typing import Tuple


class RKHSFunctionalDP:
    """
    Implements the Discretization-Consistent Functional Differential Privacy
    mechanism via RKHS-calibrated Matrix-Normal noise injection.

    Theoretical References:
        - Eq. 7 (Phase 1 Antecedent):
            Phi_tilde_j = Phi_j + Z_{Phi,j},
            vec(Z_{Phi,j}) ~ N(0, sigma_Phi^2 * (K_A kron I_R))
        - Eq. 8 (Phase 2 Consequent):
            F_tilde_j = F_j + Z_{F,j},
            vec(Z_{F,j}) ~ N(0, sigma_F^2 * (K_A kron I_R))
        - Theorem 2 (Noise scale calibration):
            sigma >= (Delta_H / epsilon) * sqrt(2 ln(1.25 / delta))
    """

    def __init__(
        self,
        num_rules: int,
        anchor_size: int,
        kernel_sigma: float = 1.0,
        jitter: float = 1e-6,
    ):
        """
        :param num_rules: Number of fuzzy rules (R).
        :param anchor_size: Size of the manifold-aligned anchor set (M).
        :param kernel_sigma: Length-scale parameter for the RBF kernel.
        :param jitter: Diagonal regularization for Cholesky stability.
        """
        self.R = num_rules
        self.M = anchor_size
        self.kernel_sigma = kernel_sigma
        self.jitter = jitter

    def compute_kernel_matrix(self, anchors: np.ndarray) -> np.ndarray:
        """
        Computes the RKHS kernel matrix K_A evaluated on the anchor set.

        Equation: [K_A]_{m, m'} = K_TS(a_m, a_{m'}) = exp(-||a_m - a_{m'}||^2 / (2 sigma^2))

        :param anchors: Array of shape (M, d) containing the anchor coordinates.
        :return: Kernel matrix of shape (M, M).
        """
        sq_norms = np.sum(anchors ** 2, axis=1)
        sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (anchors @ anchors.T)
        sq_dists = np.maximum(sq_dists, 0.0)  # numerical guard against tiny negatives
        K = np.exp(-sq_dists / (2.0 * self.kernel_sigma ** 2))
        # Ensure strict positive-definiteness for Cholesky decomposition
        K += self.jitter * np.eye(self.M)
        return K

    def compute_noise_scale(
        self, sensitivity: float, epsilon: float, delta: float
    ) -> float:
        """
        Calibrates the Gaussian noise scale to satisfy (epsilon, delta)-Functional-DP
        as stated in Theorem 2.

        Equation: sigma = (Delta_H / epsilon) * sqrt(2 ln(1.25 / delta))

        :param sensitivity: The RKHS sensitivity Delta_H.
        :param epsilon: Target privacy budget.
        :param delta: Target failure probability.
        :return: Noise standard deviation sigma.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be strictly positive.")
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1).")
        sigma = (sensitivity / epsilon) * np.sqrt(2.0 * np.log(1.25 / delta))
        return float(sigma)

    def _cholesky_factor(self, K: np.ndarray) -> np.ndarray:
        """
        Computes the lower-triangular Cholesky factor L of K with retry on failure.
        Reused by the noise generation method to guarantee numerical stability.
        """
        for scale in (1.0, 10.0, 100.0):
            try:
                return np.linalg.cholesky(K + scale * self.jitter * np.eye(K.shape[0]))
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError(
            "Kernel matrix is not positive definite even after jitter."
        )

    def generate_matrix_normal_noise(
        self, sigma: float, K: np.ndarray
    ) -> np.ndarray:
        """
        Generates a noise matrix Z of shape (R, M) such that
        vec(Z) ~ N(0, sigma^2 * (K kron I_R)).

        Construction: Z = sigma * G @ L^T where G ~ N(0, I_{R x M}) and
        L = chol(K). This satisfies Cov(vec(Z)) = sigma^2 (K kron I_R).

        :param sigma: Noise standard deviation.
        :param K: Kernel matrix of shape (M, M).
        :return: Noise matrix of shape (R, M).
        """
        L = self._cholesky_factor(K)
        G = np.random.randn(self.R, self.M)
        Z = sigma * (G @ L.T)
        return Z

    def perturb_activation_profile(
        self,
        Phi: np.ndarray,
        K: np.ndarray,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Applies Eq. 7 to perturb the antecedent activation profile matrix.

        :param Phi: Activation profile matrix of shape (R, M).
        :param K: Kernel matrix of shape (M, M).
        :param sensitivity: RKHS sensitivity for the antecedent phase.
        :param epsilon: Privacy budget for this phase.
        :param delta: Failure probability for this phase.
        :return: Tuple of (perturbed matrix, noise scale used).
        """
        sigma = self.compute_noise_scale(sensitivity, epsilon, delta)
        Z = self.generate_matrix_normal_noise(sigma, K)
        return Phi + Z, sigma

    def perturb_consequent_matrix(
        self,
        F: np.ndarray,
        K: np.ndarray,
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Applies Eq. 8 to perturb the consequent output matrix.

        :param F: Consequent matrix of shape (R, M).
        :param K: Kernel matrix of shape (M, M).
        :param sensitivity: RKHS sensitivity for the consequent phase.
        :param epsilon: Privacy budget for this phase.
        :param delta: Failure probability for this phase.
        :return: Tuple of (perturbed matrix, noise scale used).
        """
        sigma = self.compute_noise_scale(sensitivity, epsilon, delta)
        Z = self.generate_matrix_normal_noise(sigma, K)
        return F + Z, sigma

    def release(
        self,
        Phi: np.ndarray,
        F: np.ndarray,
        anchors: np.ndarray,
        sens_antecedent: float,
        sens_consequent: float,
        epsilon: float,
        delta: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        End-to-end two-phase release mechanism combining Eq. 7 and Eq. 8.
        Reuses compute_kernel_matrix, perturb_activation_profile, and
        perturb_consequent_matrix.

        :return: (Phi_tilde, F_tilde, sigma_Phi, sigma_F)
        """
        K = self.compute_kernel_matrix(anchors)
        Phi_tilde, sigma_Phi = self.perturb_activation_profile(
            Phi, K, sens_antecedent, epsilon, delta
        )
        F_tilde, sigma_F = self.perturb_consequent_matrix(
            F, K, sens_consequent, epsilon, delta
        )
        return Phi_tilde, F_tilde, sigma_Phi, sigma_F
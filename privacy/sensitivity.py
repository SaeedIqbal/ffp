import numpy as np
from typing import Tuple, Dict


class SensitivityCalculator:
    """
    Computes strict L2 sensitivity bounds for the Takagi-Sugeno (TS) fuzzy model
    under the single-record replacement neighboring relation.

    Theoretical Reference: Theorem 1 (Bounded RKHS Sensitivity of TS Models).
        Δ_H <= sqrt(R) * ( (B_x * B_y / lambda_min) * sup_x ||x_tilde||_2
                          + L_phi * B_x * max_r ||beta_r||_2 )

    where:
        - B_x, B_y : clipping bounds on the input domain and output range.
        - lambda_min : minimum ridge regularization parameter for WLS.
        - L_phi : Lipschitz constant of the normalized firing strengths w.r.t.
                  the antecedent parameters.
        - R : number of fuzzy rules.
    """

    def __init__(
        self,
        num_rules: int,
        input_dim: int,
        clip_x: float,
        clip_y: float,
        lambda_min: float,
    ):
        """
        :param num_rules: Number of fuzzy rules (R).
        :param input_dim: Dimensionality of the input space (d).
        :param clip_x: Input clipping bound B_x such that X subset B_d(0, B_x).
        :param clip_y: Output clipping bound B_y such that |y| <= B_y.
        :param lambda_min: Minimum ridge regularization parameter for WLS.
        """
        if lambda_min <= 0:
            raise ValueError("lambda_min must be strictly positive for WLS stability.")
        self.R = num_rules
        self.d = input_dim
        self.B_x = clip_x
        self.B_y = clip_y
        self.lambda_min = lambda_min

    def compute_augmented_norm_bound(self) -> float:
        """
        Computes sup_{x in X} ||x_tilde||_2 where x_tilde = [1, x^T]^T.
        Equation: sqrt(1 + B_x^2)
        """
        return np.sqrt(1.0 + self.B_x ** 2)

    def compute_consequent_sensitivity(self) -> float:
        """
        Computes the sensitivity bound Delta_beta of the ridge WLS estimator
        under a single-record replacement.

        Equation (from Theorem 1, consequent phase):
            Delta_beta = (B_x * B_y / lambda_min) * sup_x ||x_tilde||_2
        """
        aug_norm = self.compute_augmented_norm_bound()
        delta_beta = (self.B_x * self.B_y / self.lambda_min) * aug_norm
        return float(delta_beta)

    def compute_lipschitz_constant(self, widths: np.ndarray) -> float:
        """
        Computes the Lipschitz constant L_phi of the normalized Gaussian firing
        strengths with respect to the antecedent centers.

        For a Gaussian membership function mu_r(x) = exp(-||x - c_r||^2 / (2 sigma_r^2)),
        the gradient w.r.t. c_r is bounded by 1 / sigma_min. The normalized firing
        strength inherits this bound up to a constant factor. We use the conservative
        bound L_phi = sqrt(2 / pi) / sigma_min.

        :param widths: Array of shape (R, d) containing the Gaussian widths.
        :return: Lipschitz constant L_phi.
        """
        sigma_min = np.min(widths)
        if sigma_min <= 0:
            raise ValueError("Gaussian widths must be strictly positive.")
        # Conservative Lipschitz bound for the normalized Gaussian partition
        L_phi = np.sqrt(2.0 / np.pi) / sigma_min
        return float(L_phi)

    def compute_antecedent_sensitivity(
        self, L_phi: float, max_beta_norm: float
    ) -> float:
        """
        Computes the sensitivity bound Delta_c induced by the antecedent phase.

        Equation (from Theorem 1, antecedent phase):
            Delta_c = L_phi * B_x * max_r ||beta_r||_2
        """
        delta_c = L_phi * self.B_x * max_beta_norm
        return float(delta_c)

    def compute_rkhs_sensitivity(
        self, widths: np.ndarray, max_beta_norm: float
    ) -> Dict[str, float]:
        """
        Computes the full RKHS sensitivity bound Delta_H as defined in Theorem 1.

        Equation:
            Delta_H = sqrt(R) * (Delta_beta + Delta_c)

        :param widths: Antecedent widths of shape (R, d).
        :param max_beta_norm: Maximum L2 norm across all consequent vectors.
        :return: Dictionary containing Delta_beta, Delta_c, and Delta_H.
        """
        delta_beta = self.compute_consequent_sensitivity()
        L_phi = self.compute_lipschitz_constant(widths)
        delta_c = self.compute_antecedent_sensitivity(L_phi, max_beta_norm)
        delta_H = np.sqrt(self.R) * (delta_beta + delta_c)

        return {
            "delta_beta": delta_beta,
            "delta_c": delta_c,
            "L_phi": L_phi,
            "delta_H": float(delta_H),
        }

    def estimate_max_beta_norm(self, betas: np.ndarray) -> float:
        """
        Estimates max_r ||beta_r||_2 from the consequent parameter matrix.

        :param betas: Consequent parameters of shape (R, d + 1).
        :return: Maximum L2 norm across rows.
        """
        norms = np.linalg.norm(betas, axis=1)
        return float(np.max(norms))
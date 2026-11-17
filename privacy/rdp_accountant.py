import numpy as np
from typing import List, Tuple, Optional


class RDPAccountant:
    """
    Rényi Differential Privacy (RDP) accountant for the sequential two-phase,
    multi-round federated pipeline.

    Theoretical References:
        - Per-phase RDP of the Gaussian mechanism of order alpha:
            epsilon_RDP(alpha) = alpha * Delta^2 / (2 * sigma^2)
        - Adaptive composition over T rounds (Eq. from methodology):
            epsilon_total(alpha) = T * (alpha * Delta_Phi^2 / (2 sigma_Phi^2)
                                      + alpha * Delta_F^2 / (2 sigma_F^2))
        - RDP-to-DP conversion:
            epsilon_DP = epsilon_RDP + log(1 / delta) / (alpha - 1)
    """

    def __init__(self, delta: float = 1e-5):
        """
        :param delta: Target failure probability for the final (epsilon, delta)-DP.
        """
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1).")
        self.delta = delta
        # Internal state: list of (sigma, sensitivity) tuples per released mechanism
        self._history: List[Tuple[float, float]] = []

    def reset(self) -> None:
        """Clears the accumulated privacy history."""
        self._history = []

    def record_mechanism(self, sigma: float, sensitivity: float) -> None:
        """
        Records a single Gaussian mechanism release.

        :param sigma: Noise standard deviation used.
        :param sensitivity: L2 sensitivity of the released function.
        """
        if sigma <= 0:
            raise ValueError("sigma must be strictly positive.")
        self._history.append((float(sigma), float(sensitivity)))

    def record_round(
        self, sigma_Phi: float, delta_Phi: float, sigma_F: float, delta_F: float
    ) -> None:
        """
        Records both phases of a single global round. Reuses record_mechanism.

        :param sigma_Phi: Noise scale for Phase 1 (antecedent).
        :param delta_Phi: Sensitivity for Phase 1.
        :param sigma_F: Noise scale for Phase 2 (consequent).
        :param delta_F: Sensitivity for Phase 2.
        """
        self.record_mechanism(sigma_Phi, delta_Phi)
        self.record_mechanism(sigma_F, delta_F)

    def compute_rdp_order(self, alpha: float) -> float:
        """
        Computes the total RDP guarantee of order alpha by summing the
        per-mechanism RDP contributions (adaptive composition theorem).

        Equation: epsilon_RDP(alpha) = sum_i alpha * Delta_i^2 / (2 sigma_i^2)

        :param alpha: Order of the Renyi divergence (alpha > 1).
        :return: Total RDP epsilon at order alpha.
        """
        if alpha <= 1.0:
            raise ValueError("RDP order alpha must be strictly greater than 1.")
        total = 0.0
        for sigma, sens in self._history:
            total += alpha * (sens ** 2) / (2.0 * sigma ** 2)
        return float(total)

    def _optimize_alpha(self, max_alpha: float = 1000.0, num_alphas: int = 2000) -> float:
        """
        Optimizes the RDP order alpha to minimize the resulting DP epsilon.
        Reused by get_privacy_spent.

        :return: Optimal alpha.
        """
        alphas = np.concatenate(
            [np.linspace(1.01, 10.0, num_alphas // 2),
             np.linspace(10.0, max_alpha, num_alphas // 2)]
        )
        best_alpha = alphas[0]
        best_eps = np.inf
        for a in alphas:
            rdp_eps = self.compute_rdp_order(a)
            dp_eps = rdp_eps + np.log(1.0 / self.delta) / (a - 1.0)
            if dp_eps < best_eps:
                best_eps = dp_eps
                best_alpha = a
        return float(best_alpha)

    def get_privacy_spent(
        self, alpha: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Converts the accumulated RDP history into a final (epsilon, delta)-DP
        guarantee via the standard RDP-to-DP conversion.

        Equation: epsilon_DP = epsilon_RDP(alpha) + log(1/delta) / (alpha - 1)

        :param alpha: If provided, uses this fixed order; otherwise optimizes.
        :return: Tuple of (epsilon, delta, optimal_alpha).
        """
        if not self._history:
            return 0.0, self.delta, 1.0

        if alpha is None:
            alpha = self._optimize_alpha()

        rdp_eps = self.compute_rdp_order(alpha)
        dp_eps = rdp_eps + np.log(1.0 / self.delta) / (alpha - 1.0)
        return float(dp_eps), self.delta, float(alpha)

    def check_budget(self, target_epsilon: float) -> bool:
        """
        Verifies whether the accumulated privacy cost remains within the target budget.

        :param target_epsilon: Maximum allowed epsilon.
        :return: True if the budget is not exceeded.
        """
        epsilon, _, _ = self.get_privacy_spent()
        return epsilon <= target_epsilon

    def get_history_length(self) -> int:
        """Returns the number of recorded mechanism releases."""
        return len(self._history)

    def get_round_count(self) -> int:
        """
        Infers the number of completed global rounds assuming two phases per round.
        Reuses get_history_length.
        """
        return self.get_history_length() // 2
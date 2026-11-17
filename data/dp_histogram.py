import numpy as np
from typing import List, Tuple

class DPHistogramMechanism:
    """
    Implements the Differentially Private Coarse Histogram (DP-Hist) mechanism 
    for manifold-aligned anchor placement.
    
    Equations referenced from the manuscript:
    - Eq. 9: Local histogram count vector h_{j,k}
    - Eq. 10: Gaussian mechanism for privatized counts \tilde{h}_j
    - Eq. 11: Global aggregated DP histogram \bar{h}
    """

    def __init__(self, epsilon_h: float, delta_h: float, num_bins_per_dim: int = 5):
        """
        :param epsilon_h: Privacy budget for the histogram mechanism.
        :param delta_h: Failure probability for the histogram mechanism.
        :param num_bins_per_dim: Number of bins per feature dimension (B^{1/d}).
        """
        self.epsilon_h = epsilon_h
        self.delta_h = delta_h
        self.num_bins_per_dim = num_bins_per_dim
        self.sigma_h = self._compute_noise_scale()

    def _compute_noise_scale(self) -> float:
        """
        Computes the Gaussian noise scale \sigma_h for L2 sensitivity \Delta_h = \sqrt{2}.
        Equation: \sigma_h >= (\sqrt{2} / \epsilon_h) * \sqrt{2 \ln(1.25 / \delta_h)}
        """
        delta_h = max(self.delta_h, 1e-10) # Prevent log(0)
        return (np.sqrt(2) / self.epsilon_h) * np.sqrt(2 * np.log(1.25 / delta_h))

    def compute_local_histogram(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the local histogram count vector h_j for client data X.
        Returns the flattened count vector and the bin edges.
        """
        d = X.shape[1]
        bins_per_dim = [self.num_bins_per_dim] * d
        
        # Compute histogram using numpy
        hist, edges = np.histogramdd(X, bins=bins_per_dim, range=[(0, 1)] * d)
        
        # Flatten the histogram to a 1D vector
        h_j = hist.flatten()
        
        return h_j, edges

    def add_gaussian_noise(self, h_j: np.ndarray) -> np.ndarray:
        """
        Applies the Gaussian mechanism to the local histogram.
        Equation: \tilde{h}_j = h_j + z_j, where z_j ~ N(0, \sigma_h^2 I_B)
        """
        noise = np.random.normal(0, self.sigma_h, size=h_j.shape)
        return h_j + noise

    def aggregate_histograms(self, privatized_histograms: List[np.ndarray]) -> np.ndarray:
        """
        Computes the global aggregated DP histogram.
        Equation: \bar{h} = (1/K) \sum_{j=1}^K \tilde{h}_j
        """
        K = len(privatized_histograms)
        if K == 0:
            raise ValueError("No client histograms provided for aggregation.")
            
        aggregated = np.mean(privatized_histograms, axis=0)
        return aggregated

    def select_anchors(self, aggregated_hist: np.ndarray, edges: np.ndarray, M: int) -> np.ndarray:
        """
        Selects the top-M anchors based on the aggregated DP histogram.
        Equation: I_M = argmax_{S} \sum_{k \in S} \hat{p}_k
        Anchor coordinate: a_m = centroid of hyper-rectangle b_{k_m}
        """
        # Compute estimated probability mass \hat{p}_k
        # Equation: \hat{p}_k = max(\bar{h}_k, 0) / \sum max(\bar{h}_l, 0)
        positive_hist = np.maximum(aggregated_hist, 0)
        total_mass = np.sum(positive_hist)
        
        if total_mass == 0:
            # Fallback to uniform selection if all bins are zeroed out by noise
            p_hat = np.ones_like(positive_hist) / len(positive_hist)
        else:
            p_hat = positive_hist / total_mass
            
        # Select top-M bins
        M = min(M, len(p_hat))
        top_indices = np.argsort(p_hat)[::-1][:M]
        
        # Compute centroids for selected bins
        anchors = []
        for idx in top_indices:
            # Convert flattened index back to multi-dimensional index
            multi_idx = np.unravel_index(idx, [self.num_bins_per_dim] * len(edges))
            
            # Compute centroid of the hyper-rectangle
            centroid = []
            for dim_idx, bin_idx in enumerate(multi_idx):
                low = edges[dim_idx][bin_idx]
                high = edges[dim_idx][bin_idx + 1]
                centroid.append((low + high) / 2.0)
                
            anchors.append(centroid)
            
        return np.array(anchors)

    def generate_anchors(self, client_datasets: List[Tuple[np.ndarray, np.ndarray]], M: int) -> np.ndarray:
        """
        End-to-end pipeline to generate manifold-aligned anchors from federated client data.
        """
        privatized_histograms = []
        edges = None
        
        for X_client, _ in client_datasets:
            h_j, edges = self.compute_local_histogram(X_client)
            h_tilde_j = self.add_gaussian_noise(h_j)
            privatized_histograms.append(h_tilde_j)
            
        aggregated_hist = self.aggregate_histograms(privatized_histograms)
        anchors = self.select_anchors(aggregated_hist, edges, M)
        
        return anchors
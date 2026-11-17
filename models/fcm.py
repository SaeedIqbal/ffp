import numpy as np

class FuzzyCMeans:
    """
    Fuzzy C-Means (FCM) algorithm for antecedent parameter estimation.
    Estimates the cluster centers \mathbf{c}_r and widths \boldsymbol{\sigma}_r.
    """

    def __init__(self, num_rules: int, fuzzifier: float = 2.0, max_iter: int = 100, tol: float = 1e-4):
        """
        :param num_rules: Number of clusters/rules (R).
        :param fuzzifier: Fuzzification index (m > 1).
        :param max_iter: Maximum number of iterations.
        :param tol: Convergence tolerance for center updates.
        """
        self.R = num_rules
        self.m = fuzzifier
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = 1e-8

    def fit(self, X: np.ndarray) -> tuple:
        """
        Fits the FCM model to the input data X.
        Updates centers and widths iteratively.
        
        :param X: Input data of shape (N, d).
        :return: Tuple of (centers, widths) each of shape (R, d).
        """
        N, d = X.shape
        
        # Initialize centers randomly from the dataset
        indices = np.random.choice(N, self.R, replace=False)
        centers = X[indices].copy()
        widths = np.ones((self.R, d)) * 0.1  # Initial widths
        
        for iteration in range(self.max_iter):
            # 1. Compute distances and update membership matrix U
            # dist: (N, R)
            dist = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2) ** 2
            dist = np.maximum(dist, self.epsilon)
            
            # Update memberships: u_{ir} = 1 / \sum_k (||x_i - c_r|| / ||x_i - c_k||)^{2/(m-1)}
            power = 2.0 / (self.m - 1.0)
            u = (1.0 / dist) ** power
            u = u / np.sum(u, axis=1, keepdims=True)
            
            # 2. Update centers
            # Equation: \mathbf{c}_r = \frac{\sum_{i=1}^n u_{ir}^m x_i}{\sum_{i=1}^n u_{ir}^m}
            u_m = u ** self.m
            new_centers = np.dot(u_m.T, X) / (np.sum(u_m, axis=0, keepdims=True).T + self.epsilon)
            
            # 3. Update widths (variances)
            # Equation: \sigma_{r,l}^2 = \frac{\sum_{i=1}^n u_{ir}^m (x_{i,l} - c_{r,l})^2}{\sum_{i=1}^n u_{ir}^m}
            new_widths = np.zeros((self.R, d))
            for r in range(self.R):
                weights = u_m[:, r]
                sum_weights = np.sum(weights) + self.epsilon
                diff_sq = (X - centers[r]) ** 2
                new_widths[r] = np.sqrt(np.average(diff_sq, axis=0, weights=weights) + self.epsilon)
            
            # Check convergence
            center_shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            widths = new_widths
            
            if center_shift < self.tol:
                break
                
        return centers, widths
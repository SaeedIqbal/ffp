import numpy as np

class WeightedLeastSquares:
    """
    Weighted Least Squares (WLS) for consequent parameter estimation.
    Estimates the linear coefficients \boldsymbol{\beta}_r for each rule.
    """

    def __init__(self, lambda_reg: float = 1e-3):
        """
        :param lambda_reg: Tikhonov regularization parameter (\lambda_{\text{reg}} > 0).
        """
        self.lambda_reg = lambda_reg

    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        firing_strengths: np.ndarray
    ) -> np.ndarray:
        """
        Fits the consequent parameters for all rules.
        Equation: \boldsymbol{\beta}_r = (\tilde{X}^\top W_r \tilde{X} + \lambda_{\text{reg}} I)^{-1} \tilde{X}^\top W_r Y
        
        :param X: Input data of shape (N, d).
        :param y: Target outputs of shape (N,).
        :param firing_strengths: Normalized firing strengths \phi_r(x_i) of shape (N, R).
        :return: Consequent parameters \boldsymbol{\beta}_r of shape (R, d + 1).
        """
        N, d = X.shape
        R = firing_strengths.shape[1]
        
        # Augment X with a column of ones: \tilde{X} = [1, X]
        X_aug = np.hstack([np.ones((N, 1)), X])  # (N, d + 1)
        
        betas = np.zeros((R, d + 1))
        
        # Precompute \tilde{X}^\top \tilde{X} structure to optimize, though W_r changes per rule
        # We compute per rule to maintain strict adherence to the WLS formulation.
        for r in range(R):
            # W_r = \text{diag}(\phi_r(x_1), \dots, \phi_r(x_n))
            # To avoid creating a full (N, N) diagonal matrix, we use element-wise multiplication.
            # \tilde{X}^\top W_r \tilde{X} = (X_aug * \sqrt{W_r})^\top (X_aug * \sqrt{W_r})
            # However, standard formulation:
            phi_r = firing_strengths[:, r]  # (N,)
            
            # X_aug.T @ W_r @ X_aug
            XtWX = (X_aug.T * phi_r) @ X_aug  # (d + 1, d + 1)
            
            # Add Tikhonov regularization: \lambda_{\text{reg}} I
            XtWX += self.lambda_reg * np.eye(d + 1)
            
            # \tilde{X}^\top W_r Y
            XtWy = (X_aug.T * phi_r) @ y  # (d + 1,)
            
            # Solve the linear system: \boldsymbol{\beta}_r = XtWX^{-1} XtWy
            # Using np.linalg.solve for numerical stability instead of explicit inversion
            betas[r] = np.linalg.solve(XtWX, XtWy)
            
        return betas
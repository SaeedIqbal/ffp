import numpy as np
from typing import Tuple, Dict, List

class DataPreprocessor:
    """
    Handles feature scaling, target clipping, and data splitting.
    Ensures inputs satisfy the bounded domain assumptions: X \subset [0, 1]^d and |y| <= B_y.
    """

    def __init__(self, clip_y: float = 10.0, test_ratio: float = 0.2, val_ratio: float = 0.1):
        """
        :param clip_y: Maximum absolute value for the target variable (B_y).
        :param test_ratio: Fraction of data reserved for the hold-out test set.
        :param val_ratio: Fraction of data reserved for validation.
        """
        self.clip_y = clip_y
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.feature_min = None
        self.feature_max = None

    def fit(self, X: np.ndarray) -> None:
        """
        Computes the min and max bounds for Min-Max scaling to [0, 1]^d.
        """
        self.feature_min = np.min(X, axis=0)
        self.feature_max = np.max(X, axis=0)
        # Prevent division by zero for constant features
        self.feature_range = np.maximum(self.feature_max - self.feature_min, 1e-8)

    def transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies Min-Max scaling to features and clips the target variable.
        
        Equations:
        x' = (x - min) / (max - min)
        y' = clip(y, -B_y, B_y)
        """
        if self.feature_min is None:
            self.fit(X)
            
        X_scaled = (X - self.feature_min) / self.feature_range
        y_clipped = np.clip(y, -self.clip_y, self.clip_y)
        
        return X_scaled, y_clipped

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fits the scaler and transforms the data in one step."""
        self.fit(X)
        return self.transform(X, y)

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Splits the dataset into training, validation, and test sets.
        """
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        test_size = int(n_samples * self.test_ratio)
        val_size = int(n_samples * self.val_ratio)
        
        test_idx = indices[:test_size]
        val_idx = indices[test_size:test_size + val_size]
        train_idx = indices[test_size + val_size:]
        
        return {
            'train': (X[train_idx], y[train_idx]),
            'val': (X[val_idx], y[val_idx]),
            'test': (X[test_idx], y[test_idx])
        }
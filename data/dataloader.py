import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from .preprocessing import DataPreprocessor

class FederatedDataLoader:
    """
    Loads datasets and partitions them into K federated clients.
    Supports 'california', 'elevators', and 'anacalt' datasets.
    """

    DATASET_PATHS = {
        'california': 'california.csv',
        'elevators': 'elevators.csv',
        'anacalt': 'anacalt.csv'
    }

    def __init__(self, root_dir: str, clip_y: float = 10.0):
        """
        :param root_dir: Root directory containing the dataset CSV files.
        :param clip_y: Target clipping bound (B_y).
        """
        self.root_dir = root_dir
        self.clip_y = clip_y
        self.preprocessor = DataPreprocessor(clip_y=clip_y)

    def load_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads a specific dataset from the root directory.
        Assumes the last column is the target variable y.
        """
        if dataset_name not in self.DATASET_PATHS:
            raise ValueError(f"Dataset {dataset_name} not supported. Choose from {list(self.DATASET_PATHS.keys())}")
        
        file_path = os.path.join(self.root_dir, self.DATASET_PATHS[dataset_name])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        
        df = pd.read_csv(file_path)
        data = df.values
        
        X = data[:, :-1].astype(np.float32)
        y = data[:, -1].astype(np.float32)
        
        return X, y

    def partition_clients(self, X: np.ndarray, y: np.ndarray, num_clients: int) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Partitions the global dataset into num_clients local datasets.
        Uses a simple random partition for IID simulation. 
        (Can be extended for Non-IID via Dirichlet distribution if required).
        """
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        # Split indices into K roughly equal parts
        client_indices = np.array_split(indices, num_clients)
        
        client_data = []
        for idx in client_indices:
            X_client, y_client = self.preprocessor.transform(X[idx], y[idx])
            splits = self.preprocessor.split_data(X_client, y_client)
            client_data.append(splits)
            
        return client_data

    def load_and_partition(self, dataset_name: str, num_clients: int) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Complete pipeline: Load, preprocess, and partition for federated learning.
        """
        X, y = self.load_dataset(dataset_name)
        # Fit preprocessor on the global data to ensure consistent [0, 1] scaling across clients
        X_scaled, y_clipped = self.preprocessor.fit_transform(X, y)
        
        # Re-split the globally scaled data
        global_splits = self.preprocessor.split_data(X_scaled, y_clipped)
        X_train, y_train = global_splits['train']
        
        # Partition the training data among clients
        n_train = X_train.shape[0]
        indices = np.random.permutation(n_train)
        client_indices = np.array_split(indices, num_clients)
        
        client_data = []
        for idx in client_indices:
            client_data.append({
                'train': (X_train[idx], y_train[idx]),
                'val': global_splits['val'],
                'test': global_splits['test']
            })
            
        return client_data
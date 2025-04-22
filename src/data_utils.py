"""
Data utilities for loading and processing BERT embeddings.

This module provides functions for loading and processing the pre-computed
BERT embeddings from the CLS_fea.txt file.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split


def load_bert_embeddings(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed BERT embeddings from a file.
    
    Parameters
    ----------
    file_path : str
        Path to the file containing pre-computed BERT embeddings.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - X: Feature matrix of shape (n_samples, n_features)
        - y: Target vector of shape (n_samples,)
    """
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse the embeddings
    embeddings = []
    for line in lines:
        # Split the line by commas
        values = line.strip().split(',')
        # Convert to float
        values = [float(val) for val in values]
        embeddings.append(values)
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # The first column is the label (1 for positive, 0 for negative)
    y = embeddings[:, 0].astype(int)
    
    # The rest of the columns are the features
    X = embeddings[:, 1:]
    
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    test_size : float, optional
        Proportion of the dataset to include in the test split.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - X_train: Training feature matrix
        - X_test: Testing feature matrix
        - y_train: Training target vector
        - y_test: Testing target vector
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def save_predictions(predictions: np.ndarray, probabilities: Optional[np.ndarray] = None, 
                     output_file: str = 'predictions.csv') -> None:
    """
    Save predictions to a CSV file.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted labels.
    probabilities : np.ndarray, optional
        Predicted probabilities for the positive class.
    output_file : str, optional
        Path to the output CSV file.
    """
    # Create a DataFrame
    if probabilities is not None:
        df = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities
        })
    else:
        df = pd.DataFrame({
            'prediction': predictions
        })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")


def load_test_data(file_path: str) -> np.ndarray:
    """
    Load test data for prediction.
    
    Parameters
    ----------
    file_path : str
        Path to the file containing test data.
        
    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_samples, n_features).
    """
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse the embeddings
    embeddings = []
    for line in lines:
        # Split the line by commas
        values = line.strip().split(',')
        # Convert to float
        values = [float(val) for val in values]
        embeddings.append(values)
    
    # Convert to numpy array
    embeddings = np.array(embeddings)
    
    # If the first column might be a label, remove it
    if embeddings.shape[1] == 1025:  # 1024 features + 1 label
        X = embeddings[:, 1:]
    else:
        X = embeddings
    
    return X

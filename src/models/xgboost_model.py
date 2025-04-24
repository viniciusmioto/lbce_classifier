"""
XGBoost model implementation for LBCE-BERT MVP.

This module implements the XGBoost model for predicting linear B-cell epitopes
from protein sequences using the features extracted from the sequences.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score


class XGBoostModel:
    """
    XGBoost model for predicting linear B-cell epitopes.
    
    This class implements the XGBoost classifier for predicting whether a protein
    sequence is a linear B-cell epitope based on the features extracted from the sequence.
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the XGBoost model.
        
        Parameters
        ----------
        params : Dict, optional
            XGBoost model parameters. If None, default parameters will be used.
        """
        # Default parameters based on the paper
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'eta': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 1,
            'seed': 42
        }
        
        # Use provided parameters or defaults
        self.params = params if params is not None else self.default_params
        
        # Initialize model
        self.model = None
    
    def train(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              early_stopping_rounds: int = 10, verbose: bool = True) -> None:
        """
        Train the XGBoost model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target vector of shape (n_samples,).
        eval_set : Tuple[np.ndarray, np.ndarray], optional
            Validation set for early stopping.
        early_stopping_rounds : int, optional
            Number of rounds with no improvement before early stopping.
        verbose : bool, optional
            Whether to print training progress.
        """
        # Create DMatrix for training
        dtrain = xgb.DMatrix(X, label=y)
        
        # Create validation set if provided
        watchlist = []
        if eval_set is not None:
            X_val, y_val = eval_set
            dval = xgb.DMatrix(X_val, label=y_val)
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,  # Can be adjusted based on dataset size
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds if watchlist else None,
            verbose_eval=verbose
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions using the trained model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Binary predictions of shape (n_samples,).
            
        Raises
        ------
        ValueError
            If the model has not been trained.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X)
        
        # Make predictions
        y_pred_prob = self.model.predict(dtest)
        
        # Convert probabilities to binary predictions
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions using the trained model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Probability predictions of shape (n_samples,).
            
        Raises
        ------
        ValueError
            If the model has not been trained.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X)
        
        # Make predictions
        y_pred_prob = self.model.predict(dtest)
        
        return y_pred_prob
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target vector of shape (n_samples,).
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing evaluation metrics.
            
        Raises
        ------
        ValueError
            If the model has not been trained.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Make predictions
        y_pred = self.predict(X)
        y_pred_prob = self.predict_proba(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'sensitivity': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'mcc': matthews_corrcoef(y, y_pred),
            'auroc': roc_auc_score(y, y_pred_prob)
        }
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target vector of shape (n_samples,).
        n_splits : int, optional
            Number of folds for cross-validation.
            
        Returns
        -------
        Dict[str, List[float]]
            Dictionary containing lists of evaluation metrics for each fold.
        """
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Initialize metrics
        metrics = {
            'accuracy': [],
            'precision': [],
            'sensitivity': [],
            'f1': [],
            'mcc': [],
            'auroc': []
        }
        
        # Perform cross-validation
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            self.train(X_train, y_train, verbose=False)
            
            # Evaluate model
            fold_metrics = self.evaluate(X_test, y_test)
            
            # Append metrics
            for metric, value in fold_metrics.items():
                metrics[metric].append(value)
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the model.
            
        Raises
        ------
        ValueError
            If the model has not been trained.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model.
            
        Raises
        ------
        FileNotFoundError
            If the model file is not found.
        """
        self.model = xgb.Booster()
        self.model.load_model(filepath)

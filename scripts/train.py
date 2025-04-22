#!/usr/bin/env python
"""
Training script for the simplified LBCE-BERT model.

This script trains an XGBoost model on pre-computed BERT embeddings
for linear B-cell epitope prediction.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

# Add parent directory to path for importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_bert_embeddings, split_data
from src.model import XGBoostModel
from src.evaluation import calculate_metrics, print_metrics, plot_roc_curve, plot_confusion_matrix


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train XGBoost model on BERT embeddings')
    parser.add_argument('--data_file', type=str, default='data/CLS_fea.txt',
                        help='Path to the file containing BERT embeddings')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save the trained model and results')
    parser.add_argument('--output_model', type=str, default='lbce_bert_model.pkl',
                        help='Filename for the trained model')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--cross_validate', action='store_true',
                        help='Perform cross-validation')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for cross-validation')
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load BERT embeddings
    print(f"Loading BERT embeddings from {args.data_file}...")
    X, y = load_bert_embeddings(args.data_file)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y == 1)} positive, {np.sum(y == 0)} negative")
    
    # Split data into training and testing sets
    print(f"Splitting data with test_size={args.test_size}...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size, random_state=args.random_state)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Initialize model
    print("Initializing XGBoost model...")
    model = XGBoostModel()
    
    # Cross-validation
    if args.cross_validate:
        print(f"Performing {args.n_splits}-fold cross-validation...")
        cv_metrics = model.cross_validate(X_train, y_train, n_splits=args.n_splits)
        
        # Print average metrics
        print("Cross-validation results:")
        for metric, values in cv_metrics.items():
            print(f"  {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Train model
    print("Training model...")
    model.train(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10)
    
    # Evaluate model
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_prob)
    print_metrics(metrics)
    
    # Plot ROC curve
    roc_output_file = './results/roc_curve.png'
    plot_roc_curve(y_test, y_pred_prob, roc_output_file)
    print(f"ROC curve saved to {roc_output_file}")
    
    # Plot confusion matrix
    cm_output_file = './results/confusion_matrix.png'
    plot_confusion_matrix(y_test, y_pred, cm_output_file)
    print(f"Confusion matrix saved to {cm_output_file}")
    
    # Save model
    model_path = os.path.join(args.output_dir, args.output_model)
    model.save_model(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

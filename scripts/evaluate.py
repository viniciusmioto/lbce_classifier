#!/usr/bin/env python
"""
Evaluation script for the simplified LBCE-BERT model.

This script evaluates the performance of a trained XGBoost model
on pre-computed BERT embeddings for linear B-cell epitope prediction.
"""

import os
import argparse
import numpy as np
import sys

# Add parent directory to path for importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_bert_embeddings
from src.model import XGBoostModel
from src.evaluation import calculate_metrics, print_metrics, plot_roc_curve, plot_confusion_matrix, save_metrics_to_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained XGBoost model')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the file containing BERT embeddings with labels for evaluation')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    X, y = load_bert_embeddings(args.data_file)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y == 1)} positive, {np.sum(y == 0)} negative")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = XGBoostModel()
    model.load_model(args.model)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = calculate_metrics(y, y_pred, y_pred_prob)
    print_metrics(metrics)
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, 'evaluation_metrics.txt')
    save_metrics_to_file(metrics, metrics_file)
    print(f"Metrics saved to {metrics_file}")
    
    # Plot ROC curve
    roc_file = os.path.join(args.output_dir, 'roc_curve.png')
    plot_roc_curve(y, y_pred_prob, roc_file)
    print(f"ROC curve saved to {roc_file}")
    
    # Plot confusion matrix
    cm_file = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y, y_pred, cm_file)
    print(f"Confusion matrix saved to {cm_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Prediction script for the simplified LBCE-BERT model.

This script makes predictions using a trained XGBoost model
on pre-computed BERT embeddings for linear B-cell epitope prediction.
"""

import os
import argparse
import numpy as np
import sys

# Add parent directory to path for importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_test_data, save_predictions
from src.model import XGBoostModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions using trained XGBoost model')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the file containing BERT embeddings for prediction')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to save the predictions')
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load test data
    print(f"Loading test data from {args.data_file}...")
    X = load_test_data(args.data_file)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = XGBoostModel()
    model.load_model(args.model)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)
    
    # Save predictions
    save_predictions(y_pred, y_pred_prob, args.output)
    
    # Print summary
    print("Prediction summary:")
    print(f"  Total samples: {len(y_pred)}")
    print(f"  Predicted positive: {np.sum(y_pred == 1)} ({np.sum(y_pred == 1) / len(y_pred) * 100:.2f}%)")
    print(f"  Predicted negative: {np.sum(y_pred == 0)} ({np.sum(y_pred == 0) / len(y_pred) * 100:.2f}%)")


if __name__ == "__main__":
    main()

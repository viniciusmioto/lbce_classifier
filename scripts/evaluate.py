#!/usr/bin/env python
"""
Evaluation script for LBCE-BERT MVP.

This script evaluates the performance of the trained LBCE-BERT model on a test dataset.
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# Import local modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features.aac import AACFeatureExtractor
from src.features.aap import AAPFeatureExtractor
from src.features.aat import AATFeatureExtractor
from src.features.bert import BERTEmbeddingsLoader
from src.models.xgboost_model import XGBoostModel
from src.utils.data_loader import load_dataset_from_csv
from src.utils.evaluation import calculate_metrics, print_metrics, plot_roc_curve, plot_confusion_matrix, save_metrics_to_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate LBCE-BERT model')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data CSV file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--embeddings_dir', type=str, default='data/embeddings',
                        help='Directory containing pre-computed BERT embeddings')
    parser.add_argument('--dataset', type=str, default='bcpreds',
                        help='Dataset name for BERT embeddings (default: bcpreds)')
    
    return parser.parse_args()


def extract_features(sequences: List[str], seq_ids: List[str], dataset_name: str, embeddings_dir: str) -> np.ndarray:
    """
    Extract features from protein sequences.
    
    Parameters
    ----------
    sequences : List[str]
        List of protein sequences.
    seq_ids : List[str]
        List of sequence IDs.
    dataset_name : str
        Name of the dataset for BERT embeddings.
    embeddings_dir : str
        Directory containing pre-computed BERT embeddings.
        
    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_samples, n_features).
    """
    print("Extracting features...")
    
    # Initialize feature extractors
    aac_extractor = AACFeatureExtractor()
    aap_extractor = AAPFeatureExtractor()
    aat_extractor = AATFeatureExtractor()
    bert_loader = BERTEmbeddingsLoader(embeddings_dir)
    
    # Extract AAC features
    print("  Extracting AAC features...")
    aac_features = aac_extractor.extract_batch(sequences)
    
    # Extract AAP features
    print("  Extracting AAP features...")
    aap_features = aap_extractor.extract_batch(sequences)
    
    # Extract AAT features
    print("  Extracting AAT features...")
    aat_features = aat_extractor.extract_batch(sequences)
    
    # Load BERT embeddings
    print("  Loading BERT embeddings...")
    try:
        _, bert_embeddings = bert_loader.map_sequences_to_embeddings(sequences, dataset_name)
    except Exception as e:
        print(f"  Warning: Failed to load BERT embeddings: {e}")
        print("  Using dummy BERT embeddings for demonstration purposes.")
        # Create dummy embeddings
        bert_embeddings = np.random.normal(0, 1, (len(sequences), bert_loader.get_embedding_dimension()))
    
    # Combine features
    print("  Combining features...")
    features = np.hstack([aac_features, aap_features, aat_features, bert_embeddings])
    
    print(f"  Feature matrix shape: {features.shape}")
    
    return features


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    seq_ids, sequences, labels = load_dataset_from_csv(args.test_data)
    print(f"  Loaded {len(sequences)} sequences ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")
    
    # Extract features
    features = extract_features(sequences, seq_ids, args.dataset, args.embeddings_dir)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = XGBoostModel()
    model.load_model(args.model)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(features)
    y_pred_prob = model.predict_proba(features)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = calculate_metrics(labels, y_pred, y_pred_prob)
    print_metrics(metrics)
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, 'evaluation_metrics.txt')
    save_metrics_to_file(metrics, metrics_file)
    print(f"  Saved metrics to {metrics_file}")
    
    # Plot ROC curve
    roc_file = os.path.join(args.output_dir, 'roc_curve.png')
    plot_roc_curve(labels, y_pred_prob, roc_file)
    print(f"  Saved ROC curve to {roc_file}")
    
    # Plot confusion matrix
    cm_file = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(labels, y_pred, cm_file)
    print(f"  Saved confusion matrix to {cm_file}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'seq_id': seq_ids,
        'sequence': sequences,
        'true_label': labels,
        'predicted_label': y_pred,
        'probability': y_pred_prob
    })
    results_file = os.path.join(args.output_dir, 'detailed_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"  Saved detailed results to {results_file}")


if __name__ == "__main__":
    main()

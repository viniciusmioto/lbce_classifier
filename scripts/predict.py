#!/usr/bin/env python
"""
Prediction script for LBCE-BERT MVP.

This script makes predictions using the trained LBCE-BERT model.
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions using LBCE-BERT model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input file containing protein sequences (one per line)')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output CSV file for predictions')
    parser.add_argument('--embeddings_dir', type=str, default='data/embeddings',
                        help='Directory containing pre-computed BERT embeddings')
    parser.add_argument('--dataset', type=str, default='avg_fea.txt',
                        help='Dataset name for BERT embeddings (default: avg_fea.txt)')
    
    return parser.parse_args()


def load_sequences(input_file: str) -> List[str]:
    """
    Load protein sequences from a file.
    
    Parameters
    ----------
    input_file : str
        Path to input file containing protein sequences (one per line).
        
    Returns
    -------
    List[str]
        List of protein sequences.
        
    Raises
    ------
    FileNotFoundError
        If the input file is not found.
    """
    with open(input_file, 'r') as f:
        sequences = [line.strip() for line in f.readlines() if line.strip()]
    
    return sequences


def extract_features(sequences: List[str], dataset_name: str, embeddings_dir: str) -> np.ndarray:
    """
    Extract features from protein sequences.
    
    Parameters
    ----------
    sequences : List[str]
        List of protein sequences.
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
        # Generate sequence IDs for mapping
        seq_ids = [f"seq_{i}" for i in range(len(sequences))]
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
    
    # Load sequences
    print(f"Loading sequences from {args.input}...")
    sequences = load_sequences(args.input)
    print(f"  Loaded {len(sequences)} sequences")
    
    # Extract features
    features = extract_features(sequences, args.dataset, args.embeddings_dir)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = XGBoostModel()
    model.load_model(args.model)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(features)
    y_pred_prob = model.predict_proba(features)
    
    # Create output DataFrame
    df = pd.DataFrame({
        'sequence': sequences,
        'prediction': y_pred,
        'probability': y_pred_prob
    })
    
    # Save predictions
    df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
    
    # Print summary
    print("Prediction summary:")
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Predicted epitopes: {sum(y_pred)} ({sum(y_pred) / len(y_pred) * 100:.2f}%)")
    print(f"  Predicted non-epitopes: {len(y_pred) - sum(y_pred)} ({(len(y_pred) - sum(y_pred)) / len(y_pred) * 100:.2f}%)")


if __name__ == "__main__":
    main()

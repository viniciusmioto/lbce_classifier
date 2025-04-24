#!/usr/bin/env python
"""
Simplified training script for LBCE-BERT MVP with a single embedding file and multi-sequence input files.

Usage example:
    python scripts/train.py \
      --protein_files data/proteins/neg.txt data/proteins/pos.txt \
      --embedding_file data/embeddings/CLS_fea.txt

Each protein file can contain multiple sequences (one per line). The embedding file must have one label+embedding vector per sequence, in the same order.
"""
import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure project root is on PYTHONPATH so `src` can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.aac import AACFeatureExtractor
from src.features.aap import AAPFeatureExtractor
from src.features.aat import AATFeatureExtractor
from src.models.xgboost_model import XGBoostModel
from src.utils.evaluation import calculate_metrics, print_metrics, plot_roc_curve, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Train LBCE-BERT model')
    parser.add_argument(
        '--protein_files', '-p', nargs='+', required=True,
        help='Paths to one or more protein sequence files (.txt), one sequence per line'
    )
    parser.add_argument(
        '--embedding_file', '-e', required=True,
        help='Path to a single BERT embedding file (.txt) with labels in the first column, comma-separated'
    )
    parser.add_argument(
        '--test_size', '-t', type=float, default=0.2,
        help='Fraction of data to reserve as test set (default: 0.2)'
    )
    parser.add_argument(
        '--random_state', '-r', type=int, default=42,
        help='Random seed for reproducibility'
    )
    return parser.parse_args()


def load_proteins(paths):
    """Load all sequences from given files, returning list of (seq_id, sequence)."""
    samples = []
    for path in paths:
        base = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            for idx, line in enumerate(f):
                seq = line.strip()
                if not seq:
                    continue
                seq_id = f"{base}_{idx}"
                samples.append((seq_id, seq))
    print(f"Loaded {len(samples)} sequences from {len(paths)} file(s)")
    return samples


def load_embeddings(path):
    """Load labels and embeddings from CSV-style file (no header, comma-separated)."""
    try:
        data = np.loadtxt(path, delimiter=',')
    except ValueError:
        import pandas as pd
        data = pd.read_csv(path, header=None).values
    labels = data[:, 0].astype(int)
    embeddings = data[:, 1:]
    print(f"Loaded {len(labels)} labels and {embeddings.shape[1]} embedding features.")
    return labels, embeddings


def extract_one(sequence, embedding):
    """Extract and concatenate AAC, AAP, AAT, and BERT features for one sample."""
    aac = AACFeatureExtractor().extract(sequence)
    aap = AAPFeatureExtractor().extract(sequence)
    aat = AATFeatureExtractor().extract(sequence)
    return np.hstack([aac, aap, aat, embedding])


def main():
    args = parse_args()

    # Load sequences
    samples = load_proteins(args.protein_files)
    seq_ids, sequences = zip(*samples)

    # Load embeddings and labels
    labels, embeddings = load_embeddings(args.embedding_file)

    # Validate counts
    n = len(sequences)
    if n != len(labels) or n != embeddings.shape[0]:
        sys.exit(
            f"Error: sequence count ({n}) != labels ({len(labels)}) or embeddings rows ({embeddings.shape[0]})"
        )

    # Extract features
    X = np.vstack([
        extract_one(seq, emb)
        for seq, emb in zip(sequences, embeddings)
    ])
    y = labels
    print(f"Total samples: {len(y)} (positives: {y.sum()}, negatives: {len(y)-y.sum()})")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # Train model
    print("Training XGBoost model...")
    model = XGBoostModel()
    model.train(X_train, y_train)

    # Evaluate
    print("Evaluating on test set...")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    metrics = calculate_metrics(y_test, preds, probs)
    print_metrics(metrics)

    # Save model and plots
    out_dir = 'models'
    os.makedirs(out_dir, exist_ok=True)
    model_file = os.path.join(out_dir, 'lbce_bert_model.pkl')
    model.save_model(model_file)
    print(f"Model saved to {model_file}")

    roc_file = os.path.join(out_dir, 'roc.png')
    cm_file = os.path.join(out_dir, 'cm.png')
    plot_roc_curve(y_test, probs, roc_file)
    plot_confusion_matrix(y_test, preds, cm_file)
    print(f"ROC curve and confusion matrix saved to {out_dir}")


if __name__ == '__main__':
    main()

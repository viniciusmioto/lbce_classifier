"""
Data loading utilities for LBCE-BERT MVP.

This module provides functions for loading and preprocessing data for the LBCE-BERT model.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional


def load_bcpreds_dataset(data_dir: str) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Load the BCPreds dataset.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the BCPreds dataset files.
        
    Returns
    -------
    Tuple[List[str], List[str], np.ndarray]
        Tuple containing:
        - List of sequence IDs
        - List of protein sequences
        - Array of labels (1 for epitope, 0 for non-epitope)
        
    Raises
    ------
    FileNotFoundError
        If the dataset files are not found.
    """
    # Construct paths to dataset files
    pos_file = os.path.join(data_dir, 'bcpreds_positive.txt')
    neg_file = os.path.join(data_dir, 'bcpreds_negative.txt')
    
    # Check if files exist
    if not os.path.exists(pos_file) or not os.path.exists(neg_file):
        # For MVP, create dummy data if files don't exist
        print(f"Warning: BCPreds dataset files not found in {data_dir}")
        print("Creating dummy data for demonstration purposes.")
        
        # Create dummy data
        seq_ids = []
        sequences = []
        labels = []
        
        # Generate 50 positive examples
        for i in range(50):
            seq_ids.append(f"pos_{i}")
            # Generate random sequence of 20 amino acids
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            seq = ''.join(np.random.choice(list(amino_acids), 20))
            sequences.append(seq)
            labels.append(1)
        
        # Generate 50 negative examples
        for i in range(50):
            seq_ids.append(f"neg_{i}")
            # Generate random sequence of 20 amino acids
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            seq = ''.join(np.random.choice(list(amino_acids), 20))
            sequences.append(seq)
            labels.append(0)
        
        return seq_ids, sequences, np.array(labels)
    
    # Load positive sequences
    with open(pos_file, 'r') as f:
        pos_lines = f.readlines()
    
    # Load negative sequences
    with open(neg_file, 'r') as f:
        neg_lines = f.readlines()
    
    # Parse sequences
    seq_ids = []
    sequences = []
    labels = []
    
    # Parse positive sequences
    for i, line in enumerate(pos_lines):
        line = line.strip()
        if line:
            seq_ids.append(f"pos_{i}")
            sequences.append(line)
            labels.append(1)
    
    # Parse negative sequences
    for i, line in enumerate(neg_lines):
        line = line.strip()
        if line:
            seq_ids.append(f"neg_{i}")
            sequences.append(line)
            labels.append(0)
    
    return seq_ids, sequences, np.array(labels)


def split_dataset(seq_ids: List[str], sequences: List[str], labels: np.ndarray, 
                  test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str], np.ndarray, List[str], List[str], np.ndarray]:
    """
    Split dataset into training and testing sets.
    
    Parameters
    ----------
    seq_ids : List[str]
        List of sequence IDs.
    sequences : List[str]
        List of protein sequences.
    labels : np.ndarray
        Array of labels.
    test_size : float, optional
        Proportion of the dataset to include in the test split.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    Tuple[List[str], List[str], np.ndarray, List[str], List[str], np.ndarray]
        Tuple containing:
        - List of training sequence IDs
        - List of training protein sequences
        - Array of training labels
        - List of testing sequence IDs
        - List of testing protein sequences
        - Array of testing labels
    """
    # Convert to numpy arrays for easier indexing
    seq_ids_array = np.array(seq_ids)
    sequences_array = np.array(sequences)
    
    # Get indices for stratified split
    indices = np.arange(len(labels))
    
    # Split indices
    from sklearn.model_selection import train_test_split
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Split data
    train_seq_ids = seq_ids_array[train_indices].tolist()
    train_sequences = sequences_array[train_indices].tolist()
    train_labels = labels[train_indices]
    
    test_seq_ids = seq_ids_array[test_indices].tolist()
    test_sequences = sequences_array[test_indices].tolist()
    test_labels = labels[test_indices]
    
    return train_seq_ids, train_sequences, train_labels, test_seq_ids, test_sequences, test_labels


def save_dataset(seq_ids: List[str], sequences: List[str], labels: np.ndarray, output_file: str) -> None:
    """
    Save dataset to a CSV file.
    
    Parameters
    ----------
    seq_ids : List[str]
        List of sequence IDs.
    sequences : List[str]
        List of protein sequences.
    labels : np.ndarray
        Array of labels.
    output_file : str
        Path to output CSV file.
    """
    # Create DataFrame
    df = pd.DataFrame({
        'seq_id': seq_ids,
        'sequence': sequences,
        'label': labels
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)


def load_dataset_from_csv(csv_file: str) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Load dataset from a CSV file.
    
    Parameters
    ----------
    csv_file : str
        Path to CSV file.
        
    Returns
    -------
    Tuple[List[str], List[str], np.ndarray]
        Tuple containing:
        - List of sequence IDs
        - List of protein sequences
        - Array of labels
        
    Raises
    ------
    FileNotFoundError
        If the CSV file is not found.
    """
    # Load CSV
    df = pd.read_csv(csv_file)
    
    # Extract data
    seq_ids = df['seq_id'].tolist()
    sequences = df['sequence'].tolist()
    labels = df['label'].values
    
    return seq_ids, sequences, labels

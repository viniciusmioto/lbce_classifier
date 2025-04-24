"""
BERT embeddings loader for LBCE-BERT MVP.

This module implements functionality to load pre-computed BERT embeddings
from the original LBCE-BERT repository.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple


class BERTEmbeddingsLoader:
    """
    Loads pre-computed BERT embeddings for protein sequences.
    
    This class handles loading and processing of pre-computed BERT embeddings
    from the original LBCE-BERT repository, eliminating the need to generate
    embeddings from scratch.
    """
    
    def __init__(self, embeddings_dir: str):
        """
        Initialize the BERT embeddings loader.
        
        Parameters
        ----------
        embeddings_dir : str
            Directory containing pre-computed BERT embeddings files.
        """
        self.embeddings_dir = embeddings_dir
        self.embedding_dim = 1024  # Default BERT embedding dimension
        self.embeddings_cache = {}  # Cache for loaded embeddings
        
    def load_embeddings(self, dataset_name: str) -> Dict[str, np.ndarray]:
        """
        Load pre-computed BERT embeddings for a specific dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset (e.g., 'BCPreds').
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping sequence IDs to their BERT embeddings.
            
        Raises
        ------
        FileNotFoundError
            If the embeddings file for the specified dataset is not found.
        """
        # Check if embeddings are already cached
        if dataset_name in self.embeddings_cache:
            return self.embeddings_cache[dataset_name]
        
        # Construct path to embeddings file
        embeddings_path = os.path.join(self.embeddings_dir, dataset_name)
        
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"BERT embeddings directory not found: {embeddings_path}")
        
        # Load embeddings from files
        embeddings = {}
        
        # In a real implementation, this would parse the actual file format
        # For MVP, we'll implement a simplified version that assumes a specific format
        try:
            # Look for positive and negative embedding files
            pos_file = os.path.join(embeddings_path, 'positive_embeddings.npy')
            neg_file = os.path.join(embeddings_path, 'negative_embeddings.npy')
            
            if os.path.exists(pos_file) and os.path.exists(neg_file):
                # Load numpy arrays
                pos_embeddings = np.load(pos_file)
                neg_embeddings = np.load(neg_file)
                
                # Create sequence IDs
                pos_ids = [f"pos_{i}" for i in range(pos_embeddings.shape[0])]
                neg_ids = [f"neg_{i}" for i in range(neg_embeddings.shape[0])]
                
                # Add to embeddings dictionary
                for i, seq_id in enumerate(pos_ids):
                    embeddings[seq_id] = pos_embeddings[i]
                
                for i, seq_id in enumerate(neg_ids):
                    embeddings[seq_id] = neg_embeddings[i]
            else:
                # Alternative: look for a combined embeddings file
                combined_file = os.path.join(embeddings_path, 'embeddings.npy')
                ids_file = os.path.join(embeddings_path, 'sequence_ids.txt')
                
                if os.path.exists(combined_file) and os.path.exists(ids_file):
                    # Load embeddings
                    all_embeddings = np.load(combined_file)
                    
                    # Load sequence IDs
                    with open(ids_file, 'r') as f:
                        seq_ids = [line.strip() for line in f.readlines()]
                    
                    # Add to embeddings dictionary
                    for i, seq_id in enumerate(seq_ids):
                        embeddings[seq_id] = all_embeddings[i]
                else:
                    print(f"Warning: Could not find embedding files in {embeddings_path}")
                    print("Using dummy embeddings for demonstration purposes.")
                    
                    # Create dummy embeddings for demonstration
                    np.random.seed(42)  # For reproducibility
                    dummy_count = 100  # Arbitrary number for demonstration
                    
                    for i in range(dummy_count):
                        if i < dummy_count // 2:
                            seq_id = f"pos_{i}"
                        else:
                            seq_id = f"neg_{i - dummy_count // 2}"
                        
                        embeddings[seq_id] = np.random.normal(0, 1, self.embedding_dim)
        
        except Exception as e:
            print(f"Error loading BERT embeddings: {e}")
            print("Using dummy embeddings for demonstration purposes.")
            
            # Create dummy embeddings for demonstration
            np.random.seed(42)  # For reproducibility
            dummy_count = 100  # Arbitrary number for demonstration
            
            for i in range(dummy_count):
                if i < dummy_count // 2:
                    seq_id = f"pos_{i}"
                else:
                    seq_id = f"neg_{i - dummy_count // 2}"
                
                embeddings[seq_id] = np.random.normal(0, 1, self.embedding_dim)
        
        # Cache the embeddings
        self.embeddings_cache[dataset_name] = embeddings
        
        return embeddings
    
    def get_embedding(self, sequence_id: str, dataset_name: str) -> np.ndarray:
        """
        Get the BERT embedding for a specific sequence.
        
        Parameters
        ----------
        sequence_id : str
            ID of the sequence.
        dataset_name : str
            Name of the dataset.
            
        Returns
        -------
        np.ndarray
            BERT embedding for the sequence.
            
        Raises
        ------
        KeyError
            If the sequence ID is not found in the embeddings.
        """
        embeddings = self.load_embeddings(dataset_name)
        
        if sequence_id not in embeddings:
            raise KeyError(f"Sequence ID '{sequence_id}' not found in embeddings for dataset '{dataset_name}'")
        
        return embeddings[sequence_id]
    
    def get_embeddings_batch(self, sequence_ids: List[str], dataset_name: str) -> np.ndarray:
        """
        Get BERT embeddings for a batch of sequences.
        
        Parameters
        ----------
        sequence_ids : List[str]
            List of sequence IDs.
        dataset_name : str
            Name of the dataset.
            
        Returns
        -------
        np.ndarray
            Array of BERT embeddings for the sequences.
        """
        embeddings = self.load_embeddings(dataset_name)
        
        # Get embeddings for each sequence ID
        batch_embeddings = []
        for seq_id in sequence_ids:
            if seq_id in embeddings:
                batch_embeddings.append(embeddings[seq_id])
            else:
                # If sequence ID not found, use zeros
                print(f"Warning: Sequence ID '{seq_id}' not found in embeddings for dataset '{dataset_name}'")
                batch_embeddings.append(np.zeros(self.embedding_dim))
        
        return np.array(batch_embeddings)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the BERT embeddings.
        
        Returns
        -------
        int
            Dimension of the BERT embeddings.
        """
        return self.embedding_dim
    
    def map_sequences_to_embeddings(self, sequences: List[str], dataset_name: str) -> Tuple[List[str], np.ndarray]:
        """
        Map protein sequences to their corresponding BERT embeddings.
        
        This is a simplified implementation for the MVP. In a real implementation,
        this would match sequences to their pre-computed embeddings based on sequence content.
        
        Parameters
        ----------
        sequences : List[str]
            List of protein sequences.
        dataset_name : str
            Name of the dataset.
            
        Returns
        -------
        Tuple[List[str], np.ndarray]
            Tuple containing list of sequence IDs and array of BERT embeddings.
        """
        # Load embeddings
        embeddings = self.load_embeddings(dataset_name)
        
        # Get list of sequence IDs
        seq_ids = list(embeddings.keys())
        
        # For MVP, we'll just assign embeddings in order
        # In a real implementation, this would match sequences to embeddings based on content
        assigned_ids = []
        assigned_embeddings = []
        
        for i, seq in enumerate(sequences):
            if i < len(seq_ids):
                assigned_ids.append(seq_ids[i])
                assigned_embeddings.append(embeddings[seq_ids[i]])
            else:
                # If we run out of embeddings, use zeros
                dummy_id = f"dummy_{i}"
                assigned_ids.append(dummy_id)
                assigned_embeddings.append(np.zeros(self.embedding_dim))
        
        return assigned_ids, np.array(assigned_embeddings)

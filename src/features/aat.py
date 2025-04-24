"""
Amino Acid Trimer (AAT) antigenicity scale feature extraction.

This module implements the Amino Acid Trimer antigenicity scale feature extraction method,
which calculates features based on the antigenicity values of amino acid triplets in a protein sequence.
"""

import numpy as np
from typing import List, Dict, Union


class AATFeatureExtractor:
    """
    Extracts Amino Acid Trimer (AAT) antigenicity scale features from protein sequences.
    
    AAT represents the antigenicity values of amino acid triplets in a protein sequence.
    This is an extension of the AAP method to consider three consecutive amino acids.
    """
    
    def __init__(self, aat_scale_file=None):
        """
        Initialize the AAT feature extractor.
        
        Parameters
        ----------
        aat_scale_file : str, optional
            Path to a file containing the AAT antigenicity scale values.
            If None, default values will be used.
        """
        # Standard amino acids
        self.amino_acids = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
        ]
        
        # Initialize AAT scale with default values
        # These values are simplified for the MVP and should be replaced with actual values from the paper
        # In a real implementation, these would be loaded from a file or database
        self.aat_scale = self._initialize_default_aat_scale()
        
        # Load custom scale if provided
        if aat_scale_file:
            self._load_aat_scale(aat_scale_file)
    
    def _initialize_default_aat_scale(self) -> Dict[str, float]:
        """
        Initialize default AAT scale values.
        
        Returns
        -------
        Dict[str, float]
            A dictionary mapping amino acid triplets to their antigenicity scale values.
        """
        # Create a subset of possible amino acid triplets (for simplicity in MVP)
        # In a real implementation, all 8000 triplets would be included
        aa_triplets = []
        for aa1 in self.amino_acids:
            for aa2 in self.amino_acids:
                for aa3 in self.amino_acids:
                    aa_triplets.append(f"{aa1}{aa2}{aa3}")
        
        # For MVP, initialize with random values between 0 and 1
        # In a real implementation, these would be the actual values from the paper
        np.random.seed(42)  # For reproducibility
        scale_values = np.random.uniform(0, 1, len(aa_triplets))
        
        return dict(zip(aa_triplets, scale_values))
    
    def _load_aat_scale(self, file_path: str) -> None:
        """
        Load AAT scale values from a file.
        
        Parameters
        ----------
        file_path : str
            Path to the file containing AAT scale values.
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            self.aat_scale = {}
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        aa_triplet = parts[0]
                        value = float(parts[1])
                        self.aat_scale[aa_triplet] = value
        except Exception as e:
            print(f"Error loading AAT scale file: {e}")
            print("Using default AAT scale values instead.")
    
    def extract(self, sequence: str) -> np.ndarray:
        """
        Extract AAT features from a protein sequence.
        
        Parameters
        ----------
        sequence : str
            A protein sequence consisting of amino acid letters.
            
        Returns
        -------
        np.ndarray
            A numpy array containing the AAT features.
        """
        # Convert sequence to uppercase
        sequence = sequence.upper()
        
        # Get all adjacent amino acid triplets in the sequence
        triplets = [sequence[i:i+3] for i in range(len(sequence)-2)]
        
        # Calculate the average AAT score for the sequence
        aat_scores = []
        for triplet in triplets:
            if triplet in self.aat_scale:
                aat_scores.append(self.aat_scale[triplet])
        
        # If no valid triplets found, return zeros
        if not aat_scores:
            return np.zeros(1)
        
        # Return the average AAT score as a feature
        return np.array([np.mean(aat_scores)])
    
    def extract_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Extract AAT features from a batch of protein sequences.
        
        Parameters
        ----------
        sequences : List[str]
            A list of protein sequences.
            
        Returns
        -------
        np.ndarray
            A numpy array of shape (n_sequences, 1) containing the AAT features.
        """
        return np.array([self.extract(seq) for seq in sequences])
    
    @property
    def feature_names(self) -> List[str]:
        """
        Get the names of the features.
        
        Returns
        -------
        List[str]
            A list of feature names.
        """
        return ["AAT_score"]
    
    @property
    def feature_dimension(self) -> int:
        """
        Get the dimension of the feature vector.
        
        Returns
        -------
        int
            The dimension of the feature vector.
        """
        return 1

"""
Amino Acid Composition (AAC) feature extraction.

This module implements the Amino Acid Composition feature extraction method,
which calculates the frequency of each amino acid in a protein sequence.
"""

import numpy as np
from typing import List, Dict, Union


class AACFeatureExtractor:
    """
    Extracts Amino Acid Composition (AAC) features from protein sequences.
    
    AAC represents the frequency of each of the 20 standard amino acids in a protein sequence.
    """
    
    def __init__(self):
        """Initialize the AAC feature extractor."""
        # Standard amino acids
        self.amino_acids = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
        ]
        
    def extract(self, sequence: str) -> np.ndarray:
        """
        Extract AAC features from a protein sequence.
        
        Parameters
        ----------
        sequence : str
            A protein sequence consisting of amino acid letters.
            
        Returns
        -------
        np.ndarray
            A numpy array of length 20 containing the frequency of each amino acid.
        """
        # Convert sequence to uppercase
        sequence = sequence.upper()
        
        # Initialize counts dictionary
        aa_count = {aa: 0 for aa in self.amino_acids}
        
        # Count occurrences of each amino acid
        for aa in sequence:
            if aa in aa_count:
                aa_count[aa] += 1
        
        # Calculate frequencies
        seq_length = len(sequence)
        aa_freq = np.array([aa_count[aa] / seq_length for aa in self.amino_acids])
        
        return aa_freq
    
    def extract_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Extract AAC features from a batch of protein sequences.
        
        Parameters
        ----------
        sequences : List[str]
            A list of protein sequences.
            
        Returns
        -------
        np.ndarray
            A numpy array of shape (n_sequences, 20) containing the AAC features.
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
        return [f"AAC_{aa}" for aa in self.amino_acids]
    
    @property
    def feature_dimension(self) -> int:
        """
        Get the dimension of the feature vector.
        
        Returns
        -------
        int
            The dimension of the feature vector.
        """
        return len(self.amino_acids)

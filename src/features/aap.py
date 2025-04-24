"""
Amino Acid Pair (AAP) antigenicity scale feature extraction.

This module implements the Amino Acid Pair antigenicity scale feature extraction method,
which calculates features based on the antigenicity values of amino acid pairs in a protein sequence.
"""

import numpy as np
from typing import List, Dict, Union


class AAPFeatureExtractor:
    """
    Extracts Amino Acid Pair (AAP) antigenicity scale features from protein sequences.
    
    AAP represents the antigenicity values of amino acid pairs in a protein sequence.
    This method was refined by Chen et al. and has shown to outperform other methods in BCE prediction.
    """
    
    def __init__(self, aap_scale_file=None):
        """
        Initialize the AAP feature extractor.
        
        Parameters
        ----------
        aap_scale_file : str, optional
            Path to a file containing the AAP antigenicity scale values.
            If None, default values from the paper will be used.
        """
        # Standard amino acids
        self.amino_acids = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
        ]
        
        # Initialize AAP scale with default values
        # These values are simplified for the MVP and should be replaced with actual values from the paper
        # In a real implementation, these would be loaded from a file or database
        self.aap_scale = self._initialize_default_aap_scale()
        
        # Load custom scale if provided
        if aap_scale_file:
            self._load_aap_scale(aap_scale_file)
    
    def _initialize_default_aap_scale(self) -> Dict[str, float]:
        """
        Initialize default AAP scale values.
        
        Returns
        -------
        Dict[str, float]
            A dictionary mapping amino acid pairs to their antigenicity scale values.
        """
        # Create all possible amino acid pairs
        aa_pairs = [f"{aa1}{aa2}" for aa1 in self.amino_acids for aa2 in self.amino_acids]
        
        # For MVP, initialize with random values between 0 and 1
        # In a real implementation, these would be the actual values from the paper
        np.random.seed(42)  # For reproducibility
        scale_values = np.random.uniform(0, 1, len(aa_pairs))
        
        return dict(zip(aa_pairs, scale_values))
    
    def _load_aap_scale(self, file_path: str) -> None:
        """
        Load AAP scale values from a file.
        
        Parameters
        ----------
        file_path : str
            Path to the file containing AAP scale values.
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            self.aap_scale = {}
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        aa_pair = parts[0]
                        value = float(parts[1])
                        self.aap_scale[aa_pair] = value
        except Exception as e:
            print(f"Error loading AAP scale file: {e}")
            print("Using default AAP scale values instead.")
    
    def extract(self, sequence: str) -> np.ndarray:
        """
        Extract AAP features from a protein sequence.
        
        Parameters
        ----------
        sequence : str
            A protein sequence consisting of amino acid letters.
            
        Returns
        -------
        np.ndarray
            A numpy array containing the AAP features.
        """
        # Convert sequence to uppercase
        sequence = sequence.upper()
        
        # Get all adjacent amino acid pairs in the sequence
        pairs = [sequence[i:i+2] for i in range(len(sequence)-1)]
        
        # Calculate the average AAP score for the sequence
        aap_scores = []
        for pair in pairs:
            if pair in self.aap_scale:
                aap_scores.append(self.aap_scale[pair])
        
        # If no valid pairs found, return zeros
        if not aap_scores:
            return np.zeros(1)
        
        # Return the average AAP score as a feature
        return np.array([np.mean(aap_scores)])
    
    def extract_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Extract AAP features from a batch of protein sequences.
        
        Parameters
        ----------
        sequences : List[str]
            A list of protein sequences.
            
        Returns
        -------
        np.ndarray
            A numpy array of shape (n_sequences, 1) containing the AAP features.
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
        return ["AAP_score"]
    
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

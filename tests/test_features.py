"""
Unit tests for feature extraction modules.

This module contains tests for the feature extraction modules (AAC, AAP, AAT, BERT).
"""

import unittest
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.aac import AACFeatureExtractor
from src.features.aap import AAPFeatureExtractor
from src.features.aat import AATFeatureExtractor
from src.features.bert import BERTEmbeddingsLoader


class TestAACFeatureExtractor(unittest.TestCase):
    """Tests for the AAC feature extractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = AACFeatureExtractor()
        self.sequence = "ACDEFGHIKLMNPQRSTVWY"  # All 20 standard amino acids
    
    def test_extract_single(self):
        """Test extraction of AAC features from a single sequence."""
        features = self.extractor.extract(self.sequence)
        
        # Check shape
        self.assertEqual(features.shape, (20,))
        
        # Check that all values are 0.05 (1/20 for each amino acid)
        self.assertTrue(np.allclose(features, 0.05))
    
    def test_extract_batch(self):
        """Test extraction of AAC features from a batch of sequences."""
        sequences = [self.sequence, self.sequence]
        features = self.extractor.extract_batch(sequences)
        
        # Check shape
        self.assertEqual(features.shape, (2, 20))
        
        # Check that all values are 0.05 (1/20 for each amino acid)
        self.assertTrue(np.allclose(features, 0.05))
    
    def test_feature_names(self):
        """Test that feature names are correctly generated."""
        names = self.extractor.feature_names
        
        # Check that there are 20 feature names
        self.assertEqual(len(names), 20)
        
        # Check that all feature names start with "AAC_"
        for name in names:
            self.assertTrue(name.startswith("AAC_"))


class TestAAPFeatureExtractor(unittest.TestCase):
    """Tests for the AAP feature extractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = AAPFeatureExtractor()
        self.sequence = "ACDEFGHIKLMNPQRSTVWY"  # All 20 standard amino acids
    
    def test_extract_single(self):
        """Test extraction of AAP features from a single sequence."""
        features = self.extractor.extract(self.sequence)
        
        # Check shape
        self.assertEqual(features.shape, (1,))
        
        # Check that the value is a float
        self.assertTrue(isinstance(features[0], np.float64))
    
    def test_extract_batch(self):
        """Test extraction of AAP features from a batch of sequences."""
        sequences = [self.sequence, self.sequence]
        features = self.extractor.extract_batch(sequences)
        
        # Check shape
        self.assertEqual(features.shape, (2, 1))
    
    def test_feature_names(self):
        """Test that feature names are correctly generated."""
        names = self.extractor.feature_names
        
        # Check that there is 1 feature name
        self.assertEqual(len(names), 1)
        
        # Check that the feature name is "AAP_score"
        self.assertEqual(names[0], "AAP_score")


class TestAATFeatureExtractor(unittest.TestCase):
    """Tests for the AAT feature extractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = AATFeatureExtractor()
        self.sequence = "ACDEFGHIKLMNPQRSTVWY"  # All 20 standard amino acids
    
    def test_extract_single(self):
        """Test extraction of AAT features from a single sequence."""
        features = self.extractor.extract(self.sequence)
        
        # Check shape
        self.assertEqual(features.shape, (1,))
        
        # Check that the value is a float
        self.assertTrue(isinstance(features[0], np.float64))
    
    def test_extract_batch(self):
        """Test extraction of AAT features from a batch of sequences."""
        sequences = [self.sequence, self.sequence]
        features = self.extractor.extract_batch(sequences)
        
        # Check shape
        self.assertEqual(features.shape, (2, 1))
    
    def test_feature_names(self):
        """Test that feature names are correctly generated."""
        names = self.extractor.feature_names
        
        # Check that there is 1 feature name
        self.assertEqual(len(names), 1)
        
        # Check that the feature name is "AAT_score"
        self.assertEqual(names[0], "AAT_score")


class TestBERTEmbeddingsLoader(unittest.TestCase):
    """Tests for the BERT embeddings loader."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a dummy embeddings directory for testing
        self.embeddings_dir = "/tmp/bert_embeddings"
        self.loader = BERTEmbeddingsLoader(self.embeddings_dir)
        self.dataset_name = "test_dataset"
        self.sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"]
    
    def test_get_embedding_dimension(self):
        """Test getting the embedding dimension."""
        dim = self.loader.get_embedding_dimension()
        
        # Check that the dimension is a positive integer
        self.assertTrue(isinstance(dim, int))
        self.assertTrue(dim > 0)
    
    def test_map_sequences_to_embeddings(self):
        """Test mapping sequences to embeddings."""
        # This test will use dummy embeddings since we don't have actual embeddings
        seq_ids, embeddings = self.loader.map_sequences_to_embeddings(self.sequences, self.dataset_name)
        
        # Check that the number of sequence IDs matches the number of sequences
        self.assertEqual(len(seq_ids), len(self.sequences))
        
        # Check that the embeddings shape is correct
        self.assertEqual(embeddings.shape, (len(self.sequences), self.loader.get_embedding_dimension()))


if __name__ == '__main__':
    unittest.main()

"""
Integration test for the LBCE-BERT MVP.

This module tests the integration between different components of the LBCE-BERT MVP.
"""

import unittest
import numpy as np
import os
import sys
import tempfile

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.aac import AACFeatureExtractor
from src.features.aap import AAPFeatureExtractor
from src.features.aat import AATFeatureExtractor
from src.features.bert import BERTEmbeddingsLoader
from src.models.xgboost_model import XGBoostModel
from src.utils.data_loader import split_dataset
from src.utils.evaluation import calculate_metrics


class TestIntegration(unittest.TestCase):
    """Integration tests for the LBCE-BERT MVP."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create dummy data for testing
        np.random.seed(42)
        self.seq_ids = [f"seq_{i}" for i in range(50)]
        self.sequences = ["".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), 20)) for _ in range(50)]
        self.labels = np.random.randint(0, 2, 50)
        
        # Split data
        self.train_seq_ids, self.train_sequences, self.train_labels, self.test_seq_ids, self.test_sequences, self.test_labels = split_dataset(
            self.seq_ids, self.sequences, self.labels, test_size=0.2, random_state=42
        )
        
        # Initialize feature extractors
        self.aac_extractor = AACFeatureExtractor()
        self.aap_extractor = AAPFeatureExtractor()
        self.aat_extractor = AATFeatureExtractor()
        self.bert_loader = BERTEmbeddingsLoader("/tmp/bert_embeddings")
        
        # Initialize model
        self.model = XGBoostModel()
    
    def test_end_to_end_pipeline(self):
        """Test the end-to-end pipeline from feature extraction to prediction."""
        # Extract features for training data
        train_aac_features = self.aac_extractor.extract_batch(self.train_sequences)
        train_aap_features = self.aap_extractor.extract_batch(self.train_sequences)
        train_aat_features = self.aat_extractor.extract_batch(self.train_sequences)
        
        # Simulate BERT embeddings for training data
        embedding_dim = 10
        train_bert_features = np.random.normal(0, 1, (len(self.train_sequences), embedding_dim))
        
        # Combine features for training data
        train_features = np.hstack([train_aac_features, train_aap_features, train_aat_features, train_bert_features])
        
        # Train model
        self.model.train(train_features, self.train_labels, verbose=False)
        
        # Extract features for test data
        test_aac_features = self.aac_extractor.extract_batch(self.test_sequences)
        test_aap_features = self.aap_extractor.extract_batch(self.test_sequences)
        test_aat_features = self.aat_extractor.extract_batch(self.test_sequences)
        
        # Simulate BERT embeddings for test data
        test_bert_features = np.random.normal(0, 1, (len(self.test_sequences), embedding_dim))
        
        # Combine features for test data
        test_features = np.hstack([test_aac_features, test_aap_features, test_aat_features, test_bert_features])
        
        # Make predictions
        y_pred = self.model.predict(test_features)
        y_pred_prob = self.model.predict_proba(test_features)
        
        # Calculate metrics
        metrics = calculate_metrics(self.test_labels, y_pred, y_pred_prob)
        
        # Check that metrics dictionary contains expected keys
        expected_keys = ['accuracy', 'precision', 'sensitivity', 'f1', 'mcc', 'auroc']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check that metric values are between 0 and 1
        for key, value in metrics.items():
            self.assertTrue(0 <= value <= 1)


if __name__ == '__main__':
    unittest.main()

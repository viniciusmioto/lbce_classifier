"""
Unit tests for utility functions.

This module contains tests for the utility functions (data_loader, evaluation).
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import split_dataset, save_dataset, load_dataset_from_csv
from src.utils.evaluation import calculate_metrics, save_metrics_to_file


class TestDataLoader(unittest.TestCase):
    """Tests for the data loader utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create dummy data for testing
        np.random.seed(42)
        self.seq_ids = [f"seq_{i}" for i in range(100)]
        self.sequences = ["".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), 20)) for _ in range(100)]
        self.labels = np.random.randint(0, 2, 100)
    
    def test_split_dataset(self):
        """Test splitting the dataset."""
        # Split dataset
        train_seq_ids, train_sequences, train_labels, test_seq_ids, test_sequences, test_labels = split_dataset(
            self.seq_ids, self.sequences, self.labels, test_size=0.2, random_state=42
        )
        
        # Check that the split sizes are correct
        self.assertEqual(len(train_seq_ids), 80)
        self.assertEqual(len(train_sequences), 80)
        self.assertEqual(len(train_labels), 80)
        self.assertEqual(len(test_seq_ids), 20)
        self.assertEqual(len(test_sequences), 20)
        self.assertEqual(len(test_labels), 20)
        
        # Check that the labels are binary (0 or 1)
        self.assertTrue(np.all(np.isin(train_labels, [0, 1])))
        self.assertTrue(np.all(np.isin(test_labels, [0, 1])))
    
    def test_save_and_load_dataset(self):
        """Test saving and loading the dataset."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            # Save dataset
            save_dataset(self.seq_ids, self.sequences, self.labels, tmp.name)
            
            # Load dataset
            loaded_seq_ids, loaded_sequences, loaded_labels = load_dataset_from_csv(tmp.name)
            
            # Check that the loaded data matches the original data
            self.assertEqual(len(loaded_seq_ids), len(self.seq_ids))
            self.assertEqual(len(loaded_sequences), len(self.sequences))
            self.assertEqual(len(loaded_labels), len(self.labels))
            
            # Check that the labels are the same
            np.testing.assert_array_equal(loaded_labels, self.labels)


class TestEvaluation(unittest.TestCase):
    """Tests for the evaluation utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create dummy data for testing
        np.random.seed(42)
        self.y_true = np.random.randint(0, 2, 100)
        self.y_pred = np.random.randint(0, 2, 100)
        self.y_prob = np.random.random(100)
    
    def test_calculate_metrics(self):
        """Test calculating evaluation metrics."""
        # Calculate metrics
        metrics = calculate_metrics(self.y_true, self.y_pred, self.y_prob)
        
        # Check that metrics dictionary contains expected keys
        expected_keys = ['accuracy', 'precision', 'sensitivity', 'f1', 'mcc', 'auroc']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check that metric values are between 0 and 1
        for key, value in metrics.items():
            self.assertTrue(0 <= value <= 1)
    
    def test_save_metrics_to_file(self):
        """Test saving metrics to a file."""
        # Calculate metrics
        metrics = calculate_metrics(self.y_true, self.y_pred, self.y_prob)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            # Save metrics
            save_metrics_to_file(metrics, tmp.name)
            
            # Check that the file exists and is not empty
            self.assertTrue(os.path.exists(tmp.name))
            self.assertTrue(os.path.getsize(tmp.name) > 0)
            
            # Read the file and check that it contains the metrics
            with open(tmp.name, 'r') as f:
                content = f.read()
                for key in metrics.keys():
                    self.assertIn(key, content.lower())


if __name__ == '__main__':
    unittest.main()

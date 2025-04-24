"""
Unit tests for model implementation.

This module contains tests for the XGBoost model implementation.
"""

import unittest
import numpy as np
import os
import sys
import tempfile

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.xgboost_model import XGBoostModel


class TestXGBoostModel(unittest.TestCase):
    """Tests for the XGBoost model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = XGBoostModel()
        
        # Create dummy data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 20)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 20)
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_train_and_predict(self):
        """Test training the model and making predictions."""
        # Train model
        self.model.train(self.X_train, self.y_train, verbose=False)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Check that predictions have the correct shape
        self.assertEqual(y_pred.shape, self.y_test.shape)
        
        # Check that predictions are binary (0 or 1)
        self.assertTrue(np.all(np.isin(y_pred, [0, 1])))
    
    def test_predict_proba(self):
        """Test making probability predictions."""
        # Train model
        self.model.train(self.X_train, self.y_train, verbose=False)
        
        # Make probability predictions
        y_pred_prob = self.model.predict_proba(self.X_test)
        
        # Check that predictions have the correct shape
        self.assertEqual(y_pred_prob.shape, self.y_test.shape)
        
        # Check that probabilities are between 0 and 1
        self.assertTrue(np.all(y_pred_prob >= 0))
        self.assertTrue(np.all(y_pred_prob <= 1))
    
    def test_evaluate(self):
        """Test evaluating the model."""
        # Train model
        self.model.train(self.X_train, self.y_train, verbose=False)
        
        # Evaluate model
        metrics = self.model.evaluate(self.X_test, self.y_test)
        
        # Check that metrics dictionary contains expected keys
        expected_keys = ['accuracy', 'precision', 'sensitivity', 'f1', 'mcc', 'auroc']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check that metric values are between 0 and 1
        for key, value in metrics.items():
            self.assertTrue(0 <= value <= 1)
    
    def test_cross_validate(self):
        """Test cross-validation."""
        # Perform cross-validation
        cv_metrics = self.model.cross_validate(self.X_train, self.y_train, n_splits=2)
        
        # Check that metrics dictionary contains expected keys
        expected_keys = ['accuracy', 'precision', 'sensitivity', 'f1', 'mcc', 'auroc']
        for key in expected_keys:
            self.assertIn(key, cv_metrics)
        
        # Check that each metric is a list of length n_splits
        for key, values in cv_metrics.items():
            self.assertEqual(len(values), 2)
    
    def test_save_and_load_model(self):
        """Test saving and loading the model."""
        # Train model
        self.model.train(self.X_train, self.y_train, verbose=False)
        
        # Make predictions before saving
        y_pred_before = self.model.predict(self.X_test)
        
        # Save model to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            model_path = tmp.name
            self.model.save_model(model_path)
            
            # Create a new model instance
            new_model = XGBoostModel()
            
            # Load the saved model
            new_model.load_model(model_path)
            
            # Make predictions with the loaded model
            y_pred_after = new_model.predict(self.X_test)
            
            # Check that predictions are the same
            np.testing.assert_array_equal(y_pred_before, y_pred_after)


if __name__ == '__main__':
    unittest.main()

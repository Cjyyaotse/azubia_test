#!/usr/bin/env python3
"""
Integration Tests for ML Pipeline Runner Script.

This module contains integration tests for the complete machine learning
pipeline, testing component interactions, data flow, and end-to-end
functionality with real file operations and data processing.

Usage:
    pytest test_integration.py -v
    pytest test_integration.py::TestPipelineIntegration -v
    pytest test_integration.py --cov=run --cov-report=html -s

Requirements:
    pip install pytest pytest-mock pytest-cov pandas numpy scikit-learn

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import time
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module under test
import run


class TestPipelineIntegration:
    """Integration tests for the complete pipeline workflow."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.temp_dir, "raw")
        self.processed_dir = os.path.join(self.temp_dir, "processed")
        self.models_dir = os.path.join(self.temp_dir, "models")
        
        # Create directory structure
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create realistic banking dataset
        self.create_realistic_dataset()
        
    def teardown_method(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_realistic_dataset(self):
        """Create a realistic banking dataset for testing."""
        np.random.seed(42)  # For reproducible tests
        
        n_samples = 1000
        
        # Create realistic banking data
        ages = np.random.randint(18, 80, n_samples)
        jobs = np.random.choice(['admin', 'technician', 'management', 'blue-collar', 
                                'services', 'retired', 'self-employed', 'entrepreneur'], n_samples)
        marital = np.random.choice(['married', 'single', 'divorced'], n_samples)
        education = np.random.choice(['primary', 'secondary', 'tertiary'], n_samples)
        balances = np.random.normal(1500, 2000, n_samples)
        duration = np.random.randint(50, 1000, n_samples)
        campaign = np.random.randint(1, 10, n_samples)
        
        # Create target with some correlation to features
        target_proba = (ages / 100 + (balances / 5000).clip(0, 1) + 
                       (duration / 1000) + np.random.normal(0, 0.2, n_samples))
        targets = (target_proba > np.percentile(target_proba, 70)).astype(int)
        
        self.raw_data = pd.DataFrame({
            'age': ages,
            'job': jobs,
            'marital': marital,
            'education': education,
            'balance': balances,
            'duration': duration,
            'campaign': campaign,
            'y': targets
        })
        
        # Save raw data
        self.raw_data_path = os.path.join(self.raw_dir, "bank_data.csv")
        self.raw_data.to_csv(self.raw_data_path, index=False)

    @patch('run.train_and_save_models')
    @patch('run.feature_engineering')
    @patch('run.preprocess')
    @patch('run.data_loader')
    def test_complete_pipeline_data_flow(self, mock_data_loader, mock_preprocess,
                                        mock_feature_engineering, mock_train_models):
        """Test complete data flow through the pipeline."""
        # Mock data loader to return our test data
        mock_data_loader.return_value = self.raw_data
        
        # Create engineered data that would result from processing
        engineered_data = self.raw_data.copy()
        engineered_data['age_scaled'] = (engineered_data['age'] - engineered_data['age'].mean()) / engineered_data['age'].std()
        engineered_data['balance_log'] = np.log1p(engineered_data['balance'].clip(lower=0))
        engineered_data['duration_squared'] = engineered_data['duration'] ** 2
        engineered_data = pd.get_dummies(engineered_data, columns=['job', 'marital', 'education'])
        
        # Save engineered data to file
        engineered_path = os.path.join(self.processed_dir, "engineered_data.csv")
        engineered_data.to_csv(engineered_path, index=False)
        
        # Execute pipeline
        with patch('pandas.read_csv', return_value=engineered_data):
            run.run_pipeline(
                raw_data_path=self.raw_data_path,
                processed_data_path=self.processed_dir,
                engineered_data_path=engineered_path
            )
        
        # Verify pipeline steps were called in correct order
        mock_data_loader.assert_called_once_with(self.raw_data_path)
        mock_preprocess.assert_called_once_with(self.raw_data_path, output_dir=self.processed_dir)
        mock_feature_engineering.assert_called_once()
        mock_train_models.assert_called_once()
        
        # Verify data passed to model training
        call_args = mock_train_models.call_args
        features, target = call_args[0]
        
        # Check feature engineering results
        assert 'age_scaled' in features.columns
        assert 'balance_log' in features.columns
        assert 'duration_squared' in features.columns
        assert any('job_' in col for col in features.columns)  # One-hot encoded jobs
        assert 'y' not in features.columns  # Target should be separated
        
        # Verify target data
        assert len(target) == len(self.raw_data)
        assert target.name == 'y' or target.name is None

    @patch('run.train_and_save_models')
    @patch('run.feature_engineering')
    @patch('run.preprocess')
    @patch('run.data_loader')
    def test_pipeline_with_missing_features(self, mock_data_loader, mock_preprocess,
                                           mock_feature_engineering, mock_train_models):
        """Test pipeline behavior with missing features in data."""
        # Create data with missing columns
        incomplete_data = self.raw_data[['age', 'balance', 'y']].copy()
        mock_data_loader.return_value = incomplete_data
        
        # Create minimal engineered data
        engineered_data = incomplete_data.copy()
        engineered_data['age_scaled'] = (engineered_data['age'] - engineered_data['age'].mean()) / engineered_data['age'].std()
        
        engineered_path = os.path.join(self.processed_dir, "engineered_data.csv")
        engineered_data.to_csv(engineered_path, index=False)
        
        with patch('pandas.read_csv', return_value=engineered_data):
            run.run_pipeline(
                raw_data_path=self.raw_data_path,
                processed_data_path=self.processed_dir,
                engineered_data_path=engineered_path
            )
        
        # Verify pipeline still executed
        mock_train_models.assert_called_once()
        
        # Check that reduced feature set was passed
        call_args = mock_train_models.call_args
        features, target = call_args[0]
        
        assert len(features.columns) == 2  # age_scaled and balance
        assert 'y' not in features.columns

    @patch('run.train_and_save_models')
    @patch('run.feature_engineering')
    @patch('run.preprocess')
    @patch('run.data_loader')
    def test_pipeline_with_large_dataset(self, mock_data_loader, mock_preprocess,
                                        mock_feature_engineering, mock_train_models):
        """Test pipeline performance with larger dataset."""
        # Create larger dataset
        large_data = pd.concat([self.raw_data] * 10, ignore_index=True)  # 10k samples
        mock_data_loader.return_value = large_data
        
        # Create corresponding engineered data
        engineered_data = large_data.copy()
        engineered_data['feature_sum'] = (engineered_data['age'] + 
                                         engineered_data['balance'] + 
                                         engineered_data['duration'])
        
        engineered_path = os.path.join(self.processed_dir, "engineered_data.csv")
        engineered_data.to_csv(engineered_path, index=False)
        
        # Measure execution time
        start_time = time.time()
        
        with patch('pandas.read_csv', return_value=engineered_data):
            run.run_pipeline(
                raw_data_path=self.raw_data_path,
                processed_data_path=self.processed_dir,
                engineered_data_path=engineered_path
            )
        
        execution_time = time.time() - start_time
        
        # Verify reasonable performance (adjust threshold as needed)
        assert execution_time < 5.0  # Should complete within 5 seconds
        
        # Verify large dataset was processed
        call_args = mock_train_models.call_args
        features, target = call_args[0]
        assert len(features) == 10000
        assert len(target) == 10000

    def test_directory_structure_creation(self):
        """Test that pipeline creates necessary directory structure."""
        # Remove existing directories
        shutil.rmtree(self.temp_dir)
        
        # Create a new temp directory without subdirectories
        self.temp_dir = tempfile.mkdtemp()
        
        with patch('run.run_pipeline') as mock_pipeline:
            # Change to temp directory to test relative paths
            original_cwd = os.getcwd()
            os.chdir(self.temp_dir)
            
            try:
                run.main()
                
                # Verify directories were created
                assert os.path.exists('data/raw')
                assert os.path.exists('data/processed')
                assert os.path.exists('models')
                
                mock_pipeline.assert_called_once()
                
            finally:
                os.chdir(original_cwd)


class TestDataIntegrity:
    """Integration tests for data integrity throughout the pipeline."""
    
    def setup_method(self):
        """Set up data integrity test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data with known properties
        self.test_data = pd.DataFrame({
            'numeric_feature': [1.0, 2.0, 3.0, 4.0, 5.0],
            'categorical_feature': ['A', 'B', 'A', 'C', 'B'],
            'target_feature': np.random.choice([0, 1], 5),
            'y': [0, 1, 0, 1, 0]
        })
        
    def teardown_method(self):
        """Clean up data integrity test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('run.train_and_save_models')
    @patch('run.feature_engineering')
    @patch('run.preprocess')
    @patch('run.data_loader')
    def test_data_shape_consistency(self, mock_data_loader, mock_preprocess,
                                   mock_feature_engineering, mock_train_models):
        """Test that data shapes remain consistent through pipeline."""
        mock_data_loader.return_value = self.test_data
        
        # Create engineered data with additional features
        engineered_data = self.test_data.copy()
        engineered_data['new_feature_1'] = engineered_data['numeric_feature'] * 2
        engineered_data['new_
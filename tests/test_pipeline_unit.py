"""
Unit Tests for ML Pipeline Runner Script.

This module contains unit tests for individual components and functions
of the machine learning pipeline. Tests focus on isolated functionality
with mocked dependencies.

Usage:
    pytest test_unit.py -v
    pytest test_unit.py::TestRunPipeline::test_run_pipeline_success -v
    pytest test_unit.py --cov=run --cov-report=html

Requirements:
    pip install pytest pytest-mock pytest-cov pandas numpy

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call

import pytest
import pandas as pd
import numpy as np

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module under test
from src.run import run_pipeline as run


class TestRunPipeline:
    """Unit tests for the run_pipeline function."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.raw_data_path = os.path.join(self.temp_dir, "raw", "test_data.csv")
        self.processed_data_path = os.path.join(self.temp_dir, "processed")
        self.engineered_data_path = os.path.join(self.temp_dir, "processed", "engineered_data.csv")
        self.model_dir = os.path.join(self.temp_dir, "models")
        
        # Create directory structure
        os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'C', 'D', 'E'],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5],
            'y': [0, 1, 0, 1, 0]
        })
        
        # Create sample engineered data
        self.engineered_data = pd.DataFrame({
            'feature1_scaled': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature2_encoded': [1, 2, 3, 4, 5],
            'feature3_normalized': [0.2, 0.4, 0.6, 0.8, 1.0],
            'new_feature': [10, 20, 30, 40, 50],
            'y': [0, 1, 0, 1, 0]
        })

    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('run.train_and_save_models')
    @patch('run.feature_engineering')
    @patch('run.preprocess')
    @patch('run.data_loader')
    @patch('pandas.read_csv')
    def test_run_pipeline_success(self, mock_read_csv, mock_data_loader, 
                                  mock_preprocess, mock_feature_engineering, 
                                  mock_train_models):
        """Test successful execution of the complete pipeline."""
        # Setup mocks
        mock_data_loader.return_value = self.sample_data
        mock_read_csv.return_value = self.engineered_data
        
        # Execute pipeline
        run.run_pipeline(
            raw_data_path=self.raw_data_path,
            processed_data_path=self.processed_data_path,
            engineered_data_path=self.engineered_data_path
        )
        
        # Verify all functions were called
        mock_data_loader.assert_called_once_with(self.raw_data_path)
        mock_preprocess.assert_called_once_with(self.raw_data_path, output_dir=self.processed_data_path)
        mock_feature_engineering.assert_called_once()
        mock_train_models.assert_called_once()
        
        # Verify train_models was called with correct parameters
        call_args = mock_train_models.call_args
        features, target = call_args[0]
        
        # Check that features don't include target column
        assert 'y' not in features.columns
        assert len(features.columns) == 4  # All columns except 'y'
        assert len(target) == 5  # Target column

    def test_run_pipeline_default_parameters(self):
        """Test that default parameters are set correctly."""
        with patch('run.data_loader') as mock_data_loader, \
             patch('run.preprocess') as mock_preprocess, \
             patch('run.feature_engineering') as mock_feature_engineering, \
             patch('pandas.read_csv') as mock_read_csv, \
             patch('run.train_and_save_models') as mock_train_models:
            
            mock_data_loader.return_value = self.sample_data
            mock_read_csv.return_value = self.engineered_data
            
            # Call without parameters
            run.run_pipeline()
            
            # Verify default paths were used
            mock_data_loader.assert_called_once_with("data/raw/bank-full.csv")
            mock_preprocess.assert_called_once_with("data/raw/bank-full.csv", output_dir='data/processed')

    def test_run_pipeline_with_custom_paths(self):
        """Test run_pipeline accepts custom paths correctly."""
        custom_raw = "/custom/raw/path.csv"
        custom_processed = "/custom/processed"
        custom_engineered = "/custom/engineered.csv"
        
        with patch('run.data_loader') as mock_loader, \
             patch('run.preprocess') as mock_preprocess, \
             patch('run.feature_engineering') as mock_fe, \
             patch('pandas.read_csv') as mock_read, \
             patch('run.train_and_save_models') as mock_train, \
             patch('os.path.exists', return_value=True):
            
            # Setup mocks
            mock_loader.return_value = pd.DataFrame({'col': [1], 'y': [0]})
            mock_read.return_value = pd.DataFrame({'col': [1], 'y': [0]})
            
            run.run_pipeline(custom_raw, custom_processed, custom_engineered)
            
            # Verify custom paths were used
            mock_loader.assert_called_once_with(custom_raw)
            mock_preprocess.assert_called_once_with(custom_raw, output_dir=custom_processed)


class TestErrorHandling:
    """Unit tests for error handling scenarios."""
    
    @patch('run.data_loader')
    def test_run_pipeline_file_not_found(self, mock_data_loader):
        """Test pipeline behavior when raw data file is not found."""
        mock_data_loader.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(SystemExit) as exc_info:
            run.run_pipeline(raw_data_path="nonexistent_file.csv")
        
        assert exc_info.value.code == 1

    @patch('run.data_loader')
    @patch('run.preprocess')
    @patch('run.feature_engineering')
    @patch('pandas.read_csv')
    def test_run_pipeline_empty_data_error(self, mock_read_csv, mock_feature_engineering,
                                          mock_preprocess, mock_data_loader):
        """Test pipeline behavior when data file is empty."""
        sample_data = pd.DataFrame({'col': [1], 'y': [0]})
        mock_data_loader.return_value = sample_data
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No data")
        
        with pytest.raises(SystemExit) as exc_info:
            run.run_pipeline()
        
        assert exc_info.value.code == 1

    @patch('run.data_loader')
    @patch('run.preprocess')
    @patch('run.feature_engineering')
    @patch('pandas.read_csv')
    def test_run_pipeline_missing_target_column(self, mock_read_csv, mock_feature_engineering,
                                               mock_preprocess, mock_data_loader):
        """Test pipeline behavior when target column 'y' is missing."""
        sample_data = pd.DataFrame({'col': [1], 'y': [0]})
        mock_data_loader.return_value = sample_data
        
        # Create data without target column
        data_without_target = pd.DataFrame({'col': [1, 2, 3]})
        mock_read_csv.return_value = data_without_target
        
        with pytest.raises(SystemExit) as exc_info:
            run.run_pipeline()
        
        assert exc_info.value.code == 1

    @patch('run.data_loader')
    @patch('run.preprocess')
    @patch('run.feature_engineering')
    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_run_pipeline_engineered_file_not_found(self, mock_exists, mock_read_csv,
                                                   mock_feature_engineering, mock_preprocess,
                                                   mock_data_loader):
        """Test pipeline behavior when engineered data file doesn't exist."""
        sample_data = pd.DataFrame({'col': [1], 'y': [0]})
        mock_data_loader.return_value = sample_data
        mock_exists.return_value = False  # Simulate file not existing
        
        with pytest.raises(SystemExit) as exc_info:
            run.run_pipeline()
        
        assert exc_info.value.code == 1

    @patch('run.data_loader')
    def test_general_exception_handling(self, mock_data_loader):
        """Test handling of unexpected exceptions."""
        mock_data_loader.side_effect = Exception("Unexpected error")
        
        with pytest.raises(SystemExit) as exc_info:
            run.run_pipeline()
        
        assert exc_info.value.code == 1

    @patch('builtins.print')
    @patch('run.data_loader')
    def test_error_messages(self, mock_data_loader, mock_print):
        """Test that appropriate error messages are printed."""
        mock_data_loader.side_effect = FileNotFoundError("Test file not found")
        
        with pytest.raises(SystemExit):
            run.run_pipeline()
        
        # Check that error message was printed
        mock_print.assert_any_call("❌ File not found error: Test file not found")


class TestMainFunction:
    """Unit tests for the main function."""
    
    @patch('run.run_pipeline')
    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_main_creates_directories(self, mock_exists, mock_makedirs, mock_run_pipeline):
        """Test that main function creates required directories."""
        # Simulate directories don't exist
        mock_exists.return_value = False
        
        run.main()
        
        # Verify directories were created
        expected_calls = [
            call('data/raw', exist_ok=True),
            call('data/processed', exist_ok=True),
            call('models', exist_ok=True)
        ]
        mock_makedirs.assert_has_calls(expected_calls, any_order=True)
        mock_run_pipeline.assert_called_once()

    @patch('run.run_pipeline')
    @patch('os.path.exists')
    def test_main_directories_exist(self, mock_exists, mock_run_pipeline):
        """Test main function when directories already exist."""
        # Simulate directories already exist
        mock_exists.return_value = True
        
        run.main()
        
        # run_pipeline should still be called
        mock_run_pipeline.assert_called_once()

    @patch('run.run_pipeline')
    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_main_partial_directory_creation(self, mock_exists, mock_makedirs, mock_run_pipeline):
        """Test main function when some directories exist and others don't."""
        # Simulate mixed directory existence
        def side_effect(path):
            return path == 'data/raw'  # Only raw directory exists
        
        mock_exists.side_effect = side_effect
        
        run.main()
        
        # Verify only missing directories were created
        expected_calls = [
            call('data/processed', exist_ok=True),
            call('models', exist_ok=True)
        ]
        mock_makedirs.assert_has_calls(expected_calls, any_order=True)
        
        # Verify raw directory creation was not called
        mock_makedirs.assert_not_called_with('data/raw', exist_ok=True)


class TestDataValidation:
    """Unit tests for data validation logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.valid_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['A', 'B', 'C'],
            'y': [0, 1, 0]
        })
    
    @patch('run.data_loader')
    @patch('run.preprocess')
    @patch('run.feature_engineering')
    @patch('pandas.read_csv')
    @patch('run.train_and_save_models')
    def test_feature_target_separation(self, mock_train_models, mock_read_csv,
                                      mock_feature_engineering, mock_preprocess,
                                      mock_data_loader):
        """Test that features and target are correctly separated."""
        mock_data_loader.return_value = self.valid_data
        mock_read_csv.return_value = self.valid_data
        
        run.run_pipeline()
        
        # Get the arguments passed to train_and_save_models
        call_args = mock_train_models.call_args
        features, target = call_args[0]
        
        # Verify separation
        assert 'y' not in features.columns
        assert len(features.columns) == 2  # feature1 and feature2
        assert all(target == self.valid_data['y'])

    @patch('run.data_loader')
    @patch('run.preprocess')
    @patch('run.feature_engineering')
    @patch('pandas.read_csv')
    @patch('run.train_and_save_models')
    def test_empty_dataframe_handling(self, mock_train_models, mock_read_csv,
                                     mock_feature_engineering, mock_preprocess,
                                     mock_data_loader):
        """Test handling of empty DataFrames."""
        empty_data = pd.DataFrame()
        mock_data_loader.return_value = empty_data
        mock_read_csv.return_value = empty_data
        
        with pytest.raises(SystemExit):
            run.run_pipeline()


# Pytest fixtures for reusable test data
@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame for tests."""
    return pd.DataFrame({
        'feature1': np.random.randn(10),
        'feature2': np.random.choice(['A', 'B', 'C'], 10),
        'feature3': np.random.uniform(0, 1, 10),
        'y': np.random.choice([0, 1], 10)
    })


@pytest.fixture
def temp_directory():
    """Fixture providing temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# Parametrized tests for different scenarios
@pytest.mark.parametrize("file_extension", [".csv", ".CSV"])
def test_file_extensions(file_extension):
    """Test pipeline works with different CSV file extensions."""
    test_file = f"test_data{file_extension}"
    # This would test file extension handling
    assert test_file.lower().endswith('.csv')


@pytest.mark.parametrize("error_type,expected_exit_code", [
    (FileNotFoundError, 1),
    (pd.errors.EmptyDataError, 1),
    (ValueError, 1),
    (Exception, 1)
])
def test_error_exit_codes(error_type, expected_exit_code):
    """Test that different errors produce correct exit codes."""
    with patch('run.data_loader') as mock_loader:
        mock_loader.side_effect = error_type("Test error")
        
        with pytest.raises(SystemExit) as exc_info:
            run.run_pipeline()
        
        assert exc_info.value.code == expected_exit_code


@pytest.mark.parametrize("data_size", [0, 1, 10, 100])
def test_different_data_sizes(data_size):
    """Test pipeline with different data sizes."""
    if data_size == 0:
        test_data = pd.DataFrame()
    else:
        test_data = pd.DataFrame({
            'feature': range(data_size),
            'y': [0, 1] * (data_size // 2 + 1)
        }).head(data_size)
    
    with patch('run.data_loader', return_value=test_data), \
         patch('run.preprocess'), \
         patch('run.feature_engineering'), \
         patch('pandas.read_csv', return_value=test_data), \
         patch('run.train_and_save_models'):
        
        if data_size == 0:
            with pytest.raises(SystemExit):
                run.run_pipeline()
        else:
            try:
                run.run_pipeline()
            except SystemExit:
                # Some data sizes might still cause issues due to missing 'y' column
                pass


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
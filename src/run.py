#!/usr/bin/env python3
"""
Machine Learning Pipeline Runner Script.

This script orchestrates a complete machine learning pipeline including:
- Data loading from raw CSV files
- Data preprocessing and cleaning
- Feature engineering
- Model training and saving

Usage:
    python run.py
"""

import os
import sys
import pandas as pd

# Import pipeline functions (adjusted for src/ folder structure)
from data_loader import load_data as data_loader
from preprocess import preprocess_pipeline as preprocess
from feature_engineering import run_feature_engineering as feature_engineering
from train_model import train_and_save_models


def run_pipeline(raw_data_path=None, processed_data_path=None, engineered_data_path=None):
    """
    Execute the complete machine learning pipeline.

    Args:
        raw_data_path (str, optional): Path to raw data file.
        processed_data_path (str, optional): Directory for processed data.
        engineered_data_path (str, optional): Path to engineered data file.

    Returns:
        None
    """
    if raw_data_path is None:
        raw_data_path = "data/raw/bank-full.csv"
    if processed_data_path is None:
        processed_data_path = "data/processed"
    if engineered_data_path is None:
        engineered_data_path = "data/processed/engineered_data.csv"

    print("🔁 Starting machine learning pipeline...\n")

    try:
        # Step 1: Load raw data
        print("📥 Loading raw data...")
        raw_data = data_loader(raw_data_path)
        print(f"   ✓ Loaded {len(raw_data)} records from {raw_data_path}")

        # Step 2: Data preprocessing
        print("⚙️ Preprocessing data...")
        preprocess(raw_data_path, output_dir=processed_data_path)
        print(f"   ✓ Preprocessed data saved to {processed_data_path}")

        # Step 3: Feature engineering
        print("🛠️ Running feature engineering...")
        preprocessed_file = os.path.join(processed_data_path, 'preprocessed_data.csv')
        feature_engineering(preprocessed_file, processed_data_path)
        print(f"   ✓ Engineered features saved to {processed_data_path}")

        # Step 4: Load engineered data for model training
        print("📂 Loading engineered data for model training...")
        if not os.path.exists(engineered_data_path):
            raise FileNotFoundError(f"Engineered data file not found: {engineered_data_path}")

        engineered_data = pd.read_csv(engineered_data_path)
        print(
            f"   ✓ Loaded {len(engineered_data)} records "
            f"with {len(engineered_data.columns)} features"
        )

        if 'y' not in engineered_data.columns:
            raise ValueError("Target column 'y' not found in engineered data")

        features = engineered_data.drop(columns=['y'])
        target = engineered_data['y']

        # Step 5: Train and save models
        print("🤖 Training and saving models...")
        model_directory = 'models'
        os.makedirs(model_directory, exist_ok=True)

        train_and_save_models(features, target, model_dir=model_directory)
        print(f"   ✓ Models saved to {model_directory}")

        print("\n✅ Pipeline completed successfully!")

    except FileNotFoundError as file_error:
        print(f"❌ File not found error: {file_error}")
        sys.exit(1)
    except pd.errors.EmptyDataError as data_error:
        print(f"❌ Data error: {data_error}")
        sys.exit(1)
    except ValueError as value_error:
        print(f"❌ Value error: {value_error}")
        sys.exit(1)
    except Exception as general_error:  # Catch-all for unexpected issues
        print(f"❌ Unexpected error occurred: {general_error}")
        sys.exit(1)


def main():
    """
    Main function to execute the pipeline.
    """
    required_dirs = ['data/raw', 'data/processed', 'models']
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"⚠️  Creating missing directory: {directory}")
            os.makedirs(directory, exist_ok=True)

    run_pipeline()


if __name__ == "__main__":
    main()

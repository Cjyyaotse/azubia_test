import os
import sys
import pandas as pd
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import train_model

def test_load_data_columns():
    features, target = train_model.load_data('data/processed/engineered_data.csv')
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
    assert target.name == 'y'

def test_get_models_keys():
    models = train_model.get_models()
    assert 'logistic_regression' in models
    assert 'random_forest' in models
    assert 'xgboost' in models

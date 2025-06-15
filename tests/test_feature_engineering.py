import os
import sys
import pandas as pd
import pytest

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import feature_engineering as fe

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'age': [30, 40],
        'job': ['admin.', 'technician'],
        'marital': ['married', 'single'],
        'education': ['tertiary', 'secondary'],
        'default': ['no', 'no'],
        'housing': ['yes', 'no'],
        'loan': ['no', 'yes'],
        'contact': ['cellular', 'telephone'],
        'month': ['may', 'jun'],
        'day_of_week': ['mon', 'tue'],
        'duration': [100, 200],
        'campaign': [1, 2],
        'pdays': [999, 10],
        'previous': [0, 1],
        'poutcome': ['nonexistent', 'success']
    })

def test_engineer_features_output_shape(sample_df):
    df_feat = fe.engineer_features(sample_df)
    assert isinstance(df_feat, pd.DataFrame)
    assert df_feat.shape[0] == 2

def test_encode_features_output(sample_df):
    df_feat = fe.engineer_features(sample_df)
    df_encoded = fe.encode_features(df_feat)
    assert isinstance(df_encoded, pd.DataFrame)
    assert df_encoded.isnull().sum().sum() == 0

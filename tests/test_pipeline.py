import pytest
import pandas as pd
import numpy as np
import os

from src import data_loader, preprocess, feature_engineering, train_model, evaluate

@pytest.fixture(scope="module")
def raw_data():
    path = "data/raw/bank-additional.csv"
    df = data_loader.load_csv(path)
    return df

def test_data_loader(raw_data):
    assert isinstance(raw_data, pd.DataFrame)
    assert not raw_data.empty
    assert "y" in raw_data.columns

def test_preprocess(raw_data):
    df_clean = preprocess.clean_data(raw_data.copy())
    assert isinstance(df_clean, pd.DataFrame)
    assert df_clean.isnull().sum().sum() == 0
    assert "y" in df_clean.columns

def test_feature_engineering(raw_data):
    df_clean = preprocess.clean_data(raw_data.copy())
    df_feat = feature_engineering.engineer_features(df_clean.copy())
    df_encoded = feature_engineering.encode_features(df_feat)
    assert isinstance(df_encoded, pd.DataFrame)
    assert "zero_duration" in df_encoded.columns
    assert "was_previously_contacted" in df_encoded.columns

def test_class_balancing(raw_data):
    df_clean = preprocess.clean_data(raw_data.copy())
    df_feat = feature_engineering.engineer_features(df_clean.copy())
    df_encoded = feature_engineering.encode_features(df_feat.copy())
    df_balanced = feature_engineering.handle_class_imbalance(df_encoded, target="y")
    assert df_balanced["y"].value_counts().nunique() == 1  # balanced classes

def test_model_training(raw_data):
    df_clean = preprocess.clean_data(raw_data.copy())
    df_feat = feature_engineering.engineer_features(df_clean.copy())
    df_encoded = feature_engineering.encode_features(df_feat.copy())
    df_balanced = feature_engineering.handle_class_imbalance(df_encoded.copy(), target="y")

    X = df_balanced.drop("y", axis=1)
    y = df_balanced["y"]
    model = train_model.train_random_forest(X, y)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

def test_model_evaluation(raw_data):
    df_clean = preprocess.clean_data(raw_data.copy())
    df_feat = feature_engineering.engineer_features(df_clean.copy())
    df_encoded = feature_engineering.encode_features(df_feat.copy())
    df_balanced = feature_engineering.handle_class_imbalance(df_encoded.copy(), target="y")

    X = df_balanced.drop("y", axis=1)
    y = df_balanced["y"]

    model = train_model.train_random_forest(X, y)
    metrics = evaluate.evaluate_model(model, X, y)

    assert isinstance(metrics, dict)
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert metric in metrics
        assert 0 <= metrics[metric] <= 1

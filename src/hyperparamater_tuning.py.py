"""
evaluate.py

Loads trained classification models and evaluates them using a test dataset.
Prints key metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

Author: Your Name
Date: 2025-06-14
"""

import os
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Configuration
DATA_PATH = 'data/raw/bank.csv'
MODEL_DIR = 'models'
TARGET_COL = 'y'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: str):
    """
    Load external test data from CSV.

    Args:
        path (str): Path to CSV.

    Returns:
        X_test (pd.DataFrame), y_test (pd.Series)
    """
    df = pd.read_csv(path, delimiter=';')
    X_test = df.drop(columns=[TARGET_COL])
    y_test = df[TARGET_COL]
    return X_test, y_test



def load_models(model_dir: str) -> dict:
    """
    Load all trained models from a directory.

    Args:
        model_dir (str): Directory path containing model files.

    Returns:
        dict: Dictionary of model names and loaded model objects.
    """
    models = {}
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl'):
            model_name = filename.replace('.pkl', '')
            models[model_name] = joblib.load(os.path.join(model_dir, filename))
    return models


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and return metrics.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True labels.

    Returns:
        dict: Metrics.
    """
    y_pred = model.predict(X_test)

    # Use predict_proba or decision_function for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_scores = model.decision_function(X_test)
    else:
        y_scores = y_pred  # fallback

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_scores)
    }


def main():
    X_test, y_test = load_data(DATA_PATH)
    models = load_models(MODEL_DIR)

    print("Model Performance on Test Set\n" + "-" * 40)
    for name, model in models.items():
        print(f"\n{name.upper()}")
        metrics = evaluate_model(model, X_test, y_test)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()

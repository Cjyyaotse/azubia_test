"""
train_models.py

This script trains multiple classification models on the feature-engineered bank marketing dataset.
It saves each trained model to disk for future evaluation and inference.

Models Trained:
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine
"""

import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Configuration
DATA_PATH = 'data/processed/engineered_data.csv'
MODEL_DIR = 'models'
TARGET_COL = 'y'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the dataset and split into features and target.

    Args:
        path (str): Path to the engineered dataset.

    Returns:
        tuple: Features (X), Target (y)
    """
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def get_models() -> dict:
    """
    Define a dictionary of models to train.

    Returns:
        dict: A dictionary of model name and model instance.
    """
    return {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
        'svm': SVC(probability=True, random_state=RANDOM_STATE)
    }


def train_and_save_models(X: pd.DataFrame, y: pd.Series, model_dir: str):
    """
    Train multiple models and save them to disk.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        model_dir (str): Directory to save trained models.
    """
    os.makedirs(model_dir, exist_ok=True)
    models = get_models()

    # Split once and use for all models
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        model_path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")


def main():
    X, y = load_data(DATA_PATH)
    train_and_save_models(X, y, MODEL_DIR)


if __name__ == "__main__":
    main()

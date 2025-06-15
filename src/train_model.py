"""
train_models.py

This script trains multiple classification models on the feature-engineered bank marketing dataset.
It saves each trained model to disk for future evaluation and inference.

Models Trained:
- Logistic Regression
- Random Forest
- XGBoost
"""

import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Configuration
DATA_PATH = 'data/processed/engineered_data.csv'
MODEL_DIR = 'models'
OUTPUT_DIR = 'data/output'
TARGET_COL = 'y'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and split into features and target."""
    df = pd.read_csv(path)
    features = df.drop(columns=[TARGET_COL])
    target = df[TARGET_COL]
    return features, target


def get_models() -> dict:
    """Define a dictionary of models to train."""
    return {
        'logistic_regression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    }


def train_and_save_models(features: pd.DataFrame, target: pd.Series, model_dir: str):
    """Train models, evaluate them, save model and metrics to disk."""
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = get_models()

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(features_train, target_train)

        # Save model
        model_path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Saved {name} to {model_path}")

        # Evaluate
        target_pred = model.predict(features_test)
        if hasattr(model, "predict_proba"):
            target_scores = model.predict_proba(features_test)[:, 1]
        elif hasattr(model, "decision_function"):
            target_scores = model.decision_function(features_test)
        else:
            target_scores = target_pred

        metrics = {
            "accuracy": round(accuracy_score(target_test, target_pred), 4),
            "precision": round(precision_score(target_test, target_pred), 4),
            "recall": round(recall_score(target_test, target_pred), 4),
            "f1_score": round(f1_score(target_test, target_pred), 4),
            "roc_auc": round(roc_auc_score(target_test, target_scores), 4)
        }

        print(f"📊 Performance on Test Set ({name}):")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value}")

        # Save metrics to JSON
        metrics_path = os.path.join(OUTPUT_DIR, f"{name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"📁 Saved metrics to {metrics_path}")


def main():
    """Main function to load data and train models."""
    features, target = load_data(DATA_PATH)
    train_and_save_models(features, target, MODEL_DIR)


if __name__ == "__main__":
    main()

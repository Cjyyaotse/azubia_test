"""
Test Model Script

Loads a test dataset, applies feature engineering and encoding,
uses the trained model to predict outcomes, and evaluates performance.
Saves predictions and performance plots.
"""

import os
import sys
import json
import importlib.util

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)


def import_module_from_src(filename, name=None):
    """
    Dynamically import a module from the src directory by filename.
    """
    full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
    spec = importlib.util.spec_from_file_location(name or filename, full_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name or filename] = module
    spec.loader.exec_module(module)
    return module


# Load Modules
feature_mod = import_module_from_src('feature_engineering.py', 'feature_engineering')
engineer_features = feature_mod.engineer_features
encode_features = feature_mod.encode_features

# Constants
MODEL_PATH = os.path.join("models", "random_forest.pkl")
DATA_PATH = os.path.join("data", "raw", "bank-additional.csv")
METRICS_PATH = "data/output/test_metrics.json"
PREDICTIONS_PATH = "data/output/test_predictions.csv"
ROC_PATH = "reports/figures/test_model_roc_curve.png"
CM_PATH = "reports/figures/test_model_confusion_matrix.png"

# Load Model
model = joblib.load(MODEL_PATH)

# Load Dataset
df = pd.read_csv(DATA_PATH, delimiter=';')

# Separate Features and Target
X = df.drop(columns=['y'])
y_true = df['y'].map({'yes': 1, 'no': 0})

# Feature Engineering
X_eng = engineer_features(X)
X_enc = encode_features(X_eng)

# Align columns to match those used during model training
if hasattr(model, "feature_names_in_"):
    expected_cols = list(model.feature_names_in_)
    X_enc = X_enc.reindex(columns=expected_cols, fill_value=0)
else:
    print("Warning: Model does not have 'feature_names_in_' attribute. Ensure columns match manually.")

# Predict
y_pred = model.predict(X_enc)
y_prob = model.predict_proba(X_enc)[:, 1]

# Append Predictions
results_df = df.copy()
results_df['prediction'] = y_pred
results_df['probability'] = y_prob

# Evaluate Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)

print("📊 Model Performance on Test Set:")
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔁 Recall:    {recall:.4f}")
print(f"📈 F1 Score:  {f1:.4f}")
print(f"🧪 ROC AUC:   {roc_auc:.4f}")

metrics = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "roc_auc": round(roc_auc, 4)
}

# Save Metrics
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)
print(f"📁 Test performance metrics saved to {METRICS_PATH}")

# Save Predictions
os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
results_df.to_csv(PREDICTIONS_PATH, index=False)
print(f"✅ Predictions saved to {PREDICTIONS_PATH}")

# Save ROC Curve
os.makedirs(os.path.dirname(ROC_PATH), exist_ok=True)
fpr, tpr, _ = roc_curve(y_true, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig(ROC_PATH)
plt.close()
print(f"🧭 ROC curve saved to {ROC_PATH}")

# Save Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(CM_PATH)
plt.close()
print(f"🧩 Confusion matrix saved to {CM_PATH}")

# test_model.py

"""
Test Model Script

Loads a test dataset, applies feature engineering and encoding,
uses the trained model to predict outcomes, and evaluates performance.
Saves predictions and performance plots.

"""

import os
import sys
import pandas as pd
import joblib
import importlib.util
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Dynamic Import Utility
def import_module_from_src(filename, name=None):
    full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
    spec = importlib.util.spec_from_file_location(name or filename, full_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name or filename] = module
    spec.loader.exec_module(module)
    return module

# Load Modules
# ======================
feature_mod = import_module_from_src('feature_engineering.py', 'feature_engineering')
engineer_features = feature_mod.engineer_features
encode_features = feature_mod.encode_features

# ======================
# Load Model
# ======================
model_path = os.path.join("models", "random_forest.pkl")
model = joblib.load(model_path)

# ======================
# Load Dataset
# ======================
data_path = os.path.join("data", "raw", "bank-additional.csv")
df = pd.read_csv(data_path, delimiter=';')

# ======================
# Separate Features and Target
# ======================
X = df.drop(columns=['y'])
y_true = df['y'].map({'yes': 1, 'no': 0})

# ======================
X_eng = engineer_features(X)
X_enc = encode_features(X_eng)

# Align columns to match those used during model training
if hasattr(model, "feature_names_in_"):
    expected_cols = list(model.feature_names_in_)
    # Drop unseen columns and reorder to match training
    X_enc = X_enc.reindex(columns=expected_cols, fill_value=0)
else:
    print("Warning: Model does not have 'feature_names_in_' attribute. Ensure columns match manually.")

y_pred = model.predict(X_enc)
y_prob = model.predict_proba(X_enc)[:, 1]

# ======================
# Append Predictions
# ======================
results_df = df.copy()
results_df['prediction'] = y_pred
results_df['probability'] = y_prob

# ======================
# Performance Metrics
# ======================
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

metrics_path = "data/output/test_metrics.json"
os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"📁 Test performance metrics saved to {metrics_path}")

# ======================
# Save Predictions
# ======================

os.makedirs("data/output", exist_ok=True)
results_df.to_csv("data/output/test_predictions.csv", index=False)
print("✅ Predictions saved to output/test_predictions.csv")

# ======================
# Save Plots
# ======================
os.makedirs("reports/figures", exist_ok=True)

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
roc_path = "reports/figures/test_model_roc_curve.png"
plt.savefig(roc_path)
plt.close()
print(f"🧭 ROC curve saved to {roc_path}")

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
cm_path = "reports/figures/test_model_confusion_matrix.png"
plt.savefig(cm_path)
plt.close()
print(f"🧩 Confusion matrix saved to {cm_path}")

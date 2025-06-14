"""
bank_feature_engineering.py

This module performs feature engineering on the preprocessed UCI Bank Marketing dataset.
It includes:
- Deriving new features
- Encoding categorical variables
- Handling class imbalance in the target variable 'y'

Output:
- A processed dataset ready for modeling

"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib


def load_preprocessed_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['zero_duration'] = (df['duration'] == 0).astype(int)
    df['was_previously_contacted'] = (df['pdays'] != -1).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])
    df['balance_bin'] = pd.cut(df['balance'], bins=[-1e6, 0, 1000, 5000, 1e6], labels=['negative', 'low', 'medium', 'high'])
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_num'] = df['month'].map(month_map)
    df['has_loan_and_housing'] = ((df['loan'] == 'yes') & (df['housing'] == 'yes')).astype(int)
    df['total_contacts'] = df['previous'] + df['campaign']
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    df['num_unknowns'] = df[cat_cols].apply(lambda x: (x == 'unknown').sum(), axis=1)
    df['contact_effectiveness'] = df.apply(lambda row: row['previous'] / (row['pdays'] + 1)
                                           if row['pdays'] > 0 else 0, axis=1)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col != 'y':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    os.makedirs('models', exist_ok=True)
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    return df


def handle_class_imbalance(df: pd.DataFrame, target: str = 'y') -> pd.DataFrame:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    if df[target].dtype == 'object':
        df[target] = LabelEncoder().fit_transform(df[target])

    X = df.drop(columns=[target])
    y = df[target]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    df_bal = pd.DataFrame(X_res, columns=X.columns)
    df_bal[target] = y_res
    return df_bal


def save_engineered_data(df: pd.DataFrame, output_dir: str, filename: str = 'engineered_data.csv'):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, filename), index=False)


def run_feature_engineering(preprocessed_path: str, output_dir: str) -> pd.DataFrame:
    df = load_preprocessed_data(preprocessed_path)
    df = engineer_features(df)
    df = encode_features(df)
    df = handle_class_imbalance(df, target='y')
    save_engineered_data(df, output_dir)

    print("\nFinal columns used for training:")
    print(df.drop(columns=['y']).columns.tolist())

    return df


if __name__ == "__main__":
    try:
        PREPROCESSED_PATH = 'data/processed/preprocessed_data.csv'
        OUTPUT_DIR = 'data/processed'
        df_final = run_feature_engineering(PREPROCESSED_PATH, OUTPUT_DIR)
        print("Successfully performed feature engineering")
    except Exception as e:
        print(f"Error during feature engineering: {e}")

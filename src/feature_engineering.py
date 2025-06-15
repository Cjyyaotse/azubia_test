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

import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


def load_preprocessed_data(path: str) -> pd.DataFrame:
    """Load preprocessed CSV data into a DataFrame."""
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate new features from existing columns."""
    df['zero_duration'] = (df['duration'] == 0).astype(int)
    df['was_previously_contacted'] = (df['pdays'] != -1).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 100],
                              labels=['<30', '30-40', '40-50', '50+'])

    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month_num'] = df['month'].map(month_map)

    df['has_loan_and_housing'] = ((df['loan'] == 'yes') & (df['housing'] == 'yes')).astype(int)
    df['total_contacts'] = df['previous'] + df['campaign']

    cat_cols = ['job', 'marital', 'education', 'default', 'housing',
                 'loan', 'contact', 'month', 'poutcome']
    df['num_unknowns'] = df[cat_cols].apply(lambda x: (x == 'unknown').sum(), axis=1)

    df['contact_effectiveness'] = df.apply(
        lambda row: row['previous'] / (row['pdays'] + 1) if row['pdays'] > 0 else 0,
        axis=1
    )

    df = df.drop(columns='balance', errors='ignore')
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode categorical features and save encoders."""
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
    """Apply SMOTE to balance the dataset based on the target variable."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    if df[target].dtype == 'object':
        df[target] = LabelEncoder().fit_transform(df[target])

    features = df.drop(columns=[target])
    labels = df[target]

    smote = SMOTE(random_state=42)
    features_resampled, labels_resampled = smote.fit_resample(features, labels)

    df_balanced = pd.DataFrame(features_resampled, columns=features.columns)
    df_balanced[target] = labels_resampled
    return df_balanced


def save_engineered_data(df: pd.DataFrame, output_dir: str, filename: str = 'engineered_data.csv'):
    """Save the final engineered DataFrame to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, filename), index=False)


def run_feature_engineering(preprocessed_path: str, output_dir: str) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = load_preprocessed_data(preprocessed_path)
    df = engineer_features(df)
    df = encode_features(df)
    df = handle_class_imbalance(df, target='y')
    save_engineered_data(df, output_dir)

    return df


if __name__ == "__main__":
    try:
        PREPROCESSED_PATH = 'data/processed/preprocessed_data.csv'
        OUTPUT_DIR = 'data/processed'
        df_final = run_feature_engineering(PREPROCESSED_PATH, OUTPUT_DIR)
        print("\nFinal columns used for training:")
        print(df_final.drop(columns=['y']).columns.tolist())
        print("Successfully performed feature engineering")
    except Exception as error:
        print(f"Error during feature engineering: {error}")

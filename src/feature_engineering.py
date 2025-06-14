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


def load_preprocessed_data(path: str) -> pd.DataFrame:
    """
    Load preprocessed data from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features to improve model performance.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Binary feature for zero duration calls
    df['zero_duration'] = (df['duration'] == 0).astype(int)

    # Indicator if client was previously contacted
    df['was_previously_contacted'] = (df['pdays'] != -1).astype(int)

    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])

    # Balance bins
    df['balance_bin'] = pd.cut(df['balance'], bins=[-10000, 0, 1000, 5000, 1e6],
                               labels=['negative', 'low', 'medium', 'high'])

    # Convert month to numeric value
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_num'] = df['month'].map(month_map)

    # Interaction: has both loan and housing loan
    df['has_loan_and_housing'] = ((df['loan'] == 'yes') & (df['housing'] == 'yes')).astype(int)

    # Total contacts (previous + current)
    df['total_contacts'] = df['previous'] + df['campaign']

    # Count of 'unknown' values in key categorical columns
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    df['num_unknowns'] = df[cat_cols].apply(lambda x: sum(x == 'unknown'), axis=1)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables.

    Args:
        df (pd.DataFrame): DataFrame with categorical features.

    Returns:
        pd.DataFrame: DataFrame with encoded features.
    """
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col != 'y':  # Don't encode target here
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    return df


def handle_class_imbalance(df: pd.DataFrame, target: str = 'y') -> pd.DataFrame:
    """
    Balance the dataset using SMOTE.

    Args:
        df (pd.DataFrame): DataFrame with imbalanced target.
        target (str): Target column name.

    Returns:
        pd.DataFrame: Balanced DataFrame.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    # Encode target if it's not numeric
    if df[target].dtype == 'object':
        df[target] = LabelEncoder().fit_transform(df[target])

    X = df.drop(columns=[target])
    y = df[target]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced[target] = y_resampled

    return df_balanced


def save_engineered_data(df: pd.DataFrame, output_dir: str, filename: str = 'engineered_data.csv'):
    """
    Save the final dataset after feature engineering.

    Args:
        df (pd.DataFrame): Final DataFrame to save.
        output_dir (str): Directory to save the file.
        filename (str): Output filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, filename), index=False)


def run_feature_engineering(preprocessed_path: str, output_dir: str) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Args:
        preprocessed_path (str): Path to the preprocessed dataset.
        output_dir (str): Output directory.

    Returns:
        pd.DataFrame: Final processed DataFrame.
    """
    df = load_preprocessed_data(preprocessed_path)
    df = engineer_features(df)
    df = encode_features(df)
    df = handle_class_imbalance(df, target='y')
    save_engineered_data(df, output_dir)
    return df


# Run when executed directly
if __name__ == "__main__":
    try:
        PREPROCESSED_PATH = 'data/processed/preprocessed_data.csv'
        OUTPUT_DIR = 'data/processed'

        df_final = run_feature_engineering(PREPROCESSED_PATH, OUTPUT_DIR)
        print("Successfully performed feature engineering")
    except Exception as error:
        print(f"Error performing feature enginerring, error:{error}")

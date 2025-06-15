"""
bank_preprocessing.py

This module handles data preprocessing for the UCI Bank Marketing dataset.
It includes steps such as:
- Initial data loading and inspection
- Memory usage optimization
- Outlier detection
- Validation checks
- Summary reporting
The cleaned DataFrame is saved to data/preprocessed for downstream analysis.
"""

import os
import pandas as pd
import numpy as np


def load_data(path: str, delimiter: str = ';') -> pd.DataFrame:
    """
    Load the dataset from the specified CSV file path.

    Args:
        path (str): Path to the CSV file.
        delimiter (str): Delimiter used in the file.

    Returns:
        pd.DataFrame: Loaded raw data.
    """
    return pd.read_csv(path, delimiter=delimiter)


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage by downcasting numerical and categorical columns.

    Args:
        df (pd.DataFrame): DataFrame to be optimized.

    Returns:
        pd.DataFrame: Optimized DataFrame.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in numerical_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if pd.api.types.is_integer_dtype(df[col]):
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype(np.int32)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')

    for col in categorical_cols:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')

    return df


def detect_outliers(df: pd.DataFrame) -> dict:
    """
    Detect outliers in numerical columns using IQR method.

    Args:
        df (pd.DataFrame): DataFrame to analyze.

    Returns:
        dict: Summary of outliers per column.
    """
    outliers = {}
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        if count > 0:
            outliers[col] = {
                'count': count,
                'percentage': (count / len(df)) * 100,
                'min': df[col].min(),
                'max': df[col].max(),
                'lower_bound': lower,
                'upper_bound': upper
            }

    return outliers


def validate_data(df: pd.DataFrame) -> list:
    """
    Perform basic validation checks on the dataset.

    Args:
        df (pd.DataFrame): DataFrame to validate.

    Returns:
        list: List of detected issues.
    """
    issues = []

    for col in ['age', 'duration', 'campaign', 'pdays']:
        if col in df.columns and (df[col] < 0).sum() > 0:
            issues.append(f"{col} contains negative values")

    if 'duration' in df.columns and (df['duration'] == 0).sum() > 0:
        issues.append("duration contains zero values")

    return issues


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary of the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to summarize.

    Returns:
        pd.DataFrame: Summary DataFrame.
    """
    summary = pd.DataFrame(index=df.columns)
    summary['Data_Type'] = df.dtypes
    summary['Non_Null_Count'] = df.count()
    summary['Unique_Values'] = df.nunique()
    summary['Memory_MB'] = df.memory_usage(deep=True, index=False) / 1024**2
    summary.reset_index(inplace=True)
    summary.rename(columns={'index': 'Column'}, inplace=True)
    return summary


def save_data(df: pd.DataFrame, output_dir: str, filename: str = 'preprocessed_data.csv'):
    """
    Save the DataFrame to a specified directory.

    Args:
        df (pd.DataFrame): DataFrame to save.
        output_dir (str): Directory where the file will be saved.
        filename (str): Name of the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, filename), index=False)


def preprocess_pipeline(csv_path: str, output_dir: str) -> pd.DataFrame:
    """
    Main preprocessing pipeline.

    Args:
        csv_path (str): Path to raw CSV file.
        output_dir (str): Directory where processed data will be saved.

    Returns:
        pd.DataFrame: Final preprocessed DataFrame.
    """
    df = load_data(csv_path)
    df.drop_duplicates(inplace=True)
    df = optimize_memory(df)
    outlier_info = detect_outliers(df)
    print(f"Outlier info: {outlier_info}")
    validation_issues = validate_data(df)
    print(f"Validation_issues: {validation_issues}")

    # Save preprocessed DataFrame
    save_data(df, output_dir)

    return df  # optionally return summary, outlier_info, validation_issues too


# Run pipeline when executed directly
if __name__ == "__main__":
    RAW_DATA_PATH = 'data/raw/bank-full.csv'
    OUTPUT_DIRECTORY = 'data/processed'

    df_final = preprocess_pipeline(RAW_DATA_PATH, OUTPUT_DIRECTORY)

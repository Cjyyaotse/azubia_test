"""Data loader module for loading CSV files into pandas DataFrames."""

import pandas as pd


def load_data(filepath):
    """
    Load preprocessed data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(filepath, delimiter=';')
    return df


if __name__ == "__main__":
    try:
        df = load_data("data/raw/bank.csv")
        print("DataFrame loaded successfully")
        print(f"Dataframe statistics:{df.describe()}")
    except (FileNotFoundError, pd.errors.ParserError) as error:
        print(f"Error loading DataFrame, error: {error}")
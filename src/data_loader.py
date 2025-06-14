"""Data loader module for loading CSV files into pandas DataFrames."""

import pandas as pd


def load_data(filepath):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(filepath, delimiter=';')


if __name__ == "__main__":
    try:
        df = load_data("data/raw/bank.csv")
        print("DataFrame loaded successfully")
        print(f"Dataframe statistics:{df.describe()}")
    except (FileNotFoundError, pd.errors.ParserError) as error:
        print(f"Error loading DataFrame, error: {error}")
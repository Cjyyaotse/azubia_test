import os
import sys
import pandas as pd
import pytest

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import data_loader  


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file for testing."""
    data = "age;job;marital\n30;admin.;single\n40;technician;married"
    file_path = tmp_path / "test.csv"
    file_path.write_text(data)
    return str(file_path)


def test_load_data_returns_dataframe(sample_csv):
    """Test that load_data returns a DataFrame."""
    df = data_loader.load_data(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ['age', 'job', 'marital']


def test_load_data_file_not_found():
    """Test that loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data_loader.load_data("nonexistent.csv")


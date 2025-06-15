import os
import sys
import pandas as pd
import numpy as np
import pytest

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import preprocess as bp
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'age': [25, 45, 35, 60, 85],
        'duration': [100, 0, 200, 300, 100],
        'campaign': [1, 2, 1, 2, -1],
        'pdays': [999, -1, 5, 10, 20],
        'job': ['admin.', 'technician', 'blue-collar', 'retired', 'admin.']
    })


def test_optimize_memory(sample_df):
    optimized = bp.optimize_memory(sample_df.copy())
    assert pd.api.types.is_integer_dtype(optimized['age'])



def test_detect_outliers(sample_df):
    outliers = bp.detect_outliers(sample_df)
    assert isinstance(outliers, dict)
    assert 'age' in outliers or 'campaign' in outliers  # based on IQR, at least one likely


def test_validate_data(sample_df):
    issues = bp.validate_data(sample_df)
    assert isinstance(issues, list)
    assert any("negative" in issue or "zero" in issue for issue in issues)


def test_summarize(sample_df):
    summary = bp.summarize(sample_df)
    assert isinstance(summary, pd.DataFrame)
    assert 'Column' in summary.columns
    assert summary.shape[0] == sample_df.shape[1]


def test_save_and_load(tmp_path, sample_df):
    save_path = tmp_path / "saved.csv"
    bp.save_data(sample_df, tmp_path, "saved.csv")
    assert save_path.exists()
    loaded = pd.read_csv(save_path)
    assert loaded.shape == sample_df.shape


def test_load_data(tmp_path):
    file = tmp_path / "file.csv"
    file.write_text("a;b;c\n1;2;3")
    df = bp.load_data(str(file), delimiter=';')
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['a', 'b', 'c']


# 🔄 Integration test
def test_preprocess_pipeline_integration(tmp_path):
    content = """age;duration;campaign;pdays;job
25;100;1;999;admin.
45;0;2;-1;technician
35;200;1;5;blue-collar
60;300;2;10;retired
85;100;-1;20;admin."""
    csv_path = tmp_path / "bank.csv"
    csv_path.write_text(content)

    output_dir = tmp_path / "processed"
    df = bp.preprocess_pipeline(str(csv_path), str(output_dir))

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 5
    assert os.path.exists(os.path.join(output_dir, "preprocessed_data.csv"))

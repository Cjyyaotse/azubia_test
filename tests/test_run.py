import os
import sys
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import run

# Sample dataframe fixture
@pytest.fixture
def sample_engineered_df():
    return pd.DataFrame({
        'feature1': [0.1, 0.2],
        'feature2': [1, 0],
        'y': [1, 0]
    })


def test_missing_directories(tmp_path, monkeypatch):
    """Test if missing directories are created without error."""
    monkeypatch.setattr(run, "run_pipeline", lambda *args, **kwargs: None)

    # Remove any directory assumption
    for d in ['data/raw', 'data/processed', 'models']:
        full_path = tmp_path / d
        if full_path.exists():
            os.rmdir(full_path)

    monkeypatch.chdir(tmp_path)
    run.main()

    assert (tmp_path / 'data/raw').exists()
    assert (tmp_path / 'data/processed').exists()
    assert (tmp_path / 'models').exists()


@patch("src.run.data_loader")
@patch("src.run.preprocess")
@patch("src.run.feature_engineering")
@patch("src.run.train_and_save_models")
@patch("pandas.read_csv")
def test_run_pipeline_success(
    mock_read_csv,
    mock_train,
    mock_feat_eng,
    mock_preprocess,
    mock_loader,
    sample_engineered_df,
    tmp_path
):
    """Integration-style test for successful pipeline run."""
    raw_data_path = tmp_path / "bank.csv"
    raw_data_path.write_text("dummy;csv")

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    engineered_file = processed_dir / "engineered_data.csv"
    sample_engineered_df.to_csv(engineered_file, index=False)

    mock_loader.return_value = pd.DataFrame({'a': [1, 2]})
    mock_read_csv.return_value = sample_engineered_df

    run.run_pipeline(str(raw_data_path), str(processed_dir), str(engineered_file))

    mock_loader.assert_called_once()
    mock_preprocess.assert_called_once()
    mock_feat_eng.assert_called_once()
    mock_train.assert_called_once()


@patch("run.data_loader", side_effect=FileNotFoundError("Missing file"))
def test_run_pipeline_file_not_found(mock_loader):
    """Check if file not found exits with error code."""
    with pytest.raises(SystemExit) as excinfo:
        run.run_pipeline("nonexistent.csv", "data/processed", "data/processed/engineered_data.csv")
    assert excinfo.value.code == 1


@patch("run.data_loader", return_value=pd.DataFrame({'a': [1]}))
@patch("pandas.read_csv", side_effect=pd.errors.EmptyDataError("Empty file"))
def test_run_pipeline_empty_data_error(mock_read_csv, mock_loader):
    with pytest.raises(SystemExit) as excinfo:
        run.run_pipeline("some.csv", "data/processed", "data/processed/engineered_data.csv")
    assert excinfo.value.code == 1


@patch("run.data_loader", return_value=pd.DataFrame({'a': [1]}))
@patch("pandas.read_csv", return_value=pd.DataFrame({'x1': [0.1], 'x2': [0.2]}))  # Missing 'y'
def test_run_pipeline_value_error_due_to_missing_target(mock_read_csv, mock_loader):
    with pytest.raises(SystemExit) as excinfo:
        run.run_pipeline("some.csv", "data/processed", "data/processed/engineered_data.csv")
    assert excinfo.value.code == 1

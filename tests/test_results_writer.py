import numpy as np
import pandas as pd
import pytest

from core.results_writer import (
    create_export_dir,
    export_spike_times,
    export_data,
    export_hazard_excel,
)


# --- Test create_export_dir --- #
def test_create_export_dir(tmp_path):
    folder_path = tmp_path
    analysis_folder_name = "test_export"
    export_dir = create_export_dir(folder_path, analysis_folder_name)

    # Check that the directory exists
    assert export_dir.is_dir()
    # Check that the directory is located inside tmp_path
    assert export_dir.parent == folder_path


# --- Test export_spike_times --- #
def test_export_spike_times(tmp_path):
    # Create synthetic data_export with two clusters.
    data_export = {
        0: np.array([100, 200, 300]),
        1: np.array([150, 250, 350]),
    }
    # Use a temporary directory as export_dir.
    export_dir = tmp_path / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    csv_path = export_dir / "spike_times_by_cluster_time_ms.csv"
    txt_export_dir = export_dir / "txt_files_for_clampfit_import"
    txt_export_dir.mkdir(parents=True, exist_ok=True)

    # Call export_spike_times
    export_spike_times(data_export, csv_path, txt_export_dir)

    # Check that the CSV file exists.
    assert csv_path.is_file()

    # Check that the text files directory exists and contains a file for each cluster.
    assert txt_export_dir.is_dir()
    for cid in data_export.keys():
        txt_file = txt_export_dir / f"spike_times_Cluster_{cid}_time_ms.txt"
        assert txt_file.is_file()


# --- Test export_hazard_excel --- #
def test_export_hazard_excel(tmp_path):
    # Create synthetic DataFrames for hazard and ISI data.
    bin_starts = np.array([0.0, 0.1, 0.2])

    # Synthetic main ISI histogram DataFrame.
    isi_df = pd.DataFrame(
        {
            "Bin_Starts": bin_starts,
            "Cluster_0": np.array([5, 3, 2]),
        }
    )

    # Synthetic hazard function DataFrame.
    hazard_df = pd.DataFrame(
        {
            "Bin_Starts": bin_starts,
            "Cluster_0": np.array([0.5, 0.3, 0.2]),
        }
    )

    # Synthetic hazard summary DataFrame.
    hazard_summary_df = pd.DataFrame(
        {
            "Cluster": ["Cluster_0"],
            "Peak Early Hazard": [0.5],
            "Mean Late Hazard": [0.2],
            "Hazard Ratio": [2.5],
        }
    )

    # Synthetic baseline ISI histogram DataFrame.
    baseline_isi_df = pd.DataFrame(
        {
            "Bin_Starts": bin_starts,
            "Cluster_0_Baseline": np.array([2, 2, 1]),
        }
    )

    # Synthetic baseline hazard DataFrame.
    baseline_hazard_df = pd.DataFrame(
        {
            "Bin_Starts": bin_starts,
            "Cluster_0_Baseline": np.array([0.2, 0.1, 0.05]),
        }
    )

    # Synthetic baseline hazard summary DataFrame.
    baseline_hazard_summary_df = pd.DataFrame(
        {
            "Cluster": ["Cluster_0_Baseline"],
            "Peak Early Hazard": [0.2],
            "Mean Late Hazard": [0.05],
            "Hazard Ratio": [4.0],
        }
    )

    # Use a temporary directory as export_dir.
    export_dir = tmp_path / "export_hazard"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Call export_hazard_excel.
    export_hazard_excel(
        export_dir,
        hazard_df,
        hazard_summary_df,
        isi_df,
        baseline_isi_df,
        baseline_hazard_df,
        baseline_hazard_summary_df,
    )

    # Check that the Excel file exists.
    excel_path = export_dir / "isi_and_hazard_analysis.xlsx"
    assert excel_path.is_file()


def test_export_data_creates_files(tmp_path):
    output_dir = tmp_path / "output"
    analysis_results_dir = output_dir / "analysis_results"
    images_dir = analysis_results_dir / "firing_rate_images"
    txt_export_dir = analysis_results_dir / "txt_files_for_clampfit_import"

    export_data(
        data_export={0: np.array([100, 200, 300])},
        baseline_fr_dict={0: 5.0},
        data_folder_path=output_dir,
        bin_size=0.1,
        start_time=0.0,
        end_time=1.0,
        baseline_start=None,
        baseline_end=None,
    )

    # Ensure files exist in the correct location
    assert (analysis_results_dir / "firing_rates_by_cluster.xlsx").exists()
    assert (analysis_results_dir / "spike_times_by_cluster_time_ms.csv").exists()
    assert images_dir.exists()
    assert txt_export_dir.exists()


@pytest.fixture
def mock_data():
    """Fixture to provide sample spike times."""
    return {
        0: np.array([100, 200, 300]),
        1: np.array([150, 250, 350]),
    }


@pytest.fixture
def mock_firing_rate():
    """Fixture to return fake firing rate data."""
    return {"Time Intervals (s)": np.array([0.0, 0.1])}, None  # raw_data, delta_data


@pytest.fixture
def mock_firing_df():
    """Fixture to return a fake DataFrame for firing rates."""
    return pd.DataFrame({"Time Intervals (s)": [0.0, 0.1]}), None


@pytest.fixture
def mock_baselined_df():
    """Fixture to return a fake DataFrame for baseline stats."""
    return pd.DataFrame({"Cluster": ["Cluster_0"], "Mean Firing Rate": [5.0]})


@pytest.fixture
def mock_output_dirs(tmp_path):
    """Fixture to provide temporary output directories."""
    return tmp_path / "export", tmp_path / "images", tmp_path / "txt"


def test_export_data(
    mock_data,
    mock_firing_rate,
    mock_firing_df,
    mock_baselined_df,
    tmp_path,
    monkeypatch,
):
    """Tests that export_data correctly processes data and creates necessary files."""

    # Use tmp_path to create an isolated test directory
    output_dir = tmp_path / "output"

    # Mock dependencies except for make_output_folders
    monkeypatch.setattr("core.results_writer.export_spike_times", lambda *args: None)
    monkeypatch.setattr(
        "core.results_writer.calculate_firing_rate", lambda *args: mock_firing_rate
    )
    monkeypatch.setattr(
        "core.results_writer.create_firing_rate_dataframes",
        lambda *args: mock_firing_df,
    )
    monkeypatch.setattr(
        "core.results_writer.create_baselined_df", lambda *args: mock_baselined_df
    )

    # Call function (make_output_folders runs normally)
    export_dir, images_dir, df_raw = export_data(
        data_export=mock_data,
        baseline_fr_dict={0: 5.0, 1: 6.0},
        data_folder_path=output_dir,
        bin_size=0.1,
        start_time=0.0,
        end_time=1.0,
        baseline_start=0.0,
        baseline_end=0.5,
    )

    # Assert that the directories were actually created
    assert export_dir.exists()
    assert images_dir.exists()

    # Ensure firing rate output DataFrame is not None
    assert df_raw is not None


def test_export_data_no_spikes(tmp_path):
    """Test export_data when there are no spikes in data_export."""

    output_dir = tmp_path / "output"

    export_dir, images_dir, df_raw = export_data(
        data_export={},  # Empty data export
        baseline_fr_dict={},
        data_folder_path=output_dir,
        bin_size=0.1,
        start_time=0.0,
        end_time=1.0,
        baseline_start=None,
        baseline_end=None,
    )

    # Ensure function returns directories but no DataFrame
    assert export_dir.exists()
    assert images_dir.exists()
    assert df_raw is None


def test_export_data_xlsx_content(tmp_path):
    output_dir = tmp_path / "output"

    export_data(
        data_export={0: np.array([100, 200, 300])},
        baseline_fr_dict={0: 5.0},
        data_folder_path=output_dir,
        bin_size=0.1,
        start_time=0.0,
        end_time=1.0,
        baseline_start=0.0,
        baseline_end=0.5,
    )

    xlsx_path = output_dir / "analysis_results" / "firing_rates_by_cluster.xlsx"
    assert xlsx_path.exists()

    # Verify content of the Excel file
    with pd.ExcelFile(xlsx_path) as xls:
        assert "Firing_Rates_Raw" in xls.sheet_names
        df = pd.read_excel(xls, sheet_name="Firing_Rates_Raw")
        assert not df.empty  # Ensure there is data in the sheet

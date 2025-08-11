import numpy as np
import pandas as pd

from core.isi_hazard import (
    calculate_isi_histogram,
    compute_hazard_values,
    compute_hazard_summary,
    calculate_hazard_function,
)

# --- Test for calculate_isi_histogram --- #


def test_calculate_isi_histogram_without_baseline():
    # Create synthetic data_export dictionary:
    # Assume two clusters with spike times (in ms).
    data_export = {
        # spikes at 0.1s, 0.2s, ... 0.5s
        0: np.array([100, 200, 300, 400, 500]),
        # spikes at 0.15s, 0.25s, ... 0.55s
        1: np.array([150, 250, 350, 450, 550]),
    }
    time_bin = 0.1  # seconds
    max_isi_time = 0.5  # seconds

    # Call the function without baseline specification.
    isi_df, baseline_isi_df = calculate_isi_histogram(
        data_export,
        baseline_start=None,
        baseline_end=None,
        time_bin=time_bin,
        max_isi_time=max_isi_time,
    )

    # Check that the returned DataFrame has a "Bin_Starts" column and columns for each cluster.
    expected_columns = {"Bin_Starts", "Cluster_0", "Cluster_1"}
    assert set(isi_df.columns) == expected_columns
    # Since spike times are very regular, we expect nonzero counts in some bins.
    assert isi_df["Cluster_0"].sum() > 0
    # When no baseline is provided, baseline_isi_df should be None.
    assert baseline_isi_df is None


def test_calculate_isi_histogram_with_baseline():
    # Create synthetic data_export dictionary:
    data_export = {
        # spikes at 0.1s, 0.2s, ... 0.5s
        0: np.array([100, 200, 300, 400, 500]),
    }
    time_bin = 0.1  # seconds
    max_isi_time = 0.5  # seconds
    # Specify baseline period: for example, between 0.15 and 0.45 seconds.
    baseline_start = 0.15
    baseline_end = 0.45

    isi_df, baseline_isi_df = calculate_isi_histogram(
        data_export,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        time_bin=time_bin,
        max_isi_time=max_isi_time,
    )

    # Check that the main isi_df has the expected column.
    expected_columns = {"Bin_Starts", "Cluster_0"}
    assert set(isi_df.columns) == expected_columns
    # baseline_isi_df should not be None.
    assert baseline_isi_df is not None
    # It should have a column "Bin_Starts" and "Cluster_0_Baseline".
    expected_baseline_columns = {"Bin_Starts", "Cluster_0_Baseline"}
    assert set(baseline_isi_df.columns) == expected_baseline_columns


# --- Test for compute_hazard_values --- #


def test_compute_hazard_values():
    # Create a synthetic ISI histogram DataFrame.
    bin_starts = np.array([0.0, 0.1, 0.2])
    # For two clusters, provide arbitrary counts.
    data = {
        "Bin_Starts": bin_starts,
        "Cluster_0": np.array([10, 5, 0]),
        "Cluster_1": np.array([8, 4, 2]),
    }
    isi_df = pd.DataFrame(data)
    # Compute hazard values.
    hazard_df = compute_hazard_values(isi_df, bin_starts)
    # Check that hazard_df has the same columns.
    assert set(hazard_df.columns) == set(isi_df.columns)
    # Ensure that hazard values are computed and are floats.
    for col in hazard_df.columns:
        if col != "Bin_Starts":
            assert hazard_df[col].dtype == float


# --- Test for compute_hazard_summary --- #


def test_compute_hazard_summary():
    # Create a synthetic hazard DataFrame.
    bin_starts = np.array([0.0, 0.1, 0.2, 0.3])
    data = {
        "Bin_Starts": bin_starts,
        "Cluster_0": np.array([0.2, 0.5, 0.3, 0.1]),
    }
    hazard_df = pd.DataFrame(data)
    # Set early threshold to 0.15s and late period from 0.2 to 0.3s.
    early_time = 0.15
    late_time_start = 0.2
    late_time_end = 0.3
    summary_df = compute_hazard_summary(
        hazard_df, bin_starts, early_time, late_time_start, late_time_end
    )
    # Check that summary_df contains a row for "Cluster_0" and has the expected columns.
    expected_cols = {"Cluster", "Peak Early Hazard", "Mean Late Hazard", "Hazard Ratio"}
    assert set(summary_df.columns) == expected_cols
    # Ensure the "Cluster" column contains "Cluster_0".
    assert "Cluster_0" in summary_df["Cluster"].values


# --- Test for calculate_hazard_function --- #


def test_calculate_hazard_function_without_baseline():
    # Create a synthetic ISI histogram DataFrame.
    bin_starts = np.linspace(0, 0.4, 5)  # e.g., [0.0, 0.1, 0.2, 0.3, 0.4]
    data = {
        "Bin_Starts": bin_starts,
        "Cluster_0": np.array([5, 3, 2, 0, 0]),
    }
    isi_df = pd.DataFrame(data)
    # Call calculate_hazard_function without baseline.
    hazard_df, summary_df, baseline_hazard_df, baseline_summary_df = (
        calculate_hazard_function(
            isi_df,
            baseline_isi_df=None,
            early_time=0.15,
            late_time_start=0.2,
            late_time_end=0.3,
        )
    )
    # Ensure that baseline outputs are None.
    assert baseline_hazard_df is None
    assert baseline_summary_df is None
    # Check that hazard_df and summary_df are DataFrames.
    assert isinstance(hazard_df, pd.DataFrame)
    assert isinstance(summary_df, pd.DataFrame)


def test_calculate_hazard_function_with_baseline():
    # Create a synthetic main ISI histogram DataFrame.
    bin_starts = np.linspace(0, 0.4, 5)
    main_data = {
        "Bin_Starts": bin_starts,
        "Cluster_0": np.array([5, 3, 2, 0, 0]),
    }
    isi_df = pd.DataFrame(main_data)
    # Create a synthetic baseline ISI histogram DataFrame.
    baseline_data = {
        "Bin_Starts": bin_starts,
        "Cluster_0_Baseline": np.array([2, 2, 1, 0, 0]),
    }
    baseline_isi_df = pd.DataFrame(baseline_data)
    hazard_df, summary_df, baseline_hazard_df, baseline_summary_df = (
        calculate_hazard_function(
            isi_df,
            baseline_isi_df,
            early_time=0.15,
            late_time_start=0.2,
            late_time_end=0.3,
        )
    )
    # Check that baseline hazard outputs are DataFrames.
    assert isinstance(baseline_hazard_df, pd.DataFrame)
    assert isinstance(baseline_summary_df, pd.DataFrame)

    # For hazard function DataFrames, they should have "Bin_Starts" column.
    for df in [hazard_df, baseline_hazard_df]:
        assert "Bin_Starts" in df.columns

    # For summary DataFrames, they should have the expected summary columns.
    expected_summary_cols = {
        "Cluster",
        "Peak Early Hazard",
        "Mean Late Hazard",
        "Hazard Ratio",
    }
    for df in [summary_df, baseline_summary_df]:
        assert set(df.columns) == expected_summary_cols

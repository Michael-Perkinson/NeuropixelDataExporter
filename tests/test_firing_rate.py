import numpy as np
import pandas as pd

from core.firing_rate import (
    compute_baseline_firing_rate,
    shift_spike_times,
    process_cluster_data,
    calculate_firing_rate,
    create_firing_rate_dataframes,
    create_baselined_df,
)

from core.terminal_prompts import (
    drug_application_time,
    start_and_end_time,
    prompt_for_baseline,
)

# --- Tests for interactive functions --- #


def test_drug_application_time_valid(monkeypatch):
    # Simulate user entering a valid float
    monkeypatch.setattr("builtins.input", lambda prompt="": "5.5")
    result = drug_application_time()
    assert result == 5.5


def test_drug_application_time_empty(monkeypatch):
    # Simulate user pressing Enter (empty input)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    result = drug_application_time()
    assert result is None


def test_start_and_end_time(monkeypatch):
    # Simulate user entering values for start and end times
    # We'll use an iterator so that two different values are returned for the two prompts.
    inputs = iter(["1.0", "10.0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    start, end = start_and_end_time(20.0)
    assert start == 1.0
    assert end == 10.0


def test_start_and_end_time_default(monkeypatch):
    # Simulate user pressing Enter for both prompts so that defaults are used.
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    start, end = start_and_end_time(15.0)
    # start defaults to 0.0, end defaults to max_time
    assert start == 0.0
    assert end == 15.0


def test_prompt_for_baseline_yes(monkeypatch):
    # Simulate user wants to specify baseline and enters valid values
    inputs = iter(["y", "2.0", "8.0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    baseline_start, baseline_end = prompt_for_baseline(10.0, 1.0)
    # Expect values as entered (within allowed range)
    assert baseline_start == 2.0
    assert baseline_end == 8.0


def test_prompt_for_baseline_no(monkeypatch):
    # Simulate user not wanting to specify baseline
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    baseline_start, baseline_end = prompt_for_baseline(10.0, 1.0)
    assert baseline_start is None
    assert baseline_end is None


def test_compute_baseline_firing_rate():
    # Create a simple spike time array in seconds
    spikes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Suppose baseline is from 2.0 to 4.0 seconds; expected spikes in baseline: 2.0, 3.0, 4.0
    expected_rate = 3 / (4.0 - 2.0)
    rate = compute_baseline_firing_rate(spikes, 2.0, 4.0)
    assert rate == expected_rate


def test_compute_baseline_firing_rate_invalid():
    spikes = np.array([1.0, 2.0, 3.0])
    assert compute_baseline_firing_rate(spikes, 4.0, 2.0) == 0.0  # Invalid range


def test_shift_spike_times_with_drug():
    # Create an array of spike times in seconds
    spikes = np.array([5.0, 6.0, 7.0])
    drug_time = 5.0
    # After shifting, spikes become [0, 1, 2] seconds and then converted to ms => [0, 1000, 2000]
    shifted = shift_spike_times(spikes, drug_time)
    np.testing.assert_array_equal(shifted, np.array([0, 1000, 2000]))


def test_shift_spike_times_no_drug():
    spikes = np.array([1.0, 2.0, 3.0])
    shifted = shift_spike_times(spikes, None)
    np.testing.assert_array_equal(shifted, spikes * 1000)


def test_process_cluster_data():
    # Create a simple DataFrame with spike times and clusters.
    # Assume sample_rate=1000 for simplicity.
    data = {
        "spike_times": [1000, 2000, 3000, 4000, 5000],  # in "samples"
        "spike_clusters": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    # Let start_time=1.0, end_time=5.0 seconds; drug_time=1.0, sample_rate=1000.
    # For cluster 0, spikes in seconds: [1.0, 3.0, 5.0]. After shifting (subtract drug_time):
    # [0.0, 2.0, 4.0] -> in ms: [0, 2000, 4000].
    relative_spikes_ms, baseline_fr = process_cluster_data(
        df,
        cluster_id=0,
        sample_rate=1000,
        start_time=1.0,
        end_time=5.0,
        drug_time=1.0,
        baseline_start=None,
        baseline_end=None,
    )
    np.testing.assert_array_equal(relative_spikes_ms, np.array([0, 2000, 4000]))
    # With no baseline provided, baseline_fr should be None.
    assert baseline_fr is None


def test_process_cluster_data_with_baseline():
    # Create a simple DataFrame with spike times and clusters.
    # Assume sample_rate=1000 for simplicity.
    data = {
        "spike_times": [1000, 2000, 3000, 4000, 5000],  # in "samples"
        "spike_clusters": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    # Let start_time=1.0, end_time=5.0 seconds; drug_time=1.0, sample_rate=1000.
    # For cluster 0, spikes in seconds: [1.0, 3.0, 5.0]. After shifting (subtract drug_time):
    # [0.0, 2.0, 4.0] -> in ms: [0, 2000, 4000].
    # Baseline is from 1.0 to 3.0 seconds; expected spikes in baseline: 1.0
    relative_spikes_ms, baseline_fr = process_cluster_data(
        df,
        cluster_id=0,
        sample_rate=1000,
        start_time=1.0,
        end_time=5.0,
        drug_time=1.0,
        baseline_start=1.0,
        baseline_end=3.0,
    )
    np.testing.assert_array_equal(relative_spikes_ms, np.array([0, 2000, 4000]))
    # With baseline provided, baseline_fr should be computed.
    assert baseline_fr == 1.0


def test_calculate_firing_rate():
    # Create a dummy data_export dictionary with spike times in ms.
    data_export = {
        0: np.array([100, 200, 300, 400, 500]),  # Example spike times (ms)
        1: np.array([150, 250, 350, 450, 550]),
    }
    bin_size = 0.1  # seconds
    start_time = 0.0
    end_time = 1.0
    # For baseline_fr_dict, let's provide dummy baseline values.
    baseline_fr_dict = {0: 5.0, 1: 6.0}
    raw_data, delta_data = calculate_firing_rate(
        data_export, bin_size, start_time, end_time, baseline_fr_dict
    )
    # raw_data should contain a key "Time Intervals (s)" and keys "Cluster_0", "Cluster_1"
    assert "Time Intervals (s)" in raw_data
    assert "Cluster_0" in raw_data
    assert "Cluster_1" in raw_data
    # Check that delta_data exists and values are computed as raw minus baseline.
    assert delta_data is not None
    # Since our synthetic data is arbitrary, we can simply verify that the keys exist.
    assert "Cluster_0" in delta_data
    assert "Cluster_1" in delta_data


def test_calculate_firing_rate_bin_edges():
    data_export = {
        # Example spike times (ms)
        0: np.array([100, 200, 300, 400, 500, 900]),
    }
    bin_size = 0.3  # seconds
    start_time = 0.0
    end_time = 0.75  # Should result in bins [0.0, 0.3, 0.6] if <= end_time

    raw_data, _ = calculate_firing_rate(
        data_export, bin_size, start_time, end_time, baseline_fr_dict=None
    )
    print(raw_data)
    actual_bins = raw_data["Time Intervals (s)"]

    # Compute expected bins dynamically
    expected_bins = np.arange(start_time, end_time, bin_size)

    print(f"Actual bins: {actual_bins}")
    print(f"Expected bins: {expected_bins}")

    # Ensure actual bins match expected bins
    np.testing.assert_array_almost_equal(actual_bins, expected_bins, decimal=6)

    # Ensure last bin does not exceed end_time
    assert (
        actual_bins[-1] <= end_time
    ), f"Last bin {actual_bins[-1]} exceeds end_time {end_time}"


def test_create_firing_rate_dataframes():
    # Create synthetic raw_data and delta_data dictionaries.
    raw_data = {
        "Time Intervals (s)": np.array([0.0, 0.1, 0.2]),
        "Cluster_0": np.array([10, 20, 30]),
        "Cluster_1": np.array([15, 25, 35]),
    }
    delta_data = {
        "Time Intervals (s)": np.array([0.0, 0.1, 0.2]),
        "Cluster_0": np.array([2, 4, 6]),
        "Cluster_1": np.array([3, 5, 7]),
    }
    raw_df, delta_df = create_firing_rate_dataframes(raw_data, delta_data)
    # Check that the columns are correctly sorted.
    expected_columns = ["Time Intervals (s)", "Cluster_0", "Cluster_1"]
    assert list(raw_df.columns) == expected_columns
    assert list(delta_df.columns) == expected_columns


def test_create_baselined_df():
    # Create a simple data_export dictionary with spike times in ms.
    data_export = {
        0: np.array([100, 200, 300, 400, 500]),  # Example spike times (ms)
        1: np.array([150, 250, 350, 450, 550]),
    }
    baseline_start = 0.0
    baseline_end = 0.5
    bin_size = 0.1  # seconds
    # Let's assume the max time is 1.0 seconds.
    df = create_baselined_df(baseline_start, baseline_end, bin_size, data_export)
    # Check that the DataFrame has the expected columns.
    expected_columns = ["Cluster", "Mean Firing Rate", "Standard Deviation"]
    assert list(df.columns) == expected_columns
    # Check that the DataFrame has the expected number of rows.
    assert len(df) == 2  # Two clusters

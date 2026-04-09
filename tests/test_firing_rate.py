import pytest
import numpy as np
import pandas as pd

from src.core.firing_rate import (
    compute_baseline_stats,
    shift_spike_times_to_ms,
    process_cluster_data,
    calculate_firing_rate,
    create_firing_rate_dataframes,
    create_baselined_df,
    SAMPLE_RATE_HZ,
)


def test_compute_baseline_stats():
    spikes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Window 2.0–4.0 s, spikes at 2.0, 3.0, 4.0 → 3/2 = 1.5 Hz
    mean, _ = compute_baseline_stats(spikes, 2.0, 4.0)
    assert mean == pytest.approx(1.5)


def test_compute_baseline_stats_invalid():
    spikes = np.array([1.0, 2.0, 3.0])
    mean, _ = compute_baseline_stats(spikes, 4.0, 2.0)
    assert mean == 0.0


def test_shift_spike_times_with_drug():
    spikes = np.array([5.0, 6.0, 7.0])
    shifted = shift_spike_times_to_ms(spikes, drug_time_s=5.0)
    np.testing.assert_array_equal(shifted, np.array([0.0, 1000.0, 2000.0]))


def test_shift_spike_times_no_drug():
    spikes = np.array([1.0, 2.0, 3.0])
    shifted = shift_spike_times_to_ms(spikes, drug_time_s=None)
    np.testing.assert_array_equal(shifted, spikes * 1000.0)


def _make_df_at_30k(spike_times_s: list[float], spike_clusters: list[int]) -> pd.DataFrame:
    """Build a DataFrame with spike_times stored as samples at SAMPLE_RATE_HZ."""
    samples = [int(t * SAMPLE_RATE_HZ) for t in spike_times_s]
    return pd.DataFrame({"spike_times": samples, "spike_clusters": spike_clusters})


def test_process_cluster_data_no_baseline():
    # Cluster 0 spikes at 1s, 3s, 5s; cluster 1 at 2s, 4s
    df = _make_df_at_30k(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0, 1, 0, 1, 0],
    )
    raw, baseline_dict = process_cluster_data(
        df,
        cluster_ids=[0],
        start_time=1.0,
        end_time=5.0,
        drug_time=None,
        baseline_start=None,
        baseline_end=None,
    )
    # drug_time=None → times kept as-is, converted to ms
    np.testing.assert_array_almost_equal(raw[0], np.array([1000.0, 3000.0, 5000.0]))
    assert baseline_dict is None


def test_process_cluster_data_with_baseline():
    df = _make_df_at_30k(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [0, 1, 0, 1, 0],
    )
    raw, baseline_dict = process_cluster_data(
        df,
        cluster_ids=[0],
        start_time=1.0,
        end_time=5.0,
        drug_time=None,
        baseline_start=1.0,
        baseline_end=3.0,
    )
    # Spikes in [1.0, 3.0] for cluster 0: 1.0, 3.0 → 2 spikes / 2s = 1.0 Hz mean
    assert baseline_dict is not None
    mean_fr, sd_fr = baseline_dict[0]
    assert mean_fr == pytest.approx(1.0)
    assert sd_fr >= 0.0


def test_calculate_firing_rate():
    data_export = {
        0: np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        1: np.array([150.0, 250.0, 350.0, 450.0, 550.0]),
    }
    baseline_stats_dict = {0: (5.0, 1.0), 1: (6.0, 1.0)}
    raw_data, delta_data = calculate_firing_rate(
        data_export, bin_size=0.1, start_time=0.0, end_time=1.0,
        baseline_stats_dict=baseline_stats_dict,
    )
    assert "Time Intervals (s)" in raw_data
    assert "Cluster_0" in raw_data
    assert "Cluster_1" in raw_data
    assert delta_data is not None
    assert "Cluster_0" in delta_data
    assert "Cluster_1" in delta_data


def test_calculate_firing_rate_bin_edges():
    data_export = {0: np.array([100.0, 200.0, 300.0, 400.0, 500.0, 900.0])}
    raw_data, _ = calculate_firing_rate(
        data_export, bin_size=0.3, start_time=0.0, end_time=0.75,
        baseline_stats_dict=None,
    )
    actual_bins = raw_data["Time Intervals (s)"]
    expected_bins = np.arange(0.0, 0.75, 0.3)
    np.testing.assert_array_almost_equal(actual_bins, expected_bins, decimal=6)
    assert actual_bins[-1] <= 0.75


def test_create_firing_rate_dataframes():
    raw_data = {
        "Time Intervals (s)": np.array([0.0, 0.1, 0.2]),
        "Cluster_0": np.array([10.0, 20.0, 30.0]),
        "Cluster_1": np.array([15.0, 25.0, 35.0]),
    }
    delta_data = {
        "Time Intervals (s)": np.array([0.0, 0.1, 0.2]),
        "Cluster_0": np.array([2.0, 4.0, 6.0]),
        "Cluster_1": np.array([3.0, 5.0, 7.0]),
    }
    raw_df, delta_df = create_firing_rate_dataframes(raw_data, delta_data)
    expected_columns = ["Time Intervals (s)", "Cluster_0", "Cluster_1"]
    assert list(raw_df.columns) == expected_columns
    assert list(delta_df.columns) == expected_columns


def test_create_baselined_df():
    baseline_stats_dict = {0: (1.5, 0.0), 1: (2.0, 0.0)}
    df = create_baselined_df(baseline_stats_dict)
    assert list(df.columns) == ["Cluster", "Mean Firing Rate (Hz)"]
    assert len(df) == 2
    assert df.loc[df["Cluster"] == "Cluster_0", "Mean Firing Rate (Hz)"].iloc[0] == pytest.approx(1.5)
    assert df.loc[df["Cluster"] == "Cluster_1", "Mean Firing Rate (Hz)"].iloc[0] == pytest.approx(2.0)

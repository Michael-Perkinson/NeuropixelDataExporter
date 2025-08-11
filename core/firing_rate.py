import numpy as np
import pandas as pd

from typing import Optional


def compute_baseline_firing_rate(
    spikes: np.ndarray, baseline_start: float, baseline_end: float
) -> float:
    """
    Compute the baseline firing rate for a cluster based on spike times.

    Args
        spikes: Array of spike times in seconds.
        baseline_start: Start time of the baseline window.
        baseline_end: End time of the baseline window.

    Returns
        Baseline firing rate in spikes per second. Returns 0.0 if duration is zero or negative.
    """
    if baseline_end <= baseline_start:
        return 0.0

    baseline_spikes = spikes[(spikes >= baseline_start) & (spikes <= baseline_end)]
    baseline_duration = baseline_end - baseline_start

    return len(baseline_spikes) / baseline_duration


def shift_spike_times(spikes: np.ndarray, drug_time: float | None) -> np.ndarray:
    """
    Shift spike times relative to the drug application time and convert to milliseconds.

    Args
        spikes: Array of spike times in seconds.
        drug_time: Time of drug application in seconds, or None.

    Returns
        Array of spike times in milliseconds, shifted to start at zero if drug_time is provided.
    """
    if drug_time is not None:
        spikes = spikes - drug_time
    return spikes[spikes >= 0] * 1000


def process_cluster_data(
    df: pd.DataFrame,
    cluster_id: list[int],
    start_time: float,
    end_time: float,
    drug_time: float | None,
    baseline_start: float | None,
    baseline_end: float | None,
) -> tuple[np.ndarray, float | None]:
    """
    Process spike data for a single cluster: filter spikes, shift times, and compute baseline firing rate.

    Args
        df: DataFrame containing spike times for all clusters.
        cluster_id: Cluster ID to process.
        start_time: Start time for analysis (seconds).
        end_time: End time for analysis (seconds).
        drug_time: Drug application time (seconds), or None.
        baseline_start: Baseline period start time (seconds), or None.
        baseline_end: Baseline period end time (seconds), or None.

    Returns
        A tuple (relative_spikes_ms, baseline_firing_rate):
        - relative_spikes_ms: Shifted spike times in milliseconds.
        - baseline_firing_rate: Computed baseline firing rate, or None if baseline was not specified.
    """
    sample_rate = 30000
    
    cluster_df = df.loc[df["spike_clusters"] == cluster_id]
    
    cluster_df["spike_times"] = pd.to_numeric(
        cluster_df["spike_times"], errors='coerce')
    spike_times_array: np.ndarray = cluster_df["spike_times"].values.astype(
        np.float64)
    
    spikes_in_seconds = spike_times_array / sample_rate

    baseline_fr = None
    if baseline_start is not None and baseline_end is not None:
        baseline_fr = compute_baseline_firing_rate(
            spikes_in_seconds, baseline_start, baseline_end
        )

    filtered_spikes = spikes_in_seconds[
        (spikes_in_seconds >= start_time) & (spikes_in_seconds <= end_time)
    ]
    relative_spikes_ms = shift_spike_times(filtered_spikes, drug_time)
    return relative_spikes_ms, baseline_fr


def calculate_firing_rate(
    data_export: dict[int, np.ndarray],
    bin_size: float,
    start_time: float,
    end_time: float,
    baseline_fr_dict: Optional[dict[int, float | None]] = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """
    Calculate raw firing rates and delta firing rates (if baseline provided) for each cluster.

    Args
        data_export: Dictionary mapping cluster IDs to spike times in milliseconds.
        bin_size: Bin size in seconds.
        start_time: Start time for analysis (seconds).
        end_time: End time for analysis (seconds).
        baseline_fr_dict: (Optional) Dictionary mapping cluster IDs to their baseline firing rate.

    Returns
        A tuple (raw_data, delta_data):
        - raw_data: Dictionary of firing rates per bin.
        - delta_data: Dictionary of firing rates minus baseline values, or None if no baseline provided.
    """
    bins = np.arange(start_time, end_time + bin_size, bin_size)

    # Ensure bins do not exceed end_time
    if bins[-1] > end_time:
        bins = bins[bins <= end_time]
        print(
            f"Warning: Last bin exceeded end_time ({end_time}s) and has been removed."
        )

    raw_data = {"Time Intervals (s)": bins}
    delta_data = {} if baseline_fr_dict is not None else None

    for channel, spikes in data_export.items():
        spike_times_sec = spikes / 1000.0
        counts, _ = np.histogram(spike_times_sec, bins=bins)
        raw_data[f"Cluster_{channel}"] = counts / bin_size

        if delta_data is not None and baseline_fr_dict is not None:
            current_baseline_value_raw = baseline_fr_dict.get(channel)
            
            if current_baseline_value_raw is not None:
                baseline_value: float = float(current_baseline_value_raw)
                
                delta_data[f"Cluster_{channel}"] = (
                    raw_data[f"Cluster_{channel}"] - baseline_value
                )

    return raw_data, delta_data


def create_firing_rate_dataframes(
    raw_data: dict[str, np.ndarray], delta_data: dict[str, np.ndarray] | None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Convert raw and delta firing rate dictionaries into sorted Pandas DataFrames.

    Args
        raw_data: Dictionary with raw firing rates.
        delta_data: Dictionary with delta firing rates, or None.

    Returns
        A tuple (raw_df, delta_df):
        - raw_df: DataFrame of raw firing rates.
        - delta_df: DataFrame of delta firing rates, or None if not applicable.
    """
    sorted_cluster_columns = sorted(
        [col for col in raw_data if col.startswith("Cluster_")],
        key=lambda x: int(x.split("_")[1]),
    )
    sorted_columns = ["Time Intervals (s)"] + sorted_cluster_columns

    # Truncate "Time Intervals (s)" to match other columns
    min_length = min(
        len(raw_data["Time Intervals (s)"]),
        *[len(raw_data[col]) for col in sorted_cluster_columns],
    )

    raw_data["Time Intervals (s)"] = raw_data["Time Intervals (s)"][:min_length]

    for col in sorted_cluster_columns:
        raw_data[col] = raw_data[col][:min_length]

    raw_df = pd.DataFrame(raw_data)[sorted_columns]

    delta_df = None
    if delta_data:
        delta_data["Time Intervals (s)"] = raw_data["Time Intervals (s)"]
        delta_sorted_columns = ["Time Intervals (s)"] + [
            col for col in sorted_cluster_columns if col in delta_data
        ]
        delta_df = pd.DataFrame(delta_data)[delta_sorted_columns]

    return raw_df, delta_df


def create_baselined_df(
    baseline_start: float,
    baseline_end: float,
    bin_size: float,
    data_export: dict[int, np.ndarray],
) -> pd.DataFrame:
    """
    Create a DataFrame from the baseline firing rates dictionary.

    Args
        baseline_start: Start time of the baseline period.
        baseline_end: End time of the baseline period.
        bin_size: Bin size for firing rate calculation.
        data_export: Dictionary of baseline firing rates per cluster.

    Returns
        DataFrame containing mean firing rate and standard deviation per cluster.
    """
    baseline_bins = np.arange(baseline_start, baseline_end, bin_size)
    baselined_data = []

    for channel, spikes in data_export.items():
        spike_times_sec = spikes / 1000.0
        counts, _ = np.histogram(spike_times_sec, bins=baseline_bins)
        firing_rates = counts / bin_size
        baselined_data.append(
            {
                "Cluster": f"Cluster_{channel}",
                "Mean Firing Rate": firing_rates.mean(),
                "Standard Deviation": firing_rates.std(),
            }
        )

    return pd.DataFrame(baselined_data)

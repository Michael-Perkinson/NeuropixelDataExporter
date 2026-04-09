from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SAMPLE_RATE_HZ: int = 30000


def compute_baseline_stats(
    spikes_s: NDArray[np.float64],
    baseline_start: float,
    baseline_end: float,
) -> tuple[float, float]:
    """
    Compute mean firing rate in the baseline window.
    Returns (mean_hz, 0.0) — SD retained in signature for compatibility but always 0.
    """
    if baseline_end <= baseline_start:
        return 0.0, 0.0

    n = int(((spikes_s >= baseline_start) & (spikes_s <= baseline_end)).sum())
    mean = n / (baseline_end - baseline_start)
    return mean, 0.0


def shift_spike_times_to_ms(
    spikes_s: NDArray[np.float64],
    drug_time_s: float | None,
) -> NDArray[np.float64]:
    """
    Shift spike times relative to drug time (if provided), drop negatives, convert to ms.
    """
    if drug_time_s is not None:
        spikes_s = spikes_s - drug_time_s

    spikes_s = spikes_s[spikes_s >= 0.0]
    return spikes_s * 1000.0


def process_one_cluster(
    df: pd.DataFrame,
    cluster_id: int,
    start_time_s: float,
    end_time_s: float,
    drug_time_s: float | None,
    baseline_start_s: float | None,
    baseline_end_s: float | None,
) -> tuple[NDArray[np.float64], tuple[float, float] | None]:
    """
    Process a single cluster:
      - Convert spike times from samples -> seconds using fixed SAMPLE_RATE_HZ
      - Optionally compute baseline firing rate stats (mean, SD) from raw spike times
      - Filter to analysis window [start_time_s, end_time_s]
      - Shift by drug time (if provided) and convert to ms

    Returns (spikes_ms, (mean_fr, sd_fr) or None).
    """
    cluster_times = df.loc[df["spike_clusters"] == cluster_id, "spike_times"]
    spike_samples = pd.to_numeric(
        cluster_times, errors="coerce").to_numpy(dtype=np.float64)

    spike_samples = spike_samples[~np.isnan(spike_samples)]
    spikes_s: NDArray[np.float64] = spike_samples / float(SAMPLE_RATE_HZ)

    baseline_stats: tuple[float, float] | None = None
    if baseline_start_s is not None and baseline_end_s is not None:
        baseline_stats = compute_baseline_stats(
            spikes_s, baseline_start_s, baseline_end_s)

    # Filter to analysis window
    in_window = (spikes_s >= start_time_s) & (spikes_s <= end_time_s)
    filtered_s = spikes_s[in_window]

    spikes_ms = shift_spike_times_to_ms(filtered_s, drug_time_s)
    return spikes_ms, baseline_stats


def process_cluster_data(
    df: pd.DataFrame,
    cluster_ids: list[int],
    start_time: float,
    end_time: float,
    drug_time: float | None,
    baseline_start: float | None,
    baseline_end: float | None,
) -> tuple[dict[int, NDArray[np.float64]], dict[int, tuple[float, float]] | None]:
    """
    Process multiple clusters and return:
      - raw_spikes_ms: dict[cluster_id -> spike times (ms)]
      - baseline_stats_dict: dict[cluster_id -> (mean_fr, sd_fr)] or None
    """
    raw_spikes_ms: dict[int, NDArray[np.float64]] = {}
    baseline_stats_dict: dict[int, tuple[float, float]] | None = (
        {} if (baseline_start is not None and baseline_end is not None) else None
    )

    for cid in cluster_ids:
        spikes_ms, baseline_stats = process_one_cluster(
            df=df,
            cluster_id=cid,
            start_time_s=start_time,
            end_time_s=end_time,
            drug_time_s=drug_time,
            baseline_start_s=baseline_start,
            baseline_end_s=baseline_end,
        )
        raw_spikes_ms[cid] = spikes_ms
        if baseline_stats_dict is not None and baseline_stats is not None:
            baseline_stats_dict[cid] = baseline_stats

    return raw_spikes_ms, baseline_stats_dict


def calculate_firing_rate(
    data_export: dict[int, NDArray[np.float64]],
    bin_size: float,
    start_time: float,
    end_time: float,
    baseline_stats_dict: dict[int, tuple[float, float]] | None = None,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.float64]] | None]:
    """
    Calculate raw firing rates and delta firing rates (if baseline provided) for each cluster.

    Returns:
      raw_data: keys "Time Intervals (s)" and "Cluster_{id}"
      delta_data: same keys, values are (raw - baseline_mean) when baseline is available
    """
    bins = np.arange(start_time, end_time + bin_size,
                     bin_size, dtype=np.float64)

    if bins.size > 0 and bins[-1] > end_time:
        bins = bins[bins <= end_time]

    raw_data: dict[str, NDArray[np.float64]] = {"Time Intervals (s)": bins}
    delta_data: dict[str, NDArray[np.float64]] | None = (
        {} if baseline_stats_dict is not None else None
    )

    for channel, spikes_ms in data_export.items():
        spike_times_sec = spikes_ms / 1000.0
        counts, _ = np.histogram(spike_times_sec, bins=bins)
        fr = counts.astype(np.float64) / float(bin_size)

        raw_data[f"Cluster_{channel}"] = fr

        if delta_data is not None and baseline_stats_dict is not None:
            stats = baseline_stats_dict.get(channel)
            if stats is not None:
                delta_data[f"Cluster_{channel}"] = fr - stats[0]

    return raw_data, delta_data


def create_firing_rate_dataframes(
    raw_data: dict[str, NDArray[np.float64]],
    delta_data: dict[str, NDArray[np.float64]] | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Convert raw and delta firing rate dictionaries into sorted Pandas DataFrames.
    """
    cluster_cols = sorted(
        [col for col in raw_data.keys() if col.startswith("Cluster_")],
        key=lambda x: int(x.split("_")[1]),
    )
    sorted_columns = ["Time Intervals (s)"] + cluster_cols

    # Ensure equal lengths
    min_length = min(
        raw_data["Time Intervals (s)"].size,
        *[raw_data[col].size for col in cluster_cols],
    )
    raw_data["Time Intervals (s)"] = raw_data["Time Intervals (s)"][:min_length]
    for col in cluster_cols:
        raw_data[col] = raw_data[col][:min_length]

    raw_df = pd.DataFrame(raw_data)[sorted_columns]

    delta_df: pd.DataFrame | None = None
    if delta_data:
        delta_data["Time Intervals (s)"] = raw_data["Time Intervals (s)"]
        delta_sorted_columns = ["Time Intervals (s)"] + [
            col for col in cluster_cols if col in delta_data
        ]
        delta_df = pd.DataFrame(delta_data)[delta_sorted_columns]

    return raw_df, delta_df


def create_baselined_df(
    baseline_stats_dict: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    """
    Create a DataFrame of mean baseline firing rate per cluster.
    Expects baseline_stats_dict from process_cluster_data: {cluster_id: (mean_hz, _)}.
    """
    rows: list[dict[str, float | str]] = []
    for channel, (mean_fr, _) in sorted(baseline_stats_dict.items()):
        rows.append(
            {
                "Cluster": f"Cluster_{channel}",
                "Mean Firing Rate (Hz)": mean_fr,
            }
        )
    return pd.DataFrame(rows)

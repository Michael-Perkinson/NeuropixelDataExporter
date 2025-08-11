import numpy as np
import pandas as pd


def calculate_isi_histogram(
    data_export: dict[int, np.ndarray],
    baseline_start: float | None = None,
    baseline_end: float | None = None,
    time_bin: float = 0.01,
    max_isi_time: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Calculate ISI histograms for each cluster.

    Args
        data_export: Dictionary mapping cluster IDs to arrays of spike times (in ms).
        baseline_start: Baseline period start time (seconds), or None.
        baseline_end: Baseline period end time (seconds), or None.
        time_bin: Bin width for the ISI histogram (seconds).
        max_isi_time: Maximum ISI value to consider (seconds).

    Returns
        A tuple containing:
        - A DataFrame with ISI histogram data for each cluster.
        - A DataFrame with baseline ISI histogram data if baseline is provided, otherwise None.
    """
    baseline_data = (
        {} if baseline_start is not None and baseline_end is not None else None
    )

    n_bins = int(max_isi_time / time_bin)
    bin_edges = np.arange(0, (n_bins + 1) * time_bin, time_bin)
    isi_data = {"Bin_Starts": bin_edges[:-1]}

    for channel, spikes in data_export.items():
        spikes_sec = spikes / 1000.0  # Convert ms to seconds
        isis = np.diff(spikes_sec)
        hist, _ = np.histogram(isis, bins=bin_edges)
        isi_data[channel] = hist

        if baseline_data is not None:
            baseline_spikes = spikes_sec[
                (spikes_sec >= baseline_start) & (spikes_sec <= baseline_end)
            ]
            baseline_isis = np.diff(baseline_spikes)
            baseline_hist, _ = np.histogram(baseline_isis, bins=bin_edges)
            baseline_data[channel] = baseline_hist

    isi_df = pd.DataFrame(
        {
            "Bin_Starts": isi_data["Bin_Starts"],
            **{f"Cluster_{ch}": isi_data[ch] for ch in isi_data if ch != "Bin_Starts"},
        }
    )

    baseline_isi_df = None
    if baseline_data is not None:
        baseline_isi_df = pd.DataFrame(
            {
                "Bin_Starts": bin_edges[:-1],
                **{f"Cluster_{ch}_Baseline": baseline_data[ch] for ch in baseline_data},
            }
        )

    return isi_df, baseline_isi_df


def compute_hazard_values(isi_df: pd.DataFrame, bin_starts: np.ndarray) -> pd.DataFrame:
    """
    Compute hazard function values for each cluster.

    Args
        isi_df: DataFrame containing ISI histogram counts.
        bin_starts: Array of bin start times (seconds).

    Returns
        DataFrame with hazard function values per cluster.
    """
    hazard_data = {"Bin_Starts": bin_starts}
    for channel in isi_df.columns:
        if channel == "Bin_Starts":
            continue

        counts = isi_df[channel].values
        total_spikes = counts.sum()
        cumsum_counts = np.cumsum(counts)
        hazard_values = np.divide(
            counts,
            np.maximum(total_spikes - cumsum_counts + counts, 1),
            where=(cumsum_counts < total_spikes),
            out=np.zeros_like(counts, dtype=float),
        )
        hazard_data[channel] = hazard_values

    return pd.DataFrame(hazard_data)


def compute_hazard_summary(
    hazard_df: pd.DataFrame,
    bin_starts: np.ndarray,
    early_time: float,
    late_time_start: float,
    late_time_end: float,
) -> pd.DataFrame:
    """
    Compute summary metrics for the hazard function.

    Args
        hazard_df: DataFrame containing hazard function values.
        bin_starts: Array of bin start times (seconds).
        early_time: Time threshold defining the early hazard period (seconds).
        late_time_start: Start of the late hazard period (seconds).
        late_time_end: End of the late hazard period (seconds).

    Returns
        DataFrame summarizing peak early hazard, mean late hazard, and hazard ratio per cluster.
    """
    summary = []
    for channel in hazard_df.columns:
        if channel == "Bin_Starts":
            continue

        hazard_values = hazard_df[channel].values
        early_mask = bin_starts <= early_time
        late_mask = (bin_starts >= late_time_start) & (bin_starts <= late_time_end)

        peak_early = hazard_values[early_mask].max() if early_mask.any() else 0
        mean_late = hazard_values[late_mask].mean() if late_mask.any() else 0
        hazard_ratio = peak_early / mean_late if mean_late > 0 else float("nan")

        summary.append(
            {
                "Cluster": channel,
                "Peak Early Hazard": peak_early,
                "Mean Late Hazard": mean_late,
                "Hazard Ratio": hazard_ratio,
            }
        )

    return pd.DataFrame(summary)


def calculate_hazard_function(
    isi_df: pd.DataFrame,
    baseline_isi_df: pd.DataFrame | None = None,
    early_time: float = 0.07,
    late_time_start: float = 0.41,
    late_time_end: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Calculate hazard functions and summary metrics for each cluster.

    Args
        isi_df: DataFrame with ISI histogram data.
        baseline_isi_df: Optional DataFrame with baseline ISI histogram data.
        early_time: Time threshold for the early hazard period.
        late_time_start: Start time for the late hazard period.
        late_time_end: End time for the late hazard period.

    Returns
        A tuple containing:
        - Hazard function DataFrame for main data.
        - Hazard summary DataFrame for main data.
        - Hazard function DataFrame for baseline, or None if not provided.
        - Hazard summary DataFrame for baseline, or None if not provided.
    """
    bin_starts = isi_df["Bin_Starts"].values
    hazard_df = compute_hazard_values(isi_df, bin_starts)
    hazard_summary_df = compute_hazard_summary(
        hazard_df, bin_starts, early_time, late_time_start, late_time_end
    )

    baseline_hazard_df = None
    baseline_hazard_summary_df = None
    if baseline_isi_df is not None:
        baseline_bin_starts = baseline_isi_df["Bin_Starts"].values
        baseline_hazard_df = compute_hazard_values(baseline_isi_df, baseline_bin_starts)
        baseline_hazard_summary_df = compute_hazard_summary(
            baseline_hazard_df,
            baseline_bin_starts,
            early_time,
            late_time_start,
            late_time_end,
        )

    return hazard_df, hazard_summary_df, baseline_hazard_df, baseline_hazard_summary_df

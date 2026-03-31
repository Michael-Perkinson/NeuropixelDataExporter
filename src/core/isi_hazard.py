import numpy as np
import pandas as pd


def calculate_windowed_isi(
    data_export: dict[int, np.ndarray],
    window_start: float,
    window_end: float,
    time_bin: float = 0.01,
    max_isi_time: float = 0.75,
    col_suffix: str = "",
) -> pd.DataFrame:
    """
    ISI histogram for spikes within [window_start, window_end] seconds.

    col_suffix is appended to each cluster column name (e.g. '_Pre', '_Post').
    """
    n_bins = int(max_isi_time / time_bin)
    bin_edges = np.arange(0, (n_bins + 1) * time_bin, time_bin)
    data: dict[str, np.ndarray] = {"Bin_Starts": bin_edges[:-1]}
    for cid, spikes_ms in data_export.items():
        spikes_s = spikes_ms / 1000.0
        windowed = spikes_s[(spikes_s >= window_start)
                            & (spikes_s <= window_end)]
        isis = np.diff(windowed)
        hist, _ = np.histogram(isis, bins=bin_edges)
        data[f"Cluster_{cid}{col_suffix}"] = hist
    return pd.DataFrame(data)


def calculate_isi_histogram(
    data_export: dict[int, np.ndarray],
    baseline_start: float | None = None,
    baseline_end: float | None = None,
    time_bin: float = 0.01,
    max_isi_time: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Calculate ISI histograms for each cluster over the full recording.

    If baseline_start and baseline_end are provided, also returns a baseline ISI DataFrame
    with columns named Cluster_N_Baseline.
    """
    baseline_data: dict[int, np.ndarray] | None = (
        {} if baseline_start is not None and baseline_end is not None else None
    )

    n_bins = int(max_isi_time / time_bin)
    bin_edges = np.arange(0, (n_bins + 1) * time_bin, time_bin)
    isi_data: dict[str, np.ndarray] = {"Bin_Starts": bin_edges[:-1]}

    for channel, spikes in data_export.items():
        spikes_sec = spikes / 1000.0
        isis = np.diff(spikes_sec)
        hist, _ = np.histogram(isis, bins=bin_edges)
        isi_data[channel] = hist  # type: ignore[index]

        if baseline_data is not None:
            assert baseline_start is not None and baseline_end is not None
            baseline_spikes = spikes_sec[
                (spikes_sec >= baseline_start) & (spikes_sec <= baseline_end)
            ]
            baseline_hist, _ = np.histogram(
                np.diff(baseline_spikes), bins=bin_edges)
            baseline_data[channel] = baseline_hist

    isi_df = pd.DataFrame({
        "Bin_Starts": isi_data["Bin_Starts"],
        **{f"Cluster_{ch}": isi_data[ch] for ch in isi_data if ch != "Bin_Starts"},
    })

    baseline_isi_df = None
    if baseline_data is not None:
        baseline_isi_df = pd.DataFrame({
            "Bin_Starts": bin_edges[:-1],
            **{f"Cluster_{ch}_Baseline": baseline_data[ch] for ch in baseline_data},
        })

    return isi_df, baseline_isi_df


def compute_hazard_values(isi_df: pd.DataFrame, bin_starts: np.ndarray) -> pd.DataFrame:
    """Compute hazard function values from an ISI histogram DataFrame."""
    hazard_data: dict[str, np.ndarray] = {"Bin_Starts": bin_starts}
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
    Summarise hazard metrics per cluster.

    Args:
        early_time: upper bound for the early hazard window (seconds).
        late_time_start / late_time_end: bounds for the late hazard window.
    """
    summary = []
    for channel in hazard_df.columns:
        if channel == "Bin_Starts":
            continue
        hazard_values = hazard_df[channel].values
        early_mask = bin_starts <= early_time
        late_mask = (bin_starts >= late_time_start) & (
            bin_starts <= late_time_end)
        peak_early = hazard_values[early_mask].max() if early_mask.any() else 0
        mean_late = hazard_values[late_mask].mean() if late_mask.any() else 0
        hazard_ratio = peak_early / \
            mean_late if mean_late > 0 else float("nan")
        summary.append({
            "Cluster": channel,
            "Peak Early Hazard": peak_early,
            "Mean Late Hazard": mean_late,
            "Hazard Ratio": hazard_ratio,
        })
    return pd.DataFrame(summary)


def calculate_hazard_function(
    isi_df: pd.DataFrame,
    baseline_isi_df: pd.DataFrame | None = None,
    early_time: float = 0.07,
    late_time_start: float = 0.41,
    late_time_end: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Compute hazard functions and summary metrics from an ISI histogram.

    Returns (hazard_df, summary_df, baseline_hazard_df, baseline_summary_df).
    Baseline outputs are None if baseline_isi_df is not provided.
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
        baseline_hazard_df = compute_hazard_values(
            baseline_isi_df, baseline_bin_starts)
        baseline_hazard_summary_df = compute_hazard_summary(
            baseline_hazard_df, baseline_bin_starts,
            early_time, late_time_start, late_time_end,
        )

    return hazard_df, hazard_summary_df, baseline_hazard_df, baseline_hazard_summary_df

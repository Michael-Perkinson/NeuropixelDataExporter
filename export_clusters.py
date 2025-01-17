import os
import sys
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from pandas import ExcelWriter
import plotly.graph_objects as go

REQUIRED_FILES = ["spike_times.npy", "spike_clusters.npy", "cluster_group.tsv"]


def file_chooser() -> str | None:
    """
    Prompts the user to select a folder within the current script's directory.

    Returns:
        str or None: The path to the selected folder, or None if no folder is selected.
    """
    root = tk.Tk()
    root.withdraw()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    folder_path = filedialog.askdirectory(
        initialdir=script_dir, title="Select a folder containing the required files"
    )

    if not folder_path:
        print("No folder selected.")
        return None

    print(f"""Selected folder: {folder_path}""")
    return folder_path


def find_specific_files_in_folder(
    folder_path: str, file_names: list[str]
) -> dict[str, str]:
    """
    Finds specific files in the provided folder.

    Parameters:
        folder_path (str): Path to the folder containing the files.
        file_names (list): List of file names to search for.

    Returns:
        dict: A dictionary mapping file names to their full paths.
    """
    file_paths = {}
    for file in file_names:
        full_path = os.path.join(folder_path, file)
        if os.path.exists(full_path):
            file_paths[file] = full_path
    return file_paths


def choose_and_validate_folder() -> dict[str, str]:
    """
    Prompts the user to choose a folder and validates the presence of required files.
    If required files are missing, prompts the user to pick another folder until valid
    files are found or the user cancels.

    Returns:
        dict[str, str]: A dictionary mapping required file names to their paths.
    """
    while True:  # Loop until valid files are found or the user cancels
        folder_path = file_chooser()
        if not folder_path:
            print("No folder selected. Exiting.")
            sys.exit(1)  # Exit the script if the user cancels folder selection

        # Find required files in the selected folder
        file_paths = find_specific_files_in_folder(folder_path, REQUIRED_FILES)
        missing_files = [file for file in REQUIRED_FILES if file not in file_paths]

        if missing_files:
            print(f"""Error: Missing required files: {', '.join(missing_files)}""")
            print("Please select another folder or cancel to exit.")
        else:
            return file_paths


def drug_application_time() -> float | None:
    """
    Prompts the user to enter the drug application time in seconds or skip if not provided.

    Returns:
        float or None: The entered drug application time in seconds, or None if skipped or invalid.
    """
    try:
        drug_time = input(
            "Enter the time at which the drug was applied (s), or press Enter to skip: "
        )
        return float(drug_time) if drug_time else None
    except ValueError:
        print("Invalid input. Ignoring drug application time.")
        return None


def start_and_end_time(max_time: float) -> tuple[float, float]:
    """
    Prompts the user to enter start and end times for a plot, defaulting to the full range if omitted.

    Parameters:
        max_time (float): The maximum allowable time for the end of the plot.

    Returns:
        tuple[float, float]: A tuple containing the start and end times in seconds.
    """

    def parse_input(prompt: str, default: float) -> float:
        """Helper to parse user input with a default fallback."""
        value = input(prompt).strip()
        return float(value) if value else default

    start_time = parse_input(
        "Enter the start time of the plot (seconds), or press Enter to start from 0: ",
        0.0,
    )
    end_time = parse_input(
        (
            f"""Enter the end time of the plot (seconds), or press Enter to use {max_time:.2f}: """
        ),
        max_time,
    )

    return start_time, end_time


def prompt_for_baseline(
    max_time: float, min_time: float = 0
) -> tuple[float, float] | tuple[None, None]:
    """
    Asks the user if they want to specify a baseline time range.
    Returns (None, None) if no baseline is specified.
    Otherwise returns (baseline_start, baseline_end).

    The minimum baseline time is adjusted if the start time is greater than 0.
    """
    if (
        input("Do you want to specify a baseline period? (y/n): ").strip().lower()
        != "y"
    ):
        return None, None

    try:
        baseline_start = float(
            input(f"""Enter baseline start time (>= {min_time:.2f}s): """)
        )
        baseline_end = float(input(f"""Enter baseline end time (<= {max_time:.2f}s): """))

        # Ensure the baseline start time is greater than or equal to min_time
        if baseline_start < min_time:
            print(
                f"""Baseline start time must be >= {min_time:.2f}s.
                Using the minimum time."""
            )
            baseline_start = min_time
        if baseline_end > max_time:
            print(
                f"""Baseline end time must be <= {max_time:.2f}s.
                Using the maximum time."""
            )
            baseline_end = max_time
        return baseline_start, baseline_end

    except ValueError:
        print("Invalid input for baseline times. Skipping baseline.")
        return None, None


def channels_or_labels_to_export() -> tuple[list[int], list[str]]:
    """
    Prompts user for channels or labels to export and parses the input.

    Returns:
        tuple[list[int], list[str]]: A tuple containing:
            - A sorted list of unique integer channel identifiers.
            - A sorted list of unique group labels (strings).
    """
    user_input = input(
        "Enter the channels or labels you want to export (separated by commas): "
    ).strip()

    # Parse input into a list of cleaned strings
    inputs = [item.strip() for item in user_input.split(",") if item.strip()]

    # Separate numeric channels and text labels
    channels = sorted({int(item) for item in inputs if item.isdigit()})
    labels = sorted({item for item in inputs if not item.isdigit()})

    # Provide clear feedback to the user
    if channels:
        print(f"""Selected channels: {channels}""")
    else:
        print("No numeric channels were selected.")

    if labels:
        print(f"""Selected labels: {labels}""")
    else:
        print("No group labels were selected.")

    return channels, labels


def load_spike_data(
    spike_times_path: str, spike_clusters_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads spike times and spike cluster data from .npy files.

    Parameters:
        spike_times_path (str): Path to the .npy file containing spike times.
        spike_clusters_path (str): Path to the .npy file containing spike cluster IDs.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - spike_times: Array of spike times.
            - spike_clusters: Array of spike cluster IDs.
    """
    return np.load(spike_times_path).ravel(), np.load(spike_clusters_path).ravel()


def create_label_lookup(group_labels_path: str) -> np.ndarray:
    """
    Creates a label lookup array that maps cluster IDs to group labels.

    Parameters:
        group_labels_path (str): Path to the .tsv file containing cluster IDs and group labels.
            The file should have columns:
            - "cluster_id": Integer IDs of clusters.
            - "group": Corresponding group labels for each cluster.

    Returns:
        np.ndarray: An array where each index corresponds to a cluster ID and its value is the group label.
                    Unmapped indices are labeled as "unknown".
    """
    # Check for the correct group label column
    group_label_column = None

    cluster_group = pd.read_csv(group_labels_path, sep="\t")
    max_cluster_id = cluster_group["cluster_id"].max() + 1
    group_labels_array = np.full(max_cluster_id, "unknown", dtype=object)

    if "group" in cluster_group.columns:
        group_label_column = "group"
    elif "KSLabel" in cluster_group.columns:
        group_label_column = "KSLabel"
    else:
        raise ValueError(
            "Neither 'group' nor 'KSLabel' column found in the group labels dataframe."
        )

    # Assign the values from the appropriate column
    group_labels_array[cluster_group["cluster_id"].values] = cluster_group[
        group_label_column
    ].values

    return group_labels_array


def filter_by_labels(
    spike_clusters: np.ndarray,
    group_labels_array: np.ndarray,
    labels_to_include: list[str],
) -> np.ndarray:
    """
    Creates a mask for filtering by group labels.

    Parameters:
        spike_clusters (np.ndarray): Array of spike cluster IDs.
        group_labels_array (np.ndarray): Array mapping cluster IDs to group labels.
        labels_to_include (list[str]): List of group labels to include.

    Returns:
        np.ndarray: A boolean mask for the selected labels.
    """
    valid_cluster_ids = np.where(np.isin(group_labels_array, labels_to_include))[0]
    print(
        f"""Filtered cluster IDs for labels
        {labels_to_include}: {valid_cluster_ids}"""
    )
    return np.isin(spike_clusters, valid_cluster_ids)


def filter_by_channels(
    spike_clusters: np.ndarray, channels_to_include: list[int]
) -> np.ndarray:
    """
    Creates a mask for filtering by channel IDs.

    Parameters:
        spike_clusters (np.ndarray): Array of spike cluster IDs.
        channels_to_include (list[int]): List of channel IDs to include.

    Returns:
        np.ndarray: A boolean mask for the selected channels.
    """
    return np.isin(spike_clusters, channels_to_include)


def filter_data(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    group_labels_array: np.ndarray,
    labels_to_include: list[str] = None,
    channels_to_include: list[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters spike data by user-specified labels and/or channels.

    Parameters:
        spike_times (np.ndarray): Array of spike times.
        spike_clusters (np.ndarray): Array of spike cluster IDs.
        group_labels_array (np.ndarray): Array mapping cluster IDs to group labels.
        labels_to_include (list[str], optional): List of group labels to include.
        channels_to_include (list[int], optional): List of channel IDs to include.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - filtered_spike_times: Filtered array of spike times.
            - filtered_spike_clusters: Filtered array of spike cluster IDs.
            - filtered_group_labels: Filtered array of group labels.
    """
    # Initialize an empty mask
    mask = np.zeros_like(spike_clusters, dtype=bool)

    # Apply label filtering
    if labels_to_include:
        mask |= filter_by_labels(spike_clusters, group_labels_array, labels_to_include)

    # Apply channel filtering
    if channels_to_include:
        mask |= filter_by_channels(spike_clusters, channels_to_include)

    # Apply the mask to filter data
    filtered_spike_times = spike_times[mask]
    filtered_spike_clusters = spike_clusters[mask]
    filtered_group_labels = group_labels_array[spike_clusters[mask]]

    return filtered_spike_times, filtered_spike_clusters, filtered_group_labels


def construct_dataframe(
    spike_times: np.ndarray, spike_clusters: np.ndarray, group_labels: np.ndarray
) -> pd.DataFrame:
    """
    Constructs a DataFrame from spike times, spike clusters, and group labels.

    Parameters:
        spike_times (np.ndarray): Array of spike times.
        spike_clusters (np.ndarray): Array of spike cluster IDs.
        group_labels (np.ndarray): Array of group labels.

    Returns:
        pd.DataFrame: A DataFrame containing spike times, spike clusters, and group labels.
    """
    return pd.DataFrame(
        {
            "spike_times": spike_times,
            "spike_clusters": spike_clusters,
            "group": group_labels,
        }
    )


def create_export_dir(folder_path: str, analysis_folder_name: str) -> str:
    """
    Creates a directory for exporting analysis results.

    Combines the provided folder path with a subdirectory name (analysis folder name),
    creates the directory if it does not already exist, and returns the full path.

    Parameters:
        folder_path (str): The path to the base folder where the analysis directory should be created.
        analysis_folder_name (str): The name of the subdirectory for storing exported analysis files.

    Returns:
        str: The full path to the created export directory.
    """
    export_dir = os.path.join(folder_path, analysis_folder_name)
    os.makedirs(export_dir, exist_ok=True)
    return export_dir


def export_spike_times(data_export: dict[int, np.ndarray], export_dir: str) -> None:
    """
    Exports spike times for each cluster to CSV and text files.

    Saves spike times for all clusters to a single CSV file and individual text files,
    organized within a subdirectory for easy access.

    Parameters:
        data_export (dict[int, np.ndarray]): A dictionary where keys are cluster IDs and
                                             values are arrays of spike times (in milliseconds).
        export_dir (str): The directory where the CSV and text files will be saved.

    Returns:
        None: This function saves files directly to the specified directory.
    """
    # Create and export CSV
    max_length = max(len(arr) for arr in data_export.values())
    df_spikes = pd.DataFrame(
        {
            f"""Cluster_{cid}""": np.pad(
                arr, (0, max_length - len(arr)), mode="constant", constant_values=np.nan
            )
            for cid, arr in data_export.items()
        }
    )
    csv_path = os.path.join(export_dir, "spike_times_by_cluster_time_ms.csv")
    df_spikes.to_csv(csv_path, index=False)
    print(f"""Spike times exported to {csv_path}""")

    # Create and export text files
    txt_export_dir = os.path.join(export_dir, "txt_files_for_clampfit_import")
    os.makedirs(txt_export_dir, exist_ok=True)
    for cid, arr in data_export.items():
        out_file = os.path.join(
            txt_export_dir, f"""spike_times_Cluster_{cid}_time_ms.txt"""
        )
        np.savetxt(out_file, arr, fmt="%.4f")
    print(f"""Text files saved to {txt_export_dir}""")


def export_data(
    data_export: dict[int, np.ndarray],
    baseline_fr_dict: dict[int, float | None],
    folder_path: str,
    bin_size: float,
    start_time: float,
    end_time: float,
    drug_time: float | None,
    baseline_start: float | None,
    baseline_end: float | None,
) -> str:
    """
    Exports analyzed spike data, firing rates, and interactive plots to an output directory.

    This function creates an export directory and performs the following tasks:
        1. Filters out clusters with no spike data.
        2. Exports spike times for each cluster to text files.
        3. Calculates raw and delta firing rates for each cluster.
        4. Saves firing rates as Excel sheets:
            - "Firing_Rates_Raw": Raw firing rates for each cluster and time bin.
            - "Delta_from_Baseline" (optional): Delta firing rates relative to the baseline.
            - "Baseline_Stats" (optional): Mean and standard deviation of firing rates for each cluster during the baseline period.
        5. Exports interactive firing rate plots as HTML files.

    Parameters:
        data_export (dict[int, np.ndarray]): A dictionary where keys are cluster IDs and
                                             values are arrays of spike times (in milliseconds).
        baseline_fr_dict (dict[int, float | None]): A dictionary mapping cluster IDs to their
                                                    baseline firing rates. None if unavailable.
        folder_path (str): Path to the folder containing the original data files.
        bin_size (float): Size of time bins (in seconds) for firing rate calculations.
        start_time (float): Start time for spike time analysis (in seconds).
        end_time (float): End time for spike time analysis (in seconds).
        drug_time (float | None): Time of drug application (in seconds), if applicable.
        baseline_start (float | None): Start time of the baseline period (in seconds).
        baseline_end (float | None): End time of the baseline period (in seconds).

    Returns:
        str: The path to the export directory where all files are saved.

    Raises:
        ValueError: If no spikes are available for export after filtering.
    """
    # Create export directory
    analysis_folder_name = f"""{os.path.basename(folder_path)}_analysed"""
    export_dir = create_export_dir(folder_path, analysis_folder_name)

    # Filter out clusters with zero spikes
    data_export = {cid: arr for cid,
                   arr in data_export.items() if len(arr) > 0}
    if not data_export:
        print("No spikes to export. Exiting.")
        return export_dir

    # Export spike times
    export_spike_times(data_export, export_dir)

    # Calculate firing rates
    raw_data, delta_data = calculate_firing_rate(
        data_export, bin_size, start_time, end_time, baseline_fr_dict
    )

    df_raw, df_delta = create_firing_rate_dataframes(raw_data, delta_data)

    # Calculate baseline statistics
    if baseline_start is not None and baseline_end is not None:
        baseline_bins = np.arange(
            baseline_start, baseline_end + bin_size, bin_size)
        baseline_stats = []

        for channel, spikes in data_export.items():
            # Convert spike times to seconds
            spike_times_sec = spikes / 1000.0
            # Calculate firing rates for baseline bins
            counts, _ = np.histogram(spike_times_sec, bins=baseline_bins)
            firing_rates = counts / bin_size
            # Append statistics
            baseline_stats.append(
                {
                    "Cluster": f"""Cluster_{channel}""",
                    "Mean Firing Rate": np.mean(firing_rates),
                    "Standard Deviation": np.std(firing_rates),
                }
            )

        # Convert stats to DataFrame
        baseline_stats_df = pd.DataFrame(baseline_stats)
        print(baseline_stats_df.columns)
        # Sort cluster names using your custom function
        sorted_clusters = sort_cluster_columns(
            baseline_stats_df["Cluster"].tolist())
        baseline_stats_df = baseline_stats_df.set_index(
            "Cluster").loc[sorted_clusters].reset_index()
        
        baseline_start_fmt = f"""{baseline_start:.2f}""".rstrip(
            "0").rstrip(".")
        baseline_end_fmt = f"""{baseline_end:.2f}""".rstrip("0").rstrip(".")
        sheet_name = f"""Baseline_Stats ({baseline_start_fmt}s - {baseline_end_fmt}s)"""

    # Export firing rates and baseline statistics to Excel
    xlsx_path = os.path.join(export_dir, "firing_rates_by_cluster.xlsx")

    with ExcelWriter(xlsx_path) as writer:
        df_raw.to_excel(writer, sheet_name="Firing_Rates_Raw", index=False)
        if df_delta is not None and len(df_delta.columns) > 1:
            df_delta.to_excel(
                writer, sheet_name="Delta_from_Baseline", index=False)
        if baseline_start is not None and baseline_end is not None:
            baseline_stats_df.to_excel(
                writer, sheet_name=sheet_name, index=False
            )
    print(
        f"""Firing rates (raw/delta) and baseline statistics exported to {xlsx_path}""")

    # Export firing rate plots
    images_dir = os.path.join(export_dir, "firing_rate_images")
    os.makedirs(images_dir, exist_ok=True)
    export_firing_rate_html(df_raw, images_dir, bin_size, drug_time)
    print(f"""Interactive firing rate plots exported to {images_dir}""")

    return export_dir

def export_firing_rate_html(
    firing_rate_df: pd.DataFrame,
    images_export_dir: str,
    bin_size: float,
    drug_time: float | None,
) -> None:
    """
    Creates and exports interactive HTML plots for firing rates of each cluster.

    Generates bar plots for the firing rates of each cluster, based on the provided DataFrame.
    The plots include a time axis, firing rate values, and an optional drug application marker.
    Each plot is saved as an interactive HTML file in the specified export directory.

    Parameters:
        firing_rate_df (pd.DataFrame): A DataFrame containing firing rate data.
                                       The first column should be 'Time Intervals (s)' with time bins,
                                       and subsequent columns should represent firing rates for clusters.
        images_export_dir (str): The directory where HTML plots will be saved.
        bin_size (float): The size of time bins (in seconds), included in plot titles and file names.
        drug_time (float | None): The time (in seconds) of drug application, marked on the plots if provided.

    Returns:
        None: Saves HTML plots directly to the specified directory.
    """
    bin_times = firing_rate_df["Time Intervals (s)"]

    for channel in firing_rate_df.columns[1:]:
        fig = go.Figure()

        # Add bar trace for firing rates
        fig.add_trace(
            go.Bar(
                x=bin_times,
                y=firing_rate_df[channel],
                name=f"""Cluster {channel}""",
                marker_color="black",
            )
        )

        # Add drug application marker if specified
        if drug_time is not None:
            max_y = max(firing_rate_df[channel]) * 1.1
            fig.add_shape(
                type="line",
                x0=drug_time,
                y0=0,
                x1=drug_time,
                y1=max_y,
                line={"color": "red", "width": 2, "dash": "dash"},
            )
            fig.add_annotation(
                x=drug_time,
                y=max_y,
                text="Drug Application",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20,
            )

        # Customize layout
        fig.update_layout(
            title=(
                f"""Firing Rate Histogram for Cluster {channel} """
                f"""(Bin Size: {bin_size}s)"""
            ),
            xaxis_title="Time (s)",
            yaxis_title="Firing Rate (Hz)",
            plot_bgcolor="white",
            bargap=0,
            xaxis={"showline": True, "linecolor": "black", "showgrid": False},
            yaxis={
                "showline": True,
                "linecolor": "black",
                "showgrid": False,
                "ticks": "outside",
            },
        )

        # Save HTML file
        html_path = os.path.join(
            images_export_dir,
            f"""Firing_Rate_Cluster_{channel}_BinSize_{bin_size}s.html""",
        )
        fig.write_html(html_path)

    print(
        f"""Interactive firing rate HTMLs for clusters saved to {images_export_dir}"""
    )


def extract_data(
    df: pd.DataFrame,
    drug_time: float | None,
    start_time: float,
    end_time: float,
    sample_rate: int,
    baseline_start: float | None = None,
    baseline_end: float | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, float | None]]:
    """
    Filters spike times in a DataFrame by specified start/end times and optionally computes a
    baseline firing rate for each cluster if baseline_start and baseline_end are provided.

    Parameters:
        df (pd.DataFrame): DataFrame of spikes, each row has spike_times, spike_clusters, etc.
        drug_time (float | None): Time of drug application (seconds), or None if not applicable.
        start_time (float): Start time for analysis (seconds).
        end_time (float): End time for analysis (seconds).
        sample_rate (int): Sampling rate in Hz.
        baseline_start (float | None): Start time of the baseline window (seconds), or None if no baseline.
        baseline_end (float | None): End time of the baseline window (seconds), or None if no baseline.

    Returns:
        tuple:
          - dict[int, np.ndarray]: Dictionary keyed by cluster ID with filtered/shifted spike times (ms).
          - dict[int, float | None]: Dictionary keyed by cluster ID with baseline firing rate (spikes/s)
            if computed, or None if no baseline.
    """
    cluster_ids = df["spike_clusters"].unique()
    data_export = {}
    baseline_fr_dict = {}

    for cluster_id in cluster_ids:
        # Process each cluster
        relative_spikes_ms, baseline_fr = process_cluster_data(
            df,
            cluster_id,
            sample_rate,
            start_time,
            end_time,
            drug_time,
            baseline_start,
            baseline_end,
        )
        data_export[cluster_id] = relative_spikes_ms
        baseline_fr_dict[cluster_id] = baseline_fr

    return data_export, baseline_fr_dict


def process_cluster_data(
    df: pd.DataFrame,
    cluster_id: int,
    sample_rate: int,
    start_time: float,
    end_time: float,
    drug_time: float | None,
    baseline_start: float | None,
    baseline_end: float | None,
) -> tuple[np.ndarray, float | None]:
    """
    Processes spike data for a single cluster, including filtering and baseline computation.

    Parameters:
        df (pd.DataFrame): DataFrame containing spike times and cluster IDs.
        cluster_id (int): The cluster ID to process.
        sample_rate (int): Sampling rate in Hz.
        start_time (float): Start time for analysis (seconds).
        end_time (float): End time for analysis (seconds).
        drug_time (float | None): Time of drug application (seconds), or None if not applicable.
        baseline_start (float | None): Start time of the baseline window (seconds), or None if no baseline.
        baseline_end (float | None): End time of the baseline window (seconds), or None if no baseline.

    Returns:
        tuple:
            - np.ndarray: Filtered and shifted spike times (ms).
            - float | None: Baseline firing rate (spikes/s), or None if not computed.
    """
    cluster_df = df.loc[df["spike_clusters"] == cluster_id]
    spikes_in_seconds = cluster_df["spike_times"].values / sample_rate

    # Compute baseline firing rate if applicable
    baseline_fr = None
    if baseline_start is not None and baseline_end is not None:
        baseline_fr = compute_baseline_firing_rate(
            spikes_in_seconds, baseline_start, baseline_end
        )

    # Filter spikes for the analysis window and shift relative to drug time
    filtered_spikes = spikes_in_seconds[
        (spikes_in_seconds >= start_time) & (spikes_in_seconds <= end_time)
    ]
    relative_spikes_ms = shift_spike_times(filtered_spikes, drug_time)

    return relative_spikes_ms, baseline_fr


def compute_baseline_firing_rate(
    spikes: np.ndarray, baseline_start: float, baseline_end: float
) -> float:
    """
    Computes the baseline firing rate for a cluster.

    Parameters:
        spikes (np.ndarray): Array of spike times in seconds.
        baseline_start (float): Start time of the baseline window (seconds).
        baseline_end (float): End time of the baseline window (seconds).

    Returns:
        float: Baseline firing rate (spikes/s).
    """
    baseline_spikes = spikes[(spikes >= baseline_start) & (spikes <= baseline_end)]
    baseline_duration = baseline_end - baseline_start
    return len(baseline_spikes) / baseline_duration if baseline_duration > 0 else 0.0


def shift_spike_times(spikes: np.ndarray, drug_time: float | None) -> np.ndarray:
    """
    Shifts spike times relative to the drug application time.

    Parameters:
        spikes (np.ndarray): Array of spike times in seconds.
        drug_time (float | None): Time of drug application in seconds, or None if not applicable.

    Returns:
        np.ndarray: Shifted spike times in milliseconds.
    """
    if drug_time is not None:
        spikes = spikes - drug_time
    return spikes[spikes >= 0] * 1000  # Convert to milliseconds


def calculate_firing_rate(
    data_export: dict[int, np.ndarray],
    bin_size: float,
    start_time: float,
    end_time: float,
    baseline_fr_dict: dict[int, float | None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """
    Calculates raw and delta firing rates for each cluster using the specified bin size.

    Parameters:
        data_export (dict[int, np.ndarray]): Dictionary where keys are cluster IDs and
                                             values are arrays of spike times in milliseconds.
        bin_size (float): Bin size for firing rate calculation (in seconds).
        start_time (float): Start time for the analysis window (in seconds).
        end_time (float): Maximum time for the analysis window (in seconds).
        baseline_fr_dict (dict[int, float | None] | None, optional): Dictionary mapping cluster IDs
                                                                     to baseline firing rates (spikes/s).

    Returns:
        tuple:
            - dict[str, np.ndarray]: Raw firing rates dictionary, including bin times and cluster rates.
            - dict[str, np.ndarray] | None: Delta firing rates dictionary, or None if no baseline provided.
    """
    bins = np.arange(start_time, end_time + bin_size, bin_size)
    if bins[-1] > end_time:
        bins = bins[:-1]

    raw_data = {"Time Intervals (s)": bins[:-1]}
    delta_data = {} if baseline_fr_dict else None

    for channel, spikes in data_export.items():
        # Convert spike times from ms to seconds
        spike_times_sec = spikes / 1000.0

        # Calculate raw firing rates
        counts, _ = np.histogram(spike_times_sec, bins=bins)
        raw_data[f"""Cluster_{channel}"""] = counts / bin_size

        # Calculate delta firing rates if baseline is provided
        if delta_data is not None:  # Ensure delta_data is a dictionary
            baseline_fr = baseline_fr_dict.get(channel)
            if baseline_fr is not None:
                delta_data[f"""Cluster_{channel}"""] = (
                    raw_data[f"""Cluster_{channel}"""] - baseline_fr
                )

    return raw_data, delta_data


def sort_cluster_columns(cluster_columns: list[str]) -> list[str]:
    """
    Sorts cluster columns numerically based on the cluster ID.

    Parameters:
        cluster_columns (list[str]): List of cluster column names.

    Returns:
        list[str]: Sorted cluster column names.
    """
    # Sort cluster columns numerically based on raw_data
    sorted_cluster_columns = sorted(
        [col for col in cluster_columns if col.startswith("Cluster_")],
        key=lambda x: int(x.split("_")[1]),
    )
    return sorted_cluster_columns


def create_firing_rate_dataframes(
    raw_data: dict[str, np.ndarray], delta_data: dict[str, np.ndarray] | None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Converts raw and delta firing rate dictionaries into sorted DataFrames.

    Parameters:
        raw_data (dict[str, np.ndarray]): Raw firing rates dictionary, including bin times and cluster rates.
        delta_data (dict[str, np.ndarray] | None): Delta firing rates dictionary, or None.

    Returns:
        tuple:
            - pd.DataFrame: Sorted DataFrame of raw firing rates.
            - pd.DataFrame | None: Sorted DataFrame of delta firing rates, or None.
    """
    sorted_cluster_columns = sort_cluster_columns(raw_data)
    sorted_columns = ["Time Intervals (s)"] + sorted_cluster_columns

    # Create raw DataFrame
    raw_df = pd.DataFrame(raw_data)[sorted_columns]

    # Create delta DataFrame only with matching columns
    delta_df = None
    if delta_data:
        # Ensure "Time Intervals (s)" exists in delta_data
        delta_data["Time Intervals (s)"] = raw_data["Time Intervals (s)"]

        delta_sorted_columns = ["Time Intervals (s)"] + [
            col for col in sorted_cluster_columns if col in delta_data
        ]
        delta_df = pd.DataFrame(delta_data)[delta_sorted_columns]

    return raw_df, delta_df


def calculate_isi_histogram(
    data_export: dict[int, np.ndarray],
    baseline_start: float | None = None,
    baseline_end: float | None = None,
    time_bin: float = 0.01,
    max_isi_time: float = 0.75,
) -> pd.DataFrame:
    """
    Calculates interspike interval (ISI) histograms for each cluster.

    Parameters:
        data_export (dict[int, np.ndarray]): Dictionary where keys are cluster IDs (integers) and
                                            values are arrays of spike times (in milliseconds).
        baseline_start (float, optional): Start time (in seconds) for the baseline period. If None, no baseline is considered.
        baseline_end (float, optional): End time (in seconds) for the baseline period. If None, no baseline is considered.
        time_bin (float, optional): Bin width for the ISI histogram (in seconds). Default is 0.01s.
        max_isi_time (float, optional): Maximum ISI (in seconds) to include in the histogram. Default is 0.75s.

    Returns:
        tuple:
            - pd.DataFrame: A DataFrame containing:
                - 'Bin_Starts': The left edges of each ISI bin (from 0 to max_isi_time).
                - One column per cluster: Counts of ISIs falling into each bin.
            - pd.DataFrame | None: If baseline is provided, a DataFrame containing baseline ISI histograms. Otherwise, None.
    """
    baseline_data = {} if baseline_start is not None else None

    n_bins = int(max_isi_time / time_bin)
    bin_edges = np.arange(0, (n_bins + 1) * time_bin, time_bin)
    isi_data = {"Bin_Starts": bin_edges[:-1]}

    for channel, spikes in data_export.items():
        # Compute ISIs for the entire time period
        spikes_in_seconds = spikes / 1000.0
        isis = np.diff(spikes_in_seconds)
        isi_histogram, _ = np.histogram(isis, bins=bin_edges)
        isi_data[channel] = isi_histogram

        # If baseline is provided, calculate ISI for the baseline period
        if baseline_data is not None:
            baseline_spikes = spikes_in_seconds[
                (spikes_in_seconds >= baseline_start)
                & (spikes_in_seconds <= baseline_end)
            ]
            baseline_isis = np.diff(baseline_spikes)
            baseline_histogram, _ = np.histogram(baseline_isis, bins=bin_edges)
            baseline_data[channel] = baseline_histogram

    # Formatting the DataFrame
    formatted_columns = {"Bin_Starts": isi_data["Bin_Starts"]}

    for key, value in isi_data.items():
        if key != "Bin_Starts":
            formatted_columns[f"""Cluster_{key}"""] = value

    baseline_isi_df = None
    if baseline_data:
        baseline_data["Bin_Starts"] = bin_edges[:-1]
        baseline_columns = {"Bin_Starts": baseline_data["Bin_Starts"]}
        for key, value in baseline_data.items():
            if key != "Bin_Starts":
                baseline_columns[f"""Cluster_{key}_Baseline"""] = value
        baseline_isi_df = pd.DataFrame(baseline_columns)

    return pd.DataFrame(formatted_columns), baseline_isi_df


def export_hazard_excel(
    export_dir: str,
    hazard_df: pd.DataFrame,
    hazard_summary_df: pd.DataFrame,
    isi_df: pd.DataFrame,
    baseline_isi_df: pd.DataFrame | None = None,
    baseline_hazard_df: pd.DataFrame | None = None,
    baseline_hazard_summary_df: pd.DataFrame | None = None,
) -> None:
    """
    Exports hazard and ISI analysis data to an Excel file with multiple sheets.

    The exported Excel file contains:
        - "ISI_Histogram": Interspike interval histogram data.
        - "Hazard_Function": Hazard function values over time bins for each channel.
        - "Hazard_Summary": Summary metrics of the hazard function
            (e.g., peak early hazard, mean late, ratio).

    Parameters:
        export_dir (str): Directory where the Excel file will be saved.
        hazard_df (pd.DataFrame): DataFrame containing hazard function values for each channel.
        hazard_summary_df (pd.DataFrame): DataFrame with summary metrics of the hazard function.
        isi_df (pd.DataFrame): DataFrame containing ISI histograms.
        baseline_df (pd.DataFrame): DataFrame containing baseline hazard function values.

    Returns:
        None: The function writes the Excel file to `export_dir` and prints a confirmation message.
    """
    excel_path = os.path.join(export_dir, "isi_and_hazard_analysis.xlsx")

    with ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        isi_df.to_excel(writer, sheet_name="ISI_Histogram", index=False)
        hazard_df.to_excel(writer, sheet_name="Hazard_Function", index=False)
        hazard_summary_df.to_excel(writer, sheet_name="Hazard_Summary", index=False)

        # Write baseline data if available
        if baseline_isi_df is not None:
            baseline_isi_df.to_excel(
                writer, sheet_name="Baseline_ISI_Histogram", index=False
            )
        if baseline_hazard_df is not None:
            baseline_hazard_df.to_excel(
                writer, sheet_name="Baseline_Hazard_Function", index=False
            )
        if baseline_hazard_summary_df is not None:
            baseline_hazard_summary_df.to_excel(
                writer, sheet_name="Baseline_Hazard_Summary", index=False
            )

    print(f"""ISI and hazard data exported to {excel_path}""")


def calculate_hazard_function(
    isi_df: pd.DataFrame,
    baseline_isi_df: pd.DataFrame | None = None,
    early_time: float = 0.07,
    late_time_start: float = 0.41,
    late_time_end: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Calculates hazard function values for each channel and key hazard metrics.

    Parameters:
        isi_df (pd.DataFrame): DataFrame containing ISI histogram data. Should have:
            - 'Bin_Starts': Left edges of ISI bins.
            - One column per channel with ISI histogram counts.
        baseline_isi_df (pd.DataFrame | None): Optional DataFrame for baseline ISI histogram data.
        early_time (float): Upper threshold (in seconds) for the "early" hazard region.
        late_time_start (float): Start of the "late" hazard region (in seconds).
        late_time_end (float): End of the "late" hazard region (in seconds).

    Returns:
        tuple:
            - pd.DataFrame: Hazard function values with 'Bin_Starts' and one column per channel.
            - pd.DataFrame: Hazard summary metrics for each channel.
            - pd.DataFrame | None: Baseline hazard function values, or None if no baseline provided.
            - pd.DataFrame | None: Baseline hazard summary metrics, or None if no baseline provided.
    """
    # Calculate hazard function for the main ISI data
    isi_bin_starts = isi_df["Bin_Starts"].values
    hazard_df = compute_hazard_values(isi_df, isi_bin_starts)

    # Compute summary metrics for main data
    hazard_summary_df = compute_hazard_summary(
        hazard_df, isi_bin_starts, early_time, late_time_start, late_time_end
    )

    # If baseline ISI data is provided, calculate baseline hazard functions and metrics
    if baseline_isi_df is not None:
        baseline_isi_bin_starts = baseline_isi_df["Bin_Starts"].values
        baseline_hazard_df = compute_hazard_values(
            baseline_isi_df, baseline_isi_bin_starts
        )
        baseline_hazard_summary_df = compute_hazard_summary(
            baseline_hazard_df,
            baseline_isi_bin_starts,
            early_time,
            late_time_start,
            late_time_end,
        )
        return (
            hazard_df,
            hazard_summary_df,
            baseline_hazard_df,
            baseline_hazard_summary_df,
        )

    # If no baseline ISI data, return None for baseline data
    return hazard_df, hazard_summary_df, None, None


def compute_hazard_values(isi_df: pd.DataFrame, bin_starts: np.ndarray) -> pd.DataFrame:
    """
    Compute hazard values for each channel.

    Parameters:
        isi_df (pd.DataFrame): DataFrame containing ISI histogram data.
        bin_starts (np.ndarray): Left edges of ISI bins.

    Returns:
        pd.DataFrame: DataFrame containing hazard values for each channel.
    """
    hazard_data = {"Bin_Starts": bin_starts}

    for channel in isi_df.columns[1:]:
        isi_counts = isi_df[channel].values
        total_spikes = np.sum(isi_counts)

        # Compute hazard values
        cumsum_isi = np.cumsum(isi_counts)
        hazard_values = np.divide(
            isi_counts,
            np.maximum(total_spikes - cumsum_isi + isi_counts, 1),
            where=(cumsum_isi < total_spikes),
            out=np.zeros_like(isi_counts, dtype=float),
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
    """Compute hazard summary metrics for each channel.

    Parameters:
        hazard_df (pd.DataFrame): DataFrame containing hazard values for each channel.
        bin_starts (np.ndarray): Left edges of ISI bins.
        early_time (float): Upper threshold (in seconds) for the "early" hazard region.
        late_time_start (float): Start of the "late" hazard region (in seconds).
        late_time_end (float): End of the "late" hazard region (in seconds).

    Returns:
        pd.DataFrame: DataFrame containing summary metrics for each channel.
    """
    summary_data = []

    for channel in hazard_df.columns[1:]:
        hazard_values = hazard_df[channel].values

        # Early hazard metrics
        early_mask = bin_starts <= early_time
        peak_early_hazard = hazard_values[early_mask].max() if early_mask.any() else 0

        # Late hazard metrics
        late_mask = (bin_starts >= late_time_start) & (bin_starts <= late_time_end)
        mean_late_hazard = hazard_values[late_mask].mean() if late_mask.any() else 0

        # Hazard ratio
        hazard_ratio = (
            peak_early_hazard / mean_late_hazard if mean_late_hazard > 0 else np.nan
        )

        # Append summary data
        summary_data.append(
            {
                "Cluster": channel,
                "Peak Early Hazard": peak_early_hazard,
                "Mean Late Hazard": mean_late_hazard,
                "Hazard Ratio": hazard_ratio,
            }
        )

    return pd.DataFrame(summary_data)


def get_user_input(
    file_paths: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list]]:
    """
    Handles user input for filtering data based on channels or labels.
    Loads raw spike data and creates a label lookup array.

    Parameters:
        file_paths (dict[str, str]): Paths to required input files.

    Returns:
        tuple: Contains spike times, spike clusters, group labels array,
        and user filtering selections.
    """
    spike_times_path = file_paths["spike_times.npy"]
    spike_clusters_path = file_paths["spike_clusters.npy"]
    group_labels_path = file_paths["cluster_group.tsv"]

    # Prompt user for channels/labels to include
    channels_to_include, labels_to_include = channels_or_labels_to_export()

    # Load raw spike data
    spike_times, spike_clusters = load_spike_data(spike_times_path, spike_clusters_path)
    group_labels_array = create_label_lookup(group_labels_path)

    return (
        spike_times,
        spike_clusters,
        group_labels_array,
        {
            "channels_to_include": channels_to_include,
            "labels_to_include": labels_to_include,
        },
    )


def process_filtered_data(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    group_labels_array: np.ndarray,
    user_filters: dict[str, list],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Filters data based on user selections and computes the maximum recording time.

    Parameters:
        spike_times (np.ndarray): Raw spike times.
        spike_clusters (np.ndarray): Raw spike cluster IDs.
        group_labels_array (np.ndarray): Group label mappings.
        user_filters (dict): User-selected filters for channels and labels.

    Returns:
        tuple: Filtered spike times, spike clusters, group labels, and maximum time.
    """
    filtered_spike_times, filtered_spike_clusters, filtered_labels = filter_data(
        spike_times,
        spike_clusters,
        group_labels_array,
        labels_to_include=user_filters["labels_to_include"],
        channels_to_include=user_filters["channels_to_include"],
    )

    max_time = spike_times[-1] / 30000 if len(spike_times) > 0 else 0

    print(f"""Recording length: {max_time:.2f}s""")

    return filtered_spike_times, filtered_spike_clusters, filtered_labels, max_time


def prepare_filtered_data(file_paths: dict) -> tuple[pd.DataFrame, float]:
    """
    Loads spike data, applies user-specified filters, and prepares the recording DataFrame.

    Parameters:
        file_paths (dict): Dictionary containing paths to required files
        (e.g., spike times, clusters).

    Returns:
        tuple:
            - pd.DataFrame: Filtered spike data containing spike times, clusters, and labels.
            - float: Maximum recording time in seconds.
    """
    spike_times, spike_clusters, group_labels_array, user_filters = get_user_input(
        file_paths
    )
    filtered_spike_times, filtered_spike_clusters, filtered_labels, max_time = (
        process_filtered_data(
            spike_times, spike_clusters, group_labels_array, user_filters
        )
    )
    recording_dataframe = construct_dataframe(
        filtered_spike_times, filtered_spike_clusters, filtered_labels
    )
    return recording_dataframe, max_time


def get_user_parameters(
    max_time: float,
) -> tuple[float | None, float, float, float, float | None, float | None]:
    """
    Collects user-defined parameters for data analysis, including drug application time,
    start and end times, bin size, and baseline period.

    Parameters:
        max_time (float): Maximum recording time in seconds.

    Returns:
        tuple:
            - float | None: Drug application time in seconds, or None if not specified.
            - float: Start time for analysis in seconds.
            - float: End time for analysis in seconds.
            - float: Bin size for firing rate calculation in seconds.
            - float | None: Start time of the baseline period, or None if not specified.
            - float | None: End time of the baseline period, or None if not specified.
    """
    drug_time = drug_application_time()
    start_time, end_time = start_and_end_time(max_time)
    bin_size = float(
        input(
            "Enter the bin size for firing rate calculation (s), or press Enter for 1s: "
        )
        or 1.0
    )
    if start_time is not None:
        min_time = start_time
    if end_time is not None:
        max_time = end_time

    baseline_start, baseline_end = prompt_for_baseline(max_time, min_time)
    print(
        f"""Analyzing data from {start_time}s to {end_time}s with bin size {bin_size}s"""
    )
    if drug_time:
        print(f"""Drug application time: {drug_time:.2f}s""")
    return drug_time, start_time, end_time, bin_size, baseline_start, baseline_end


def highlight_baseline_data(export_dir: str) -> str:
    pass


def main():
    """
    Orchestrates the complete data processing and analysis pipeline.

    Steps:
        - Select and validate the folder containing required files.
        - Load and filter data based on user input.
        - Collect user-defined parameters for analysis.
        - Perform data extraction, ISI, and hazard analysis.
        - Export results to files.
    """
    # Choose folder and validate files
    file_paths = choose_and_validate_folder()

    # Load and filter data
    recording_dataframe, max_time = prepare_filtered_data(file_paths)

    # Collect user-defined parameters
    drug_time, start_time, end_time, bin_size, baseline_start, baseline_end = (
        get_user_parameters(max_time)
    )

    # Extract data for analysis
    raw_fr_dict, baseline_fr_dict = extract_data(
        recording_dataframe,
        drug_time,
        start_time,
        end_time,
        30000,  # sample_rate
        baseline_start,
        baseline_end,
    )

    # Export spike times and firing rates
    export_dir = export_data(
        raw_fr_dict,
        baseline_fr_dict,
        os.path.dirname(file_paths["spike_times.npy"]),
        bin_size,
        start_time,
        end_time,
        drug_time,
        baseline_start,
        baseline_end,
    )

    if baseline_start is not None:
        formated_export_dir = highlight_baseline_data(export_dir)

    # Perform ISI and hazard analysis
    isi_df, baseline_isi_df = calculate_isi_histogram(
        raw_fr_dict, baseline_start, baseline_end
    )

    hazard_df, hazard_summary_df, baseline_hazard_df, baseline_hazard_summary_df = (
        calculate_hazard_function(isi_df, baseline_isi_df)
    )

    # Export hazard and ISI analysis, including baseline data if available
    export_hazard_excel(
        export_dir,
        hazard_df,
        hazard_summary_df,
        isi_df,
        baseline_isi_df,
        baseline_hazard_df,
        baseline_hazard_summary_df,
    )


if __name__ == "__main__":
    main()

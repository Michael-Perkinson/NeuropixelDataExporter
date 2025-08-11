import numpy as np
import pandas as pd
from core.file_manager import load_spike_data, create_label_lookup
from core.terminal_prompts import channels_or_labels_to_export


def filter_by_labels(
    spike_clusters: np.ndarray,
    group_labels_array: np.ndarray,
    labels_to_include: list[str],
) -> np.ndarray:
    """
    Generate a mask selecting spikes from clusters with specified group labels.

    Args
        spike_clusters: Array of spike cluster IDs.
        group_labels_array: Lookup array mapping cluster IDs to group labels.
        labels_to_include: List of labels to include.

    Returns
        Boolean mask for the selected spikes.
    """
    valid_cluster_ids = np.where(np.isin(group_labels_array, labels_to_include))[0]
    print(f"Filtered cluster IDs for labels {labels_to_include}: {valid_cluster_ids}")
    return np.isin(spike_clusters, valid_cluster_ids)


def filter_by_channels(
    spike_clusters: np.ndarray, channels_to_include: list[int]
) -> np.ndarray:
    """
    Generate a mask selecting spikes from specified channels.

    Args
        spike_clusters: Array of spike cluster IDs.
        channels_to_include: List of channel IDs to include.

    Returns
        Boolean mask for the selected spikes.
    """
    return np.isin(spike_clusters, channels_to_include)


def filter_data(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    group_labels_array: np.ndarray,
    labels_to_include: list[str] | None = None,
    channels_to_include: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply channel and/or label filtering to spike data.

    Args
        spike_times: Array of spike times.
        spike_clusters: Array of spike cluster IDs.
        group_labels_array: Array mapping cluster IDs to group labels.
        labels_to_include: Optional list of group labels to filter by.
        channels_to_include: Optional list of channel IDs to filter by.

    Returns
        Filtered spike times, spike cluster IDs, and group labels.
    """
    mask = np.zeros_like(spike_clusters, dtype=bool)

    if labels_to_include:
        mask |= filter_by_labels(spike_clusters, group_labels_array, labels_to_include)

    if channels_to_include:
        mask |= filter_by_channels(spike_clusters, channels_to_include)

    filtered_spike_times = spike_times[mask]
    filtered_spike_clusters = spike_clusters[mask]
    filtered_group_labels = group_labels_array[spike_clusters[mask]]

    return filtered_spike_times, filtered_spike_clusters, filtered_group_labels


def construct_dataframe(
    spike_times: np.ndarray, spike_clusters: np.ndarray, group_labels: np.ndarray
) -> pd.DataFrame:
    """
    Construct a DataFrame from spike times, cluster IDs, and group labels.

    Args
        spike_times: Array of spike times.
        spike_clusters: Array of spike cluster IDs.
        group_labels: Array of group labels corresponding to each spike.

    Returns
        DataFrame containing the spike data.
    """
    return pd.DataFrame(
        {
            "spike_times": spike_times,
            "spike_clusters": spike_clusters,
            "group": group_labels,
        }
    )


def get_all_data_from_files(
    file_paths: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw spike data and labels.

    Args
        file_paths: Dictionary mapping file names to their paths.

    Returns
        Tuple containing spike times, spike cluster IDs, and group labels.
    """
    spike_times, spike_clusters = load_spike_data(
        file_paths["spike_times.npy"], file_paths["spike_clusters.npy"]
    )
    group_labels_array = create_label_lookup(file_paths["cluster_group.tsv"])

    return spike_times, spike_clusters, group_labels_array


def process_filtered_data(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    group_labels_array: np.ndarray,
    user_filters: dict[str, list],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply filtering based on user-selected channels and labels.

    Args
        spike_times: Array of spike times.
        spike_clusters: Array of spike cluster IDs.
        group_labels_array: Array mapping cluster IDs to group labels.
        user_filters: Dictionary with filtering criteria:
            - "labels_to_include" (list of labels)
            - "channels_to_include" (list of channels)

    Returns
        Filtered spike times, spike cluster IDs, and group labels.
    """
    filtered_spike_times, filtered_spike_clusters, filtered_labels = filter_data(
        spike_times,
        spike_clusters,
        group_labels_array,
        labels_to_include=user_filters.get("labels_to_include", []),
        channels_to_include=user_filters.get("channels_to_include", []),
    )

    return filtered_spike_times, filtered_spike_clusters, filtered_labels


def calc_max_time(spike_times: np.ndarray) -> float:
    """
    Calculate the maximum recording time in seconds.

    Args
        spike_times: Array of spike times.

    Returns
        Maximum recording time in seconds.
    """
    return spike_times[-1] / 30000 if len(spike_times) > 0 else 0


def prepare_filtered_data(file_paths: dict[str, str]) -> tuple[pd.DataFrame, float]:
    """
    Load, filter, and construct a DataFrame of spike data.

    - Loads spike data and cluster labels.
    - Filters based on user-specified channels or labels.
    - Constructs a DataFrame of filtered data.

    Args
        file_paths: Dictionary mapping required file names to their paths.

    Returns
        A tuple containing:
        - DataFrame of filtered spike data.
        - Maximum recording time (seconds) based on filtered data.
    """
    spike_times, spike_clusters, group_labels_array = get_all_data_from_files(
        file_paths
    )

    user_filters = channels_or_labels_to_export()

    filtered_spike_times, filtered_spike_clusters, filtered_labels = (
        process_filtered_data(
            spike_times, spike_clusters, group_labels_array, user_filters
        )
    )

    max_time = calc_max_time(filtered_spike_times)

    recording_dataframe = construct_dataframe(
        filtered_spike_times, filtered_spike_clusters, filtered_labels
    )

    return recording_dataframe, max_time

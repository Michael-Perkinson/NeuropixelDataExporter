from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.core.file_manager import KS_LABEL_FILES, create_label_lookup, load_spike_data


def filter_by_labels(
    spike_clusters: NDArray[np.int64],
    group_labels_array: NDArray[np.object_],
    labels_to_include: list[str],
) -> NDArray[np.bool_]:
    """
    Boolean mask selecting spikes whose cluster IDs belong to clusters with the given labels.
    """
    valid_cluster_ids = np.where(
        np.isin(group_labels_array, labels_to_include))[0]
    return np.isin(spike_clusters, valid_cluster_ids)


def filter_by_channels(
    spike_clusters: NDArray[np.int64],
    channels_to_include: list[int],
) -> NDArray[np.bool_]:
    """
    Boolean mask selecting spikes whose cluster IDs are in channels_to_include.
    """
    return np.isin(spike_clusters, channels_to_include)


def filter_data(
    spike_times: NDArray[np.float64],
    spike_clusters: NDArray[np.int64],
    group_labels_array: NDArray[np.object_],
    labels_to_include: list[str] | None = None,
    channels_to_include: list[int] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.object_]]:
    """
    Apply channel and/or label filtering to spike data.
    """
    mask: NDArray[np.bool_] = np.zeros_like(spike_clusters, dtype=np.bool_)

    if labels_to_include:
        mask |= filter_by_labels(
            spike_clusters, group_labels_array, labels_to_include)

    if channels_to_include:
        mask |= filter_by_channels(spike_clusters, channels_to_include)

    filtered_spike_times = spike_times[mask]
    filtered_spike_clusters = spike_clusters[mask]
    # group_labels_array maps cluster_id -> label, so index by cluster IDs for selected spikes
    filtered_group_labels = group_labels_array[spike_clusters[mask]]

    return filtered_spike_times, filtered_spike_clusters, filtered_group_labels


def construct_dataframe(
    spike_times: NDArray[np.float64],
    spike_clusters: NDArray[np.int64],
    group_labels: NDArray[np.object_],
) -> pd.DataFrame:
    """
    Construct a DataFrame from spike times, cluster IDs, and group labels.
    """
    return pd.DataFrame(
        {
            "spike_times": spike_times,
            "spike_clusters": spike_clusters,
            "group": group_labels,
        }
    )


def get_all_data_from_files(
    file_paths: dict[str, Path],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.object_]]:
    """
    Load raw spike data and labels from the validated Kilosort folder.

    Expects keys:
      - "spike_times.npy"
      - "spike_clusters.npy"
      - one of KS_LABEL_FILES
    """
    spike_times, spike_clusters = load_spike_data(
        file_paths["spike_times.npy"], file_paths["spike_clusters.npy"]
    )

    label_path: Path | None = None
    for fname in KS_LABEL_FILES:
        if fname in file_paths:
            label_path = file_paths[fname]
            break
    if label_path is None:
        raise FileNotFoundError("No label file path found in file_paths.")

    group_labels_array = create_label_lookup(label_path)
    return spike_times, spike_clusters, group_labels_array


def process_filtered_data(
    spike_times: NDArray[np.float64],
    spike_clusters: NDArray[np.int64],
    group_labels_array: NDArray[np.object_],
    user_filters: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.object_]]:
    """
    Apply filtering based on user-selected channels and labels.

    user_filters should include:
      - "labels_to_include": list[str]
      - "channels_to_include": list[int]
    """
    labels_to_include = user_filters.get("labels_to_include", [])
    channels_to_include = user_filters.get("channels_to_include", [])

    filtered_spike_times, filtered_spike_clusters, filtered_labels = filter_data(
        spike_times,
        spike_clusters,
        group_labels_array,
        labels_to_include=labels_to_include,
        channels_to_include=channels_to_include,
    )

    return filtered_spike_times, filtered_spike_clusters, filtered_labels


def calc_max_time(spike_times: NDArray[np.float64]) -> float:
    """
    Calculate the maximum recording time in seconds.
    Assumes spike_times are in samples at 30 kHz (Kilosort convention).
    """
    return float(spike_times[-1] / 30000.0) if spike_times.size > 0 else 0.0


def resolve_labels_to_cluster_ids(
    group_labels_array: NDArray[np.object_],
    labels: list[str],
) -> list[int]:
    """
    Return sorted cluster IDs whose label is in the given labels list.
    """
    return sorted(int(i) for i in np.where(np.isin(group_labels_array, labels))[0])


def prepare_filtered_data(file_paths: dict[str, Path]) -> tuple[pd.DataFrame, float]:
    """
    Load all spike data and return as a DataFrame with max recording time.
    No filtering is applied here — the GUI selects clusters downstream.

    Returns:
      - DataFrame of all spike data (spike_times, spike_clusters, group)
      - Maximum recording time (seconds)
    """
    spike_times, spike_clusters, group_labels_array = get_all_data_from_files(
        file_paths)

    max_time = calc_max_time(spike_times)
    recording_dataframe = construct_dataframe(
        spike_times, spike_clusters, group_labels_array[spike_clusters]
    )

    return recording_dataframe, max_time

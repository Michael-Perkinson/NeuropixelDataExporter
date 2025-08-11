import numpy as np
import pandas as pd

from core.spike_filter import (
    filter_by_labels,
    filter_by_channels,
    filter_data,
    construct_dataframe,
    get_all_data_from_files,
    process_filtered_data,
    prepare_filtered_data,
)


def test_filter_by_labels():
    spike_clusters = np.array([0, 1, 2, 3, 4])
    group_labels_array = np.array(["A", "B", "A", "C", "B"])
    labels_to_include = ["A"]
    mask = filter_by_labels(spike_clusters, group_labels_array, labels_to_include)
    # Expected: only clusters with group "A", which are at indices 0 and 2.
    expected_mask = np.isin(spike_clusters, [0, 2])
    np.testing.assert_array_equal(mask, expected_mask)


def test_filter_by_channels():
    spike_clusters = np.array([0, 1, 2, 3, 4])
    channels_to_include = [1, 3]
    mask = filter_by_channels(spike_clusters, channels_to_include)
    expected_mask = np.isin(spike_clusters, channels_to_include)
    np.testing.assert_array_equal(mask, expected_mask)


def test_filter_data():
    spike_times = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    spike_clusters = np.array([0, 1, 2, 3, 4])
    group_labels_array = np.array(["A", "B", "A", "C", "B"])
    # Filter by labels "A"
    filtered_times, filtered_clusters, filtered_labels = filter_data(
        spike_times, spike_clusters, group_labels_array, labels_to_include=["A"]
    )
    # Only indices with group "A" should remain: indices 0 and 2.
    expected_times = np.array([0.1, 0.3])
    expected_clusters = np.array([0, 2])
    expected_labels = np.array(["A", "A"])
    np.testing.assert_array_equal(filtered_times, expected_times)
    np.testing.assert_array_equal(filtered_clusters, expected_clusters)
    np.testing.assert_array_equal(filtered_labels, expected_labels)


def test_construct_dataframe():
    spike_times = np.array([0.1, 0.2])
    spike_clusters = np.array([0, 1])
    group_labels = np.array(["A", "B"])
    df = construct_dataframe(spike_times, spike_clusters, group_labels)
    expected_df = pd.DataFrame(
        {
            "spike_times": spike_times,
            "spike_clusters": spike_clusters,
            "group": group_labels,
        }
    )
    pd.testing.assert_frame_equal(df, expected_df)


def test_get_all_data_from_files(tmp_path):
    # Create dummy files in tmp_path
    required_files = ["spike_times.npy", "spike_clusters.npy", "cluster_group.tsv"]
    spike_times = np.array([0.1, 0.2])
    spike_clusters = np.array([0, 1])
    np.save(tmp_path / "spike_times.npy", spike_times)
    np.save(tmp_path / "spike_clusters.npy", spike_clusters)
    df = pd.DataFrame({"cluster_id": [0, 1], "group": ["A", "B"]})
    df.to_csv(tmp_path / "cluster_group.tsv", sep="\t", index=False)
    file_paths = {f: str(tmp_path / f) for f in required_files}

    # Call get_user_input
    times, clusters, group_array = get_all_data_from_files(file_paths)
    np.testing.assert_array_equal(times, spike_times)
    np.testing.assert_array_equal(clusters, spike_clusters)


def test_process_filtered_data():
    # Create synthetic data for testing.
    spike_times = np.array([0.1, 0.2, 0.3, 0.4])
    spike_clusters = np.array([0, 1, 0, 1])
    group_labels_array = np.array(["A", "B"])
    # Set user_filters to filter by channel 1 only.
    user_filters = {"channels_to_include": [1], "labels_to_include": []}
    filtered_times, filtered_clusters, filtered_labels = process_filtered_data(
        spike_times, spike_clusters, group_labels_array, user_filters
    )
    expected_times = np.array([0.2, 0.4])
    expected_clusters = np.array([1, 1])
    expected_labels = np.array(["B", "B"])
    np.testing.assert_array_equal(filtered_times, expected_times)
    np.testing.assert_array_equal(filtered_clusters, expected_clusters)
    np.testing.assert_array_equal(filtered_labels, expected_labels)


def test_prepare_filtered_data(monkeypatch, tmp_path):
    # Create dummy required files
    required_files = ["spike_times.npy", "spike_clusters.npy", "cluster_group.tsv"]
    spike_times = np.array([0.1, 0.2, 0.3])
    spike_clusters = np.array([0, 1, 0])
    np.save(tmp_path / "spike_times.npy", spike_times)
    np.save(tmp_path / "spike_clusters.npy", spike_clusters)
    df = pd.DataFrame({"cluster_id": [0, 1], "group": ["A", "B"]})
    df.to_csv(tmp_path / "cluster_group.tsv", sep="\t", index=False)
    file_paths = {f: str(tmp_path / f) for f in required_files}

    # Monkeypatch input so that channels_or_labels_to_export returns "0, A"
    monkeypatch.setattr("builtins.input", lambda prompt="": "0, A")
    df_out, max_time = prepare_filtered_data(file_paths)
    # Verify DataFrame columns.
    assert set(df_out.columns) == {"spike_times", "spike_clusters", "group"}

    expected_max_time = spike_times[-1] / 30000
    assert max_time == expected_max_time

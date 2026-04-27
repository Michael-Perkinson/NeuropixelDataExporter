import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.core.file_manager import (
    validate_ks_folder,
    find_specific_files_in_folder,
    load_spike_data,
    create_label_lookup,
    get_recording_duration,
    make_output_folders,
    make_specific_folder,
    KS_REQUIRED,
    KS_LABEL_FILES,
    SAMPLE_RATE,
)

# --- Test for find_specific_files_in_folder --- #
def test_find_specific_files_in_folder(tmp_path):
    # Create dummy files in the temporary directory
    always_files = list(KS_REQUIRED)
    one_of_files = list(KS_LABEL_FILES)
    for file in always_files + [one_of_files[0]]:
        (tmp_path / file).write_text("dummy content")

    file_paths = find_specific_files_in_folder(tmp_path, always_files, one_of_files)

    # Check that required files and one label file are found
    assert set(always_files) <= set(file_paths.keys())
    assert any(f in file_paths for f in one_of_files)


# --- Test for load_spike_data --- #
def test_load_spike_data(tmp_path):
    # Create dummy numpy arrays
    spike_times = np.array([0.1, 0.2, 0.3])
    spike_clusters = np.array([1, 2, 3])

    # Save them to temporary npy files
    spike_times_path = tmp_path / "spike_times.npy"
    spike_clusters_path = tmp_path / "spike_clusters.npy"
    np.save(spike_times_path, spike_times)
    np.save(spike_clusters_path, spike_clusters)

    # Load the data using our function
    loaded_times, loaded_clusters = load_spike_data(
        spike_times_path, spike_clusters_path
    )

    # Use numpy testing to compare arrays
    np.testing.assert_array_equal(loaded_times, spike_times)
    np.testing.assert_array_equal(loaded_clusters, spike_clusters)


# --- Test for create_label_lookup --- #
def test_create_label_lookup_group(tmp_path):
    # Create a dummy TSV file with cluster labels
    tsv_file = tmp_path / "cluster_group.tsv"
    data = {"cluster_id": [0, 1, 2], "group": ["A", "B", "C"]}
    df = pd.DataFrame(data)
    df.to_csv(tsv_file, sep="\t", index=False)

    # Create lookup array using our function
    lookup, _ = create_label_lookup(tsv_file)

    # Check that the lookup returns the expected group labels
    assert lookup[0] == "A"
    assert lookup[1] == "B"
    assert lookup[2] == "C"


def test_create_label_lookup_kslabel(tmp_path):
    # Create a dummy TSV file with cluster labels
    tsv_file = tmp_path / "cluster_group.tsv"
    data = {"cluster_id": [0, 1, 2], "KSLabel": ["good", "mua", "noise"]}
    df = pd.DataFrame(data)
    df.to_csv(tsv_file, sep="\t", index=False)

    # Create lookup array using our function
    lookup, _ = create_label_lookup(tsv_file)

    # Check that the lookup returns the expected group labels
    assert lookup[0] == "good"
    assert lookup[1] == "mua"
    assert lookup[2] == "noise"


def test_create_label_lookup_random_str(tmp_path):
    # No recognised label column → all clusters get "unknown"
    tsv_file = tmp_path / "cluster_group.tsv"
    data = {"cluster_id": [0, 1, 2], "random": ["A", "B", "C"]}
    df = pd.DataFrame(data)
    df.to_csv(tsv_file, sep="\t", index=False)

    lookup, _ = create_label_lookup(tsv_file)
    assert all(lookup[i] == "unknown" for i in range(3))


# --- Tests for validate_ks_folder --- #
def test_choose_and_validate_ks_folder_success(tmp_path):
    # Create the always-required files plus one label file.
    always_required = list(KS_REQUIRED)
    one_of_required = list(KS_LABEL_FILES)
    for file in always_required + [one_of_required[0]]:
        (tmp_path / file).write_text("dummy content")

    file_paths = validate_ks_folder(tmp_path, always_required, one_of_required)
    assert set(always_required) <= set(file_paths.keys())
    assert any(f in file_paths for f in one_of_required)


def test_choose_and_validate_ks_folder_cancel():
    # When a required file is missing, validate_ks_folder should raise FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        validate_ks_folder(Path("invalid_path"), list(KS_REQUIRED), list(KS_LABEL_FILES))


def test_make_specific_folder(tmp_path):
    # Create a specific folder within the temporary directory
    folder_name = "specific_folder"
    specific_folder = make_specific_folder(tmp_path, folder_name)

    # Check that the specific folder exists and is located inside tmp_path
    assert specific_folder.is_dir()
    assert specific_folder.parent == tmp_path
    assert specific_folder.name == folder_name


def test_create_label_lookup_missing_cluster_id(tmp_path):
    tsv_file = tmp_path / "cluster_group.tsv"
    pd.DataFrame({"wrong_col": [0, 1]}).to_csv(tsv_file, sep="\t", index=False)
    with pytest.raises(ValueError, match="cluster_id"):
        create_label_lookup(tsv_file)


def test_create_label_lookup_neurontype(tmp_path):
    tsv_file = tmp_path / "cluster_info.tsv"
    data = {"cluster_id": [0, 1], "group": ["good", "mua"], "neurontype": ["PMNC", ""]}
    pd.DataFrame(data).to_csv(tsv_file, sep="\t", index=False)
    lookup, log = create_label_lookup(tsv_file)
    assert lookup[0] == "PMNC"
    assert lookup[1] == "mua"
    assert any("neurontype" in msg for msg in log)


def test_validate_ks_folder_missing_label(tmp_path):
    for f in KS_REQUIRED:
        (tmp_path / f).write_text("x")
    with pytest.raises(FileNotFoundError, match="Missing label"):
        validate_ks_folder(tmp_path, list(KS_REQUIRED), list(KS_LABEL_FILES))


def test_get_recording_duration(tmp_path):
    samples = np.array([0, 15000, 30000, 90000])  # last = 3.0 s at 30 kHz
    path = tmp_path / "spike_times.npy"
    np.save(path, samples)
    assert get_recording_duration(path) == 3.0


def test_get_recording_duration_empty(tmp_path):
    path = tmp_path / "spike_times.npy"
    np.save(path, np.array([]))
    assert get_recording_duration(path) is None


def test_get_recording_duration_missing():
    assert get_recording_duration(Path("nonexistent.npy")) is None


def test_find_specific_files_returns_only_present(tmp_path):
    (tmp_path / "spike_times.npy").write_text("x")
    result = find_specific_files_in_folder(tmp_path, ["spike_times.npy"], ["cluster_group.tsv"])
    assert "spike_times.npy" in result
    assert "cluster_group.tsv" not in result


def test_make_output_folders(tmp_path):
    # Create output folders using the function
    analysis_dir, images_dir, txt_dir = make_output_folders(tmp_path)

    # Check that the directories are created and are inside tmp_path
    assert analysis_dir.is_dir()
    assert images_dir.is_dir()
    assert txt_dir.is_dir()
    assert analysis_dir.parent == tmp_path
    assert images_dir.parent == analysis_dir
    assert txt_dir.parent == analysis_dir
    assert images_dir.name == "firing_rate_images"
    assert txt_dir.name == "txt_files_for_clampfit_import"

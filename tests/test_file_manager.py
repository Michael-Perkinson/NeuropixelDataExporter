import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.core.file_manager import (
    validate_ks_folder,
    find_specific_files_in_folder,
    load_spike_data,
    create_label_lookup,
    make_output_folders,
    make_specific_folder,
)

# --- Test for find_specific_files_in_folder --- #
def test_find_specific_files_in_folder(tmp_path):
    # Create dummy files in the temporary directory
    required_files = ["spike_times.npy",
                      "spike_clusters.npy", "cluster_group.tsv"]
    for file in required_files:
        (tmp_path / file).write_text("dummy content")

    # Call the function with the temporary directory path
    file_paths = find_specific_files_in_folder(tmp_path, required_files)

    # Check if all required files are found and paths exist
    assert set(file_paths.keys()) == set(required_files)
    for file in required_files:
        assert (tmp_path / file).exists()


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
    lookup = create_label_lookup(tsv_file)

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
    lookup = create_label_lookup(tsv_file)

    # Check that the lookup returns the expected group labels
    assert lookup[0] == "good"
    assert lookup[1] == "mua"
    assert lookup[2] == "noise"


def test_create_label_lookup_random_str(tmp_path):
    # Create a dummy TSV file with cluster labels
    tsv_file = tmp_path / "cluster_group.tsv"
    data = {"cluster_id": [0, 1, 2], "random": ["A", "B", "C"]}
    df = pd.DataFrame(data)
    df.to_csv(tsv_file, sep="\t", index=False)

    # Create lookup array using our function
    with pytest.raises(ValueError):
        create_label_lookup(tsv_file)


# --- Tests for choose_and_validate_ks_folder --- #
def test_choose_and_validate_ks_folder_success(monkeypatch, tmp_path):
    # Create dummy required files in the temporary directory.
    required_files = ["spike_times.npy",
                      "spike_clusters.npy", "cluster_group.tsv"]
    for file in required_files:
        (tmp_path / file).write_text("dummy content")

    # Monkeypatch file_chooser to return our tmp_path as a Path object.
    monkeypatch.setattr(
        "pyside_gui.file_chooser.file_chooser", lambda: tmp_path)

    # Call validate_ks_folder and check if it returns the correct file paths.
    file_paths = validate_ks_folder(tmp_path, required_files)
    assert set(file_paths.keys()) == set(required_files)
    for file in required_files:
        assert (tmp_path / file).exists()


def test_choose_and_validate_ks_folder_cancel(monkeypatch):
    # Monkeypatch file_chooser to simulate cancellation (return None)
    monkeypatch.setattr("pyside_gui.file_chooser.file_chooser", lambda: None)

    # When no folder is selected, choose_and_validate_ks_folder should raise a FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        validate_ks_folder(Path("invalid_path"), ["file1", "file2"])


def test_make_specific_folder(tmp_path):
    # Create a specific folder within the temporary directory
    folder_name = "specific_folder"
    specific_folder = make_specific_folder(tmp_path, folder_name)

    # Check that the specific folder exists and is located inside tmp_path
    assert specific_folder.is_dir()
    assert specific_folder.parent == tmp_path
    assert specific_folder.name == folder_name


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

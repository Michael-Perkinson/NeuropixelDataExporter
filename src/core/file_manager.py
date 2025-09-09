from typing import Sequence
from pathlib import Path
import numpy as np
import pandas as pd

# Always required from Kilosort output
KS_REQUIRED = [
    "spike_times.npy",
    "spike_clusters.npy",
]

# One-of required: clustering annotation
KS_LABEL_FILES = [
    "cluster_KSLabel.tsv",   # Kilosort4
    "cluster_group.tsv",     # Phy / Kilosort3
]


def find_specific_files_in_folder(
    folder_path: Path, file_names: list[str]
) -> dict[str, Path]:
    """
    Search for specific files in a folder and return their full paths if found.

    Args
        folder_path: The directory where the files should be located.
        file_names: List of filenames to search for.

    Returns
        A dictionary mapping filenames to their full paths if they exist.
    """
    return {
        file: folder_path / file for file in file_names if (folder_path / file).exists()
    }


def validate_ks_folder(
    folder_path: Path,
    always_required: Sequence[str],
    one_of_required: Sequence[str],
) -> dict[str, str]:
    """
    Validate Kilosort output folder.

    Args:
        folder_path: Path to the folder being validated.
        always_required: Filenames that must all be present.
        one_of_required: A list of filenames where at least one must exist.

    Returns:
        A dict mapping filenames (present) to their resolved full paths.

    Raises:
        FileNotFoundError if any always-required file is missing
        or if none of the one-of-required files are found.
    """
    # Check always-required files
    resolved: dict[str, str] = {}
    for fname in always_required:
        fpath = folder_path / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required file: {fname}")
        resolved[fname] = str(fpath)

    # Check one-of-required files
    found = None
    for fname in one_of_required:
        fpath = folder_path / fname
        if fpath.exists():
            resolved[fname] = str(fpath)
            found = fname
            break
    if found is None:
        raise FileNotFoundError(
            f"Missing label file, need one of: {', '.join(one_of_required)}"
        )

    return resolved



def load_spike_data(
    spike_times_path: str, spike_clusters_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and return spike times and cluster IDs from .npy files.

    Args
        spike_times_path: Path to the spike times file.
        spike_clusters_path: Path to the spike clusters file.

    Returns
        A tuple containing:
        - A flattened NumPy array of spike times.
        - A flattened NumPy array of cluster IDs.
    """
    return np.load(spike_times_path).ravel(), np.load(spike_clusters_path).ravel()


def create_label_lookup(group_labels_path: str) -> np.ndarray:
    """
    Generate a lookup array mapping cluster IDs to group labels.

    The TSV file must contain a "cluster_id" column and either "group" or "KSLabel" for labels.
    Any cluster without a specified label is set to "unknown".

    Args
        group_labels_path: Path to the TSV file containing cluster labels.

    Returns
        A NumPy array mapping cluster IDs to their labels.

    Raises
        ValueError if the required label columns are missing.
    """
    cluster_group = pd.read_csv(group_labels_path, sep="\t")
    max_cluster_id = cluster_group["cluster_id"].max() + 1
    group_labels_array = np.full(max_cluster_id, "unknown", dtype=object)

    if "group" in cluster_group.columns:
        label_column = "group"
    elif "KSLabel" in cluster_group.columns:
        label_column = "KSLabel"
    else:
        raise ValueError("Expected 'group' or 'KSLabel' column in the file.")

    group_labels_array[cluster_group["cluster_id"].values] = cluster_group[
        label_column
    ].values
    return group_labels_array


def make_output_folders(data_folder_path: Path) -> tuple[Path, Path, Path]:
    """
    Create output folders for analysis results, images, and text files.

    Args
        data_folder_path: Path to the main data folder.

    Returns
        A tuple containing:
        - The path to the analysis results folder.
        - The path to the image export folder.
        - The path to the text export folder.
    """
    export_dir = make_specific_folder(data_folder_path, "analysis_results")
    images_dir = make_specific_folder(export_dir, "firing_rate_images")
    txt_export_dir = make_specific_folder(export_dir, "txt_files_for_clampfit_import")
    return export_dir, images_dir, txt_export_dir


def make_specific_folder(folder_path: Path, folder_name: str) -> Path:
    """
    Create a folder inside the specified directory if it doesn't exist.

    Args
        folder_path: The parent directory where the folder will be created.
        folder_name: Name of the folder to create.

    Returns
        The full path to the created folder.
    """
    folder = folder_path / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder

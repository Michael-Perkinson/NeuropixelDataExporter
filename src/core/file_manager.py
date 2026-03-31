from typing import Sequence, Final
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# Always required from Kilosort output
KS_REQUIRED: Final[Sequence[str]] = (
    "spike_times.npy", "spike_clusters.npy")

# One of these must be present for cluster labels
KS_LABEL_FILES: Final[Sequence[str]] = (
    "cluster_KSLabel.tsv", "cluster_group.tsv")


def find_specific_files_in_folder(
    folder_path: Path,
    always_required: Sequence[str],
    one_of_required: Sequence[str],
) -> dict[str, Path]:
    """Return a filename → Path mapping for files present in folder_path."""
    wanted = list(always_required) + list(one_of_required)
    return {
        name: (folder_path / name)
        for name in wanted
        if (folder_path / name).exists()
    }


def validate_ks_folder(
    folder_path: Path,
    always_required: Sequence[str],
    one_of_required: Sequence[str],
) -> dict[str, Path]:
    """
    Validate a Kilosort output folder.

    Raises FileNotFoundError if any always_required file is missing,
    or if none of the one_of_required label files are present.
    """
    resolved: dict[str, Path] = {}

    for fname in always_required:
        fpath = (folder_path / fname).resolve()
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required file: {fname}")
        resolved[fname] = fpath

    for fname in one_of_required:
        fpath = (folder_path / fname).resolve()
        if fpath.exists():
            resolved[fname] = fpath
            break
    else:
        raise FileNotFoundError(
            f"Missing label file, need one of: {', '.join(one_of_required)}"
        )

    return resolved


def load_spike_data(
    spike_times_path: str | Path,
    spike_clusters_path: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Load and return spike times and cluster IDs from .npy files."""
    return (
        np.load(str(spike_times_path)).ravel(),
        np.load(str(spike_clusters_path)).ravel(),
    )


def create_label_lookup(group_labels_path: str | Path) -> NDArray[np.object_]:
    """Return array mapping cluster_id → label string ("unknown" if missing)."""
    cluster_group = pd.read_csv(str(group_labels_path), sep="\t")

    if "cluster_id" not in cluster_group.columns:
        raise ValueError("Expected 'cluster_id' column in the file.")

    if "group" in cluster_group.columns:
        label_column = "group"
    elif "KSLabel" in cluster_group.columns:
        label_column = "KSLabel"
    else:
        raise ValueError("Expected 'group' or 'KSLabel' column in the file.")

    cluster_ids = pd.to_numeric(
        cluster_group["cluster_id"], errors="raise").astype(int).to_numpy()
    labels = cluster_group[label_column].astype(str).to_numpy()

    max_cluster_id = int(cluster_ids.max()) + 1 if cluster_ids.size else 0
    group_labels_array = np.full(max_cluster_id, "unknown", dtype=np.object_)
    group_labels_array[cluster_ids] = labels
    return group_labels_array


def make_output_folders(data_folder_path: Path) -> tuple[Path, Path, Path]:
    """Create and return (analysis_results/, firing_rate_images/, txt_files/) directories."""
    export_dir = make_specific_folder(data_folder_path, "analysis_results")
    images_dir = make_specific_folder(export_dir, "firing_rate_images")
    txt_export_dir = make_specific_folder(export_dir, "txt_files_for_clampfit_import")
    return export_dir, images_dir, txt_export_dir


def make_specific_folder(folder_path: Path, folder_name: str) -> Path:
    """Create folder_path/folder_name if it doesn't exist and return the path."""
    folder = folder_path / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder

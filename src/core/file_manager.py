from typing import Sequence, Final
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import pandas as pd

# Always required from Kilosort output
KS_REQUIRED: Final[Sequence[str]] = (
    "spike_times.npy", "spike_clusters.npy")

# Checked in priority order — first found wins
KS_LABEL_FILES: Final[Sequence[str]] = (
    "cluster_info.tsv", "cluster_KSLabel.tsv", "cluster_group.tsv")


SAMPLE_RATE: Final[float] = 30000.0


def get_recording_duration(spike_times_path: str | Path) -> float | None:
    """Return recording duration in seconds from spike_times.npy, or None on failure."""
    try:
        raw = np.load(str(spike_times_path)).ravel()
        if raw.size:
            return round(float(raw[-1]) / SAMPLE_RATE, 2)
    except Exception:
        pass
    return None


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


def create_label_lookup(
    group_labels_path: str | Path,
    spike_clusters: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.object_], list[str]]:
    """Return (label_array, log_messages) where label_array maps cluster_id → label string.

    Priority for label source per cluster (first non-blank wins):
      1. neurontype column (custom Phy annotation, e.g. PMNC/NMNC) — only in cluster_info.tsv
      2. group column (Phy curation: good/mua/noise)
      3. KSLabel column (Kilosort's original label)
      4. "unknown" fallback

    spike_clusters: if provided, the array is sized to cover any cluster IDs present in
    the spike data even if absent from the label file (e.g. Kilosort noise clusters that
    were never manually labelled in Phy).
    """
    log: list[str] = []
    path = Path(group_labels_path)
    df = pd.read_csv(str(path), sep="\t")

    if "cluster_id" not in df.columns:
        raise ValueError("Expected 'cluster_id' column in the file.")

    is_cluster_info = path.name == "cluster_info.tsv"

    cluster_ids = pd.to_numeric(
        df["cluster_id"], errors="raise").astype(int).to_numpy()

    # Build label array: neurontype > group > KSLabel > "unknown"
    n = len(cluster_ids)
    labels = np.full(n, "unknown", dtype=np.object_)

    if "KSLabel" in df.columns:
        ks = df["KSLabel"].fillna("").astype(str).to_numpy()
        mask = ks != ""
        labels[mask] = ks[mask]

    if "group" in df.columns:
        grp = df["group"].fillna("").astype(str).to_numpy()
        mask = grp != ""
        labels[mask] = grp[mask]

    if is_cluster_info and "neurontype" in df.columns:
        nt = df["neurontype"].fillna("").astype(str).to_numpy()
        mask = nt != ""
        labels[mask] = nt[mask]
        count = int(mask.sum())
        if count:
            log.append(f"  {count} cluster(s) have a neurontype annotation.")

    max_from_tsv = int(cluster_ids.max()) + 1 if cluster_ids.size else 0
    max_from_spikes = (int(spike_clusters.max()) + 1
                       if spike_clusters is not None and spike_clusters.size
                       else 0)
    max_cluster_id = max(max_from_tsv, max_from_spikes)

    group_labels_array = np.full(max_cluster_id, "unknown", dtype=np.object_)
    group_labels_array[cluster_ids] = labels
    return group_labels_array, log


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

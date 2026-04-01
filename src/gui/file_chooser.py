from PySide6.QtWidgets import QFileDialog
from pathlib import Path


def file_chooser(parent=None, start_dir: Path | None = None) -> Path | None:
    """Open a folder dialog. Starts in start_dir if provided, otherwise cwd."""
    folder_path = QFileDialog.getExistingDirectory(
        parent,
        "Select a folder containing the required files",
        str(start_dir) if start_dir and start_dir.exists() else str(Path.home()),
    )

    if not folder_path:
        return None

    return Path(folder_path)

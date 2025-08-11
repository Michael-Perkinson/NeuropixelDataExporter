from PySide6.QtWidgets import QFileDialog
from pathlib import Path


def file_chooser(parent=None) -> Path | None:
    """
    Open a folder dialog using PySide6 for the user to select a folder.

    Returns
        Path object of the selected folder, or None if canceled.
    """
    folder_path = QFileDialog.getExistingDirectory(
        parent,
        "Select a folder containing the required files",
        str(Path.cwd())
    )

    if not folder_path:
        return None

    return Path(folder_path)

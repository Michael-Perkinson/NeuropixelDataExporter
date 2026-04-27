import numpy as np
import pandas as pd
import pytest
from src.core.interactive_plot import export_firing_rate_html


def _make_df():
    return pd.DataFrame({
        "Time Intervals (s)": np.array([0, 1, 2, 3]),
        "Cluster_0": np.array([5, 10, 15, 20]),
    })


def test_export_firing_rate_html_missing_time_column(tmp_path):
    df = pd.DataFrame({"Cluster_0": [1, 2, 3]})
    with pytest.raises(ValueError, match="Time Intervals"):
        export_firing_rate_html(df, tmp_path, 1.0, [])


def test_export_firing_rate_html_interval_drug_event(tmp_path):
    """Drug event with a numeric end time renders shaded region (interval path)."""
    df = _make_df()
    drug_events = [{"name": "Drug", "start": 1.0, "end": 2.0}]
    export_firing_rate_html(df, tmp_path, 1.0, drug_events)
    assert any(f.suffix == ".html" for f in tmp_path.iterdir())


def test_export_firing_rate_html(tmp_path):
    # Create a synthetic firing_rate_df
    time_intervals = np.array([0, 1, 2, 3])
    df = pd.DataFrame(
        {
            "Time Intervals (s)": time_intervals,
            "Cluster_0": np.array([5, 10, 15, 20]),
            "Cluster_1": np.array([4, 8, 12, 16]),
        }
    )
    bin_size = 1.0
    drug_events = [{"name": "Saline", "start": 1.5, "end": None}]

    # Create a temporary directory for the exported plots.
    images_export_dir = tmp_path / "plots"
    images_export_dir.mkdir(parents=True, exist_ok=True)

    # Call the function to export HTML plots.
    export_firing_rate_html(df, images_export_dir, bin_size, drug_events)

    # Check that an HTML file exists for each cluster.
    for cluster in ["Cluster_0", "Cluster_1"]:
        html_path = (
            images_export_dir / f"Firing_Rate_{cluster}_BinSize_{bin_size}s.html"
        )
        assert html_path.is_file(), f"HTML file {html_path} does not exist"
        # Optionally, check that the file is non-empty.
        assert html_path.stat().st_size > 0

import numpy as np
import pandas as pd
from src.core.interactive_plot import export_firing_rate_html


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

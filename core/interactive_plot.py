from pathlib import Path
import plotly.graph_objects as go
import pandas as pd


def export_firing_rate_html(
    firing_rate_df: pd.DataFrame,
    images_export_dir: Path,
    bin_size: float,
    drug_time: float | None,
) -> None:
    """
    Generate interactive HTML bar plots for firing rate data of each cluster.

    Args
        firing_rate_df: DataFrame containing firing rate data. The first column must be 'Time Intervals (s)',
                        and the remaining columns correspond to clusters.
        images_export_dir: Directory where the HTML files will be saved.
        bin_size: Bin size used in the firing rate calculation (seconds).
        drug_time: Time of drug application (seconds), used to annotate plots if provided.
    """
    images_export_dir.mkdir(parents=True, exist_ok=True)
    time_bins = firing_rate_df["Time Intervals (s)"]

    for cluster in firing_rate_df.columns:
        if cluster == "Time Intervals (s)":
            continue

        fig = go.Figure()

        # Add bar trace for the firing rate histogram.
        fig.add_trace(
            go.Bar(
                x=time_bins,
                y=firing_rate_df[cluster],
                name=cluster,
                marker_color="blue",
            )
        )

        # Add vertical marker for drug application time if provided.
        if drug_time is not None:
            max_y = firing_rate_df[cluster].max() * 1.1
            fig.add_shape(
                type="line",
                x0=drug_time,
                y0=0,
                x1=drug_time,
                y1=max_y,
                line=dict(color="red", width=2, dash="dash"),
            )
            fig.add_annotation(
                x=drug_time,
                y=max_y,
                text="Drug Application",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-20,
            )

        # Customize layout.
        fig.update_layout(
            title=f"Firing Rate Histogram for {cluster} (Bin Size: {bin_size}s)",
            xaxis_title="Time (s)",
            yaxis_title="Firing Rate (Hz)",
            plot_bgcolor="white",
            bargap=0,
            xaxis=dict(showline=True, linecolor="black", showgrid=False),
            yaxis=dict(showline=True, linecolor="black", showgrid=False),
        )

        # Save plot as an interactive HTML file.
        html_path = (
            images_export_dir / f"Firing_Rate_{cluster}_BinSize_{bin_size}s.html"
        )
        fig.write_html(html_path)

    print(f"Interactive plots saved to {images_export_dir}")

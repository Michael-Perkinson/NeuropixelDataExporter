from __future__ import annotations

from pathlib import Path
from typing import Sequence, TypedDict

import pandas as pd
import plotly.graph_objects as go


class DrugEvent(TypedDict):
    name: str
    start: float
    end: float | None


def export_firing_rate_html(
    firing_rate_df: pd.DataFrame,
    images_export_dir: Path,
    bin_size: float,
    drug_events: Sequence[DrugEvent] | None = None,
) -> None:
    """
    Generate interactive HTML bar plots for firing rate data of each cluster.

    Drug events:
      - end is None -> point event (vertical line)
      - end is not None -> interval event (shaded region)

    Notes:
      - Plotly "shapes" do not appear in the legend, so we add invisible/dummy
        scatter traces to create a legend entry per drug.
    """
    images_export_dir.mkdir(parents=True, exist_ok=True)

    time_col = "Time Intervals (s)"
    if time_col not in firing_rate_df.columns:
        raise ValueError(f"Expected column '{time_col}' in firing_rate_df")

    time_bins = firing_rate_df[time_col]

    # Deterministic palette (repeat if > palette length)
    palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]

    events = list(drug_events) if drug_events else []

    for cluster in firing_rate_df.columns:
        if cluster == time_col:
            continue

        fig = go.Figure()

        # Bars
        fig.add_trace(
            go.Bar(
                x=time_bins,
                y=firing_rate_df[cluster],
                name=cluster,
            )
        )

        # Drug overlays
        if events:
            # Use per-cluster max for sensible y placement
            y_max = float(firing_rate_df[cluster].max())
            y_pad = y_max * 0.10 if y_max > 0 else 1.0
            y_top = y_max + y_pad

            for i, ev in enumerate(events):
                name = str(ev["name"])
                start = float(ev["start"])
                end = ev.get("end", None)
                color = palette[i % len(palette)]

                # Add a dummy trace so the drug appears in the legend
                # (shapes don't show up in legend)
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color=color, width=3),
                        name=name,
                        showlegend=True,
                    )
                )

                if end is None:
                    # Point event -> vertical line + label
                    fig.add_shape(
                        type="line",
                        x0=start,
                        y0=0,
                        x1=start,
                        y1=y_top,
                        line=dict(color=color, width=2, dash="dash"),
                    )
                    fig.add_annotation(
                        x=start,
                        y=y_top,
                        text=name,
                        showarrow=True,
                        arrowhead=2,
                        ax=0,
                        ay=-20,
                    )
                else:
                    end_f = float(end)
                    # Interval event -> shaded region + boundary lines
                    fig.add_vrect(
                        x0=start,
                        x1=end_f,
                        fillcolor=color,
                        opacity=0.18,
                        line_width=0,
                        layer="below",
                    )
                    fig.add_shape(
                        type="line",
                        x0=start,
                        y0=0,
                        x1=start,
                        y1=y_top,
                        line=dict(color=color, width=2),
                    )
                    fig.add_shape(
                        type="line",
                        x0=end_f,
                        y0=0,
                        x1=end_f,
                        y1=y_top,
                        line=dict(color=color, width=2),
                    )
                    # One label in the middle of the interval
                    mid = start + (end_f - start) / 2.0
                    fig.add_annotation(
                        x=mid,
                        y=y_top,
                        text=name,
                        showarrow=False,
                    )

        fig.update_layout(
            title=f"Firing Rate Histogram for {cluster} (Bin Size: {bin_size}s)",
            xaxis_title="Time (s)",
            yaxis_title="Firing Rate (Hz)",
            bargap=0,
        )

        html_path = images_export_dir / \
            f"Firing_Rate_{cluster}_BinSize_{bin_size}s.html"
        fig.write_html(html_path)

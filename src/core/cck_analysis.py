from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

# CCK constants (5-min protocol)
CCK_WINDOW_S: float = 300.0             # pre/post window
CCK_BIN_S: float = 60.0                 # bin width → 5 bins each side
CCK_THRESHOLD_HZ: float = 0.5           # delta >= this → Putative Oxytocin
CCK_STABILITY_HZ_PER_MIN: float = 0.1   # pre-window slope flag

# PE constants (1-min protocol)
PE_WINDOW_S: float = 60.0               # pre/post window
PE_BIN_S: float = 10.0                  # bin width → 6 bins each side
PE_THRESHOLD_HZ: float = -0.5           # delta <= this → Putative Vasopressin
PE_STABILITY_HZ_PER_MIN: float = 0.1    # pre-window slope flag


def _bin_spikes(
    spikes_ms: NDArray[np.float64],
    window_start_s: float,
    window_end_s: float,
    n_bins: int,
) -> NDArray[np.float64]:
    """Bin spikes into n_bins equal-width bins; return firing rates (Hz)."""
    bin_edges = np.linspace(window_start_s, window_end_s, n_bins + 1)
    spikes_s = spikes_ms / 1000.0
    counts, _ = np.histogram(spikes_s, bins=bin_edges)
    bin_width = (window_end_s - window_start_s) / n_bins
    return counts.astype(np.float64) / bin_width


def _baseline_regression(
    pre_rates: NDArray[np.float64],
    bin_s: float,
) -> tuple[float, float]:
    """
    Linear regression of pre-event firing rates.

    x-axis: bin centres in minutes relative to event (negative = before).
    Returns (slope_hz_per_min, r_squared).
    """
    n = len(pre_rates)
    bin_min = bin_s / 60.0
    bin_centres = np.array([(-n + i + 0.5) * bin_min for i in range(n)])

    if pre_rates.std() == 0.0:
        return 0.0, float("nan")

    result = stats.linregress(bin_centres, pre_rates)
    return float(result.slope), float(result.rvalue ** 2)


def _stability_note(slope_hz_per_min: float, threshold: float) -> str:
    if abs(slope_hz_per_min) <= threshold:
        return ""
    direction = "Rising" if slope_hz_per_min > 0 else "Falling"
    return f"⚠ {direction} baseline"


def _analyse_cell_type_protocol(
    data_export: dict[int, NDArray[np.float64]],
    event_time_s: float,
    window_s: float,
    bin_s: float,
    stability_threshold: float,
    classify_fn: Callable[[float], str],
    sheet_label: str,
) -> pd.DataFrame:
    """
    Shared analysis engine for CCK and PE protocols.

    Args:
        data_export: cluster_id → spike times in ms (absolute, not shifted).
        event_time_s: injection time in seconds from recording start.
        window_s: pre and post window duration in seconds.
        bin_s: bin width in seconds.
        stability_threshold: slope (Hz/min) above which baseline is flagged.
        classify_fn: function(delta_hz) → classification string.
        sheet_label: protocol name used in column headers ("CCK"/"PE").
    """
    n_bins = int(round(window_s / bin_s))
    pre_start = event_time_s - window_s
    pre_end = event_time_s
    post_start = event_time_s
    post_end = event_time_s + window_s

    rows: list[dict[str, object]] = []

    for cluster_id, spikes_ms in data_export.items():
        pre_rates = _bin_spikes(spikes_ms, pre_start, pre_end, n_bins)
        post_rates = _bin_spikes(spikes_ms, post_start, post_end, n_bins)

        pre_mean = float(pre_rates.mean())
        post_mean = float(post_rates.mean())
        delta = post_mean - pre_mean
        slope, r2 = _baseline_regression(pre_rates, bin_s)

        row: dict[str, object] = {"Cluster": f"Cluster_{cluster_id}"}

        for i, v in enumerate(pre_rates, 1):
            row[f"Pre_Bin_{i}_Hz"] = round(v, 4)
        for i, v in enumerate(post_rates, 1):
            row[f"Post_Bin_{i}_Hz"] = round(v, 4)

        row["Pre_Mean_FR_Hz"] = round(pre_mean, 4)
        row["Post_Mean_FR_Hz"] = round(post_mean, 4)
        row["Delta_FR_Hz"] = round(delta, 4)
        row["Baseline_Slope_Hz_per_min"] = round(slope, 4)
        row["R_squared"] = round(r2, 4) if not np.isnan(r2) else float("nan")
        row["Classification"] = classify_fn(delta)
        row["Notes"] = _stability_note(slope, stability_threshold)

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            "Cluster",
            key=lambda s: s.str.extract(r"(\d+)")[0].astype(int),
        ).reset_index(drop=True)

    return df


# CCK protocol

def _classify_cck(delta_hz: float) -> str:
    if delta_hz >= CCK_THRESHOLD_HZ:
        return "Putative Oxytocin"
    return "Putative Vasopressin"


def analyse_cck_response(
    data_export: dict[int, NDArray[np.float64]],
    cck_time_s: float,
) -> pd.DataFrame:
    """
    Classify clusters based on firing rate change 5 min before vs after CCK.

    Putative Oxytocin: delta >= CCK_THRESHOLD_HZ (0.5 Hz).
    Pre-window linear regression flags an unstable baseline if slope > CCK_STABILITY_HZ_PER_MIN.
    """
    return _analyse_cell_type_protocol(
        data_export=data_export,
        event_time_s=cck_time_s,
        window_s=CCK_WINDOW_S,
        bin_s=CCK_BIN_S,
        stability_threshold=CCK_STABILITY_HZ_PER_MIN,
        classify_fn=_classify_cck,
        sheet_label="CCK",
    )


# PE protocol

def _classify_pe(delta_hz: float) -> str:
    if delta_hz <= PE_THRESHOLD_HZ:
        return "Putative Vasopressin"
    return "Putative Oxytocin"


def analyse_pe_response(
    data_export: dict[int, NDArray[np.float64]],
    pe_time_s: float,
) -> pd.DataFrame:
    """
    Classify clusters based on firing rate change 1 min before vs after PE.

    Putative Vasopressin: delta <= PE_THRESHOLD_HZ (-0.5 Hz).
    Uses 10-second bins (6 bins per side).
    """
    return _analyse_cell_type_protocol(
        data_export=data_export,
        event_time_s=pe_time_s,
        window_s=PE_WINDOW_S,
        bin_s=PE_BIN_S,
        stability_threshold=PE_STABILITY_HZ_PER_MIN,
        classify_fn=_classify_pe,
        sheet_label="PE",
    )

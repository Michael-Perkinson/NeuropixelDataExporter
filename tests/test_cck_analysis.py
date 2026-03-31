import numpy as np
import pytest

from src.core.cck_analysis import (
    analyse_cck_response,
    analyse_pe_response,
    CCK_WINDOW_S,
    CCK_BIN_S,
    PE_WINDOW_S,
    PE_BIN_S,
)


def _make_spikes_ms(rate_hz: float, start_s: float, end_s: float) -> np.ndarray:
    """Generate evenly-spaced spike times (ms) at the given rate."""
    n = int((end_s - start_s) * rate_hz)
    return np.linspace(start_s, end_s, n) * 1000.0


# --- analyse_cck_response ---

def test_cck_putative_oxytocin():
    """Cluster firing more after CCK → Putative Oxytocin."""
    cck_time = CCK_WINDOW_S  # onset at 300 s
    # Low pre-CCK rate, high post-CCK rate
    pre_spikes = _make_spikes_ms(1.0, 0.0, cck_time)
    post_spikes = _make_spikes_ms(5.0, cck_time, cck_time + CCK_WINDOW_S)
    spikes = np.concatenate([pre_spikes, post_spikes])

    df = analyse_cck_response({0: spikes}, cck_time_s=cck_time)

    assert len(df) == 1
    assert df.loc[0, "Classification"] == "Putative Oxytocin"
    assert df.loc[0, "Delta_FR_Hz"] > 0.5


def test_cck_putative_vasopressin():
    """Cluster firing less after CCK → Putative Vasopressin."""
    cck_time = CCK_WINDOW_S
    pre_spikes = _make_spikes_ms(5.0, 0.0, cck_time)
    post_spikes = _make_spikes_ms(1.0, cck_time, cck_time + CCK_WINDOW_S)
    spikes = np.concatenate([pre_spikes, post_spikes])

    df = analyse_cck_response({0: spikes}, cck_time_s=cck_time)

    assert df.loc[0, "Classification"] == "Putative Vasopressin"
    assert df.loc[0, "Delta_FR_Hz"] < 0.5


def test_cck_output_columns():
    """Output DataFrame has all expected columns."""
    cck_time = CCK_WINDOW_S
    spikes = _make_spikes_ms(2.0, 0.0, cck_time * 2)
    df = analyse_cck_response({0: spikes}, cck_time_s=cck_time)

    n_pre_bins = int(round(CCK_WINDOW_S / CCK_BIN_S))
    expected = (
        ["Cluster"]
        + [f"Pre_Bin_{i}_Hz" for i in range(1, n_pre_bins + 1)]
        + [f"Post_Bin_{i}_Hz" for i in range(1, n_pre_bins + 1)]
        + ["Pre_Mean_FR_Hz", "Post_Mean_FR_Hz", "Delta_FR_Hz",
           "Baseline_Slope_Hz_per_min", "R_squared", "Classification", "Notes"]
    )
    assert list(df.columns) == expected


def test_cck_multiple_clusters_sorted():
    """Clusters are sorted numerically in the output."""
    cck_time = CCK_WINDOW_S
    data = {
        5: _make_spikes_ms(2.0, 0.0, cck_time * 2),
        1: _make_spikes_ms(2.0, 0.0, cck_time * 2),
        10: _make_spikes_ms(2.0, 0.0, cck_time * 2),
    }
    df = analyse_cck_response(data, cck_time_s=cck_time)
    assert list(df["Cluster"]) == ["Cluster_1", "Cluster_5", "Cluster_10"]


def test_cck_no_spikes():
    """Cluster with no spikes produces delta of 0 → Putative Vasopressin."""
    cck_time = CCK_WINDOW_S
    df = analyse_cck_response({0: np.array([])}, cck_time_s=cck_time)
    assert len(df) == 1
    assert df.loc[0, "Pre_Mean_FR_Hz"] == 0.0
    assert df.loc[0, "Post_Mean_FR_Hz"] == 0.0


# --- analyse_pe_response ---

def test_pe_putative_vasopressin():
    """Cluster inhibited by PE → Putative Vasopressin."""
    pe_time = PE_WINDOW_S
    pre_spikes = _make_spikes_ms(5.0, 0.0, pe_time)
    post_spikes = _make_spikes_ms(0.5, pe_time, pe_time + PE_WINDOW_S)
    spikes = np.concatenate([pre_spikes, post_spikes])

    df = analyse_pe_response({0: spikes}, pe_time_s=pe_time)

    assert df.loc[0, "Classification"] == "Putative Vasopressin"
    assert df.loc[0, "Delta_FR_Hz"] <= -0.5


def test_pe_putative_oxytocin():
    """Cluster excited by PE → Putative Oxytocin."""
    pe_time = PE_WINDOW_S
    pre_spikes = _make_spikes_ms(1.0, 0.0, pe_time)
    post_spikes = _make_spikes_ms(5.0, pe_time, pe_time + PE_WINDOW_S)
    spikes = np.concatenate([pre_spikes, post_spikes])

    df = analyse_pe_response({0: spikes}, pe_time_s=pe_time)

    assert df.loc[0, "Classification"] == "Putative Oxytocin"


def test_pe_output_columns():
    """Output DataFrame has all expected columns."""
    pe_time = PE_WINDOW_S
    spikes = _make_spikes_ms(2.0, 0.0, pe_time * 2)
    df = analyse_pe_response({0: spikes}, pe_time_s=pe_time)

    n_bins = int(round(PE_WINDOW_S / PE_BIN_S))
    expected = (
        ["Cluster"]
        + [f"Pre_Bin_{i}_Hz" for i in range(1, n_bins + 1)]
        + [f"Post_Bin_{i}_Hz" for i in range(1, n_bins + 1)]
        + ["Pre_Mean_FR_Hz", "Post_Mean_FR_Hz", "Delta_FR_Hz",
           "Baseline_Slope_Hz_per_min", "R_squared", "Classification", "Notes"]
    )
    assert list(df.columns) == expected

import numpy as np
import pandas as pd
import pytest

from src.core.isi_hazard import compute_hazard_values
from src.core.results_writer import export_data


def test_last_occupied_hazard_bin():
    bins = np.array([0., .1, .2])
    result = compute_hazard_values(pd.DataFrame({"Bin_Starts": bins, "Cluster_1": [10, 5, 0]}), bins)
    np.testing.assert_allclose(result["Cluster_1"], [2 / 3, 1, 0])


def export_example(tmp_path, events=None):
    export_data({1: np.array([100., 600.]), 2: np.array([])}, None,
                tmp_path, 1., 0., 2., None, None, export_txt=False,
                drug_events=events, cluster_group_map={1: "good", 2: "good"})
    return tmp_path / "analysis_results" / "firing_rates_by_cluster.xlsx"


def test_silent_cluster_and_full_label_mean(tmp_path):
    path = export_example(tmp_path)
    raw = pd.read_excel(path, sheet_name="Binned_Firing_Rates")
    assert "Cluster_2" in raw
    assert raw["Cluster_2"].tolist() == [0., 0.]
    means = pd.read_excel(path, sheet_name="Mean_by_Label")
    assert means["Mean_good_Hz"].tolist() == [1., 0.]


@pytest.mark.parametrize("names", [["Drug", "Drug"], ["Drug/A", "Drug:A"], ["abcdefghijklmnopqrst1", "abcdefghijklmnopqrst2"]])
def test_drug_sheets_do_not_collide(tmp_path, names):
    events = [dict(name=name, start=float(i), end=None, pre_time=0., post_time=2.) for i, name in enumerate(names)]
    path = export_example(tmp_path, events)
    with pd.ExcelFile(path) as book:
        sheets = [s for s in book.sheet_names if s.startswith("Peri_")]
        assert len(sheets) == 2
        assert pd.read_excel(book, sheet_name=sheets[0])["Time Intervals (s)"].tolist() == [0., 1.]
        assert pd.read_excel(book, sheet_name=sheets[1])["Time Intervals (s)"].tolist() == [-1., 0.]


def make_worker(tmp_path, **overrides):
    from src.gui.gui_controller import AnalysisWorker
    duration = overrides.pop("recording_duration", 120.)
    times = np.arange(overrides.pop("first_spike", .5), duration, .5)
    np.save(tmp_path / "spike_times.npy", times * 30000)
    np.save(tmp_path / "spike_clusters.npy", np.ones(len(times), dtype=int))
    pd.DataFrame({"cluster_id": [1], "group": ["good"]}).to_csv(tmp_path / "cluster_group.tsv", sep="\t", index=False)
    args = dict(file_paths={name: tmp_path / name for name in ["spike_times.npy", "spike_clusters.npy", "cluster_group.tsv"]},
                cluster_ids=[1], start_time=0., end_time=119.5, bin_size=10.,
                use_baseline=False, baseline_start=None, baseline_end=None,
                run_hazard=True, peri_hazard=True, early_hazard_start=0., early_hazard_end=30.,
                export_all_graphs=False, export_txt=False, export_peri_drug=False,
                cck_time=None, pe_time=None, mean_label_data=False, plot_events=[], active_drug_events=[])
    args.update(overrides)
    return AnalysisWorker(**args)


def test_incomplete_protocol_rejected_before_export(tmp_path):
    worker = make_worker(tmp_path, pe_time=60., end_time=90.)
    with pytest.raises(ValueError, match="PE"):
        worker._run()
    assert not (tmp_path / "analysis_results").exists()


def test_hazard_uses_full_recording_and_independent_drug_events(tmp_path):
    event = dict(name="Drug", start=20., end=None)
    worker = make_worker(tmp_path, start_time=60., hazard_drug_events=[event])
    worker._run()
    path = tmp_path / "analysis_results" / "isi_and_hazard_analysis.xlsx"
    with pd.ExcelFile(path) as book:
        full = pd.read_excel(book, sheet_name="Full_ISI")
        assert full["Cluster_1"].sum() == 238
        early_name = next(s for s in book.sheet_names if s.startswith("Early_ISI"))
        assert pd.read_excel(book, sheet_name=early_name)["Cluster_1"].sum() == 59
        assert "Drug_PreDrug_ISI" in book.sheet_names


def test_delta_names_and_guides_remain_distinct(tmp_path):
    events = [dict(name=name, start=0., end=None, pre_time=0., post_time=2.)
              for name in ["Drug", "drug", "Drug_Delta"]]
    export_data({1: np.array([100., 600.])}, {1: (1., 0.)}, tmp_path,
                1., 0., 2., 0., 1., drug_events=events, export_txt=False)
    path = tmp_path / "analysis_results" / "firing_rates_by_cluster.xlsx"
    with pd.ExcelFile(path) as book:
        peri = [s for s in book.sheet_names if s.startswith("Peri_")]
        assert len(peri) == 6
        guide = pd.read_excel(book, sheet_name="Sheet_Guide")
        assert set(peri) <= set(guide["Sheet"])
        assert "Peri_Drug" in peri
        assert "Peri_Drug_Delta" in peri


def test_hazard_duplicate_names_and_guide(tmp_path):
    events = [dict(name="Drug/A", start=start, end=start + 5.) for start in [20., 40.]]
    make_worker(tmp_path, hazard_drug_events=events)._run()
    path = tmp_path / "analysis_results" / "isi_and_hazard_analysis.xlsx"
    with pd.ExcelFile(path) as book:
        peri = [s for s in book.sheet_names if "_PreDrug_" in s or "_EndDrug_" in s]
        assert len(peri) == 12
        guide = pd.read_excel(book, sheet_name="Summary")
        assert set(peri) <= set(guide["Sheet"])


def test_all_silent_selected_clusters_are_exported(tmp_path):
    export_data({1: np.array([])}, None, tmp_path, 1., 0., 2., None, None,
                export_txt=False, cluster_group_map={1: "good"})
    path = tmp_path / "analysis_results" / "firing_rates_by_cluster.xlsx"
    assert pd.read_excel(path, sheet_name="Mean_by_Label")["Mean_good_Hz"].tolist() == [0., 0.]


@pytest.mark.parametrize("protocol,onset,end", [("pe", 60., 120.), ("cck", 300., 600.)])
def test_complete_protocol_still_exports(tmp_path, protocol, onset, end):
    worker = make_worker(tmp_path, recording_duration=end + 1., end_time=end, first_spike=.25,
                         **{f"{protocol}_time": onset})
    worker._run()
    path = tmp_path / "analysis_results" / "firing_rates_by_cluster.xlsx"
    result = pd.read_excel(path, sheet_name=f"{protocol.upper()}_Cell_Typing")
    assert abs(result.loc[0, "Delta_FR_Hz"]) < .01


def test_cck_cannot_extend_past_actual_recording(tmp_path):
    worker = make_worker(tmp_path, cck_time=300., end_time=600.)
    with pytest.raises(ValueError, match="CCK"):
        worker._run()
    assert not (tmp_path / "analysis_results").exists()

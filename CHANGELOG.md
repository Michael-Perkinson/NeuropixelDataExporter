# Changelog

## [2.0.0] - 2026-03-31

This release is a complete rewrite of the tool. The original monolithic terminal script (`export_clusters.py`) has been replaced by a PySide6 desktop application with a fully graphical interface, background threading, multi-drug support, cell-type classification, and a restructured multi-sheet Excel output.

---

### Architecture

- **Complete rewrite**: replaced `export_clusters.py` (interactive terminal prompts) with a PySide6 GUI application launched via `main.py`
- **Package structure**: all source code moved into `src/` with a `core/` module (analysis logic) and `gui/` module (interface), separating concerns cleanly
- **Background threading**: analysis now runs on a `QThread` (`AnalysisWorker`) so the GUI stays responsive during long exports; the Run button disables while running and re-enables on completion
- **Session persistence**: most settings (checkboxes, theme) are saved to `.neuropixel_gui_last_session.json` and restored on next launch
- **Settings import/export**: full settings can be saved to and loaded from a user-chosen JSON file via the Settings menu
- **Fixed window size**: application window is locked at 1080×720 to prevent layout distortion on different screens

### New Modules

| File | Purpose |
| --- | --- |
| `src/core/cck_analysis.py` | CCK and PE cell-type classification engine with shared analysis protocol |
| `src/core/file_manager.py` | Kilosort folder validation, required file checks, label file discovery |
| `src/core/firing_rate.py` | Firing rate calculation and dataframe construction |
| `src/core/input_parser.py` | Parsing of cluster/label inputs and drug event fields |
| `src/core/interactive_plot.py` | Per-cluster Plotly HTML export with drug overlays |
| `src/core/isi_hazard.py` | ISI histogram and hazard function calculation |
| `src/core/results_writer.py` | Multi-sheet Excel export engine |
| `src/core/spike_filter.py` | Spike data loading, windowed ISI helper, label resolution |
| `src/gui/gui_controller.py` | Controller: input validation, worker management, settings I/O |
| `src/gui/view.py` | MainWindow: all UI widget construction |
| `src/gui/gui_themes.py` | QSS stylesheets for light and dark themes |

### Removed

- `export_clusters.py` — replaced by the GUI application
- `src/controller/flow_controller.py` — dead code depending on deleted terminal prompts module
- `src/core/terminal_prompts.py` — all terminal-based prompting removed
- `tests/test_terminal_prompts.py` — tested the deleted module

---

### New Features

#### GUI Layout

- Two-column layout: Analysis settings (left, 40%) and Drug Event management (right, 60%)
- Left column contains: Analysis group (clusters, time range, bin size, FR baseline, ISI hazard window), Cell Typing tabs (CCK/PE), Output Options grid
- Right column contains: Add Drug Event form, drug event table, Run button, and log panel
- Light theme by default (blue accent `#0077b6`, neutral background `#f0f2f5`); dark mode toggle under Settings menu

#### Cluster Selection

- Accepts numeric cluster IDs, Kilosort/Phy group labels (e.g. `good`, `mua`), or a mix of both in one field
- Label dropdown auto-populates from the selected folder's label file
- Clicking a label in the dropdown appends it to the text field

#### Cell-Type Classification — CCK (IV)

- New `CCK_Cell_Typing` sheet in the firing rates Excel file
- Protocol: compare mean firing rate in 5 min pre-window vs 5 min post-window using 1-minute bins
- Classification: delta ≥ +0.5 Hz → Putative Oxytocin; delta < +0.5 Hz → Putative Vasopressin (binary, no dead zone)
- Pre-window linear regression used to flag baseline instability
- Sheet columns: Cluster, Pre_Mean_FR_Hz, Post_Mean_FR_Hz, Delta_FR_Hz, Classification, Baseline_Stability
- Baseline_Stability categories: Stable (≤0.1 Hz/min), Small/Medium/Large rising or falling drift, with ⚠ for large (>0.6 Hz/min)
- Window validation: warns in the log if the analysis start/end clips the required 5-minute pre/post windows
- All protocol thresholds defined as named constants in `cck_analysis.py` for easy tuning

#### Cell-Type Classification — PE (IV)

- New `PE_Cell_Typing` sheet in the firing rates Excel file
- Protocol: compare mean firing rate in 1 min pre-window vs 1 min post-window using 10-second bins
- Classification: delta ≤ −0.5 Hz → Putative Vasopressin; delta > −0.5 Hz → Putative Oxytocin (binary)
- Same baseline stability analysis and sheet format as CCK

#### Drug Event Management

- Named drug events with: name, route (Microdialysis / IV), start time (s), optional end time (s)
- End time field accepts `max` to indicate the drug runs to the end of the recording
- Optional peri-drug window per event, entered as a single value (`60` = symmetric) or `pre/post` pair (`300/60`)
- Multiple drugs supported simultaneously in a scrollable table
- Right-click a table row to remove it

#### Peri-Drug Firing Rate Sheets

- One `Peri_<Drug>` sheet per drug event that has a peri window configured
- Time axis reset so t=0 corresponds to drug onset
- Post-window clipped to recording end with a warning in the Summary sheet if a finite post-time exceeds the recording; `max` post-times do not generate spurious warnings
- Peri window shown in the Summary sheet as an absolute time range (e.g. `3600.0s – 4800.0s`) rather than raw offsets

#### Mean by Label

- When enabled, adds a `Mean_by_Label` sheet with firing rates averaged across all clusters sharing the same Phy group label
- For each drug with a peri window, also adds a corresponding `MeanPeri_<Drug>` sheet with the same label-averaged rates aligned to drug onset

#### ISI / Hazard Analysis Restructure

- **Full recording**: `Full_ISI`, `Full_Hazard`, `Full_Hazard_Summary` sheets (renamed from previous `ISI_Histogram`, `Hazard_Function`, `Hazard_Summary`)
- **Early recording reference window**: new "ISI Hazard Window" field in the Analysis section (default 0–600 s, configurable); always computed when hazard is enabled; produces `Early_ISI (X–Ys)`, `Early_Hazard (X–Ys)`, `Early_Hazard_Summary` sheets. This is separate from the firing rate baseline and is intended as a pre-drug reference hazard shape
- **Peri-drug hazard** (new checkbox): for each drug with a peri window, computes ISI histograms and hazard functions for the pre and post epochs separately, and writes them as `Peri_<Drug>_ISI` and `Peri_<Drug>_Hazard` sheets with `_Pre` / `_Post` column suffixes so pre and post can be compared directly
- `calculate_windowed_isi()` added to `isi_hazard.py` to compute ISI for an arbitrary time window with optional column suffix

#### Output Options Checkboxes

- 2×3 grid of toggleable output options, all enabled by default:
  - **Binned ISI & Hazard** — full + early window ISI/hazard Excel file
  - **Peri-drug Hazard** — per-drug pre/post epoch ISI and hazard sheets (replaces the old "Baseline ISI & Hazard" checkbox)
  - **Export TXT (Clampfit)** — per-cluster `.txt` files for Clampfit import
  - **Export All Graphs** — per-cluster interactive HTML firing rate plots
  - **Mean by Label** — label-averaged firing rate sheets
  - **Peri-drug Sheets** — peri-drug firing rate Excel sheets

#### Interactive Plots

- Per-cluster Plotly bar charts exported as HTML files
- Drug events with both start and end time rendered as shaded regions
- Drug events with only a start time rendered as dashed vertical lines with annotation
- `max` drug end times resolved to the actual recording end before plotting

#### Summary Sheet

- First sheet in `firing_rates_by_cluster.xlsx` (moved to front automatically)
- Records: analysis window, bin size, firing rate baseline window, CCK and PE injection times and protocol details, putative OT/VP neuron counts per protocol, drug event onset/offset and peri-window time range, any clipping warnings from peri-drug sheet construction
- Drug end times shown as "max (recording end)" when applicable

---

### Changed

- **Sheet name**: `Firing_Rates_Raw` renamed to `Binned_Firing_Rates` throughout (the data is binned, not raw)
- **Sheet order in firing rates xlsx**: Summary first, Binned_Firing_Rates always last; other sheets in between in logical order
- **CCK/PE sheet format**: replaced individual pre/post bin columns with Pre_Mean_FR_Hz and Post_Mean_FR_Hz (5-minute or 1-minute averages); replaced R_squared with human-readable Baseline_Stability; removed intermediate bin columns
- **Peri-drug warning**: suppressed for `max` (infinite) post-window values; only warns when a specific finite time genuinely exceeds the recording end
- **Baseline checkbox label**: renamed from "Baseline" to "FR Baseline" to distinguish from the ISI Hazard Window below it
- **ISI Hazard Window**: moved out of the checkboxes and into the Analysis group as a dedicated configurable field with clearly distinct label and tooltip
- **CCK/PE time fields**: no longer saved/restored between sessions (intentionally blank on launch to avoid stale values being used accidentally)

---

### Fixed

- GUI freeze during analysis — resolved by moving all heavy computation to `AnalysisWorker(QThread)`
- Label-based cluster filtering not working — resolved by loading the dataframe and resolving labels to cluster IDs on the main thread before the worker starts
- Peri-drug time axis not reset — fixed in `_build_peri_sheet()` by subtracting drug onset from the Time Intervals column
- `export_hazard_excel()` positional argument error — optional arguments are now passed as keyword-only
- All output checkboxes now default to checked (previously defaulted to unchecked, causing no outputs)
- `prepare_filtered_data` unused top-level import warning in `gui_controller.py` — removed; function is now imported locally where needed
- `isi_hazard` functions now validate window/baseline ranges and non-negative monotonic spike times, with dedicated tests for empty windows and non-monotonic input ordering

---

## [1.3.0] - 2025-01-16

Added support for baseline ISI and hazard function calculations, enabling separate analysis of baseline and post-drug data. Exported baseline ISI and hazard analysis to separate sheets in the Excel output file if baseline data is available. Updated `calculate_hazard_function` to handle baseline ISI data and produce separate hazard analysis for baseline and post-treatment data.

## [1.2.0] - 2025-01-16

Added support for specifying custom labels for filtering, enabling better analysis based on user-defined groupings. Added baseline calculation for delta firing rates, allowing more accurate computation of firing rate changes relative to baseline periods. Added support for organising output files in clearly labelled folders for easier navigation.

## [1.1.0] - 2024-10-31

Added support for hazard function calculation with key metrics.

## [1.0.0] - 2024-10-29

Initial release with spike time filtering, firing rate calculation, and ISI histograms.

# Changelog

## [1.3.0] - 2025-01-16

### Added

- Added support for baseline ISI and hazard function calculations, enabling separate analysis of baseline and post-drug data.
- Exported baseline ISI and hazard analysis to seperate sheets in the Excel output file if baseline data is available.
- Updated `calculate_hazard_function` to handle baseline ISI data and produce separate hazard analysis for baseline and post-treatment data.

## [1.2.0] - 2025-01-16

### Added

- Support for specifying custom labels for filtering, enabling better analysis based on user-defined groupings.
- Baseline calculation for delta firing rates, allowing more accurate computation of firing rate changes relative to baseline periods.
- Support for organizing output files in clearly labeled folders for easier navigation.

## [1.1.0] - 2024-10-31

### Added

- Support for hazard function calculation with key metrics.

## [1.0.0] - 2024-10-29

### Added

- Initial release with spike time filtering, firing rate calculation, and ISI histograms.

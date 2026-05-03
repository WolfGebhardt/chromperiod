# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-05-03

First production-stable release, coinciding with submission of the accompanying manuscript to *Nature* (tracking number 2026-05-13187, May 3, 2026):

> Gebhardt, W. H. (2026). A scaling law of periodic radial geometry organises eukaryotic chromosomes.

### Added

- `CITATION.cff` for GitHub citation widget and reference managers.
- `CHANGELOG.md` (this file).
- README expanded with method rationale, manuscript context, and reproducibility-bundle pointer.
- Setup.py keywords expanded to include "nuclear organization" and "chromosome architecture".

### Changed

- Version bumped from 0.1.0 → 1.0.0 to mark paper-submission state.
- Development status classifier changed from "3 - Alpha" to "5 - Production/Stable".
- Manuscript reference in `chromperiod/__init__.py` docstring updated to the current submitted title.
- Author name normalised to "Wolf Henning Gebhardt" across `setup.py`, `__init__.py`, and `CITATION.cff`.
- Repository URL casing normalised to `WolfGebhardt/chromperiod` to match the canonical GitHub handle.

### Fixed

- Removed nested duplicate package directories (`chromperiod/chromperiod/`, `chromperiod/chromperiod/chromperiod/`) that had accumulated from copy-iteration during the manuscript-bundle build process. The package layout is now a flat `chromperiod/` source tree at repository root.

### Reproducibility

The complete analysis pipeline that produced the results in the manuscript is provided in the supplementary submission bundle:

- 11 Tier-1 reviewer-runnable figure regeneration scripts in `05_code/figure_regen/`
- 7 rebuttal-prep computational tests in `05_code/rebuttal_prep/`
- Per-cell-line phase maps for K562, GM12878, HCT116, MCF-7, IMR-90 in hg38 and hg19 in `04_supplementary_data/`
- Cross-species scaling-law table (71 chromosomes, 4 phyla) in `04_supplementary_data/cross_species_scaling_m2_recomputed.csv`

## [0.1.0] - 2026-01

Initial public release. Consecutive-peak CWT pipeline with Paul m=2 wavelet, AR(1) red-noise null, IAAFT surrogate null, basic phase classification, and Hi-C-eigenvector concordance. Tested on K562, GM12878, MCF-7 DNase-seq.

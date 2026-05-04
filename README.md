# chromperiod

**Detecting periodic radial geometry of eukaryotic chromosomes from accessibility data.**

> **Patent notice.** The methods implemented in this repository are the subject of a Swiss patent application filed 4 May 2026 by Wolf Henning Gebhardt (sole inventor). PCT international application is planned within the 12-month priority window. This software is provided under the **chromperiod Research and Non-Commercial Use License** (see [LICENSE](LICENSE)). Use is permitted free of charge for personal, academic, and non-commercial research purposes. **Commercial use** — including but not limited to clinical diagnostic products, drug-discovery pipelines, cell- or gene-therapy quality-control systems, biomarker-based tests, software-as-a-service offerings, or paid analytical services — **requires a separate written license** from the patent holder. See [PATENT_NOTICE.md](PATENT_NOTICE.md) for the complete patent disclosure. Commercial licensing: w.gebhardt@protonmail.com.

`chromperiod` is a Python package implementing the consecutive-peak continuous wavelet transform (CWT) analysis pipeline that reveals a conserved supra-compartment radial fold of eukaryotic chromosomes. The method takes standard DNase-seq, ATAC-seq, or H3K4me3 ChIP-seq narrowPeak files as input and returns the dominant period, sig95 spectral concentration, and per-bin wavelet phase, with no requirement for chromosome conformation capture (Hi-C) data.

The central observation, established across 71 chromosomes in human, mouse, chicken, and *Drosophila*, is that wavelength scales with chromosome length as **T ∝ L^0.83**, and that each chromosome traces approximately **N ≈ 5** radial excursions between nuclear speckles and the lamina, regardless of size. Wavelet phase predicts radial nuclear position by TSA-seq with R² = 0.90, and the relationship is preserved at single-cell resolution in 240/240 mouse cortical neurons (Dip-C), 14/14 GM12878 nuclei (Dip-C 3D structures), and 1,787 IMR-90 nuclei (MERFISH chromatin tracing).

## Installation

```bash
pip install chromperiod
```

Or from source:

```bash
git clone https://github.com/WolfGebhardt/chromperiod
cd chromperiod
pip install -e .
```

## Quick start

```python
from chromperiod import consecutive_peaks_cwt

# Run CWT on one chromosome from a narrowPeak file
result = consecutive_peaks_cwt('K562_DNase.narrowPeak', chromosome='chrX')
print(result)
# CWTResult(chrom=chrX, n_peaks=8082, dominant_period=28.4 Mbp, sig95=39.4%, ar1=0.150)

# Genome-wide
from chromperiod import run_genome_wide_cwt
results = run_genome_wide_cwt('K562_DNase.narrowPeak')
for r in results:
    print(f'{r.chrom}: T={r.dominant_period:.1f} Mbp, sig95={r.sig95:.1%}')
```

## Method

The pipeline applies a complex Paul wavelet (m=2) to the consecutive sequence of accessibility peak signal values, ordered by genomic position, with inter-peak distances discarded. This coordinate change suppresses the spacing-distance autocorrelation that masks supra-compartment periodicity in standard Hi-C-eigenvector and bin-based analyses, and reveals an orderly periodic pattern in the chromosome's accessible-chromatin surface chain.

For an extended introduction to the rationale (the "consecutive-peak coordinate" change, the surface-vs-bulk reading, the connection to the active-nuclear-compartment / inactive-nuclear-compartment framework), see the manuscript citation below.

## Output

Each `CWTResult` provides:

- `chrom`: chromosome identifier
- `n_peaks`: peak count after filtering
- `dominant_period`: dominant period in megabase pairs (Mbp)
- `sig95`: fraction of global wavelet spectrum power above the AR(1) red-noise null
- `ar1`: lag-1 autocorrelation of the input series
- `cwt_phase`: per-bin wavelet phase (unwrapped)
- `cwt_amplitude`: per-bin wavelet amplitude at the dominant period

## Reproducibility

A self-contained submission bundle accompanies the manuscript. It includes the chromperiod canonical 500-kb pipeline, per-figure data sources, phase maps for K562, GM12878, HCT116, MCF-7, and IMR-90 in hg38 and hg19, and per-figure regeneration scripts. The bundle is provided to editors and reviewers as supplementary material and will be deposited publicly upon acceptance.

This repository contains the core library; the submission bundle is a separate supplement.

## Citation

If you use chromperiod or the consecutive-peak CWT pipeline in your work, please cite:

> Gebhardt, W. H. (2026). **A scaling law of periodic radial geometry organises eukaryotic chromosomes.** Submitted to *Nature*.

A `CITATION.cff` file is provided in the repository root for GitHub's citation widget.

## License

This software is released under the **chromperiod Research and Non-Commercial Use License v1.0** (see [LICENSE](LICENSE)). Permitted Use — free of charge — covers personal, academic, and non-commercial research use, plus a 90-day commercial-evaluation window.

**Commercial use is not permitted under the LICENSE alone** and requires a separate written license from the patent holder. Categories of commercial use include, but are not limited to: clinical diagnostic products, drug-discovery pipelines, cell- or gene-therapy quality-control systems, biomarker-based tests, software-as-a-service offerings, paid analytical services, and use in connection with regulated healthcare services or medical decision-making.

The methods implemented are the subject of a Swiss patent application filed 4 May 2026 (Wolf Henning Gebhardt, sole inventor); see [PATENT_NOTICE.md](PATENT_NOTICE.md) for the complete patent disclosure. Commercial licensing inquiries: **w.gebhardt@protonmail.com**.

## Author

Wolf Henning Gebhardt, Independent Researcher, Bad Homburg, Germany.
ORCID: [0000-0003-2091-1437](https://orcid.org/0000-0003-2091-1437)
Correspondence: w.gebhardt@protonmail.com

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a full version history.

# chromperiod: Detecting periodic chromatin organization from accessibility data

`chromperiod` is a Python package implementing the consecutive-peaks continuous wavelet transform (CWT) analysis pipeline for detecting periodic chromatin accessibility patterns from standard DNase-seq, ATAC-seq, or ChIP-seq narrowPeak files — without requiring chromosome conformation capture (Hi-C) data.

The method reveals that A/B chromatin compartments are arranged with a characteristic periodicity at the multi-megabase scale, conserved from *Drosophila* to human. This periodic organization is detectable from widely available accessibility data and corresponds to Hi-C A/B compartment identity with high concordance (Fisher's OR = 10.88 on human chrX in MCF-7 cells).

## Installation

```bash
pip install chromperiod
```

Or from source:

```bash
git clone https://github.com/wolfgebhardt/chromperiod
cd chromperiod
pip install -e .
```

## Quickstart

```python
from chromperiod import consecutive_peaks_cwt
from chromperiod.plotting import plot_scalogram, plot_gws

# Run CWT on chrX from a narrowPeak file
result = consecutive_peaks_cwt('peaks.narrowPeak', chromosome='chrX')
print(result)
# CWTResult(chrom=chrX, n_peaks=8082, dominant_period=28.4 Mbp, sig95=39.4%, ar1=0.150)

# Plot scalogram and GWS
fig, ax = plot_scalogram(result, period_units='mbp')
fig.savefig('scalogram_chrX.png', dpi=150, bbox_inches='tight')

fig2, ax2 = plot_gws(result, period_units='mbp')
fig2.savefig('gws_chrX.png', dpi=150, bbox_inches='tight')
```

## Method

The consecutive-peaks CWT treats the ordered sequence of chromatin accessibility peak scores as a time series and applies the continuous wavelet transform (CWT) using the Paul wavelet of order m=4, following Torrence & Compo (1998). Key features:

- **Coordinate-free representation**: peaks are indexed by position in the genome, not by genomic coordinate. This removes the dominant source of autocorrelation (clustering of accessible sites in gene-rich regions) and exposes periodic fluctuations in chromatin state.
- **Paul wavelet m=4**: provides good time-frequency localization with a well-defined Fourier period conversion factor (λ = 4π/(2m+1)).
- **AR1 red-noise significance**: analytical significance threshold based on the chi²(2) distribution (T&C eq. 18), with Monte Carlo permutation testing for non-parametric validation.
- **No scale division**: power is normalized as |W_n(s)|²/σ² without dividing by scale, preserving the relative amplitude of oscillations at different periods.
- **GWS-peak dominant period**: the dominant period is defined as the argmax of the global wavelet spectrum (time-averaged power outside the cone of influence).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `peaks_file` | required | Path to narrowPeak/BED file or DataFrame |
| `chromosome` | `None` | Chromosome to analyze (e.g. `'chrX'`). `None` = all peaks |
| `wavelet` | `'paul'` | Wavelet type: `'paul'`, `'morlet'`, or `'dog'` |
| `order` | `4` | Wavelet order (m=4 for Paul, ω₀=6 for Morlet, m=2 for DOG) |
| `n_scales` | `80` | Number of log-spaced scales |
| `period_min` | `10` | Minimum period in peak-index units |
| `period_max` | `7000` | Maximum period in peak-index units |
| `significance_level` | `0.95` | AR1 significance level |
| `signal_column` | `'signalValue'` | Column name or index for the analysis signal |

## Output: CWTResult

| Attribute | Description |
|-----------|-------------|
| `.power` | Wavelet power matrix (n_scales × n_peaks) |
| `.significance` | Boolean significance mask |
| `.gws` | Global wavelet spectrum |
| `.periods` | Period array in peak-index units |
| `.periods_mbp` | Period array in Mbp (if spacing available) |
| `.dominant_period` | GWS-peak dominant period (peak-index units) |
| `.dominant_period_mbp` | GWS-peak dominant period in Mbp |
| `.sig95` | Fraction of significant area outside COI |
| `.ar1_alpha` | Lag-1 autocorrelation |
| `.coi` | Cone of influence array |
| `.coi_frac` | COI-accessible fraction |

## Example data

See `examples/example_data/README.md` for instructions to download the MCF-7 chrX DNase-seq data (ENCODE file ENCFF250GOB) used in the manuscript.

## Citation

If you use `chromperiod` in your research, please cite:

> Gebhardt WH (2026) Consecutive chromatin accessibility peaks reveal periodic compartmental organization of eukaryotic chromosomes. *bioRxiv* [DOI to be added]

and the underlying wavelet method:

> Torrence C, Compo GP (1998) A practical guide to wavelet analysis. *Bull Am Meteorol Soc* 79(1):61–78.

## License

MIT License. Copyright (c) 2026 Wolf H. Gebhardt.

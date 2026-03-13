# Example Data

## MCF-7 chrX DNase-seq (ENCODE ENCFF250GOB)

The primary example dataset used in the manuscript is the MCF-7 breast cancer cell
DNase-seq narrowPeak file from ENCODE:

- **Experiment**: ENCSR000EPH
- **File accession**: ENCFF250GOB
- **Assembly**: GRCh38
- **Format**: narrowPeak (BED6+4)
- **Peaks**: 214,851 genome-wide; 8,082 on chrX

### Download

```bash
wget https://www.encodeproject.org/files/ENCFF250GOB/@@download/ENCFF250GOB.bed.gz
gunzip ENCFF250GOB.bed.gz
mv ENCFF250GOB.bed ENCFF250GOB.narrowPeak
```

Or using the ENCODE portal:
1. Go to https://www.encodeproject.org/files/ENCFF250GOB/
2. Click "Download" to get the narrowPeak file.

### Usage

```python
from chromperiod import consecutive_peaks_cwt

result = consecutive_peaks_cwt('ENCFF250GOB.narrowPeak', chromosome='chrX')
print(result)
# Expected output:
# CWTResult(chrom=chrX, n_peaks=8082, dominant_period=28.4 Mbp, sig95=39.4%, ar1=0.150)
```

## GM12878 DNase-seq (ENCODE ENCFF762CRQ)

A second example dataset from the GM12878 B-lymphoblastoid cell line:

- **Experiment**: ENCSR000EMT
- **File accession**: ENCFF762CRQ
- **Assembly**: GRCh38
- **Peaks**: 76,116 genome-wide

```bash
wget https://www.encodeproject.org/files/ENCFF762CRQ/@@download/ENCFF762CRQ.bed.gz
gunzip ENCFF762CRQ.bed.gz
```

## Mouse heart DNase-seq (ENCODE ENCFF391DTE)

Mouse heart tissue data in Hotspot3 BED format:

- **Experiment**: ENCSR000CBF
- **File accession**: ENCFF391DTE
- **Assembly**: mm10
- **Format**: BED hotspot (max_density in column 5)

```bash
wget https://www.encodeproject.org/files/ENCFF391DTE/@@download/ENCFF391DTE.bed.gz
gunzip ENCFF391DTE.bed.gz
```

```python
from chromperiod import consecutive_peaks_cwt

# Use max_density signal column (Hotspot3 format)
result = consecutive_peaks_cwt(
    'ENCFF391DTE.bed',
    chromosome='chrX',
    signal_column='max_density',
)
```

# Patent Notice

The methods implemented in this repository are the subject of pending patent protection.

## Filing record

- **Swiss patent application** filed 4 May 2026 with the Swiss Federal Institute of Intellectual Property (IPI / IGE), Stauffacherstrasse 65/59 g, CH-3003 Bern.
- **Sole inventor and applicant:** Wolf Henning Gebhardt.
- **Application number:** to be assigned (formal acknowledgement pending; e-filing receipt confirmed by IGE mail system on 4 May 2026).
- **Title:** *Methods and systems for detecting periodic supra-compartment chromosome organisation from chromatin accessibility data and for clinical, oncology, and biomarker applications thereof.*
- **Service address in Switzerland (Art. 13 PatG):** Prof. Dr. Peter Seitz, Bergstrasse 35, 8902 Urdorf, Switzerland.

## Planned international filings

- **PCT international application** within the 12-month Paris Convention priority window (deadline: ~4 May 2027), claiming priority from the Swiss filing.
- **National-phase entry** at month 30 (deadline: ~4 November 2028) anticipated in at least: European Patent Office, United States Patent and Trademark Office, and selected additional jurisdictions.

## What is covered

The patent application describes methods for analysing chromatin accessibility data (DNase-seq, ATAC-seq, ChIP-seq narrowPeak, or equivalent) using a continuous wavelet transform applied to a consecutive-peak coordinate representation, including but not limited to:

- The foundational method of representing accessibility peaks as a consecutive-index sequence and applying a complex wavelet transform (Paul, Morlet, or equivalent) to recover the supra-compartment periodic phase.
- Aggregation of per-peak phase values to genomic bins by amplitude-weighted circular mean.
- AR(1) red-noise null testing of CWT power and computation of the sig95 spectral concentration metric.
- Application of the recovered phase to: clinical interpretation of copy-number variants and structural variants; oncology somatic-variant phase-matching; laminopathy biomarker assays; cell- and gene-therapy chromosomal-integrity quality control; drug-target prioritisation; and cross-domain non-genomic signal-processing applications.

This software repository (`chromperiod`) implements the **foundational** consecutive-peak CWT pipeline. It does not implement the clinical, oncology, biomarker, cell-therapy, drug-target, or cross-domain application embodiments, which are the subject of additional claims in the patent application.

## What this means for users

Users who are individuals, non-profit academic or research institutions, or government agencies, and who use this software solely for non-commercial research or teaching as defined in Section 1 of the LICENSE, receive a personal patent licence sufficient to perform their use. **No commercial patent rights are granted by use of this software.**

If you intend to use this software, the methods it implements, or any derivative thereof, in or for the benefit of any commercial product, service, pipeline, or for-profit entity — including but not limited to clinical diagnostics, cell- or gene-therapy quality control, drug-discovery pipelines, software-as-a-service offerings, or paid analytical services — you must obtain a separate written commercial licence from the patent holder before use.

## Commercial licensing inquiries

Wolf Henning Gebhardt
Independent Researcher, Bad Homburg, Germany
Email: w.gebhardt@protonmail.com

Please include in your inquiry: your organisation, the intended use, the expected scope (research evaluation only, internal pipeline, embedded in a commercial product, etc.), and the timeline. Initial response is typically within 5 working days.

## Updates

This notice will be updated when:

- The Swiss application number is formally assigned (expected within 1–2 weeks of filing).
- The PCT application is filed and an international application number is assigned.
- Any national-phase application is filed.
- Any patent grants.

The most recent version of this notice is the one in this repository's `main` branch.

## Independence of this notice from the Nature manuscript

This patent notice is independent of the manuscript *A scaling law of periodic radial geometry organises eukaryotic chromosomes* (submitted to *Nature*, 3 May 2026). The manuscript discloses the scientific findings; this notice describes the legal status of the methods. Both are by the same author, but the legal status of the methods is governed by the patent filings, not by the manuscript or by any preprint deposit thereof.

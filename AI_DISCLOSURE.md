# AI Tool Disclosure

This document describes the use of artificial-intelligence tools, including large language models (LLMs), in the development of the `chromperiod` software and the accompanying scientific manuscript.

## Inventorship and authorship

The consecutive-peak continuous wavelet transform method that underlies this software, and the scientific findings reported in the accompanying manuscript, were conceived and developed by Wolf Henning Gebhardt. Conception traces to the inventor's independent work between approximately 2014 and 2017 at the Institute of Molecular Biology (IMB), Mainz, and predates the broad availability of large-language-model AI tools (ChatGPT public release November 2022; comparable foundation models available 2023+). Contemporaneous notes, software files, and correspondence from that period are retained by the inventor as evidence of independent prior conception. The 2026 manuscript and patent filing develop the initial observation into a quantitative, cross-species, mechanism-linked phenomenon.

**Wolf Henning Gebhardt is the sole inventor of the methods claimed in the Swiss patent application filed 4 May 2026 (and any subsequent PCT and national-phase filings tracing priority to that application). Wolf Henning Gebhardt is the sole author of the accompanying manuscript.**

## How AI tools have been used

Between approximately 2024 and 2026, AI tools — including but not limited to Anthropic's Claude family of models — were used as research assistants in the development of this software and manuscript. The categories of use include:

- **Software implementation and review**: drafting, refactoring, debugging, and reviewing portions of the Python source code in `chromperiod/`. All code changes were reviewed, tested, and approved by the human author before commit.
- **Statistical analysis assistance**: drafting analysis scripts, computing summary statistics, generating diagnostic plots, and running pre-committed disposition tests. All statistical claims in the manuscript are derived from human-reviewed analyses run on real data; provenance is documented in the supplementary submission bundle (`05_code/figure_regen/` and `05_code/rebuttal_prep/`).
- **Manuscript drafting and editing**: generating draft text, polishing prose, sweeping for consistency, drafting figure captions, and producing reviewer-response material. All scientific claims in the manuscript reflect the human author's understanding and judgement.
- **Research tooling**: drafting BIOMNI prompts, organising the supplementary bundle, building reproducibility checks, and maintaining the project change log.

AI tools have **not** been used to:

- Conceive the underlying scientific method (the consecutive-peak CWT representation, the AR(1) red-noise null framework, the cross-species scaling-law analysis, the per-bin amplitude-weighted phase aggregation, or any of the application embodiments described in the patent).
- Author or sign the patent application.
- Independently make scientific claims that were not reviewed and approved by the human author.
- Replace human judgement on inventorship, scientific interpretation, or publication decisions.

## Compatibility with editorial and patent-office policies

This disclosure is consistent with:

- The **Nature** editorial policy on the use of large language models in research articles (2023), which permits LLM use as a tool subject to disclosure but does not permit LLM authorship.
- The **2023 USPTO Inventorship Guidance for AI-Assisted Inventions** (88 Fed. Reg. 32168, May 2023, and subsequent guidance), under which a natural-person inventor remains validly named when the human made a "significant contribution" to the conception, with AI tools used for implementation, optimisation, or analysis.
- The **European Patent Office** position established in EBoA G 1/19 and subsequent guidance, under which inventorship is reserved for natural persons.
- The **Swiss patent law** position (Art. 3 PatG), under which a natural-person inventor is required.

## Updates

This disclosure will be updated if:

- The category or extent of AI-tool use changes materially.
- A formal patent grant requires a more specific disclosure.
- A journal of publication requires a different format of disclosure.

The most recent version is the one in this repository's `main` branch.

## Contact

Questions about this disclosure: w.gebhardt@protonmail.com

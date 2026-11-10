# Identity Ladder: CVPR 2027 working draft

Started 8 November 2026 from the ISBI draft and the saved identity-ladder experiments. This is a manuscript development folder, not a submission-ready paper. The ISBI source and existing experiment outputs are preserved.

Working title: **Identity Ladder: Measuring Identity, Symmetry, and Variability in Human Images**.

**Current research direction, updated 8 November:** the user specifies a cross-anatomy study of identity, bilateral symmetry and variability, comparison of image distances, and inpainting evaluation. The planned main figure is a spatial difference atlas across brain, eye, fundus, face and pupil-related views. See [`research_brief.md`](research_brief.md) for the current scope, figure specification and experiment plan. The expanded `main.tex`/`main.pdf` now include bilateral-comparison methods, saved GRAPE/UPOL/UTIRIS representation comparisons, and a clearly labeled atlas/inpainting extension. The new atlas and reconstruction experiments are not reported as completed.

The contribution is an evaluation protocol for asking whether an image distance responds to target-specific information, nuisance variation, or population typicality. The percentile transformation is established statistical machinery, not a new metric. The ladder describes controlled comparison conditions and testable partial orders, not a universal biological ordering.

## Files

- `outline.md`: claim-first outline, written before the manuscript.
- `research_brief.md`: latest user-directed cross-anatomy scope, difference-atlas design, and proposed experiments; distinguishes planned work from saved evidence.
- `ARTIFACTS.md`: checked inventory of available artifacts, measured evidence, MRI preparation and missing experiments.
- `inventory.py`, `artifact_inventory.json`: reproducible public-file counts and hashes; no participant data access.
- `main.tex`: anonymous English manuscript draft in the official CVPR 2027 style.
- `refs.bib`: selected references copied from the ISBI bibliography, with any additional records obtained from Crossref.
- `evidence.json`: aggregate values and source hashes used in the manuscript.
- `prepare.py`: prepares figures, tables, references and the evidence manifest from existing results; does not run experiments or read participant images.
- `development.md`: unresolved experiments and editorial decisions.
- `citation_report.md`: automatic bibliography verification, when completed.
- `verification.md`: build, provenance and visual-review record.

## Evidence boundary

The draft reports existing synthetic experiments and exploratory aggregate GRAPE/UTIRIS pilots. It does not claim a validated scalar identity metric, a new state-of-the-art reconstruction method, a trained-segmenter result, or a population-level real-MRI validation. A high calibrated rank against unrelated donors is not proof of identity preservation. Synthetic seed ranges are not confidence intervals.

## Venue checks, 8 November 2026

The official [call for papers](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) gives enrollment on 10 November and submission on 16 November, both 23:59 AoE. In Dubai these are **11 November and 17 November at 15:59**, respectively. The [author guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines) require the official template, anonymity, at most eight content pages, and an LLM Usage Statement. Additional pages may contain references and that statement.

**Confirmed by the user on 8 November: the ISBI paper is submitted and under review.** This development draft reuses mean/shrinkage and label analyses that appear in the ISBI source. It is not a submission-eligible independent contribution as currently scoped. Before enrollment, resolve overlap under the current CVPR dual-submission policy and disclose related submissions as required. A change of title or prose does not establish a distinct contribution. No withdrawal of ISBI has been requested or performed. See `development.md` for the scope matrix.

Official template: [CVPR 2027 author kit](https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip). Any template files included here are third-party material under their original terms.

## Build in WSL Ubuntu

From the project root, run `python3 paper/cvpr2027/prepare.py`, then `cd paper/cvpr2027 && tectonic main.tex` (or use a TeX installation with `latexmk -pdf main.tex`). The preparation script needs NumPy and Matplotlib. The TeX engine may download standard packages.

The expanded development build is **11 pages: ten content pages and one page containing the LLM Usage Statement and references**. It is a working reading copy, not an eight-page submission version. Three tables are generated from saved aggregates; the new representation table contains 18 score cells. All ten references retain the earlier DOI-verification record. No new experiments or data acquisition were performed. See `verification.md` for the current build and provenance checks.

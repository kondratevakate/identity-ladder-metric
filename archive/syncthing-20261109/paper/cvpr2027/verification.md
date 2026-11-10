# Draft verification, 8 November 2026

## Current expanded development build

The rebuilt PDF has 11 pages: ten content pages and one page for the LLM Usage Statement and references. This is the expanded reading copy, not a claim of compliance with the submission page limit. No font, margin or style changes were used to fit text.

The new table has nine representation rows and 18 score cells generated directly from `grape_calibration.json` and `iris_calibration.json`. Preparation verifies finite scores, bounds and common recorded unit/pair counts within each dataset. An additional check confirms that all 18 rounded values occur in the manuscript and that all 33 inventoried result JSON/CSV and ISBI source/bibliography files retain their recorded SHA-256 values. Six result JSONs now contribute to the manuscript evidence manifest. These are provenance/build checks, not a rerun of the research experiments or historical test suite.

The TeX source grew from approximately 3,971 to 5,583 English word-like tokens, a net increase of 1,612 by the same regular-expression count. This includes command/label tokens and is an approximate writing-volume comparison, not a formal abstract or submission word count.

All 11 pages were rendered with Poppler and visually reviewed as contact sheets. The new table fits within the page and its pair/unit columns remain legible. The final log contains no overfull boxes, undefined citations/references or missing characters. Extracted page text contains no replacement characters. The previously recorded upstream `lineno.sty` UTF-8 warning remains. Ten references are still cited and match the bibliography; no new bibliographic entries were introduced and the earlier DOI verification was not rerun.

## Earlier eight-page build, retained as history

The development PDF builds with Tectonic 0.17.0 in WSL Ubuntu using the official CVPR 2027 style. It contains seven content pages and one page for the LLM Usage Statement and references. No margins, font sizes or conference style rules were changed to achieve that count.

All eight pages were rendered with Poppler and visually inspected, including the two tables and five figures. The final TeX log contains no undefined citations, undefined references, missing glyphs or overfull boxes. Tectonic reports an invalid UTF-8 sequence in the upstream `lineno.sty` package; no missing character is present in the document output.

All ten bibliography entries were verified by DOI against Crossref by `bib-verify`. The new RETFound record was generated from cached Crossref metadata; ampersand and apostrophe escaping were subsequently normalized for TeX without changing bibliographic identity. The preparation script checks that every bibliography entry is cited and that result hashes match the evidence manifest.

Numeric table cells are generated directly from saved aggregate JSON files. Other quoted synthetic sensitivity findings were checked against the saved reports. Historical test-suite counts in the project handoff were not rerun and are not claimed as tests of this manuscript. No participant images were read and no new experiments were run.

The original ISBI manuscript and bibliography were preserved. Their SHA-256 values are recorded in `evidence.json`. There is no `.git` directory in this checkout; no commit or push is claimed.

This is a working draft, not a submission-ready independent CVPR paper. ISBI is under review, as confirmed by the user. The overlap matrix and unperformed experiments in `development.md` remain material to the final scope.

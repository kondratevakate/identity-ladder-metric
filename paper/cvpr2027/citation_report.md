# Bibliography verification and manual resolutions

Checked 8 November and refreshed 9 November 2026. Every entry was passed through the installed bib-verify script against Crossref/arXiv. The unmodified automatic output is retained in `citation_report.json`; it is not overwritten with manual green labels. Publisher metadata responses are cached locally by `fetch_bibliography.py`. The current bibliography contains 21 entries.

Eighteen entries resolve without automatic flags. Three require the following primary-source resolutions:

| Key | Automatic flag | Primary-source check and decision |
|---|---|---|
| dobes2004 | Author overlap 50%; escalated to substituted | Crossref for DOI 10.1078/0030-4026-00388 returns Dobeš, Machala, Tichavský, Pospíšil, matching the bibliography's four authors, title, year and DOI. The checker's surname normalizer removes non-ASCII letters without decoding TeX accents. `Dobe\v{s}` versus `Dobeš` and `Pospíšil` cause a false mismatch. Retain correct renderable TeX names and the original checker flag. |
| betti2023 | needs_review: arXiv year 2022 versus conference year 2023 | The [official PMLR proceedings](https://proceedings.mlr.press/v202/stucki23a.html) explicitly give ICML 2023, volume 202, pages 32698–32727, with the same five authors and title. arXiv 2211.15272 is the earlier preprint. Retain the published conference year. |
| fwd2025 | substituted: arXiv year 2023 versus conference year 2025 | The [official ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6605c7d1c1ae1521b328d036df5be1ae-Abstract-Conference.html) and its paper give ICLR 2025, the same title and authors; Hilde/Hildegard Kuehne is a name-form variation. arXiv 2312.15289 first appeared in 2023. Retain the published conference year; this is not a different paper. |

Earlier checks caught and corrected a DOI underscore escaped for TeX, the online-versus-print year for Rothe et al., and a missing font glyph in the rendered Dobeš name. These were corrected in the metadata-generation code rather than in the checker. Brain-ID is cited as the ECCV work, with the publisher's 2025 proceedings publication year retained.

Bibliographic existence does not validate a manuscript claim. CARE, LMSB and Brain-ID were read as primary sources for their stated objectives, and the manuscript does not claim a completed empirical comparison against them. TM/lDDT references identify the original protein-comparison ideas; the manuscript explicitly defines its intensity adaptations. Betti matching is cited in its original segmentation setting, separately from the unavailable full P3 intensity results.

Verification is complete with documented manual resolution of these three tool limitations. The automatic report alone is not an all-pass report.

The BraTS task paper was added for the competition-oriented revision. Its arXiv identifier, title and leading authors were checked against the primary record; bib-verify resolves it by arXiv with no field mismatch. The entry uses first posting year 2023 and explicitly notes the version-3 revision in September 2024. The 2026 task/evaluator sources are separately recorded in decision_coverage.json; no announced 2027 format is inferred from them.

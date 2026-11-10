# Identity Ladder CVPR 2027 first draft

This is a reading draft assembled from the available Git snapshots, not a submission-ready claim of completed cross-domain reconstruction experiments. The primary text is `main.tex` / `main.pdf`; the separate supplement is `supplementary.tex` / `supplementary.pdf`. The two canonical reading PDFs are versioned on this author-review branch. Build caches and temporary renderings are ignored.

Latest author clarification, 9 November: the scope is inpainting broadly and a proposed audit of conditional generation across population slices. BraTS remains one application. The title is **Identity Ladder: Population References for Inpainting and Conditional Generation**. `POPULATION_AUDIT_FRAMING.md`, `population_audit.json`, `POPULATION_AUDIT_COVERAGE.md` and `population_audit_manifest.json` separate measured donor/metric evidence from proposed generator subgroup profiles. The earlier BraTS decision map remains an application-specific history. No generator experiment or new metric run was performed.

The main figure is `fig/identity_ladder_provenance_v2.{pdf,png}`: a white editorial layout with panel a (cohort provenance), panel b (one primary heatmap dataset per modality), restrained leader lines and larger maps. QTIM, CelebA, GRAPE P8 and UTIRIS NIR P2 are retained. A shared exterior display mask removes the MRI background and applies a common crop; internal dark anatomy and retained source colors are preserved. This is a display operation, not a measurement mask or a new statistic. The numerical scale stays 0 / 1 / 2.4; V remains row-scaled. `provenance_infographic_v2.json` records sources, crops, counts, illustration provenance and the pixel-preservation assertion. The synthetic bust is decorative. Secondary datasets remain in the supplementary atlas.

The working branch is `codex/cvpr2027-manuscript` on the owner's public fork, based on `80a094c7c3cab36d51d0e290b0634cca0ccf8d36` after incorporating 33 new commits. The related ISBI source is preserved. `REPOSITORY_UPDATE.md` describes the update and consolidation.

## Rebuild in WSL Ubuntu

Run from the repository root with Python 3.12, the packages in `requirements-paper.txt`, Tectonic 0.17 and Poppler's `pdftotext` installed:

```bash
python -m pip install -r paper/cvpr2027/requirements-paper.txt
python paper/cvpr2027/build.py --tectonic /path/to/tectonic --render
```

The preparation stage reads existing aggregate results only. `source_snapshot.py` pins the manuscript evidence to `a48523d` directly through Git objects, so a build cannot silently substitute newer P3/face results. New exports are available in the checkout and await a separately reviewed scientific revision. It creates every table, copies source figures, imports two immutable bilateral-branch JSON blobs, and assembles the atlas using the saved Gaussian brain panels. It does not run a metric experiment, inspect participant images, or reconstruct private maps. The bilateral commit `82ed7fe8a52e557854cf4cd2d02eb47d5803d7cb` must be available in the clone; fetch `origin/donor-ladder-bilateral` if necessary.

`src/ladder_maps_all_figure.py` is the imported historical private-data generator, with a provenance notice. It is not called by the paper build; its original footer must not be mistaken for the current Gaussian atlas specification.

`evidence_manifest.json` and `atlas_provenance.json` record source hashes, producing commits, generator hashes and transformations. Font/metadata creation dates can change output PDF bytes across builds; numerical source identity is checked independently. The complete source atlas retains fundus P8/P9 and iris P2; the main provenance figure selects the four primary rows described above. The full iris P3 export has arrived through the fork, and the base face pipeline/protocol was recovered from the Syncthing folder. They are preserved; the text and figures retain the explicitly frozen evidence version pending integration review.

Bibliography refresh is separate from the offline manuscript preparation: `fetch_bibliography.py` retrieves Crossref metadata and retains the manually checked official-proceedings entries. Every entry was checked using the installed bib-verify script. `citation_report.json` preserves its automatic verdicts and `citation_report.md` resolves three known false flags against primary sources. The build reads this report; it does not silently claim to have rerun an external verifier.

## Review records

`ASSEMBLY_PLAN.md` defines the iteration cycle. `TEXT_DECISIONS.md` is its append-only decision history. `MATERIALS_COVERAGE.md` and `materials_inventory.json` map the existing materials to manuscript roles. `SCIENTIFIC_REVIEW.md`, `REPRODUCTION_AUDIT.md`, `METADATA_REVIEW.md` and `CVPR_AUDIT.md` record separate scientific, reproduction-evidence, title/abstract and venue checks. These are author-facing development records, not automatically part of the anonymous submission.

Full repository tests were run in the WSL environment: **417 passed, 7 skipped, 1 failed** (9 November; received face code included). The single remaining failure is `tests/test_real_cases.py::test_guard_refuses_changed_inputs_and_missing_amendment`: the frozen real-cases guard requires Python 3.11, while this host's test environment is Python 3.12. No experiment guard was relaxed to make the suite green. The build and table/source checks are separate from this historical environment constraint; successful tests would not establish independent experimental reproduction.

Before submission, the author still needs to settle ISBI overlap, dataset-specific consent/ethics and portal eligibility, integrate the missing safe exports, and resolve experiment QC gates. The author explicitly authorized this branch publication and consolidation on 9 November. Submission and upstream publication remain separate actions. No individual faces, eye images, subject identifiers or restricted UTIRIS images were added to this draft.

Earlier loose review ZIPs and superseded rendered variants are preserved in the branch history and removed from the final working tree. Use the canonical source/PDF paths above.

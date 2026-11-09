# Open-repository readiness review

Reviewed 9 November 2026 by Codex. Branch: `codex/cvpr2027-manuscript`; reviewed commit: `f7d2580e50cd69079bfc80f1ac24febacf441f64`. The GitHub connector confirms that the owner's fork is already **public** and its default branch is `master`. This review concerns readiness for a coherent, reusable research release, not permission to make that already-public repository visible.

**Verdict: not ready for a general research release yet.** The useful core is public code, saved aggregate controls, frozen protocols, explicit provisional/withdrawn evidence and a buildable review manuscript. The blocking work is primarily publication-policy consistency, a scientifically accurate landing page and a documented reproducible environment. A final best metric, a universal identity score or a completed generator benchmark is not required to release an honestly scoped work in progress.

## Scope and actual checks

The current tree inventory covers **694 tracked files (81,151,834 bytes)** across code, tests, results, protocols, papers, figures, cloud helpers and recovered history. The history scan covered **2358 reachable objects, 1308 blobs and 1203 UTF-8 texts/archive members**, including ten ZIP blobs, over all locally fetched refs. No blob was skipped for the 25 MB size cap. Recognizable OpenAI, GitHub, Hugging Face, Slack, AWS and private-key formats produced **zero matches**. The narrow JSON identifier-key scan also produced zero candidates. These are limited scans, not proof against unknown/encoded credentials, identifiers embedded in figures, or all privacy risks. No gitleaks/entropy scanner was run.

Full tests in WSL/Python 3.12 after cleanup: **417 passed, 7 skipped, 1 failed**. The failure is the existing Python-3.11 frozen guard in `test_guard_refuses_changed_inputs_and_missing_amendment`; it was not relaxed. This does not establish that a fresh Python 3.11 installation passes. The manuscript build/render succeeds with eight content pages, ten main pages total and fifteen supplementary pages. No metric experiment, registration run, cloud job or generative-model benchmark was launched.

The figure review sampled the main infographic, recovered v6f face maps, the older photo-like face ladder and rebuilt eye panel against their source/provenance documents. The last two are documented synthetic illustrations. A realistic face was not treated as a confirmed participant-data leak merely because it looked photographic. The real QTIM example in the landing README is explicitly described as real and CC0, creating a policy-consistency question rather than a hidden-data discovery. Raw dataset extensions and the inspected JSON keys did not reveal a raw dataset export; all PDF/PNG content has not been independently rights-cleared or de-identification certified.

## Findings requiring resolution

### OR-01 · P1: Data and figure publication policy conflicts with the blanket licence

**Location:** `README.md:174-197; docs/DATASETS.md:17-19,49-64; docs/PROTOCOL_donor_ladder_face.md:16`.

The README permits a real QTIM example and applies CC BY to figures/results; the registry says real-case images never enter Git. CelebA source terms restrict research use and derived data. Aggregate/mean status alone does not settle the applicable permission.

**Smallest useful remedy:** Create a per-artifact dataset/licence/publication matrix; reconcile the QTIM exception; confirm which derived assets may carry the stated licence. Preserve provenance and do not re-host restricted raw data.

### OR-02 · P1: Public entry-point claims are stronger than the measured evidence

**Location:** `README.md:12-14,92-116; paper/cvpr2027/sec/5_synthesis.tex:22-38; results/real_cases.json:2`.

The README moves from a variance-component model to unconditional mean/sibling and Dice conclusions. The paper correctly limits interpretation, registration remains provisional, and older c/rho/label results are withdrawn.

**Smallest useful remedy:** Rewrite the landing README around controlled metric/reference audits and current evidence status; label theoretical assumptions and synthetic mechanisms; link the frozen paper and withdrawn-results warning.

### OR-03 · P1: A clean experiment environment is not specified or verified

**Location:** `README.md:154-172; src/real_cases.py:132-138; paper/cvpr2027/requirements-paper.txt`.

Only manuscript requirements are installable. No root dependency/environment manifest exists. The advertised experiment runtime is Python 3.11; the executed 3.12 suite has one expected guard failure.

**Smallest useful remedy:** Provide separate pinned experiment, paper and optional-modality environments; test installation in a fresh Python 3.11 environment without weakening frozen guards.

### OR-04 · P1: The default test command performs an unmarked model download and uses a Windows cache path

**Location:** `tests/test_donor_ladder_brain_metrics_t23.py:54-61; src/donor_ladder_brain_metrics_t23.py:17,71-80; src/donor_ladder_brain.py:24-26`.

The full suite loaded pretrained AlexNet and produced a roughly 234 MB checkpoint beneath a literal C:\Projects\data directory in WSL. DATA_ROOT redirected the later runs to the ignored cache.

**Smallest useful remedy:** Separate offline unit tests from pretrained-model integration tests, mark the latter, document size/cache/network requirements and configure platform-neutral cache roots.

### OR-05 · P1: The English-only publication requirement is not met by active generated material

**Location:** `results/METRICS_STATUS.md; src/metrics_board.py; inventory.cyrillic_text_files`.

Twenty tracked text files contain Cyrillic, including fifteen outside the explicitly historical Syncthing archive. Current generator strings can reintroduce Russian after a manual report translation.

**Smallest useful remedy:** Translate active publication text and generator strings together; keep original historical/quoted evidence explicitly identified rather than rewriting its content.

### OR-06 · P2: The default branch and the reviewed research branch expose different project states

**Location:** `GitHub repository metadata; origin/master; codex/cvpr2027-manuscript; paper/cvpr2027/source_snapshot.py:12`.

The public fork defaults to master. The reviewed paper lives on the Codex branch and pins a48523d evidence while newer exports are present.

**Smallest useful remedy:** Choose and document a release entry point/tag, branch purpose and evidence manifest; update the default landing documentation through an author-approved fork change.

### OR-07 · P2: Result JSONs contain non-standard NaN literals

**Location:** `results/cross_modality/agreement.json; results/cross_modality/corr_all.json; results/donor_ladder_face*.json`.

Ten current result files contain 334 unquoted NaN constants. Python accepts them; strict JSON consumers need an explicit missingness representation.

**Smallest useful remedy:** Add a versioned strict export using null plus unavailable/undefined/not-run status and a schema. Preserve original historical result bytes and never replace missing values with zero.

### OR-08 · P2: Automation and contributor release metadata are absent

**Location:** `Tracked-tree inventory: .github/workflows/, CITATION.cff, CONTRIBUTING.md, SECURITY.md`.

No CI workflow or citation/contribution/security document was found. Repository security settings and branch protections were not inspected.

**Smallest useful remedy:** Add offline CI, an opt-in integration job, citation metadata, contribution/data-output rules and a private vulnerability-reporting route; verify settings separately.

### OR-09 · P2: Ignore rules do not cover common credential and model-cache locations

**Location:** `.gitignore:1-14; observed Windows-literal cache directory`.

external/ and common build files are ignored, but .env, key files, general data/cache directories and checkpoint extensions are not generally covered. No actual credential was detected by the limited scan.

**Smallest useful remedy:** Add scoped exclusions for local secrets, private data and weights, plus an outgoing-artifact check. Keep explicit exceptions for intentionally versioned public evidence.

### OR-10 · P2: Newest exports need version-specific scientific integration and a unified status view

**Location:** `docs/HANDOFF_iris_metrics.md:29-43; results/iris_ladder_metrics.json:amendment_p3; paper/cvpr2027/source_snapshot.py`.

P3 is now present, H4 superiority is not established, and three H5 readouts mark interval_trusted false. The current paper intentionally uses P2. The metrics-board source does not cover all newer modality results.

**Smallest useful remedy:** Publish an evidence-version/status index and review P3/brain/face/IXI exports before updating claims. Keep point estimates, flagged uncertainty and not-run model experiments separate.

### OR-11 · P2: Some realistic-looking illustrations need provenance at the asset entry point

**Location:** `results/figures/identity_ladder_faces_photo.png; docs/figures_draft.md:44-50; paper/cvpr2027/fig/assets/`.

The older photographic face ladder is documented as fictional ChatGPT imagery, not a participant photo. That explanation is in a separate document; the new bust has a stronger prompt/tool record.

**Smallest useful remedy:** Add an asset manifest and nearby synthetic/mean/template labels for every public visual; distinguish illustrations from measurements and dataset derivatives.

### OR-12 · P2: Runtime and evidence provenance are sensitive to line endings and local paths

**Location:** `results/METRICS_STATUS.json:source_sha256; paper/cvpr2027/README.md; inventory.source_files_with_windows_literals`.

Nineteen source/cloud files include Windows path literals. Regenerating the board changed source byte hashes and CSV line endings without changing measured rows.

**Smallest useful remedy:** Use explicit configuration and record raw-byte and LF-normalized hashes where appropriate; add .gitattributes for new source/output conventions while retaining frozen original evidence.

## Primary-source rights checks

The [official CelebA agreement](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) restricts use to non-commercial research, addresses derived data and restricts further dataset distribution. This supports a rights-review requirement; it does not by itself decide the legal status of every aggregate statistic. Do not infer a blanket unrestricted licence for every derived asset from removal of participant identifiers.

The [UTIRIS maintainer page](https://utiris.wordpress.com/) identifies CIPCE/University of Tehran copyright and requests citation; the inspected page did not provide an open redistribution grant. The [IMDB-WIKI project page](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) describes Internet-crawled celebrity images and metadata. Download availability and a citation request do not document a universal grant to relicense underlying images. Keep source terms distinct from the project's MIT/CC BY declarations. This review does not change dataset permissions or assert that every aggregate is prohibited.

No credential exposure was established. If later review establishes sensitive content, deleting its current path alone does not remove it from Git history; [GitHub's removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository) explains the coordination required. No history rewrite, force push, visibility change, default-branch change or message to a third party was performed.

## Scientific and reproduction interpretation

The new generative-metric tiles compare metrics/reference controls; their name is not evidence of a completed generative-model subgroup benchmark. Keep target strata, donor cohorts and acquisition domains separate. Registration confounding remains visible. A conditional-mean MSE result is not automatically an identity result, a general Dice theorem, a causal demographic disparity or fairness validation.

The full P3 JSON was inspected directly during this review. Its H4 T2 Holm values are 1.0 for the three prespecified LPIPS/Betti readouts: the requested superiority is not established, which does not prove equivalence or a reverse biological claim. Its H5 mean/stranger point ratios are about 19.7 for FID and 415.5 for FWD; three associated records explicitly mark their intervals untrusted. H6's NIR LPIPS-AlexNet order is recorded as repeat 0.188648, look-alike 0.318836, stranger 1.005276, with repeat upper bound 0.220548. These are existing reference-control measurements, not a new rerun or model-output comparison. The manuscript's a48523d/P2 freeze is intentional and should remain explicit until a separate integration review.

`audits/open_repository_claims_20261109.json` applies the installed repro-judge rubric to narrowly stated scientific interpretations. It is a Codex documentary review, not a paid/model judge, not the calibration benchmark and not a numerical release-readiness score. The existing paper reproduction review remains historical and source-version specific.

## Release decision

No confirmed P0 credential/raw-identifier exposure was found in this bounded review. Five P1 findings and seven P2 findings are recorded in the CSV. Resolve the P1 items, retain a clear work-in-progress label, and perform the named clean-environment/data-rights checks before describing the project as a reproducible open research release. New metric experiments are not a prerequisite for these documentation and packaging fixes. The separate registration-QC gate still governs future scientific runs.

See `OPEN_REPOSITORY_RELEASE_PLAN.md` for acceptance criteria, and `audits/open_repository_inventory_20261109.json` for the file/hash inventory and scan limits. Repository settings, branch protections, institutional ethics/consent records, all original private preprocessing and third-party rights were not independently verified.

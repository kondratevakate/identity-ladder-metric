# Text decision log

This append-only log records manuscript decisions, their evidence and checks. A working verdict is not a substitute for source results. Dates refer to manuscript assembly on 8 November 2026.

## Iteration 00: establish the authoritative source

**Inputs:** workstation handoff; cvpr2027-paper at a48523d; the older non-Git Syncthing copy.

**Finding:** the branch contains a manuscript skeleton, while the earlier local draft used a narrower and older evidence set. The Git branch and its protocols are the assembly authority.

**Decision:** clone the fork outside Syncthing, retain the branch, preserve the ISBI source, and write only the paper plus the explicitly requested figure-generator import. Do not overwrite the branch with the old monolithic draft.

**Checks:** verified branch, remote and commit; read AGENTS.md and the reproduction workflow. No push was performed.

## Iteration 01: distinguish current, provisional and absent evidence

**Inputs:** general donor protocol; brain amendments through 21; fundus amendments; face and iris handoffs; saved result JSONs.

**Finding:** brain registration remains pre-QC, current accessible fundus tables are P7, face tier-1 has an approved v6f frame but lacks its base pipeline/protocol, and the latest full iris P3 is absent from Git.

**Decision:** label source-dependent provisional results P, distinguish unavailable from not run and undefined, exclude superseded support-sensitive block searches, and retain newer-protocol specifications without claiming their sweeps finished.

**Scientific consequence:** the text can describe metric profiles but cannot infer a validated biological identity ordering. Historical low-level ocular pilots are not primary evidence for the newer iris strip panel.

**Checks:** read endpoint counts and scale definitions directly from aggregate files; generated source hashes and producing commits. No experiments were rerun.

## Iteration 02: make claims match their measurement scope

**Inputs:** QTIM tier-1 R profiles, face v6f pixel and ArcFace reference profiles, P2 iris profiles and directional tests.

**Finding:** repeat-anchored and self-anchored values differ in their zero. The face ArcFace reference is whole-image donor identity, whereas the pixel distance is local. A failure of T2 does not prove equivalence.

**Decision:** retain both anchors by name, state the spatial-scope mismatch explicitly, and report the attenuation confidence intervals separately from the one-sided test decision. Do not call a context donor an identity-preserving inpainting.

**Change:** replaced section TODOs with methods, measured-results prose, limitations and a distinct LLM statement. Added source-generated coverage, brain, face and iris tables.

**Checks:** preparation reads aggregate files only. Initial full pytest: 344 passed, 8 skipped, 6 failures caused by missing optional dependencies and incompatible environment versions. These failures were preserved and are not reported as passes; environment alignment is in progress.

## Iteration 03: author decisions and the first-draft boundary

**Inputs:** author replies to the three handoff questions and the requested iteration/skill workflow.

**Decisions:** assemble now from the available Git snapshot; prepare the main figure with Gaussian brain and provisional fundus; Codex will integrate face files after they arrive. Apply scientific-writing, scientific-critical-thinking, conference-submission-audit, repro-judge and academic-discoverability, with decisions logged on every iteration.

**Figure consequence:** the private-data map pipeline cannot run on this host. Use immutable published aggregate PDF panels, replace the obsolete brain row with the existing Gaussian panel, regenerate annotations from its JSON, and record that this is PDF-panel assembly rather than a new map computation. Preserve the complete source-derived atlas in the supplement and identify P2 iris as P2.

**Next checks:** assess figure scales and privacy; inventory all available materials; perform claim-level reproduction and scientific reviews; complete the metadata editorial pass; render and audit the final first-draft PDFs.

## Iteration 04: recover the available bilateral evidence

**Inputs:** immutable bilateral commit 82ed7fe, its P8/P9 frame-only and completed v1 fundus metric JSONs, saved atlas PDF and generator; current Gaussian brain maps.

**Finding:** the available Git history contains newer fundus aggregates than the initial CVPR checkout. P7 is therefore no longer the strongest accessible frame-only source. The still newer running extension is a different source.

**Decision and change:** preserve the exact bilateral JSON blobs in paper inputs, generate the primary fundus table and coverage from P8/P9, retain P7 as history, and add the v1-versus-P2 metric availability registry. Assemble the requested atlas now from immutable aggregate PDF panels, replace the stale brain row with Gaussian panels, and retain secondary iris rows in the supplement. Scalar brain annotations are pooled, not slice-only estimates.

**Checks:** source hashes and producing commits are recorded; common diverging ratio scales were checked against the source generators. Fixed an imported PDF footer-font resource issue detected by pdftotext. No map arrays were recomputed and no participant exemplar was introduced. New generator code has a source hash because it has not yet been committed.

## Iteration 05: material coverage and claim-level review

**Inputs:** 263 tracked scientific artifacts across 14 material families; CARE iteration-2 rows; the existing repro-judge icml2026_v2 rubric; current manuscript and numerical source records.

**Finding:** historical simulations, unregistered pilots, fixed real-case demonstrations and CARE score reaggregation answer different questions from the main anatomical donor panels. Author PASS/F/P labels do not by themselves establish reproduction.

**Decision and change:** map every inventoried material to a text role or exclusion reason. Include the narrow CARE audit and historical correlation/size sensitivity in the supplement; keep synthetic evidence as mechanism controls. The Codex documentary claim review assigns five verified, two toy and one inconclusive verdicts (12/16 under the unchanged rubric), with no probability or independent-rerun interpretation. Preserve the P3 H4 failure report as pending numerical integration rather than inventing a figure.

**Checks:** distinguished person versus eye, local pixels versus whole-image ArcFace, repeat versus self anchors, and failure of a directional test versus equivalence. Added explicit intensity definitions for TM-like and lDDT-like adaptations with verified original-method references. Scientific and reproduction review files retain the remaining confounders and provenance limits.

## Iteration 06: title, abstract and citation fidelity

**Inputs:** academic-discoverability skill, evidence ledger, ten generated query intents and three near misses; Crossref/arXiv and official primary proceedings; bib-verify output.

**Decision and change:** apply “Identity Ladder: Auditing Image Similarity for Individual Anatomy” to main and supplement. The abstract states the four domains, controls and two measured comparisons, retaining their different anchors/scopes and provisional boundaries. No universal identity metric or trained-model benchmark is claimed. Keep the generated query panel as an editorial check, not a measured retrieval gain.

**Checks:** the title is eight words and the abstract 214 words. All bibliography entries go through bib-verify. Retain correct published conference years for Betti matching and FWD despite automatic preprint-year flags; document primary-source resolutions and the surname-diacritic false flag without rewriting the automatic report. Corrected the escaped DOI, print year and rendered accent issues in the metadata generator.

## Iteration 07: build, tests and venue-file review

**Inputs:** official current CVPR 2027 author guide; official author-kit Git tag CVPR2027-v1; compiled main and supplement; full repository tests.

**Finding:** the initial main reading copy uses eight content pages, with LLM statement before references on the next page. The supplementary initially had a blank title-only page and an overfull float page. The direct kit ZIP download timed out, while the primary official Git tag was reachable and matched the local style byte-for-byte.

**Decision and change:** remove the blank supplementary cover break, separate and bound the historical figure pages, scan identifying text and unresolved references, and visually inspect rendered pages. Treat compliance of inspected files separately from unknown portal, overlap and ethics facts. Keep paper build requirements and commands distinct from experimental reproduction.

**Code verification:** full pytest completed with 349 passed, 8 skipped, 1 failed. The remaining real-cases guard correctly refuses Python 3.12 where its frozen protocol requires 3.11. Do not relax another owner's guard or report an all-pass suite. Tests downloaded a model into an accidental Windows-style relative cache directory under WSL; remove this session-generated cache from the working tree after the run.

**Remaining boundary:** first-draft delivery does not wait for unavailable private exports, as authorized. Final submission and push still require the author's PDF review and resolution of the separately listed source/QC/administrative facts. Final PDF hashes, layout checks and page counts are recorded in verification.json after the corrected build.

## Iteration 08: read the figure at manuscript size

**Finding:** the first eight-column atlas was valid as an aggregate assembly but its scaled annotation text was too small on the main-paper page. The PDF-panel API also exposes off-clip images inside nested forms, so image-bound extraction alone can crop a replacement panel incorrectly.

**Decision and change:** retain six key columns and all six primary-domain/dataset rows in the main figure, redraw labels and the scale at native manuscript size, and keep all eight columns/eight rows in the supplement. Use original source panel bounds for original rows and explicit replacement bounds for brain. Retain the exact displayed source annotations, with a fit assertion for every new text box. No source map or measured value is recalculated.

**Scientific clarification:** the main caption now distinguishes pooled mean-error map summaries and 16-donor context means from anchored, window-level, often single-donor metric tables. The last column is labeled mean-fill V rather than implying a separately established biological variance. Brain remains Gaussian/provisional, fundus P8/P9 provisional, and the iris panel P2 rather than P3.

**Checks:** visually inspected the enlarged atlas and the complete main/supplement contact sheet; inspected main numerical-table pages at reading resolution. Corrected missing text-box fits and the brain crop before accepting the new figure. A final rebuild updates the exact PDF hashes and file audit.

## Iteration 09: accept the reviewable first draft

**Final artifacts:** main PDF is nine pages total: eight content pages and one page for the LLM statement/references. The separate supplement is eleven pages. No section TODO remains. Tables and figures have source/generator hashes and producing-commit records. The final generated report records no overfull boxes, missing glyphs, unresolved references or identifying-text hits in either PDF. Rendered-page inspection confirms legible main-atlas labels and no blank cover, clipped captions or overflowing supplementary figures.

**Verification limits retained:** the TeX build still emits upstream lineno UTF-8 and included-PDF-version warnings, without observed missing text or rendering defects. The full test result remains 349 passed, 8 skipped and one Python-version guard failure. Seventeen of twenty bibliography entries have unflagged automatic verification; three primary-source manual resolutions are documented, with the automatic flags preserved. These are distinct checks and none establishes independent experimental reproduction.

**Delivery decision:** the available-snapshot first-draft objective is complete. Continue later with full P3 aggregate integration, receipt of the face base files, experiment QC and author-specific submission facts. The paper is local, uncommitted and unpushed. The legacy private-data generator is explicitly labeled historical; the current paper entry point is build.py and its Gaussian PDF-panel assembler. The generated metric board was refreshed and left outside the intended paper change set.

## Iteration 10: hand off independent paper and result review (9 November 2026)

**Request:** prepare a handoff for reviewing the manuscript and its scientific results.

**Checks:** re-read the current Git state, stored PDF verification, scientific/reproduction audits and full-test summary. The branch is still at a48523d, and the new draft remains local, uncommitted and unpushed. The saved PDF hashes and evidence boundary are unchanged. No experiment, new citation verification or conference-rule audit was performed for this handoff.

**Decision and change:** create HANDOFF_REVIEW.md with exact checkout/base, source-to-claim anchors, review priorities, provenance/scale pitfalls, missing exports, prior-check limits and concrete independent-review outputs. Warn that a GitHub checkout alone does not contain this draft. Provide an author-review overlay ZIP containing current public manuscript materials, reading PDFs and a hashed snapshot manifest, excluding private data, model caches, Git metadata and transient build files. Existing author audits must be challenged rather than inherited as reviewer verdicts.

**Boundary:** the reviewer is asked to inspect and report before rewriting. No experiment, commit, push or message to another agent is requested. The handoff preserves the registration gate, result ownership and the author's earlier decisions to proceed from this Git snapshot, use Gaussian brain panels with provisional fundus, and integrate base face files only after receipt.

## Iteration 11: decompose the actual competition decision (9 November 2026)

**Author clarification:** the paper should help choose evaluation for next year's BraTS inpainting competition. It need not announce a final metric, but must advance that decision.

**Finding:** the first draft emphasized cross-anatomy interpretations without mapping them explicitly to adoption requirements. Metric failure profiles, clinical/task validity, model-ranking stability and evaluator deployment are different evidence questions. A familiar conditional-mean advantage under MSE is not, by itself, a reason to reject a metric for a reference-distortion objective.

**Decision and change:** create twelve independent decision questions in decision_coverage.json and a generated source-backed assessment in BRATS_METRIC_COVERAGE.md. Record the investigated scope, decision progress, missing evidence and source paths for each. Use qualitative coverage, with no overall percentage or fictitious adoption score. The most direct missing link is a shared-case comparison of actual inpainting methods against an independent anatomy/usefulness criterion.

**Checks:** inspected source aggregate/protocol/code files and current official 2026 task/evaluator sources. Treat 2027 as the author's intended application, not announced rules. No new metric experiment or patient-data analysis was run; the registration/QC priority remains in force.

## Iteration 12: recover useful implementation evidence and reject invalid history

**Inputs:** saved brats_panel_checks outputs for torchmetrics 1.1.2 and 1.9.0 with inpainting 0.0.12; src/brats_panel_checks.py; official evaluator source blob 2141df7775ff08a4599f19daa9b43acf717d63b2; label robustness and historical real-case files.

**Finding:** the saved synthetic probes establish two-dimensional SSIM window behavior for the recorded four-dimensional layout, and plain-PSNR reference-range clipping. On eleven constructed fills, plain PSNR has Kendall agreement 0.890909 with RMSE, versus 1.0 for the epsilon variant. Thus an unconditional claim of exact MSE/PSNR rank redundancy would be false for the actual recorded implementation. This is code-level evidence, not an independent anatomical signal or an estimate of leaderboard instability.

**Decision and change:** report the implementation findings in main results and a source-generated supplementary table. Keep monotone equivalence conditional on identical scored errors and ranges. Explicitly exclude both real_cases aggregate files, whose source headers withdraw all c/rho/label outputs after the intensity-scaling error; they cannot count as provisional supporting MRI evidence. Mark these exclusions in the material inventory.

**Scientific consequence:** synthetic Dice/topology and a segmenter's agreement remain proxy tests, whole-image ArcFace remains a donor reference, and stranger calibration does not create anatomical information. The present controls narrow what a score may be interpreted to mean, while direct model-ranking validity remains open.

## Iteration 13: revise the manuscript around progress toward selection

**Decision and change:** apply the title “Identity Ladder: Toward Metric Selection for Brain MRI Inpainting,” retaining a bounded aim. Rewrite the abstract, opening motivation, synthesis and conclusion around the competition decision. Add the twelve-question coverage mapping, group it into a main-text table, retain the complete mapping in the supplement, and move the overview cohort table and control-profile plot there. Cohort counts and measured modality results remain in main text. The claim-first outline is preserved in BRATS_METRIC_COVERAGE.md.

**Checks:** refreshed scientific/reproduction and metadata reviews. Added and verified the official BraTS task-paper citation by arXiv; the bibliography now has 21 entries, with 18 unflagged automatic resolutions and the same three documented manual resolutions. The new abstract is 246 words; the nine-word title is consistent in both documents. No trained-model benchmark, universal metric or validated replacement score is claimed.

**Venue/layout correction:** the first revised build had nine content pages because a delayed main float spilled onto a separate page. Moving the redundant cohort overview to the supplement returned the main content to eight pages without changing type size or margins. The official CVPR 2027 ZIP became reachable; its style differs from the original official Git tag by four indentation spaces only. Copy the ZIP's exact style bytes and record both provenance states rather than claiming the earlier Git hash equals the ZIP hash.

**Remaining verification:** retain the earlier full-test result as an 8 November check rather than claiming a new run. Verify current generation, hashes, layout, anonymity and bibliography separately. The earlier handoff archive remains an immutable first-draft snapshot; a separate archive will carry this revision and its updated decision map.

## Iteration 14: verify the decision-oriented reading copy

**Outcome:** the revised main PDF has eight content pages plus two LLM/reference pages; the supplement has fourteen pages. The generated main/supplement checks report no overfull boxes, missing glyphs, unresolved references or identifying-text hits. The main coverage table and full twelve-question supplement were inspected at reading size. An overflowing CARE table was moved to a new supplementary page and re-rendered; its footer is clear. The official kit style matches byte-for-byte.

**Evidence checks:** regenerated every table from its saved JSON or explicitly authored coverage specification, refreshed the source/hash manifests, and reran bib-verify for all 21 entries. The three prior manual citation resolutions remain visible. The material inventory now separates evaluator probes and withdrawn MRI results into their own categories. No new metric experiment, model inference or statistical ranking was run. The prior full pytest result is retained with its original date; it is not presented as a new all-pass run.

**Delivery:** provide the twelve-question assessment, revised PDFs and updated review handoff/package. Keep the earlier review ZIP unchanged as the first-draft snapshot. The practical result is a better specified evaluation question, concrete implementation/control findings and an explicit route to direct model-ranking validation, not a declared winning metric. The working branch and Git HEAD remain unchanged, with no commit or push.

## Iteration 15: supersede the BraTS-centered framing

**Author correction:** frame the paper around inpainting generally, then its extension to how conditional generative models behave across population slices. BraTS is one application, not the paper's organizing goal.

**Finding:** the existing experiments interrogate metrics and donor/reference candidates. Metadata-matched means define donor cohorts; they do not establish model performance in corresponding target strata. Dataset/modality contrasts are not demographic disparities, and an initial protocol's strata-not-run list does not describe all later/private runs. No generator subgroup evidence was found in the accessible snapshot.

**Decision and change:** apply “Identity Ladder: Population References for Inpainting and Conditional Generation,” revise the abstract/introduction/synthesis, and make population_audit.json the active twelve-question evidence assessment. Retain the BraTS-specific map as application history. Define a proposed paired output-minus-control profile, rather than claiming that the new model experiment has already been executed.

## Iteration 16: make the proposed population audit interpretable

**Checks:** a separate read-only agent critique inspected the source protocols and proposed equation. Adopt its requirements: oriented paired distances; average seeds within target; fixed support and common target/mask eligibility independent of generated content; failures reported; fill/composite/whole-image scope explicit; independent-person/family uncertainty. FID/FWD remain set-level endpoints outside the paired equation.

**Scientific decisions:** keep target strata, donor cohorts and acquisition domains separate. Retain raw, common-reference and stratum-matched readouts. A difference between group profiles characterizes the model--metric system and needs sensitivity to references, acquisition and task composition before attribution to the generator. Reduced variance alone is not mode collapse; a descriptive gap is not causal discrimination or fairness validation. Unconditional samples and unobserved counterfactual targets require a separate reference formulation. The current evidence remains human-anatomy donor/metric data; broader transfer and generator profiles are proposed.

**Change:** add POPULATION_AUDIT_FRAMING.md, generated coverage and a source-hash manifest. Preserve the source/version and QC limitations. No model inference or new metric experiment was launched.

## Iteration 17: implement the requested source-provenance infographic

**Author instruction:** use the supplied white/light-blue health-interface aesthetic to show where data came from, and retain one heatmap dataset per modality in the main figure.

**Decision and change:** build a pale provenance panel with four factual cohort cards around an anonymous decorative white 3D bust, followed by four heatmap rows. Use QTIM, CelebA, GRAPE P8 and UTIRIS NIR P2 based on protocol/comparison coverage: GRAPE has a repeat rung and NIR is primary in the iris protocol. Keep IMDb-WIKI, PAPILA and secondary iris sets in the full supplementary atlas and measured result tables. No clinical gauge, health rating or arbitrary ring score is copied from the visual references.

**Generation boundary:** the built-in image tool generated only the decorative bust; prompt/tool/asset provenance is saved under fig/assets/. A separate figure-building agent assembled vector labels, counts and source panels in make_provenance_infographic.py. Actual heatmaps, ratios/CIs and scales are imported from saved aggregates. Selected PDF clips are rendered at four-times source resolution to avoid hidden off-clip text; this is a documented display transformation, not map recomputation. The LLM statement and caption disclose the illustration separately from measured evidence.

**Checks:** inspected the new infographic, verified primary-row scalars and source counts, retained the GRAPE repeat endpoint's smaller denominator, and removed an ambiguous “independent source cohorts” subtitle. The graphic has one dataset per modality and preserves the Gaussian brain row, provisional gates, P2-versus-P3 distinction and separate V scale.

## Iteration 18: typesetting and verification of the population-reference draft

**Finding and correction:** the word “References” in the new title triggered a false bibliography-order failure in the file checker. Identify the bibliography heading at the start of a line instead of any occurrence of the word. The first full build also deferred the coverage table to a ninth content page. Declare that overview earlier and shorten the application-specific evaluator summary, keeping all numerical probe results in the supplement; no fonts or margins are compressed.

**Outcome:** the corrected main PDF has eight content pages and two LLM/reference pages; the supplement has fourteen pages. File checks report no overfull boxes, missing glyphs, unresolved references or identifying-text hits. Source/generator/asset manifests and PDF hashes are refreshed. Bibliography entries and measured results are unchanged, so the previous 21-entry verification and dated full-test result remain separate from this revision's generation/layout checks. The read-only scientific critique found no blocking proposed-versus-measured overclaim; its model--metric interpretation safeguard was adopted.

**Delivery boundary:** provide the new infographic/updated reading PDFs and a separate population/provenance review package. Preserve both earlier review archives as historical snapshots. All current work remains local, uncommitted and unpushed; no experimental owner files or source measurements were changed beyond the required regenerated metric board.

## Iteration 19: revise the main figure toward an editorial journal layout

**Author request:** use a Cell/Nature-inspired scientific layout, remove the dark MRI exterior, reconsider scale and review the figure text. This is a style direction, not a claim of compliance with either journal's figure rules.

**Decision and evidence:** use a white a/b layout, thin source leaders, fewer decorative containers, 50-point-high maps instead of 42 and 7-point numeric labels. Keep one dataset per modality. Replace ambiguous labels with reference anatomy, opposite side, context-selected mean (16 donors), and mean-fill error V. Explain pooled-map versus anchored-window numbers, row-specific V, modality-specific display sizes, source-frame F and provisional P in the caption. The synthetic bust remains decorative and unchanged.

**Display integrity:** remove only border-connected locations that are black/white/transparent in all six aligned MRI panels. Apply one common crop and a light outline outside retained support. The retained RGBA pixels are unchanged; 522 internal dark reference pixels are preserved. The exterior display mask is not a measurement/acquisition mask or validated segmentation. Gaussian windows, eligibility, scientific arrays, annotations and the 0/1/2.4 color scale are unchanged. Record the proof in provenance_infographic_v2.json. A read-only figure/text review found no blocking presentation issue; this is not experimental replication.

**Checks:** the PDFs build with eight main-content pages, ten main pages total and fifteen supplementary pages; file checks find no layout/reference warnings or identifying-text hits. A caption sentence beginning with References exposed a heading-detector false positive. Detect the exact PDF bibliography text line instead, preserving the page-limit check. No new metric/model experiment was run.

## Iteration 20: consolidate the fork and recover local-only work

**Author authorization:** inspect repository changes, create a separate branch on the owner's fork, preserve all work there and remove redundant local copies. The author explicitly chose to preserve unique Syncthing material before deleting that folder. This supersedes the earlier no-push instruction for this fork branch only.

**Finding:** origin/cvpr2027-paper advanced by 33 commits, from a48523d to 80a094c, changing 82 paths. Full iris P3, newer face/brain metrics, IXI and cross-modality summaries are now available. Preserve them without claiming they have already been scientifically integrated. Pin the draft's existing evidence to a48523d through Git object reads so P2 is never silently replaced by P3. Update availability wording and review records.

**Recovery:** compare all 456 non-external Syncthing files. Preserve 131 identical canonical files, 124 exact/LF-normalized history blobs, 37 received face files at canonical paths and 164 unique historical versions in an archive. The archive keeps matching registration variants without overwriting experiment owners' current sources. Make the face metric loader repository-relative after receipt. The second repro folder contains only an empty .git. Consolidate ignored external runtimes once locally and verify hashes before removing the old source.

**Language and privacy:** new manuscript, labels, documentation and commit messages are English. Explicit immutable-history exceptions are the original Russian BRATS literature notes and legacy metrics-board source/report, identified in the archive README. These remain historical, unverified records. Scientific images are aggregate/mean/drawn maps, not participant photographs. No credentials, restricted iris images or subject-level exports are staged.

**Validation:** full pytest after receiving the face files: 417 passed, 7 skipped, 1 failed. The remaining real_cases guard failure is the existing frozen Python-3.11 requirement in this Python-3.12 environment; do not relax the guard. The pre-recovery upstream suite had 401 passed, 8 skipped and the same failure. Tests are implementation checks, not independent experiment reproduction. Preserve first, verify the remote commit, then remove loose historical archives/renderings and the two old folders. No upstream PR or conference submission is part of this authorization.

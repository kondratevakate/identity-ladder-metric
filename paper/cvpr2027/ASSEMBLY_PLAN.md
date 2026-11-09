# CVPR 2027 assembly plan and evidence boundary

Source branch: cvpr2027-paper, initial snapshot a48523dccd2b70dae2615717d62d8725e5183528.
Working clone: a separate Git checkout outside the Syncthing source copy.
Owner: Codex for paper/cvpr2027 only. Experiment code and source aggregates remain owned by their experiment sessions.

## Claim-first outline

The introduction distinguishes reference distortion, population realism, anatomical agreement, and individual fidelity. It positions the audit against CARE, OCT LMSB, Brain-ID, perceptual similarity, and restoration theory using verified primary references. It does not claim those methods have been experimentally defeated here.

Methods define the donor comparison conditions, acquisition and family dependencies, the mean-error unit V, ratio-of-sums aggregation, residual magnitude/alignment, and the distinct R_repeat and R_self scales. They retain missing support as missing, matched selection controls, oracle labeling, Gaussian map windows, and each source's bootstrap scheme. Latest amendments are read even where the corresponding new experiment has not completed.

Results use measured aggregate files directly. The brain tier-1 panel remains provisional under its registration gate. The face v6f tier-1 result has an approved frame but the underlying face pipeline/protocol is missing in this checkout. Iris P2 is separate from the newer full P3. Historical unregistered ocular pilots are not promoted to primary evidence. Fundus P8/P9 aggregates recovered from the frozen bilateral Git snapshot are primary; P7 is historical, and block-search results predating the missing-support fix are excluded.

The synthesis explains why symmetry, acquisition stability, residual amplitude and individual identity cannot be reduced to one unsupported ordering. It distinguishes metric stress tests from a trained inpainting-model benchmark. The supplement replaces template text with definitions, source-level evidence tables, controls, uncertainty, and reproducibility limitations.

## Current claim-to-source map

| Claim or artifact | Source | Treatment |
|---|---|---|
| Brain mean, mirror, shrinkage and wrong-detail profile | results/donor_ladder_brain_metrics.json | P; quantitative table and intervals; no biological identity conclusion before registration gate |
| Face pixels versus identity-oriented reference | results/face_ladder_metrics.json | Approved v6f frame; report available tier-1 aggregates and missing base-protocol provenance |
| Iris repeat, fellow-eye and context-donor profiles | results/iris_ladder_metrics.json | P2 panel only; retain lid-shadow QC note and independent person counts |
| Fundus coverage and fixed-composition maps | results/donor_ladder_2d_facefmt.json at 82ed7fe, preserved in paper inputs | P8/P9, provisional; P7 historical; exclude superseded block searches |
| HACA3+ paired contrast sensitivity | results/donor_ladder_brain_haca3_stage1.json | P; stage 1 only, visual verdict pending, not a validated harmonized metric result |
| Gaussian brain maps | results/figures/fig_ladder_maps_common.pdf; map/grid JSONs | Aggregate-only and Gaussian; source commits and hashes recorded by script |
| Full cross-anatomy teaser | immutable bilateral atlas PDF and saved Gaussian brain PDF | Assembled now as authorized; replace the stale brain row, preserve fundus P8/P9 and iris P2 labels; source PDFs/transformations in atlas_provenance.json |
| Full iris P3 H4 failure, H5/H6 support | Workstation handoff only; aggregates not in Git snapshot | Reported by handoff, not numerically verified; do not substitute P2 values for P3 |
| Brain-ID metric | No completed endpoint | Not run; related work only |

## Pending inputs and publication boundary

The author decided to assemble the first draft from the available Git snapshot, prepare the Gaussian-brain teaser now with provisional fundus, and let Codex integrate the face pipeline/protocol after receiving those files. Full P3 aggregates remain absent; their absence does not stop this first draft and is not filled with old results.

All figures copied into fig/ and tables in tab/ must be produced by the paper preparation script with input hashes and producing commits. No individual images, subject identifiers, private row tables, or restricted iris images may enter this public fork. Only explicit owned paths may be committed after running the full tests. No push occurs before Kate has seen the rendered PDFs. The related ISBI submission remains separate and is not modified.

## Required iteration cycle, requested by the author

1. Inventory every available result, protocol, figure and handoff, including relevant other-branch artifacts. Map each material to main text, supplement, historical context, or an explicitly explained exclusion. Source absence and superseded versions remain visible.
2. Assemble prose from those sources with scientific-writing: claim-first outline, then complete paragraphs, one sentence per source line. Use scientific-critical-thinking to test alternative explanations, scope, uncertainty, support selection, circular targets and the difference between a failed directional test and equivalence.
3. Review manuscript claims with repro-judge's existing icml2026_v2 rubric. Use a Codex evidence review here, not a paid/model judge run. Keep verified/falsified/toy/inconclusive distinct from the experiment's F/P labels, and distinguish documented evidence, checked artifacts and actual reruns.
4. Check the current CVPR 2027 official author guide and kit with conference-submission-audit. Inspect the rendered main and supplement, anonymity, actual content-page count, LLM statement, citations, figure privacy, ethics and overlap. Portal and author-only facts remain unknown where not observed.
5. Apply academic-discoverability to title and abstract: evidence ledger, generated query-intent panel and near misses, three candidate titles, factual/length checks and one recommendation. Do not claim measured search ranking gains from editorial wording.
6. Verify every citation with bib-verify and primary metadata, regenerate all tables/figures with provenance, build and render PDFs, and run the appropriate code checks. Run the full existing tests before any commit; commit explicit owned paths only.
7. Append an iteration entry to TEXT_DECISIONS.md: inputs, findings, accepted/rejected wording, concrete changes, verification performed, remaining uncertainty and the next decision. Preserve earlier entries and failed checks. Repeat until the first draft and its evidence/audit reports are reviewable.

## First-draft deliverables

The first draft consists of main.tex/main.pdf, the separate supplementary.tex/PDF, generated tables, aggregate-only figures and their provenance, MATERIALS_COVERAGE.md, TEXT_DECISIONS.md, REPRODUCTION_AUDIT.md, SCIENTIFIC_REVIEW.md, METADATA_REVIEW.md and CVPR_AUDIT.md. The draft can be delivered with clearly recorded source or administrative gaps; it must not be called submission-ready while those gaps remain unresolved.

## 9 November: decision-oriented continuation requested by the author

The practical objective is to advance the choice of a metric or evaluation rule for the next BraTS inpainting competition. A final winner is not required; specific validated constraints, candidate limitations and a discriminating next evaluation constitute progress. The original cross-anatomy evidence remains the experimental basis, with no new metric runs authorized here.

The continuation first decomposes the decision into twelve independently assessable questions in decision_coverage.json, with source locations, qualitative coverage and missing validation. It then inspects official task/evaluator sources and saved implementation probes, brings the code-level findings into results, and rewrites the title, abstract, introduction and synthesis around the practical decision. Full question coverage, cohort overview and the control-profile plot remain in the supplement. Withdrawn real-case measurements are excluded explicitly. All generated tables retain their inputs and hashes.

Reproduction review, title/abstract intent checks, bibliography verification, current CVPR file checks and rendered-page review are refreshed for the changed draft. No overall completeness percentage is manufactured. The highest-value later study compares frozen real inpainting outputs on shared held-out cases against an independent anatomical/task criterion, then evaluates rank stability, mask strata, sampling and deployment rules under a separately frozen protocol.

## Superseding scope: general inpainting and population audits

The later author instruction makes inpainting the controlled entry point to evaluating conditional generation across population slices. BraTS is an application rather than the paper's central framing. The active assessment is population_audit.json and its generated coverage/manifest; the earlier BraTS-specific decision files remain historical application records.

The text separates the measured instrument/reference audit, the proposed model-output audit and its proposed extension across target strata. Target strata, independent donor cohorts and acquisition domains have distinct roles. Proposed paired profiles use common cross-model eligibility and support fixed independently of output content, average seeds within target, and retain raw/common/matched references. Model--metric group contrasts need uncertainty and sensitivity to reference/task composition; they are not automatically fairness or mode-collapse results. Unconditional samples and unobserved counterfactual targets require different validation.

The author's additional visual instruction is implemented as a pale anatomical-provenance infographic with four factual source cards and one primary dataset per modality. Only the anonymous decorative bust is AI-generated. Scientific heatmaps are copied from saved aggregate panels with fixed scales; no map or metric is regenerated. Protocol coverage determines the primary rows, not favorable ordering. Full secondary rows remain in the supplement. This is a manuscript/figure revision, with the registration priority and experimental ownership unchanged.

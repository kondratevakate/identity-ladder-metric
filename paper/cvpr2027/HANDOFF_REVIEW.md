# Handoff: independent review of the Identity Ladder paper and results

**Current framing and figure request:** general inpainting is the controlled entry point to a proposed audit of conditional generation across population slices. BraTS is one application. Read POPULATION_AUDIT_FRAMING.md, population_audit.json, POPULATION_AUDIT_COVERAGE.md and population_audit_manifest.json first. Current experiments measure donor/reference and metric responses; generated subgroup profiles remain proposed. Review the distinction between target strata, donor cohorts and acquisition domains, and the model--metric interpretation of a group gap. The new main illustration shows data provenance and one dataset per modality; the bust is decorative and synthetic, while all heatmaps/counts are source-based. The canonical review copy is in branch codex/cvpr2027-manuscript. Earlier loose review ZIPs are preserved in Git history.

Prepared 9 November 2026 for Kate. Task: independently review the first CVPR 2027 draft and the evidence behind its claims. Review correctness, scientific contribution, numerical traceability and presentation before editing prose. The existing development audits are inputs to challenge, not independent reviewer verdicts.

**Superseded BraTS-specific clarification, 9 November:** the paper's practical purpose is to advance metric selection for next year's BraTS inpainting competition. It need not identify a final winning metric. Read `BRATS_METRIC_COVERAGE.md`, `decision_coverage.json` and `decision_evidence_manifest.json` alongside the revised paper. Review whether each of the twelve decision questions is correctly matched to evidence, whether the saved evaluator probes provide useful progress, and whether the remaining direct-model/independent-anatomy validation is stated accurately. The earlier ZIP preserves the first version; the separate `identity_ladder_review_20261109_brats_framing.zip` carries this revision. Use verification.json for the current PDF hashes and counts.

## 1. Locate the actual draft

- Windows checkout: `C:\Users\Lenovo\source\identity-ladder-cvpr2027`.
- WSL checkout: `/mnt/c/Users/Lenovo/source/identity-ladder-cvpr2027`.
- Branch: `cvpr2027-paper`.
- HEAD: `a48523dccd2b70dae2615717d62d8725e5183528`, 8 November 2026, 14:03:31 +04:00.
- `origin`: `https://github.com/kondratevakate/identity-ladder-metric.git` (Kate's public fork).
- `upstream` also exists and points to koflerlab. Do not push there or create an upstream PR.

**The entire new draft is still local, uncommitted and unpushed. Cloning or checking out the GitHub branch alone gives the older skeleton, not this draft. Do not reset, clean or overwrite the working checkout.** This checkout is separate from the non-Git Syncthing copy under `D:\projects`; it is also separate from the workstation's `C:\Projects` worktrees. Neither private workstation data nor its uncommitted face files were accessed here.

For a different review computer, use the accompanying ZIP as an overlay in a separate clone at this base commit. The overlay contains the current paper sources, generated figures/tables, both reading PDFs, audit records, the historical atlas generator and a snapshot manifest. It excludes `.git`, raw images, private rows, model weights and build caches. The baseline experiment sources/results come from Git; the frozen bilateral commit below must also be available for a rebuild. The ZIP is an author-review package, not an anonymous submission archive.

## 2. Read in this order

All paths below are repository-relative.

1. `paper/cvpr2027/main.pdf`: ten pages total, eight content pages plus two pages for the LLM statement/references.
2. `paper/cvpr2027/supplementary.pdf`: fifteen pages.
3. `paper/cvpr2027/main.tex`, `supplementary.tex`, `sec/*.tex`, and `main.bib`.
4. `paper/cvpr2027/evidence_manifest.json` and `atlas_provenance.json`: selected records, source hashes, producing commits and assembly transformations.
5. `paper/cvpr2027/SCIENTIFIC_REVIEW.md`, `REPRODUCTION_AUDIT.md`, `METADATA_REVIEW.md`, `CVPR_AUDIT.md`, `citation_report.{json,md}`, and `verification.json`.
6. `paper/cvpr2027/TEXT_DECISIONS.md`, `ASSEMBLY_PLAN.md`, `MATERIALS_COVERAGE.md` and `materials_inventory.json`: decisions and the inventory of 263 tracked scientific artifacts across 14 families.
7. Root `AGENTS.md`, `docs/HANDOFF.md`, `docs/REPRODUCTION_WORKFLOW.md`, the general donor protocol and each modality protocol. Brain amendments run through 21; do not stop at the skeleton's older citations.

Title: **Identity Ladder: Population References for Inpainting and Conditional Generation**. Revised abstract: 219 words. Review a bounded reference/framework contribution: no generator subgroup performance, fairness guarantee or unconditional-generation identity criterion has been demonstrated.

## 3. Evidence map and numerical spot checks

These are reading anchors, not substitutes for checking the full JSON records, intervals, eligibility, protocols and code.

| Block | Authoritative accessible source | Key observation and boundary |
|---|---|---|
| Brain QTIM | `results/donor_ladder_brain_metrics.json`; map/grid/size and HACA3 stage-1 JSONs | Provisional affine alignment: 221 targets, 168 families, 23 repeats. MSE mean 0.412716, mirror 0.717274 on `R_rep`; attenuated repeat -0.013516. Registration QC gate is unresolved. |
| Fundus P8/P9 | `results/donor_ladder_2d_facefmt.json` and `results/fundus_ladder_metrics.json` at bilateral commit `82ed7fe8a52e557854cf4cd2d02eb47d5803d7cb`; byte-identical copies in `paper/cvpr2027/inputs/` | Provisional frame-only data, not the newer extension. GRAPE repeat Q about 0.489, fellow eye 1.330, stranger 2.012; PAPILA fellow eye 1.467, stranger 1.992 and no repeat. Historical P7 is not primary. |
| Face v6f | `results/face_ladder_metrics.json`, `docs/PROTOCOL_face_metrics.md`, `docs/HANDOFF_face_metrics.md` | 921 CelebA targets and 678 IMDb-WIKI targets, with 921/453 independent clusters. CelebA local context-donor MSE `R_self` 0.399476 versus whole-image ArcFace donor-reference 0.982955. These are different spatial scopes. |
| Iris P2 | `results/iris_ladder_metrics.json`, `docs/PROTOCOL_iris_metrics.md`, `docs/HANDOFF_iris_metrics.md` | NIR 156 eyes/78 people, VIS 158/79, UPOL 128/64; VIS repeat has 157 eyes. Q repeat/fellow: NIR 0.407/1.256, VIS 0.569/1.293, UPOL 0.177/1.118. Mask/lid-shadow caveat remains. |
| Contrast sensitivity | `results/donor_ladder_brain_haca3_stage1.json` | Mean absolute paired shift: deep 0.120959 to 0.016368; parietal 0.059131 to 0.033166, 23 pairs. Visual verdict pending; no demonstration of preserved individual detail. |
| CARE score aggregation | `results/care_iteration2.json` | 36 matched rows, 63 mismatches, 8 coverage-blocked and 1 unavailable. This audits released-score aggregation, not CT reconstruction or segmentation reruns. |
| Synthetic mechanisms | `results/toy_ladder.json`, `label_ladder.json`, `population_null_synthetic.json`, `label_robustness.json`, `pupil_ladder.json` and related reports | Known-target attenuation, wrong detail and pupil dilation are toy/mechanism evidence. They do not establish biological identity preservation or a universal Dice/topology theorem. |

**Evidence unavailable at the manuscript freeze (a48523d):** the full iris P3 aggregate export, the base face pipeline/full v6f protocol and newer face exports, newer fundus extension, private animation and completed Brain-ID metric. The workstation reports that iris P3 H4 (Betti-related) fails while H5/H6 hold; the numeric outputs are absent here. Do not promote that handoff statement to a verified numerical result or replace it with P2. Codex is authorized to integrate face files after receiving them; the base face files have now been received and preserved. Full iris P3 and newer brain/face/IXI aggregates are also present after the fork update. This handoff's scientific numbers describe the frozen draft, not a review of those newer exports.

## 4. Review questions, in priority order

**Scientific contribution and claim scope.** Identify the strongest supported contribution and any unsupported identity, symmetry, reconstruction or biological inference. Distinguish a new evaluation design from the familiar squared-error advantage of an expectation. Evaluate positioning against CARE, OCT LMSB, Brain-ID, LPIPS/SSIM, restoration theory and biometric identity. No completed empirical comparison against those systems is claimed.

**Numerical and statistical consistency.** Trace every headline number, table and central plot to its exact source key. Check ratio of sums versus mean of ratios, matched support for each denominator, target/family exclusion, population differences between repeat anchors and other rungs, selection-pool sizes, dependent regions/eyes, bootstrap units and multiplicity. Review missing-support handling in experiment code where available; passing a unit test does not certify historical private preprocessing. Failed T2 superiority is not equivalence or proof of the reverse.

**Coordinates and comparison units.** Keep Q (mean error = 1), `R_rep` (repeat = 0, unrelated = 1), `R_self` (self = 0, unrelated = 1), and unrelated-reference rank c distinct. Do not combine them into a cross-domain leaderboard. Check whether each identity unit is a person, eye, hemisphere or acquisition. F means a source-recorded approved/frozen frame, not independent replication. P is provisional. Unavailable, undefined, not run, invalid and absent-from-snapshot are different states.

**Metric definitions and adapted measures.** Audit directions, tolerances and calibration for MSE/RMSE/PSNR/SSIM, residual correlation, TM-/GDT-/lDDT-like intensity scores and identity references. A norm or correlation is not automatically an individual-identity measure. Distribution-level FID/FWD must not become per-eye distances; intensity-filtration Betti matching is an adaptation from segmentation. Inspect protocol amendments and measured outcomes without making expectation tables into biological ground truth.

**Atlas validity and readability.** The main journal-style infographic has panel a with cohort provenance and panel b with QTIM, CelebA, GRAPE P8 and UTIRIS NIR P2 heatmaps. The MRI exterior mask/common crop is display-only; retained pixels, Gaussian windows, numerical scale and statistics are unchanged. The pale outline is graphical, not a validated segmentation. Maps use modality-specific display sizes; V has its own scale, context maps average 16 donors, and GRAPE repeat coverage is smaller than its full cohort. F is source-frame approval and P is provisional evidence. See provenance_infographic_v2.json. Secondary datasets remain in the supplementary atlas; these are donor controls, not generated subgroup outcomes.

**Missing evidence and version history.** Make a prioritized list of which conclusions survive without private exports/QC and what evidence would change each conclusion. Do not merge P7/P8/P9, iris pilot/P2/P3, old size sweeps/latest object-relative protocol, or proposed/running/completed endpoints. Do not conceal a measured failure to obtain a cleaner paper narrative.

## 5. Checks already performed, and their limits

- Both PDFs compiled and were rendered/inspected. The saved file audit found no overfull boxes, missing glyphs, unresolved references or identifying-text hits. It does not certify anonymity of every upstream artifact or biological validity.
- The main PDF SHA-256 is `67b9794a751297c26eca8654817cf104fbf5767cb88ec96aa94d85b122074c3f`; supplement is `2f76c86c34ff8f7c627b1aed71d808f08cf20adbcaf7940f52014cfd31e2a074`.
- Full pytest after consolidation: **417 passed, 7 skipped, 1 failed** (9 November). Earlier 8 November result: 349 passed, 8 skipped, 1 failed. Failure: `tests/test_real_cases.py::test_guard_refuses_changed_inputs_and_missing_amendment`; its frozen guard requires Python 3.11, while the test environment is 3.12. No guard was relaxed. These tests are not experiment replication.
- Bibliography: 21 entries checked by bib-verify. Eighteen unflagged; three manual resolutions in `citation_report.md`: Dobeš/TeX accent normalization, Betti matching preprint-versus-conference year, and FWD preprint-versus-conference year. Recheck those primary sources independently. Automatic flags were preserved, not overwritten.
- Existing Codex repro-judge review: five verified, two toy, one inconclusive claims; 12/16 under `icml2026_v2`. This is a documentary development review, not a truth probability, independent experiment rerun or paper-quality score. Do not simply inherit its labels.
- CVPR file audit was refreshed on 9 November 2026. The official kit ZIP became reachable; its style now matches byte-for-byte. It differs from the original official Git style only by four indentation spaces. Recheck current official rules when assessing submission readiness. Author portal, ISBI overlap, dataset consent and institutional ethics basis remain unverified.

## 6. Skills and required reviewer outputs

Use the installed `scientific-critical-thinking`, `scientific-writing`, `repro-judge`, `conference-submission-audit`, `academic-discoverability` and citation/bib-verify workflows. Skills are under `C:\Users\Lenovo\.codex\skills\`; each has `SKILL.md`. Read the actual relevant skill/rubric rather than relying on this summary. No paid/model judge or calibration run was used during draft assembly.

Start with an independent reading before looking at the existing verdicts. Produce English review artifacts in the paper directory without overwriting the development audits:

1. `REVIEW_independent.md`: concise overall assessment, strongest defensible contribution to inpainting/conditional-generation evaluation across populations, then findings ordered P0 (correctness/evidence blocker), P1 (major scientific weakness), P2 (clarity/presentation). Every finding needs a source location, evidence, impact and smallest useful remedy. Distinguish measured references from proposed model-output profiles; do not require a final metric winner as the only contribution.
2. `REVIEW_claims.csv`: claim, manuscript location, source/key, unit/scale, evidence status, verified/falsified/toy/inconclusive verdict, reason and remaining evidence needed. The rubric verdict is separate from F/P.
3. `REVIEW_revision_plan.md`: fixes possible from current sources versus missing exports/QC/new experiments; recommended title/abstract changes only where supported; explicit submission blockers and uncertainty.
4. Append an iteration to `TEXT_DECISIONS.md` with evidence inspected, accepted/rejected interpretations, checks actually run and unresolved questions. Do not erase the earlier history.

Review first. Do not silently rewrite the paper, change measurements, remove inconvenient outcomes or relax protocols. No new metric experiment is authorized by this handoff: root AGENTS.md retains the registration-QC priority. Read other sessions' results without editing their experiment files. The author subsequently authorized a separate fork branch and local-copy consolidation on 9 November; see REPOSITORY_UPDATE.md. No submission or upstream PR is authorized here. Anything eventually pushed to Kate's public fork must be English and exclude raw faces/eyes, subject IDs, restricted UTIRIS images and private rows.

## 7. Rebuild only if useful for the review

Use WSL Ubuntu for project commands. Existing local tools:

```bash
cd /mnt/c/Users/Lenovo/source/identity-ladder-cvpr2027
/tmp/identity-cvpr-venv/bin/python paper/cvpr2027/build.py \
  --tectonic /tmp/identity-cvpr-tools/tectonic --render
```

Portable setup is in `paper/cvpr2027/README.md` and `requirements-paper.txt`. Fetch the frozen bilateral commit in a separate reviewer clone if absent. `build.py` regenerates tables/figures, builds both PDFs, runs the file checks and optionally renders pages; it does not rerun experiments or bib-verify. PDF timestamps can change byte hashes after rebuilding. Compare scientific inputs and selected records separately.

Do not rerun experiments to make an already documented evidence gap disappear. A new result requires its own protocol/QC and provenance, followed by a separately reviewed manuscript update.

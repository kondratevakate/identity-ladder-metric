# Scientific review of the first draft

Method: scientific-critical-thinking plus scientific-writing. This is a manuscript-development review, not independent peer review or an experiment rerun.

## Conclusions retained

The current material supports descriptive, source-specific metric profiles. It also supports a design for separating reference distortion, residual amplitude, target alignment and acquisition variation. It does not establish a universal biological ordering, a fused identity score, a new reconstruction method, or a completed cross-domain model benchmark.

## Consequential reasoning checks and text decisions

| Issue | Evidence or alternative explanation | Text decision |
|---|---|---|
| Mean advantage versus identity | Lower expected squared error can follow from variance reduction without target-specific information | Describe metric preference; do not interpret the mean as better preserved personal anatomy |
| Circular reference target | Early reality-B ordering used measured squared error; later axioms remain normative rather than biological ground truth | Retain empirical profiles and explicit assumptions; do not present fit to the axioms as independent identity validation |
| Repeat versus self anchors | A repeat photograph has acquisition and appearance differences; self has exact analytic distance | Name R_repeat and R_self separately and do not combine them into a cross-domain leaderboard |
| Local pixels versus ArcFace | One comparison is a local window; the other is a whole donor image | Report the descriptive disagreement and explicitly exclude an inpainted-composite identity interpretation |
| Failed directional test | Non-significance of repeat superiority does not imply equivalence or significant reverse ordering | State not established for T2 and report the attenuation interval separately |
| Biological and acquisition confounding | Brain registration remains unresolved; bilateral images share capture factors; longitudinal anatomy can change | Keep P status and avoid causal scanner, symmetry or biological claims |
| Missing support and selected donors | Padding can win a search or pull an estimated mean toward a filler | Keep validity masks throughout, exclude superseded block searches, retain eligible counts |
| Pooling and uncertainty | Both eyes, regions and repeated pairs are dependent; ratio of means and mean of ratios differ | Preserve independent people/families, source weights and one-/two-way bootstrap distinctions |
| Gaussian maps and display scale | A smoothing/display convention is not an empirical identity theorem | Preserve masked Gaussian windows and common ratio scale; distinguish V's row-specific scale and scalar annotation scope |
| Topology and distribution distances | Betti matching was developed for segmentations; FID/FWD are set-level; intensity adaptation and support selection can alter interpretation | Keep the adaptation and unit explicit; do not invent individual scores or full P3 outcomes |
| Window geometry | The latest object-relative sweep is specified but not completely evidenced by the older size file | Report the new definition as a protocol and the old sweep as historical sensitivity |
| Synthetic controls | Known-target shrinkage is algebraically constructed; pupil parameters are simulated | Keep toy/mechanism evidence separate from real-anatomy validation |
| Label overlap | Pointwise modal-label optimality does not prove a general Dice optimum | Retain only conditional synthetic observations, with no universal Dice theorem |
| CARE reproduction | Reaggregating released score files differs from rerunning reconstruction and segmentation | Report the narrow consistency audit; do not refute CARE's broader method from mismatched aggregates |
| Publication and ethics | Prior ISBI overlap, dataset consent and institutional basis require author-specific facts | Keep these unresolved in the author audit; do not invent consent, approval or submission eligibility |

## Remaining scientific uncertainties

The first draft cannot remove registration bias, establish the origin of every upstream number, or validate private image QC from public aggregates alone. Face base-pipeline provenance and the full P3 extension remain pending receipt. Population means and withheld donors are useful controls but do not supply the hidden biological truth of an inpainting task. These limitations determine claim strength and are not repaired by successful typesetting, citation verification, or a text-only reproduction judge.

## 9 November: evaluate progress toward the BraTS decision

The principal question is now the choice of a future competition evaluation rule. BRATS_METRIC_COVERAGE.md decomposes it into twelve independent requirements and documents the scope of each experiment. Qualitative coverage is separate from a measured performance score and from a final adoption decision. No new metric experiment was run.

The most consequential clarification is that a conditional expectation winning under squared error is not by itself evidence of an incorrect ranking under a reference-distortion objective. The organizers must decide how much individual fidelity, plausible counterfactual anatomy and downstream utility matter. An observed artificially masked healthy region and unobserved healthy tissue inside an actual lesion provide different references. Donor provenance is not automatically the desired clinical target.

Saved official-evaluator probes add concrete code-level progress. The recorded four-dimensional SSIM layout is two-dimensional in its spatial window; plain PSNR clipping prevents a blanket assertion of exact redundancy with MSE. This does not demonstrate an independent anatomical property of PSNR. The remaining server/shape/mask-boundary contract and model-rank validity still require testing.

The two historical real_cases result files explicitly withdraw all c, correlation and label outputs. Their source presence is not supporting evidence and must not be relabeled provisional. The current QTIM metric panel is separately provisional under its own alignment/QC state. Shape/topology/threshold-segmentation toys and the CARE aggregation audit do not establish MRI downstream utility. Whole-image ArcFace donor comparisons do not validate local MRI inpainting. Stranger calibration cannot create anatomical information absent from the base distance.

The highest-value future validation uses outputs from genuinely different methods on the same untouched cases, independently assessed anatomical/task errors and uncertainty of the resulting method rankings. Between-person donor registration is a present evidence limitation, while native-grid target/prediction comparison need not introduce that registration. Any future run remains subject to the project's explicit registration/QC priority and a separate frozen protocol; this writing revision does not launch it.

## Superseding population-audit framing

Inpainting is the controlled conditional-generation setting; a target population, a donor reference cohort and an acquisition domain are different axes. The revised methods define a proposed model-output-minus-control profile, with oriented paired distances, fixed support, common cross-model eligibility, per-target seed averaging and independent-person/family uncertainty. FID/FWD remain set-level endpoints outside that equation. The experiment set still measures reference/metric behavior, not generator subgroup performance.

A separate read-only agent critique checked these distinctions against the protocol and current text. It identified metadata-matched face means and the initial strata-not-run list as comparator/output specifications, not generated-group results; the latter statement must remain version-specific. Its useful correction was adopted: a group contrast characterizes the model--metric system and requires sensitivity analysis for reference, acquisition and task composition before attributing a disparity to a generator. No fairness, causal discrimination, mode-collapse or general unconditional-generation validation is claimed.

The main illustration now explains cohort provenance using an anonymous synthetic bust and four source cards. The measured maps remain saved aggregate panels: QTIM, CelebA, GRAPE P8 and UTIRIS NIR P2, chosen by protocol/rung coverage. Rendering/cropping source panels is a display operation, not recomputation from private images. All secondary datasets remain in the full supplementary atlas. The bust is decorative; its generated status and full prompt/tool provenance are recorded separately from scientific data. A separate figure-building agent checked displayed ratios and crop/scales; this is presentation verification, not experimental reproduction.

The broader population-audit framing does not by itself establish novelty against all generative-model auditing literature. That positioning requires further verified comparison before a strong priority or universality claim. The present title and text make a bounded framework/reference contribution.

## Journal-style figure display review

The a/b white layout and larger maps replace the earlier application-like cards. The shared MRI exterior mask/common crop affects only display. Source anatomy and retained pixel colors remain intact, including enclosed dark structures; the pale outline is not an anatomical segmentation. The caption distinguishes pooled map error, anchored window metrics, row-specific V, modality display sizes, F/P and P8/P2 source versions. Color limits are unchanged. See provenance_infographic_v2.json.

The fork update now supplies full P3 and newer brain/face/IXI exports. This availability does not retroactively verify them or update this draft's frozen tables. REPOSITORY_UPDATE.md separates preservation from the next scientific integration review.

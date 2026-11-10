# BraTS inpainting: metric-selection questions and experimental coverage

Author clarification, 9 November 2026: the paper should advance the choice of evaluation for the next BraTS inpainting competition. A useful outcome is a justified narrowing of choices and explicit next tests; naming a winning metric is not required.

BraTS 2027 is the author's intended application, not a claim about announced 2027 rules. The external reference is the published 2026 task and evaluator.

Coverage describes what has actually been investigated. It is not a percentage of the overall problem solved, a quality score or a metric recommendation.

## Claim-first manuscript outline

Introduction: state the competition decision, distinguish observed-reference recovery from unobserved healthy counterfactual tissue, and define why similarity alone does not fix the evaluation target. Methods: retain the ladder as a candidate-audit instrument. Results: add existing official-evaluator probes before the donor comparisons. Synthesis: map each decision requirement to measured evidence and identify the unfilled model-validation step. Conclusion: report what has become testable rather than announcing an unsupported replacement metric.

## Coverage by independent decision question

| ID | Subtask | Existing coverage | Consequence for selection |
|---|---|---|---|
| D01 | What should a winning reconstruction preserve? | Conceptually developed; objective weights unresolved | Specify the evaluation target before assigning a metric failure. A conditional mean can be optimal for squared-error prediction, so its advantage alone does not establish a bad competition ranking. |
| D02 | What does the implemented evaluator actually measure? | Substantial implementation-probe coverage | Freeze implementation, shape convention, mask support, data range, clipping and versions. Metric names alone are insufficient to specify the competition rule. |
| D03 | Does a metric notice removal of individual detail? | Mechanism established; MRI comparison provisional | Require an attenuation/mean control and report how it changes the score. A high pixel or stranger-calibrated rank alone cannot establish correct individual detail. |
| D04 | Does it distinguish wrong detail from missing detail? | Controlled mechanism/proxy coverage | Reject interpreting residual magnitude or visual realism alone as correct anatomy. State the desired tradeoff before evaluating a candidate distance. |
| D05 | Can nuisance acquisition differences be separated from anatomical error? | Partial; MRI remains provisional | Use repeat/acquisition controls separately from wrong-anatomy controls. Invariance is valuable only if it does not erase information relevant to the selected task. |
| D06 | Does the score reflect anatomical structure and useful downstream behavior? | Synthetic/indirect evidence only | Do not assume that adding Dice, a topology descriptor or a segmentation pipeline automatically fixes averaging bias. Specify structure, reference labels and downstream purpose. |
| D07 | Is the result stable across 3D masks, tissue regions and scales? | Partial spatial coverage; challenge transfer untested | Report mask size, geometry, anatomical location and boundary separately instead of assuming a scalar average describes all voids. |
| D08 | Do candidate metrics make useful distinctions among real inpainting methods? | Not tested in a valid comparative BraTS experiment | The present evidence can expose failure modes; it cannot identify the metric that best ranks a U-Net, diffusion model or other competition entry. |
| D09 | What independent criterion establishes that one metric gives a better decision? | Proxy references; no direct MRI criterion validation | Metric-to-metric agreement is not validity. Validate against an independently defined anatomy/error/usefulness criterion, keeping uncertainty in that criterion. |
| D10 | Will the aggregation produce a stable, non-redundant leaderboard? | Metric dependence/probes partial; model ranks untested | Check effective weighting of correlated distances, per-case versus pooled aggregation, ties, missing runs and case-bootstrap rank stability before adopting a rank sum or composite. |
| D11 | How should stochastic outputs and unavoidable ambiguity be scored fairly? | Conceptual controls only; not evaluated | Freeze whether the competition requests one output or several, the sampling budget and selection rule. Do not mix best-of-N, mean-of-N and one-sample evaluations without specifying their different targets. |
| D12 | Can the chosen rule run reliably and reproducibly in the competition? | Partial engineering evidence; no deployment validation | Treat deterministic preprocessing, failures, runtime, dependencies and reference availability as selection criteria, separate from scientific validity. |

## D01: What should a winning reconstruction preserve?

The ladder separates reference distortion, residual amplitude, target alignment and acquisition. Synthetic controls distinguish omitted from incorrect detail. The task may value a particular observed healthy region, plausible counterfactual tissue or downstream utility; those are different targets.

**Progress:** Specify the evaluation target before assigning a metric failure. A conditional mean can be optimal for squared-error prediction, so its advantage alone does not establish a bad competition ranking.

**Still needed:** Organizer agreement on acceptable uncertainty and the relative costs of blur, plausible but incorrect anatomy and downstream failure. No patient-specific healthy counterfactual is observed inside an actual lesion.

**Sources:** `docs/PROTOCOL_donor_ladder_general.md`, `results/toy_ladder.json`, `results/label_ladder.json`.

## D02: What does the implemented evaluator actually measure?

Saved inpainting 0.0.12 probes with torchmetrics 1.1.2 and 1.9.0 inspect tensor-axis behavior, percentile normalization, PSNR clipping and within-case ordering. They are synthetic code probes, not official-server reproduction.

**Progress:** Freeze implementation, shape convention, mask support, data range, clipping and versions. Metric names alone are insufficient to specify the competition rule.

**Still needed:** A container pinned to the actual organizer deployment, server parity, exceptional-input handling and accepted 3D orientation/boundary behavior on challenge volumes.

**Sources:** `src/brats_panel_checks.py`, `results/brats_panel_checks_tm1.1.2.json`, `results/brats_panel_checks_tm1.9.0.json`.

## D03: Does a metric notice removal of individual detail?

Analytical and synthetic averaging/attenuation controls, a provisional QTIM mean-versus-mirror profile, and face pixel/identity-reference disagreements expose score interpretations. Different face measurement scopes are explicit.

**Progress:** Require an attenuation/mean control and report how it changes the score. A high pixel or stranger-calibrated rank alone cannot establish correct individual detail.

**Still needed:** Validated BraTS reconstructions showing loss of recoverable target-specific detail, separated from uncertainty, registration and acquisition. Evidence that an alternative metric improves actual model decisions.

**Sources:** `results/toy_ladder.json`, `results/population_null_synthetic.json`, `results/donor_ladder_brain_metrics.json`, `results/face_ladder_metrics.json`.

## D04: Does it distinguish wrong detail from missing detail?

Norm-matched wrong residuals, attenuation and noise distinguish magnitude from target direction. QTIM T3a/T3b and T2 are separate tests; a failed superiority test is not equivalence. Constructed controls do not supply clinical error costs.

**Progress:** Reject interpreting residual magnitude or visual realism alone as correct anatomy. State the desired tradeoff before evaluating a candidate distance.

**Still needed:** Matched-realism anatomical errors in actual 3D model outputs, independent error labels and agreed penalties for hallucination versus omission.

**Sources:** `results/donor_ladder_brain_metrics.json`, `results/iris_ladder_metrics.json`, `results/toy_ladder.json`.

## D05: Can nuisance acquisition differences be separated from anatomical error?

Repeat captures across modalities, synthetic pupil dilation and HACA3 stage-1 paired contrast shifts probe acquisition sensitivity. HACA3 contrast reduction is not a demonstration of preserved individual structure.

**Progress:** Use repeat/acquisition controls separately from wrong-anatomy controls. Invariance is valuable only if it does not erase information relevant to the selected task.

**Still needed:** Accepted MRI registration/QC, scanner/site/contrast stratification and a measurement of both nuisance tolerance and sensitivity to real anatomical errors.

**Sources:** `results/donor_ladder_brain_haca3_stage1.json`, `results/donor_ladder_brain_metrics.json`, `results/face_ladder_metrics.json`, `results/iris_ladder_metrics.json`, `results/pupil_ladder.json`.

## D06: Does the score reflect anatomical structure and useful downstream behavior?

Label robustness explores Dice and simple shape/topology descriptors under threshold segmentation and different priors. CARE is a released-score aggregation audit. Neither validates real BraTS downstream anatomy.

**Progress:** Do not assume that adding Dice, a topology descriptor or a segmentation pipeline automatically fixes averaging bias. Specify structure, reference labels and downstream purpose.

**Still needed:** Real 3D MRI with independently checked anatomical labels, blinded or task-relevant error assessments, and actual parcellation/registration/segmentation outcomes.

**Sources:** `results/label_ladder.json`, `results/label_robustness.json`, `results/label_robustness_report.md`, `results/care_iteration2.json`.

## D07: Is the result stable across 3D masks, tissue regions and scales?

Random brain windows, historical size/shell sweeps and Gaussian spatial maps localize variation. The latest object-relative sweep is specified but not established as completed. Cross-anatomy domains stress mechanisms rather than validate BraTS transfer.

**Progress:** Report mask size, geometry, anatomical location and boundary separately instead of assuming a scalar average describes all voids.

**Still needed:** Irregular challenge masks, 3D continuity, boundary seams, small critical structures and prespecified size/site strata in actual model outputs.

**Sources:** `results/donor_ladder_brain_size.json`, `results/donor_ladder_brain_maps.json`, `results/donor_ladder_brain_map_grid.json`, `docs/PROTOCOL_donor_ladder_brain.md`.

## D08: Do candidate metrics make useful distinctions among real inpainting methods?

Donor fills and artificial controls are available. Frozen protocols mention challenge models, but a protocol, cached model or fixed illustration is not a completed model-level comparison. Withdrawn real-case results are excluded.

**Progress:** The present evidence can expose failure modes; it cannot identify the metric that best ranks a U-Net, diffusion model or other competition entry.

**Still needed:** Frozen predictions from several genuinely different methods on shared held-out BraTS cases, with controls and endpoint-specific eligibility.

**Sources:** `docs/PROTOCOL_real_cases.md`, `docs/DECISION_TREES.md`, `docs/HANDOFF.md`.

## D09: What independent criterion establishes that one metric gives a better decision?

Known latent synthetic structure and the whole-image ArcFace donor reference constrain interpretation. Neither is an independent reference for local healthy-brain completion quality.

**Progress:** Metric-to-metric agreement is not validity. Validate against an independently defined anatomy/error/usefulness criterion, keeping uncertainty in that criterion.

**Still needed:** Blinded MRI reader assessment or independently adjudicated task errors on an untouched validation split, with agreement and uncertainty reported.

**Sources:** `results/face_ladder_metrics.json`, `results/toy_ladder.json`, `results/label_robustness.json`.

## D10: Will the aggregation produce a stable, non-redundant leaderboard?

Within-case monotonicity is known for several MSE-derived variants; actual plain PSNR clipping breaks exact rank identity in the saved probe. Rung correlations and family/position intervals exist, but are not bootstrap confidence for a competition-method ranking.

**Progress:** Check effective weighting of correlated distances, per-case versus pooled aggregation, ties, missing runs and case-bootstrap rank stability before adopting a rank sum or composite.

**Still needed:** Method-by-case scores and independent quality labels; rank/reversal uncertainty across cases and strata; frozen weights and a prespecified treatment of failed cases.

**Sources:** `results/brats_panel_checks_tm1.9.0.json`, `results/donor_ladder_metrics_correlation.json`, `results/donor_ladder_brain_metrics.json`.

## D11: How should stochastic outputs and unavoidable ambiguity be scored fairly?

Averaging/shrinkage mechanisms motivate separating a point prediction from a distribution of plausible completions. Saved donor averages are not a model ensemble or a calibrated predictive distribution.

**Progress:** Freeze whether the competition requests one output or several, the sampling budget and selection rule. Do not mix best-of-N, mean-of-N and one-sample evaluations without specifying their different targets.

**Still needed:** Repeated predictions from stochastic methods, fixed sample/compute budgets and validation of an appropriate quality/diversity or probabilistic evaluation rule.

**Sources:** `results/toy_ladder.json`, `results/population_null_synthetic.json`, `docs/PROTOCOL_donor_ladder_general.md`.

## D12: Can the chosen rule run reliably and reproducibly in the competition?

Versioned aggregate artifacts, validity-mask tests, source hashes, a metric registry and the existing repository test run support reproducibility practice. Advanced endpoints remain blocked or absent; the historical Python-version guard failure is retained.

**Progress:** Treat deterministic preprocessing, failures, runtime, dependencies and reference availability as selection criteria, separate from scientific validity.

**Still needed:** Pinned evaluator container, measured runtime/memory at challenge scale, deterministic reference behavior, licensing/access checks and a hidden-test deployment rehearsal.

**Sources:** `docs/REPRODUCTION_WORKFLOW.md`, `results/iris_ladder_metrics.json`, `paper/cvpr2027/inputs/fundus_ladder_metrics.json`, `paper/cvpr2027/README.md`.

## Concrete implementation progress already available

These deterministic synthetic probes were previously executed and are inspected here, not rerun. They establish behavior of the recorded implementation, not clinical prevalence or server parity.

| Probe | Saved value, torchmetrics 1.9.0 | Meaning |
|---|---|---|
| SSIM original / joint first-axis permutation | 0.956163 / 0.956163 | Consistent with the documented 4D input being treated as 2D batches/channels |
| PSNR reported / unclamped-error expression | 12.316670 / 9.091105 dB | Reference-range clipping changes the evaluated error |
| Plain PSNR / PSNR_eps rank agreement with RMSE | 0.890909 / 1.000000 | Exact monotone redundancy applies only when the same error is scored; the plain variant is a counterexample |

The corresponding 1.1.2 source is retained separately. The constructed panel has eleven fills in one synthetic volume; this is not eleven independent patients or a study of model leaderboard stability.

## What can be concluded now

The strongest progress is a set of discriminating controls and implementation specifications. MSE/RMSE quantify reference distortion; a mean advantage cannot by itself identify the wrong ranking under that objective. SSIM requires a precise spatial/implementation contract. Residual correlation measures direction and misses positive amplitude shrinkage; c is a calibration, not new identity information. TM/lDDT-style intensity scores remain provisional candidates. Face ArcFace is an indirect donor reference, and advanced iris metrics do not validate an MRI endpoint. No candidate has yet passed all requirements for selecting the next competition metric.

## Next work ordered by value for the decision

First, freeze the intended output and loss tradeoff, mask/reference rules and evaluator contract. While the registration gate remains unresolved, continue only the authorized QC/source-integration and manuscript work. Then, under a separate frozen protocol consistent with that gate, compare genuinely different inpainting methods and controls on the same held-out brain cases, obtain an independent anatomy/usefulness criterion, and test candidate-method rankings with case-level uncertainty. Freeze a final metric or composite using development cases and validate it on untouched cases. Set stochastic-output and computational rules before seeing final ranks.

For native-grid inpainting outputs, target and prediction already share coordinates; unnecessary between-person registration is not a prerequisite for that future comparison. Donor-ladder registration remains a limitation of the present proxy evidence. No new experiment is launched by this assessment.

## Explicit exclusions

- `results/real_cases.json`: WITHDRAWN: incorrect white-matter-peak scaling; all c/rho/label results invalid.
- `results/real_cases_parietal.json`: WITHDRAWN for the same scaling failure; cannot be counted as provisional supporting evidence.

## Source and verification boundary

Documentary assessment of existing experiments, protocols and code. No new metric experiment, patient-data analysis or competition ranking was executed.
 Source hashes and extracted probe values are in `decision_evidence_manifest.json`. Coverage judgments are authored scientific assessments, not experimental measurements. Existing F/P labels and documentary reproduction verdicts remain separate.

- [2026 task-level named metrics and masked-region evaluation; not a 2027 rule.](https://github.com/BraTS-inpainting/2026_challenge/blob/main/evaluation/README.md), checked 2026-11-09.
- [Healthy/unhealthy masks and reference availability.](https://github.com/BraTS-inpainting/2026_challenge/blob/main/dataset/README.md), checked 2026-11-09.
- [Official evaluator source inspected; current source inspection is separate from the frozen 0.0.12 probes.](https://github.com/BraTS-inpainting/inpainting/blob/main/inpainting/challenge_metrics_2023.py), checked 2026-11-09.
- [Official task motivation; first posted 2023, v3 revised 2024.](https://arxiv.org/abs/2305.08992v3), checked 2026-11-09.

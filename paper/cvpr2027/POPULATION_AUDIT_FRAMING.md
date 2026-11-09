# Identity Ladder: from inpainting to population audits of conditional generation

Author clarification, 9 November 2026: the paper concerns inpainting broadly, with an extension to how generative models behave across population slices. BraTS is one application of the evaluation design. The existing experiments characterize reference candidates and metrics in human anatomy; a population-stratified audit of actual generator outputs is a proposed next step.

## Claim-first writing outline

The introduction starts with inpainting as a controlled conditional-generation task: visible context, withheld content and an explicit target unit. The ladder supplies reference alternatives for interpreting a completion's score. Methods define the optional generator audit and separate target populations, donor reference cohorts and acquisition domains. Results retain only measured donor/metric observations. Synthesis shows how they can support model profiles across strata and where validation is missing. The main illustration explains data provenance and shows one primary dataset per modality; all secondary panels remain available in the supplement.

## Three layers of contribution

First, calibrate the interpretation of the measuring instrument: repeat captures, contralateral anatomy, related donors where available, unrelated donors, global/matched means, attenuation and incorrect-detail controls. These are different comparison conditions, not a universal biological ordering.

Second, apply that instrument to actual completions. On the same held-out targets, masks and eligible support, compare a model with those controls. A model may achieve low reference error while reducing residual variation; retain detailed variation whose direction does not match the target; or react strongly to acquisition. These are hypotheses to test on model outputs, not outcomes established by the present donor experiments.

Third, repeat the model/control comparison across prespecified target populations, regions and acquisition domains. Ask whether a particular control preference, loss of relevant detail or nuisance sensitivity differs by group. Generalization to broader conditional generation requires a defined identity-bearing condition and meaningful reference unit. Pure unconditional generation has no intended individual target, so paired identity fidelity requires a separate formulation; distribution and coverage questions cannot be replaced by the same target-pair score.

## Three axes that must remain distinct

| Axis | Meaning | Present evidence |
|---|---|---|
| Target stratum | The subset whose generated outputs are being evaluated, for example a prespecified age range or a recorded metadata category where available | No generated subgroup profiles in the accessible results |
| Donor reference cohort | The independent people/eyes/images supplying controls, global means or metadata-matched means | Measured donor/reference profiles; matched means are controls, not subgroup model performance |
| Acquisition domain | Scanner, capture modality, illumination or other measurement conditions | Repeat and contrast sensitivity, with domain-specific QC limits |

Age and recorded-sex labels in a donor mean do not establish a demographic audit. Different datasets or modalities do not themselves demonstrate demographic disparities. Do not infer unavailable attributes from images. The initial face protocol's strata-not-run list describes that protocol version, while the accessible tier-1 aggregate has no generated subgroup audit; neither statement determines the status of unreceived private runs.

## Proposed model audit, not an executed experiment

For target x_i and mask M_i, visible context c_i is paired with K fixed-budget outputs y_ik = G(c_i, M_i, z_k). Define target strata before inspecting scores. Use held-out people/families and independent reference pools, and keep each model-versus-rung contrast paired on the same eligible targets and support. Actual regions, mask geometry and model conditioning must remain visible in the report.

A proposed descriptive readout is the mean oriented distance of an output minus that of control r, indexed by stratum a, metric m and rung r. Negative values mean metric preference over that control, not a probability of correct identity. We do not fuse this array into a new scalar score. Raw distances and the existing Q/R/c coordinates can accompany it with explicit units, anchors and coverage.

Report a fixed common reference alongside a stratum-matched reference. Common references help keep cross-group comparisons interpretable; matched references reveal proximity to the group's typical anatomy. Reporting only group-normalized scores can hide raw differences or entangle them with reference variance. Sampling budgets, donor pool sizes, mask/location difficulty, and acquisition must be comparable or explicitly stratified. Marginal and within-context sampling variation answer different questions. Reduced variation or preference for a mean alone is not proof of mode collapse.

Uncertainty must respect independent people/families, repeated masks and seeds, and estimated reference pools. Retain eligibility, unavailable relationships and small-group uncertainty; a missing subgroup is not a good score. Group gaps and worst-group summaries are descriptive where estimable, not automatic causal discrimination or fairness claims. Any execution requires a separately frozen protocol consistent with the existing registration/QC priority.

## What is already supported, and what is not

A group contrast describes the evaluated model--metric system. Attributing it to the generator requires sensitivity analyses for the metric, reference cohort, acquisition and task composition. The readout uses paired, oriented distances with an observed reference; set-level FID/FWD require their own definition. Average seeds within targets before the target mean, keep common cross-model eligibility independent of generated content, and report failures instead of dropping them.

Synthetic controls and saved real donor profiles support investigating averaging, wrong detail and acquisition effects. The brain/fundus panels remain provisional. Whole-image face ArcFace is a donor reference, not identity measured inside an inpainted composite. Matched demographic means are measured comparators, not generated subgroup results. Full iris P3 remains absent. The withdrawn real-case aggregates remain excluded. Neither the measuring metric nor the generator has been validated as fair, clinically useful or identity-preserving across population slices by these results.

The proposed extension is useful because it specifies a testable profile and the data needed to estimate it. POPULATION_AUDIT_COVERAGE.md and population_audit_manifest.json provide the current source-backed coverage. No research-generator outputs were evaluated and no new metric experiment was run; the decorative illustration uses separately disclosed image generation.

## Main-figure choices requested by the author

Use a light ice-blue/white anatomical-provenance illustration with four source callouts, and retain one primary heatmap row per modality: QTIM for brain, CelebA for face, GRAPE P8 for fundus, UTIRIS NIR P2 for iris. QTIM is the available brain panel, CelebA supplies the primary face reference, GRAPE retains a repeat rung, and NIR is the protocol's primary iris set. These choices are based on protocol/comparison coverage, not favorable score ordering. IMDb-WIKI, PAPILA and secondary iris datasets remain in the full supplementary atlas and result tables.

The white synthetic bust is decorative and explicitly disclosed; no scientific image, heatmap or measurement is generated. Dataset names, cohort counts, arrows and measured panels are assembled from recorded sources. Source-card denominators are qualified, including the smaller GRAPE repeat endpoint. Frosted cards and a pale background borrow the references' visual language without importing clinical gauges, health scores or commercial interface elements.

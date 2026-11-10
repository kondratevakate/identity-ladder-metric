# Registration: comparison and visual QC

Single protocol: `docs/PROTOCOL_registration.md`, Amendment 9. **Gate NOT MET.**

Saved method-comparison eligibility is not visual acceptance. No scores were recomputed.

| Dataset | Completed methods | Intended pairs per method | Saved recommendation | Visual status |
|---|---|---|---|---|
| GRAPE | G0, G1, G2, H1, H2 | 437 | G1; tied: G1, H1 | NOT ESTABLISHED |
| UTIRIS | I0, I1, I2, I3, I4, I5, H1 | 234 | not registrable under this protocol | NOT ESTABLISHED |
| UPOL | I0, I1, I2, I3, I4, I5, H1 | 192 | not registrable under this protocol | NOT ESTABLISHED |

[Frozen comparison and metric definitions](comparison.md). Q1a: raw error in pixels on the 224 grid, lower better; Q1b fundus Dice higher better; iris centre error lower better. Per-method JSON retains endpoint-specific denominators and intervals. Historical c is calibrated, provisional, and not rerun.

## Visual evidence, kept separate by method

| Evidence | Intended | Reviewed | Pass | Fail | Unreviewable | Not reviewed |
|---|---:|---:|---:|---:|---:|---:|
| SIFT / iris development history | 1465 | 13 | 0 | 6 | 7 | 1452 |
| Legacy G1-family frame sheets | 755 | 0 | 0 | 0 | 0 | 755 |
| Current shared G1 development | 755 | 7 | 0 | 6 | 1 | 748 |

Visual acceptance = pass / intended, raw proportion, higher better; >=90% in every prespecified stratum, including mirrors. These rows are different methods, not additive reviews.

QTIM: 1 repaired development scan; 4 reviewed box/branch decisions, 0 pass, 0 fail, 4 unreviewable. Full MRI denominator: 334 required comparisons; 72 targets / 68 families.

Current G1 reviews are development evidence, with Kate review pending; no acceptance is inferred. Next: repair independently checked disc/fovea localization; independent iris anatomy; QTIM local anatomical review and remaining-case validation. No identity-metric release.

## MRI full composite geometry: one development scan

Saved image replayed exactly: True.
Full pull Jacobian: 0 nonpositive / 1882851 valid brain voxels; raw unitless determinant, positivity required.
Roundtrip error: P95 0.038748 mm / 3766 points; raw physical error, lower better.
These checks do not establish preservation of individual anatomy or visual acceptance. [Audit](qc/mri_composite_v1.json); [frozen cohort](qc/mri_manifest_v1.json).

## Modality-specific pretrained development (not a release)

Original tables above are preserved. No visual acceptance or identity scoring is inferred.

- [fundus_pretrained_v1](qc/fundus_pretrained_v1.json): 7/7 pilot rows inspected; 3 fail, 4 unreviewable; scope: G5 homography only; G5S geometry saved separately, not visually accepted.
- [iris_pretrained_v1](qc/iris_pretrained_v1.json): 6/6 pilot rows inspected; 0 fail, 6 unreviewable; scope: segmentation feasibility only; registration NOT RUN.
- [synthmorph_pilot_v1](qc/synthmorph_pilot_v1.json): runtime exit 1; 0 saved outputs; visual acceptance NOT ESTABLISHED.
- [synthmorph_pilot_v2_raw](qc/synthmorph_pilot_v2_raw.json): runtime exit 137; 0 saved outputs; visual acceptance NOT ESTABLISHED.
- [synthmorph_pilot_v3_raw192](qc/synthmorph_pilot_v3_raw192.json): runtime exit 0; 4 saved outputs; visual acceptance NOT ESTABLISHED.

## Sequential fixed MRI methods: exposed development pair

Official SynthMorph rigid and affine completed; joint extent-192 output reused. Each method uses the same T1w scan/template. No local landmark acceptance or validation release.

- [fixed_mri_v1](qc/fixed_mri_v1.json): saved development evidence; not additive to the frozen visual gate denominator.
- [fixed_mri_anatomy_v1](qc/fixed_mri_anatomy_v1.json): saved development evidence; not additive to the frozen visual gate denominator.
- [fixed_mri_geometry_v2](qc/fixed_mri_geometry_v2.json): saved development evidence; not additive to the frozen visual gate denominator.
- [brainmorph_pilot_v1](qc/brainmorph_pilot_v1.json): runtime exit 1; 0 output files; visual acceptance NOT ESTABLISHED.
- [brainmorph_pilot_v2](qc/brainmorph_pilot_v2.json): runtime exit 1; 0 output files; visual acceptance NOT ESTABLISHED.
- [brainmorph_pilot_v3](qc/brainmorph_pilot_v3.json): runtime exit 0; 10 output files; visual acceptance NOT ESTABLISHED.
- [brainmorph_anatomy_v1](qc/brainmorph_anatomy_v1.json): saved development evidence; not additive to the frozen visual gate denominator.
- [brainmorph_support_v1](qc/brainmorph_support_v1.json): saved development evidence; not additive to the frozen visual gate denominator.

## Fixed MRI cohort: affine preparation in progress

95/95 scans completed; 0 runtime failures; 334/334 QC sheets prepared.
Preparation is not visual acceptance. Original denominators and metric pause remain unchanged. [Live aggregate](qc/mri_validation_v1.json).

Agent visual review: 10/334 comparisons; 8 pass, 0 fail, 2 unreviewable. Kate acceptance PENDING; gate NOT MET.
[Review by stratum](qc/mri_validation_review_v1.json).

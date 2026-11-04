# Protocol: donor-averaging ladder on brain MRI (QTIM, 2 Nov 2026)

Recorded before any result of this experiment was computed. It is the brain counterpart of
`PROTOCOL_donor_ladder_fundus.md`, with the same fills, measures and summary.

## Decision

On 2 Nov 2026 Kate approved this run ("Yes, run it") after being told that MRI registration QC is not met. All
results are labelled **pre-QC**. The scans use the existing affine registration only. SyN is not used here, because
it can warp individual folding towards the template.

## Data

QTIM (`siblings-mri`, OpenNeuro ds004169): every T1w scan already prepared by `real_cases.prepare`.

- Preparation: SynthStrip, N4, affine ANTs registration to the nilearn MNI 2009a symmetric template, 1 mm grid.
- The run uses `prep/scan_*_brain.nii.gz` from `real-cases-run` and `real-cases-run-expanded`, deduplicated by
  participant and session.
- Counts: 244 scans, 221 people, 23 people with a second session, 168 families (53 sibling pairs and 4 co-twin
  pairs among the prepared scans).
- No new download and no new registration.

## Preprocessing (fixed)

- Skull-stripped volume, own brain mask = voxels above zero.
- Each scan is z-scored inside its own brain mask.
- A patch counts as valid for a scan if at least 95% of its voxels lie inside that scan's mask.

## Patches (fixed before results)

- 32 cubes of 16 mm, with a 4 mm context shell (24 mm window).
- Centres are drawn once with seed 20261102 from the positions where at least 90% of the window lies inside the
  template brain mask (nilearn MNI152 1 mm brain mask). The template mask, not the data, defines where patches
  can go.
- The two frozen boxes of the real-case protocol (`deep` and `parietal`) are reported separately.
- The same positions are used for every target and donor.
- Mirroring: reflection across the template midline, x to -x in millimetres.

## Cohorts

Every pool excludes the target's whole family.

| Cohort | Donors |
|---|---|
| `population` | session-1 scans of people from other families |
| `population_both` | the same plus their mirrored hemispheres |
| `cohort` | same sex, age within 2 years |
| `knn` | the k nearest donors by Euclidean distance on the context shell of that patch |
| `other_hemisphere` | the target's own scan, mirrored |
| `sibling` | the first different-age member of the family |
| `cotwin` | the first same-age member of the family |
| `repeat` | the same person's session 2 |

## Measures

- Identical to the fundus protocol: k in {1, 2, 4, 8, 16, 32, 64} plus all donors; error and spread per target
  and patch, normalized by V, the mean error of the full population mean.
- Fit of (1 - R2)(1 + 1/k) per cohort; r = 1 - (error/V)/2 for single same-person or relative donors.
- 95% intervals by bootstrap over families (2000 resamples).
- Targets are session-1 scans.
- The co-twin rung has only 4 pairs and is reported as descriptive only.

## Outputs

- Private, `C:\Projects\data\projects\identity-ladder-metric\donor-ladder-brain-20261102-v1\`: patch arrays and
  per-target rows.
- Public: `results/donor_ladder_brain.json` (aggregates only).

## Amendment 1 (2 Nov 2026, after a degenerate first run)

The first run read `scan_*_brain.nii.gz` as the image. That file is the binary brain mask, so every z-scored cube
was zero and every error was 0. No interpretable value was produced. Intensities now come from `scan_*.nii.gz`
(N4-corrected, registered), and the brain mask from `scan_*_brain.nii.gz`. A scan with no intensity variation
inside its mask now stops the run. The degenerate cache and rows were deleted. Nothing else changes.

## Amendment 2 (3 Nov 2026, bilateral and look-alike variants; recorded before these results)

Kate asked for three design changes (3 Nov 2026). They are new variants in `src/donor_ladder_brain_bilateral.py`;
the original variants are kept unchanged and re-run in the same pass, so old and new sit side by side. Still
**pre-QC, affine registration only**; the registration gate in `AGENTS.md` is not met.

- **Sides.** Every position is run twice, on the left and on the right hemisphere. The right side is the same
  experiment on the mirrored scan (x to -x mm), so cubes stay voxel-aligned. A position counts as left if its
  centre has x < 0 mm. `orig` is the side of the position as drawn (it must reproduce the original run). Each side
  is normalized by its own V; `L+R` pools both sides' rows per target (mean over patches and sides) and is
  normalized by the pooled V.
- **Look-alikes (kNN), three variants**, all within the same side and excluding the target's family:
  (a) `knn_oracle`: nearest donors by the distance between the target's true cube and the donors' cubes (an
  upper bound, it uses the answer); (b) `knn`: by the context shell (original); (c) `knn_mirror`: by the
  distance between the target's own mirrored contralateral cube and the donors' cubes.
- **Bilateral donor, two variants**: (a) `other_hemisphere`: the mirrored cube at the same location (original);
  (b) `mirror_block`: the target's shell is matched against shells in its own mirrored hemisphere, within a
  search window of +-8 mm (1 mm steps, 17^3 offsets) on each axis around the mirrored location; the cube at the
  best-matching offset is the fill. Candidate cubes must have >= 95% of voxels inside the brain mask, and the x
  offset is restricted so that the candidate overlaps the target's own cube in x no more than the unshifted
  mirrored cube does.
- **Strangers on both sides**: `population` k = 1 is the stranger's same-side cube; `stranger_other_side` is the
  first stranger (same draw order) with a valid contralateral cube, mirrored; `stranger_block` runs the same
  +-8 mm block search in that stranger's contralateral hemisphere. `stranger_block` is the control for
  `mirror_block`, which otherwise gains from choosing the best of 4913 candidates.
- Table rows use k = 16 for look-alikes and all donors for "same sex and age", as in the ISBI Table 1.
- Outputs: private rows and region arrays in `donor-ladder-brain-bilateral-20261103-v1\`; public aggregates in
  `results/donor_ladder_brain_bilateral.json`.

## Amendment 3 (3 Nov 2026, whole-hemisphere block search; recorded before these results)

Kate asked to search the whole contralateral hemisphere instead of a +-8 mm window. Two new single-donor fills,
computed in `src/donor_ladder_brain_hemisphere.py` and added to the same rows and summary; the +-8 mm variants stay.

- `mirror_hemi`: the target's shell (unchanged) is matched against every cube position of its own mirrored scan
  whose cube lies wholly on the target's side of the template midline (x <= 0 mm for left, x >= 0 mm for right,
  1 mm steps), i.e. anywhere in the contralateral hemisphere; candidates need >= 95% of cube voxels inside the
  brain mask; the cube at the best match is the fill.
- `stranger_hemi`: the same search in the contralateral hemisphere of the first stranger of the population draw
  (the stranger of `population` k = 1); control for `mirror_hemi`.
- Shell voxels outside the scan grid count as zero (outside the brain). Distances are computed with FFT
  cross-correlation in float64.

## Amendment 4 (4 Nov 2026, size grid and oracle searches; recorded before these results)

Kate asked for the dependence on patch size, shell thickness and their ratio, and for an oracle version of the
mirror search. Computed in `src/donor_ladder_brain_size.py` and `src/donor_ladder_brain_hemisphere.py`.

- **Grid.** Cube edge p in {8, 12, 16, 24, 32} mm and shell s in {2, 4, 8, 12} mm, 20 configurations. The 32
  random cubes keep the centres of the original 16 mm draw (start = centre - p/2); the frozen `deep` and `parietal`
  boxes keep their size and vary only s. Everything else (validity >= 95%, family exclusion, L/R/L+R views, seeds,
  +-8 mm block search with its overlap rule, k = 16 look-alikes) is as in Amendment 2. Voxels outside the scan
  grid count as zero (outside the brain).
- **Oracle searches** (upper bounds, they use the answer): `mirror_block_oracle` / `stranger_block_oracle` pick
  the +-8 mm offset whose cube is closest to the target's true cube (in the grid), and `mirror_hemi_oracle` /
  `stranger_hemi_oracle` do the same over the whole contralateral hemisphere (16 mm cubes, 4 mm shell only).
- Reported per configuration: L+R error over V with 95% family-bootstrap intervals for single donors, k = 16
  look-alikes, same sex and age (all) and the dataset mean. The shell does not change fills that do not use it
  (strangers, relatives, mirror at the same place, means); those vary with p only.
- Outputs: private rows in `donor-ladder-brain-size-20261104-v1\`; public aggregates in
  `results/donor_ladder_brain_size.json`; figure `results/figures/fig_size_shell_dependence.{png,pdf}`.
- **Trim (4 Nov 2026, Kate, for runtime; decided before the remaining configurations were computed).** Run time
  grows with the region volume (about 14 h left for the full grid). Kept: every configuration already finished
  (cubes 8 and 12 mm with all shells, 16 mm with a 2 mm shell), cube 16 mm with shells 4, 8 and 12 mm (with the
  boxes), cube 32 mm with shells 2, 4 and 8 mm. Dropped, status NOT RUN: cube 24 mm (all shells) and cube 32 mm
  with a 12 mm shell.

## Amendment 5 (4 Nov 2026, brain-wide variability maps; recorded before these results)

Kate asked for maps of how similar each part of the brain is to the same person, a relative and the mean, with
donors as unbiased as possible and no oracle or image-based selection. Computed in `src/donor_ladder_brain_maps.py`
from the z-scored volume cache of Amendment 4. Pre-QC, affine registration only: the maps mix anatomy with
residual misregistration and are exploratory.

- **Window**: 16 mm cube (as the random cubes), centred on every voxel of a 2 mm grid. Local MSE comes from a box
  filter of the voxel-wise squared difference. A target enters a voxel only if its window and the donor's window
  each have >= 95% of voxels inside the brain mask.
- **Donors (fixed rules, no selection on images)**: `repeat`, the same person's session 2; `mirror`, the person's
  own scan reflected at the same place, with |x| < 8 mm excluded (the window overlaps its reflection);
  `sibling`, the mean of the errors to all different-age family members; `stranger`, the exact expectation over
  every session-1 scan of other families, |x_i - mu|^2 + the pool variance (seed-free). Look-alikes, searches,
  oracles and co-twins (4 pairs) are not mapped.
- **Unit**: V, the error of the mean of all session-1 scans outside the target's family. Each map is sum of
  errors / sum of V over the same included targets at that voxel; r = 1 - ratio / 2. V itself is also mapped.
- **Reliability**: families are split once into two halves with the seed; the maps are recomputed per half and the
  voxel-wise correlation between halves is reported per map. A count map gives the number of targets per voxel.
- Outputs: private aggregate NIfTI maps in `donor-ladder-brain-maps-20261104-v1\`; public summary
  `results/donor_ladder_brain_maps.json` and figure `results/figures/fig_variability_maps.{png,pdf}`.
- **Pool and support** (added before any map was computed): the mean and the stranger pool use every session-1
  scan outside the target's family at every voxel; maps are reported only where >= 95% of session-1 scans have a
  valid window (common support), so the brain edge does not pull the mean towards the zero outside the mask.

## Amendment 6 (4 Nov 2026, same-sex-and-age map; recorded before this map was computed)

Kate asked for a map of the metadata-matched mean in place of the stranger column (the exact stranger map is about
2 everywhere by construction). `cohort`: the mean of all session-1 scans outside the target's family with the
same sex and an age within 2 years (the `cohort` rule of the cube ladder, all donors), computed from per
(sex, age) running sums; targets with fewer than 10 such donors are skipped. Same window, unit, support and
split-half reliability as Amendment 5; 1 - ratio is the share of individual variation that sex and age explain
locally (minus the pool's own noise). The stranger map stays in the JSON as a check. The figure shows the
cohort map instead of the stranger map.

## Amendment 7 (4 Nov 2026, coarse-grid ladder map with look-alikes; recorded before these results)

Kate asked for every fill of the main figure as a brain map, including k look-alikes, with contrast that shows
regional differences. Look-alikes need a neighbour search at every location, so they are mapped on a coarse grid
in `src/donor_ladder_brain_map_grid.py`; the other fills are computed there too, as a check on the 2 mm maps.

- **Centres**: 8 mm grid through the midline voxel (x = 0 mm), every centre whose 24 mm window has >= 90% of
  voxels inside the template brain mask (the rule of the random cubes); 2296 centres. Cube: 16 mm around the
  centre; shells of 2 and 4 mm from the same 24 mm window.
- **Fills at each centre**, with the cube-ladder rules (targets session 1, whole family excluded, cube validity
  >= 95%, at least 10 valid donors, V = error of the mean of the valid donors): `repeat`; `mirror` (own mirrored
  scan, same place; centres with |x| < 8 mm excluded); `sibling` (mean error over all valid different-age family
  members); `stranger` (exact expectation over the valid donors); `cohort` (mean of the valid donors of the same
  sex and age within 2 years, at least 10); `knn16_s4`, `knn16_s2` (mean of the 16 donors closest on the 4 mm or
  2 mm shell; selection by context only, no oracle).
- Each centre: sum of errors / sum of V over its targets; split-half reliability over the two family halves of
  Amendment 5. Figures: one common colour scale, and one with a per-map range (2nd to 98th percentile around the
  map's median) so that regional differences are visible; both show the split-half r next to each map.
- Outputs: private window cache and aggregate maps in `donor-ladder-brain-maps-20261104-v1\`; summary in
  `results/donor_ladder_brain_map_grid.json`; figures `results/figures/fig_ladder_maps_{common,contrast}.{png,pdf}`.

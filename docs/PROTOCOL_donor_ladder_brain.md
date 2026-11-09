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

## Amendment 8 (4 Nov 2026, independent halves, replication display, coverage; before the revised maps)

After a review of the exploratory maps (Kate agreed with it except on contrast):

- **Wording.** Colour = local MSE in the 16 mm window over the error of the dataset mean; the r under a map is
  the Pearson correlation between the maps of two family halves (reproducibility of the pattern), not a
  correlation with the donor.
- **Independent halves.** In Amendment 5 both halves shared the pool of the denominator (and of the stranger and
  cohort fills). The half maps are now computed with every pool restricted to the families of that half; the
  main map keeps the full pool. Template and registration remain shared, so a high split-half r is compatible
  with stable anatomy and with systematic residual misregistration alike.
- **Contrast.** Kate wants per-map contrast to show regional differences. In the stretched version, voxels where
  the two independent halves deviate from the map's median in opposite directions are shown faded, so the eye
  sees what replicates. The common-scale version is kept.
- **Coverage.** A figure with the repeat and mirror maps on the common scale next to the number of targets per
  voxel. The stranger map is a control of the expected level (about 2); sibling and cohort maps go to the
  appendix.
- The 8 mm look-alike grid of Amendment 7 used shared pools for its halves; an independent-half rerun is listed
  as pending until it is done.

## Amendment 9 (6 Nov 2026, outside the brain mask is stored as 0; flagged, not yet changed)

Found while fixing the same issue in the fundus block search (general protocol, Amendment 9.4). Volumes are
z-scored inside the brain mask and 0 outside, and 0 is a real value (the mean intensity), not a gap. Three places let
those zeros into an error or a distance:
- Block search (`search_terms`, `block_search` in `src/donor_ladder_brain_bilateral.py`, also used by the
  hemisphere, size and oracle runs): a candidate needs only its cube to be valid (`VALID` = 0.95 of voxels inside
  the mask); the shell used for matching, of the candidate and of the target, may reach outside the mask with no
  limit and enters the distance as zeros.
- Cubes in every ladder run: up to 5% of the cube may lie outside the mask, and those voxels count as 0 in the
  error of every donor.
- Maps (`src/donor_ladder_brain_maps.py`, Amendments 5-8) and the 8 mm grid: a window counts when 95% of it is
  inside the mask, so the same up-to-5% applies.
Near the cortex the outside of the brain is a real boundary (brain shape is individual), but encoding it as the mean
intensity is arbitrary; the shell term is the likely part of the Amendment 4 finding that thin shells favour boundary
smoothness. All brain numbers so far keep this behaviour and stay pre-QC. Planned change, with the nonlinear
registration rerun: matched and scored voxels must be inside the mask of both target and donor, the cube and
window rule becomes 100% inside (or a masked mean over the shared voxels, decided before rerunning), plus a
regression test like the fundus one.

## Amendment 10 (6 Nov 2026, metric panel tier 1 on the cubes; recorded before these results)

Kate: compute the CVPR metric panel on the brain ladder first, look at the result, then add the remaining 117
ses-02 scans. Data as Amendment 2 (221 targets, 23 repeats, the bilateral region arrays of
`donor-ladder-brain-bilateral-20261103-v1`, 32 random cubes + deep + parietal, sides L and R, view LR primary).
Pre-QC: affine only; the registration gate is not met (`results/registration/status.md`). Script
`src/donor_ladder_brain_metrics.py`, output `results/donor_ladder_brain_metrics.json`.

- Tier 1 = metrics defined on a 16 mm cube without a network or whole-brain context. Tiers 2 (persistence /
  Betti matching, Topology Distance; new library) and 3 (MS-SSIM, LPIPS, Brain-ID embeddings, FID / FWD,
  plausibility, latent geometry; need the fill inserted into the whole scan and pretrained weights) are not run here.
- Missing is never zero (general protocol section 8, closes the cube part of Amendment 9 for this run). Scored voxels
  E = target mask AND donor mask; a mean fill averages, at each voxel, only the pool members valid there (a voxel with
  no valid member is not in E). Squared error is also reported over all cube voxels (`mse_allvox`), which must
  reproduce `donor_ladder_brain_bilateral.json` (same donors, same seeds) before any other number is read.
- Metrics, each on E: MSE, RMSE, MAE, PSNR (range = target max - min on E), PSNR_fixed (range 8 z), SSIM (3D,
  Gaussian sigma 1.5 voxels, normalised convolution with E as weights so no outside voxel enters, constants for an
  8 z range), Pearson r of fill and target, ACC_dataset = Pearson r of (f - mu) and (x - mu) with mu the dataset mean,
  ACC_cohort (mu = same-sex, age +-2 y mean), amplitude ratio ||f - mu|| / ||x - mu|| (descriptive, not a closeness
  score), and lDDT-like local agreement: over voxel pairs within 2 voxels, the share with |delta f - delta x| below
  0.25, 0.5, 1 and 2 z, averaged over the four thresholds.
- Fills: repeat; mirror (other hemisphere, same place); sibling; co-twin; stranger (first of the population draw);
  16 strangers (for c); random means k = 2, 4, 8, 16; dataset mean; same-sex-and-age mean; 16 look-alikes by shell;
  16 oracle look-alikes (true cube; upper bound). Block searches are left out (Amendment 9).
- Controls (fills built from the truth, `PROTOCOL_real_cases.md` C7): shrink(d, lambda) = mu + lambda (d - mu) of the
  repeat and the mirror with lambda = r of that rung (r = 1 - mse_allvox ratio / 2, from the bilateral JSON, LR,
  random cubes), the MSE-optimal shrink; wrong detail = mu + the first stranger's residual made orthogonal to
  (x - mu) and scaled to ||x - mu||; noise = mu + Gaussian noise (seeded) with the SD of (x - mu) on E.
- Per metric the sign is set so that higher = closer to the target. Calibrated c = P(stranger worse) + P(tie) / 2
  against the 16 strangers. Position on the 0 = dataset mean, 1 = repeat scale = (m - m_mean) / (m_repeat - m_mean),
  ratio of means over targets with a repeat.
- Tests per metric, paired per target and side, averaged per family, one-sided Wilcoxon signed-rank on family
  means: T1 repeat beats the dataset mean; T2 the repeat beats its shrink with lambda = r (a metric that prefers the
  shrink rewards averaging); T3a the repeat beats wrong detail; T3b the repeat beats noise. Holm over all metric x
  test hypotheses of the dataset. Fewer than 10 families: descriptive only.
- Intervals: bootstrap over families, 2000 resamples, seed 20261102. Every value with its target and family counts.

## Amendment 11 (6 Nov 2026, ladder by blocks, expectations and the R scale; recorded before these results)

Kate's review of the Amendment 10 figures: the x axis must not presuppose the order it tests, magic numbers (k = 16)
go, and the expectations of an identity-faithful metric are written down before the numbers are read. Same data
and status as Amendment 10 (pre-QC, affine only, 221 targets, 23 repeats). Output replaces
`results/donor_ladder_brain_metrics.json` (Amendment 10 numbers stay in git history).

- Scale. For every metric, d = distance (lower = closer; similarities are negated). R(rung) = (mean d(rung) -
  mean d(repeat)) / (mean d(stranger) - mean d(repeat)); repeat = 0 (measurement noise only), stranger = 1 (no shared
  individuality). The stranger anchor is the per-target mean over the 16 seeded strangers. Means over targets per
  rung; anchors over all targets that have them; 95% intervals by bootstrap over families (2000, one resample set
  shared by all rungs so contrasts have intervals).
- Blocks on the x axis (order inside a block is by design, not by expected score):
  1. same person: repeat, own mirror (other hemisphere, same place);
  2. relatives: co-twin (descriptive, few), sibling;
  3. scaffold-selected (achievable by an inpainter): look-alike = the single stranger with the nearest scaffold
     (k = 1); mirror block search (own contralateral side, offset within +-8 mm chosen by the scaffold);
     stranger block search (the same search in the first stranger: control for the search itself);
  4. averages with no selection: stranger (k = 1), mean of 2, sex-and-age mean, dataset mean;
  5. oracle bounds (not achievable): oracle look-alike (single stranger nearest on the true cube), oracle mirror
     and oracle stranger block search;
  6. controls: repeat shrunk, mirror shrunk (lambda = r, Amendment 10), wrong detail, white noise.
  k = 4, 8, 16 and the 16-look-alike mean are dropped: the 1 + 1/k shape is known, and averaging belongs to block 4.
- Scaffold rule (closes Amendment 9 for this run). The scaffold distance uses only voxels inside the brain mask of
  both the target and the candidate: d = mean of (x - y)^2 over those voxels; a candidate (or a block-search
  offset) needs at least 50% of the scaffold shared and its cube >= 95% inside its own mask. The oracle distance uses
  the cube voxels inside both masks in the same way. Block search: x offsets restricted as in Amendment 2; the
  search is computed by FFT correlation of masked terms. Regression tests with known geometry, including a filler
  region outside the mask that would win if counted.
- Pool size. Look-alikes and oracles depend on the pool size (here 218 to 220 people from other families); it is
  reported with every result. For comparisons across datasets the pool will be subsampled (seeded) to the smallest
  dataset's pool; recorded when the second dataset is added.
- Expectations of an identity-faithful metric (pass / fail per metric, random cubes, LR view; boxes descriptive):
  E1 own mirror closer than a stranger: upper 95% bound of R(mirror) < 1.
  E2 sibling closer than a stranger: upper bound of R(sibling) < 1.
  E3 no reward for averaging: lower bound of R(mean of 2) >= 0.9 and of R(dataset mean) >= 0.9 (tolerance 0.1;
     the stranger level within 10%). Also reported: the slope R(dataset mean) - R(stranger).
  E4 no reward for shrinkage: lower bound of R(shrunk) - R(original) >= -0.1, for the repeat and the mirror.
  E5 wrong detail and white noise not closer than a stranger: lower bound of R >= 0.9.
  Measurements only (no expectation): own mirror against sibling; the scaffold block; the oracle block.
  Kate's answer "да" (6 Nov 2026) to the proposed defaults: mirror against sibling stays a measurement; tolerance 0.9.
- Reproduction: mse_allvox of repeat, mirror, sibling, co-twin, stranger, mean of 2, sex-age mean and dataset mean
  must equal the bilateral rows; the look-alike changes by design (new scaffold rule) and its agreement with the old
  k = 1 choice is reported.

## Amendment 12 (6 Nov 2026, protein-style metrics and two-way intervals; recorded before these results)

Kate (6 Nov 2026): add the protein-structure metrics (docs/REPRODUCTION_BACKLOG.md theme 7) in two groups, and
resample both families and positions. Same data and status as Amendment 11.

- Tolerance from repeat scans, no magic numbers. sigma_rep(position, side, target) = root mean squared voxel
  difference between session 1 and session 2 over voxels valid in both, pooled over the repeat pairs of the other
  families (leave-family-out; at least 5 pairs, else the metric is not computed for that cell).
- Group 1, new base metrics (scored on E like every metric; a rerun of the fills):
  tm_like = mean over E of 1 / (1 + (|f - x| / sigma_rep)^2) (TM-score idea: all voxels, a continuous kernel instead
  of cutoffs, the scale from a reference not from the metric); gdt_like = mean over thresholds t in sigma_rep x
  {0.5, 1, 2, 4} of the share of E with |f - x| < t (GDT idea); lddt_rep = lDDT-like with thresholds sigma_rep x
  {0.5, 1, 2, 4} (lDDT tolerance from an ensemble of reference scans); rmse_rep = RMSE / sigma_rep (noise-normalised,
  comparable across positions and mask sizes; the size-normalised RMSD idea).
- Group 2, calibrations on the saved rows, for every base metric (no rerun):
  Tnorm z = (s - mean of the 16 strangers) / SD of the 16 strangers, per target, position and side (impostor-cohort
  normalisation); EVT P-value = upper tail probability of z under a generalised extreme value distribution fitted to
  the pooled stranger z of the same position and side (TM-score significance against random pairs); P(same | z) =
  posterior that the fill is the same person, from Gaussian densities of repeat z and stranger z per position, prior
  0.5 (posterior of the same fold given the score). Reported per rung as means over targets; c and R stay.
- Intervals: for the random cubes, two-way bootstrap, families and positions resampled independently (2000, seed
  20261102); the family-only interval is also reported. Boxes: families only (one position each). Expectations
  E1-E5 are read from the two-way intervals.
- Later, not in this amendment: a reduced panel of weakly correlated metrics for the main figure (Kate, 6 Nov 2026).

## Amendment 13 (6 Nov 2026, the reference reality is "do not lie"; shrinkage is a separate experiment)

Kate chose (6 Nov 2026), after the two options were laid out: the reference is reality B, "do not lie": correct
individual detail is good, wrong detail is worse than none. Consequences, recorded before the summary is redrawn:

- Target ladder (R, repeat = 0, stranger = 1): a fill with no individual information (a mean) sits between them,
  closer for more averaging (the squared-error shape 1 + 1/k is the reference: mean of 2 above the dataset mean);
  a stranger's wrong detail sits at 1; wrong detail of the same amplitude at 1; white noise, which is wrong and not
  anatomy, above 1. Relatives and the own mirror below 1.
- Expectations replacing E3-E5 of Amendment 11 (E1, E2 unchanged):
  E3B averages better than a stranger and ordered by k: upper 95% bound of R(dataset mean) < 1 and of
      R(mean of 2) < 1, and R(dataset mean) < R(mean of 2) < R(stranger) (point estimates);
  E5B wrong detail not better than a stranger: lower bound of R(wrong detail) >= 0.9;
  E6B white noise worse than a stranger: lower bound of R(white noise) > 1.
  The Amendment 11 E3 and E4 stay in the JSON as descriptive numbers.
- In reality B the line between honest hedging and lost detail is not a pass / fail property of the ladder, so the
  blends with the mean (repeat and own mirror, lambda = r) leave the ladder figure and get their own table: per
  metric, R of the donor, R of the blend, their difference with its interval, and whether the metric prefers the
  blend ("rewards fading the right detail"). An oracle blend (best single person, lambda = its r) is added to that
  table in the next rerun.
- Reading the audit under B: squared error follows the target ladder except that it cannot tell white noise from a
  stranger; this is the main pixel-metric failure B exposes. The Amendment 11 statement that averages "should" sit at
  the stranger level belongs to reality A and is not used.

## Amendment 14 (6 Nov 2026, an independent reality-B target; recorded before it is scored)

Problem found in Amendment 13: its target ladder took each single rung's shared fraction r from the measured squared
error, so squared error matched the target by construction. Kate: build an independent target. The target now
follows from the axioms of reality B alone and uses no measured metric value:

- Scale: repeat = 0 (measurement noise counts as no error), unrelated person = 1 (full wrong detail).
- A1 "no information is half way": a fill with no individual detail, neither right nor wrong (the population
  mean), sits at 0.5.
- A2 "wrong detail scales with its amount": the mean of k unrelated people carries wrong detail whose variance is
  1/k of one person's (independent deviations average), so R(mean of k) = 0.5 + 0.5 / k: k = 1 gives 1, k = 2 gives
  0.75, all people give 0.5. This is a property of averaging, not of a metric.
- A3 wrong detail of the same amplitude as a person's (stranger residual, orthogonal to the target's) sits at 1.
- A4 white noise is wrong and not anatomy: strictly above 1 (no value is set; only the side).
- Rungs that carry an unknown amount of correct detail (own mirror, sibling, co-twin, look-alike, oracle) get no
  value, only order: each below 1; oracle look-alike not above the look-alike; repeat below all of them.
- Sex-and-age mean: treated as a population mean (0.5); its small cohort-specific part is ignored.

Score per metric (random cubes, both hemispheres, two-way intervals of Amendment 12):
- value fit: root mean square of R - target over the rungs with a value (dataset mean, mean of 2, sex-age mean,
  wrong detail), and each rung's deviation with its interval;
- side constraints: white noise above 1 (lower bound > 1); each partly-correct rung below 1 (upper bound < 1);
  oracle look-alike <= look-alike (point estimates).
The Amendment 13 target (squared-error model) stays in the JSON history as "MSE-derived target", not used for
scoring.

## Amendment 15 (6 Nov 2026, segmentation-based metrics on the boxes; recorded before any fill is segmented)

Approved by Kate on 6 Nov 2026 ("можно считать сегментации"), after the draft below was written and before any
volume was built or segmented. At most 2 segmentations in parallel (memory shared with other jobs).
Local Docker Desktop failed to start (6 Nov 2026); Kate gave an explicit yes (6 Nov 2026) to running this step with
the QTIM volumes on the project's GCP machine (the MRI avatar audit VM), if it has room.
Kate chose FastSurfer segmentation with DKT parcellation on the GPU (6 Nov 2026), replacing the D8 timing choice
below, for the oracle, every core-rung volume and the blur sweep volumes. No FreeSurfer surface reconstruction
(recon-surf) in this paper (Kate, 6 Nov 2026): it runs on CPU only, about an hour per volume, and an honest
within-subject comparison would need the longitudinal stream; that is parked for a separate paper. Meshes, when
used here, are derived from the FastSurfer labels (marching cubes, locally in each box), so they are fixed by the
labels and add no run-to-run noise. Time per volume is measured on one scan first and sets the number of targets
(budget written here before the run). Mesh-based scores (surface Euler characteristic and components, BrainPrint,
varifold or vertex distance to the oracle mesh, label-based thickness) are defined in a later amendment before they
are computed; this amendment scores labels only (aseg Dice, DKT Dice, accuracy).
Input clip (6 Nov 2026, found at the timing step, before any score): 9 of 17 targets carry a few extreme intensity
outliers (max over the 99.9th percentile of nonzero voxels above 10 000), and FastSurfer's conform step then maps the
whole brain to one uint8 bin and returns an empty segmentation. Fix, applied by the segmentation runner and not to
the built volumes (which stay identical to the metric rows): every volume of target t is clipped to [0, c_t], with
c_t the 99.9th percentile of the nonzero voxels of t's oracle, one c_t per target, float32 kept. Oracle and rungs
get the same transform, so labels stay comparable. The runner logs c_t and, per box and rung, the number of scored
fill voxels above c_t; if that share exceeds 1% of a box's scored voxels the rung is reported and flagged.
Distance metrics (Amendments 11 to 14) are unaffected; they were computed on the unclipped volumes.
Clip result (6 Nov 2026): 0 scored fill voxels above c_t in every target, box and rung; no rung flagged.
Number of targets, fixed before the run (Kate, 6 Nov 2026, "All 17 targets"): N = 17, all eligible targets, no
subset. Basis: 75 to 77 s per volume measured on the L4 (two clipped oracles, 2 CPU threads under load), about 221
volumes, about 2.5 h with 2 containers in parallel (3.6 GB GPU each). The run starts after Kate's verdict on the
visual QC of one target.
Visual QC of one target (7 Nov 2026; Kate delegated the verdict to the analysis session before going offline): all 13
volumes segmented, labels placed correctly, outside-box Dice 0.98 to 1.00, no label band along the box faces, blur
sweep monotone in both boxes. Passed; the 17-target run was released. Observation recorded before any group score:
in the deep box FastSurfer reconstructs the basal ganglia and thalamus from the surrounding context, so noise and
dataset-mean fills keep near-oracle shapes (aseg Dice about 0.65 on this target against 0.96 for the repeat); the
parietal box has the wider range (0.94 against 0.55).

Device (7 Nov 2026, before any rung is scored): no L4 GPU was available in the allowed EU regions, and Kate chose her free local CPU first. All 224 volumes run on CPU, one device for all: native FastSurfer v2.5.3 (the pinned image's tag) in WSL, CPU torch 2.7.1, the same flags and the same clipped inputs. Device check on the t009 oracle: CPU and GPU labels differ in 722 of 1.46 M labelled voxels, mean per-label Dice 0.9994 (min 0.9957). About 23 min per volume at 4 threads; timing.json records device and threads. Per-volume times are not a clean D8 runtime figure while other local jobs share the CPU.

Device, remainder (9 Nov 2026, before any of the remaining 145 volumes is scored): Kate, 9 Nov (quoted by the ISBI paper session): "Надо ещё fastserfer доделать". 79 of 224 volumes are done on the laptop; the remaining 145 (about 30 h of laptop CPU) run on a cloud CPU VM in Kate's own GCP project, so the laptop stays free for the metric runs. Same device class (CPU, no GPU): native FastSurfer v2.5.3, CPU torch 2.7.1, the same flags, the same clipped inputs (sha256 of each staged file checked on the VM against the local copy), the same resumable runner (`local/fastsurfer_native_run.sh`). Region and machine: the cheapest allowed EU region by catalog price (STUDIES-038; 9 Nov table: europe-west1 n2 spot 0.0965 USD/h per 8 vCPU), spot with stop on preemption, rerun skips DONE volumes. Reproduction check before the remainder: 2 volumes already done on the laptop (t009_oracle, t016_oracle) are segmented again on the VM; every label present in either must have Dice >= 0.995 on both volumes, else the remainder does not start and the mismatch is reported. Only labels and logs come back; the VM and its disk are deleted at the end, also on failure, and the actual cost is recorded. Per-volume timing on the VM is recorded with the device in the run JSON.

Device, remainder, outcome (9 Nov 2026): reproduction check passed (t009_oracle 6 voxels differ, t016_oracle 8; minimum per-label Dice 0.99974 over 96 labels). The 145 remaining volumes ran on n2-standard-32 spot in europe-west1-b, 8 runners x 4 threads, 08:25 to 11:33 UTC, 0 failed, median 575 s per volume; 224 of 224 volumes are done. The labels of the 2 reproduction volumes are kept from the laptop. The VM outputs were downloaded to the private folder and removed from the bucket; all study data was wiped from the VM disk at 12:58 UTC. The VM itself was then handed to the face metrics session for a face-only job, and that session deletes it. FastSurfer VM time 06:50 to 12:58 UTC (6.1 h, including setup and the reproduction check): about USD 2.4 at the spot catalog price (estimate from the catalog price; the billing export lags by a day).

Kate: add segmentation metrics, kept simple. Same data and status as Amendment 11 (pre-QC, affine only).
Script `src/donor_ladder_brain_seg.py`; private volumes and labels in
`donor-ladder-brain-seg-20261106-v1`; public aggregates `results/donor_ladder_brain_seg.json`.

- Where: the deep and parietal boxes only, as drawn (left hemisphere, scan frame). The two boxes do not overlap and
  lie about 25 mm apart (14 mm in y, 21 mm in z), so one volume per (target, fill) carries the fill in both boxes; each box is scored on
  its own.
- Fills: the core rungs of the main figure only, built exactly as in `run_position` (same donors, same seeds):
  repeat, own mirror, sibling, stranger, look-alike, oracle look-alike, mean of 2, dataset mean, white noise.
  The fill (z units) is put back into intensity with the target's own brain mean and SD, on box voxels inside the
  target brain mask where the fill is valid; box voxels inside the target mask where the fill is missing get the
  dataset mean (counted, never scored); everything outside the boxes is the target's own scan.
- Targets: targets with a valid repeat in both boxes, one per family, sorted by participant id; an evenly spaced
  subset (np.linspace over that list) whose size follows from the measured time per volume and a 4 hour budget,
  minimum 5. The size is written here before segmentation starts.
- Segmenter (DECISION_TREES D8): FastSurfer (`--seg_only`, CPU, pinned digest in `src/real_cases.py`) if one
  volume takes 5 minutes or less on this machine, otherwise SynthSeg (`--keepgeom`). Timed on one target's
  unaltered scan before any fill; timing is not a result. Labels are resampled to the template grid with nearest
  neighbour. Oracle = the same segmenter on the unaltered target.
- Scores per box, on S = box AND target mask AND fill valid: Dice per label with at least 50 oracle voxels in S,
  their mean (dice_mean), and voxel accuracy. Topology (Euler number, components) is not run here.
- Blur sweep first (D8 branch test): the unaltered target blurred with a Gaussian of sigma 1, 2, 3, 4 mm, inserted in
  both boxes inside the target mask, segmented, Dice against the oracle. The fill results are read only if the
  bootstrap 95% interval of the Dice drop at 3 mm lies above 0 in that box; otherwise that box is reported as
  uninformative for this segmenter.
- Scale: d = 1 - dice_mean (and 1 - accuracy); R as Amendment 11 with repeat = 0 and the single stranger
  (k = 1, the first population draw) = 1, since the 16-stranger anchor would cost 16 more volumes per target.
  Intervals: bootstrap over targets (one per family), 2000, seed 20261102. With so few targets the numbers are
  descriptive; no pass / fail against the Amendment 14 target is claimed, the deviations are only shown.
- Reproduction: for every inserted fill, the squared error on E against the target must equal `mse` of the same
  (target, position, side, fill) in the Amendment 12 rows; otherwise nothing is segmented.
- A failed segmentation is retried once, then counted as failed for its rung (D8); more than 10% failed switches the
  segmenter.

## Amendment 16 (6 Nov 2026, drawn maps use a masked Gaussian window, not a cube; recorded before the rerun)

Kate caught square local-error maps again (brain lesson L8, rule STUDIES-019). The brain-wide variability maps of
Amendments 5, 6 and 8 averaged the squared difference over a 16 mm cube, which draws squares around any focal error.
Change in `src/donor_ladder_brain_maps.py`: the local error is a masked Gaussian average, sd = 16 mm / sqrt(12) (the sd
of the 16 mm box, about 4.6 mm), counting only voxels inside both scans' brain masks (target mask alone for the
stranger and cohort terms); the 2 mm output grid stays. Validity stays a box: a voxel is kept when at least 95% of the
16 mm cube around it lies inside the masks (a fraction, not a drawn value). Everything else is unchanged (donors,
V, halves, support, midline). On one target the change is visible only in shape (a square blob becomes round); the
mean over kept voxels differs by 3%. Figures from these maps are redrawn after Kate approves the before/after slice.
`tests/test_no_box_windows.py` fails on any box filter in map code not marked as a validity fraction.

## Amendment 17 (6 Nov 2026, HACA3+ contrast harmonization as a sensitivity analysis; recorded before any output)

Why: repeat scans differ by -0.4 to -0.5 z in the deep box between sessions (Amendment 10 notes), so part of
d_repeat is scanner/session contrast, not anatomy. The HACA3+ pilot of 4 Nov (one repeat pair, identity-ladder-metric
`docs/REGISTRATION_QC_HANDOFF.md`) reduced that gap visibly; Kate asked to apply it to the ladder (6 Nov 2026).
Method: HACA3+ (Hays et al., MIDL 2026), upstream commit 4ec722b8, weights checked against the upstream SHA-256
pointers. Inputs are the 244 prepared volumes the ladder already uses (221 ses-01, 23 ses-02; N4, ANTs affine to MNI
2009a symmetric, 1 mm), centred crop/pad to the model grid 192 x 224 x 192 with a check that no brain-mask voxel is
cropped; settings as in the pilot (background removal off, 56 batches per orientation, fusion output). No extra clip:
HACA3+ normalises by the 95th percentile of the volume and clips at 5 times it.
Target contrast (Kate, 6 Nov 2026): one fixed target for every scan, the cohort median. For each ses-01 scan the
model's own target encoder gives theta (2) and eta (2) exactly as for a target image (axial slices 100-119 of its
224-grid, mean over slices) and its norm value (95th percentile); the target is the per-component median over the 221
ses-01 scans, frozen in `haca3_target.json` before any harmonized image is produced. ses-02 scans do not enter it.
Output: the fused image is put back on the 197 x 233 x 189 grid; the brain masks are unchanged and everything outside
them stays missing; the ladder z-scores in the mask as before.
Stage 1 (gate): the 23 repeat pairs (46 scans). QC sheets per pair, deep and parietal box, before and after, one
shared window per pair; numbers: mean z difference ses-02 minus ses-01 in each box, before and after. Kate's verdict
before stage 2. Stage 2: all 244 scans, the ladder rerun into a new folder (v4), v3 kept.
Expectations, stated now: H1 the between-session shift in the deep box shrinks; H2 the order repeat < own mirror <
stranger is kept; H3 (the risk) harmonization pulls individuals towards the mean: c = P(stranger worse than repeat)
drops, or d_stranger falls relative to d_repeat. H3 holding would be the regression to the mean inside the
harmonizer, reported as a finding, not hidden. Results are labelled pre-QC and sensitivity, never replacing v3.

## Amendment 18 (7 Nov 2026, the 8 mm look-alike grid uses Gaussian weights inside the cube; recorded before the rerun)

Kate approved smoothing the last square column ("16 look-alikes*", Amendment 7) after Amendment 16 removed the
squares from the 2 mm maps. Two sources made it square: each centre averaged the squared error uniformly over its
16 mm cube (lesson L8), and the 8 mm grid was drawn with nearest-neighbour upsampling.
Change 1, analysis (`src/donor_ladder_brain_map_grid.py`): at every centre each error, V and the stranger term is a
weighted mean over the 16 mm cube, with Gaussian weights of sd 16 mm / sqrt(12) (about 4.6 mm, as in Amendment 16),
truncated at the cube and normalised to sum 1. The look-alikes are still chosen on the 2 or 4 mm shell outside the
cube, so selection stays on context only and the weights never touch it. The stranger term stays exact (V plus the
weighted donor variance). Centres, validity (95% of the cube inside the mask), donors, halves, midline and the window
cache are unchanged. As before, the cube values are taken as cached; voxels outside the brain mask (at most 5% of a
valid cube) are not separately masked, unlike the 2 mm maps; this is a known difference, unchanged here.
Change 2, drawing (`src/donor_ladder_brain_ladder_maps_figure.py`): the grid is upsampled to the 2 mm frame by
normalised trilinear interpolation (values and a validity weight interpolated separately, kept where the weight is at
least 0.5), not nearest neighbour. This changes no number; split-half r and the whole-map MSE under the column are
still computed on the grid centres.
The uniform-cube outputs of 4 Nov are kept: private NIfTIs in `grid_box_20261104/` of the private maps folder and the
JSON in git (a425727). Expectation, stated now: pooled ratios move by a few per cent (the Gaussian puts more weight
on the cube centre, where the shell context predicts less), split-half r stays above 0.95 for the look-alikes.
Result (7 Nov 2026, 558 min on a shared CPU): pooled ratios with Gaussian weights against the uniform cube: repeat
0.281 to 0.278, mirror 1.527 to 1.535, sibling 1.842 to 1.843, stranger 1.991 (unchanged), cohort 1.007 (unchanged),
look-alikes on the 4 mm shell 0.735 to 0.753 and on the 2 mm shell 0.696 to 0.719 (+2.4% and +3.3%, as expected).
Split-half r of the look-alikes 0.983 and 0.982 (above 0.95, as expected). Grid against the 2 mm maps at the grid
centres: r 0.986 (repeat), 0.996 (mirror), 0.986 (sibling), 0.987 (cohort); the stranger ratio is about 2
everywhere by construction, so its r is near 0 and carries no information. The column is drawn without squares; the
independent half maps now exist for the grid, so its contrast panel is faded like the others.

## Amendment 19 (7 Nov 2026, IXI as a second brain cohort for the non-family rungs; recorded before any IXI scan is prepared)

Kate chose IXI (7 Nov 2026, "IXI") as the second brain dataset next to QTIM. Why: QTIM is one site and young adult
twins; IXI is 3 London centres and adults 20 to 86 years, so it tests whether the ladder order and "the dataset mean
beats related anatomy" hold in an older, multi-site population. IXI has no repeats and no families: the repeat and
sibling rungs are not available there, reported as such, never filled.
Data: registry id `ixi` (CC BY-SA 3.0, verified 2026-10-18), local copy
`datasets/neuroimaging/ixi/version-unverified/raw`. Aggregates read on 7 Nov 2026: 581 T1 files (Guys 322, HH 185,
IOP 74); `IXI.xls` has 619 rows for 593 IDs; 563 T1 scans have an ID with an age. Eligible: one T1 per ID with a
non-missing sex and age; an ID whose duplicate rows disagree on sex or by more than 1 year on age is excluded and
counted (D7), never resolved by hand. No other selection (STUDIES-025).
Preparation, identical to QTIM (`src/real_cases.py` strip and prepare): SynthStrip brain mask (model version 1, as in
the pinned FreeSurfer 8.0.0 image used for QTIM), N4 on a 2 mm copy, ANTs affine registration of the masked scan to
the nilearn MNI 2009a symmetric template (same iterations, seed 123, template brain mask), 1 mm output, own mask
carried by nearest neighbour; z-scored in the own mask by the ladder. If SynthStrip runs outside the pinned image,
its masks are first checked on 5 already prepared QTIM scans: brain-mask Dice against the QTIM masks must be at
least 0.99 for every scan, else stop and report.
QC before any ladder number (repository gate and STUDIES-018): the D7 automatic rules (unreadable, box under 95%
inside its own brain mask, template correlation below median - 3 MAD flagged for review) and a visual sheet of all
flagged scans plus a random 30 (seed 123), axial, coronal and sagittal through both boxes. Kate's verdict, or the
analysis session's if she delegates it, before stage 2.
Stage 2, the cube ladder of Amendments 10 and 11 on the same two frozen boxes and the same random positions, IXI
only (targets and donors from IXI): own mirror, stranger (exact expectation), 16 look-alikes by the 2 and 4 mm shell
and by the true box (oracle), same sex and age (within 2 years, at least 10 donors), dataset mean (leave-one-out).
Site sensitivity: the stranger, same sex and age and dataset-mean rungs also with donors from the target's own
centre only. Units and metrics as in QTIM; the R scale has no repeat anchor in IXI, so it is reported on the QTIM
anchors only, flagged as cross-cohort.
Expectations, stated now: E1 dataset mean below stranger in both boxes (QTIM: about 1 against 2 by construction);
E2 own mirror between them in the deep box and near the stranger in the parietal box, as in QTIM; E3 same-centre
donors lower V and every averaged rung by a few per cent, without changing the order; E4 older targets have larger
V (ventricles, atrophy) but the same order. A reversal of E1 or E2 is reported, not explained away.
Private outputs in `projects/identity-ladder-metric/ixi-ladder-20261107-v1/`; the repository holds aggregates only.
Status: pre-QC, affine registration only, like QTIM.

## Amendment 20 (7 Nov 2026, metric panel tiers 2 and 3 on the frozen boxes, CPU part; recorded before any result)

Kate approved computing the whole 26-metric panel for the brain, weights and libraries included (7 Nov 2026, in the
metrics-explorer session: "скачивай веса конечно считай все что запланировали", "библиотеку ставь"). This amendment
defines the part that runs on this CPU-only machine; the rest is listed at the end with its blocker.
Scope: the two frozen boxes (deep, parietal; positions 32 and 33 of Amendment 10), both sides, the same 221 targets,
donors, fills, controls, seeds and scored voxels E as Amendment 10 and 11 (`src/donor_ladder_brain_metrics.py`,
the new metrics are added to the same rows). The 32 random cubes are not scored for these tiers: one pass over them
costs days on this CPU (timed on 7 Nov: persistence plus Wasserstein about 0.25 s per pair on a 16 mm cube of noise,
LPIPS about 15 ms per slice). Private rows in `donor-ladder-brain-metrics-20261107-v4/`; v3 stays.
- MS-SSIM (group A): three scales (the box, then twice halved). Downsampling is the mean of the valid voxels of each
  2 x 2 x 2 block; a coarse voxel is in E when at least half of its 8 children are. At each scale the masked SSIM
  terms of Amendment 10 (normalised convolution, sigma 1.5 voxels, constants for an 8 z range); contrast-structure at
  every scale and luminance at the coarsest, each averaged over E and clipped at 0, combined with the first three
  Wang et al. weights (0.0448, 0.2856, 0.3001, as in piq 0.8.0) renormalised to sum 1. Higher = closer.
- LPIPS (group A), 2.5D: AlexNet, lpips 0.1.4 (linear layers v0.1). The three central orthogonal slices of the box;
  z mapped to [-1, 1] by z / 4 and clipped, the grey slice repeated to 3 channels, bilinearly upsampled so the short
  side is 64 px. A slice is scored only if every voxel of it is in E (no filler enters the network); the value is
  the mean over the scored slices, NaN when none is scored, and the count of scored slices is kept. Lower = closer.
- Amplitude against the matched mean (group A, descriptive like `amplitude`): ||f - mu_c|| / ||x - mu_c|| on E,
  mu_c the same-sex-and-age mean.
- Intensity persistence (group B): sublevel-set cubical persistence of the box (gudhi 3.13), voxels outside E set to
  +inf so they never enter; diagrams in dimensions 0, 1 and 2, finite intervals only (essential classes dropped,
  counted); distance = sum over dimensions of the 1-Wasserstein distance (internal p = 2, POT backend).
  Lower = closer.
- Tests, c, scale and intervals as Amendments 10 to 12, Holm over all metric x test hypotheses of this run, reported
  next to tier 1, never merged into its Holm family after the fact.
Before the full run: a smoke test on 3 targets with timing, and a visual sheet of one target (target, repeat,
dataset mean, noise) with the LPIPS slices and the persistence diagrams, looked at before the full run.
Expectations, stated now: MS-SSIM orders like SSIM; LPIPS ranks the dataset mean worse than SSIM does (a blurred
mean is perceptually far) but the repeat still best; persistence ranks the dataset mean and the shrink far from the
target (smoothing removes small features) and noise far as well (it adds features), so it may pass T2 where pixel
metrics fail. Any of these failing is reported.
Not in this amendment, each with its blocker: Betti matching in 3D (Betti-Matching-3D needs a C++ build; no compiler
here, build on the project's Linux VM); identity embeddings (angle, norm, PFE, PLDA) on a brain encoder (Brain-ID
download approved; needs the fill inside the whole scan, 2 boxes x about 221 volumes, CPU-feasible after FastSurfer
frees the CPU, defined in its own amendment; PFE needs an uncertainty head the encoder may not have); FID, FWD and
Topology Distance (set-level, per rung; own amendment); plausibility and latent geometry (pullback, Fisher-Rao,
LDDMM): need a generative brain model and realistically a GPU.
Smoke test (7 Nov 2026, 3 targets with a repeat, both boxes, 2 workers on a CPU shared with FastSurfer): 5.9 min,
so about 7 h for the 221 targets. The QC sheet of one target was looked at: slices and fills are the right ones; in
the deep box LPIPS orders repeat 0.046 < dataset mean 0.27 < noise 0.48. Two observations, recorded before the full
run and not acted on (the definitions above stay): (1) in the parietal box the central slices often leave the brain
mask, so LPIPS is missing for 24 of 64 parietal rows (0 scored slices); (2) persistence W1 is not invariant to an
intensity offset: on this target the repeat (box mean 0.48 z against 0.95 z for the target, the between-session
shift of Amendment 17) is farther from the target (173.7) than the dataset mean (125.4), which has few, short-lived
features; over the 3 targets the deep-box means are repeat 96.7, mirror 51.1, dataset mean 112.0. Persistence may
fail T1 in the deep box for this reason; that will be reported, not repaired after the fact.

Result (8 Nov 2026, full run, 221 targets, both boxes, L+R; pre-QC). `results/donor_ladder_brain_metrics_t23.json`,
figure `results/figures/fig_brain_metrics_t23.{png,pdf}`; summary added to `src/donor_ladder_brain_metrics_t23.py`
(`--summary-only`) after the run, with the definitions above unchanged. Holm over 24 hypotheses (3 metrics x 4 tests x
2 boxes). T1 (repeat beats the dataset mean), T3a (wrong detail) and T3b (noise) pass for all three metrics in both
boxes; T2 (repeat beats the repeat shrunk to the mean) fails in the deep box for all three and passes in the parietal
box. R (0 = repeat, 1 = stranger), deep box: dataset mean 0.50 (MS-SSIM), 3.31 (LPIPS), 1.06 (persistence); noise
0.79, 5.67, 28.3. Expectations as stated: LPIPS ranks the dataset mean far worse than SSIM-type metrics; MS-SSIM rates
noise closer than a stranger (E6B fails). Not as expected: persistence puts the own other side and the look-alikes
closer than the repeat in the deep box (R -0.55 and -0.2 to -0.3; possibly the session intensity offset of observation (2) above, not tested), and
does not pass E2 there; parietal LPIPS is missing for 7013 of 11923 rows (59%), counted, never filled.

## Amendment 21 (8 Nov 2026, window size in object units; recorded before any result at the new sizes)

Kate, 8 Nov 2026, metrics-explorer session: "да переделай честно". One window in mm covers a different share of each
modality's object, so the window is now set as a fraction f of the object's own extent, with the same fractions for
brain, face, fundus and iris (cross-modality rule of the metrics-explorer session, 8 Nov 2026).
- **Object extent.** L = cbrt(mean brain-mask volume) over the 244 prepared scans (221 ses-01 + 23 ses-02) in the
  analysis space (MNI 2009a symmetric, 1 mm, affine): mean 2046 ml (sd 67 ml), **L = 127.0 mm**. This is the
  registered space, not native space (the template mask itself is 1883 ml); the metrics are computed in this space,
  so this is the L that matters.
- **Sizes.** Cube edge w = f x L rounded to the voxel: f = 1/16 -> 8 mm, f = 1/8 -> 16 mm, f = 1/4 -> 32 mm. The
  current 16 mm window is f = 16 / 127.0 = 0.126, i.e. the f = 1/8 point: the tier-1 run (Amendments 10 to 13,
  `donor-ladder-brain-metrics-20261106-v3`) and the running tier-2/3 run (Amendment 20) are the reference and are
  not redone.
- **Shell and search.** The context shell keeps its current ratio to the cube, s = w/4 (2, 4, 8 mm; the 3D analogue
  of the 2D ring w/2), so f = 1/8 is exactly the reference configuration. The +-8 mm block-search half-window is a
  tolerance for registration misalignment, not an object fraction, and stays 8 mm at every size (as in Amendment 4).
  SSIM sigma and the lDDT radius are voxel-scale metric constants and stay unchanged.
- **Positions.** The same 32 random centres as the 16 mm draw (start = centre - w/2). A centre is kept at size w only
  if >= 90% of its window (w + 2s) lies inside the template brain mask and the window lies inside the grid (the rule
  of `donor_ladder_brain.positions`); dropped centres are counted and reported, never replaced or filled. Target and
  donor validity (>= 95% of the cube inside the scan's brain mask) is unchanged. The frozen `deep` and `parietal`
  boxes (32 x 32 x 24 mm, f about 0.23) are anatomical regions and are not rescaled: rescaling them would change the
  anatomy asked about. They appear only in the reference runs.
- **What runs at the new sizes.** The tier-1 panel exactly as Amendments 10 to 13 (metrics, fills, controls, tests,
  c, R, two-way intervals) on the random cubes at 8 and 32 mm, in `src/donor_ladder_brain_metrics_size.py`, with the
  cube size, shell and half-window passed as explicit arguments. Tier 2/3, set-level (FID, FWD, Topology Distance)
  and model metrics: only at f = 1/8.
- **Check.** mse_allvox of the shared donors must reproduce the Amendment 4 rows of the same configuration
  (`rows_p8_s2`, `rows_p32_s8`) on the kept positions, as the 16 mm run reproduces the bilateral rows; if it does
  not, no summary is written.
- **Reported.** L; f and w per size; kept positions per size; per metric the rung order and R at each size; rank
  stability = Spearman correlation of the rung order (family-bootstrap mean of each fill, random cubes, L+R view,
  fills present at all three sizes, the reference strangers excluded) between each pair of sizes. A claim holds only
  if it holds at all three sizes; otherwise it is stated with its scale.
- **Expectations (stated now).** The repeat is best and noise worst at every size and on every metric that passes
  T1 at 16 mm. The order of the middle rungs (mirror, sibling, look-alike, single stranger) may change with size;
  the dataset mean is expected to look relatively better at 32 mm (larger cubes average more structure, which a mean
  keeps) and worse at 8 mm. Spearman between adjacent sizes is expected >= 0.8 for MSE-type metrics; lower values
  are reported as scale dependence, not repaired.
- **Order and gates.** Smoke test on 3 targets at 32 mm with timing; a visual QC sheet (kept and dropped centres
  on the template; for one target at two positions the cube and its fills at 8, 16 and 32 mm) looked at, and Kate's
  verdict, before the full runs; then 8 mm, then 32 mm, at most 2 workers, started as the CPU allows. Private rows in
  `donor-ladder-brain-metrics-size-20261108-v1\`; public aggregates only in `results/donor_ladder_brain_metrics_size.json`.
  Pre-QC (affine only), like every brain result.

Result (9 Nov 2026, full runs at 8 and 32 mm; QC sheet and the 3-target 32 mm smoke looked at by the agent before the
full runs, under Kate's overnight rule of 8 Nov that the agent checks each experiment by eye on a small portion
first). L = 127.0 mm; 8 mm (f 1/16) keeps 32 of 32 cubes, 32 mm (f 1/4) keeps 20 of 32 (dropped 0, 6, 7, 9, 11, 12,
15, 17, 20, 21, 23, 28). Check: mse_allvox of every shared donor reproduces `rows_p8_s2` and `rows_p32_s8` on the kept
positions (max abs difference 0 to 5e-7); look-alikes differ by design. Tests (Holm over 55 per size): T1 13/13,
T3a 14/14, T3b 14/14 metrics pass and T2 0/14 at both sizes, as at 16 mm. Rank stability (Spearman of the order of 18
fills): 8 vs 16 mm 0.89 to 0.99, 16 vs 32 mm 0.90 to 0.97 over the 14 metrics (MSE 0.92 and 0.94, SSIM 0.91 and 0.93,
lDDT-like 0.97 and 0.96); 8 vs 32 mm 0.72 to 0.91. Expectations: adjacent Spearman >= 0.8 holds for every metric. The
repeat is best at every size except against the control "repeat blended with mean", which beats it on 12 to 13 of 14
metrics at all three sizes (regression to the mean, as at 16 mm). Noise is not the worst rung on MSE-type metrics at
any size (it stays below the stranger, R 0.83, 0.92, 0.96 for MSE), as at 16 mm; it is worst on lDDT-like at every
size. Not as expected: the dataset mean does not move with size (MSE R 0.41, 0.41, 0.42). Scale dependence: the
fixed rungs (own other side, sibling, means) stay within 0.1 R across sizes, but the search rungs move closer to the
repeat as the cube shrinks with the search kept at +-8 mm (MSE R at 8, 16, 32 mm: own other side best fit 0.29,
0.58, 0.62; best look-alike that knows the answer 0.11, 0.35, 0.53; look-alike by context 0.26, 0.52, 0.66). A
search-based donor is therefore stated with its window size. `results/donor_ladder_brain_metrics_size.json`, figure
`results/figures/fig_brain_window_sizes`. Pre-QC, affine only.

## Amendment 22 (9 Nov 2026, LPIPS VGG and set-level FID/FWD on the frozen boxes; recorded before any result)

Kate, 9 Nov 2026 (quoted by the ISBI paper session): "Gpu для метрик разрешаю 20$" and "только сохрани старые
результаты а это сделай новыми". The work fits the laptop CPU (measured: LPIPS VGG 54 ms per slice pair at 64 px,
1 thread), so no GPU or cloud is used and no data leaves the laptop. Amendment 20 outputs stay untouched; this is a
new version.

- **Run.** One new pass over the two frozen boxes (deep, parietal; 221 targets, both sides) with the same donors,
  fills, scored voxels E and seed as Amendment 20 (`run_position` with the extra hook `vgg_set`). Tier-1 columns are
  recomputed and must equal the Amendment 20 rows (`donor-ladder-brain-metrics-20261107-v4`) on every shared row,
  which proves the fills are the same; otherwise no summary is written. Private rows and crops in
  `donor-ladder-brain-metrics-20261109-v5\`; public aggregates in `results/donor_ladder_brain_metrics_t23_v5.json`.
- **LPIPS VGG.** `lpips` 0.1, net vgg (torchvision VGG16 weights from the project TORCH_HOME), on exactly the
  Amendment 20 slices: the three central orthogonal slices lying fully inside E, z / 4 clipped to [-1, 1], bilinear
  to a 64 px short side, mean over the slices; NaN when no slice lies inside E, never 0. Lower is better. Summarised
  like Amendment 20 (raw, c, R with family-bootstrap intervals, T1-T3 on the L+R view of each box, one Holm family
  over both boxes for this metric only; the Amendment 20 Holm family is not changed).
- **Set-level FID and FWD per rung.** Cell = (target, side, slice axis) in one box. Sample = the central 24 x 24
  square of that central slice (24 mm is the shortest box edge), used only when it lies fully inside E of that
  target-fill pair, z / 4 clipped and mapped to [0, 1], bilinear to 64 x 64. For each rung the target set and the
  fill set are the samples of the same cells (cells valid for that rung). FID: Inception-v3 pool features
  (pytorch-fid weights, 2048-d), Frechet distance computed exactly through the n x n cross-Gram matrix because n is
  below 2048. FWD: Frechet distance per orthonormal Haar packet (level 2, 16 packets), unweighted mean, as in the
  face panel. Reported per box view (`deep/LR`, `parietal/LR`) and rung: fid, fwd, n_cells, and a 95% interval from
  100 bootstrap resamples of families (all cells of a family move together; duplicates allowed). Floor: the target
  set split into two family-disjoint halves (`_target_split_half`). Reference strangers (str00-str15) are not rungs.
  Topology Distance stays not run (it needs equal-sized groups; own amendment if wanted).
- **Expectations (stated now).** LPIPS VGG orders the rungs like LPIPS AlexNet (Spearman of the rung order >= 0.8 in
  the deep box) and passes T1. Set level: every single-donor rung (repeat, own other side, sibling, look-alikes,
  stranger) is close to the split-half floor, because each of these sets is a sample of real brains; set-level
  distances therefore cannot rank individual identity. The mean fills (dataset mean, mean of 2, cohort mean) and the
  blended controls have reduced variance and score clearly worse than a stranger on FID and FWD, the opposite of
  MSE; noise scores worst on FWD.
- **Order and gates.** Smoke on 3 targets with timing; one QC look at the slices and crops of one target (target,
  repeat, stranger, dataset mean, noise) and at the smoke numbers, then the full run, 2 workers. Pre-QC (affine only).


Amendment 22, result (9 Nov 2026). The v5 pass reproduces the Amendment 20 rows exactly (25,495 shared rows; mse,
ssim, lddt, mse_allvox max difference 0, same NaN pattern). LPIPS VGG: the rung order agrees with LPIPS AlexNet
(Spearman of R over 18 rungs 0.965 in both boxes; expectation >= 0.8, met), and T1 passes (deep/LR: repeat closer
than the dataset mean for 23 of 23 targets, Holm p 4.8e-7; met); T2 does not pass (repeat vs repeat shrunk to the
mean, 3 of 23), as for the other tier-1 metrics. Set level (FID / FWD; floor = family-disjoint split half of the
target set, deep 11.2 / 0.026, parietal 28.0 / 0.092): the single-donor rungs are at or below the floor (deep:
own other side 0.0, sibling 0.4, stranger 3.1, look-alikes 4.0 to 4.9; parietal: 0.0, 0.6, 9.3, 47 to 48), except
the repeat (deep 25.7, parietal 58.4), whose set has only 138 and 87 cells against about 1,300 and 880, so its FID
carries the small-sample bias; the expectation that set-level distances cannot rank individual identity is met. The
own-other-side and sibling sets reproduce the target set almost exactly by construction (the other side of a target
is itself a target sample; siblings are themselves targets), so their near-zero values say nothing about identity.
The mean fills and blended controls score far worse than a stranger, the opposite of MSE (deep FID: dataset mean
316, cohort mean 243, mean of 2 23.0, other side shrunk 153, against stranger 3.1; FWD likewise); met. Not met:
noise is not the worst on FWD. The dataset mean is worse than noise in both boxes (deep 0.391 vs 0.374; parietal
1.542 vs 1.193, where the cohort mean, 1.256, is also worse); reported, not explained away.
## Amendment 23 (9 Nov 2026, IXI minimal run: 100 targets, other side as the R anchor; amends Amendment 19; recorded before any IXI scan is prepared)

Kate, 9 Nov 2026 (quoted by the ISBI paper session): "мб еще запустим один датасет по мозгам ixi чтобы было для
сравнения?". Purpose: a second brain dataset for the dataset-comparison result, as faces have CelebA and IMDb-WIKI and
iris has three sets. Amendment 19 stays the full design; this is its smallest version that answers the comparison.
It runs after Amendment 22 finishes, never in parallel with it, and only when free RAM and disk cover its peak
(STUDIES-044, checked and logged before each stage).

- **Data and integrity.** Registry id `ixi`, local copy `datasets/neuroimaging/ixi/version-unverified/raw` (payload
  integrity not verified by the registry). Every file used is checked before use: full gzip decompression, nibabel
  load, 3D shape, finite voxels, sha256 recorded in the private manifest. A failing file is excluded and counted,
  never repaired. Eligibility as Amendment 19 (one T1 per ID with sex and age from `IXI.xls`; IDs whose duplicate
  rows disagree are excluded and counted).
- **Sample.** 100 targets drawn from the eligible scans, stratified by centre (Guys, HH, IOP; proportional, largest
  remainder) and age tertile within centre, random within strata, seed 20261109. No selection by image content
  (STUDIES-025). The same 100 scans are the donor pool (leave the target out).
- **Preparation.** Identical to QTIM and Amendment 19: SynthStrip mask (pinned FreeSurfer 8.0.0 image), N4 on a 2 mm
  copy, ANTs affine to the nilearn MNI 2009a symmetric template (same settings, seed 123), 1 mm, own mask by nearest
  neighbour, z-scored in the own mask. If SynthStrip runs from another image, the Amendment 19 Dice check on 5 QTIM
  scans (>= 0.99 each) comes first. D7 automatic QC rules as in QTIM.
- **Positions and metrics.** The two frozen boxes and the same random 16 mm cubes (Gaussian window of Amendment 18)
  as QTIM, both sides. Tier-1 metrics of Amendments 10 to 12 (`src/donor_ladder_brain_metrics.py`, unchanged). Tier 2
  and 3, LPIPS VGG and set level are not run on IXI here.
- **Rungs.** Own other side (reference), look-alike by context (2 and 4 mm shell search) and best look-alike
  (oracle), same-sex age-matched mean (the 10 same-sex donors nearest in age; the median age gap is reported),
  dataset mean (leave-one-out), stranger from the same centre and stranger from a different centre (16 each, exact
  expectation; a rung with fewer than 16 eligible donors uses all and reports n), mean of 2 strangers, reference
  shrunk to the mean (QTIM lambdas), wrong detail, white noise. Not available in IXI and reported as n/a, never
  filled or drawn as 0: repeat, sibling, co-twin, cohort mean.
- **R and tests.** IXI has no repeat, so R uses the other side as the reference, as PAPILA does:
  R = (d(rung) - d(other side)) / (d(strangers) - d(other side)), d(strangers) = mean over 16 reference strangers drawn
  from all centres (as QTIM str00-str15).
  Tests as the cross-modality rule (`tests_uniform.py`) with reference = other side: T1 n/a (it needs a repeat), T2 =
  other side shrunk to the mean, T3a = wrong detail, T3b = noise, Wilcoxon over targets (no families), Holm within
  IXI. For the comparison, QTIM R is also recomputed on the same other-side anchor from its existing rows (no rerun).
- **Expectations (stated now).** E1 the dataset mean is closer than a stranger on MSE-type metrics (R < 1), as in
  QTIM. E2 best look-alike < look-alike by context < stranger. E3 the same-centre stranger is closer than the
  different-centre stranger on intensity metrics (scanner effect, as in the image-ladder run). E4 the order of the
  shared rungs on the other-side anchor agrees with QTIM (Spearman >= 0.8 in the deep box). A failure is reported,
  not explained away.
- **Order and gates.** Integrity check of all 100 files; preparation of 3 targets plus their QC sheet (three slices
  through both boxes) looked at before the rest; smoke ladder on 3 targets with timing; then the full run on 2
  workers. Private outputs in `projects/identity-ladder-metric/ixi-ladder-20261109-v1/`; the repository holds
  aggregates only, in `results/donor_ladder_brain_ixi.json` (keys as `results/donor_ladder_brain_metrics.json`, with
  `R_anchor = "other_side"`). If the files fail integrity, SynthStrip cannot run without a download Kate has not
  approved, or the runtime does not fit, the run is written as a limitation with a time estimate. Pre-QC, affine
  only, like QTIM.


Amendment 23, preparation outcome (9 Nov 2026, before any IXI ladder number). Kate, 9 Nov (quoted by the ISBI paper
session): "на машине". Eligibility: 581 T1 files, 15 without a spreadsheet row, 3 without sex or age, 2 with
disagreeing rows; 561 eligible. Sample of 100 (seed 20261109): Guys 56, HH 32, IOP 12; age 23 to 78 (median 47);
60 female. Integrity check: 100 of 100 pass. Preparation (`src/ixi_prep.py`, `gcp/ixi_prep_vm.sh`) ran on a GCP
n2-standard-8 spot VM in europe-west1-b (the 32-vCPU project quota did not allow more next to the face job),
13:16 to 14:44 UTC, with the pinned FreeSurfer 8.0.0 image (digest checked on the VM; no Dice check needed) and the
laptop's antspyx 0.6.3, nibabel 5.4.2, numpy 1.26.4; the ANTs calls are those of `real_cases.prepare_scan` (checked
by a test). SynthStrip 12.7 s and preparation 18.5 s per scan (median), 0 failures. Only the prepared volumes,
masks and per-scan JSON came back; the VM and its disk were deleted at 14:44 UTC. Cost about USD 0.15 at the spot
catalog price (estimate; billing export lags a day). 3-scan smoke sheet looked at before the rest: masks on the
brain edge, consistent registration, no N4 gradient; pass.
D7: a first version of the IXI rule also excluded scans whose parietal box or mirror lay under 95% inside the own
mask (8 scans, parietal 0.936 to 0.951, at the vertex near the edge of the IXI field of view). That is stricter than
QTIM (`real_cases.qc_flags` excludes on the deep box only); it was corrected to the QTIM rule before any ladder
number, with a test. The parietal box and every cube use the ladder's per-cube 95% validity rule (missing rows,
never zero), as in QTIM. Result: 0 excluded; 5 flagged by template correlation (median 0.542, MAD 0.038; QTIM median
0.513), looked at: 1 with enlarged ventricles, 1 with the lower IOP contrast, 3 unremarkable; registration and masks
fine in all 5, kept. Random 30 (seed 123) looked at: 30 of 30 pass. Verdict by the brain metrics session, delegated
by the ISBI paper session ("look at the QC sheet yourself"). The IXI ladder runs after Amendment 22 finishes.

Amendment 23, ladder result (9 Nov 2026). Script `src/donor_ladder_brain_ixi.py`, which calls the QTIM tier-1 code
unchanged (region arrays, `run_position_at`, `score`, `ladder_R`, `tests`). The run took about 37 min on 2 workers
for 100 targets and 34 positions. Output: `results/donor_ladder_brain_ixi.json`, `R_anchor = "other_side"`.
Not available, reported as n/a and never filled: the repeat, sibling, co-twin and cohort-mean rungs; T1;
acc_cohort; and the four protein-style metrics, which need the repeat-scan tolerance. The protein-style metrics
were removed from the IXI metric list at the summary step, after the rows existed, because their rows are NaN by
construction. In the parietal box, 91 of 100 targets have a valid box; the rest are cut at the vertex and are
missing, never zero. Median gap of the age-matched mean: 2.8 years.
Expectations, with MSE R and 95% family-bootstrap intervals on the L+R views:
- **E1, met.** The dataset mean is closer than a stranger: deep 0.31 [0.24, 0.37], parietal -2.53, random cubes
  -0.46. The age-matched mean gives the same picture: 0.32, -2.24, -0.37.
- **E2, met.** Best look-alike < look-alike by context < stranger: deep -0.05 < 0.07 < 1.16; parietal -1.77 < -1.20
  < 1.19; random cubes -0.78 < -0.43 < 0.97.
- **E3, met.** The same-centre stranger is closer than the different-centre stranger in every view: deep 0.62
  [0.56, 0.68] against 1.19 [1.13, 1.25]; parietal 0.53 against 1.14; random cubes 0.63 against 1.17. The scanner
  effect is about half the stranger scale.
- **E4, met.** The QTIM rows were recomputed on the same other-side anchor, with no rerun. Spearman of R over the 12
  shared rungs in the deep box: mse 0.972, ssim 0.958, pearson 0.965.
- **Tests (MSE, Holm within IXI).** T3a and T3b pass in every view. T2 passes in the deep box only (60% of targets);
  it fails in the parietal box and in the random cubes, as T2 does in QTIM.
- **Parietal box.** The other side is almost as far as a stranger here, so the parietal R values have large
  magnitudes and wide intervals; read the order, not the size.
Pre-QC, affine only.
## Amendment 24 (9 Nov 2026, mean of 16 look-alikes and Betti matching 3D; recorded before any result)

Kate, 9 Nov 2026 evening, on the list of remaining items (quoted by the ISBI paper session): "запускай".

**A. Mean of 16 look-alikes, QTIM and IXI.** This reverses the Amendment 11 line "k = 4, 8, 16 and the 16-look-alike
mean are dropped" for the k = 16 mean only, on Kate's decision, so that the rung exists for the brain as it does
for fundus and iris (cross-modality key `lookalike16`).
- Fill: the per-voxel masked mean of the 16 donors with the smallest scaffold distance, computed with the same
  distances `masked_dist` already uses to pick the k = 1 look-alike (Amendment 11 scaffold rule: shell voxels inside
  both masks, at least 50% of the scaffold shared, donor cube at least 95% inside its own mask, other families
  only). There is no new search. If fewer than 16 donors are eligible, the mean uses all of them and n is reported.
  The pools are 218 to 220 people (QTIM) and 99 (IXI). E = target mask AND the voxels where at least one of the 16
  donors is valid (the rule of every mean fill).
- Positions, sides, targets, metrics: as the tier-1 rows (QTIM Amendments 10 to 12; IXI Amendment 23 with its
  metric list), all 34 positions, both sides.
- R anchors: QTIM repeat = 0, stranger = 1 (Amendment 11). IXI other side = 0, stranger = 1 (Amendment 23).
  Intervals and views as before.
- Reproduction gate: the nearest of the 16 must reproduce the committed k = 1 look-alike rows on every metric
  (same donor, same E, same means). Otherwise no summary is written.
- Old results stay untouched. Private rows: `donor-ladder-brain-lookalike16-20261109-v1/` (QTIM) and
  `ixi-ladder-20261109-v1/lookalike16_parts/` (IXI). Public: `results/donor_ladder_brain_lookalike16.json`, with
  `lookalike16` and, for reference, `lookalike` and `dataset_mean` per dataset, view and metric.
- Expectation (stated now): on MSE-type metrics the 16-look-alike mean is closer than the single look-alike and
  than the dataset mean in the random cubes (averaging selected donors removes their individual detail while
  keeping the scaffold-matched part). On SSIM, pearson and lddt it is reported without an expectation.
- Order: smoke on 3 targets with timing, then QTIM, then IXI, one job at a time on the laptop CPU.

**B. Betti matching 3D, QTIM (IXI if the timing allows it within the budget).**
- Library: Betti-Matching-3D (github.com/nstucki/Betti-Matching-3D), built from source on a Linux VM, commit
  recorded at clone. Same settings as the 2D library in fundus Amendment 9.15 and iris P3 (Betti-matching at commit
  3e7eca26): sublevel filtration, absolute (not relative). The matched and unmatched parts are computed with the
  same formula as `iris_ladder_metrics_extra.betti_parts`, over dimensions 0, 1 and 2 (dimension 2 exists only in
  3D; also reported as dims 0-1 only, so it is comparable with 2D). Lower = closer for both parts.
- Input: the 16 mm random cubes of the tier-1 rows, z clipped to [-3, 3] and mapped linearly to [0, 1]. Only cubes
  where E covers all 4096 voxels are used (no filler enters the filtration). Topology subset S as in fundus: per
  target and side, 2 wholly valid random positions, drawn with seed [20261102, target, side]. All rungs of the
  tier-1 rows for those cells.
- Agreement check before any full run: on 2 targets, the 3D library on a single 16 x 16 slice (a 16 x 16 x 1
  volume) must give the same matched and unmatched parts in dimensions 0 and 1 as the 2D library on the same slice,
  within 1e-6 relative. Otherwise stop and report.
- Compute: a Linux GCP spot VM in the cheapest allowed EU region (gcp/prices.py, STUDIES-038), resumable. Only the
  cube pairs of S are uploaded, as float16 npz keyed by row index; no scans, no IDs. The VM and its disk are
  deleted at the end, also on failure, and the cost is reported. Cap: USD 2 for this item. If the timed estimate is
  higher, stop and report before the full run.
- Summary: R and tests as tier 1 (QTIM repeat anchor; IXI other-side anchor), on the subset S. Private rows in
  `donor-ladder-brain-betti3d-20261109-v1/`; public `results/donor_ladder_brain_betti3d.json`.
- Expectations (stated now): Betti matching ranks noise far (it adds features; passes T3b) and the dataset mean far
  (smoothing removes features). It may, like persistence in Amendment 20, put the own other side ahead of the
  repeat because of the session intensity offset; that is reported, not repaired.

# Protocol: the donor ladder for any imaging modality (5 Nov 2026)

General protocol distilled from the brain ladder (`PROTOCOL_donor_ladder_brain.md`, Amendments 1-8) and the
fundus ladder (`PROTOCOL_donor_ladder_fundus.md`), with Kate's critical review of 5 Nov 2026. A modality
protocol instantiates every section below and is recorded before its results. Results stay **pre-QC** until the
registration gate in `AGENTS.md` is met for that modality.

## 1. Question and unit

Fill a region of a target image with tissue from donors of decreasing relatedness and score it against the
truth. If a metric ranks an average of other people above related anatomy, it rewards averaging.

- Unit: V, the error of the mean of all donors outside the target's family (or person, if no relatives). For
  squared error a single stranger scores 2, the mean 1, and r = 1 - error / 2 is the share of individual
  variation shared with the donor.
- For any other metric (SSIM, LPIPS, segmentation overlap, ...) the stranger level is not 2 by theory: report
  each metric against its own empirical stranger and mean levels on the same scale, and check for saturation.
- Metric panel: fixed before results. Squared error is always first, because it has the closed-form check.

## 2. Rungs (define each for the modality, or mark it unavailable)

| Rung | Brain MRI | Fundus | Iris | Face photo |
|---|---|---|---|---|
| repeat | session 2 | other visit, same eye | another image, same eye | another photo |
| mirror | other hemisphere, same scan | fellow eye, same visit | other eye | other face half |
| relative | sibling / co-twin | if available | if available | if available |
| metadata mean | sex, age | sex, age, device | sex, age, device | sex, age |
| look-alikes | k nearest by context | same | same | same |
| dataset mean | all other families | same | same | same |

For every rung, state what the donor shares with the target besides the person (scan, camera, visit, interval,
device). The mirror inside one scan shares acquisition noise and registration; a fellow eye shares the camera
and the visit. Report the repeat interval and device change.

## 3. Common frame, mask and normalization

- A common frame is mandatory: MRI affine to a template (no nonlinear warp, which erases folding); fundus
  disc-fovea similarity frame, disc-right; iris polar unwrapping with rotation alignment and radius
  normalization (pupil size); faces landmark alignment with the same pose for repeats.
- Misregistration hurts same-person rungs and leaves the mean almost untouched, so before QC the "mean beats
  relatives" finding can be overstated. Every report says so.
- Validity mask per image (brain mask, field of view, iris annulus without lids and glints, face mask).
- Normalization: z-score inside the mask per channel. Any harmonization is fitted on a fixed reference pool of
  people outside every compared pair; otherwise it lowers V.

## 4. Regions

- Primary geometry fixed in advance (brain: 16 mm cube, 4 mm shell; fundus: 48 px patch, 16 px ring on the
  256 grid). Random positions drawn once with the seed where the window is inside the template/field of view in
  >= 90% of images, plus any frozen anatomical boxes. Same positions for every target and donor.
- Validity: >= 95% (brain) or 100% (fundus) of region pixels inside each image's mask.
- Sensitivity grid (exploratory): region size x context width, same centres.
- Lesion-shaped masks are the realistic case for inpainting; run them when a lesion-mask source exists.

## 5. Fills and mandatory controls

- Fixed-rule donors: repeat, mirror at the same place, relatives (all of them, averaged per target), stranger
  same side and other side, metadata mean, dataset mean, averages of k random strangers.
- Selection-based fills: look-alikes by context (k nearest on the shell/ring), by the target's own mirror
  region, and the mirror block search (best context match within a window, and over the whole contralateral
  side). **Every selection has two controls**: the same selection applied to a stranger, and an oracle version
  that selects with the true region (an upper bound, never a method).
- Both sides: each experiment on the left and right (hemisphere, eye laterality) and their average.
- Reproduction: a new implementation first reproduces the previous numbers exactly.

## 6. Statistics

- Ratio = sum of errors / sum of V over targets; 95% intervals by bootstrap over families (or persons),
  2000 resamples; denominators with every value; rungs with fewer than 10 targets are descriptive only.
- Averages of k: compare with (1 - R2)(1 + 1/k).

## 7. Maps

- Local error in a sliding window (box filter of the squared difference) for fixed-rule donors; look-alikes on
  a coarser grid. Same rules and unit as the regions.
- Reproducibility: Pearson r between maps from two independent halves of families (pools restricted to each
  half). Per-map contrast is allowed; voxels where the halves disagree on the side of the median are faded.
- Show coverage (targets per voxel) next to the maps.

## 8. Process

- Every design change is an amendment written before its results; tests with known geometry for each new
  function; aggregates only leave the private folder; metrics board before and after; commit and push in steps.

## 9. Fundus instance (run 5 Nov 2026, `src/donor_ladder_2d.py`)

Data, frame, grid, positions and cohorts as in `PROTOCOL_donor_ladder_fundus.md` (PAPILA, GRAPE; disc-right
frame; green channel z-scored in the field of view; 32 patches of 48 px with a 16 px ring, support 90%; metric
box). Added, recorded before these results:

- Sides: target eye laterality R and L, and their average (LR).
- Mirror = fellow eye reflected into the frame, same place (`fellow_eye`); `fellow_block`: best ring match
  within +-8 px in the fellow eye; `fellow_block_oracle`: best match on the true patch. Controls:
  `stranger_other_side` (a stranger's other eye at the same place), `stranger_block`, `stranger_block_oracle`
  (the same searches in that stranger's other eye). The stranger is the first of the population draw with a
  valid other eye.
- Look-alikes, k = 16: `knn` (ring, as before), `knn_oracle` (true patch), `knn_fellow` (distance from the
  target's fellow-eye patch to donors' patches).
- Sensitivity: patch 24, 48, 72 px x ring 4, 8, 16, 24 px on the same centres.
- Maps on the canonical frame: 48 px window, fixed donors (fellow eye, other visit, stranger expectation,
  metadata mean), unit V, independent person halves.
- Outputs: private rows in `donor-ladder-2d-20261105-v1\`; public `results/donor_ladder_2d.json` and figures
  `results/figures/fig_ladder2d_*`.

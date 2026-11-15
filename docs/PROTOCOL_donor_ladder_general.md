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
- Missing is never zero (added 6 Nov 2026 after Amendment 9.4; applies to every dataset and modality). Pixels or
  voxels outside the field of view, brain mask or any acquisition mask are missing values. Store a validity mask
  with every image; every error, distance, mean, variance and normalisation uses only pixels valid in all images it
  touches (target, donor, every pool member), or leaves that location out. Never let a filler value (0, the
  z-score mean, padding) enter a statistic. Each new dataset gets a regression test in which a filler-valued region
  outside the mask would win or shift the result if it were counted.
- Keep the acquired field (added 6 Nov 2026). Do not crop away part of the field of view for convenience (a centre
  square, a fixed ROI); embed the whole field in a larger canvas with the outside marked missing. Any crop that
  drops acquired tissue is a design change: an amendment that states how much is lost and why.

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

### Amendment 9.1 (5 Nov 2026, illumination correction for fundus; recorded before these results)

Review of the fundus maps showed that z-scoring the green channel leaves large-scale illumination (vignetting,
brightness around the disc) in the images, so V and the fellow-eye rung may partly measure lighting shared by the
camera and visit. Following the background-estimation approach of Foracchia, Grisan and Ruggeri (Med Image
Anal 2005, doi:10.1016/j.media.2004.07.001), the fundus analogue of N4:

- `flat` arm: in the canonical frame at 512 px, the green channel is divided by a background estimate - a median
  filter over 52 px (computed at 128 px with a 13 px window, outside the field of view filled from the nearest
  inside pixel, upsampled bilinearly) - then downsampled to 256 and z-scored in the field of view.
- First run: PAPILA, primary geometry only (48 px patch, 16 px ring), same positions, seeds and fills; compared
  with the z-score arm. GRAPE and the sensitivity grid follow if Kate approves.
- Deep enhancement or harmonization networks (GFE-Net, PCE-Net and similar) are out of scope here (Kate,
  5 Nov 2026: next paper).

### Amendment 9.2 (5 Nov 2026, versioned fundus preprocessing, iterated under visual QC)

Visual QC of five evenly spaced PAPILA images and a 40-image check showed that the fovea detector of the
disc-fovea frame (`ladder2d_register.fundus_landmarks`, Amendment 2b of the fundus protocol) returns the outer edge
of its search band (0.75 of the field-of-view radius, 213 px on 512) instead of the fovea: vignetting makes the
band edge the darkest point. The frame is therefore anchored on the disc only, with rotation and scale set by
that edge point. It also showed that the 52 px background median of 9.1 includes the optic disc. Kate (5 Nov 2026):
version every preprocessing, record it here before it runs, and iterate until it passes visual QC.

Versions (each is frozen once recorded; results always name their version):

- **P0** - fundus protocol Amendment 2b frame (fovea detector above) + green channel z-scored in the field of
  view. All fundus ladder results up to commit 8669120 use P0. Status: **fails visual QC** (fovea).
- **P0.1** - P0 frame + the 9.1 background division (52 px median). Status: **fails visual QC** (fovea; disc in
  the background). Its runs were stopped before any result.
- **P1** - landmarks and background redone (`src/fundus_preprocess.py`):
  1. Disc: as P0 (brightest point of the smoothed red+green inside the eroded field of view); disc radius
     estimated from the bright blob (half-maximum area of the smoothed brightness around the disc).
  2. Background (for the fovea search and for the analysis): median filter of about 100 px (computed at 128 px
     with a 25 px window) after removing a circle of 1.5 disc radii around the disc and filling it from the
     nearest outside pixel; the green channel is divided by it.
  3. Fovea: darkest point of the smoothed background-divided green inside the same band as P0; accepted only if
     it is a strict local minimum away from the band edge (more than 5 px inside) and its distance from the disc
     lies in 1.8 to 3.5 disc diameters. Otherwise the image is counted as a frame failure and left out (never
     silently accepted).
  4. Frame and analysis: the Amendment 2b similarity frame from these landmarks; green / background,
     downsampled to 256, z-scored in the field of view.
- Visual QC per version: 20 images evenly spaced in each dataset table (PAPILA, GRAPE), showing raw + both
  landmark sets, the frame with target positions, the background and the analysis image, plus the frame-failure
  count. The version passes only when the reviewer (Claude first, then Kate) finds the disc and fovea correct on
  every shown image that the version accepts; otherwise the next version is recorded here and the loop repeats.
- **P1 QC result (6 Nov 2026, 20 + 20 sheets).** PAPILA: 5 of 20 accepted, at least one with a wrong disc (image
  307) and one with a wrong fovea (435); PAPILA photographs are disc-centred and the fovea is mostly at or beyond
  the field edge, so a disc-fovea frame does not fit this dataset. GRAPE: 8 of 20 accepted, all 8 correct (fovea
  on its target), but 12 rejected wrongly: the ~100 px background is about the size of the fovea and removes its
  darkness, so the minimum moved to the band edge. Status: **fails visual QC**.
- **P2** (recorded before it runs):
  1. Laterality from the clinical metadata (OD / OS), not from the landmark order; OS images are reflected so the
     disc is on the right.
  2. Disc and disc radius as P1.
  3. Fovea (GRAPE): search the temporal side only (left of the disc after reflection), in the annulus of 1.8 to
     3.5 disc diameters, within 1 disc diameter vertically, on the green channel divided by a broad background
     (Gaussian of 60 px sigma over the field of view with the disc circle filled), smoothed by 8 px; accept a
     minimum more than 3 px inside the annulus, else a counted frame failure.
  4. Frames per dataset, never mixed inside one: GRAPE uses the disc-fovea similarity frame; PAPILA uses a
     disc-only frame (translation of the disc to the disc target and scaling of the disc radius to 30 px, no
     rotation), because its fovea is mostly not in the field.
  5. Analysis image as P1 (green / background with the disc removed, z-scored), background ~100 px.
- **P2 QC result (6 Nov 2026).** GRAPE: 15 of 20 accepted, all with the fovea on the dark spot and aligned in the
  frame; rejections mostly from an underestimated disc radius (the 1.8-3.5 disc-diameter annulus then lies inside
  the true fovea distance: images 66, 198, 364, probably 99) and one correct rejection (image 0, wrong disc in a
  myopic fundus); image 464 shows a dark artefact next to the disc (disc partly left in the background). PAPILA:
  disc wrong on dark or glare images (281, 307, 333, probably 102), and the disc radius is too unstable to set
  the scale (frames zoomed in and out). Status: **fails visual QC**.
- **P3** (recorded before it runs):
  1. Disc: brightest point of the 12 px-smoothed red + green divided by its broad background (Gaussian of 60 px
     sigma, outside the field of view filled) inside the field of view eroded by 25 px; accepted only if that
     ratio is at least 1.3, otherwise a counted frame failure.
  2. Disc radius: half-maximum radius on the same ratio image, bounded to 0.08-0.20 of the field-of-view radius;
     used only to remove the disc (2 radii) from backgrounds, never to set the scale.
  3. Fovea (GRAPE): temporal side, distance 0.40 to 0.95 of the field-of-view radius from the disc, within 0.30
     of that radius vertically, on the green channel divided by its broad background with the disc filled,
     smoothed by 8 px; accepted if more than 3 px inside the search region and darker than 0.9 of the region's
     median, otherwise a counted frame failure.
  4. Frames: GRAPE the disc-fovea similarity frame; PAPILA translation only (disc to the disc target, no rotation
     or scale: one camera, one field of view).
  5. Analysis image as P2 with the disc removed over 2 radii.
- **P3 QC result (6 Nov 2026).** GRAPE: 19 of 20 accepted; 18 correct, image 198 wrong (fovea placed at the dark
  field edge); image 464 correctly rejected (wrong disc). PAPILA: 17 of 20 accepted; image 281 wrong (disc on a
  small glare spot); 128, 307, 333 rejected for low disc contrast (faint discs, counted); on many images the disc
  point sits on the brightest part (the cup) about half a radius from the disc centre, a systematic offset of the
  translation-only frame. Status: **fails visual QC**.
- **P4** (recorded before it runs):
  1. Disc candidates in order of the P3 ratio score; for each, the connected region with ratio at least halfway
     between the peak and 1 (within 0.25 of the field-of-view radius) gives the disc centre (its centroid) and
     radius (from its area). A candidate whose radius is outside 0.05-0.22 of the field-of-view radius is
     discarded (glare spots, large bright areas) and its surroundings suppressed; up to 3 candidates, then a
     counted frame failure. The score threshold 1.3 stays.
  2. Fovea (GRAPE) as P3, with the search region also kept at least 0.25 of the field-of-view radius inside the
     field-of-view border.
  3. Frames and analysis image as P3.
- **P4 QC result (6 Nov 2026).** GRAPE: 19 of 20 accepted, all plausible; image 198 now has the fovea at the
  expected temporal distance but without a clear dark spot (uncertain). PAPILA unchanged: image 281 still on a
  glare spot in a dark corner (the background division amplifies it), and the disc point still sits on the bright
  cup, because the region above half-maximum is the cup and pale temporal rim, not the whole disc. Status:
  **fails visual QC** (PAPILA).
- **P5** (recorded before it runs): disc centre = maximum of the P3 ratio image smoothed at disc scale (Gaussian of
  25 px sigma instead of 12), among candidates whose raw red + green (12 px smoothing) is at least the 75th
  percentile of the field of view (rejects glare in dark corners); score threshold 1.3, up to 3 candidates. Disc
  radius from the P4 bright region around that centre, bounded to 0.08-0.20 of the field-of-view radius, used only
  to remove the disc. Fovea, frames and analysis image as P4.
- **P5 QC result (6 Nov 2026).** PAPILA: image 281 fixed (disc found), but image 153 is now rejected although its
  disc is clearly visible, and on several images the point still sits on the bright cup. Hand-tuned detection
  shows diminishing returns. Status: **fails visual QC**.
- **P6** (recorded before it runs):
  - PAPILA: the disc comes from the dataset's expert segmentations (`ExpertsSegmentations/Contours`,
    `*_disc_exp1.txt` and `*_disc_exp2.txt`, original-pixel x y): centre = mean of the two polygon centroids,
    radius = mean of sqrt(area / pi), mapped to the 512 grid by the centre square crop and resize of
    `ladder2d.load_base`; translation-only frame as P3. An image without both contours is a counted frame failure.
  - GRAPE: P4 unchanged (all 19 accepted images plausible on the P4 sheets; image 198 uncertain).
- **P6 QC result (6 Nov 2026, Claude's review).** PAPILA: 20 of 20 accepted; the expert disc encloses the disc on
  every image (faint and dark ones included) and the frames put it at one point with consistent vessel exits;
  on the very dark image 102 the dark periphery turns into noise after the background division (mostly outside
  the analysed windows). GRAPE (= P4): 19 of 20 accepted, fovea on its dark spot on 18, image 198 uncertain (no
  clear dark spot, plausible position); image 464 a counted frame failure (wrong disc). Status: **passes Claude's
  visual QC; awaiting Kate's review** before any ladder rerun.
- **Kate's review of P6 (6 Nov 2026).** GRAPE image 464 (very bright) is a wrong rejection: the disc is where it
  should be. Diagnosis: on a bright image the disc's ratio to the broad background (1.32) is no higher than a
  darker corner's (1.33), which the background division lifts; by raw brightness the corner is at the 36th
  percentile of the field of view and the true disc at the 100th.
- **P7** (recorded before it runs): GRAPE as P4 with the P5 raw-brightness gate on disc candidates (raw red +
  green, 12 px smoothing, at least the 75th percentile of the field of view). PAPILA unchanged from P6.
- **P7 QC result (6 Nov 2026, Claude's review).** GRAPE: 20 of 20 accepted; image 464 now correct (disc on the
  disc, fovea on the faint central dark spot; the dark blob next to the disc in its analysis image is the
  haemorrhage, a real finding); the other 19 unchanged from P4 (198 still uncertain). PAPILA = P6 (passed).
  Status: **passes Claude's visual QC; awaiting Kate's review** before any ladder rerun.
- **Kate approved P7 (6 Nov 2026, "да давай").** The fundus ladder is rerun on P7 (`--arm=p7` in
  `src/donor_ladder_2d.py`): primary geometry and maps first for PAPILA and GRAPE, then the sensitivity grid.
  Laterality comes from the metadata, so the P0 side-conflict exclusion no longer applies; frame failures are
  counted per dataset. P0 results stay in `results/donor_ladder_2d.json` for comparison; P7 results go to
  `results/donor_ladder_2d_p7.json`.

### Amendment 9.3 (6 Nov 2026, fixed-composition maps; recorded before the revised maps)

Kate's review of the P7 maps: straight steps and diamond-shaped edges. Cause: each pixel averaged only the targets
whose 48 px window lies inside their field of view, and after alignment many fields share similar borders, so the
set of targets (and of pool donors) jumps along lines. The pattern is a change of composition, not anatomy.

- Maps now use a fixed composition: region R = pixels where the window is valid in at least 95% of the dataset's
  aligned images; targets and pool members = the images whose window is valid at every pixel of R; a donor
  (fellow eye, other visit) counts only if it is in the same set. Every pixel of R then averages the same people.
  The JSON reports |R| and the number of images kept. Applies to the fundus maps from this amendment on; the cube
  ladder (patch rows) is unaffected.
- **Kate's review of the P7 analysis images (6 Nov 2026):** the black outside and the dark border add nothing and
  must leave the analysis; the GRAPE camera notch (the "ear" on the field border) must go as well. The P7
  field-of-view mask is a brightness threshold, so it keeps the notch and the dark transition rim, where the
  background division leaves bright or dark rings.
- **P8** (recorded before it runs): P7 landmarks and frames unchanged; the analysis mask in the frame becomes:
  threshold (red > 0.1) -> largest connected component -> morphological opening with a disc of 0.12 of the field
  radius (removes the notch and other protrusions) -> erosion by 0.03 of the field radius (removes the dark rim).
  This mask is used for the background estimate, the z-scoring and every validity rule (patches, maps).
- **P9** (6 Nov 2026, recorded before it runs; Kate: keep the acquired field, section 8). P0-P8 load a centre
  square of 1934 x 1934 from the 2576 x 1934 PAPILA photo, dropping about 220 raw px of field on each side, and the
  P7 translation onto the 512 canvas clips more. Measured on all 488 images: one camera mask (circle of diameter
  ~2377 raw px, cut at the top and bottom by the sensor); disc offset from the field centre after reflecting left
  eyes, raw px: dx -460 to 619 (1st-99th percentile -228 to 503), dy -515 to 309 (-314 to 213). P9, PAPILA only:
  the whole photo is resized by k = 512 / 3120 (field diameter + the 1st-99th percentile spread of dx, + rounding
  margin), the disc (mean expert contour, raw) is moved by translation only to (278, 247), the middle of the
  positions that keep every 1st-99th percentile offset fully on the canvas (the offsets are asymmetric) (scale 1, no
  per-image disc-size normalisation; Hemelings et al. 2023 keep natural disc-size variation for the same reason),
  outside the photo is missing; the P8 clean mask, background division (window scaled by k / k_P8 so it covers the
  same retina) and z-scoring follow, on the 256 grid. Costs, stated in advance: one grid pixel is 12.2 raw px
  instead of 7.6 (resolution x 0.62), so a 48 px window covers 1.6 x more retina than in P8; images with an offset
  beyond the 1st-99th percentile lose a sliver of field at the canvas edge (counted and reported). A finer grid
  (384) would keep the P8 resolution but touches every fixed position on the 256 grid; deferred, Kate's call.
  GRAPE keeps P8 in this version (its crop loss is measured and reported separately).

### Amendment 9.4 (6 Nov 2026, outside the field of view is missing in the block search)

Found by Kate's review of single PAPILA cases: pixels outside the field of view are stored as 0, which in z units is
a real value (the mean), not a gap. The maps (9.3) and the fixed-rule and kNN donors already required full validity.
The block search (`fellow_block`, `stranger_block` and their oracles) did not: the shell of the target and of the
candidate could reach outside the field of view and enter the distance as zeros. From now on the target's matched
pixels must all lie inside its field of view (else no block donor for that patch), and a candidate needs its patch
and its matched pixels inside its own. Regression test:
`tests/test_donor_ladder_2d.py::test_search_treats_outside_the_field_of_view_as_missing_not_zero` (fails on the old
code). Block-search rows in `results/donor_ladder_2d.json` and `results/donor_ladder_2d_p7.json` predate the fix and
are superseded by the next rerun. The brain block search encodes the outside of the brain mask the same way; not
changed here; recorded in the brain protocol as Amendment 9. The same review caught one display-only
slip that never reached a result file: a private case sheet averaged the dataset mean over images including
their zeros outside the field of view; it now averages only images valid at each pixel.

### Amendment 9.5 (6 Nov 2026, drawn maps use a masked Gaussian window, not a square; recorded before the rerun)

Brain lesson L8 (rule STUDIES-019): drawn local-error maps never use a square window. The 2D local-error maps of
Amendment 9.3 (`maps()` in `src/donor_ladder_2d.py`) averaged the squared difference over a 48 px square. They now use
a masked Gaussian with sd = 48 px / sqrt(12) (the sd of that square, about 13.9 px), counting only pixels inside the
target's field of view and, for donor terms with a single donor image (fellow eye, other visit), inside the donor's
too. The fixed region and fixed image set of Amendment 9.3 stay defined by the 48 px square fraction (a validity
fraction, not a drawn value). Figures from these maps are redrawn after Kate approves a before/after view.

## 10. Amendment 10 (14 Nov 2026): one window size, f = 1/8, for the cross-dataset summary; RQ4 size curve

Recorded before any run of this amendment. Kate, 14 Nov 2026: "да давай, зафиксируй и пересчитай". The repo-level
pause of metric experiments (AGENTS.md, 2 Nov, until registration QC passes) still stands; Kate asked for this run
explicitly. Brain (QTIM, IXI), retina (GRAPE, PAPILA) and iris (UTIRIS NIR/VIS, UPOL) stay labelled provisional.

**Why.** The rule of 8 Nov (window w = f x L in object units, f in {1/16, 1/8, 1/4}; brain Amendment 21, face F3,
iris P4) was run as a sweep, but the cross-dataset summary (`results/cross_modality/common_all.json`) still takes each
dataset at its historical window. Measured f of those windows: brain 16 mm / L 127.0 mm = 0.126; faces 16 px / 79.5 px
= 0.201 (CelebA) and 16 / 79.5 = 0.201 (IMDb-WIKI); iris 16 px / (256 / pi) = 0.196; retina 16 px / L, L = sqrt(mean
field-of-view area in the 128 px frame): PAPILA L = 78.1 px (488 eyes, mask `M` of `donor_ladder_2d_facefmt.down2`,
measured 14 Nov, geometry only) -> f = 0.205; GRAPE to be measured the same way. So only the brain is at f = 1/8; faces,
iris and retina are at about 0.2. The single look-alike moves with f (MSE, R on the stranger scale: brain 0.26 / 0.52 /
0.66 at 1/16, 1/8, 1/4; CelebA 0.16 / 0.30 / 0.40 / 0.45 at 1/16, 1/8, 0.2, 1/4), while the dataset mean, the shrunk
true person and wrong detail do not. RQ4 therefore depends on f and must be reported at one f.

**Rule.**
1. Reference size for every cross-dataset statement: f = 1/8, w = round_half_up(L / 8). Brain 16 mm (QTIM L = 127.0
   mm; IXI is in the same MNI analysis space and uses the same 16 mm cube, its own L is not measured: stated as such);
   faces 10 px; iris 10 px; retina round_half_up(L / 8) per dataset (PAPILA 10 px; GRAPE from its own L). The
   historical f ~ 0.2 results of faces, iris and retina are kept unchanged as the f = 0.2 point of the size curve.
2. Context (ring) per modality as in the existing sweeps, no new rule: brain shell s = w / 4 with the +-8 mm block
   search fixed in mm (Amendment 21); faces, iris and retina ring = ceil(w / 2) (F3, P4, fundus 9.14 with w in place
   of 16). Grid, validity thresholds, pools, seeds, lambdas and E are unchanged; positions at each size are reported,
   never filled.
3. RQ4 size curve: f in {1/16, 1/8, 1/4} (plus the historical ~0.2 point where it exists). Rungs: single look-alike
   (k = 1, context ring or shell), mean of 16 look-alikes (`lookalike16`, the 16 nearest by the same ring distance,
   masked mean; fewer than 16 eligible donors = missing, not a smaller mean), oracle look-alike (nearest on the hidden
   window), dataset mean, true person (second image or other visit: `repeat`), the shrunk true person and the
   stranger anchor. Scale: R_self of `src/cross_modality/common_frame.py` (0 = target itself, 1 = mean of the 16
   seeded strangers; d = MSE over E), with a 95% bootstrap interval over the dataset's unit (family, person or brain
   target; 1000 reps). Faces at 5 / 10 / 20 px use the F3 subset (100 evenly spaced targets per source, the same at
   every size; their 16 px point is the same subset of the full rows). Brain: QTIM random cubes kept at each size
   (Amendment 21: 32 of 32 at 8 mm, 20 of 32 at 32 mm; 16 mm on the same 32), all targets. IXI has no size geometry
   (no 8 or 32 mm run): only its 16 mm point, stated as missing at the other sizes.
4. What is computed new: `lookalike16` at every swept size where it is missing (brain 8 and 32 mm; faces 5, 10, 20 px
   on the subset; iris and retina come with their own panel, which already holds `lookalike16` and the oracle); the
   full-cohort tier-1 face rows at 10 px with `lookalike16` (needed for the f = 1/8 summary input; F3 planned all
   targets but only the subset was run); iris 10 px for UTIRIS VIS and UPOL (NIR exists, P4), and 5 / 20 px for
   UTIRIS NIR (primary set) if CPU time allows; retina at 10 px (f = 1/8) and at 1/16, 1/4 if the inputs exist
   (GRAPE needs its metadata table from the `re_identification` checkout; the only copy on this machine is the
   Syncthing version store `C:\Projects\.stversions\...\externale_identification-50bf2ea`, used read-only).
   Retina geometry (measured 14 Nov before any run, `src/rq4_fundus_size.py --geometry`): PAPILA L = 78.1 px (488
   eyes), GRAPE L = 85.5 px (239 eyes); the 16 px window is f = 0.205 and 0.187. Sizes: PAPILA 5 / 10 / 20 px, GRAPE
   5 / 11 / 21 px (rings 3 / 5 / 10 and 3 / 6 / 11). Retina runs the panel's tier-1 rungs and metrics only (the
   Amendment 9.15 extras are off); its wrapper reproduced the stored 16 px PAPILA rows exactly on a 10-eye smoke.
   CPU budget: where a full-cohort run at 1/16 is too long (retina, iris: the number of windows grows as 1 / f^2),
   1/16 runs on evenly spaced eyes (every 4th of each side, about 25%), and the 1/8 and 1/4 points of that set are
   also reported on the same eyes, so the curve is paired; the full-cohort 1/8 run is the summary input.
5. Gates. (a) Each pass recomputes the k = 1 look-alike next to the new fill; it must equal the stored k = 1 rows of
   the same size (brain |diff| <= 1e-5, faces <= 1e-9, on every metric and the same missing pattern), else nothing is
   written. (b) Smoke run with timing on a few targets per modality, and a private QC sheet (target, k = 1 look-alike,
   mean of 16, oracle, at each size, evenly spaced targets) shown to Kate before the full runs. (c) Disk and RAM are
   checked before each long run; at most 3 workers while other sessions run; progress with ETA in the log.
6. Outputs. Private rows: `C:\Projects\data\projects\identity-ladder-metric\rq4-window-size-20261114-v1\`. Public
   aggregates only: `results/rq4_window_size.json` and `results/figures/fig_rq4_window_size.{png,pdf}`;
   the f = 1/8 summary inputs for faces and iris (and retina when run) in
   `results/cross_modality/common_f8_v1.json`, same format as `common_all.json`, brain entries copied unchanged (they
   are already f = 1/8). `common_all.json`, `generative_block.py`, `fig_metric_summary_R.json` and the summary-figure
   scripts are not edited here; the summary figure is redrawn from the new file by its owner.
7. Claims. An RQ4 statement in the paper is made at f = 1/8 and states its f; a statement about the look-alike's
   position relative to the dataset mean holds only if it holds at all three sizes, otherwise it is given with its
   scale (rule 4 of the 8 Nov note).
8. f = 1/16 (14 Nov 2026, after the QC sheets, before any 1/16 result of this amendment was read). At 5 px the retina
   and face windows hold almost no structure by eye (25 pixels; vessels are 2 to 4 px wide in the 128 px frame), and
   a 3 px ring nearly determines a 5 px window, so 1/16 mostly measures image smoothness, not the person. Kate kept it
   on the curves ("ладно пускай будет для кривых"): 1/16 stays a point of the RQ4 curve, labelled in the figure and
   text as the interpolation end, and no RQ4 claim rests on it alone (rule 7 applies to 1/8, ~0.2 and 1/4).

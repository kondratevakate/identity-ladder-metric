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
- No square windows in maps (added 6 Nov 2026; Kate found the same square blocks in the face maps earlier and again
  in the fundus maps). Local errors and every other map statistic use a Gaussian window with the standard deviation
  of the intended box (window / sqrt(12)), masked (weight / weight of valid pixels). A square box draws squares.
  Square windows are allowed only for validity checks (share of a window inside a mask) and for patch definitions
  of the ladder itself (cubes, squares), never for what is drawn as a map. Known square-window maps: the old fundus
  maps (`donor_ladder_2d.maps`, superseded by Amendment 9.5-9.6) and the brain maps (brain protocol, Amendment 10).
- Result spec (added 6 Nov 2026, Kate: "a checkable spec that triggers on every result computed on data").
  Every runner that computes ladder results declares `"spec": "ladder-result-spec/1"` in its JSON and calls
  `src/result_spec.py` before writing; `tests/test_result_spec.py` re-checks every declaring file in `results/`
  on each test run (tests run before every commit). Hard rules stop the run; soft rules are written into the JSON
  (`spec_checks`) and printed. Hard: every rung has ratio, a 95% CI containing it, n targets >= 10, persons <= targets
  and a named bootstrap unit (person or family; never images or eyes); the pools are leave-person-out (asserted at
  run time: a target's own person, all eyes or sides, is never in its pool, mean or look-alike search); every
  mirror-type rung reports both directions separately (left-right sides: target on the left vs on the right; within
  one image: each half against the reflected other half), and direction counts add up (between sides: the two
  directions partition the targets; within an image: each direction covers every target); a stranger rung exists.
  Soft: stranger ratio in [1.85, 2.15] (theory 2); the two directions of a mirror rung within |ln(a/b)| <= 0.15
  (else: check orientation and reflection before reading the result). Brain results adopt the spec at their rerun.
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

### Amendment 9.5 (6 Nov 2026, fundus ladder maps in the face-figure format; recorded before these results)

Kate asked for the fundus results in the same form as the face figure (`src/donor_ladder_face.py`, frame v6):
one row per dataset (PAPILA, then GRAPE), columns Mean fundus / Another photo / Mirror side / Sibling / Stranger /
16 look-alikes / Same sex and age / Dataset mean, one scale (1 = dataset mean, 2 = stranger), and under each map
MSE = ratio x mean [95% CI], split-half r, n. Definitions follow the face script, adapted only where fundus differs:
- Inputs: PAPILA P9, GRAPE P8 (Amendment 9.2), downsampled 2x to the face frame size (128 px; a pixel is valid only
  if its 4 source pixels are). Window 16 px, look-alikes on a 4 px grid by the 8 px ring, k = 16, 8 strangers, as
  for faces. On PAPILA P9 the 16 px window spans 390 raw px, on GRAPE P8 the same share of the frame.
- Targets: one image per eye (earliest valid visit). Pools per side, leave-person-out (the target's own person,
  both eyes, never enters a pool, a mean or a look-alike search).
- Donors: Another photo = the same eye's closest other visit (GRAPE; PAPILA has one visit: not available).
  Mirror side = the fellow eye (already reflected into the same orientation). Sibling: not available in either
  dataset. Stranger = mean error over 8 random same-side people. 16 look-alikes = the face `lookalike_fill` with
  masks (ring distance over pixels valid in both, candidates sharing < half of the ring ineligible, a window pixel
  needs >= 8 contributing neighbours), plus a person-exclusion matrix. Same sex and age = mean of same-side people of
  the same sex within 5 years (GRAPE: same device too), >= 10 people. Dataset mean = leave-person-out mean.
- Missing is never zero (section 8): every error uses the joint mask target x donor x valid region; the reference
  (dataset mean) error is summed over the same pixels. Valid region = pixels valid in >= 90% of the side's pool.
- Numbers: ratio = sum over targets of rung error / sum of mean error on the same pixels; 95% CI by 2000 bootstrap
  resamples of persons; split-half r between maps from two independent person halves (pools inside each half);
  n = targets with that donor. Local maps: masked box mean, a pixel needs >= half of its window valid.
Pre-QC: the camera spot, the dark-area threshold and the window scale (P10 candidates) are not fixed yet.

### Amendment 9.6 (6 Nov 2026, two mirror rungs for eyes; recorded before these results)

Kate: within one eye the symmetry axis is horizontal (the disc-fovea line, the horizontal raphe; the upper and
lower vascular arcades roughly mirror each other), not vertical as in faces and brains; and because eyes come in
pairs, the fellow eye is an extra rung of the ladder. In Amendment 9.5 "Mirror side" was the fellow eye. From now on:
- Mirror side = the same eye reflected about the horizontal line through the disc-fovea axis (the analogue of the
  face and brain mirror). A strip of 2.5 px either side of the axis (128 px frame) is excluded, as the face midline.
  GRAPE P8: the frame puts disc and fovea on one row (y = 63.625 on the 128 frame), so the axis is exact. PAPILA P9: the
  fovea is not located (it is mostly outside the 30 deg field), so the axis is the horizontal line through the disc
  (y = 61.375); each eye's real disc-fovea tilt is not removed, which can only make this mirror worse (stated in the
  figure).
- Fellow eye = the other eye of the same person, reflected into the same orientation (the former "Mirror side").
- Figure columns: Mean fundus / Another photo / Mirror side / Fellow eye / Sibling / Stranger / 16 look-alikes /
  Same sex and age / Dataset mean. Everything else as in Amendment 9.5.
- Correction recorded with 9.6 (before the rerun): Amendment 9.5 said "masked box mean", which deviates from the face
  figure (frame v6 uses a Gaussian window with the box's standard deviation, 16 / sqrt(12) px, face Amendment 4.7);
  the square box produced square blocks in the maps (Kate). Local errors now use that Gaussian. Display: in each row
  every panel shows only pixels where all its maps are defined (no grey margin where one rung is unfilled).

### Amendment 9.7 (6 Nov 2026, eye mirror combinations, directions, result spec; recorded before these results)

Two eyes give three mirror-type rungs, each reported with both directions (section 8, result spec):
- Mirror side (within one eye, top-bottom about the disc-fovea line): directions = upper half vs the reflected lower
  half, and lower half vs the reflected upper half (pixels split at the axis).
- Fellow eye: directions = right eye as target (left eye as donor) and left eye as target (right eye as donor).
- Fellow eye mirrored (new rung, the "diagonal"): the fellow eye also reflected top-bottom about the axis (2.5 px strip
  excluded), so the upper arcade of one eye meets the lower arcade of the other; directions as for the fellow eye.
Figure columns: Mean fundus / Another photo / Mirror side / Fellow eye / Fellow eye mirrored / Sibling / Stranger /
16 look-alikes / Same sex and age / Dataset mean. The JSON declares the spec and carries its checks.

### Amendment 9.8 (6 Nov 2026, groupwise fundus registration before the ladder; recorded before any result)

Kate chose registration first: Amendment 9.7 showed that every paired rung (even the fellow eye with no reflection)
rises from ~1.1-1.3 x mean along the disc row to ~1.8-1.9 x mean over the arcades, while look-alikes (which adapt to
local position through their ring) stay flat: paired donors are only aligned at the disc (PAPILA: translation;
GRAPE: disc-fovea similarity), so their vessels miss the target's.
Design (no pairwise target-donor optimisation, which would favour some rungs; the brain analogue is MNI):
- Per dataset, every analysis image (PAPILA P9, GRAPE P8; 256 grid) is registered to one groupwise template with
  ANTsPy 0.6.3 (`ants.registration`), with the fixed (template) mask and the moving (image) mask in every stage
  (`mask_all_stages`), so the filler outside a field never acts as an edge. Template_0 = valid-pixel mean of the
  current frame; template_k = valid-pixel mean of all images warped at iteration k; 3 iterations.
- Two arms, both reported (selection sweep rule): R1 = affine to the template; R2 = R1 then SyN (ANTs defaults for
  2D unless QC says otherwise). Warped masks: linear interpolation, valid only where the warped mask is ~1.
- Statement of cost: R2 bends each eye toward the group's geometry, removing part of the geometric (vessel layout)
  individuality; the ladder after R2 measures appearance after shape normalisation (as nonlinear MNI for brains).
- Gate (visual-qc-gate): prototype on 40 evenly spaced images per dataset; QC sheets (before / R1 / R2 with the
  template's vessel outline over each image) and an alignment video; Kate's verdict before any ladder rerun.
- Settings fixed after the first prototype failed (before any result): ANTs' default centre-of-mass start divides by
  ~0 on z-scored images (translation ~4e10 px, every warped mask empty), so registration starts from the identity
  (the disc frame is already the start). Metric Mattes (ANTs default; global correlation also sane; mean squares
  diverged: scale 0.87-1.13, shifts up to 28 px). Determinism: ANTS_RANDOM_SEED=1 and one ITK thread per process
  (repeats identical), parallel across processes. A failed registration is counted and reported; that image is
  missing everywhere, never silently dropped.
- R2b (6 Nov 2026, Kate likes affine + SyN but R2 visibly bends the field; recorded before running): the same as R2
  with a smoother, shorter SyN: flow_sigma 6 (R2: 3), total_sigma 1 (R2: 0), reg_iterations (20, 10, 0) (R2:
  ANTsPy default (40, 20, 0)). Compared with R2 on the same 40 images per dataset by (a) the image's vessel-pattern
  correlation with the leave-one-out template (high-pass, sigma 6 px; edge excluded) and (b) bending = mean |log det
  Jacobian| of the SyN warp inside the image's field (0 = no bending). The arm with the better trade-off goes to
  Kate's QC; both stay reported.

### Amendment 9.9 (6 Nov 2026, H1: one harmonised frame for both fundus datasets, as for faces; before any result)

Kate: stop tuning registration (R2's warps of the field look odd) and use preprocessing that makes the two datasets
alike, as the face pipeline does (both face datasets go through the same landmarks, template, scale and crop).
H1, applied identically to PAPILA and GRAPE:
- Load: the whole photo padded to a square of its longer side (outside the photo missing), box-resized to 512
  (PAPILA 2576 x 1934; GRAPE square, 1556 / 1611 / 2136). Left eyes reflected (disc right).
- Disc: one detector for both, the P7 brightness detector (disc_p4 + raw-brightness gate). PAPILA expert contours
  are not used for the frame; they validate the detector (centre error in disc radii, radius ratio), reported.
- Frame: translation of the detected disc centre to (256, 256) and one scale factor per dataset so that its median
  detected disc radius becomes 30 px (512 grid); individual disc size is kept (as Hemelings et al. 2023). No
  rotation, no deformation.
- Field: a disc-centred circle of radius 225 px (7.5 median disc radii) in both datasets; outside a photo, its
  clean mask (P8) or that circle is missing.
- Intensity: as P8 (green / background with the disc removed, z-scored inside the mask), 256 grid.
- Prototype gate: 40 evenly spaced images per dataset; QC = detector validation on PAPILA, sheets (photo with
  detected and expert disc, H1 frame, analysis image), the two dataset means side by side; Kate's verdict first.
- Outcome of H1 (6 Nov 2026, Kate's QC): rejected. Moving the disc to the centre cuts away much of GRAPE's field
  (its disc sits near the field edge), and the shared detector behaves differently at the two pixel scales (PAPILA:
  radius 0.71 of the expert's). Each dataset stays in its own frame (PAPILA P9, GRAPE P8; Amendments 9.5-9.7).

### Amendment 9.10 (6 Nov 2026, figure: GRAPE first, GRAPE's field outlined on PAPILA)

Kate: show that GRAPE covers more retina and that PAPILA is cropped. In the face-format figure GRAPE is the first
row; every PAPILA panel carries a dashed outline of GRAPE's displayed field mapped into PAPILA's frame by disc
position (frame targets) and disc size. Disc size is measured the same way in both datasets: on the dataset mean,
the bright blob around the disc target above half the way from its peak (within 10 px) to the median of the 15-30 px
ring, r = sqrt(area / pi) (a radial profile from the target failed on PAPILA, whose bright cup lies off the
geometric disc centre). Display only; no number changes.

# Protocol: the donor ladder for face photographs (5 Nov 2026)

Face instance of `PROTOCOL_donor_ladder_general.md` (branch `donor-ladder-bilateral`), recorded before any face
result. Goal: the face analogue of the brain map figure (target, repeat, mirror side, sibling, stranger, 16
look-alikes, same sex and age, dataset mean; one scale, 1 = dataset mean, 2 = stranger; split-half r under each map).
Kate decided on 5 Nov 2026 that this session builds the face frame and its QC itself. Results are pre-QC until a
human visual check of the frame, including mirrored cases, passes the gate in `AGENTS.md`.

## Data (registry ids)

- `celeba`: aligned crops, identity labels, attributes, official partition. Targets: test partition. Donor pool
  (strangers, means, look-alikes): train partition, disjoint identities. No relatives.
- `imdb-wiki` with `imdb-wiki-kinship-20261104`: targets are people in Wikidata full-sibling pairs (half and step
  siblings excluded); pool: identities in no kinship pair. Age = photo year minus birth year.
- Privacy and licences: aggregates only leave the private folder; no image, file name, identity id or name in any
  output, log or git. Figures show maps on the mean face of the pool, never a person.

## Landmarks and pose

MediaPipe Face Landmarker v2 (`mediapipe==1.2.2`, model `face_landmarker.task`, SHA-256 in the model manifest,
Apache-2.0). An image is usable if exactly one face is found. Pose (yaw, pitch, roll) from the facial transformation
matrix; expression from the blendshapes mouthSmileLeft/Right and jawOpen.

## Frame, mask, normalization

- Similarity transform (least squares) of three points, left iris centre, right iris centre and mouth centre, to
  fixed positions in a 128 x 128 frame: eyes (44, 52) and (84, 52), mouth (64, 92). No nonlinear warp.
- Mirror: horizontal flip about the frame midline x = 63.5, which is the facial midline by construction.
- Validity mask per image: face-oval polygon of the mesh warped into the frame. A pixel enters a comparison only if
  it is inside both masks; a map pixel is shown where >= 90% of targets contribute.
- Grey level Y = 0.299 R + 0.587 G + 0.114 B, z-scored inside the image's own mask.
- Pose filter (primary): |yaw| < 10 deg and |pitch| < 10 deg for every target and donor image.
- Quantitative frame QC: residual of the remaining landmarks (nose tip, mouth corners, face oval) to the pool median
  after alignment, in units of the inter-ocular distance; images above the 99th percentile of the pool residual are
  excluded and counted. A private contact sheet (aligned and mirrored) is written for Kate's visual check; the agent
  does not view face images.

## Rungs (columns)

| Column | CelebA | IMDb-WIKI | Shared with the target besides the person |
|---|---|---|---|
| Repeat | another photo of the identity, nearest in pose and expression | same | camera, session and date unknown; may be the same event |
| Mirror side | the target's own photo flipped | same | everything except side-specific anatomy, lighting asymmetry and residual pose |
| Sibling | unavailable | photo of each full sibling nearest in pose and expression; the error is averaged over siblings (expected single-sibling error, as for strangers) | family environment; different photos |
| Stranger | expectation over 8 random pool people | same | nothing by design |
| 16 look-alikes | mean of the 16 pool images nearest on the ring around each window (8 px grid) | same | context (light, pose, skin tone) through selection |
| Same sex and age | mean of the pool cell Male x Young | mean of the pool cell sex x 5-year age band | sex, age |
| Dataset mean | mean of the pool | mean of the pool | none |

One target image per person: the most frontal usable image; the repeat is chosen among the person's other images.
Selection controls (general protocol section 5) are exploratory here: the look-alike selection applied to a stranger
and the oracle look-alikes (selected on the true window).

## Maps and numbers

- Local error: box filter (16 px) of the squared difference inside the joint mask. V map: the same for the dataset
  mean. Ratio map = sum over targets of the rung's local error / sum of the V map. Under each map: the ratio over
  all valid pixels, and split-half r between maps from two random halves of persons (IMDb: halves of families; the
  pool is restricted to the same half for the mean and look-alikes).
- Denominators (targets per column) are printed with every value; columns with fewer than 10 targets are not shown.
- Bootstrap over persons (families for IMDb), 2000 resamples, for the whole-face ratios.

## Amendment 1 (5 Nov 2026, after the first map, before its rerun)

Seen: the first map (look-alikes drawn as 8 px blocks on 28 central cells; a blue strip on the midline of the mirror
column). Changes:

1. Look-alikes become a full fill image like every other column: neighbours (k = 16, nearest on the ring) are chosen
   on a 4 px grid wherever the 4 x 4 centre block is valid; each pixel takes the mean of its cell's neighbours; the
   map then uses the same 16 px box error as all columns. Ring distance uses only ring pixels valid in the pool
   coverage mask, so cells near the face edge are no longer dropped.
2. Mirror side: pixels within 8 px of the midline (x in [55.5, 71.5]) are excluded, because there the mirrored image
   is compared with itself.
3. Sensitivity (reported, not in the main figure): look-alike ring 4 px.

## Amendment 2 (5 Nov 2026): preprocessing v2, the face analogue of the MRI pipeline

Seen before writing: the Amendment 1 map, the look-alike outlier sheets (drawings, re-photographed prints, glasses,
hands, strong expressions) and duplicate counts (IMDb: 78 crop groups shared by different people; 9 sibling and 8
parent-child pairs with a near-identical photo). Kate approved on 5 Nov 2026. v1 outputs are kept; v2 writes
`frame_v2.npz`, `results/donor_ladder_face_v2.json` and `fig_face_ladder_maps_v2`.

| MRI step (`prepare_scan`) | Face step v2 |
|---|---|
| SynthStrip brain mask | MediaPipe Selfie Multiclass segmenter (Apache-2.0): mask = face-skin class within the convex hull of the mesh face oval |
| N4 bias field (field on a 4x shrunk copy) | illumination field: mask-normalized Gaussian smoothing of the grey level in the frame, sigma 16 px (0.4 inter-ocular distance), division |
| ANTs Affine to the symmetric MNI template | least-squares 2D affine of all 478 landmarks to a symmetric template shape (pool mean shape after v1 alignment, averaged with its mirror image through nearest-point correspondence) |
| linear resampling, nearest-neighbour mask | same, 128 x 128 frame |
| z-score inside own mask | same |
| no SyN | no mesh (UV) warp; UV warp only as a later sensitivity |
| QC: correlation with the template, below median - 3 MAD flagged, not excluded | same (template = mean pool image); flags are reported and shown to Kate |

Exclusion rules (images; fixed now, counts reported per rule and source):

1. No single face or |yaw|, |pitch| >= 10 deg (as v1).
2. Occlusion, CelebA: attributes Eyeglasses, Wearing_Hat or Bangs.
3. Occlusion, both datasets (segmenter, fractions of the oval hull): accessories > 1%, body skin > 2% (hands),
   hair > 5%. Agreement with the CelebA attributes is reported as a check, thresholds are not tuned on it.
4. Expression: blendshape smile >= 0.5 or jawOpen >= 0.3.
5. Duplicates: two images of different people with correlation > 0.9 in the frame are both excluded. Within a person,
   a donor with correlation > 0.9 to the target cannot serve as the repeat.
6. No inpainting of occlusions: filling them with a model would put averaged tissue into the very comparison under
   audit.

Not addressed in v2 (needs an identity/quality model, separate approval): drawings, re-photographed prints and other
people's faces under a person's label. Their effect is reported through the outlier sheets.

## Amendment 3 (5 Nov 2026): look-alike fills without block seams

Seen: v2 outlier sheets, where the look-alike fill is a mosaic of 4 x 4 blocks and the local error shows square
blobs at block borders (Kate). Change: every 4 px grid node fills its whole 16 x 16 window with the mean of its 16
neighbours; overlapping windows are averaged with a separable tent weight (1 at the window centre, falling linearly
to the window edge). Selection of neighbours is unchanged. Written to `_v2` outputs again (v2 numbers before this
amendment are kept in the chat record and the outlier score file of that run).

## Amendment 4 (5 Nov 2026): more IMDb-WIKI targets, map display rule (v3)

Seen: the v2 map after Amendment 3 (IMDb-WIKI 68 targets in 49 families; split-half r near zero for repeat,
sibling and sex-age). Kate approved the proposal ordered by effect. Outputs `_v3`; v2 is kept.

1. IMDb-WIKI: every single-face crop of every person in a kinship pair enters the landmark pass (the cap of 20 crops
   per person is removed); the added rows are appended to `images.json` and `landmarks.npz` (v2 copies kept as
   `*_v2.*`).
2. IMDb-WIKI targets: every person in a full-sibling or parent-child pair with a usable target and repeat. The
   sibling column uses only those with a full sibling. Families = connected components over full-sibling and
   parent-child pairs.
3. Pose threshold unchanged (10 deg); all Amendment 2 exclusion rules unchanged.
4. Display rule for maps: a column's map is drawn only if its split-half r >= 0.5; otherwise the panel shows the
   whole-face ratio with its interval and "map not reproducible".
5. Identity/quality filtering is a separate later step (model download needs Kate's approval of the specific file).
6. Images flagged by the template correlation (below median - 3 MAD) are excluded, not only flagged. Unlike an MRI
   scan, a failed face fit puts another part of the head in the frame (Kate saw an ear where a profile was fitted
   as frontal, 5 Nov 2026).
7. Local error window: Gaussian with sigma = 16 / sqrt(12) = 4.6 px (the standard deviation of the 16 px box)
   instead of the box, for every column; a box turns point-like errors (sclera, highlights) into squares (Kate).
   Whole-face ratios are unaffected (they use unsmoothed errors).

## Amendment 5 (5 Nov 2026): identity consistency and a narrower midline strip (v4)

Seen: v2 cases with another person as the repeat (IMDb case 8: a woman's target, a man's repeat), drawings and
re-photographed prints among outliers; the 16 px midline strip removes the nose from the mirror map (Kate). Kate
approved the InsightFace `buffalo_l` download (non-commercial research licence; SHA-256 in its manifest).

1. Identity embedding: ArcFace ResNet50 (`w600k_r50.onnx`), 112 x 112 crop aligned by a similarity transform of five
   MediaPipe points (iris centres, nose tip 1, mouth corners 61 and 291) to the ArcFace template of
   `insightface/utils/face_align.py`; input (RGB - 127.5) / 127.5; embeddings L2-normalized.
2. Every landmark-usable frontal image (|yaw|, |pitch| < 10 deg, the images that can enter the analysis) of a target
   or pool person is embedded (changed before any identity result: the first run on all usable images was stopped
   for speed). Person centroid: normalized mean of the
   person's embeddings; recomputed once from the images with cosine >= 0.3 to the first centroid.
3. Exclusion `identity_inconsistent`: cosine to the final centroid < 0.3. A person with exactly two usable images
   whose mutual cosine is < 0.3: both excluded (`identity_unresolved`). Thresholds fixed now, not tuned.
4. Mirror column: excluded strip narrowed to |x - 63.5| < 2.5 px, i.e. pixels whose mirrored partner is closer
   than the horizontal autocorrelation length of the individual residual (5 px: autocorrelation 0.60 at 4 px,
   0.41 at 6 px; measured on the v2 pool before any v4 result).
5. Everything else as v3. Outputs `_v4`.

## Amendment 6 (5 Nov 2026): minimal preprocessing, QA and selection (v5, primary)

Seen: v3 results; Kate judged the accumulated exclusions and the map display rule to be pre-selection. One principle
now: exclude only images that are not a valid measurement of the labelled person's face, with criteria that never
use a ladder metric; handle every other nuisance by preprocessing or as a reported stratum. v4 (strict selection)
is kept as a sensitivity analysis.

Preprocessing (unchanged from v2, the MRI analogues): face-skin mask within the oval hull (occluders such as
glasses, hair and hands drop out at pixel level), illumination field division, 478-landmark affine to the symmetric
template, z-score inside the mask.

QA exclusions (validity only): no single face (landmarks); template-correlation flag (fit failure); identity
inconsistent or unresolved (Amendment 5); face-skin mask under 2000 px (too little face to measure); duplicates
shared by different people.

Acquisition condition: |yaw| and |pitch| < 10 deg (the analogue of one scanner orientation; needed for the mirror).

No longer exclusions: CelebA occlusion attributes, segmenter occlusion fractions, expression. Expression becomes a
reported stratum: targets with smile < 0.5 and jawOpen < 0.3 (neutral) versus the rest.

Repeat: primary nearest in pose and expression (rung definition); sensitivity a random other image of the person.

Reporting: every map is drawn, with its split-half r; no map is hidden (Amendment 4.4 withdrawn). A flow table gives
images and people after each step, per dataset and role. Mirror strip and Gaussian window as in Amendment 5 / 4.7.

## Amendment 7 (6 Nov 2026): erratum for look-alike masks; glasses, facial hair and makeup as pixel masks (v6, primary)

Seen: v5 maps, QC sheets, cases and look-alike outliers. Kate saw glasses with transparent or tinted lenses whose
lenses stayed in the mask, a hand, heavy and theatrical makeup, and moustaches and beards that the look-alike fill
cannot reproduce. Kate decided (6 Nov 2026): the estimand is the bare face; facial hair, theatrical makeup and bright
lipstick are treated like accessories, and their pixels are removed from every calculation (not the whole image).
Everything is fixed here before any v6 result. v5 is kept as a sensitivity analysis (beard and makeup in).

Erratum (affects v1 to v5). `lookalike_fill` ignored the masks of pool images: pixels outside a pool image's mask are
stored as 0 and entered the ring distance and the neighbour mean as real values. This pulled the fill towards 0 and
matched neighbours on the shape of their occlusions. Correction:

1. Ring distance: mean squared difference over ring pixels inside both the target's and the candidate's mask;
   candidates sharing fewer than half of the target's available ring pixels are not eligible.
2. Fill: each pixel is the tent-weighted mean over the neighbours whose mask contains that pixel; a pixel with fewer
   than 8 of the 16 neighbours stays unfilled.
3. A unit test checks that a pool with holes gives the same fill as the same pool without holes and does not shrink
   to 0. The v5 maps are recomputed with the correction (`_v5` outputs; the uncorrected files are kept as
   `*_prefix`).

Masks (all in the 128 px frame after the template affine; each pixel removed is counted per rule, source and sex):

4. Glasses. Trigger: fraction of segmenter accessory pixels in the eye band (rows 36 to 68) above a threshold chosen
   by Youden's J against the CelebA Eyeglasses attribute (reported with sensitivity and false positive rate; the
   attribute is not used for the mask itself). When triggered, removed: two template lens ellipses centred on the
   template eye centres (horizontal semi-axis 17 px, vertical 12 px, i.e. 0.42 and 0.30 inter-ocular distances), the
   bridge between them (+-3 px around the eye line), and the accessory pixels dilated by 3 px.
5. Colour deviation (facial hair, theatrical makeup, face paint). Per image, the skin reference is the median CIELAB
   colour of mask pixels in the upper-cheek and nose zone (between the eye band and the nose base, outside the eye
   and brow polygons). A pixel is removed when its CIELAB distance to the reference, with lightness weighted 0.5,
   exceeds the 99th percentile of the same distance in clean CelebA reference images (No_Beard, no Heavy_Makeup, no
   Wearing_Lipstick, no Eyeglasses). The flag map is cleaned by a 3 x 3 median filter. Lips, eye and brow polygons
   are exempt (they differ from skin by anatomy) and are handled by rule 6 and the segmenter. The rule is applied to
   all images; facial-hair sensitivity is reported as the fraction of the mouth-to-chin zone removed in CelebA images
   with and without No_Beard, and of makeup against Heavy_Makeup.
6. Bright lipstick. The lip polygon (MediaPipe outer lip contour) is removed when the lips' median CIELAB a* minus
   the skin reference a* exceeds the 95th percentile of the same contrast among men (CelebA Male, both datasets'
   male sex labels), a reference group that does not wear lipstick.
7. Greyscale photographs (median skin chroma below 3): rules 5 and 6 use lightness only; counted.

Coverage. `valid` (where maps and the dataset mean are defined) is now: anatomical coverage, the oval hull inside the
photograph, in >= 90% of the pool, and at least 50% of the pool unmasked at that pixel. Before, both were the same
mask; with facial hair removed in about 17% of the pool, the old rule would have removed the chin and upper lip from
every map.

Strata added to the report: glasses (triggered or not), facial hair (more than 10% of the mouth-to-chin zone removed
by rule 5), and the fraction of the face removed by rules 4 to 6. QC for Kate: images with the most pixels removed
per rule, original next to the masked frame. Everything else as v5. Outputs `_v6`.

## Amendment 7b (6 Nov 2026): symmetric colour rule (masks P7b)

Seen: v6 mask QC (Kate): the colour rule removed one-sided shadows, not beards; agreement with CelebA attributes was
weak (facial hair 8.5% vs 1.4%, lipstick 21.9% vs 10.5%, makeup 2.7% vs 5.0%). Kate proposed symmetry. Facial hair,
makeup, face paint and lipstick are bilateral; side lighting is not. Change to rule 7.5: the per-pixel statistic is
s = min(d(x, y), d(mirror x, y)), the colour distance of the pixel and of its mirror partner about the frame midline
(both must be eligible skin; otherwise the pixel is not flagged). Threshold: the 99th percentile of s in the same
clean CelebA reference images. Rule 7.6 (lipstick) is withdrawn: the v6 QC showed it removing natural lips,
including men's (Kate: lips stay); lips remain exempt from the colour rule. The nose tip and alae (hull of
MediaPipe points 1, 2, 4, 19, 64, 94, 97, 98, 99, 129, 168, 219, 240, 294, 326, 327, 328, 358, 439, 460, dilated
2 px) are added to the exempt zones: dark nostrils are bilateral anatomy and were removed by the colour rule (Kate). Everything else in Amendment 7 unchanged. Gate before any full run: smoke run and a
QC sheet of 24 evenly spaced images (original, flags of the old rule, flags of the new rule), looked at by the agent
and by Kate.

## Amendment 7c (6 Nov 2026): facial hair by an image-level texture detector and a landmark zone (masks P7c)

Seen: 7b smoke QC (Kate and the agent): one-sided shadows were gone, but ginger and grey beards were missed, and the
rule flagged frown lines on the midline (where a pixel's mirror partner is its neighbour), the jaw contour and
isolated pixels at the oval edge. Pixel colour thresholds answer "does this pixel differ from skin", not "is this a
beard". Kate approved (6 Nov 2026):

1. The colour rule (7.5, 7b) is withdrawn entirely; theatrical makeup and occluders the segmenter misses (a
   microphone, hair on the forehead) are handled through the look-alike outlier sheets, with a recorded reason.
2. Facial-hair detector, image level, colour-blind: logistic regression (L2, lambda 30) on the grey z-scored frame
   inside the skin mask, lower face rows 76-123 and columns 24-103: 2 x 2 block means (960 values) and Sobel gradient
   magnitude averaged in 8 x 8 blocks (60 values), standardized. Trained on CelebA men of the v5 frame table, target
   CelebA No_Beard = 0. Measured before this amendment, 5-fold cross-validation by person: AUC within men 0.90 (No_Beard),
   0.86 (Mustache), 0.90 (Goatee), 0.69 (5_o_Clock_Shadow); ArcFace embeddings gave 0.71 and were not used. Threshold:
   Youden's J on the cross-validated scores; final weights fitted on all CelebA men. Applied to every image (both
   sexes); trigger rates are reported per dataset and sex.
3. When triggered, the facial-hair zone is removed: anatomical oval below the nose base (MediaPipe point 2), lips
   excluded. Glasses as in Amendment 7.4. Nothing else changes from v6.
4. Gate: smoke QC on 24 images evenly spaced in the table, 12 evenly spaced CelebA men with No_Beard = 0, and 12
   evenly spaced IMDb men (a domain the detector was not trained on); looked at by the agent, then by Kate, before
   the full run. Outputs `_v6c`.

## Amendment 7d (6 Nov 2026): facial hair stays in the face (masks P7d, final for v7)

Seen: 7c smoke QC (agent and Kate). The texture detector found dark, ginger and grey beards, but its CelebA label
counts stubble as facial hair, it missed a long white beard, fired on two women (open mouth, a microphone), and the
zone removed the whole lower third of the face for about 40% of men. Kate decided (6 Nov 2026): facial hair is kept
("we cannot beat it"); no facial-hair mask. Final pixel masks: segmenter face skin in the oval hull and the
photograph, plus glasses (Amendment 7.4). The colour rule stays withdrawn (7c.1). The in-house detector
(`train_beard`, `beard_v6c.json`, CV AUC 0.91 within CelebA men) is not used at all, not even for a stratum (Kate:
an own detector is not what the paper needs, and it fired on women). Facial hair is reported only where the dataset
labels it: CelebA men by the published attribute No_Beard. v6 to v6c outputs are kept as the record of these
attempts.

## Amendment 8 (6 Nov 2026): selection without pixel similarity, symmetric sampling, age and pose (v7)

Seen: the v5 selection flow diagram (Kate asked for a fresh look at what is wrong for the task). Written while v6 was
still running; no v6 result had been seen. v6 stays the comparison point. Kate's decisions in brackets.

1. Fit-failure QC without pixels [agreed]. The template-correlation rule correlates each image with the pool mean,
   i.e. the denominator of the ladder; it removes atypical faces and shrinks the scale. Primary: landmark residual
   after the 478-landmark affine to the symmetric template, in inter-ocular distances, excluded above the 99th
   percentile of the pool (the v1 frame-QC rule, now applied after the template affine). The old rule becomes a
   sensitivity analysis. Reported: overlap of the two flag sets, and a private sheet for Kate of images flagged by
   only one of them (to check that fitted profiles, the "ear" case, are still caught).
2. Identity filter and age [check; keep age in the analysis]. Measured before writing (IMDb, 8,726 embedded images
   with an age, persons with >= 3 images): exclusion rate by age gap to the person's median age 3.7% (0-5 y), 5.1%
   (5-10 y), 5.8% (10-20 y), 7.3% (20-40 y); 9.7% of excluded versus 5.0% of kept images are >= 15 y from the
   median. The ArcFace filter stays (it removes other people, a validity issue) and age becomes explicit:
   IMDb repeat strata by the target-repeat age gap (0-4, 5-14, >= 15 y), and a rung "age-distant repeat", the
   person's usable image with the largest age gap. The age gap of the identity-excluded images is reported.
3. Symmetric sampling of the pool [Kate asked 2 or 3 images]. The pool mean uses one image per person (the most
   frontal); the number sampled only sets the chance that a usable frontal image exists (3 images: 1,938 of 2,500
   CelebA pool people survive pose, versus 978 of 1,000 targets with all their images). Pool persons now contribute
   up to 10 images each (all if fewer; same identities, seed 2026), added to the landmark pass as new rows. Two
   images would lower survival further.
4. Pose sensitivity [check 15 and 20 deg]. Prep runs once with |yaw|, |pitch| < 20 deg; the maps stage filters
   targets, repeats and pool at 10 (primary), 15 and 20 deg. The mirror rung is reported only at 10 deg (it needs a
   frontal face); at 15 and 20 deg it is shown as exploratory. All three are written (`_v7_p10`, `_v7_p15`,
   `_v7_p20`) and reported side by side with denominators. Note from the pose-band preview (16 evenly spaced
   images per band, seen by the agent and Kate before any v7 result): MediaPipe pitch has median +4.6 deg (IQR -1.2
   to 9.8) over 90,437 images, so half of the 10-15 deg band is near-frontal faces exceeding 10 deg in pitch only;
   yaw above about 15 deg visibly compresses the far cheek. Kate kept one threshold on max(|yaw|, |pitch|) at
   10 / 15 / 20 deg; the pitch offset is reported in the methods.
5. Masks and everything else as v6 (Amendment 7). The flow diagram (`flow` stage) is regenerated for every version.

## Amendment 9 (6 Nov 2026): v6 frozen as the face result; v7 parked

Seen: v7 prep at 20 deg and its fit-QC gate sheets (A: 699 images excluded by the landmark-fit rule; B: 2,152 images
the template-correlation rule would flag but v7 keeps), evenly spaced, looked at by the agent and Kate. No v7 map was
computed. Findings: the landmark-fit rule acts as a pose filter (96% of its exclusions lie in the 15-20 deg band,
median |yaw| 18.5 deg versus 6.0 deg in kept images; none below 10 deg); it does not select expressions (CelebA
Smiling 0.49 excluded versus 0.54 kept); 67% of its exclusions are women versus 53% of kept images. The
template-correlation flags (B) were mostly sunglasses, hats and caps, dark skin and hard lighting, i.e. appearance,
not misfit.

Decision (Kate, 6 Nov 2026): across v2 to v6 the rungs moved by at most about 0.05 (v5 to v6: CelebA repeat 1.29 to
1.32, mirror 0.96 to 1.01, look-alikes 0.55 to 0.56; IMDb repeat 1.39 to 1.40, sibling 1.73 to 1.74, look-alikes
0.58 to 0.59) and their order never changed. v6 is the frozen face result. Open point: the v6 run still carries the
withdrawn colour rule (31.7% of images, median 36 px) and lipstick rule (12.7%, median 309 px); the final mask rule
(Amendment 7d: segmenter plus glasses) has not yet been run on the v6 selection.

## Amendment 10 (6 Nov 2026): v6f, the v6 selection with the final masks (frozen face result)

Kate decided (6 Nov 2026) to rerun v6 with the final mask rule rather than freeze masks she had withdrawn. v6f is v6
in every respect (rows of images.json before Amendment 8.3, pose < 10 deg, identity filter from identity.npz, the
same template rows, template-correlation QC, valid rule and masked look-alike fill of Amendment 7) except the masks:
segmenter face skin plus glasses only (Amendment 7d); no colour rule, no lipstick rule, no facial-hair detector. The
mask rules per version are an explicit table in code (`mask_rules`, with a unit test), because '_v6f' sorts after
'_v6c' as a string. Gate: a QC sheet of evenly spaced v6f frames with their masks, seen by the agent and Kate, and
the v6f maps next to v6. Outputs `_v6f`. v6f replaces v6 as the frozen face result of Amendment 9; v6 becomes a
sensitivity row.

Result and approval (6 Nov 2026). v6f: CelebA 921 targets, IMDb-WIKI 678 (247 with a sibling). Rungs within 0.03 of
v6, same order (CelebA repeat 1.30, mirror 0.99, look-alikes 0.54; IMDb repeat 1.40, sibling 1.73, look-alikes 0.57).
Mask QC of 24 evenly spaced frames (agent): glasses removed as lens ellipses, lips and facial hair kept, no shadow
removal; one glasses false positive (a woman without glasses, both eyes removed), consistent with the measured false
positive rate of about 4%, kept and reported. Kate approved v6f as the final face result. Earlier versions are reported in one supplementary table as robustness to preprocessing. v7
(Amendment 8: landmark-fit QC, pool of up to 10 images, age strata, pose 10/15/20 deg) is parked: its prep outputs
stay in the private folder, no maps are run, and it is not reported except for the template-correlation finding in
Limitations 1.

## Limitations (recorded 6 Nov 2026, Kate; both modalities)

1. Population template as reference. Both pipelines register every image to one population template (faces: the
   symmetric pool-mean shape; MRI: MNI 2009a symmetric) and QC against it. Any QC by similarity to the template
   (intensity correlation) removes or flags atypical anatomy (faces: unusual faces; MRI: atrophy, large ventricles)
   and so acts on the denominator of the ladder. Faces: in the frozen v6 result the template-correlation rule is the
   QC; the v7 gate (Amendment 9) showed that the images it flags are mostly sunglasses, headwear, dark skin and hard
   lighting, so it can under-represent these groups. A landmark rule (Amendment 8.1) avoids this but is parked.
   MRI: `prepare_scan` only flags (median - 3 MAD), it does not exclude; the flag rate by atypicality should still be
   reported.
2. Same-person comparisons pass through the population template. Target and repeat are each resampled to the template
   rather than to each other, so interpolation and template bias enter the repeat rung twice and the repeat is
   judged in a population frame. The preferable design for the repeat (and mirror) rungs is a within-subject
   reference: an unbiased template built from the person's own images, as in the FreeSurfer longitudinal stream for
   MRI, with every image of the person resampled once to it. For faces, the analogue is a per-person mean landmark
   shape (from all usable images of the person) to which the target and repeat are aligned. Not implemented in v1 to
   v7; to be added as a sensitivity analysis (reference to be verified before it enters the paper).
3. Status (Kate, 6 Nov 2026): items 1 and 2 are limitations of the paper as a whole. For the current runs, landmark
   alignment to the population template is accepted. Backlog: optimise the alignment so that it keeps more individual
   variability (within-subject template, or a registration that removes pose and acquisition but not shape).

## Process

Code: `src/donor_ladder_face.py`, tests `tests/test_donor_ladder_face.py`. Private outputs:
`C:\Projects\data\projects\identity-ladder-metric\face-ladder-20261105-v1\`. Public: aggregate JSON and the figure
only. Any change after the first map is an amendment recorded here before its results.

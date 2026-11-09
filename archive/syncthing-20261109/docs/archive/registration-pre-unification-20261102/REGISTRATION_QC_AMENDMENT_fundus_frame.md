# Registration QC amendment: GRAPE in the disc-fovea frame (2 Nov 2026)

Status: recorded before any sheet of this amendment is generated or inspected. Gate NOT MET. No metric
scoring is authorized. This is an addendum to `REGISTRATION_QC_PROTOCOL.md` (owned by another session,
left unchanged); where the two differ for GRAPE, this file states which applies and why. Iris and MRI are
not affected.

## Why

The GRAPE development sheets (`registration-qc-fundus-20261102-v2`) reviewed a keypoint registration
(SIFT, RANSAC similarity). Four problems, seen on those sheets:

1. **Wrong method under review.** Scoring uses the disc-fovea frame of 2D Amendment 2b
   (`ladder2d_register.fundus_canonical`), not keypoints. QC must review the transform that scoring
   would apply. Keypoints also cannot match a fellow eye (all fellow pairs no-fit) or a stranger, which
   2b already documents.
2. **Crop without local features.** The diagnostic box (183 to 329 on the 512 canvas) equals the metric
   mask `MASK[80:144, 80:144]` on the 224 grid. In the disc-fovea frame it spans fovea (x = 181) to disc
   (x = 330): the papillomacular region. In tessellated eyes it holds almost no retinal vessels, so the
   "three local features within two pixels" criterion can rarely be met there. Such rows stay
   UNREVIEWABLE; the criterion is not relaxed.
3. **Fitted points lie outside the crop.** Matches cluster around the disc and arcades, so the transform
   is unconstrained inside the box.
4. **Edge panel cannot show misalignment.** v2 drew Canny edges of the CLAHE-enhanced anchor and of the
   un-enhanced moving image, so the moving edges are mostly missing.

## What changes for GRAPE

**Registration under review.** The 2b frame exactly as committed: `ladder2d.load_base(path, 512, rgb=True)`,
then `fundus_landmarks` and `fundus_canonical`. Code hash recorded per run. Fellow eye: the moving image's
canonical frame is reflected (x to 511 - x), as 2b specifies for the caller. Reflected-input control: the
moving input is reflected, explicitly reflected back, then canonicalized. Missing landmarks are a
preparation failure (black panels), never an unregistered fallback.

**Landmark check comes first.** The frame is defined by two detected points, so each image is reviewed
for:

- disc: the marked point lies inside the visible optic disc rim;
- fovea: the marked point lies within 0.5 disc diameters (about 30 pixels at 512) of the visible foveal
  centre.

If the fovea is not visible (for example a tessellated, myopic macula), the row is UNREVIEWABLE, not
PASS. A wrong landmark is FAIL for every comparison that uses that image. Bright peripapillary atrophy
next to the disc is a known risk for the disc detector and is checked explicitly.

**Three boxes per comparison, labelled and never swapped:**

| Box | Canonical 512 frame (x0, y0, x1, y1) | Role |
|---|---|---|
| `metric` | 183, 183, 329, 329 | The existing metric mask. Required for any 2D-ladder scoring. |
| `arcade_sup` | 208, 72, 304, 168 | New. Superior temporal arcade, about 2 disc diameters above the disc-fovea line. |
| `arcade_inf` | 208, 344, 304, 440 | New. Inferior temporal arcade, mirror of `arcade_sup`. |

The arcade boxes are placed from anatomy alone: disc-fovea distance 149 px, about 2.5 disc diameters, so
one disc diameter is about 60 px. They were fixed before any image of this run was generated. They are
candidates for the inpainting paper's mask. They do not replace the metric mask, and passing them does
not release metric scoring on the metric mask.

**Review sheet.**

- Row 1: anchor and moving source images with the detected disc (circle) and fovea (cross), and all three
  boxes mapped back to source coordinates; then both canonical frames with the boxes and target points.
- One row per box: anchor crop, moving crop, vessel overlay, checkerboard.
- Vessel overlay: green-channel black top-hat with identical parameters on both images; anchor red,
  moving green, overlap yellow. This is an aid to inspection, not a landmark.

Same display window for both images of a pair.

**Artifact check.** A small structure at the same sensor position in images of different eyes or people
is treated as an optical artifact, not a retinal landmark, and is not counted among the three local
features.

**Unchanged.**

- Review population: same frozen manifest (`registration-qc-independent-20261102-v1/manifest.json`), same
  first-pair-per-stratum development pilot, no reselection or replacement.
- Denominators, PASS/FAIL/UNREVIEWABLE definitions and the 90% rule per stratum, including the fellow-eye
  stratum.
- Kate's acceptance, agent review and automated flags stay separate fields.
- The development pilot covers only 2 people in 7 rows. It is a development set, not evidence of
  generalization.

**Outputs.**

- Private: `registration-qc-fundus-frame-20261102-v1/`.
- Public: `results/registration_qc_fundus_frame.json`, which holds counts only, with no IDs, paths or
  images.

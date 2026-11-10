# Additional datasets for registration QC — 2026-11-02

Scope: dataset/input audit and registration QC only. Identity metrics remain paused. New datasets cannot replace failed cases in the frozen GRAPE/UPOL/UTIRIS gate. This screen does not confer a visual pass or change its denominator. Pixel-viewing authorization in this chat covers GRAPE, UPOL, UTIRIS and QTIM; no new dataset pixels were sent to the model during this screen.

## Verified local inventory

Canonical raw roots from the shared registry were enumerated, excluding nested reparse points in the saved audit. Counts are files, not independent participants. Evidence: private `C:/Projects/data/projects/identity-ladder-metric/dataset-screen-20261102-v1/local_inventory.json`.

| Dataset | Verified local content | Use and limitation |
|---|---|---|
| PAPILA, figshare 14798004 v2 | 488 original JPEG + 488 contour-overlay JPEG; 1,952 canonical contour files + 11 additional files | First local addition for bilateral fundus geometry. 244 people, both eyes according to the source paper. No longitudinal repeat claim. Use original images only. |
| RIDB, Mendeley v1 | 100 JPEG | Source confirms 20 people, five samples each. Timing and allocation to eyes are not established: neither bilateral nor longitudinal eligibility is confirmed. |
| Tongji contactless palmprint | 12,000 original TIFF + 12,000 derived ROI BMP | Both palms and repeat sessions per registry; promising non-eye mirror/crop control. Originals and ROIs are the same observations. Subject/session mapping needs a private audit before selection. |
| AMI ear | 700 JPEG | Registry describes same-sitting variants and bilateral images, not longitudinal repeats. |
| 11k Hands | 11,076 JPEG + one metadata CSV | Hand-side and surface metadata must be checked before pairing. Different views are not independent people or verified temporal repeats. |
| Dorsal hand veins | 1,782 TIFF + 1,782 PNG | Two representations must not be counted as two cohorts. Registry describes bilateral and two-month repeats; mapping and format equivalence still need verification. |

SIMON is also already local: the existing registry audit reports 99 T1 acquisitions, 73 sessions, **one person**. This screen confirmed its raw directory exists and contains 599 NIfTI files across modalities; it did not revalidate the T1/session selection. Useful for scanner/time nuisance QC, not an independent population. IXI is already local but cross-sectional; it is not a replacement for a repeat-scan control. FEI, MUCT and partial SpeakingFaces are additional registry-listed candidates, lower priority for the present eye/MRI registration failures; their files were not recounted here.

### PAPILA structural QC completed

`src/registration_dataset_audit.py` reads annotations locally and emits aggregates only. Private result: `dataset-screen-20261102-v1/papila_structure.json`.

- Original-source denominator: 488 images, 244 OD and 244 OS. All 488 have a corresponding contour-overlay image.
- Expected canonical annotation denominator: 488 images × disc/cup × two experts = 1,952. All 1,952 parse as finite two-column polygons with at least three points. This is a structural input check, not anatomical accuracy or visual acceptance.
- Eleven additional contour files are preserved and not silently substituted for canonical annotations.
- All 976 JPEG files have distinct SHA256 hashes. Directory roles establish that half are overlays; distinct byte hashes do not imply independent observations. Both groups have dimensions 2576 × 1934.
- Never fit image registration using `ImagesWithContours`: drawn lines would contaminate correspondence estimation. Independent contours may inform coarse anatomy QC, but disc/cup contours do not supply three distributed local vessel correspondences.

## Primary-source search

| Candidate | Source-supported content | Access checked today | Priority |
|---|---|---|---|
| FIRE | 129 images from 39 people; 134 registration pairs, S=71/P=49/A=14; paired landmarks and FOV masks | Official page indexed with 264 MB download; direct page requests timed out. Primary paper accessible. Archive not acquired or verified. Data license not established in this screen. | First external fundus registration benchmark. |
| IRISSEG-CC | Manual pupil/iris/eyelid parameters for BioSec 1,200, CASIA v3 Interval 2,655, MobBIO 800 images | Official GitHub contains annotation ZIPs; original images are separate. Repository MIT license is not authorization for the source-image datasets. No download here. | Directly relevant to independent pupil/limbus and eyelid checks. |
| IRISSEG-EP | Elliptical iris and polynomial eyelid boundaries; 7,966 masks across four source datasets | Official page currently offers a request form, not an unrestricted verified archive. No request submitted. | Better geometric reference for the failed concentric-circle assumption. |
| Maclaren test–retest | Primary paper: 3 people, 120 T1 volumes, two scans per session with repositioning, 20 sessions over 31 days per person | Primary publication located; current archive/version retrieval not verified. | Small external MRI repeat-control candidate after current QTIM audit. |
| ON-Harmony | Primary resource publication: 20 people, more than 160 multimodal sessions | Publication located; current release/file inventory not verified. | Scanner/domain-shift control; freeze release and available T1 subset before use. |

IRISSEG does **not** annotate UPOL or UTIRIS. It can validate a segmentation approach externally, but cannot replace manual/independent anatomy review on the frozen current cases. Eyelash occlusion is not included in the published IRISSEG annotation, so a visible-tissue mask needs additional handling.

UBIPr's official old URL now advertises a moved location; a working archive was not verified. AFHIRIS and OLIVES remain search candidates, not verified acquisitions. Historical `DATASETS.md` labels such as FIRE “unverifiable” are superseded only to the extent supported above: source and annotation design verified, archive availability not yet verified.

## Next discriminating steps

1. Complete independent local landmark checks on the same GRAPE development pairs and continue pupil/limbus/eyelid geometry on the same iris failures. No case replacement or lowering of the gate.
2. Resolve FIRE archive access and freeze an external registration-only development/validation split with landmarks reserved for evaluation; include explicit reflected-input controls. Do not train and assess on the same correspondences.
3. Use PAPILA original images plus separately retained expert contours for a new, separately frozen bilateral anatomy QC set after its image-viewing scope is authorized. Inspect the 11 extra contours before choosing any noncanonical version.
4. Finish QTIM native/crop/composite-transform audit and freeze its intended repeat/mirror denominator. SIMON or Maclaren can add repeat controls, not erase current QTIM failures.

Current gate unchanged: latest 2D development review has 13 cases, 0 passes, 6 fails, 7 unreviewable; full intended 2D set is 1,465 comparisons. MRI four box/branch decisions on one repaired scan remain unreviewable. The 90% gate is not met.

## Primary references

- FIRE official: https://projects.ics.forth.gr/cvrl/fire/ ; paper: https://www.aivojournal.com/index.php/AIVO/article/download/42/90/540
- IRISSEG-CC: https://github.com/HalmstadUniversityBiometrics/Iris-Segmentation-Groundtruth-Database ; description: https://mw.hh.se/caisr/index.php/Iris_Segmentation_Groundtruth
- IRISSEG-EP: https://www.wavelab.at/sources/Hofbauer14b/
- PAPILA: https://www.nature.com/articles/s41597-022-01388-1 ; release DOI: https://doi.org/10.6084/m9.figshare.14798004.v2
- RIDB: https://data.mendeley.com/datasets/tjw3zwntv6/1 (CC BY 4.0 stated)
- Tongji: https://cslinzhang.github.io/ContactlessPalm/
- Maclaren: https://doi.org/10.1038/sdata.2014.37
- ON-Harmony: https://pmc.ncbi.nlm.nih.gov/articles/PMC11992253/

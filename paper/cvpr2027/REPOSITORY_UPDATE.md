# Fork update and local consolidation, 9 November 2026

Branch: `codex/cvpr2027-manuscript`. Remote: the owner's fork `kondratevakate/identity-ladder-metric`. Base before manuscript commits: `80a094c7c3cab36d51d0e290b0634cca0ccf8d36`.

## New material on the fork

The fork advanced by 33 commits and 82 changed paths after the manuscript's a48523d evidence freeze. Additions include full iris P3 and its window analysis, face LPIPS-VGG and sizes, brain Amendment 22 LPIPS/FID/FWD, IXI Amendment 23 (100 targets), completed FastSurfer volume processing, and cross-modality generative-metric summaries. These are new metric/control exports, not proof that model-output subgroup evaluation was run. FastSurfer completion is not a completed Brain-ID embedding metric.

The iris handoff reports an H4 failure, H5 point-value support with flagged bootstrap intervals, and H6 support. These exports now exist in Git. Their scientific integration, denominator checks and statistical review remain separate from this preservation task. The manuscript build explicitly loads its original evidence from Git, preventing silent P2/P3 substitution.

## Local material and cleanup

The former Syncthing folder contained 456 files outside `external/`. Its separate CVPR draft, registration/QC work and face pipeline were compared by content hashes. The preservation manifest identifies canonical files, branch-history blobs, LF-only copies and unique archived versions. Registration variants stay archived with their own matching context; they do not overwrite current experiment owners' code. The face loader now defaults to the repository-relative recovered source.

The second folder, `identity-ladder-repro`, contained only an empty `.git` directory and no work or Git objects. One canonical clone is retained. Ignored third-party runtimes are consolidated locally rather than published. Loose review archives, temporary build copies and superseded figure renderings are removed only after a preservation commit is verified on the fork. Their unique historical content remains recoverable from Git.

Immutable historical-language exceptions are listed in the archive README; no Russian text is introduced into new manuscript/documentation or figure labels. No participant images, identifiers, credentials or restricted UTIRIS photographs are included in the added public artifacts.

## Commits incorporated

```text
ff760ce | Face F2 panel: resumable jobs of 50 targets per position
8443499 | Face protocol: Amendment F3.1, minimal object-unit sweep for the night
f4bd614 | Face F3: rank-stability stage and window-size figure
e2df476 | Iris Amendment P3: full-run aggregates, hypotheses H4-H6 and figures
82ae541 | HANDOFF: night plan 8-9 Nov with Kate's stop rule and paper scope
4c58f32 | Cross-modality: dataset card table and generative/topology block (LPIPS, Betti, persistence, iris code, FID, FWD)
a4702c4 | HANDOFF: explorer night items done (dataset card, generative block)
6461725 | Iris Amendment P4: panel at 10 px (f = 1/8) next to the 16 px reference, UTIRIS NIR
eaed1e4 | Brain tier 2/3 summary on both boxes; window sweep at 8 mm (Amendment 21)
85d5a02 | Face Amendment F2: full panel aggregates on CelebA and IMDb-WIKI (v6f)
cdbb4d7 | Cross-modality generative block: add faces F2 and brain Amendment 20 (LPIPS, Betti, persistence, FID, FWD)
79b151a | Face F3.1: window sizes in face-extent units, rung-order stability at 5/10/20 px
f628055 | HANDOFF: night plan status 9 Nov 00:20 (faces F2/F3, brain, fundus, iris, integrity limitation)
92623f8 | Brain window sweep: 32 mm run, rank stability, Amendment 21 result
a1b9854 | Brain protocol: Amendment 22 (LPIPS VGG, set-level FID/FWD), before results
4f45c9a | Face Amendment F2.1: LPIPS VGG as a new pass on the F2 fills (protocol and code before results)
8128418 | Brain protocol: Amendment 23 (IXI minimal run, other side as R anchor), before any IXI scan
fd17303 | FastSurfer remainder on a GCP CPU VM: protocol device note and runner, before scoring
ec5388d | FastSurfer CPU VM: version check runs in the venv; no silent exit under set -e
9862e99 | Face panel: refuse to start when workers do not fit in free RAM
5e8c53b | HANDOFF: backlog item, downstream significance threshold per dataset (Kate, 9 Nov)
26b9dd3 | Cross-modality: values behind the generative ranks and the mean-of-16-look-alikes rung
b0a6653 | Brain FastSurfer: 224/224 volumes done (laptop 79 + GCP CPU remainder 145)
893d2a1 | Face F2.1: IMDb-WIKI jobs on a cloud VM without CelebA rows; cross-machine reproduction note
5c0055b | IXI (Amendment 23): 100 scans prepared on a GCP VM, D7 QC passed
2240ea8 | Brain Amendment 22: LPIPS VGG and set-level FID/FWD on the frozen boxes (v5)
014077d | Cross-modality generative block: brain Amendment 22 (LPIPS VGG, set-level FID/FWD)
6c581d2 | Brain IXI ladder (Amendment 23): 100 targets, other side as the R anchor
8f672cf | Cross-modality: add IXI (brain Amendment 23) to the common frame, rank tiles, agreement and dataset card
6252498 | Cross-modality generative block: IXI listed as not run (tier-1 metrics only in Amendment 23)
e4a301d | Face F2.1: LPIPS VGG aggregates on CelebA and IMDb-WIKI (sibling block panel_f2_lpips_vgg)
376e108 | Cross-modality generative block: faces LPIPS VGG (Amendment F2.1)
80a094c | HANDOFF: GPU-budget items done (LPIPS VGG, brain FID/FWD, IXI) and spend so far
```

## Changed source/result paths

```text
M	docs/ANALYSIS_cross_modality.md
M	docs/HANDOFF.md
M	docs/HANDOFF_face_metrics.md
M	docs/HANDOFF_iris_metrics.md
M	docs/PROTOCOL_donor_ladder_brain.md
M	docs/PROTOCOL_face_metrics.md
M	docs/PROTOCOL_iris_metrics.md
A	gcp/fastsurfer_cpu_vm.sh
A	gcp/ixi_prep_vm.sh
M	results/METRICS_STATUS.json
M	results/METRICS_STATUS.md
M	results/cross_modality/agreement.json
M	results/cross_modality/agreement.png
M	results/cross_modality/common_all.json
A	results/cross_modality/dataset_card.json
A	results/cross_modality/dataset_card.png
M	results/cross_modality/drawn_face_thumbs.npz
A	results/cross_modality/generative_status.json
A	results/cross_modality/generative_tiles.png
M	results/cross_modality/rank_tiles.png
M	results/cross_modality/rank_tiles_by_dataset.png
M	results/cross_modality/ranks_all.json
A	results/cross_modality/ranks_generative.json
A	results/cross_modality/values_generative.json
A	results/donor_ladder_brain_ixi.json
A	results/donor_ladder_brain_metrics_size.json
A	results/donor_ladder_brain_metrics_t23.json
A	results/donor_ladder_brain_metrics_t23_v5.json
M	results/donor_ladder_brain_seg_gcp_run.json
M	results/face_ladder_metrics.json
A	results/face_ladder_sizes.json
A	results/figures/fig_brain_metrics_t23.pdf
A	results/figures/fig_brain_metrics_t23.png
A	results/figures/fig_brain_window_sizes.pdf
A	results/figures/fig_brain_window_sizes.png
A	results/figures/fig_face_window_sizes.pdf
A	results/figures/fig_face_window_sizes.png
A	results/figures/fig_iris_p3_blocks.pdf
A	results/figures/fig_iris_p3_blocks.png
A	results/figures/fig_iris_p3_setlevel.pdf
A	results/figures/fig_iris_p3_setlevel.png
A	results/figures/fig_iris_window_f8.pdf
A	results/figures/fig_iris_window_f8.png
M	results/iris_ladder_metrics.json
A	results/iris_ladder_metrics_w10.json
M	src/cross_modality/README.md
M	src/cross_modality/agreement.py
M	src/cross_modality/build_explorer.py
M	src/cross_modality/common_frame.py
A	src/cross_modality/dataset_card.py
M	src/cross_modality/drawn_rungs.py
M	src/cross_modality/explorer_template.html
A	src/cross_modality/generative_block.py
M	src/cross_modality/rank_tiles.py
A	src/donor_ladder_brain_ixi.py
M	src/donor_ladder_brain_metrics.py
M	src/donor_ladder_brain_metrics_size.py
A	src/donor_ladder_brain_metrics_size_figure.py
M	src/donor_ladder_brain_metrics_t23.py
A	src/donor_ladder_brain_metrics_t23_figure.py
A	src/donor_ladder_brain_metrics_t23_v5.py
M	src/donor_ladder_brain_seg_gcp.py
M	src/face_ladder_metrics_panel.py
M	src/face_ladder_sizes.py
A	src/face_vgg_vm.py
M	src/iris_ladder_metrics.py
M	src/iris_ladder_metrics_figures.py
A	src/iris_ladder_metrics_p3.py
A	src/iris_ladder_metrics_window.py
A	src/ixi_prep.py
A	tests/test_cross_modality.py
A	tests/test_donor_ladder_brain_ixi.py
M	tests/test_donor_ladder_brain_metrics_size.py
M	tests/test_donor_ladder_brain_metrics_t23.py
A	tests/test_donor_ladder_brain_metrics_t23_v5.py
M	tests/test_donor_ladder_brain_seg_gcp.py
M	tests/test_face_ladder_metrics_panel.py
M	tests/test_face_ladder_sizes.py
A	tests/test_face_vgg_vm.py
M	tests/test_iris_ladder_metrics.py
A	tests/test_ixi_prep.py
M	tests/test_no_box_windows.py
```

Cleanup completed after remote verification of b001d9c. The two former folders were moved to Windows Recycle Bin; one canonical Git clone and the hash-verified ignored runtime remain. See LOCAL_CLEANUP.json. Superseded renderings remain recoverable from the preservation commit.

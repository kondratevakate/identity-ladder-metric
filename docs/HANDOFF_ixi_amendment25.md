# HANDOFF: IXI Amendment 25 (paused 14 Nov 2026; completed 15 Nov 2026, see the protocol's Amendment 25 result)

Kate paused the work ("останови все пока что я потом продолжу"). Nothing is committed; nothing is running.
Worktree `C:\Projects\02_academia\studies\_worktrees\identity-ladder-ixi`, branch `cvpr2027-ixi-complete`
(from `cvpr2027-paper` at 90897a4). Pre-QC, affine registration only; the 2 Nov registration gate still stands,
Kate asked for this run explicitly.

## Goal

Close the IXI gaps in the CVPR paper: LPIPS AlexNet/VGG, persistence, FID, FWD for IXI, and an IXI row of Gaussian
error maps in the all-modalities atlas, so no caption says "IXI was not run" / "IXI has no map".

## Done (uncommitted)

- `docs/PROTOCOL_donor_ladder_brain.md`: Amendment 25 (cells = topology subset S of 24C; whole 16x16 slices as
  set-level samples; self-anchored R added before the full run; maps with donor-masked means).
- `src/donor_ladder_brain_metrics_t23.py`: hook `ixi_gen` (`extra_ixi_gen`, `whole_slices`); `_lpips` calls
  `torchvision_guard.ensure()`.
- `src/torchvision_guard.py`: Windows Application Control blocks `torchvision/_C.pyd` (WinError 4551) since
  ~13-14 Nov; the guard re-imports torchvision without its meta registrations. Checked: LPIPS VGG / AlexNet and
  persistence reproduce 64 stored QTIM rows (box 32) to 6e-17 / 4e-8 / 2e-13.
- `src/donor_ladder_brain_ixi_gen.py`: scoring, `--set-level`, `--summary-only`, `--qc`.
- `src/donor_ladder_brain_maps_ixi.py` (`fine`, `grid`) and `src/donor_ladder_brain_maps_ixi_figure.py`.
- `src/cross_modality/generative_block.py`: reads `results/donor_ladder_brain_ixi_gen.json` (self-anchored R for
  LPIPS/persistence, set level without other side and its shrunk control); `summary_figure.py` footnote.
- `paper/cvpr2027/assemble_atlas.py`: appends IXI as a ninth row (rows 0-7 keep coordinates for the infographic).
- `tests/test_ixi_amendment25.py`: 8 pass, 2 skip until results exist. Full `pytest tests/` not yet run.

## Smoke results (private: `C:\Projects\data\projects\identity-ladder-metric\`)

- Metrics, 3 targets, 6 cells, 4.7 min on 2 workers (`ixi-gen-metrics-20261114-v1\qc_smoke3.png`); Inception
  2.6 samples/s with 2 threads. Kate's verdict on the QC sheets: pass ("Да, запускай").
- Maps, first 12 scans (`ixi-maps-20261114-v1\smoke12\qc_maps.png`): fine 4 min, grid 2 min; mirror 1.47 x mean,
  stranger 1.83 (= 1 + (n-2)/n by construction); look-alike-16 and cohort empty in the smoke (too few donors).

## State at the pause

- Full metrics run was started and stopped before the first position finished: `parts/` is empty.
- Maps cache: `volumes.npy` / `masks.npy` complete (`volumes.done`); grid cache stopped at 61/100 scans and has no
  `grid.done`, so it is rebuilt from scratch on resume. The folder holds 18 GB (C: had 74 GB free); delete
  `ixi-maps-20261114-v1\cache` if space is needed, it is rebuilt.

## Resume (from the worktree, Python 3.11 global)

```
python -I src/donor_ladder_brain_ixi_gen.py --workers 2              # ~2-3 h, resumable per position, ETA printed
python -I src/donor_ladder_brain_ixi_gen.py --workers 4 --set-level  # Inception on ~17k samples, ~1 h
python -I src/donor_ladder_brain_ixi_gen.py --workers 1 --summary-only
python -I src/donor_ladder_brain_maps_ixi.py fine                    # ~20-30 min
python -I src/donor_ladder_brain_maps_ixi.py grid                    # cache rebuild + ~30 min
python -I src/donor_ladder_brain_maps_ixi_figure.py
```
Then README `src/cross_modality/README.md` steps 10, 11, 11b-11f, `python paper/cvpr2027/assemble_atlas.py`,
look at the rendered figures, write the Amendment 25 result in the protocol, `python -m pytest tests/ -q`,
commit own paths only (no push).

Caption text to change afterwards (in the review copy `identity-ladder-cvpr2027-review`, `sec/4_results.tex`):
line 53 "IXI was not run for LPIPS, persistence, FID and FWD" and line 191 "IXI has no map yet".

Follow-up, not done here: the QTIM maps average zero-filled z volumes outside donor masks (missing-is-never-zero);
IXI uses donor-masked means.

# Not run and limits


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_4bfafd8334a6", "created_at": "2026-11-10T10:06:57+00:00", "title": "Not run and limits"}
-->
Not run, with the reason (from the result files):

- Latent metrics (pullback, Fisher-Rao, LDDMM, plausibility): need a generative model with an inversion encoder per
  modality; options and prices are with Kate; nothing run.
- Brain Betti matching 3D: not available (3D and 2D libraries disagreed, brain Amendment 24 B); 2.5D used instead.
- IXI LPIPS, persistence, FID, FWD: not run (IXI scores tier-1 metrics and Betti 2.5D only).
- Brain FID / FWD: computed but excluded from Fig. 2 (repeat set 138 of 1326 windows; FID biased upward for small sets).
- Fundus window-size sweep: not run.

Limits: brain, fundus and iris results precede the registration and mask QC gates (provisional); the set-level rule
and the brain exclusion were fixed after seeing the brain value; the cross-modality analysis is exploratory.

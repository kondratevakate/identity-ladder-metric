# Claim-first outline, 8 November 2026

**Scope update:** this outline records the earlier evidence-bounded manuscript structure. The latest user-directed target is the cross-anatomy study in [`research_brief.md`](research_brief.md): identity, symmetry, variability, distance comparison and inpainting, led by a spatial difference atlas. The expanded manuscript now includes bilateral methods, saved representation comparisons and a proposed benchmark-extension section. Retain the evidence limits below until the new experiments exist.

## Central question

When a similarity metric assigns a high score to a reconstruction, what evidence supports interpreting that score as preservation of the depicted individual?

## Proposed contribution

- A structured audit combining repeat capture, nuisance interventions, later visits, bilateral/family controls where available, unrelated references, population means and wrong-detail controls.
- Conditional percentile calibration with explicit pair weighting and an abstention rule when nuisance-matched references do not exist.
- Separate reporting of distortion, discrimination, residual attenuation and target alignment. No validated scalar fusion is claimed.
- Existing synthetic counterexamples and exploratory ocular pilots that reveal why neither calibration nor a plausible image suffices.

## Abstract and introduction

Lead with two opposite errors: an identity-free average can be ranked highly, and the same simulated eye under dilation can be ranked poorly. Explain the proposed audit and the limits of current evidence. Generalize the question to reconstruction evaluation without claiming validation on natural-image reconstruction models.

## Related work

Distinguish perceptual similarity (Blau and Michaeli; LPIPS), biometric matching (Daugman), and covariate-adjusted marker placement/ROC analysis (Pepe and Longton; Janes et al.). Use Metrics Reloaded for metric-task alignment. Cite GRAPE and RETFound for the ocular pilot. This is targeted positioning, not a systematic literature review or a priority claim.

## Method

Define the intended identity unit (person, eye, anatomy) and the nuisance variable. Define calibration with half-credit for ties and its conditional/weighted AUC interpretation. Explain that rank calibration preserves ordering within a fixed reference stratum and cannot correct the underlying representation. Treat the ladder as a comparison graph: repeat versus stranger is interpretable, but bilateral versus longitudinal and scanner versus aging need not have universal orders.

Give the population-mean squared-error counterexample with equal donor variance assumptions. Distinguish an estimated finite-sample mean from the population expectation. Define residual amplitude and target projection in a fixed Euclidean representation. The oracle shrinkage path checks implementation by construction; it is not external validation. Do not extend voxel-accuracy optimality to Dice.

## Experiments and results

Use the saved population-null, pupil and label-robustness experiments. Transcribe aggregate GRAPE and UTIRIS results from JSON, including independent-unit counts and confidence intervals where available. Keep paired versus unpaired comparisons distinct. Do not use the old MRI middle-rung ordering as evidence after the audit exposed protocol confounding.

Figure plan: a new audit schematic; saved synthetic shrinkage and pupil examples; an aggregate score plot generated from JSON. No participant images or fabricated experimental illustrations.

## Discussion and conclusion

State what the audit establishes: different metrics can pass some checks and fail others. State what remains unresolved: independent biological identity validation, nuisance-normalized representations, repeat-based residual resolution, contemporary perceptual/foundation baselines and reconstructed-output evaluation.

## References and disclosure

Verify citations with the local bib-verify tool. Use anonymous author information and an accurate disclosure of AI assistance. Resolve ethics, dataset terms, related ISBI status and author approval before any submission.

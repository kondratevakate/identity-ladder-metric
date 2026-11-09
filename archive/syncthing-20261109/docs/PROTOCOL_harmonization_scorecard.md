# BACKLOG (separate paper): ladder-based harmonization scorecard (GRAPE cameras, 2 Nov 2026)

Status: BACKLOG, NOT RUN. On 2 Nov 2026 Kate stopped this before any computation: it belongs to a different paper
and is only recorded here. No scorecard value exists. `src/harmonization_scorecard.py` is an untested draft that has
never been executed. Before it is used: re-read this design, add tests, and record a new dated decision to run.

## Idea

Harmonization should remove camera or scanner effects and keep each person's individual pattern. The donor ladder
separates the two, and theory gives a target value for every check, so a failing check shows what broke.

## Data

- GRAPE colour fundus images: three cameras, the same eyes seen at several visits, sometimes on another camera.
- Images come from the donor-ladder cache (`donor-ladder-fundus-20261102-v1/GRAPE_canonical.npz`): disc-fovea
  frame, disc-right orientation, green channel, 256 grid, each image z-scored inside its field of view.
- Exclusions are those of the ladder: landmarks missing, or laterality in conflict with the disc side.
- The 32 random patches are those of the ladder (seed 20261102, support 0.90).

## Harmonization variants

Device maps are computed over all included images of a camera, inside the field of view.

| Variant | What it does |
|---|---|
| `none` | the per-image z-score only (the ladder's input) |
| `device_shift` | subtract the camera's mean image, add the pooled mean image |
| `device_location_scale` | as `device_shift`, then divide by the camera's pixelwise SD and multiply by the pooled SD (a ComBat-style location-scale step, without empirical Bayes) |
| `content_swap` (negative control) | each image keeps a share 0.7 of its own deviation from the camera mean and takes the rest from another person on the same camera: x' = m_d + 0.7(x - m_d) + sqrt(1 - 0.49)(x_other - m_d). Marginal statistics stay the same, the identity is diluted. |
| `blur` (control) | Gaussian smoothing, sigma 3 pixels, after `device_location_scale` |

## Checks

Every error is divided by the population-mean error V of the same variant, computed per target.

| # | Check | Measure | Target |
|---|---|---|---|
| 1 | Camera effect | R2 of the camera-mean fill: 1 - error(mean of other people on the target's camera) / V | 0 |
| 2 | Exchangeability across cameras | expected error of a stranger from another camera / error of the mean of that camera's pool | 2/(1 + 1/N), as within a camera |
| 3 | Equal variance | V by camera, as a max/min ratio | 1 |
| 4 | Identity kept | r = 1 - (error/V)/2 for the same eye at another visit, split into same-camera and other-camera pairs | other-camera r close to same-camera r |
| 5 | Biology kept | r of the fellow eye (reflected) | not lower than under `none` |

Strangers are always other people, excluding the target's person, and pools are the same side as the target.
Intervals: bootstrap over persons (2000 resamples).

## Caveats

- The target is included in its own camera's mean map; with more than 100 images per camera the leak is small,
  and it is stated.
- Pre-QC: the landmarks are automatic.
- This is a demonstration of the scorecard. It is not a benchmark of published harmonization methods.

## Outputs

- `results/harmonization_scorecard.json`: aggregates only.
- Rows stay in the private data root.

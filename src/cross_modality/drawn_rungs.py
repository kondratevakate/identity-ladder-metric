"""Rung thumbnails on a DRAWN face (no real person), built with src/ladder_faces.py (imported, not modified).
Like the metric fills: the target image keeps its context and only the box over one brow and eye is replaced by the
rung's content. Look-alike = closest of 25 drawn strangers by RMSE in the box; best look-alike = closest of the
80-person pool; mean of 2 = average of 2 pool people; wrong detail = a stranger's box; noise = white noise matched to
the box mean and SD; shrunk repeat = repeat box pulled halfway to the population mean. Sex-age mean is drawn as the mean of the 16 pool people closest over the whole face (illustration)."""
import sys
import numpy as np
sys.path.insert(0, "C:/Projects/02_academia/studies/_worktrees/identity-ladder-cvpr2027/src")
import ladder_faces as LF

OUT = sys.argv[1]
rng = np.random.default_rng(20261107)  # differs from LF.SEED, so no drawn stranger is the person
top, cols = LF.build()
C = {c[0]: c[3] for c in cols}
mean = C["population\nmean"]
h, w = top.shape[:2]
x0, x1, y0, y1 = LF.BOX
ys, xs = slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w))
strangers = [LF.noisy(LF.render(LF.traits(LF.draw_genome(rng)), {"shirt": LF.SHIRTS[6]}), rng) for _ in range(25)]
pool = [LF.render(LF.traits(LF.draw_genome(rng)), {"shirt": LF.SHIRTS[2]}) for _ in range(80)]
err = lambda im: LF.rmse(top, im)
box = lambda im: im[ys, xs]
src = {"repeat": C["L0"], "other_side": C["bilateral"], "cotwin": C["MZ"], "sibling": C["DZ / sibling"],
       "relative": C["DZ / sibling"],   # the pooled sibling-or-twin column of the tiles
       "stranger": C["L4"], "dataset_mean": mean, "lookalike": min(strangers, key=err), "lookalike_oracle": min(pool, key=err),
       "mean_k2": (pool[0] + pool[1]) / 2, "wrong_detail": strangers[3],
       # matched mean, illustration only: drawn faces have no sex or age, so the 16 pool people closest to the person
       # over the whole face stand in for "same sex and age"
       "cohort_mean": np.mean(sorted(pool, key=lambda im: float(np.mean((im - top) ** 2)))[:16], axis=0),
       # mean of the 16 pool people closest inside the box only (look-alike search is local), unlike the sex-age mean
       "lookalike16": np.mean(sorted(pool, key=lambda im: float(np.mean((box(im) - box(top)) ** 2)))[:16], axis=0)}
th = {}
def fill(content):
    im = top.copy()
    im[ys, xs] = content
    return im
th["target"] = top
for k, im in src.items():
    th[k] = fill(box(im))
b = box(top)
th["noise"] = fill(np.clip(rng.normal(b.mean(), b.std(), b.shape), 0, 1))
th["shrink_ref"] = fill(0.5 * box(C["L0"]) + 0.5 * box(mean))
# scanner-site strangers (IXI only), illustration: another stranger as is, and one with a contrast change standing in
# for a different scanner
th["stranger_same_centre"] = fill(box(strangers[5]))
th["stranger_other_centre"] = fill(np.clip(0.5 + 1.35 * (box(strangers[9]) - 0.5) + 0.06, 0, 1))
# crop to the eye region around the box, draw the box outline in red
pad_y, pad_x = int(0.06 * h), int(0.07 * w)
cy, cx = slice(int(y0 * h) - pad_y, int(y1 * h) + pad_y), slice(int(x0 * w) - pad_x, int(x1 * w) + pad_x)
out = {}
for k, im in th.items():
    im = im[..., :3].copy()
    a0, a1, b0, b1 = int(y0 * h), int(y1 * h) - 1, int(x0 * w), int(x1 * w) - 1
    for yy in (a0, a1):
        im[yy, b0:b1 + 1] = (0.85, 0.1, 0.1)
    for xx in (b0, b1):
        im[a0:a1 + 1, xx] = (0.85, 0.1, 0.1)
    out[k] = (np.clip(im[cy, cx], 0, 1) * 255).astype(np.uint8)
np.savez_compressed(OUT + "/drawn_face_thumbs.npz", **out)
from PIL import Image
Image.fromarray(np.concatenate([np.pad(v, ((3, 3), (3, 3), (0, 0)), constant_values=255) for v in out.values()], 1)).resize(
    (len(out) * 4 * (out["target"].shape[1] + 6), 4 * (out["target"].shape[0] + 6)), Image.NEAREST).save(OUT + "/drawn_face_thumbs_check.png")
print(list(out), out["target"].shape, {k: round(err(v), 3) for k, v in src.items()})

"""Amendment 24 B agreement check, laptop side: 2D library parts on central 16x16 slices of 2 QTIM targets
(target cube vs its own other side and vs a stranger), whole-mask cubes only. Writes slices + 2D parts to npz."""
import os, sys
import numpy as np
SRC = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, SRC)
sys.path.insert(0, r"C:\Projects\data\projects\identity-ladder-metric\models-20261107\betti-matching-2d")
import donor_ladder_brain_bilateral as BL
from iris_ladder_metrics_extra import betti_parts
vol, mask = np.load(BL.region_paths(0)[0], mmap_mode="r"), np.load(BL.region_paths(0)[1], mmap_mode="r")
c = slice(BL.W + BL.SHELL, BL.W + BL.SHELL + 16)
def unit(z):
    return (np.clip(z, -3, 3) + 3) / 6
A, Bs, P = [], [], []
k = 0
for i in range(vol.shape[0]):
    if len(A) >= 4:
        break
    for j in (i + 37,):
        x, y = vol[i, 0][c, c, c], vol[j, 0][c, c, c]
        if mask[i, 0][c, c, c].all() and mask[j, 0][c, c, c].all():
            for ax in (2,):
                a, b = unit(np.take(x, 8, ax)).astype(np.float64), unit(np.take(y, 8, ax)).astype(np.float64)
                A.append(a); Bs.append(b); P.append(betti_parts(a, b))
            xm = vol[i, 1][c, c, c]
            if mask[i, 1][c, c, c].all():
                a, b = unit(x[:, :, 8]).astype(np.float64), unit(xm[:, :, 8]).astype(np.float64)
                A.append(a); Bs.append(b); P.append(betti_parts(a, b))
np.savez(OUT, a=np.array(A), b=np.array(Bs), parts2d=np.array(P))
print(len(A), np.array(P).round(4).tolist())

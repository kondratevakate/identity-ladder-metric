import sys, json
import numpy as np
sys.path.insert(0, "/home/kondr/Betti-Matching-3D/build"); sys.path.insert(0, "/home/kondr")
import betti_matching as bm
from parts3d import parts
z = np.load("/home/kondr/check2d.npz")
out = []
for a, b, p2 in zip(z["a"], z["b"], z["parts2d"]):
    r = {"2d_lib": p2.tolist()}
    for name, (x, y) in {"vol_16x16x1": (a[:, :, None], b[:, :, None]), "array_2d": (a, b)}.items():
        x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        r[name + "_essential"] = parts(x, y, bm, dims=(0, 1), essential=True)
        r[name + "_pairs_only"] = parts(x, y, bm, dims=(0, 1), essential=False)
    out.append(r)
print(json.dumps(out, indent=0))

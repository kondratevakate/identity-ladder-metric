"""Pretrained BraTS inpainting models on the QTIM identity ladder, 16 mm cubes (f = 1/8; PROTOCOL_donor_ladder_general.md
Amendment 10, geometry of brain Amendment 10 / 11: 32 random cubes, both sides). Pre-QC (affine registration only).

The model sees the skull-stripped scan of the target with ONE 16 mm cube set to zero plus the cube mask, in the
240 x 240 x 155 BraTS grid, and returns the cube. No interpolation is applied to the scan or to the fill: the 1 mm MNI
grid is moved into the BraTS grid by axis flips, zero padding and a 155-slice z window (see `gcp/model_fills/
brain_container.py`). The fill is z-scored with the target's own in-mask mean and SD (the constants of
`donor_ladder_brain_bilateral.extract`) and scored with `donor_ladder_brain_metrics.score` on the stored region
memmaps, next to the stored rungs, as rung `model_<name>`.

Stages (rows, volumes and sheets stay in the private folder):
  export                      one array per target (scan x brain mask, target index only) and job.json for the VM
  score --model M [--smoke]   rows of model_<M> plus the gate rung (dataset_mean), which must equal the stored rows
  qc    --model M             private sheet: central slice of cube and shell for evenly spaced targets
  summ  --model M             aggregate R_self per rung (no IDs)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import donor_ladder_brain as B  # noqa: E402
import donor_ladder_brain_bilateral as BL  # noqa: E402
import donor_ladder_brain_metrics as M  # noqa: E402

RUN = "model-fills-20261116-v1"
PRIVATE = os.path.join(B.PROJ, RUN, "brain")
N_POS = 32                              # random cubes only (the two frozen boxes are not on the f = 1/8 ladder)
GATE = "dataset_mean"
TOL = 1e-9


def setup():
    import nibabel as nib
    t = B.scans()
    pos, affine, _ = B.positions(np.random.default_rng(M.SEED))
    tg = t[t.session == "ses-01"].reset_index()
    a = BL.mirror_constant(affine, nib.load(t.path.iloc[0]).shape[0])
    return t, tg, pos, affine, a


def export():
    """Per target: scan times brain mask (float32, 0 outside) and the mask, named by target row only."""
    import nibabel as nib
    t, tg, pos, affine, a = setup()
    out = os.path.join(PRIVATE, "job")
    os.makedirs(out, exist_ok=True)
    stats = []
    t0 = time.time()
    for i, f in enumerate(tg.path):
        m = nib.load(f).get_fdata(dtype=np.float32) > 0.5
        img = nib.load(f.replace("_brain.nii.gz", ".nii.gz"))
        if not np.allclose(img.affine, affine, atol=1e-3):
            raise RuntimeError("scan grid differs from the template grid")
        v = img.get_fdata(dtype=np.float32)
        mean, std = float(v[m].mean()), float(v[m].std())       # constants of BL.extract (z = (v - mean) / (std + 1e-6))
        stats.append({"target": i, "mean": mean, "std": std, "min_in_mask": float(v[m].min())})
        p = os.path.join(out, f"tgt_{i:03d}.npz")
        if not os.path.exists(p):
            np.savez_compressed(p + ".tmp.npz", vol=np.where(m, v, 0).astype(np.float32), mask=m)
            os.replace(p + ".tmp.npz", p)
        if i % 20 == 0:
            print(f"export {i + 1}/{len(tg)}, {time.time() - t0:.0f} s", flush=True)
    job = {"shape": list(v.shape), "mirror_constant": int(a), "size": B.PATCH,
           "starts": [[int(c) for c in pos[pi][1]] for pi in range(N_POS)], "stats": stats,
           "note": "arrays are RAS+ on the 1 mm MNI grid; frame 1 = mirrored x (index a - i)"}
    json.dump(job, open(os.path.join(out, "job.json"), "w"), indent=1)
    mb = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 2 ** 20
    print(f"{out}: {len(tg)} targets, {mb:.0f} MB", flush=True)


def load_fills(model):
    """Model cubes z-scored with the target's constants: {target: array [32, 2, 16, 16, 16]} (NaN = not run)."""
    d = os.path.join(PRIVATE, model)
    job = json.load(open(os.path.join(PRIVATE, "job", "job.json")))
    st = {s["target"]: s for s in job["stats"]}
    out = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("fill_") and f.endswith(".npz"):
            i = int(f[5:8])
            raw = np.load(os.path.join(d, f))["fill"].astype(np.float64)
            out[i] = ((raw - st[i]["mean"]) / (st[i]["std"] + 1e-6)).astype(np.float32)
    return out


def score_position(pi, t, tg, pos, a, fills, name):
    """Rows of the model fill and of the gate rung for one position, both sides. The target cube, its validity, the
    dataset mean, the cohort mean and the repeat tolerance are computed as in `M.run_position_at` (same statements)."""
    region, start, size = pos[pi]
    pv, pm = BL.region_paths(pi)
    vol, mask = np.load(pv, mmap_mode="r"), np.load(pm, mmap_mode="r")
    drawn = BL.side_of(start, size, M.AFFINE_HOLDER["affine"])
    shell, W = BL.SHELL, BL.W
    fam, sex, age = tg.family.to_numpy(), tg.sex.to_numpy(), tg.age.to_numpy()
    ix = tg["index"].to_numpy()
    row_of = {(p, s): k for k, (p, s) in enumerate(zip(t["person"], t["session"]))}
    cube_sl = tuple(slice(W + shell, W + shell + s) for s in size)
    rows = []
    for side in BL.SIDES:
        fo = BL.own_frame(side, drawn)
        cube_all = np.asarray(vol[:, fo][(slice(None),) + cube_sl])
        mk_all = np.asarray(mask[:, fo][(slice(None),) + cube_sl]).astype(bool)
        valid_all = mk_all.reshape(len(cube_all), -1).mean(1) >= BL.VALID
        X, MK, valid = cube_all[ix], mk_all[ix], valid_all[ix]
        rep_msd = {}
        for i in np.flatnonzero(valid):
            j2 = row_of.get((tg.person.iloc[i], "ses-02"))
            if j2 is not None and valid_all[j2]:
                both = MK[i] & mk_all[j2]
                if both.sum() >= 10:
                    rep_msd.setdefault(fam[i], []).append(float(np.mean((X[i][both] - cube_all[j2][both]) ** 2)))
        for i in np.flatnonzero(valid):
            if int(i) not in fills:
                continue
            others = np.flatnonzero((fam != fam[i]) & valid)
            pairs = [v for f_, vs in rep_msd.items() if f_ != fam[i] for v in vs]
            sigma = float(np.sqrt(np.mean(pairs))) if len(pairs) >= M.MIN_REP_PAIRS else None
            if others.size < 10:
                continue
            xi, ei = X[i], MK[i]
            mu_d, mu_d_ok = M.masked_mean(X[others], MK[others])
            cpool = others[(sex[others] == sex[i]) & (np.abs(age[others] - age[i]) <= 2)]
            mu_c = M.masked_mean(X[cpool], MK[cpool])[0] if cpool.size else None
            f = fills[int(i)][pi, fo]
            for nm, fill, ok in ((name, f, np.isfinite(f)), (GATE, mu_d, mu_d_ok)):
                if nm == name and not ok.any():
                    continue                                       # cube not run for this target: missing, never zero
                s = M.score(xi, fill, ei & ok, mu_d, mu_c, sigma)
                if s is None:
                    continue
                rows.append({"target": int(i), "patch": pi, "side": side, "drawn": side == drawn, "fill": nm, **s})
    return rows


def score(model):
    t, tg, pos, affine, a = setup()
    M._init(affine)
    fills = load_fills(model)
    name = f"model_{model}"
    t0 = time.time()
    rows = []
    for pi in range(N_POS):
        rows += score_position(pi, t[["person", "session"]].to_dict("list"), tg, pos, a, fills, name)
        print(f"position {pi + 1}/{N_POS}: rows {len(rows)}, {time.time() - t0:.0f} s", flush=True)
    r = pd.DataFrame(rows)
    ref = pd.read_csv(os.path.join(M.PRIVATE, "rows.csv.gz"))
    ref = ref[ref.patch < N_POS]
    keys = ["target", "patch", "side", "fill"]
    metrics = [c for c in M.METRICS if c in r.columns]
    g = r[r.fill == GATE].merge(ref[ref.fill == GATE], on=keys, suffixes=("", "_ref"))
    n_ref = int((ref[ref.fill == GATE].target.isin(set(r.target))).sum())
    diffs = {c: float(np.nanmax(np.abs(g[c] - g[c + "_ref"]))) for c in metrics}
    same_nan = all(bool((g[c].isna() == g[c + "_ref"].isna()).all()) for c in metrics)
    gate = {"model": model, "gate_rung": GATE, "n_gate_rows": int(len(g)), "n_stored_gate_rows_same_targets": n_ref,
            "max_abs_diff": diffs, "same_missing_pattern": same_nan,
            "passed": bool(len(g) == n_ref and len(g) > 0 and same_nan and max(diffs.values()) <= TOL),
            "n_targets": int(r.target.nunique()), "n_model_rows": int((r.fill == name).sum()),
            "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(gate, open(os.path.join(PRIVATE, model, "gate_score.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in gate.items() if k != "max_abs_diff"}), flush=True)
    if not gate["passed"]:
        raise SystemExit("stored dataset_mean rows not reproduced; model rows not kept")
    own = r[r.fill == name]
    cells = own[["target", "patch", "side"]].drop_duplicates()
    full = pd.concat([ref.merge(cells, on=["target", "patch", "side"]), own], ignore_index=True)
    full.to_csv(os.path.join(PRIVATE, model, "rows.csv.gz"), index=False, float_format="%.17g")


STR = [f"str{j:02d}" for j in range(16)]
SIM = {"ssim", "pearson", "acc_dataset", "acc_cohort", "lddt", "lddt_rep", "tm_like", "gdt_like"}
SHOW = ["repeat", "mirror", "lookalike", "dataset_mean", "shrink_repeat", "wrong_detail", "noise", "stranger"]


def r_self(r, fill, m, cell):
    sgn, d_self = (-1.0, -1.0) if m in SIM else (1.0, 0.0)
    a = r[r.fill == fill].groupby(cell)[m].mean() * sgn
    s = r[r.fill.isin(STR)].groupby(cell)[m].mean() * sgn
    j = pd.concat([a.rename("a"), s.rename("s")], axis=1).dropna()
    if j.empty:
        return None, 0
    return round(float((j.a.mean() - d_self) / (j.s.mean() - d_self)), 3), int(len(j))


def summ(model):
    r = pd.read_csv(os.path.join(PRIVATE, model, "rows.csv.gz"))
    name = f"model_{model}"
    out = {"model": model, "f": "1/8 (16 mm cube)", "n_targets": int(r[r.fill == name].target.nunique()),
           "scale": "R_self: target 0, mean of 16 random people 1, lower = closer", "R": {}}
    for m in ("mse", "ssim", "tm_like", "gdt_like", "lddt"):
        out["R"][m] = {f: r_self(r, f, m, ["target", "patch", "side"]) for f in [name] + SHOW}
    json.dump(out, open(os.path.join(PRIVATE, model, "summary.json"), "w"), indent=1)
    for m, row in out["R"].items():
        print(m, " ".join(f"{k}={v[0]}({v[1]})" for k, v in row.items()))


def qc(model, n_targets=8, positions=(0, 12, 24)):      # cubes of the budgeted run (every fourth)
    """Private sheet: central axial slice of the 16 mm cube inside its shell: target, model fill, second scan,
    dataset average, squared error of the model fill. Evenly spaced targets that the model ran on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, tg, pos, affine, a = setup()
    M._init(affine)
    fills = load_fills(model)
    have = sorted(fills)
    sel = [have[k] for k in np.unique(np.linspace(0, len(have) - 1, n_targets).round().astype(int))]
    ix = tg["index"].to_numpy()
    fam = tg.family.to_numpy()
    row_of = {(p, s): k for k, (p, s) in enumerate(zip(t.person, t.session))}
    shell, W = BL.SHELL, BL.W
    cols = ("target", f"model: {model}", "squared error", "same person,\nsecond scan", "dataset average")
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#9ecae1")
    fig, ax = plt.subplots(len(sel), len(cols) * len(positions),
                           figsize=(1.2 * len(cols) * len(positions), 1.3 * len(sel) + 0.6), squeeze=False)
    for c, pi in enumerate(positions):
        _, start, size = pos[pi]
        vol, mask = np.load(BL.region_paths(pi)[0], mmap_mode="r"), np.load(BL.region_paths(pi)[1], mmap_mode="r")
        win_sl = tuple(slice(W, W + s + 2 * shell) for s in size)
        cube_sl = tuple(slice(shell, shell + s) for s in size)
        win_all = np.asarray(vol[:, 0][(slice(None),) + win_sl])
        wm_all = np.asarray(mask[:, 0][(slice(None),) + win_sl]).astype(bool)
        X = win_all[ix][(slice(None),) + cube_sl]
        MK = wm_all[ix][(slice(None),) + cube_sl]
        valid = MK.reshape(len(X), -1).mean(1) >= BL.VALID
        z = shell + size[2] // 2
        for i, ti in enumerate(sel):
            others = np.flatnonzero((fam != fam[ti]) & valid)
            mu_d, mu_ok = M.masked_mean(X[others], MK[others])
            j2 = row_of.get((tg.person.iloc[ti], "ses-02"))
            f = fills[ti][pi, 0]
            def with_fill(fill, ok):
                w2, k2 = win_all[ix[ti]].copy(), wm_all[ix[ti]].copy()
                w2[cube_sl] = np.nan_to_num(fill)
                k2[cube_sl] = ok & MK[ti]
                return w2, k2, "gray"
            err = np.full(win_all[ix[ti]].shape, np.nan, np.float32)
            err[cube_sl] = np.where(np.isfinite(f) & MK[ti], (np.nan_to_num(f) - X[ti]) ** 2, np.nan)
            panels = [(win_all[ix[ti]], wm_all[ix[ti]], "gray"), with_fill(f, np.isfinite(f)),
                      (err, np.isfinite(err), "magma"),
                      with_fill(win_all[j2][cube_sl], wm_all[j2][cube_sl]) if j2 is not None else None,
                      with_fill(mu_d, mu_ok)]
            for q, pnl in enumerate(panels):
                a_ = ax[i, c * len(cols) + q]
                a_.set_xticks([])
                a_.set_yticks([])
                if i == 0:
                    a_.set_title(f"{cols[q]}\ncube {pi}", fontsize=6)
                if pnl is None:
                    a_.set_facecolor("#eeeeee")
                    continue
                img, ok, cm = pnl
                kw = dict(vmin=0, vmax=1) if cm == "magma" else dict(vmin=-2.5, vmax=2.5)
                a_.imshow(np.where(ok[:, :, z], img[:, :, z], np.nan).T, cmap=cmap if cm == "gray" else cm,
                          origin="lower", interpolation="nearest", **kw)
                a_.add_patch(plt.Rectangle((shell - .5, shell - .5), size[0], size[1], fill=False, ec="#d33", lw=0.6))
    fig.suptitle("PRIVATE QC QTIM, 16 mm cubes (f = 1/8), drawn side, central axial slice; red = hidden cube, outside "
                 "= 4 mm shell; light blue = outside the brain mask or not run; grey panel = no second scan", fontsize=7)
    fig.tight_layout()
    out = os.path.join(PRIVATE, model, f"qc_{model}_private.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("export", "score", "qc", "summ"))
    ap.add_argument("--model")
    a = ap.parse_args()
    if a.stage == "export":
        return export()
    if not a.model:
        raise SystemExit("--model is required")
    {"score": score, "qc": qc, "summ": summ}[a.stage](a.model)


if __name__ == "__main__":
    main()

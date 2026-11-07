"""Animated ladder maps of all modalities: one random target per row and frame, 100 frames. Pre-QC. Display only.

Same layout as `ladder_maps_all_figure.py` (faces CelebA and IMDb-WIKI v6f, brain QTIM axial z = 10 mm, fundus GRAPE
P8 and PAPILA P9), but every frame shows single comparisons: column 1 is the target itself, every other column the
local error of that target's own donor, divided by the row's typical error of the dataset mean (the mean over the 100
drawn targets of their own dataset-mean error maps), so 1 = a typical dataset-mean error and 2 = a typical stranger.
Under each panel: the target's whole-frame squared error of that donor / of the dataset mean on the same pixels.

Donors follow each modality's map rules (read-only use of their helpers): faces as `donor_ladder_face.maps` (half
"all"; one stranger instead of 8), fundus as `donor_ladder_2d_facefmt.run_dataset` (one stranger instead of 8), brain
as `donor_ladder_brain_maps.run` but with a masked 3D Gaussian window (sd = 16 mm / sqrt(12)) on a slab around
z = 10 mm, and the first different-age sibling; brain look-alikes are a selection on a coarse grid and are not
computed per target. Missing is never zero: every local error is a masked mean over pixels valid in both images.
Frames show real people: the output stays in the private folder, never in git.
Usage: python src/ladder_maps_all_animation.py   (LADDER_EXTERNAL as for donor_ladder_2d_facefmt)
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import ladder_maps_all_figure as LF  # noqa: E402

N_FRAMES = int(os.environ.get("N_FRAMES", "100"))   # smoke runs use fewer
SEED = 20261107
OUTDIR = os.path.join(LF.PROJ, "ladder-maps-animation-20261107-v1")
FACE_DIR = os.path.join(LF.PROJ, "face-ladder-20261105-v1")
COLS = [("target", "Target\n(this person)")] + LF.COLS[1:7] + [("V", "Dataset mean")]


def ratio_number(xt, f, mean, m):
    num, den = (((xt - f) ** 2) * m).sum(), (((xt - mean) ** 2) * m).sum()
    return float(num / den) if den > 0 and m.sum() >= 50 else None


# ---------------------------------------------------------------------------------------------- faces

def face_frames():
    spec = importlib.util.spec_from_file_location("donor_ladder_face", os.path.join(LF.MAIN, "src",
                                                                                     "donor_ladder_face.py"))
    Fc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Fc)
    Fc.SUF = LF.FACE_VERSION                          # explicit version (no silent default)
    imgs = json.load(open(os.path.join(FACE_DIR, "images_v6.json"), encoding="utf-8"))
    lz = np.load(os.path.join(FACE_DIR, "landmarks_v6.npz"))
    fz = np.load(os.path.join(FACE_DIR, f"frame{LF.FACE_VERSION}.npz"))
    idx, X, M = fz["idx"], fz["X"], fz["M"]
    PE = np.hstack([lz["P"][idx, :2], lz["E"][idx]])
    meta = [imgs[j] for j in idx]
    pose_ok = (np.abs(PE[:, 0]) < Fc.map_pose_max()) & (np.abs(PE[:, 1]) < Fc.map_pose_max())
    kin = json.load(open(Fc.KIN, encoding="utf-8"))["pairs"]
    lab = json.load(open(os.path.join(os.path.dirname(Fc.KIN), "qualifier_labels.json"), encoding="utf-8"))
    full_sib = [(p["a"], p["b"]) for p in kin if p["relation"] == "sibling"
                and not any("half" in lab.get(q, "") or "step" in lab.get(q, "") for q in p["kinship_qualifiers"])]
    kin_people = {x for q in kin for x in (q["a"], q["b"])}
    rows = []
    for src, label in (("celeba", "CelebA"), ("imdb", "IMDb-WIKI")):
        rr = [i for i, d in enumerate(meta) if d["src"] == src and pose_ok[i]]
        by_person = {}
        for i in rr:
            by_person.setdefault(meta[i]["person"], []).append(i)
        best = {p: min(ii, key=lambda i: (PE[i, :2] ** 2).sum()) for p, ii in by_person.items()}
        pool_p = sorted(p for p, ii in by_person.items() if meta[ii[0]]["role"] == "pool")
        sibs = {}
        for a, b in full_sib:
            if a in by_person and b in by_person:
                sibs.setdefault(a, set()).add(b)
                sibs.setdefault(b, set()).add(a)
        targets = sorted(p for p, ii in by_person.items() if meta[ii[0]]["role"] == "target" and len(ii) >= 2
                         and (src == "celeba" or p in kin_people))
        pi = np.array([best[p] for p in pool_p])
        Xp, Mp = X[pi].astype(np.float32), M[pi]
        mean, cov = Fc.masked_mean(Xp, Mp)
        valid = (fz["A"][pi].mean(0) >= 0.9) & (cov >= 0.5 * len(pi))

        def cell(i, src=src):
            d = meta[i]
            return (d["male"], d["young"]) if src == "celeba" else (d["male"], int(d["age"] // 5))
        cells = {}
        for i in pi:
            cells.setdefault(cell(i), []).append(i)
        cell_mean = {c: Fc.masked_mean(X[ii].astype(np.float32), M[ii])[0] for c, ii in cells.items() if len(ii) >= 10}
        rng = np.random.default_rng([SEED, len(src)])
        chosen = []
        for p in rng.permutation(targets):
            t = best[p]
            others = [i for i in by_person[p] if i != t and Fc._corr(X[i], X[t], M[i] & M[t]) <= Fc.DUP_R]
            if others:
                chosen.append((p, t, min(others, key=lambda i: Fc._pose_expr_dist(PE[i], PE[t]))))
            if len(chosen) == N_FRAMES:
                break
        T = np.array([t for _, t, _ in chosen])
        LFm = Fc.lookalike_fill(X[T].astype(np.float32), Xp, valid, Mt=M[T], Mp=Mp)
        midline = np.abs(np.arange(Fc.SIZE) - (Fc.SIZE - 1) / 2) >= 2.5
        frames = []
        for k, (p, t, rep) in enumerate(chosen):
            xt, mt = X[t].astype(np.float32), M[t]
            vt = mt & valid
            fills = {"repeat": (X[rep], M[rep]), "mirror": (Fc.mirror(xt), Fc.mirror(mt) & midline[None, :]),
                     "lookalike": (np.nan_to_num(LFm[k]), np.isfinite(LFm[k])), "V": (mean, valid)}
            s = rng.choice(len(pi))
            fills["stranger"] = (X[pi[s]], M[pi[s]])
            if src == "imdb" and p in sibs:
                q = sorted(sibs[p])[0]
                si = min(by_person[q], key=lambda i: Fc._pose_expr_dist(PE[i], PE[t]))
                fills["sibling"] = (X[si], M[si])
            if cell(t) in cell_mean:
                fills["sexage"] = (cell_mean[cell(t)], valid)
            frames.append(make_frame(xt, vt, mean, fills, lambda a, b, m: Fc.local_error(a, b, m)[0]))
        rows.append(finish_row("Faces", label, frames, valid, "upper"))
    return rows


# ---------------------------------------------------------------------------------------------- fundus

def fundus_frames():
    import donor_ladder_2d as D
    import donor_ladder_2d_facefmt as FF
    import fundus_ladder_metrics as FM
    FM.setup_external()
    rows = []
    for name, arm in (("GRAPE", "p8"), ("PAPILA", "p9")):
        t, img, fov, u, _, _, fellow, other = D.setup(name, arm)
        X, M = FF.down2(img, fov)
        idx = u.idx.to_numpy()
        person, side = u.person.to_numpy(), u.side.to_numpy()
        age, sex, dev = u.age.to_numpy(float), u.sex.astype(str).to_numpy(), u.device.to_numpy()
        valid = M[idx].sum(0) >= 0.9 * len(u)
        chosen = np.random.default_rng([SEED, len(name)]).permutation(len(u))[:N_FRAMES]
        frames = []
        for i in chosen:
            S = np.flatnonzero(side == side[i])
            pool = S[person[S] != person[i]]
            xt, mt = X[idx[i]], M[idx[i]]
            mean, mok = FF.masked_mean(X[idx[pool]], M[idx[pool]])
            mok = mok > 0
            vt = mt & valid & mok
            excl = (person[pool][None, :] == person[i])
            lf = FF.lookalike_fill(xt[None], mt[None], X[idx[pool]], M[idx[pool]], valid, excl)[0]
            q = np.random.default_rng([D.SEED, int(i)]).choice(pool, FF.N_STRANGER, replace=False)[0]
            fills = {"stranger": (X[idx[q]], M[idx[q]]), "lookalike": (np.nan_to_num(lf), np.isfinite(lf)),
                     "V": (mean, mok)}
            if other[i] >= 0:
                fills["repeat"] = (X[other[i]], M[other[i]])
            if fellow[i] >= 0:
                fills["mirror"] = (X[idx[fellow[i]]], M[idx[fellow[i]]])
            coh = pool[(sex[pool] == sex[i]) & (np.abs(age[pool] - age[i]) <= 5)
                       & ((dev[pool] == dev[i]) if name == "GRAPE" else True)]
            if len(coh) >= 10:
                cm, cn = FF.masked_mean(X[idx[coh]], M[idx[coh]])
                fills["sexage"] = (cm, cn >= 0.5 * len(coh))
            frames.append(make_frame(xt, vt, mean, fills, FF.local_error))
        rows.append(finish_row("Fundus", f"{name} ({arm.upper()})", frames, valid, "upper"))
    return rows


# ---------------------------------------------------------------------------------------------- brain

def brain_frames():
    import donor_ladder_brain as B
    import donor_ladder_brain_size as S
    t = B.scans()
    _, affine, _ = B.positions(np.random.default_rng(B.SEED))
    V, M = S.cache_volumes(t, affine)
    kz = int(round((10 - affine[2, 3]) / affine[2, 2]))
    sd = 16 / np.sqrt(12) / affine[0, 0]
    half = int(np.ceil(3 * sd))
    zs = slice(kz - half, kz + half + 1)
    mi = B.mirror_index(affine, V.shape[1])
    x_mm = affine[0, 0] * np.arange(V.shape[1]) + affine[0, 3]
    keep_mirror = (np.abs(x_mm) >= 8)[:, None, None]
    tg = t[t.session == "ses-01"].reset_index()
    ix, fam = tg["index"].to_numpy(), tg.family.to_numpy()
    rep = {r.person: r.Index for r in t[t.session == "ses-02"].itertuples()}
    slab = {k: np.asarray(V[k][:, :, zs], np.float32) for k in list(ix) + list(rep.values())}
    mslab = {k: np.asarray(M[k][:, :, zs]) > 0.5 for k in slab}
    allv = np.stack([slab[k] for k in ix])
    allm = np.stack([mslab[k] for k in ix])
    tot_s, tot_n = (allv * allm).sum(0, dtype=np.float64), allm.sum(0)

    def gauss(a, b, m):
        m = m.astype(np.float64)
        num = ndimage.gaussian_filter(np.where(m > 0, (a - b) ** 2, 0.0) * m, sd, mode="constant")
        den = ndimage.gaussian_filter(m, sd, mode="constant")
        return np.where(den > 1e-3, num / np.maximum(den, 1e-12), np.nan)[:, :, half]
    support = allm.mean(0)[:, :, half] >= 0.95
    chosen = np.random.default_rng([SEED, 5]).permutation(len(ix))[:N_FRAMES]
    frames = []
    for i in chosen:
        others = fam != fam[i]
        vi, mt = slab[ix[i]], mslab[ix[i]]
        own = np.flatnonzero(~others)                     # leave the target's whole family out
        s_, n_ = tot_s - (allv[own] * allm[own]).sum(0), tot_n - allm[own].sum(0)
        mean, mok = np.where(n_ > 0, s_ / np.maximum(n_, 1), 0.0), n_
        mok = mok > 0
        rng = np.random.default_rng([SEED, int(i)])
        s = rng.choice(np.flatnonzero(others))
        fills = {"stranger": (slab[ix[s]], mslab[ix[s]]), "V": (mean, mok),
                 "mirror": (vi[mi], mt[mi] & keep_mirror)}
        r = rep.get(tg.person.iloc[i])
        if r is not None:
            fills["repeat"] = (slab[r], mslab[r])
        sibs = [j for j in np.flatnonzero(fam == fam[i]) if j != i and tg.age.iloc[j] != tg.age.iloc[i]]
        if sibs:
            fills["sibling"] = (slab[ix[sibs[0]]], mslab[ix[sibs[0]]])
        coh = np.flatnonzero(others & (tg.sex.to_numpy() == tg.sex.iloc[i])
                             & (np.abs(tg.age.to_numpy(float) - float(tg.age.iloc[i])) <= 2))
        if len(coh) >= 10:
            cm, cn = (allv[coh] * allm[coh]).sum(0), allm[coh].sum(0)
            fills["sexage"] = (np.where(cn > 0, cm / np.maximum(cn, 1), 0.0), cn >= 0.5 * len(coh))
        vt = mt & mok
        fr = make_frame(vi, vt, mean, fills, gauss, mid=half)
        fr["maps"] = {k: (v.T if v is not None else None) for k, v in fr["maps"].items()}
        fr["target"] = vi[:, :, half].T
        fr["tmask"] = vt[:, :, half].T & support.T
        frames.append(fr)
    return [finish_row("Brain MRI", "QTIM\naxial z = 10 mm", frames, support.T, "lower")]


# ---------------------------------------------------------------------------------------------- frames

def make_frame(xt, vt, mean, fills, local, mid=None):
    """Local error maps of each donor against the target and the per-target whole-frame ratio."""
    out, nums = {}, {}
    for key, (f, mf) in fills.items():
        m = vt & mf.astype(bool)
        out[key] = local(xt, f, m)
        sl = (slice(None), slice(None), mid) if mid is not None else (slice(None), slice(None))
        nums[key] = ratio_number(xt[sl], f[sl], mean[sl], m[sl]) if key != "V" else None
    sl = (slice(None), slice(None), mid) if mid is not None else (slice(None), slice(None))
    return {"maps": out, "nums": nums, "target": xt[sl], "tmask": vt[sl]}


def finish_row(group, name, frames, valid, origin):
    """Row-level normalisation: divide every map by the mean over the drawn targets of their own V map."""
    vs = np.stack([fr["maps"]["V"] for fr in frames])
    with np.errstate(invalid="ignore"):
        vtyp = np.nanmean(vs, 0)
    return {"group": group, "name": name, "frames": frames, "vtyp": vtyp, "valid": valid, "origin": origin}


def render(rows, path, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    nr, nc = len(rows), len(COLS)
    fig = plt.figure(figsize=(1.32 * nc + 1.3, 1.5 * nr + 1.3))
    gs = fig.add_gridspec(nr, nc, left=0.105, right=0.995, top=0.88, bottom=0.135, wspace=0.06, hspace=0.38)
    im_ratio = None
    for i, row in enumerate(rows):
        fr = row["frames"][k % len(row["frames"])]
        valid = row["valid"] & np.isfinite(row["vtyp"]) & (row["vtyp"] > 0)
        for j, (key, title) in enumerate(COLS):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=8.2, pad=6)
            tgt = np.where(valid & fr["tmask"], fr["target"], np.nan)
            tc, = LF.crop(tgt, valid=valid)
            lo, hi = np.nanpercentile(tc, [1, 99]) if np.isfinite(tc).any() else (0, 1)
            if key not in fr["maps"] and key != "target":
                ax.set_facecolor(LF.PANEL)
                ax.text(0.5, 0.5, "not available" if not (key == "lookalike" and row["group"] == "Brain MRI")
                        else "grid only,\nnot per target", ha="center", va="center", fontsize=6.4, color=LF.MUTED,
                        transform=ax.transAxes)
                continue
            if key == "target":
                ax.imshow(tc, cmap="gray", vmin=lo, vmax=hi, origin=row["origin"], interpolation="bilinear")
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where(valid & fr["tmask"], fr["maps"][key] / row["vtyp"], np.nan)   # only the target's pixels
            rc, = LF.crop(r, valid=valid)
            ax.imshow(np.where(np.isfinite(rc), tc, np.nan), cmap="gray", vmin=lo, vmax=hi, origin=row["origin"],
                      interpolation="bilinear")             # underlay only where the map is defined (no rims)
            im_ratio = ax.imshow(rc, cmap="RdBu_r", norm=LF.NORM, alpha=0.9, origin=row["origin"],
                                 interpolation="bilinear")
            v = fr["nums"].get(key)
            ax.text(0.5, -0.05, "own dataset-mean error" if key == "V" else
                    ("" if v is None else f"{v:.2f} × mean"), ha="center", va="top", fontsize=6.6,
                    transform=ax.transAxes)
        pos = gs[i, 0].get_position(fig)
        fig.text(0.012, (pos.y0 + pos.y1) / 2, row["group"], ha="left", va="bottom", fontsize=9, weight="bold")
        fig.text(0.012, (pos.y0 + pos.y1) / 2 - 0.006, row["name"], ha="left", va="top", fontsize=7.2,
                 color=LF.MUTED)
    cax = fig.add_axes([0.32, 0.075, 0.42, 0.011])
    cb = fig.colorbar(im_ratio, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1, 1.5, 2])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("local error of the donor / typical error of the dataset mean (blue: beats the mean, "
                 "red: 2 = a typical stranger)", fontsize=7.2)
    fig.text(0.5, 0.012, f"Comparison {k + 1} of {N_FRAMES}: one random target per row. Under each panel: this "
             "target's whole-image error of the donor / of the dataset mean. Pre-QC; private, real people.",
             ha="center", fontsize=6.2, color=LF.MUTED)
    fig.suptitle("Single comparisons, every modality: where a donor is closer or farther than the dataset mean",
                 fontsize=11, y=0.985)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(os.path.join(OUTDIR, "frames"), exist_ok=True)
    rows = face_frames()
    print("faces done", flush=True)
    rows += brain_frames()
    print("brain done", flush=True)
    rows += fundus_frames()
    print("fundus done", flush=True)
    for k in range(N_FRAMES):
        render(rows, os.path.join(OUTDIR, "frames", f"f{k:03d}.png"), k)
        if k % 10 == 0:
            print(f"frame {k + 1}/{N_FRAMES}", flush=True)
    mp4 = os.path.join(OUTDIR, "ladder_maps_all_modalities_100.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1.5", "-i",
                    os.path.join(OUTDIR, "frames", "f%03d.png"), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4], check=True)
    gif = os.path.join(OUTDIR, "ladder_maps_all_modalities_100.gif")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1.5", "-i",
                    os.path.join(OUTDIR, "frames", "f%03d.png"), "-vf",
                    "scale=1100:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", gif], check=True)
    print("written", mp4, gif, flush=True)


if __name__ == "__main__":
    main()

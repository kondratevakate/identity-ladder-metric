"""Simulations for the ISBI theory: donor averaging (Prop. 1, k_eff) and linear vs nonlinear read-outs (Prop. 2).

Everything is synthetic with a known distribution; no subject data. Writes a JSON of aggregates and the paper figure.
  python src/theory_sim.py --json paper/isbi2027/theory_sim.json --fig paper/isbi2027/overleaf/fig_theory.pdf
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

R2S = (0., .3, .6)
KS = (1, 2, 4, 8, 16, 32)
RS = (.25, .5, .75)


def theory_error(r2, k):
    """Prop. 1: E||x - f_k||^2 / (D V) for k donors at a rung explaining r2 of the individual variance."""
    return (1-r2)*(1+(0 if np.isinf(k) else 1/k))


def donor_errors(rng, r2, k, n=2000, d=256):
    """Per-target squared error / D of the k-donor average (k=inf: the rung mean), with V = 1."""
    p = rng.normal(scale=np.sqrt(r2), size=(n, d))
    x = p+rng.normal(scale=np.sqrt(1-r2), size=(n, d))
    f = p if np.isinf(k) else p+rng.normal(scale=np.sqrt(1-r2), size=(k, n, d)).mean(0)
    return ((x-f)**2).mean(1), (f**2).mean(1)


def single_donor_errors(rng, r, n=2000, d=256):
    """One donor whose individual part has correlation r with the target's (same-person rung)."""
    x = rng.normal(size=(n, d))
    y = r*x+np.sqrt(1-r*r)*rng.normal(size=(n, d))
    return ((x-y)**2).mean(1)


def k_eff(spread, r2):
    """k_eff from the fill's spread around the population mean (Sec. 2.6); inf when the fill is the rung mean."""
    excess = spread-r2
    return np.inf if excess <= 0 else (1-r2)/excess


def field(rng, n, size=64, sigma=3.):
    """Unit-variance smooth Gaussian random fields."""
    f = gaussian_filter(rng.normal(size=(n, size, size)), (0, sigma, sigma), mode='wrap')
    return f/f.std(axis=(1, 2), keepdims=True)


def volume_study(rng, r2, n=200, kmax=32, frac=.2, noise=.1, size=64):
    """Sparse class (CSF-like, prevalence frac) = 1[F > t], F = sqrt(r2) Fc + sqrt(1-r2) Fi; image = label + noise.

    For each target, donors share Fc and draw new Fi. Compares three volume estimates against the target's volume:
    segment the k-average image (hard threshold at 1/2), average the k donors' segmentations, and the soft (linear)
    volume mean(fill). k=inf uses the exact conditional mean image Phi((sqrt(r2) Fc - t) / sqrt(1-r2)).
    """
    t = norm.ppf(1-frac)
    ks = [k for k in KS if k <= kmax]
    out = {m: {k: [] for k in ks+['inf']} for m in ('seg_of_mean', 'mean_of_segs', 'soft')}
    examples = None
    for i in range(n):
        fc = field(rng, 1, size)[0]
        lab = np.sqrt(r2)*fc+np.sqrt(1-r2)*field(rng, 1, size)[0] > t
        vol = lab.mean()
        dl = np.sqrt(r2)*fc+np.sqrt(1-r2)*field(rng, kmax, size) > t
        dimg = dl+rng.normal(scale=noise, size=dl.shape)
        for k in ks:
            f = dimg[:k].mean(0)
            out['seg_of_mean'][k].append((f > .5).mean()-vol)
            out['mean_of_segs'][k].append((dimg[:k] > .5).mean()-vol)
            out['soft'][k].append(f.mean()-vol)
        m = norm.cdf((np.sqrt(r2)*fc-t)/np.sqrt(1-r2)) if r2 < 1 else (fc > t).astype(float)
        out['seg_of_mean']['inf'].append((m > .5).mean()-vol)
        out['mean_of_segs']['inf'].append(m.mean()-vol)  # limit of averaging k segmentations
        out['soft']['inf'].append(m.mean()-vol)
        if i == 0:
            examples = {'truth': lab+rng.normal(scale=noise, size=lab.shape), 'donor': dimg[0], 'k4': dimg[:4].mean(0),
                        'mean': m}
    summary = {m: {str(k): {'bias': float(np.mean(v)/frac), 'rmse': float(np.sqrt(np.mean(np.square(v)))/frac)}
                   for k, v in d.items()} for m, d in out.items()}
    return summary, examples


def run(seed=2027):
    rng = np.random.default_rng(seed)
    res = {'seed': seed, 'donors': {}, 'single': {}, 'k_eff': {}, 'volume': {}}
    for r2 in R2S:
        rows = {}
        for k in KS+(np.inf,):
            err, spread = donor_errors(rng, r2, k)
            rows['inf' if np.isinf(k) else str(k)] = {'error': float(err.mean()), 'theory': theory_error(r2, k),
                                                     'k_eff': k_eff(float(spread.mean()), r2)}
        res['donors'][str(r2)] = rows
    for r in RS:
        res['single'][str(r)] = {'error': float(single_donor_errors(rng, r).mean()), 'theory': 2*(1-r)}
    examples = None
    for r2 in R2S:
        res['volume'][str(r2)], ex = volume_study(rng, r2)
        if r2 == .3:
            examples = ex
    return res, examples


def figure(res, ex, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': .5})
    fig = plt.figure(figsize=(7.0, 2.0))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=.42)
    gi = outer[0].subgridspec(2, 4, wspace=.05, hspace=.05)
    names = [('truth', 'target'), ('donor', 'one donor'), ('k4', '$k=4$'), ('mean', '$k=\\infty$')]
    for j, (key, title) in enumerate(names):
        for i in range(2):
            ax = fig.add_subplot(gi[i, j])
            img = ex[key] if i == 0 else (ex[key] > .5)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=7)
            if j == 0:
                ax.set_ylabel('image' if i == 0 else 'segm.', fontsize=7)
    fig.text(.1, .97, '(a) $R^2=0.3$', fontsize=7)
    colors = dict(zip(map(str, R2S), ('#4c72b0', '#dd8452', '#55a868')))
    ax = fig.add_subplot(outer[1])
    kk = np.array(KS)
    for r2, rows in res['donors'].items():
        ax.plot(kk, [rows[str(k)]['error'] for k in KS], 'o', ms=3, color=colors[r2])
        ax.plot(kk, [theory_error(float(r2), k) for k in KS], '-', lw=.8, color=colors[r2], label=f'$R^2={r2}$')
        ax.axhline(rows['inf']['error'], ls=':', lw=.6, color=colors[r2])
    for r, row in res['single'].items():
        ax.plot(1, row['error'], 'kx', ms=4)
        ax.annotate(f'$r={r}$', (1, row['error']), xytext=(4, 0), textcoords='offset points', fontsize=6, va='center')
    ax.set_xscale('log', base=2), ax.set_xlabel('donors averaged, $k$'), ax.set_ylabel('MSE / $V$')
    ax.set_title('(b) pixel error', fontsize=7, loc='left'), ax.legend(frameon=False, fontsize=6)
    ax = fig.add_subplot(outer[2])
    for r2, d in res['volume'].items():
        ax.plot(kk, [d['seg_of_mean'][str(k)]['bias'] for k in KS], '-o', ms=2.5, lw=.8, color=colors[r2])
        ax.plot(kk, [d['mean_of_segs'][str(k)]['bias'] for k in KS], '--s', ms=2.5, lw=.8, color=colors[r2])
    ax.axhline(0, color='k', lw=.4)
    ax.plot([], [], 'k-o', ms=2.5, lw=.8, label='segment the average')
    ax.plot([], [], 'k--s', ms=2.5, lw=.8, label='average the segmentations')
    ax.set_xscale('log', base=2), ax.set_xlabel('donors averaged, $k$'), ax.set_ylabel('relative volume bias')
    ax.set_title('(c) volume of a sparse class', fontsize=7, loc='left')
    ax.legend(frameon=False, fontsize=6, loc='center left', bbox_to_anchor=(0, .45))
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(Path(path).with_suffix('.png'), dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='paper/isbi2027/theory_sim.json')
    ap.add_argument('--fig', default='paper/isbi2027/overleaf/fig_theory.pdf')
    a = ap.parse_args()
    res, ex = run()
    Path(a.json).write_text(json.dumps(res, indent=1, default=lambda v: 'inf' if v == np.inf else v))
    figure(res, ex, a.fig)
    print(json.dumps({r2: {k: round(v['error'], 3) for k, v in rows.items()} for r2, rows in res['donors'].items()}))
    print(json.dumps(res['single']))
    print(json.dumps({r2: {m: {k: round(v['bias'], 3) for k, v in d[m].items()} for m in d}
                      for r2, d in res['volume'].items()}))

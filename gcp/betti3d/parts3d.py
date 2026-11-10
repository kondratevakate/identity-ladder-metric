"""Betti matching 3D parts with the formula of iris_ladder_metrics_extra.betti_parts (Amendment 24 B)."""
import numpy as np


def parts(a, b, bm, dims=None, essential=True):
    """(matched, unmatched) for arrays a, b in [0, 1]; dims: None = all, else e.g. (0, 1).
    essential: add the one essential 0-dim class (global minimum of each input, death = 1 in both, as the 2D
    library's end(inf) = 1), which this library does not return as a pair."""
    r = bm.compute_matching(a, b)
    nd = len(r.num_matched)
    keep = range(nd) if dims is None else [d for d in dims if d < nd]

    def split(arr, counts):
        """Per-dimension coordinate arrays (this build returns a list per dimension; a flat array is split)."""
        if isinstance(arr, (list, tuple)):
            return [np.asarray(x).reshape(-1, a.ndim) for x in arr]
        idx = np.concatenate([[0], np.cumsum(counts)])
        return [arr[idx[d]:idx[d + 1]] for d in range(nd)]

    def val(v, c):
        return v[tuple(np.asarray(c, int).T)] if len(c) else np.zeros(0)
    mb1, md1 = split(r.input1_matched_birth_coordinates, r.num_matched), split(r.input1_matched_death_coordinates, r.num_matched)
    mb2, md2 = split(r.input2_matched_birth_coordinates, r.num_matched), split(r.input2_matched_death_coordinates, r.num_matched)
    ub1, ud1 = split(r.input1_unmatched_birth_coordinates, r.num_unmatched_input1), split(r.input1_unmatched_death_coordinates, r.num_unmatched_input1)
    ub2, ud2 = split(r.input2_unmatched_birth_coordinates, r.num_unmatched_input2), split(r.input2_unmatched_death_coordinates, r.num_unmatched_input2)
    matched = unmatched = 0.0
    for d in keep:
        matched += 2 * float(np.sum((val(a, mb1[d]) - val(b, mb2[d])) ** 2 + (val(a, md1[d]) - val(b, md2[d])) ** 2))
        unmatched += float(np.sum((val(a, ub1[d]) - val(a, ud1[d])) ** 2) + np.sum((val(b, ub2[d]) - val(b, ud2[d])) ** 2))
    if essential:
        matched += 2 * float((a.min() - b.min()) ** 2)
    return matched, unmatched

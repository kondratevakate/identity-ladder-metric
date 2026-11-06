"""ladder-result-spec/1: checks every ladder result computed on data must pass (general protocol, section 8).

A results JSON declares `"spec": "ladder-result-spec/1"` and has `results: {dataset: {rung: {...}}}` plus
`spec_meta: {"bootstrap_unit": "person"|"family", "pools": "leave-person-out", "stranger_rung": name,
"mirror_rungs": {rung: "sides"|"halves"}}`. Runners call `enforce` before writing; tests/test_result_spec.py
re-checks every declaring file in results/. Hard violations raise SpecError; soft ones are returned as flags.
"""
import math

SPEC = "ladder-result-spec/1"
STRANGER_RANGE = (1.85, 2.15)
DIRECTION_TOL = 0.15                     # |ln(a / b)| between the two directions of a mirror rung


class SpecError(AssertionError):
    pass


def _rungs(block):
    return {k: v for k, v in block.items() if isinstance(v, dict) and "ratio" in v}


def check(doc):
    """List of (level, dataset, rung, message); level 'hard' or 'soft'."""
    out = []
    meta = doc.get("spec_meta", {})
    if doc.get("spec") != SPEC:
        out.append(("hard", "-", "-", f"spec is not {SPEC}"))
    if meta.get("bootstrap_unit") not in ("person", "family"):
        out.append(("hard", "-", "-", "bootstrap unit must be person or family (never images, eyes or sides)"))
    if meta.get("pools") != "leave-person-out":
        out.append(("hard", "-", "-", "pools must be declared leave-person-out"))
    stranger = meta.get("stranger_rung")
    mirrors = meta.get("mirror_rungs", {})
    for ds, block in doc.get("results", {}).items():
        rungs = _rungs(block)
        if stranger not in rungs:
            out.append(("hard", ds, str(stranger), "stranger rung missing"))
        for name, r in rungs.items():
            ci = r.get("ci95")
            if not (isinstance(ci, list) and len(ci) == 2 and ci[0] <= r["ratio"] <= ci[1]):
                out.append(("hard", ds, name, f"95% CI {ci} missing or does not contain the ratio {r['ratio']}"))
            if r.get("targets", 0) < 10:
                out.append(("hard", ds, name, f"n targets {r.get('targets')} < 10"))
            if r.get("persons", math.inf) > r.get("targets", 0):
                out.append(("hard", ds, name, "more persons than targets"))
            if name == stranger and not STRANGER_RANGE[0] <= r["ratio"] <= STRANGER_RANGE[1]:
                out.append(("soft", ds, name, f"stranger ratio {r['ratio']} outside {STRANGER_RANGE} (theory 2)"))
            if name in mirrors:
                d = r.get("directions", {})
                if len(d) != 2:
                    out.append(("hard", ds, name, f"mirror rung needs exactly 2 directions, has {sorted(d)}"))
                    continue
                (na, a), (nb, b) = sorted(d.items())
                if mirrors[name] == "sides" and a["targets"] + b["targets"] != r["targets"]:
                    out.append(("hard", ds, name, f"directions {a['targets']} + {b['targets']} != targets {r['targets']}"))
                if mirrors[name] == "halves" and not a["targets"] == b["targets"] == r["targets"]:
                    out.append(("hard", ds, name, "each half-direction must cover every target"))
                if min(a["ratio"], b["ratio"]) > 0 and abs(math.log(a["ratio"] / b["ratio"])) > DIRECTION_TOL:
                    out.append(("soft", ds, name, f"directions differ: {na} {a['ratio']:.3f} vs {nb} {b['ratio']:.3f}; "
                                                  "check orientation and reflection before reading"))
    return out


def enforce(doc):
    """Raise SpecError on any hard violation; return the soft flags (and store all checks in doc['spec_checks'])."""
    res = check(doc)
    doc["spec_checks"] = [{"level": lv, "dataset": ds, "rung": rg, "message": m} for lv, ds, rg, m in res] or "all passed"
    hard = [r for r in res if r[0] == "hard"]
    if hard:
        raise SpecError("; ".join(f"{ds}/{rg}: {m}" for _, ds, rg, m in hard))
    return [r for r in res if r[0] == "soft"]


def assert_leave_person_out(target_person, pool_persons):
    """Run-time check: the target's own person never appears in its pool."""
    if any(p == target_person for p in pool_persons):
        raise SpecError(f"target person found in its own pool")

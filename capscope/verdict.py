"""Verdict engine: one verdict, one plain-language sentence and a 0-1 growth
priority per module. Rules are evaluated in order; first match wins."""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .capture import is_ancestor, is_relative

VERDICT_ORDER = ["narrow", "passthrough", "oversmoothed", "upstream", "undertrained",
                 "redundant", "saturated", "tight", "spare"]

THRESH = {
    "narrow_width": 8,
    "oversmooth_abs": 0.05,
    "oversmooth_ratio": 0.30,
    "alpha_undertrained": 6.0,
    "redundant_frac": 0.35,
    "cka_dup": 0.98,
    "saturated": 0.85,
    "tight": 0.60,
    "alpha_min_rank": 20,     # alpha is not used for a verdict on matrices with fewer eigenvalues
    "upstream_slack": 0.10,   # carry*(1-slack)-1 <= rank <= carry*(1+slack)+1  => tracks the input
}


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _isnum(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


# ----------------------------------------------------------------------------
# upstream carry (trap 3 / 4)
# ----------------------------------------------------------------------------

def compute_carry(mods: Dict[str, dict]) -> None:
    """Propagate the narrowest point along each path to every module.

    carry(M) = min(in_dim(M), sum over producers P of info_out(P)), where
    info_out(P) = min(width(P), carry(P)). A source module (no producer)
    carries its own input width. A nonlinearity cannot add information, so
    a ReLU's info_out is exactly its carry regardless of measured rank.
    """
    order = sorted(mods.values(), key=lambda m: m["exec_order"])
    for m in order:
        in_dim = m.get("in_dim")
        width = m.get("width")
        prods = [p for p in m["producers"] if p in mods]
        if not prods:
            carry = in_dim if in_dim else (width or 0)
        else:
            total = 0
            for p in prods:
                pm = mods[p]
                pw = pm.get("width") or 0
                pc = pm.get("carry") if pm.get("carry") is not None else pw
                total += min(pw, pc) if pw else pc
            carry = min(in_dim, total) if in_dim else total
        m["carry"] = int(carry) if carry is not None else None


def _nearest_upstream(mods: Dict[str, dict], name: str, pred, skip_relatives: bool = False) -> Optional[str]:
    """BFS backwards through producers for the nearest module satisfying pred.
    With ``skip_relatives`` a module's own descendants (and ancestors) are
    walked through but never returned, so a container is compared with what
    feeds it, not with its own child."""
    seen: Set[str] = {name}
    q = deque(mods[name]["producers"])
    while q:
        p = q.popleft()
        if p in seen or p not in mods:
            continue
        seen.add(p)
        if pred(mods[p]) and not (skip_relatives and is_relative(name, p)):
            return p
        q.extend(mods[p]["producers"])
    return None


# ----------------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------------

def assign_verdicts(mods: Dict[str, dict], cka: Optional[Dict] = None, thresh: Dict = None) -> None:
    """Mutates every module dict, adding verdict / sentence / priority /
    reason fields. ``cka`` is {"names": [...], "matrix": [[...]]}."""
    T = dict(THRESH)
    if thresh:
        T.update(thresh)
    compute_carry(mods)

    # consumers (graph-node level), downstream reachability, sinks. A graph
    # node is a leaf or an emitter container (one that builds its own output).
    def _is_node(m: dict) -> bool:
        return bool(m["is_leaf"] or m.get("emits"))
    nodes = {n for n, m in mods.items() if _is_node(m)}
    for m in mods.values():
        m["consumers"] = []
    for m in mods.values():
        if m["name"] not in nodes:
            continue
        for p in m["producers"]:
            if p in nodes:
                mods[p]["consumers"].append(m["name"])
    down: Dict[str, Set[str]] = {}
    for n in sorted(nodes, key=lambda x: -mods[x]["exec_order"]):
        s: Set[str] = set()
        for c in mods[n]["consumers"]:
            s.add(c)
            s |= down.get(c, set())
        down[n] = s
    for m in mods.values():
        if m["name"] in nodes:
            m["_down"] = down.get(m["name"], set()) - {m["name"]}
        else:
            own = {l for l in nodes if is_ancestor(m["name"], l)}
            m["_down"] = set().union(*(down.get(l, set()) for l in own)) - own
            m["consumers"] = sorted({c for l in own for c in mods[l]["consumers"] if c not in own})
        m["is_sink"] = len(m["_down"]) == 0

    def _leaves_of(n: str) -> Set[str]:
        return {n} if n in nodes else {l for l in nodes if is_ancestor(n, l)}

    # best CKA partner that is neither a relative nor downstream of the module
    # (a consumer that merely copies its input is the consumer's problem)
    if cka and cka.get("names"):
        names = cka["names"]
        M = cka["matrix"]
        idx = {n: i for i, n in enumerate(names)}
        for m in mods.values():
            i = idx.get(m["name"])
            m["cka_partner"] = None
            m["cka_partner_value"] = None
            if i is None:
                continue
            best, best_v = None, -1.0
            for j, other in enumerate(names):
                if j == i or other not in mods or is_relative(m["name"], other):
                    continue
                if _leaves_of(other) & m["_down"]:
                    continue
                v = M[i][j]
                if _isnum(v) and v > best_v:
                    best, best_v = other, v
            m["cka_partner"] = best
            m["cka_partner_value"] = best_v if best is not None else None
    for m in mods.values():
        m.pop("_down", None)

    for m in mods.values():
        v, sentence, reason = _verdict_for(m, mods, T)
        m["verdict"] = v
        m["sentence"] = sentence
        m["reason"] = reason
        m["priority"] = _priority(m, T)


def _used_rank(m: dict) -> Tuple[Optional[int], str]:
    """The rank the module actually uses and where the number came from."""
    sw = m.get("sweep")
    if sw and sw.get("used_rank") is not None:
        return int(sw["used_rank"]), "truncation sweep"
    rk = m.get("rank") or {}
    if rk.get("rank99"):
        return int(rk["rank99"]), "99% activation energy"
    return None, "n/a"


def _verdict_for(m: dict, mods: Dict[str, dict], T: Dict) -> Tuple[str, str, str]:
    name = m["name"]
    width = m.get("width") or 0
    used, used_src = _used_rank(m)
    m["used_rank"] = used
    m["used_rank_source"] = used_src
    m["used_frac"] = (min(1.0, used / width) if (used is not None and width) else None)
    cap = m.get("rank_cap")
    wcap = m.get("max_weight_rank")
    carry = m.get("carry")
    typ = m.get("type", "module")

    # 1. narrow ---------------------------------------------------------------
    if width and width < T["narrow_width"]:
        role = "an output head" if m.get("is_sink") else "a very narrow layer"
        return ("narrow",
                f"Output width is {width} (below {T['narrow_width']}): this is {role}, "
                f"so capacity metrics do not apply to it.",
                "width < 8")

    # 2. passthrough ----------------------------------------------------------
    if m["is_leaf"] and m["n_params_own"] == 0:
        feeder = _nearest_upstream(mods, name, lambda x: x["n_params_total"] > 0)
        tail = f"look at {feeder}, which feeds it" if feeder else "look at whatever feeds it"
        return ("passthrough",
                f"{typ} has no parameters of its own, so there is nothing to widen here; {tail}.",
                "no parameters")

    # 3. oversmoothed (decay vs nearest upstream graph layer, trap 5) ----------
    de = m.get("dirichlet")
    if _isnum(de):
        ref = _nearest_upstream(mods, name, lambda x: _isnum(x.get("dirichlet")), skip_relatives=True)
        if ref is not None:
            dref = mods[ref]["dirichlet"]
            ratio = de / dref if dref > 0 else float("inf")
            m["dirichlet_ref"] = ref
            m["dirichlet_ratio"] = ratio
            if de < T["oversmooth_abs"] and ratio < T["oversmooth_ratio"]:
                return ("oversmoothed",
                        f"Dirichlet energy {de:.3f} is only {_fmt_pct(ratio)} of upstream {ref} "
                        f"({dref:.3f}): node features are collapsing onto their neighbours, so widening "
                        f"will not help; add residual connections or normalisation, or use fewer hops.",
                        "dirichlet decay")

    # 4. upstream: rank merely tracks a narrower input (traps 3 / 4) ----------
    if used is not None and width and carry is not None and carry < width:
        # a leaf's output rank is bounded by its weight; a container's is not
        # (nonlinearities between its layers lift it), so only leaves are capped
        eff = min(used, wcap) if (wcap is not None and m["is_leaf"]) else used
        slack = T["upstream_slack"]
        # "tracks" means used ~= carry: using far *less* than what arrives is spare, not upstream
        if carry * (1 - slack) - 1 <= eff <= carry * (1 + slack) + 1:
            src = _nearest_upstream(mods, name, lambda x: (x.get("width") or 0) <= carry
                                    and x["n_params_total"] > 0)
            if src:
                where = f"widen {src} instead"
            elif not [p for p in m["producers"] if not is_ancestor(name, p)]:
                where = "the raw input is the limit, so add input features instead"
            else:
                where = ("add capacity where that information is created (earlier layers, more "
                         "hops or richer inputs), not here")
            capnote = (f"its weight rank is capped at {wcap} by the input, " if cap == "input"
                       else ("it has no matrix weights to widen anyway, " if not m["has_matrix"] else ""))
            return ("upstream",
                    f"Uses about {eff} of {width} dims, which just tracks the roughly {carry}-wide "
                    f"information reaching it; {capnote}no width here can add information the input "
                    f"never supplied: {where}.",
                    "rank tracks upstream carry")

    # 5. undertrained ---------------------------------------------------------
    alpha = m.get("alpha")
    if (_isnum(alpha) and alpha > T["alpha_undertrained"]
            and (m.get("max_weight_rank") or 0) >= T["alpha_min_rank"]):
        return ("undertrained",
                f"Weight spectrum is still close to random (alpha {alpha:.1f} > 6): this layer needs "
                f"more steps, data or learning rate, not more neurons.",
                "alpha > 6")

    # 6. redundant ------------------------------------------------------------
    red = m.get("redundancy") or {}
    rf = red.get("redundant_frac", 0.0) or 0.0
    if rf > T["redundant_frac"]:
        return ("redundant",
                f"{_fmt_pct(rf)} of its {width} units are dead ({red.get('dead', 0)}) or near-duplicates "
                f"({red.get('dup', 0)}): the width it has is not being used, so prune or regularise "
                f"before widening.",
                "dead/duplicate units")
    pv = m.get("cka_partner_value")
    if _isnum(pv) and pv > T["cka_dup"]:
        return ("redundant",
                f"Output is almost identical to {m['cka_partner']} (CKA {pv:.3f}): the two modules are "
                f"doing the same job, so widening either is wasted.",
                "CKA duplicate")

    # 7. saturated / tight / spare --------------------------------------------
    frac = m.get("used_frac")
    if frac is None:
        return ("spare", f"No rank estimate is available for this module.", "no estimate")
    if cap == "input":
        capnote = (f"its weight rank is capped at {wcap} by the input, so widen upstream before "
                   f"widening here")
    elif cap in ("output", "square"):
        capnote = "its weight rank is capped by its own output width, so this is where width is missing"
    else:
        capnote = "it has no matrix weights, so the estimate comes from activations only"
    base = f"Uses {used} of {width} dims ({_fmt_pct(frac)}, {used_src})"
    if frac > T["saturated"]:
        return ("saturated", f"{base}: {capnote}; extra width would be absorbed here.", "used > 85%")
    if frac > T["tight"]:
        return ("tight", f"{base}: {capnote}; it is approaching capacity.", "used > 60%")
    return ("spare", f"{base}: it still has room, so widening it is unlikely to help yet.", "used <= 60%")


def _priority(m: dict, T: Dict) -> float:
    v = m["verdict"]
    frac = m.get("used_frac") or 0.0
    # marginal pressure from the sweep: relative loss increase when the module
    # loses its last used dimension, squashed to 0-1 (0.05 -> 0.5, 0.33 -> 0.87)
    sw = m.get("sweep") or {}
    pr = sw.get("pressure")
    press = (pr / (pr + 0.05)) if _isnum(pr) else 0.0
    m["pressure"] = pr if _isnum(pr) else None
    if v == "saturated":
        p = 0.70 + 0.15 * frac + 0.15 * press
    elif v == "tight":
        p = 0.35 + 0.20 * (frac - T["tight"]) / max(1e-9, T["saturated"] - T["tight"]) + 0.10 * press
    elif v == "spare":
        p = 0.15 * frac
    elif v == "upstream":
        p = 0.12
    elif v == "undertrained":
        p = 0.08
    elif v in ("redundant", "oversmoothed"):
        p = 0.05
    else:
        p = 0.0
    if v in ("saturated", "tight") and m.get("rank_cap") == "input":
        p *= 0.6
    if v in ("saturated", "tight", "spare") and m.get("is_sink"):
        p *= 0.5   # output layers: width is fixed by the task
    return float(max(0.0, min(1.0, p)))

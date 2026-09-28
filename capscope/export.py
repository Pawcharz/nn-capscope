"""Compact, self-describing export of a Report for logs and language models.

The GUI payload (``Report.to_dict``) carries everything the browser draws:
histograms, singular-value arrays, the full CKA matrix, the module hierarchy.
That is hundreds of kB and most of it is noise to a reader that wants to
decide where to add capacity. ``to_llm_dict`` keeps one record per module with
only the scalar metrics, the verdict, the sentence and the graph neighbours,
and attaches a ``legend`` that defines every field and every verdict, so the
document can be read on its own, months later, without the code or the guide.

``to_llm_markdown`` renders the same content as a Markdown document: header
and warnings, a priority-ranked table, one block per module, the edge list
and the legend.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .capture import is_relative
from .verdict import VERDICT_ORDER

FORMAT = "capscope-llm/1"

# ---------------------------------------------------------------------------
# legend: one line per exported field / verdict. Keep in step with README.
# ---------------------------------------------------------------------------

FIELDS: Dict[str, str] = {
    "name": "Module path as in model.named_modules(); dots separate parent from child.",
    "type": "Python class of the module.",
    "is_leaf": "True when the module has no child modules.",
    "emits": "True for a container that builds its own output (e.g. lin_l(agg) + lin_r(x)) instead of handing a child's output through; such containers are graph nodes.",
    "is_node": "True when the module is a node of the dataflow graph (leaf or emitting container).",
    "is_sink": "True when nothing downstream consumes the module: an output head. Its width is fixed by the task, so growth priority is halved.",
    "exec_order": "Position in the first forward pass (0 = executed first).",
    "n_calls": "Forward calls per batch (>1 means shared/reused module).",
    "n_params_own": "Parameters registered directly on the module.",
    "n_params_total": "Parameters including all children.",
    "in_dim": "Last dimension of the module's input tensor.",
    "width": "Output dimension D of the [N, D] activations. Denominator of every 'uses X of Y dims'.",
    "out_shape": "Shape of the output tensor on one batch (N rows = samples or graph nodes in the batch).",
    "producers": "Graph nodes whose outputs this module reads (recovered from the autograd graph, functional ops walked through).",
    "consumers": "Graph nodes that read this module's output.",
    "verdict": "One label per module, assigned in the order of 'verdicts' below; the first rule that fires wins.",
    "reason": "Short tag naming the rule that fired.",
    "sentence": "Plain-language explanation and the recommended action for this module.",
    "priority": "Growth priority in [0, 1]: how much adding width HERE is expected to help. Widen the highest first.",
    "used_rank": "Smallest weight rank keeping the loss within rel_tol of base (truncation sweep); without a loss, rank at 99 % activation energy.",
    "used_rank_source": "'truncation sweep' or '99% activation energy': where used_rank came from.",
    "used_frac": "used_rank / width. > 0.85 saturated, > 0.60 tight, otherwise spare.",
    "pressure": "Relative loss increase when the module loses its last used dimension (loss(used_rank-1)/base - 1). Higher = the last dimensions matter more.",
    "carry": "Narrowest width along the paths feeding the module, propagated forward through the graph; the most information that can reach it. used_rank ~= carry < width means the module only relays a narrow input ('upstream').",
    "rank_cap": "Which side bounds the weight rank min(in, out): 'input', 'output' or 'square'. 'input' means widening this layer cannot add rank; widen upstream.",
    "max_weight_rank": "min(in, out) of the weight that bounds the module's output rank.",
    "alpha": "Hill-estimator power-law exponent of the WᵀW eigenvalue tail (median over the module's matrices). 2-6 well-conditioned, > 6 undertrained (still near random init), < 2 over-trained.",
    "alpha_reading": "Verbal reading of alpha.",
    "rank.effective_rank": "exp(entropy of normalised squared singular values) of the centred activations: how many dimensions carry energy evenly.",
    "rank.stable_rank": "sum(s²) / max(s²): robust lower bound on the number of active dimensions.",
    "rank.participation_ratio": "(sum s²)² / sum s⁴: another soft count of active dimensions.",
    "rank.numerical_rank": "Number of singular values above float tolerance.",
    "rank.rank90": "Dimensions needed for 90 % of activation energy.",
    "rank.rank99": "Dimensions needed for 99 % of activation energy.",
    "rank.n_rows": "Number of sampled activation rows the ranks were computed from.",
    "sweep.max_rank": "Rank of the untouched weight (min(in, out)).",
    "sweep.base_loss": "Loss of the untouched model on the sweep batches.",
    "sweep.loss_at_1": "Loss when the weight is truncated to rank 1.",
    "sweep.curve": "loss at each swept rank k (log grid); shows how fast the module degrades as rank is removed.",
    "redundancy.dead": "Units with (near) zero variance over the sampled rows.",
    "redundancy.dup": "Units whose activations correlate above dup_thresh with another unit.",
    "redundancy.redundant_frac": "(dead + dup) / width. > 0.35 => 'redundant'.",
    "dirichlet": "Centred Dirichlet energy over the batch graph: mean ||x_i - x_j||² over edges / mean ||x_i - mean||² over nodes. ~2 independent features, -> 0 collapsed onto neighbours. Only for node-aligned outputs when edge_index_fn is given.",
    "dirichlet_ref": "Nearest upstream module with a Dirichlet value, used as the decay reference.",
    "dirichlet_ratio": "dirichlet / dirichlet of dirichlet_ref. < 0.30 together with dirichlet < 0.05 => 'oversmoothed'.",
    "mad": "Mean cosine distance between each node and its neighbours (raw features).",
    "cka_partner": "Module with the highest linear CKA to this one, excluding relatives and downstream consumers.",
    "cka_partner_value": "That CKA value in [0, 1]; > 0.98 => 'redundant' (duplicate).",
    "weights[]": "Per matrix: name, shape [out, in], max_rank, rank_cap, alpha, numerical_rank, stable_rank.",
}

VERDICTS: Dict[str, str] = {
    "narrow": "Width below narrow_width: an output head or tiny layer; capacity metrics do not apply.",
    "passthrough": "Leaf with no parameters (activation, dropout, norm without affine): nothing to widen; look at what feeds it.",
    "oversmoothed": "Graph layer whose node features collapsed onto their neighbours (Dirichlet decay vs upstream); widening will not help, add residuals/normalisation or fewer hops.",
    "upstream": "Used rank merely tracks a narrower input (carry); the information is missing earlier, so widen the named upstream module or add input features.",
    "undertrained": "Weight spectrum still near random init (alpha > alpha_undertrained); needs more training, not more neurons.",
    "redundant": "Many dead/duplicate units or a near-identical twin module (CKA); the width it has is not used, prune or regularise before widening.",
    "saturated": "used_frac > saturated: the module uses almost all of its width; extra width would be absorbed here. Highest growth priority.",
    "tight": "used_frac > tight: approaching capacity.",
    "spare": "used_frac <= tight: still has room, widening is unlikely to help yet.",
}

META_FIELDS: Dict[str, str] = {
    "model": "Class name of the inspected model.",
    "n_params": "Total parameter count.",
    "n_batches": "Batches captured for activation metrics (0 = data-free mode: weight spectra only).",
    "has_loss": "Whether a loss was given: enables the truncation sweep and pressure. Without it used_rank falls back to rank99.",
    "has_graph": "Whether an edge index was given: enables dirichlet / mad.",
    "base_loss": "Loss of the untouched model on the sweep batches.",
    "thresholds": "Verdict thresholds in force for this report (verdict.THRESH plus overrides).",
    "capscope_version": "Version of capscope that produced the report.",
    "generated_at": "UTC timestamp of the export.",
    "source": "Factory spec or report file the CLI was run on, when known.",
}

_RANK_KEYS = ("effective_rank", "stable_rank", "participation_ratio", "numerical_rank",
              "rank90", "rank99", "n_rows")
_RED_KEYS = ("dead", "dup", "dead_frac", "dup_frac", "redundant_frac")
_WEIGHT_KEYS = ("name", "shape", "max_rank", "rank_cap", "alpha", "numerical_rank", "stable_rank")
_SCALAR_KEYS = ("name", "type", "is_leaf", "emits", "is_node", "is_sink", "exec_order", "n_calls",
                "n_params_own", "n_params_total", "in_dim", "width", "out_shape",
                "producers", "consumers", "verdict", "reason", "sentence", "priority",
                "used_rank", "used_rank_source", "used_frac", "pressure", "carry",
                "rank_cap", "max_weight_rank", "alpha", "alpha_reading",
                "dirichlet", "dirichlet_ref", "dirichlet_ratio", "mad",
                "cka_partner", "cka_partner_value")


def _num(x: Any, nd: int = 4) -> Any:
    """Round floats for a compact document; leave everything else alone."""
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, float):
        return None if not math.isfinite(x) else round(x, nd)
    return x


def _module_record(m: dict) -> dict:
    rec: Dict[str, Any] = {k: _num(m.get(k)) for k in _SCALAR_KEYS}
    rk = m.get("rank")
    rec["rank"] = {k: _num(rk.get(k)) for k in _RANK_KEYS} if rk else None
    sw = m.get("sweep")
    if sw:
        rec["sweep"] = {
            "used_rank": sw.get("used_rank"), "max_rank": sw.get("max_rank"),
            "base_loss": _num(sw.get("base_loss"), 6), "loss_at_1": _num(sw.get("loss_at_1"), 6),
            "pressure": _num(sw.get("pressure")),
            "curve": {str(k): _num(l, 6) for k, l in zip(sw.get("ks", []), sw.get("losses", []))},
        }
    else:
        rec["sweep"] = None
    red = m.get("redundancy")
    rec["redundancy"] = {k: _num(red.get(k)) for k in _RED_KEYS} if red else None
    rec["weights"] = [{k: _num(w.get(k)) for k in _WEIGHT_KEYS} for w in (m.get("weights") or [])]
    return rec


def _cka_pairs(report, top: int) -> List[dict]:
    """Highest off-diagonal CKA values between graph nodes that are not
    relatives (a container and its own children are trivially similar),
    largest first."""
    cka = report.cka or {}
    names, M = cka.get("names") or [], cka.get("matrix") or []
    is_node = {m["name"]: bool(m.get("is_node")) for m in report.modules}
    pairs = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            if not (is_node.get(a) and is_node.get(b)) or is_relative(a, b):
                continue
            v = M[i][j]
            if isinstance(v, (int, float)) and math.isfinite(v):
                pairs.append({"a": a, "b": b, "cka": round(float(v), 4)})
    pairs.sort(key=lambda p: -p["cka"])
    return pairs[:top]


def to_llm_dict(report, source: Optional[str] = None, top_cka: int = 15) -> dict:
    """One self-describing dict: meta, warnings, modules ranked by priority,
    edges, top CKA pairs and a legend. JSON-safe; no arrays longer than the
    sweep grid."""
    from . import __version__
    meta = dict(report.meta)
    meta["capscope_version"] = __version__
    meta["generated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if source is not None:
        meta["source"] = str(source)
    if isinstance(meta.get("base_loss"), float):
        meta["base_loss"] = _num(meta["base_loss"], 6)
    ranked = report.ranked()
    return {
        "format": FORMAT,
        "meta": meta,
        "warnings": list(report.warnings),
        "ranked": [m["name"] for m in ranked],
        "modules": [_module_record(m) for m in ranked],
        "edges": [list(e) for e in report.edges],
        "cka_top_pairs": _cka_pairs(report, top_cka),
        "legend": {
            "how_to_read": (
                "Modules are listed by growth priority (highest first). Widen the top saturated/tight "
                "modules whose rank_cap is not 'input'; for 'upstream' verdicts follow the sentence to the "
                "module that actually limits the information; treat 'redundant', 'oversmoothed' and "
                "'undertrained' as training/architecture problems, not width problems. If warnings mention "
                "an undertrained checkpoint, ignore used_rank and pressure entirely."),
            "verdict_order": list(VERDICT_ORDER),
            "verdicts": dict(VERDICTS),
            "fields": dict(FIELDS),
            "meta": dict(META_FIELDS),
        },
    }


def to_llm_json(report, source: Optional[str] = None, indent: Optional[int] = 1) -> str:
    return json.dumps(to_llm_dict(report, source), indent=indent, allow_nan=False, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _f(x: Any, nd: int = 2, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{100 * x:.0f} %" if pct else f"{x:.{nd}f}"
    return str(x)


def to_llm_markdown(report, source: Optional[str] = None) -> str:
    d = to_llm_dict(report, source)
    meta, L = d["meta"], []
    L.append(f"# capscope report: {meta.get('model', 'model')}")
    L.append("")
    bits = [f"{meta.get('n_params', 0):,} parameters", f"{len(d['modules'])} modules",
            f"{len(d['edges'])} dataflow edges", f"{meta.get('n_batches', 0)} batches captured"]
    if meta.get("base_loss") is not None:
        bits.append(f"base loss {meta['base_loss']}")
    bits.append("loss given" if meta.get("has_loss") else "no loss (used rank = rank99)")
    bits.append("graph edges given" if meta.get("has_graph") else "no graph")
    L.append("; ".join(bits) + ".")
    L.append("")
    L.append(f"capscope {meta.get('capscope_version')} · {meta.get('generated_at')}"
             + (f" · source `{meta['source']}`" if meta.get("source") else ""))
    L.append("")
    L.append("Thresholds: " + ", ".join(f"{k} = {v}" for k, v in (meta.get("thresholds") or {}).items()) + ".")
    if d["warnings"]:
        L.append("")
        L.append("## Warnings")
        L.append("")
        L += [f"- **WARNING:** {w}" for w in d["warnings"]]
    L.append("")
    L.append("## How to read this")
    L.append("")
    L.append(d["legend"]["how_to_read"])
    L.append("")
    L.append("## Modules by growth priority")
    L.append("")
    L.append("| # | module | type | verdict | priority | width | used | used % | pressure | carry | alpha | cap | params |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in d["modules"]:
        L.append("| " + " | ".join([
            str(m["exec_order"]), f"`{m['name']}`", m["type"] or "", m["verdict"] or "",
            _f(m["priority"]), _f(m["width"]), _f(m["used_rank"]), _f(m["used_frac"], pct=True),
            _f(m["pressure"], pct=True), _f(m["carry"]), _f(m["alpha"], 1), m["rank_cap"] or "–",
            f"{m['n_params_total']:,}",
        ]) + " |")
    L.append("")
    L.append("## Module details")
    for m in d["modules"]:
        L.append("")
        kind = "leaf" if m["is_leaf"] else ("container that computes its own output" if m["emits"] else "container")
        L.append(f"### `{m['name']}` — {m['verdict']} (priority {_f(m['priority'])})")
        L.append("")
        L.append(f"{m['type']}, {kind}{', output head (sink)' if m['is_sink'] else ''}; "
                 f"{m['n_params_total']:,} params; in_dim {_f(m['in_dim'])}, width {_f(m['width'])}, "
                 f"out_shape {m['out_shape']}; carry {_f(m['carry'])}; rank cap {m['rank_cap'] or '–'} "
                 f"(max weight rank {_f(m['max_weight_rank'])}).")
        L.append("")
        L.append(f"**Verdict ({m['reason']}):** {m['sentence']}")
        L.append("")
        facts = []
        if m["used_rank"] is not None:
            facts.append(f"used rank {m['used_rank']} of {_f(m['width'])} ({_f(m['used_frac'], pct=True)}, {m['used_rank_source']})")
        if m["pressure"] is not None:
            facts.append(f"pressure {_f(m['pressure'], pct=True)}")
        if m["rank"]:
            r = m["rank"]
            facts.append(f"activation ranks: effective {_f(r['effective_rank'], 1)}, stable {_f(r['stable_rank'], 1)}, "
                         f"participation {_f(r['participation_ratio'], 1)}, numerical {_f(r['numerical_rank'])}, "
                         f"rank90 {_f(r['rank90'])}, rank99 {_f(r['rank99'])} (from {r['n_rows']} rows)")
        if m["sweep"]:
            s = m["sweep"]
            curve = ", ".join(f"{k}→{_f(v, 4)}" for k, v in s["curve"].items())
            facts.append(f"sweep: base loss {_f(s['base_loss'], 4)}, loss at rank 1 {_f(s['loss_at_1'], 4)}, "
                         f"max rank {s['max_rank']}; curve k→loss: {curve}")
        if m["alpha"] is not None:
            facts.append(f"alpha {_f(m['alpha'], 2)} ({m['alpha_reading']})")
        if m["redundancy"]:
            rd = m["redundancy"]
            facts.append(f"redundancy: {rd['dead']} dead, {rd['dup']} duplicate units ({_f(rd['redundant_frac'], pct=True)})")
        if m["dirichlet"] is not None:
            s = f"Dirichlet energy {_f(m['dirichlet'], 3)}, MAD {_f(m['mad'], 3)}"
            if m["dirichlet_ref"]:
                s += f"; {_f(m['dirichlet_ratio'], pct=True)} of upstream `{m['dirichlet_ref']}`"
            facts.append(s)
        if m["cka_partner"]:
            facts.append(f"closest CKA partner `{m['cka_partner']}` ({_f(m['cka_partner_value'], 3)})")
        if m["producers"] or m["consumers"]:
            facts.append("producers: " + (", ".join(f"`{p}`" for p in m["producers"]) or "none (source)")
                         + "; consumers: " + (", ".join(f"`{c}`" for c in m["consumers"]) or "none (sink)"))
        if m["weights"]:
            facts.append("weights: " + "; ".join(
                f"`{w['name']}` {w['shape']} rank cap {w['rank_cap']} alpha {_f(w['alpha'], 2)} "
                f"numerical rank {_f(w['numerical_rank'])}" for w in m["weights"]))
        L += [f"- {f}" for f in facts]
    L.append("")
    L.append("## Dataflow edges (producer → consumer)")
    L.append("")
    L += [f"- `{a}` → `{b}`" for a, b in d["edges"]] or ["- none recovered"]
    if d["cka_top_pairs"]:
        L.append("")
        L.append("## Most similar module pairs (linear CKA)")
        L.append("")
        L += [f"- `{p['a']}` ~ `{p['b']}`: {p['cka']:.3f}" for p in d["cka_top_pairs"]]
    L.append("")
    L.append("## Legend")
    L.append("")
    L.append("Verdicts, in the order the rules are tried (first match wins): " + " → ".join(d["legend"]["verdict_order"]) + ".")
    L.append("")
    L += [f"- **{k}**: {v}" for k, v in d["legend"]["verdicts"].items()]
    L.append("")
    L.append("Fields:")
    L.append("")
    L += [f"- `{k}`: {v}" for k, v in d["legend"]["fields"].items()]
    L.append("")
    L.append("Meta:")
    L.append("")
    L += [f"- `{k}`: {v}" for k, v in d["legend"]["meta"].items()]
    L.append("")
    return "\n".join(L)

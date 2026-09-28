"""Orchestration (``inspect``) and the ``Report`` object (GUI / HTML / table)."""
from __future__ import annotations

import http.server
import itertools
import json
import math
import os
import socketserver
import threading
import webbrowser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .capture import Capture, ModuleRecord, is_ancestor, is_relative, matrix_params
from .metrics import (activation_histogram, alpha_reading, cka_matrix, rank_metrics,
                      unit_redundancy, weight_spectrum)
from .truncation import _eval_loss, truncation_sweep
from .verdict import THRESH, assign_verdicts

_TEMPLATE = Path(__file__).parent / "gui" / "template.html"


# ----------------------------------------------------------------------------
# inspect
# ----------------------------------------------------------------------------

def inspect(
    model: nn.Module,
    loader: Any = None,
    forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
    loss_fn: Optional[Callable[[Any, Any], torch.Tensor]] = None,
    edge_index_fn: Optional[Callable[[Any], torch.Tensor]] = None,
    n_batches: int = 8,
    max_rows: int = 4096,
    sweep_batches: Optional[int] = None,
    sweep_containers: bool = True,
    rel_tol: float = 0.01,
    thresholds: Optional[Dict] = None,
    verbose: bool = False,
) -> "Report":
    """Diagnose capacity of every module of ``model`` from forward passes over
    a frozen checkpoint plus SVD on the existing weights. Nothing is trained.

    forward_fn(model, batch) -> output          (default: model(batch))
    loss_fn(output, batch) -> scalar tensor     (optional; enables the sweep)
    edge_index_fn(batch) -> LongTensor [2, E]   (optional; enables graph metrics)
    """
    log = print if verbose else (lambda *a, **k: None)
    forward_fn = forward_fn or (lambda m, b: m(b))
    batches: List[Any] = []
    if loader is not None:
        batches = list(itertools.islice(iter(loader), n_batches))
        if not batches:
            raise ValueError("loader yielded no batches")

    cap: Optional[Capture] = None
    if batches:
        log(f"[capscope] capturing activations over {len(batches)} batches")
        cap = Capture(model, forward_fn, edge_index_fn, max_rows=max_rows,
                      n_batches=len(batches)).run(batches)

    mods = _build_modules(model, cap)
    edges = cap.graph_edges() if cap else []
    if cap:
        log(f"[capscope] recovered {len(edges)} dataflow edges among {len(mods)} modules")

    # ---- CKA across all captured modules ----
    names = [n for n, m in mods.items() if m.get("_X") is not None]
    cka = {"names": [], "matrix": []}
    if names:
        mats = [mods[n]["_X"] for n in names]
        keys = [tuple(mods[n]["_keys"]) for n in names]
        C = cka_matrix(mats, keys)
        cka = {"names": names, "matrix": C.tolist()}
        log("[capscope] CKA matrix computed")

    # ---- truncation sweep ----
    base_loss = None
    if loss_fn is not None and batches:
        sb = batches[: (sweep_batches or min(len(batches), 4))]
        base_loss = _eval_loss(model, sb, forward_fn, loss_fn)
        log(f"[capscope] base loss {base_loss:.5f}; sweeping ranks")
        for n, m in mods.items():
            if not m["has_matrix"]:
                continue
            if not m["is_leaf"] and not sweep_containers:
                continue
            m["sweep"] = truncation_sweep(model, m["_module"], sb, forward_fn, loss_fn,
                                          base_loss=base_loss, rel_tol=rel_tol)
            if m["sweep"]:
                log(f"    {n:40s} used {m['sweep']['used_rank']:4d} / {m['sweep']['max_rank']}")

    assign_verdicts(mods, cka, thresholds)
    warnings = _detect_problems(mods, base_loss, loss_fn is not None, cap)

    hierarchy = _hierarchy(model, mods)
    meta = {
        "model": type(model).__name__,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "n_batches": len(batches),
        "has_loss": loss_fn is not None,
        "has_graph": edge_index_fn is not None,
        "base_loss": base_loss,
        "thresholds": {**THRESH, **(thresholds or {})},
    }
    # strip private fields
    for m in mods.values():
        for k in [k for k in m if k.startswith("_")]:
            m.pop(k)
        m["producers"] = sorted(m["producers"])
    return Report(mods, edges, hierarchy, cka, warnings, meta)


# ----------------------------------------------------------------------------
# module dicts
# ----------------------------------------------------------------------------

def _build_modules(model: nn.Module, cap: Optional[Capture]) -> Dict[str, dict]:
    mods: Dict[str, dict] = {}
    if cap is not None:
        records = cap.executed()
    else:
        records = []
        for i, (name, m) in enumerate(model.named_modules()):
            if name == "":
                continue
            r = ModuleRecord(name=name, module=m, depth=name.count(".") + 1,
                             is_leaf=len(list(m.children())) == 0,
                             n_params_own=sum(p.numel() for p in m.parameters(recurse=False)),
                             n_params_total=sum(p.numel() for p in m.parameters()),
                             has_matrix=len(matrix_params(m)) > 0)
            r.exec_order = i
            records.append(r)

    for r in records:
        m = {
            "name": r.name,
            "type": type(r.module).__name__,
            "depth": r.depth,
            "is_leaf": r.is_leaf,
            "emits": bool(r.emits),
            "is_node": bool(r.is_leaf or r.emits),
            "n_params_own": r.n_params_own,
            "n_params_total": r.n_params_total,
            "has_matrix": r.has_matrix,
            "exec_order": r.exec_order,
            "n_calls": (round(r.n_calls / max(1, cap.batches_seen)) if cap else 0),
            "out_shape": list(r.out_shape) if r.out_shape else None,
            "in_dim": r.in_dim,
            "width": r.width,
            "producers": set(r.producers),
            "_module": r.module,
            "_X": None,
            "_keys": [],
            "rank": None, "hist": None, "redundancy": None,
            "dirichlet": None, "mad": None, "sweep": None,
        }
        # activations
        X = r.activation_matrix()
        if X is not None and X.size:
            m["_X"] = X
            m["_keys"] = list(r.sample_keys)
            m["rank"] = rank_metrics(X)
            m["hist"] = activation_histogram(X)
            m["redundancy"] = unit_redundancy(X)
        if r.dirichlet:
            de = [d for d in r.dirichlet if math.isfinite(d)]
            md = [d for d in r.mad if math.isfinite(d)]
            m["dirichlet"] = float(np.mean(de)) if de else None
            m["mad"] = float(np.mean(md)) if md else None
        # weight spectra
        wp = matrix_params(r.module, recurse=True)
        m["weights"] = [{"name": (n if not r.is_leaf else n), **weight_spectrum(p)} for n, p in wp]
        alphas = [w["alpha"] for w in m["weights"] if math.isfinite(w["alpha"])]
        m["alpha"] = float(np.median(alphas)) if alphas else None
        m["alpha_reading"] = alpha_reading(m["alpha"]) if m["alpha"] is not None else "n/a"
        m["max_weight_rank"] = None
        m["rank_cap"] = None
        if m["width"] is None and m["weights"] and r.is_leaf:
            m["width"] = m["weights"][0]["shape"][0]
        if m["in_dim"] is None and m["weights"] and r.is_leaf:
            m["in_dim"] = m["weights"][0]["shape"][1]
        mods[r.name] = m
    _assign_rank_caps(mods)
    return mods


def _assign_rank_caps(mods: Dict[str, dict]) -> None:
    """The weight whose rank bounds each module's *output*.

    A leaf: its narrowest matrix. A container: the matrices of the last leaf
    executed inside it (the one that produces its output), not the narrowest
    matrix anywhere inside it. A four-layer stack whose first layer takes 15
    inputs is not "capped at 15" as a whole; nonlinearities in between let the
    later layers use their full width, and the truncation sweep measures that.
    """
    by_exec = sorted(mods.values(), key=lambda m: m["exec_order"])
    for m in mods.values():
        if not m["weights"]:
            continue
        if m["is_leaf"]:
            pool = m["weights"]
        else:
            last = None
            for c in by_exec:
                if c["is_leaf"] and c["weights"] and is_ancestor(m["name"], c["name"]):
                    last = c
            pool = last["weights"] if last is not None else m["weights"]
        binding = min(pool, key=lambda w: w["max_rank"])
        m["max_weight_rank"] = binding["max_rank"]
        m["rank_cap"] = binding["rank_cap"]


def _hierarchy(model: nn.Module, mods: Dict[str, dict]) -> List[dict]:
    """Every named module (executed or not) with its parent, so the GUI can
    collapse/expand by hierarchy. Root ('') is a virtual node."""
    out = []
    for name, m in model.named_modules():
        parent = name.rsplit(".", 1)[0] if "." in name else ("" if name else None)
        out.append({
            "name": name,
            "parent": parent,
            "type": type(m).__name__,
            "is_leaf": len(list(m.children())) == 0,
            "executed": name in mods,
            "n_params_total": int(sum(p.numel() for p in m.parameters())),
        })
    return out


def _detect_problems(mods, base_loss, has_loss, cap) -> List[str]:
    warns: List[str] = []
    if has_loss:
        swept = [m for m in mods.values() if m.get("sweep") and m["sweep"]["max_rank"] > 2
                 and m["is_leaf"]]
        if swept:
            collapsed = sum(1 for m in swept if m["sweep"]["used_rank"] <= 1)
            if collapsed / len(swept) >= 0.5:
                warns.append(
                    f"{collapsed} of {len(swept)} swept layers keep the loss within tolerance at rank 1. "
                    "Destroying weights costs nothing, which means the checkpoint has not learned the "
                    "task: this looks UNDERTRAINED and the truncation-sweep ranks are meaningless. "
                    "Train to convergence before reading capacity from this report.")
        if base_loss is not None and not math.isfinite(base_loss):
            warns.append("Base loss is not finite; the truncation sweep could not be evaluated.")
    alphas = [m["alpha"] for m in mods.values() if m.get("alpha") is not None and m["is_leaf"]]
    if alphas and np.median(alphas) > THRESH["alpha_undertrained"]:
        warns.append(
            f"Median weight-spectrum alpha is {np.median(alphas):.1f} (> 6): most layers still look "
            "like their random initialisation. The model is undertrained; capacity verdicts are "
            "unreliable until it has trained further.")
    if cap is not None:
        n_edges = len(cap.graph_edges())
        nodes = [m for m in mods.values() if m["is_node"]]
        if len(nodes) > 1 and n_edges == 0:
            warns.append("No dataflow edges were recovered. Make sure the model's parameters require "
                         "grad and that outputs are differentiable tensors.")
    return warns


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def _sanitize(o):
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    return o


class Report:
    def __init__(self, mods: Dict[str, dict], edges, hierarchy, cka, warnings, meta):
        self._mods = mods
        self.modules: List[dict] = sorted(mods.values(), key=lambda m: m["exec_order"])
        self.edges: List[tuple] = [tuple(e) for e in edges]
        self.hierarchy = hierarchy
        self.cka = cka
        self.warnings: List[str] = warnings
        self.meta = meta

    # ---- access -----------------------------------------------------------
    def __getitem__(self, name: str) -> dict:
        return self._mods[name]

    def __contains__(self, name: str) -> bool:
        return name in self._mods

    def names(self) -> List[str]:
        return [m["name"] for m in self.modules]

    def ranked(self) -> List[dict]:
        return sorted(self.modules, key=lambda m: (-m["priority"], m["exec_order"]))

    def verdict(self, name: str) -> str:
        return self._mods[name]["verdict"]

    # ---- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        return _sanitize({
            "meta": self.meta,
            "warnings": self.warnings,
            "modules": self.modules,
            "edges": [list(e) for e in self.edges],
            "hierarchy": self.hierarchy,
            "cka": self.cka,
        })

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False)

    def html(self) -> str:
        tpl = _TEMPLATE.read_text(encoding="utf-8")
        payload = self.to_json().replace("</", "<\\/")
        return tpl.replace("/*__CAPSCOPE_DATA__*/null", payload)

    def to_html(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.html(), encoding="utf-8")
        return str(p)

    # ---- terminal ---------------------------------------------------------
    def summary(self, top: Optional[int] = None, print_it: bool = True) -> str:
        rows = self.ranked()
        if top:
            rows = rows[:top]
        cols = ["#", "module", "type", "verdict", "prio", "width", "used", "used%", "press",
                "alpha", "cap", "params", "sentence"]
        table = []
        for i, m in enumerate(rows):
            used = m.get("used_rank")
            frac = m.get("used_frac")
            a = m.get("alpha")
            pr = m.get("pressure")
            table.append([
                str(m["exec_order"]), m["name"], m["type"], m["verdict"], f"{m['priority']:.2f}",
                str(m.get("width") or ""), "" if used is None else str(used),
                "" if frac is None else f"{100 * frac:.0f}",
                "" if pr is None else f"{100 * pr:.0f}%",
                "" if a is None or not math.isfinite(a) else f"{a:.1f}",
                (m.get("rank_cap") or "")[:3],
                _fmt_params(m["n_params_total"]),
                _ellipsis(m["sentence"], 70),
            ])
        widths = [max(len(c), *(len(r[j]) for r in table)) if table else len(c)
                  for j, c in enumerate(cols)]
        lines = []
        head = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
        lines.append(head)
        lines.append("-" * len(head))
        for r in table:
            lines.append("  ".join(v.ljust(w) for v, w in zip(r, widths)))
        lines.append("")
        lines.append(f"{len(self.modules)} modules, {len(self.edges)} dataflow edges, "
                     f"{_fmt_params(self.meta['n_params'])} parameters"
                     + (f", base loss {self.meta['base_loss']:.4f}" if self.meta.get("base_loss") is not None else ""))
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        out = "\n".join(lines)
        if print_it:
            print(out)
        return out

    # ---- GUI --------------------------------------------------------------
    def show(self, port: int = 0, host: str = "127.0.0.1", open_browser: bool = True,
             block: bool = True) -> str:
        """Serve the GUI on localhost and open a browser. Returns the URL."""
        html = self.html().encode("utf-8")
        payload = self.to_json().encode("utf-8")

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/data.json"):
                    body, ctype = payload, "application/json"
                else:
                    body, ctype = html, "text/html; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        srv = Server((host, port), Handler)
        url = f"http://{host}:{srv.server_address[1]}/"
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        self._server = srv
        print(f"[capscope] serving report at {url}  (Ctrl-C to stop)")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        if block:
            try:
                # join() with no timeout is uninterruptible on Windows: poll instead
                while t.is_alive():
                    t.join(0.5)
            except KeyboardInterrupt:
                print("[capscope] stopping")
            finally:
                srv.shutdown()
                srv.server_close()
        return url

    def close(self):
        srv = getattr(self, "_server", None)
        if srv is not None:
            srv.shutdown()
            srv.server_close()
            self._server = None


def _fmt_params(n: int) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return str(n)


def _ellipsis(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."

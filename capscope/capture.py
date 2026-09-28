"""Forward-hook activation capture and autograd-graph dataflow recovery.

Traps handled here (see README):

1. torch.fx / id(tensor) tracking fails on hand-written message passing.
   We walk the *autograd graph* backwards from each module's inputs through
   ``grad_fn.next_functions`` until we reach a registered producer.
2. ``id(grad_fn)`` is only stable while the Python wrapper is alive, so the
   registry keeps a strong reference to every registered ``grad_fn`` object.
   The whole bookkeeping is gated to a single grad-enabled batch and released
   right after.
6. The root module is never registered as a producer, so it cannot become a
   predecessor of everything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def iter_tensors(obj: Any, _depth: int = 0):
    """Yield every tensor reachable inside nested tuples/lists/dicts."""
    if _depth > 6:
        return
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            yield from iter_tensors(o, _depth + 1)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from iter_tensors(o, _depth + 1)
    elif _depth == 0 and hasattr(obj, "__dict__"):
        # e.g. a simple Batch/Data-like object with tensor attributes
        for o in vars(obj).values():
            if isinstance(o, torch.Tensor):
                yield o


def first_float_tensor(obj: Any) -> Optional[torch.Tensor]:
    for t in iter_tensors(obj):
        if t.is_floating_point() and t.dim() >= 1:
            return t
    return None


def flatten_rows(t: torch.Tensor) -> torch.Tensor:
    """[..., D] -> [N, D] (a 1-D tensor becomes [N, 1])."""
    if t.dim() == 0:
        return t.reshape(1, 1)
    if t.dim() == 1:
        return t.reshape(-1, 1)
    return t.reshape(-1, t.shape[-1])


def is_container(m: nn.Module) -> bool:
    return len(list(m.children())) > 0


def matrix_params(m: nn.Module, recurse: bool = True) -> List[Tuple[str, nn.Parameter]]:
    """Weight parameters with >= 2 dims (convs are viewed as [out, -1])."""
    out = []
    if recurse:
        it = m.named_parameters(recurse=True)
    else:
        it = ((n, p) for n, p in m._parameters.items() if p is not None)
    for n, p in it:
        if p.dim() >= 2:
            out.append((n, p))
    return out


def weight_as_matrix(p: torch.Tensor) -> torch.Tensor:
    return p.detach().reshape(p.shape[0], -1)


def is_ancestor(a: str, b: str) -> bool:
    """True if module ``a`` is a strict ancestor of ``b`` (by qualified name)."""
    return a != b and (a == "" or b.startswith(a + "."))


def is_relative(a: str, b: str) -> bool:
    return a == b or is_ancestor(a, b) or is_ancestor(b, a)


def _deepest(names: Sequence[str]) -> str:
    return max(names, key=lambda n: (n.count("."), len(n)))


# ----------------------------------------------------------------------------
# per-module capture record
# ----------------------------------------------------------------------------

@dataclass
class ModuleRecord:
    name: str
    module: nn.Module
    depth: int
    is_leaf: bool
    n_params_own: int
    n_params_total: int
    has_matrix: bool
    exec_order: int = -1
    n_calls: int = 0
    out_shape: Optional[Tuple[int, ...]] = None
    in_dim: Optional[int] = None
    width: Optional[int] = None
    # activation sample rows, per batch: list of [n_i, D] float32 cpu arrays
    samples: List[np.ndarray] = field(default_factory=list)
    sample_keys: List[Tuple[int, int]] = field(default_factory=list)  # (batch_idx, N)
    # graph metrics accumulated per batch
    dirichlet: List[float] = field(default_factory=list)
    mad: List[float] = field(default_factory=list)
    # dataflow: names of the modules whose outputs feed this module's inputs
    producers: Set[str] = field(default_factory=set)
    act_min: float = math.inf
    act_max: float = -math.inf

    @property
    def n_rows(self) -> int:
        return int(sum(s.shape[0] for s in self.samples))

    def activation_matrix(self) -> Optional[np.ndarray]:
        if not self.samples:
            return None
        return np.concatenate(self.samples, 0)


# ----------------------------------------------------------------------------
# capture
# ----------------------------------------------------------------------------

class Capture:
    """Runs the model over ``n_batches`` batches with forward hooks on every
    module (root included, but the root is never a graph node)."""

    def __init__(
        self,
        model: nn.Module,
        forward_fn: Callable[[nn.Module, Any], Any],
        edge_index_fn: Optional[Callable[[Any], torch.Tensor]] = None,
        max_rows: int = 4096,
        n_batches: int = 8,
        seed: int = 0,
    ):
        self.model = model
        self.forward_fn = forward_fn
        self.edge_index_fn = edge_index_fn
        self.max_rows = max_rows
        self.n_batches = max(1, n_batches)
        self.rows_per_batch = max(64, math.ceil(max_rows / self.n_batches))
        self.rng = np.random.RandomState(seed)

        self.records: Dict[str, ModuleRecord] = {}
        self.root_name = ""
        self.exec_counter = 0
        self.batches_seen = 0
        self._handles = []
        self._batch_idx = 0
        self._row_choice: Dict[Tuple[int, int], np.ndarray] = {}
        self._edge_index: Optional[torch.Tensor] = None
        self._n_nodes: Optional[int] = None

        # graph-batch bookkeeping (only alive during the grad-enabled batch)
        self._graph_mode = False
        self._registry: Dict[int, Tuple[Any, List[str]]] = {}
        self._pending_inputs: List[Tuple[str, List[Any]]] = []

        self._build_records()

    # ---- records ----------------------------------------------------------
    def _build_records(self):
        for name, m in self.model.named_modules():
            depth = 0 if name == "" else name.count(".") + 1
            own = sum(p.numel() for p in m.parameters(recurse=False))
            total = sum(p.numel() for p in m.parameters(recurse=True))
            self.records[name] = ModuleRecord(
                name=name, module=m, depth=depth, is_leaf=not is_container(m),
                n_params_own=own, n_params_total=total,
                has_matrix=len(matrix_params(m)) > 0,
            )

    # ---- hooks ------------------------------------------------------------
    def _make_hook(self, name: str):
        def hook(module, args, kwargs, output):
            self._on_forward(name, args, kwargs, output)
        return hook

    def _attach(self):
        for name, m in self.model.named_modules():
            self._handles.append(
                m.register_forward_hook(self._make_hook(name), with_kwargs=True))

    def _detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _rows_for(self, N: int) -> np.ndarray:
        key = (self._batch_idx, N)
        if key not in self._row_choice:
            if N <= self.rows_per_batch:
                idx = np.arange(N)
            else:
                idx = np.sort(self.rng.choice(N, self.rows_per_batch, replace=False))
            self._row_choice[key] = idx
        return self._row_choice[key]

    def _on_forward(self, name: str, args, kwargs, output):
        rec = self.records[name]
        if rec.exec_order < 0:
            rec.exec_order = self.exec_counter
            self.exec_counter += 1
        rec.n_calls += 1

        out_t = first_float_tensor(output)
        in_t = first_float_tensor(args)
        if in_t is None:
            in_t = first_float_tensor(kwargs)
        if in_t is not None and rec.in_dim is None:
            rec.in_dim = int(in_t.shape[-1])

        # --- graph-mode bookkeeping (traps 1 / 2 / 6) ---
        if self._graph_mode:
            if name != self.root_name:
                for t in iter_tensors(output):
                    gf = t.grad_fn
                    if gf is not None:
                        ent = self._registry.get(id(gf))
                        if ent is None:
                            self._registry[id(gf)] = (gf, [name])   # strong ref to gf
                        elif name not in ent[1]:
                            ent[1].append(name)
                gfs = []
                for t in list(iter_tensors(args)) + list(iter_tensors(kwargs)):
                    if t.grad_fn is not None:
                        gfs.append(t.grad_fn)
                self._pending_inputs.append((name, gfs))

        if out_t is None or name == self.root_name:
            return
        with torch.no_grad():
            X = flatten_rows(out_t.detach())
            if rec.out_shape is None:
                rec.out_shape = tuple(int(s) for s in out_t.shape)
                rec.width = int(X.shape[1])
            N = int(X.shape[0])
            idx = self._rows_for(N)
            Xs = X[torch.as_tensor(idx, device=X.device)].float().cpu().numpy()
            rec.samples.append(Xs)
            rec.sample_keys.append((self._batch_idx, N))
            if X.numel():
                rec.act_min = min(rec.act_min, float(X.min()))
                rec.act_max = max(rec.act_max, float(X.max()))
            if (self._edge_index is not None and out_t.dim() == 2
                    and self._n_nodes is not None and out_t.shape[0] == self._n_nodes):
                de, mad = graph_smoothness(out_t.detach(), self._edge_index)
                rec.dirichlet.append(de)
                rec.mad.append(mad)

    # ---- graph resolution -------------------------------------------------
    def _resolve_edges(self):
        """Walk backwards from every module's input grad_fns to the nearest
        registered producers (DFS through next_functions)."""
        reg = self._registry
        for name, gfs in self._pending_inputs:
            rec = self.records[name]
            for gf in gfs:
                seen: Set[int] = set()
                stack = [gf]
                while stack:
                    node = stack.pop()
                    nid = id(node)
                    if nid in seen:
                        continue
                    seen.add(nid)
                    ent = reg.get(nid)
                    if ent is not None:
                        # deepest registered module wins (a Sequential shares
                        # its grad_fn with its last child)
                        prods = [p for p in ent[1] if p != name and not is_ancestor(p, name)]
                        if prods:
                            rec.producers.add(_deepest(prods))
                            continue      # stop this path here
                    if hasattr(node, "variable"):
                        continue          # AccumulateGrad: a parameter leaf
                    for nxt, _ in getattr(node, "next_functions", ()):
                        if nxt is not None and id(nxt) not in seen:
                            stack.append(nxt)
        # release bookkeeping (trap 2: never keep the activation history alive)
        self._registry = {}
        self._pending_inputs = []

    # ---- driver -----------------------------------------------------------
    def run(self, loader) -> "Capture":
        model = self.model
        was_training = model.training
        model.eval()
        req = {n: p.requires_grad for n, p in model.named_parameters()}
        self._attach()
        try:
            for b_idx, batch in enumerate(loader):
                if b_idx >= self.n_batches:
                    break
                self._batch_idx = b_idx
                self.batches_seen = b_idx + 1
                self._set_graph_context(batch)
                if b_idx == 0:
                    # the single grad-enabled batch used for dataflow recovery
                    for p in model.parameters():
                        p.requires_grad_(True)
                    self._graph_mode = True
                    with torch.enable_grad():
                        self.forward_fn(model, batch)
                    self._graph_mode = False
                    self._resolve_edges()
                    for n, p in model.named_parameters():
                        p.requires_grad_(req[n])
                else:
                    with torch.no_grad():
                        self.forward_fn(model, batch)
        finally:
            self._detach()
            model.train(was_training)
            self._graph_mode = False
            self._registry = {}
            self._pending_inputs = []
        return self

    def _set_graph_context(self, batch):
        self._edge_index = None
        self._n_nodes = None
        if self.edge_index_fn is None:
            return
        ei = self.edge_index_fn(batch)
        if ei is None:
            return
        ei = torch.as_tensor(ei)
        if ei.dim() == 2 and ei.shape[0] != 2 and ei.shape[1] == 2:
            ei = ei.t()
        if ei.dim() != 2 or ei.shape[0] != 2 or ei.numel() == 0:
            return
        self._edge_index = ei.long()
        self._n_nodes = int(ei.max().item()) + 1
        # prefer the batch's own node-feature row count if it is consistent
        x = first_float_tensor(batch)
        if x is not None and x.dim() == 2 and x.shape[0] >= self._n_nodes:
            self._n_nodes = int(x.shape[0])

    # ---- views ------------------------------------------------------------
    def executed(self) -> List[ModuleRecord]:
        return sorted((r for r in self.records.values()
                       if r.exec_order >= 0 and r.name != self.root_name),
                      key=lambda r: r.exec_order)

    def leaf_edges(self) -> List[Tuple[str, str]]:
        """Dataflow edges between *leaf* modules. Container-level edges are
        derived by projecting these onto ancestors (never onto the root)."""
        edges = set()
        for r in self.records.values():
            if r.name == self.root_name or not r.is_leaf:
                continue
            for p in r.producers:
                if p != r.name and p != self.root_name and self.records[p].is_leaf:
                    edges.add((p, r.name))
        return sorted(edges)


# ----------------------------------------------------------------------------
# graph smoothness metrics
# ----------------------------------------------------------------------------

@torch.no_grad()
def graph_smoothness(X: torch.Tensor, edge_index: torch.Tensor) -> Tuple[float, float]:
    """Return ``(dirichlet, mad)`` for node features ``X`` [N, D].

    dirichlet: mean over edges of ||x_i - x_j||^2 divided by the mean over
      nodes of ||x_i - mean(x)||^2, on *centred* features. Random i.i.d.
      features give ~2, perfectly smoothed features give 0. Centring removes
      the common DC offset that makes un-centred normalised energy look
      small after every ReLU (trap 5). The verdict engine still only uses the
      *decay* relative to the nearest upstream graph layer.
    mad: Mean Average Distance (cosine distance to neighbours, raw features).
      Reported for completeness; it collapses whenever features share a sign.
    """
    ei = edge_index.to(X.device)
    src, dst = ei[0], ei[1]
    n = X.shape[0]
    mask = (src < n) & (dst < n)
    src, dst = src[mask], dst[mask]
    if src.numel() == 0:
        return float("nan"), float("nan")
    Xf = X.float()
    Xc = Xf - Xf.mean(0, keepdim=True)
    denom = (Xc * Xc).sum(1).mean().clamp_min(1e-12)
    diff = Xc[src] - Xc[dst]
    dirichlet = float((diff * diff).sum(1).mean() / denom)
    Xn = Xf / Xf.norm(dim=1, keepdim=True).clamp_min(1e-12)
    d = 1.0 - (Xn[src] * Xn[dst]).sum(1)
    per_node = torch.zeros(n, device=X.device).index_add_(0, dst, d)
    cnt = torch.zeros(n, device=X.device).index_add_(0, dst, torch.ones_like(d))
    has = cnt > 0
    mad = float((per_node[has] / cnt[has]).mean()) if bool(has.any()) else float("nan")
    return dirichlet, mad

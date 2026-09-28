"""SVD truncation sweep: replace each weight by its rank-k approximation,
sweep k on a log grid, measure loss, restore. The smallest k within
``rel_tol`` of the base loss is the rank the module actually uses."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .capture import matrix_params


def log_grid(rmax: int, n: int = 10) -> List[int]:
    if rmax <= 1:
        return [1]
    g = np.unique(np.round(np.geomspace(1, rmax, num=n)).astype(int))
    g = [int(k) for k in g if 1 <= k <= rmax]
    if g[-1] != rmax:
        g.append(rmax)
    return g


class _SVDCache:
    def __init__(self, params: Sequence[nn.Parameter]):
        self.params = list(params)
        self.orig = [p.detach().clone() for p in self.params]
        self.svd = []
        for p in self.params:
            M = p.detach().reshape(p.shape[0], -1).to(torch.float64)
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            self.svd.append((U, S, Vh))

    @property
    def caps(self) -> List[int]:
        return [int(S.shape[0]) for _, S, _ in self.svd]

    @torch.no_grad()
    def set_rank(self, k: int):
        for p, (U, S, Vh) in zip(self.params, self.svd):
            kk = min(k, S.shape[0])
            M = (U[:, :kk] * S[:kk]) @ Vh[:kk]
            p.copy_(M.reshape(p.shape).to(p.dtype))

    @torch.no_grad()
    def restore(self):
        for p, o in zip(self.params, self.orig):
            p.copy_(o)


@torch.no_grad()
def _eval_loss(model, batches, forward_fn, loss_fn) -> float:
    tot = 0.0
    for b in batches:
        out = forward_fn(model, b)
        l = loss_fn(out, b)
        tot += float(l.detach().item() if isinstance(l, torch.Tensor) else l)
    return tot / max(1, len(batches))


def truncation_sweep(
    model: nn.Module,
    module: nn.Module,
    batches: List[Any],
    forward_fn: Callable,
    loss_fn: Callable,
    base_loss: Optional[float] = None,
    rel_tol: float = 0.01,
    n_grid: int = 10,
    refine: bool = True,
) -> Optional[Dict]:
    """Sweep the rank of every matrix weight inside ``module`` jointly.

    Returns a dict with the (k, loss) curve, base loss, ``used_rank`` and
    ``max_rank``; None if the module has no matrix weights.
    """
    params = [p for _, p in matrix_params(module, recurse=True)]
    if not params:
        return None
    was_training = model.training
    model.eval()
    cache = _SVDCache(params)
    try:
        if base_loss is None:
            base_loss = _eval_loss(model, batches, forward_fn, loss_fn)
        rmax = max(cache.caps)
        thresh = base_loss + rel_tol * abs(base_loss) + 1e-12
        grid = log_grid(rmax, n_grid)
        curve: Dict[int, float] = {}
        for k in grid:
            cache.set_rank(k)
            curve[k] = _eval_loss(model, batches, forward_fn, loss_fn)
        curve[rmax] = curve.get(rmax, base_loss)
        ok = [k for k in sorted(curve) if curve[k] <= thresh]
        used = ok[0] if ok else rmax
        if refine and ok:
            lo = max([k for k in sorted(curve) if k < used and curve[k] > thresh], default=None)
            if lo is not None:
                while used - lo > 1:
                    mid = (lo + used) // 2
                    cache.set_rank(mid)
                    curve[mid] = _eval_loss(model, batches, forward_fn, loss_fn)
                    if curve[mid] <= thresh:
                        used = mid
                    else:
                        lo = mid
    finally:
        cache.restore()
        model.train(was_training)
    ks = sorted(curve)
    # marginal pressure: relative loss increase when the module loses its
    # single last used dimension (the bisection always evaluates used-1).
    pressure = None
    if used > 1 and (used - 1) in curve and abs(base_loss) > 0:
        pressure = max(0.0, (curve[used - 1] - base_loss) / abs(base_loss))
    return {
        "ks": ks,
        "losses": [float(curve[k]) for k in ks],
        "base_loss": float(base_loss),
        "threshold": float(thresh),
        "used_rank": int(used),
        "max_rank": int(rmax),
        "rel_tol": rel_tol,
        "loss_at_1": float(curve.get(1, float("nan"))),
        "pressure": pressure,
    }

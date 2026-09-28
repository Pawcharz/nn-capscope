"""Data-free and activation-based metrics: rank/saturation, weight spectra,
redundancy (CKA, dead and duplicate units)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

EPS32 = float(np.finfo(np.float32).eps)


# ----------------------------------------------------------------------------
# 1. rank / saturation from activations [N, D]
# ----------------------------------------------------------------------------

def rank_metrics(X: np.ndarray, center: bool = True, max_sv: int = 256) -> Dict:
    """Singular-value based capacity metrics for an activation matrix."""
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    if center and N > 1:
        X = X - X.mean(0, keepdims=True)
    if N == 0 or D == 0:
        return _empty_rank()
    try:
        s = np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        s = np.sqrt(np.maximum(np.linalg.eigvalsh(X.T @ X)[::-1], 0))
    s = np.sort(s)[::-1]
    s = s[: min(len(s), D)]
    tot = float(s.sum())
    if tot <= 0 or not np.isfinite(tot):
        return _empty_rank(D=D, n_sv=len(s))
    p = s / tot
    p = p[p > 0]
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    lam = s ** 2
    stable_rank = float(lam.sum() / lam[0]) if lam[0] > 0 else 0.0
    part_ratio = float(lam.sum() ** 2 / (lam ** 2).sum()) if (lam ** 2).sum() > 0 else 0.0
    tol = s[0] * max(N, D) * EPS32
    num_rank = int((s > tol).sum())
    cum = np.cumsum(lam) / lam.sum()
    r90 = int(np.searchsorted(cum, 0.90) + 1)
    r99 = int(np.searchsorted(cum, 0.99) + 1)
    return {
        "n_rows": int(N),
        "n_sv": int(len(s)),
        "effective_rank": eff_rank,
        "stable_rank": stable_rank,
        "participation_ratio": part_ratio,
        "numerical_rank": num_rank,
        "rank90": min(r90, len(s)),
        "rank99": min(r99, len(s)),
        "singular_values": [float(v) for v in s[:max_sv]],
        "cum_var": [float(v) for v in cum[:max_sv]],
    }


def _empty_rank(D: int = 0, n_sv: int = 0) -> Dict:
    return {"n_rows": 0, "n_sv": n_sv, "effective_rank": 0.0, "stable_rank": 0.0,
            "participation_ratio": 0.0, "numerical_rank": 0, "rank90": 0, "rank99": 0,
            "singular_values": [], "cum_var": []}


def activation_histogram(X: np.ndarray, bins: int = 40, max_vals: int = 200_000) -> Dict:
    v = np.asarray(X, dtype=np.float64).ravel()
    if v.size > max_vals:
        v = np.random.RandomState(0).choice(v, max_vals, replace=False)
    if v.size == 0:
        return {"edges": [], "counts": []}
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo + 1e-6
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts],
            "mean": float(v.mean()), "std": float(v.std()),
            "zero_frac": float((np.abs(v) < 1e-8).mean())}


# ----------------------------------------------------------------------------
# 2. weight spectra (no data needed)
# ----------------------------------------------------------------------------

def hill_alpha(eigs: np.ndarray, k: Optional[int] = None) -> Tuple[float, int]:
    """Hill estimator of the power-law tail of eigenvalues of W^T W.

    alpha = 1 + k / sum_{i<=k} log(lambda_i / lambda_{k+1}),  k ~ n/10.
    Returns (alpha, k); alpha is nan when the spectrum is too short.
    """
    lam = np.sort(np.asarray(eigs, dtype=np.float64))[::-1]
    lam = lam[lam > 0]
    n = len(lam)
    if n < 2:
        return float("nan"), 0
    if k is None:
        k = max(1, n // 10)
    k = min(k, n - 1)
    ref = lam[k]
    logs = np.log(lam[:k] / ref)
    s = float(logs.sum())
    if s <= 0:
        return float("inf"), k
    return float(1.0 + k / s), k


def weight_spectrum(W: torch.Tensor, max_sv: int = 256) -> Dict:
    """Spectrum of one weight matrix [out, in]."""
    M = W.detach().reshape(W.shape[0], -1).to(torch.float64).cpu().numpy()
    out_f, in_f = M.shape
    s = np.linalg.svd(M, compute_uv=False)
    s = np.sort(s)[::-1]
    lam = s ** 2
    alpha, k = hill_alpha(lam)
    if in_f < out_f:
        cap = "input"
    elif out_f < in_f:
        cap = "output"
    else:
        cap = "square"
    tol = s[0] * max(out_f, in_f) * EPS32 if len(s) and s[0] > 0 else 0.0
    return {
        "shape": [int(out_f), int(in_f)],
        "max_rank": int(min(out_f, in_f)),
        "rank_cap": cap,
        "alpha": alpha,
        "alpha_k": int(k),
        "numerical_rank": int((s > tol).sum()),
        "stable_rank": float(lam.sum() / lam[0]) if len(lam) and lam[0] > 0 else 0.0,
        "spectral_norm": float(s[0]) if len(s) else 0.0,
        "frob_norm": float(np.sqrt(lam.sum())),
        "singular_values": [float(v) for v in s[:max_sv]],
    }


def alpha_reading(alpha: float) -> str:
    if alpha is None or not np.isfinite(alpha):
        return "n/a"
    if alpha > 6:
        return "undertrained"
    if alpha < 2:
        return "over-trained"
    return "well-conditioned"


# ----------------------------------------------------------------------------
# 4. redundancy
# ----------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two activation matrices with identical rows."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y) ** 2
    xx = np.linalg.norm(X.T @ X)
    yy = np.linalg.norm(Y.T @ Y)
    if xx <= 0 or yy <= 0:
        return float("nan")
    return float(xy / (xx * yy))


def cka_matrix(mats: List[Optional[np.ndarray]], keys: List[Optional[tuple]]) -> np.ndarray:
    """Pairwise linear CKA; NaN where the row sets are not aligned."""
    n = len(mats)
    C = np.full((n, n), np.nan)
    # precompute centred matrices
    cen = []
    for M in mats:
        if M is None or M.size == 0:
            cen.append(None)
        else:
            M = np.asarray(M, dtype=np.float64)
            cen.append(M - M.mean(0, keepdims=True))
    gram_norm = [None if M is None else np.linalg.norm(M.T @ M) for M in cen]
    for i in range(n):
        if cen[i] is None or gram_norm[i] is None or gram_norm[i] <= 0:
            continue
        C[i, i] = 1.0
        for j in range(i + 1, n):
            if cen[j] is None or keys[i] != keys[j] or gram_norm[j] is None or gram_norm[j] <= 0:
                continue
            if cen[i].shape[0] != cen[j].shape[0]:
                continue
            v = np.linalg.norm(cen[i].T @ cen[j]) ** 2 / (gram_norm[i] * gram_norm[j])
            C[i, j] = C[j, i] = float(v)
    return C


def unit_redundancy(X: np.ndarray, dup_thresh: float = 0.95, max_rows: int = 4096) -> Dict:
    """Dead units (near-zero variance) and near-duplicate units (|corr| > thr)."""
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    if N > max_rows:
        X = X[np.random.RandomState(0).choice(N, max_rows, replace=False)]
    if N < 2 or D == 0:
        return {"dead": 0, "dup": 0, "dead_frac": 0.0, "dup_frac": 0.0, "redundant_frac": 0.0,
                "dead_units": [], "dup_pairs": []}
    var = X.var(0)
    ref = max(float(var.mean()), 1e-12)
    dead_mask = (var <= 1e-6 * ref) | (var < 1e-12)
    dead = np.where(dead_mask)[0]
    alive = np.where(~dead_mask)[0]
    dup_pairs: List[Tuple[int, int, float]] = []
    dup_units = set()
    if len(alive) >= 2:
        Z = X[:, alive]
        Z = (Z - Z.mean(0)) / np.sqrt(var[alive])
        C = (Z.T @ Z) / (Z.shape[0] - 1)
        np.fill_diagonal(C, 0.0)
        iu = np.triu_indices(len(alive), 1)
        hits = np.where(np.abs(C[iu]) > dup_thresh)[0]
        for h in hits:
            a, b = alive[iu[0][h]], alive[iu[1][h]]
            dup_pairs.append((int(a), int(b), float(C[iu[0][h], iu[1][h]])))
            dup_units.add(int(b))   # the later unit counts as the duplicate
    n_dead, n_dup = int(len(dead)), int(len(dup_units))
    return {
        "dead": n_dead, "dup": n_dup,
        "dead_frac": n_dead / D, "dup_frac": n_dup / D,
        "redundant_frac": (n_dead + n_dup) / D,
        "dead_units": [int(i) for i in dead[:64]],
        "dup_pairs": [list(p) for p in dup_pairs[:64]],
    }

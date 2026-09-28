"""Test harness model: hand-written mean-aggregation message passing (no PyG),
4 heads, deliberately mismatched widths.

Designed ground truth
---------------------
encoder      Linear(8 -> 96) + ReLU : over-wide relay of an 8-dim input  -> upstream
mp1..mp3     SAGE-style layers (64 wide) doing the real work
norm         LayerNorm in between (1-D params only)
bottleneck   Linear(64 -> 16) followed by a *functional* relu             -> saturated, top priority
heads        4 x Linear(16 -> {1,1,3,6})                                  -> narrow
nn.ReLU modules (encoder.1, mp*.act)                                     -> passthrough

The target task needs many independent latent directions (own features,
1-hop and 2-hop neighbour means, non-linearly mixed), so a 16-wide bottleneck
is genuinely full when the model is trained to convergence.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_DIM = 8
ENC_DIM = 96
HID = 64
BOTTLENECK = 16
HEAD_DIMS = {"a": 1, "b": 1, "c": 3, "d": 6}


def mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Hand-written mean aggregation over incoming edges (src -> dst)."""
    src, dst = edge_index[0], edge_index[1]
    agg = torch.zeros_like(x).index_add_(0, dst, x[src])
    deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype).index_add_(
        0, dst, torch.ones(src.shape[0], device=x.device, dtype=x.dtype))
    return agg / deg.clamp(min=1).unsqueeze(1)


class SAGELayer(nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.lin_self = nn.Linear(din, dout)
        self.lin_neigh = nn.Linear(din, dout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        agg = mean_aggregate(x, edge_index)
        return self.act(self.lin_self(x) + self.lin_neigh(agg))


class ToyGraphNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(IN_DIM, ENC_DIM), nn.ReLU())
        self.mp1 = SAGELayer(ENC_DIM, HID)
        self.mp2 = SAGELayer(HID, HID)
        self.norm = nn.LayerNorm(HID)
        self.mp3 = SAGELayer(HID, HID)
        self.bottleneck = nn.Linear(HID, BOTTLENECK)
        self.heads = nn.ModuleDict({k: nn.Linear(BOTTLENECK, d) for k, d in HEAD_DIMS.items()})

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x, ei = batch["x"], batch["edge_index"]
        h = self.encoder(x)
        h = self.mp1(h, ei)
        h = self.mp2(h, ei)
        h = self.norm(h)
        h = self.mp3(h, ei)
        z = F.relu(self.bottleneck(h))          # functional op: breaks id()-tracking, not autograd
        return {k: head(z) for k, head in self.heads.items()}


# The true leaf-level dataflow graph of ToyGraphNet.forward
TRUE_EDGES = sorted({
    ("encoder.0", "encoder.1"),
    ("encoder.1", "mp1.lin_self"), ("encoder.1", "mp1.lin_neigh"),
    ("mp1.lin_self", "mp1.act"), ("mp1.lin_neigh", "mp1.act"),
    ("mp1.act", "mp2.lin_self"), ("mp1.act", "mp2.lin_neigh"),
    ("mp2.lin_self", "mp2.act"), ("mp2.lin_neigh", "mp2.act"),
    ("mp2.act", "norm"),
    ("norm", "mp3.lin_self"), ("norm", "mp3.lin_neigh"),
    ("mp3.lin_self", "mp3.act"), ("mp3.lin_neigh", "mp3.act"),
    ("mp3.act", "bottleneck"),
    ("bottleneck", "heads.a"), ("bottleneck", "heads.b"),
    ("bottleneck", "heads.c"), ("bottleneck", "heads.d"),
})


# ----------------------------------------------------------------------------
# synthetic task
# ----------------------------------------------------------------------------

class TaskParams:
    """Fixed random projections that define the targets (shared by all graphs)."""

    def __init__(self, seed: int = 123):
        g = torch.Generator().manual_seed(seed)
        z_dim = 3 * IN_DIM                       # own, 1-hop mean, 2-hop mean
        self.W_d = torch.randn(z_dim, HEAD_DIMS["d"], generator=g) / math.sqrt(z_dim)
        self.W_c = torch.randn(z_dim, HEAD_DIMS["c"], generator=g) / math.sqrt(z_dim)
        self.w_a = torch.randn(z_dim, 4, generator=g) / math.sqrt(z_dim)
        self.w_b = torch.randn(z_dim, 4, generator=g) / math.sqrt(z_dim)
        self.V = torch.randn(z_dim, z_dim, generator=g) / math.sqrt(z_dim)


def make_graph(n_nodes: int, avg_deg: float, params: TaskParams, seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    n_edges = int(n_nodes * avg_deg)
    src = torch.randint(0, n_nodes, (n_edges,), generator=g)
    dst = torch.randint(0, n_nodes, (n_edges,), generator=g)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])   # symmetric
    x = torch.randn(n_nodes, IN_DIM, generator=g)
    m1 = mean_aggregate(x, ei)
    m2 = mean_aggregate(m1, ei)
    z = torch.cat([x, m1 * 2.0, m2 * 3.0], 1)                # rescale so hops matter
    zn = torch.tanh(z @ params.V) + z                          # non-linear mixing
    y_d = torch.tanh(zn @ params.W_d * 2.0)
    y_c = (zn @ params.W_c).argmax(1)
    pa = zn @ params.w_a
    y_a = (torch.sin(2.0 * pa[:, 0]) * pa[:, 1] + pa[:, 2] * pa[:, 3]).unsqueeze(1)
    pb = zn @ params.w_b
    y_b = (pb[:, 0] ** 2 - pb[:, 1] * pb[:, 2] + torch.cos(2.0 * pb[:, 3])).unsqueeze(1)
    return {"x": x, "edge_index": ei, "y_a": y_a, "y_b": y_b, "y_c": y_c, "y_d": y_d}


def make_loader(n_graphs: int = 12, n_nodes: int = 1500, avg_deg: float = 5.0, seed: int = 0) -> List[dict]:
    params = TaskParams()
    return [make_graph(n_nodes, avg_deg, params, seed * 1000 + i) for i in range(n_graphs)]


def loss_fn(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return (F.mse_loss(out["a"], batch["y_a"])
            + F.mse_loss(out["b"], batch["y_b"])
            + F.cross_entropy(out["c"], batch["y_c"])
            + 4.0 * F.mse_loss(out["d"], batch["y_d"]))


def baseline_loss(batches: List[dict]) -> float:
    """Loss of predicting the training mean / class prior (a 'learned nothing' reference)."""
    ya = torch.cat([b["y_a"] for b in batches]); yb = torch.cat([b["y_b"] for b in batches])
    yc = torch.cat([b["y_c"] for b in batches]); yd = torch.cat([b["y_d"] for b in batches])
    prior = torch.bincount(yc, minlength=HEAD_DIMS["c"]).float() / len(yc)
    out = {"a": ya.mean(0, keepdim=True).expand_as(ya), "b": yb.mean(0, keepdim=True).expand_as(yb),
           "c": prior.log().unsqueeze(0).expand(len(yc), -1), "d": yd.mean(0, keepdim=True).expand_as(yd)}
    return float(loss_fn(out, {"y_a": ya, "y_b": yb, "y_c": yc, "y_d": yd}))


def forward_fn(model, batch):
    return model(batch)


def edge_index_fn(batch):
    return batch["edge_index"]


# ----------------------------------------------------------------------------
# training to genuine convergence
# ----------------------------------------------------------------------------

def train(model: nn.Module, batches: List[dict], max_epochs: int = 1500, lr: float = 3e-3,
          patience: int = 150, verbose: bool = True) -> Tuple[float, int]:
    """Full-batch Adam over all graphs with plateau-based LR decay. Stops when
    the loss has not improved by >0.2% for ``patience`` epochs at the lowest LR."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=patience // 3,
                                                       threshold=2e-3, min_lr=lr / 64)
    model.train()
    best, best_epoch = float("inf"), 0
    for epoch in range(max_epochs):
        tot = 0.0
        for b in batches:
            opt.zero_grad()
            loss = loss_fn(model(b), b)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        tot /= len(batches)
        sched.step(tot)
        if tot < best * (1 - 2e-3):
            best, best_epoch = tot, epoch
        elif epoch - best_epoch > patience and opt.param_groups[0]["lr"] <= lr / 64 + 1e-12:
            break
        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch:5d}  loss {tot:.4f}  lr {opt.param_groups[0]['lr']:.2e}")
    model.eval()
    return best, epoch


def get_trained(cache_dir: Path = Path(__file__).parent / "_cache", seed: int = 0,
                verbose: bool = True) -> Tuple[nn.Module, List[dict]]:
    """Trained model + loader, cached on disk so the test suite stays fast."""
    cache_dir.mkdir(exist_ok=True)
    ck = cache_dir / f"toy_seed{seed}.pt"
    torch.manual_seed(seed)
    batches = make_loader(seed=seed)
    model = ToyGraphNet()
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        model.eval()
        return model, batches
    if verbose:
        print(f"  baseline (predict-mean) loss {baseline_loss(batches):.4f}")
    best, ep = train(model, batches, verbose=verbose)
    if verbose:
        print(f"  converged: loss {best:.4f} after {ep} epochs")
    torch.save(model.state_dict(), ck)
    return model, batches


def build_model():
    """CLI factory: ``capscope tests/toy_model.py:build_model``."""
    model, batches = get_trained(verbose=False)
    return {"model": model, "loader": batches, "forward_fn": forward_fn,
            "loss_fn": loss_fn, "edge_index_fn": edge_index_fn, "n_batches": 8}


if __name__ == "__main__":
    m, b = get_trained()
    print("train loss", float(sum(loss_fn(m(x), x) for x in b) / len(b)))

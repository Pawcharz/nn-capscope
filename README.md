# capscope

Diagnose which parts of a trained PyTorch network have run out of
representational capacity and which still have room, so you can decide where
to spend parameters when scaling up. Nothing is retrained: every diagnostic
comes from forward passes over a frozen checkpoint plus SVD on the existing
weights.

Built for plain PyTorch with hand-written message passing (no PyG / DGL
needed, but nothing stops you using them).

## Install

```bash
uv sync --all-extras        # or: pip install -e .[test]
```

Runtime dependencies are only `torch` and `numpy`. Tests additionally use
`pytest` and `playwright` (Playwright drives an already-installed Edge/Chrome,
no browser download is needed).

## Usage

```python
from capscope import inspect

report = inspect(
    model, loader,
    forward_fn=lambda m, b: m(b),            # how to call the model on a batch
    loss_fn=my_loss,                          # optional: loss_fn(output, batch) -> scalar; enables the truncation sweep
    edge_index_fn=lambda b: b["edge_index"],  # optional: LongTensor [2, E]; enables graph metrics
    n_batches=8,
)
report.show()          # serve the GUI on localhost and open a browser
report.to_html(path)   # save a self-contained HTML file
report.summary()       # terminal table
report.ranked()        # list of module dicts sorted by growth priority
report["mp2.lin_self"] # every metric for one module
```

CLI (launching the GUI is the default):

```bash
capscope path/to/file.py:build_model            # GUI
capscope path/to/file.py:build_model --html out.html --no-gui --summary
```

The factory may return a bare model (data-free mode: weight spectra only), a
`(model, loader)` tuple, or a dict with any of `model, loader, forward_fn,
loss_fn, edge_index_fn, n_batches`. See `tests/toy_model.py:build_model`.

## What it measures

Per module (leaves *and* containers, root excluded), from activations
flattened to `[N, D]`:

1. **Rank / saturation**: effective rank `exp(H(σ/Σσ))`, stable rank
   `Σσ²/σ_max²`, participation ratio, numerical rank, cumulative
   variance-explained curve, rank at 90 % and 99 % energy.
2. **Weight spectra** (no data needed): Hill estimator of the power-law tail
   of the eigenvalues of `WᵀW`, `alpha = 1 + k / Σ log(λᵢ/λ_{k+1})`, `k ≈ n/10`.
   `alpha > 6` ≈ random / undertrained, `2–6` well conditioned, `< 2`
   over-trained. Each weight is also tagged **input-capped** or
   **output-capped** (whether `min(in, out)` is the input or the output).
3. **Graph / oversmoothing**: Dirichlet energy (centred, normalised: random
   features ≈ 2, smoothed → 0) and MAD, for every module whose output is
   node-aligned with `edge_index`.
4. **Redundancy**: pairwise linear CKA across all captured modules, dead units
   (near-zero variance) and near-duplicate units (|corr| > 0.95).
5. **SVD truncation sweep** (needs `loss_fn`): every matrix weight of the
   module is replaced by its rank-k approximation, k swept on a log grid and
   bisected, loss measured, weights restored. The smallest k within 1 % of the
   base loss is the *used rank*; the relative loss increase when the module
   loses that last dimension is its *marginal pressure*.

## Verdicts

First match wins:

| verdict | rule |
|---|---|
| `narrow` | width < 8 (an output head); capacity metrics do not apply |
| `passthrough` | leaf with no parameters (ReLU, Dropout…); points at the module feeding it |
| `oversmoothed` | Dirichlet energy < 0.05 **and** < 30 % of the nearest upstream graph layer |
| `upstream` | its rank merely tracks the upstream *carry* (the narrowest point along the paths feeding it) |
| `undertrained` | alpha > 6 |
| `redundant` | > 35 % dead or duplicate units, or CKA > 0.98 with a module that is neither a relative nor downstream |
| `saturated` / `tight` / `spare` | used rank > 85 % / > 60 % / otherwise of the width |

Growth priority (0–1) rises with saturation and marginal pressure, is
discounted for input-capped weights (widen upstream instead) and for output
layers.

## Traps this tool is built around

* The dataflow graph comes from walking the **autograd graph** backwards from
  each module's inputs through `grad_fn.next_functions` to the nearest
  registered producer. `torch.fx` and `id(tensor)` tracking both break on
  functional ops (`F.relu`, `index_add_`, scatter).
* `id(grad_fn)` is only stable while the Python wrapper is alive: the registry
  holds strong references to the `grad_fn` objects, is populated during a
  single grad-enabled batch and released immediately afterwards.
* A nonlinearity inflates measured rank; the *upstream carry* propagates the
  narrowest width along each path and flags modules whose rank merely tracks
  it. Whether a weight's rank cap comes from the input or the output decides
  whether "uses its full rank" is evidence to widen here or upstream.
* Oversmoothing is judged as **decay** relative to the nearest upstream graph
  layer, not as an absolute level; energies are computed on centred features
  so a DC offset does not fake smoothing. MAD is shown but not trusted.
* Container-level edges are projections of leaf edges onto ancestors; the
  root is never a node.
* An undertrained checkpoint reports `used_rank = 1` everywhere (destroying
  weights costs nothing). capscope detects this and warns in the report
  header instead of reporting nonsense.

## GUI

Served HTML with hand-rolled SVG and vanilla JS: no extra dependencies and the
saved HTML works offline.

* **Dataflow graph**: layered DAG of the real runtime graph, nodes coloured
  by verdict, click to select, scroll-zoom, drag-pan, collapse/expand
  containers (± buttons, double-click, Expand/Collapse all), leaves-only
  toggle.
* **Module list**: sortable by growth priority, execution order, dimensions
  used, parameter count; capacity bar per row; text filter.
* **Detail panel**: verdict sentence, metric grid, activation and weight
  singular-value spectra, variance-explained curve, loss-under-truncation
  sweep, alpha gauge with the < 2 / 2–6 / > 6 zones, activation histogram,
  clickable nearest CKA neighbours, producers / consumers.
* **Similarity view**: full CKA heatmap (dark squares = redundant blocks),
  hover for values, click to select.

## Tests

```bash
uv run pytest
```

`tests/toy_model.py` is a hand-written mean-aggregation SAGE stack with four
heads and deliberately mismatched widths (an over-wide encoder relaying an
8-dim input, three 64-wide message-passing layers, a LayerNorm, a 16-wide
bottleneck, heads of width 1/1/3/6). It is trained to convergence on a
synthetic task on first run (a few minutes on CPU, cached under
`tests/_cache/`). The suite asserts the designed ground truth, the recovered
graph edge for edge, that no container is flagged redundant against its own
child, that an untrained checkpoint triggers the warning, and that the HTML
renders in a real browser with zero console errors.

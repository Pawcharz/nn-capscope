"""Quick manual run: python tests/smoke.py [out.html]"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from toy_model import get_trained, forward_fn, loss_fn, edge_index_fn, TRUE_EDGES  # noqa
from capscope import inspect  # noqa

model, batches = get_trained()
t0 = time.time()
rep = inspect(model, batches, forward_fn=forward_fn, loss_fn=loss_fn, edge_index_fn=edge_index_fn,
              n_batches=8, verbose=True)
print(f"inspect took {time.time() - t0:.1f}s")
rep.summary()
got = set(rep.edges)
true = set(TRUE_EDGES)
print("\nmissing edges:", sorted(true - got))
print("extra edges:  ", sorted(got - true))
for m in rep.modules:
    print(f"{m['name']:16s} carry={m.get('carry')} width={m.get('width')} in={m.get('in_dim')} "
          f"used={m.get('used_rank')} wcap={m.get('max_weight_rank')} cap={m.get('rank_cap')} "
          f"alpha={m.get('alpha')} dir={m.get('dirichlet')} mad={m.get('mad')} "
          f"red={(m.get('redundancy') or {}).get('redundant_frac')} cka={m.get('cka_partner')}:{m.get('cka_partner_value')}")
if len(sys.argv) > 1:
    print("wrote", rep.to_html(sys.argv[1]))

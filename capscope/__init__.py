"""capscope: diagnose which parts of a trained network have run out of
representational capacity and which still have room.

    from capscope import inspect
    report = inspect(model, loader, forward_fn=..., loss_fn=..., edge_index_fn=..., n_batches=8)
    report.show()        # GUI on localhost
    report.to_html(path) # self-contained HTML
    report.summary()     # terminal table
"""
from .report import Report, inspect
from .verdict import THRESH, VERDICT_ORDER

__all__ = ["inspect", "Report", "THRESH", "VERDICT_ORDER"]
__version__ = "0.1.0"

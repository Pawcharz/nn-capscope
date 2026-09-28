"""Command line entry point.

    capscope path/to/file.py:build_model [--html out.html] [--json data.json]
                                        [--export report.md|report.json|runs.jsonl|-]
                                        [--port 8765] [--no-open] [--summary] [--n-batches 8]
    capscope saved_report.html            # reopen a saved report: no model, no recompute

The factory may return:
  * a model                      -> data-free diagnostics (weight spectra only)
  * (model, loader)              -> activation + graph diagnostics
  * a dict with any of the keys  model, loader, forward_fn, loss_fn, edge_index_fn, n_batches
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

from .report import Report, inspect


def load_factory(spec: str):
    if ":" not in spec:
        raise SystemExit(f"expected 'path/to/file.py:factory' or 'package.module:factory', got {spec!r}")
    target, attr = spec.rsplit(":", 1)
    if target.endswith(".py") or "/" in target or "\\" in target:
        path = Path(target).resolve()
        if not path.exists():
            raise SystemExit(f"file not found: {path}")
        sys.path.insert(0, str(path.parent))
        s = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    try:
        return getattr(mod, attr)
    except AttributeError:
        raise SystemExit(f"{target} has no attribute {attr!r}")


def _normalise(obj) -> dict:
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, (tuple, list)):
        keys = ["model", "loader", "forward_fn", "loss_fn", "edge_index_fn"]
        return {k: v for k, v in zip(keys, obj)}
    return {"model": obj}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="capscope", description="Capacity diagnostics for a frozen PyTorch model.")
    ap.add_argument("factory", help="path/to/file.py:build_model, or a saved report.html / report.json to reopen")
    ap.add_argument("--html", help="save a self-contained HTML report to this path")
    ap.add_argument("--json", help="save the raw report data to this path (reopen with `capscope file.json`)")
    ap.add_argument("--export", metavar="PATH",
                    help="write a compact, self-describing summary for logs / LLMs: every module's metrics, "
                         "verdict and graph neighbours plus a legend. .md = Markdown, .jsonl = append one JSON "
                         "line, otherwise JSON; '-' prints Markdown to stdout")
    ap.add_argument("--port", type=int, default=0, help="port for the GUI server (default: random free port)")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    ap.add_argument("--no-gui", action="store_true", help="do not launch the GUI (print the summary instead)")
    ap.add_argument("--summary", action="store_true", help="also print the terminal table")
    ap.add_argument("--n-batches", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.factory.lower().endswith((".html", ".json")):
        report = Report.load(args.factory)
        if not args.quiet:
            print(f"[capscope] reloaded {args.factory}: {len(report.modules)} modules, {len(report.edges)} edges")
    else:
        factory = load_factory(args.factory)
        cfg = _normalise(factory())
        if "model" not in cfg:
            raise SystemExit("factory must return a model, (model, loader) or a dict with 'model'")
        kwargs = {k: cfg[k] for k in ("loader", "forward_fn", "loss_fn", "edge_index_fn", "n_batches") if k in cfg}
        if args.n_batches is not None:
            kwargs["n_batches"] = args.n_batches
        report = inspect(cfg["model"], verbose=not args.quiet, **kwargs)

    if args.html:
        print(f"[capscope] wrote {report.to_html(args.html)}")
    if args.json:
        print(f"[capscope] wrote {report.to_json_file(args.json)}")
    if args.export == "-":
        # utf-8 regardless of the console code page (the document uses arrows and superscripts)
        sys.stdout.flush()
        sys.stdout.buffer.write((report.to_llm("md", source=args.factory) + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()
    elif args.export:
        print(f"[capscope] wrote {report.to_llm_file(args.export, source=args.factory)}")
    if args.summary or args.no_gui:
        report.summary()
    if not args.no_gui:
        report.show(port=args.port, open_browser=not args.no_open, block=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

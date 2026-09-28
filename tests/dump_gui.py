"""Dump what the GUI shows as plain text: header, warnings, module list, every module's detail
panel (panel-below mode, so the guide text is embedded), and the guide. For reviewing the
report the way a reader would, without a browser.

    uv run python tests/dump_gui.py report.html out.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from capscope import Report  # noqa: E402


def main(src: str, dst: str) -> None:
    from playwright.sync_api import sync_playwright
    rep = Report.load(src)
    tmp = Path(dst).with_suffix(".render.html")
    rep.to_html(tmp)
    parts = []
    with sync_playwright() as p:
        browser = None
        for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
            try:
                browser = p.chromium.launch(headless=True, **kw)
                break
            except Exception:
                continue
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(tmp.resolve().as_uri())
        page.wait_for_selector("#graph .node")
        parts.append("=== HEADER ===\n" + page.locator("#meta").inner_text())
        if page.locator("#warnbox").is_visible():
            parts.append("=== WARNINGS ===\n" + page.locator("#warnbox").inner_text())
        page.locator("#sort").select_option("priority")
        parts.append("=== MODULE LIST (sorted by growth priority; columns: # module verdict prio used params) ===\n"
                     + page.locator("#list").inner_text())
        parts.append("=== DATAFLOW EDGES ===\n" + "\n".join(f"{a} -> {b}" for a, b in rep.edges))
        if page.locator("#headline").is_visible():
            parts.append("=== READING (header line) ===\n" + page.locator("#headline").inner_text())
        names = rep.cka.get("names") or []
        Mx = rep.cka.get("matrix") or []
        pairs = sorted(((Mx[i][j], names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))
                        if Mx[i][j] is not None), reverse=True)
        parts.append("=== CKA HEATMAP (the similarity view, as the 40 most similar pairs) ===\n"
                     + "\n".join(f"{v:.3f}  {a}  ~  {b}" for v, a, b in pairs[:40]))
        page.locator("#moveDetail").click()
        page.locator("[data-secs='open']").click()
        page.evaluate("document.querySelectorAll('#right details.explain').forEach(d => d.open = true)")
        for m in rep.modules:
            page.evaluate(f"capscope.select({m['name']!r})")
            page.evaluate("document.querySelectorAll('#right details.explain').forEach(d => d.open = true)")
            parts.append(f"=== MODULE PANEL: {m['name']} ===\n" + page.locator("#right").inner_text())
        page.evaluate("capscope.showGuide(true)")
        parts.append("=== GUIDE (modal) ===\n" + page.locator("#guide .body").inner_text())
        browser.close()
    Path(dst).write_text("\n\n".join(parts), encoding="utf-8")
    tmp.unlink(missing_ok=True)
    print(f"wrote {dst} ({Path(dst).stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

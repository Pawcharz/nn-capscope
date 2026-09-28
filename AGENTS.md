# AGENTS.md — capscope

AI behaviour specification for this project. Read this before touching any file.
Detailed technical reference lives in `README.md` (what is measured, verdict rules, the traps the design works around) — read it on demand, not upfront.

---

## Project orientation

capscope is a small PyTorch library that diagnoses which modules of a *trained, frozen* network have run out of representational capacity and which still have room, so the user can decide where to spend parameters when scaling up. It runs forward hooks over a few batches, recovers the real dataflow graph from autograd, computes rank / spectrum / redundancy / oversmoothing metrics and an SVD truncation sweep, then assigns each module one verdict, one plain-language sentence and a growth priority. The output is a terminal table, a self-contained HTML report and a localhost GUI. Nothing is ever retrained. It targets plain PyTorch with hand-written message passing (no PyG / DGL dependency), but must keep working on any `nn.Module`.

The picture every component serves: a user points capscope at a checkpoint and, within minutes and without training anything, gets a ranked, defensible answer to "widen *this* layer, not that one, and here is why".

---

## Git commits — never run one unasked, but do flag when it's time to ask

**Never run `git commit` unless the user's most recent message explicitly asks for a commit, right now.** This holds even mid-task, even after a string of edits the user clearly wants kept, even if the user said "commit" a few messages ago for a similar change. An earlier "commit that" does **not** authorize committing the *next* change too — wait for a fresh instruction each time. Finishing a fix or feature is not, by itself, a request to commit it. Do not commit "to be helpful" or to keep the working tree tidy.

Do track what has accumulated, though. When it reaches a coherent unit of work — a completed fix, a finished slice, a meaningful doc pass — say so and ask. Not every turn, and not never.

---

## Repository layout

```
capscope/            the library — everything importable, no side effects at import
  capture.py         forward hooks, activation sampling, autograd-graph dataflow recovery, graph smoothness
  metrics.py         rank/saturation, weight spectra (Hill alpha), CKA, dead/duplicate units
  truncation.py      SVD truncation sweep (used rank, marginal pressure)
  verdict.py         upstream carry, verdict rules, sentences, growth priority, THRESH
  report.py          inspect() orchestration, Report (summary / to_html / show / to_llm), HTML payload
  export.py          compact self-describing export (JSON / JSON lines / Markdown) with the field and verdict legend
  cli.py             `capscope file.py:factory` entry point
  gui/template.html  the GUI: hand-rolled SVG + vanilla JS, data injected as JSON
tests/               acceptance tests against a toy model with designed ground truth
  toy_model.py       hand-written SAGE stack with deliberately mismatched widths; trains once, caches to tests/_cache/
  test_capscope.py   ground-truth assertions, serialisation, GUI in a real browser (Playwright)
  smoke.py           manual run printing every raw metric per module
```

Install and run:

```bash
uv sync --all-extras                                   # torch (CPU index), numpy, pytest, playwright
uv run pytest                                          # first run trains the toy model (a few minutes on CPU)
uv run capscope tests/toy_model.py:build_model         # GUI on localhost
uv run capscope tests/toy_model.py:build_model --no-gui --html out.html
```

One root manifest: `pyproject.toml`. Runtime dependencies are only `torch` and `numpy`; keep it that way.

---

## Buckets

Code belongs in one of these buckets. The test: **what does it import, and does it have side effects?**

- **`capscope/`** — the library. Must be importable anywhere with no side effects at import. Never imports from `tests/`. `gui/template.html` belongs here because `report.py` reads it at runtime; it must stay dependency-free (no CDN scripts, no build step) so a saved report works offline.
- **`tests/`** — the toy model and the acceptance suite. Imports `capscope`; nothing imports it. The toy model is a *test fixture with designed ground truth*, not an example to copy into the library.

Dependency direction is strictly one-way:

```
tests/ ──→ capscope/
```

There is no `experiments/` bucket yet. If one is added (e.g. running capscope on real checkpoints), it imports `capscope` and is never imported by it. **Do not create a new bucket unilaterally.** If something doesn't fit an existing bucket, stop and propose the new bucket with a rationale before writing any code.

---

## Before you start

1. Read `README.md` — it is the module README for the single package (see below) and lists the design traps every change must respect.
2. State in one sentence what you read and what your plan is.

---

## Module READMEs

There is one package, so the root `README.md` serves as its module README: the "What it measures", "Verdicts" and "Traps" sections are the current-state description. Introduce per-package READMEs only if a second package appears.

**Reading rule (hard):** read `README.md` before applying any exploration tier. This is tier 0 — it precedes all search.

**Update rule (mandatory):** when you finish a task, update `README.md` if any of these changed: the public API (`inspect` signature, `Report` methods, CLI flags), a metric or its definition, a verdict rule or threshold in `THRESH`, or the GUI's features. Failure to update it is treated the same as leaving broken tests — the task is not done.

No history, no decisions, no rationale in it — only what is true right now.

---

## Exploration strategy — cheapest first

**Never read a file that wasn't first found by search.** Work through tiers in order:

| Tier | Action | When |
|---|---|---|
| 1 | Search by filename or content pattern | Always start here |
| 2 | Read the matched region only | A search hit needs its surrounding context |
| 3 | Read the whole file | The full structure is needed, e.g. understanding a class API |
| 4 | Delegate a broad search to a subagent | A "what exists here" question spanning many directories |

If a one-sentence clarifying question would save a file read, ask it.

**Never open a rendered artifact.** Compiled output — a built PDF, an exported figure image, a plot preview, a notebook's stored output — is not source. Each one enters the context as page images costing thousands of tokens, and is then re-sent on every subsequent step for the rest of the session, so an artifact opened part-way through a long session is paid for hundreds of times over.

- Read the source that produced it instead — the markup, the figure script, the cell. That is the file you can actually edit; the artifact is a derivative you cannot.
- When the question is genuinely visual — does this read clearly, is the spacing right, do these line up — **ask the user**, who has it open in front of them. A one-sentence answer is cheaper and more reliable than your reading of a rasterised page.
- Rebuilding to check that the build succeeded is fine. Reading the result back is not.

**Two narrow exceptions.** A one-off layout check after a build — delegate it to a subagent, so the page images land in that agent's context and are discarded with it, and run it once when the document is otherwise finished rather than after every edit. And when the user explicitly asks you to look at a figure, look at it. Neither exception licenses re-opening an artifact as it evolves; that loop is what this rule exists to stop.

This project's rendered artifacts: any HTML written by `Report.to_html` (it embeds the full JSON payload and is hundreds of kB), browser screenshots of the GUI, and `tests/_cache/*.pt`. `capscope/gui/template.html` is *source*, not an artifact. To check the GUI, run the Playwright test or ask the user; do not paste report HTML into context.

---

## Reusability — search before writing

Before implementing any new function, search for the concept and a few candidate names, and read any plausible match. Reuse or extend what exists rather than adding a near-duplicate; if something is clearly reusable but no existing home fits, discuss it before writing.

---

## Terminology — load-bearing

| Term | Meaning | Role |
|---|---|---|
| **width** | Output dimension of a module (`D` of its `[N, D]` activations) | The denominator of every "uses X of Y dims" statement |
| **used rank** | Smallest weight rank that keeps the loss within `rel_tol` of base (truncation sweep); falls back to rank at 99 % activation energy | What the module actually needs; `used_frac = used / width` |
| **carry** | Narrowest width along the paths feeding a module, propagated forward | Distinguishes "saturated" from "merely relaying a narrow input" (`upstream`) |
| **rank cap** | Whether a weight's `min(in, out)` is set by its input or its output | Decides whether "uses its full rank" is evidence to widen *here* or *upstream* |
| **pressure** | Relative loss increase when the module loses its last used dimension | Feeds the growth priority |
| **producers / consumers** | Dataflow edges recovered from autograd, leaf level; container edges are projections | The graph the GUI draws and the carry walks over |
| **alpha** | Hill estimator of the weight-spectrum power-law tail | `> 6` undertrained, `< 2` over-trained |

`effective rank`, `stable rank`, `numerical rank` and `rank99` are *different* numbers; never substitute one for another in a verdict or sentence. Never use *width* when you mean *used rank*. If unsure, ask before writing.

---

## Pipeline stages — touch only what you're asked to touch

```
Capture (hooks + autograd edges) → per-module metrics → CKA → truncation sweep → carry + verdicts + priority → Report (JSON → HTML / GUI / table)
```

Each stage is independent and communicates through the per-module dict built in `report._build_modules`. When working on one stage, do not modify upstream or downstream stages unless explicitly asked. If a change in one stage requires a new field in the module dict, flag it and stop — the GUI template, `summary()` and the tests all read those fields.

Two paths bypass parts of the sequence and must both keep working: **data-free mode** (`inspect(model)` with no loader: weight spectra only, no edges, no sweep, no CKA) and **no-loss mode** (loader but no `loss_fn`: used rank falls back to `rank99`). Code must handle both without conflating them with the full path.

---

## Design traps — do not regress them

The README's "Traps" section is a list of things that were tried, broke, and were replaced. Before changing anything in `capture.py` or `verdict.py`, re-read it. In particular: dataflow comes from the autograd graph, not `torch.fx` or `id(tensor)`; the `grad_fn` registry holds strong references and lives for exactly one grad-enabled batch; the root module is never a graph node; oversmoothing is *decay relative to upstream*, on centred features; a nonlinearity inflates measured rank, which is why carry exists; an undertrained checkpoint must trigger the header warning rather than a set of confident verdicts. The acceptance tests encode each of these; a change that makes one fail is a regression, not a test to relax.

---

## Long-running commands — background them

**Anything that plausibly runs over a minute — training, data builds, long sweeps, remote jobs — starts in the background.** A foreground command is killed at the tool's timeout with nothing to show for it, and the run has to start over.

- Launch it in the background, then do something that does not depend on it while it runs. Report the result when it lands.
- Before re-running anything that appeared to time out, check whether it is still running. Two copies of the same job competing for the same hardware and the same output directory is worse than waiting.
- If the run is the whole point and nothing else can proceed until it finishes, say that plainly and let the user decide whether to wait or to have you work elsewhere meanwhile.
- Print enough progress to tell a slow run from a hung one.

Here: the first `pytest` run trains the toy model (minutes on CPU) and caches it under `tests/_cache/`; later runs take under a minute. Deleting the cache forces retraining.

---

## No hardcoded paths or magic constants

No path literals anywhere in `capscope/`. Paths come from CLI flags (highest priority), function arguments, or sensible defaults as function arguments. The only file the library reads is `gui/template.html`, located relative to `__file__`.

No magic numeric constants as literals inside library code. Verdict thresholds live in `verdict.THRESH` and are overridable through `inspect(thresholds=...)`; metric tolerances (`rel_tol`, `dup_thresh`, `max_rows`, `max_sv`) are keyword arguments with defaults. A new threshold goes into `THRESH`, gets a line in the README's verdict table, and is reflected in `meta["thresholds"]` so the report is self-describing.

---

## Dated logs and records

If the project keeps a dated research log, decision record, or findings file, it is **either current or explicitly frozen** — there is no third state.

- A stale dated log is worse than none: agents read it as present state and reason from superseded facts, which is harder to catch than an absence.
- When one stops being maintained, mark it frozen **in the file itself** — a dated line at the top saying it is closed and what supersedes it. Deciding that in conversation and not writing it down is how the file goes on misleading people. Same for any doc holding results or decisions.
- Before relying on a dated doc, check its latest entry is recent relative to your work. If it plainly stopped, say so rather than quoting it as current.

---

## Reproducibility

A result nobody can regenerate is not a result.

- **Seed every source of randomness explicitly**, and record the seed with the output. Never rely on a library's default seeding. `Capture` takes a `seed`; the toy model's data, init and training are seeded; the activation-histogram and redundancy subsamples use a fixed `RandomState`.
- **A report's configuration is part of its output.** `meta` carries `n_batches`, `has_loss`, `has_graph`, `base_loss` and the resolved thresholds; keep it that way when adding options.
- **Record provenance with results** — what code version, against what checkpoint. A saved report that can't be traced back to a commit and a checkpoint is a dead end once the method moves.
- **Never edit results by hand.** If a number is wrong, fix the code and regenerate.
- **Changing a metric or a verdict rule invalidates reports computed with the old one.** Mark those superseded, or someone will later compare both without knowing they differ.

---

## Changing a load-bearing value — sweep first, edit second

**A threshold, seed, split or metric that appears in more than one place changes everywhere at once, or not at all.**

- Before editing anything, search the whole repo *and* any prose that quotes the value for every occurrence, and list them as `file:line`. Show that list, then edit.
- These values live in places that drift apart independently: `verdict.THRESH`, the verdict sentences that quote them, the README's verdict table, the GUI's alpha-gauge zones in `template.html`, and the test assertions. A change that lands in three of the five leaves the tool contradicting its own documentation.
- Regenerating the affected results, figures and tables is **part of the change**, not a follow-up task.
- Do not report the change as done while the search still returns the old value. Run it again and say what it returned.
- If the old value must survive somewhere on purpose — a definition, a published number, a comparison point — say where and why, in the write-up as well as to the user.

---

## Testing

- Every new function or class in `capscope/` gets a test in `tests/`.
- Tests use minimal synthetic inputs — the toy model and its generated graphs; never require real data or real checkpoints on disk.
- The toy model's *designed ground truth* (which module is saturated, upstream, passthrough, narrow; the exact edge set) is the acceptance bar. Do not weaken an assertion to make a change pass; if the ground truth is genuinely wrong, change the toy model's design and its docstring together.
- Data-free mode and no-loss mode must both stay covered.
- Any GUI change is verified by the Playwright test (zero console errors, the interactions it exercises); add an interaction there when adding a control.
- Run `uv run pytest` before declaring any task done; fix failures before moving on.

---

## Types and code style

- All public functions and methods in `capscope/` carry complete type annotations. Elsewhere: entry points annotated, internal helpers where non-obvious.
- Structured data uses declared types where it crosses a boundary (`ModuleRecord` is a dataclass). The per-module dict that flows through the pipeline is deliberately a plain dict because it is serialised to JSON as-is; new fields must be JSON-safe (`_sanitize` turns NaN into `null`; fields prefixed with `_` are stripped before serialisation).
- Keep import time low; `import capscope` must not import `playwright`, and the HTTP server is only built inside `show()`.
- Everything the GUI needs is in the JSON payload; the template has no other data source and no network access.

---

## Communication

Answers get read, not skimmed — length is a cost paid by the reader. Lead with the answer, then only the reasoning that changes what they would do. Two precise sentences beat five hedged paragraphs; use a table or list when the content is structured.

Explain a term or abbreviation the first time it appears. After that, calibrate: if the user uses it fluently, don't restate it; if their questions suggest it hasn't landed, briefly re-anchor the definition without being asked.

Don't pad. No restating the request before answering, no summarizing what was just said, no narrating what you're about to do when doing it is faster.

**Say when the session has become expensive.** Every step in a long session costs more time than the same step early on, because the whole accumulated context is reprocessed each time — by ~600k tokens a step can take roughly three times as long as it did at 200k. Once the context passes a few hundred thousand tokens, say so **once**, name the next natural boundary (a finished figure, an accepted result, a commit), and let the user decide whether to continue or resume there in a fresh session. A new session starts cheaply: AGENTS.md, the README and the tests carry the context that matters.

Say it once per topic, not once per turn. It is a nudge, not a nag, and the user may have good reason to keep going.

---

## Markdown formatting

Write one paragraph as one line. Never insert hard line breaks inside a paragraph to hit a column width — that overrides the editor's own wrapping and makes reflowing and diffing worse for every later reader. Line breaks separate blocks; they don't wrap text.

---

## Keeping AGENTS.md accurate

When you find that AGENTS.md is inaccurate — wrong path, obsolete section, missing convention — do **not** silently correct it mid-task. Finish the task first, then fix it as a follow-up edit in the same session and tell the user what changed and why.

---

## What to do when unsure

If the correct approach is ambiguous — which bucket code belongs in, whether a change cascades elsewhere, whether new code is worth extracting — **stop and ask** before writing. A one-sentence clarification costs nothing; a wrong implementation costs a redo.

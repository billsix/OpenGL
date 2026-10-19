# Annotate every binding in the Python port + wire a ty type-gate

## BLUF

Add an explicit type annotation to **every** binding in the `mcnopper` Python
port — every module-level global and every local, not just signatures — per the
shared Python standard (`~/.claude/reference/python-coding-standard.md`, "Type
annotations — annotate every binding"). Wire a permanent `ty` +
annotation-completeness gate. "Done" means `make type-check` is green (`ty check`
clean + `check_local_annotations.py --include-module` reports 0), `make check`
still imports all 52 demos, and the work is staged on the `python` branch.

## Context (cold-start)

- Branch: **`python`** (the port; `master`/`origin/main` is the upstream McNopper
  C examples, v3.2.7).
- Layout: `python/src/mcnopper/` is the package AND holds the demos as submodules
  (`exampleNN.py`, four nested `exampleNN/exampleNN.py` for co-located shaders).
  Console script `mcnopper N`. Editable-installed.
- Baseline gap (2026-10-19): **376 un-annotated bindings** (from repo root:
  `python tools/check_local_annotations.py --include-module python/src/mcnopper`).
- Standard nuances: loop/unpack targets typed on the line above; read-only
  container params take covariant `Mapping`/`Sequence`; parameterize generics;
  inline single-use values; comprehension loop vars stay inferred; don't fight
  the checker (leave inferred + say why). GL/GLUS/imgui libs untyped → `ty` = `Any`.

## Gate (wired this task)

- `tools/check_local_annotations.py` — vendored from runClaudeInContainer canonical.
- Dockerfile installs `ty` (`uv pip install setuptools ty`).
- `make type-check` — `ty check src/mcnopper` (package incl. demo submodules) +
  the checker. Needs `make image` rebuild first.

## Plan

1. Wire gate (Dockerfile + Makefile + vendored checker). **done**
2. Rebuild image; baseline. (in progress)
3. Annotate the core shared modules (common/gltf/gltf_gl/ibl/perlin/sh/cli); verify.
4. Fan out fresh subagents across the demos (git forbidden); verify green.
5. Stage on `python`. (Maintainer commits.)

## Status

**Priority:** 3  **Difficulty:** 7  — **COMPLETE (pending maintainer commit).**

All phases done and verified (2026-10-19):
- 376 un-annotated bindings → **0** (`check_local_annotations.py --include-module`).
- `make type-check` **green**: `ty check src/mcnopper` + checker all pass.
- `make check` imports **52 / 52** demos; `make format` clean (ruff).
- Sweep by 4 fan-out subagents (core modules + 3 demo batches).
- Central work beyond the sweep (all pre-existing issues surfaced by giving the
  port its first ruff + ty gate):
  - **ruff config**: added `[tool.ruff]` (`line-length=80`, minimal `select`) to
    `python/pyproject.toml` — the old "root pyproject governs" note was copied
    from mvp but this standalone repo has no root config, so ruff had used 88-col
    defaults and never formatted the port.
  - **43 T201 prints** `# noqa: T201` (maintainer's call to keep T201 + noqa) and
    **5 E402** (deliberate GL-setup-before-import) via `ruff --add-noqa`.
  - **3 F841** pre-existing dead locals `# noqa: F841` with reasons (NOT removed,
    per McNopper's stability mandate) — `image_desc`, `fmt` (XYZE branch),
    `dir_tex`. Candidates for a separate cleanup if desired.
  - **ty**: ~61 `unsupported-operator` on GL-`Constant` `|`/`+` (PyOpenGL's
    Constant has no typed operators — the sibling ports carry the same ignore)
    via `tasks/adhoc/annotate-every-binding/add_bitmask_ty_ignore.py` (idempotent)
    plus 3 hand-added for multi-line ORs; `make_vbo(data)` widened to
    `np.ndarray | None` + a None-guard (callers pass `Shape.tangents` etc. typed
    `| None`); example31/39 GL-enum params widened to `int | Constant`; example49
    /50 skin index narrowed with a typed local + assert; gltf `.nodes or []`.
- Rule 3 (loop/unpack targets) left inferred per the checker's exemption.

Staged on the `python` branch — maintainer commits. The one-shot codemod
`git rm`s at archive time (after the work commit).

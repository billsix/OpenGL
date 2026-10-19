# Reference the C shaders from the Python port instead of copying them

**Status:** done 2026-10-18 — implemented, container-verified (resolve 194/194, import
52/52, demos 5/13/48 compiled+ran headless), and a referenced-shader demo (15) ran on the
maintainer's display. Remaining full-parity spot-checks are ordinary maintainer display
work, not a blocker.
**Priority:** 3
**Difficulty:** 3

## BLUF

The Python port under `python/` kept its own byte-for-byte copies of each C example's
shaders. Now that the port lives in the **same repository** as the C examples, those
copies are redundant: the port references the C example's `ExampleNN/shader/*.glsl`
directly, and the 181 identical copies are deleted. Only shaders that genuinely differ
from the C stay copied (Example13's one port edit, and examples 48–50's version-lowered
`background.*`/`fullscreen.*`), plus `_glus_shaders/` (which has no committed C source).
"Done" means the port resolves every shader in and out of the container and the demos
still render on a display.

## Context

The port was originally developed in a *separate* repository from the McNopper C
examples (the upstream), so copying the shaders in was the only option — you cannot
reference another repo. It was then moved into this repository (`python/` subtree), which
made the C shaders siblings of the Python code, and made the copies pure duplication with
a drift hazard: the fidelity audit's "root cause 8" was exactly shader drift between the
copies and the C, and no copy-sync tool was carried over, so the copies were unmanaged.

This parallels the asset decision already made for the port: `find_media` reuses this
repo's `Binaries/` rather than shipping a copy (`tasks/reference/python-port/gl-and-imgui-gotchas.md`
§4). Shaders now follow the same principle.

## The decision (A), and the alternative (B) that was rejected

Decided by the maintainer (William Emerison Six <billsix@gmail.com>, 2026-10-18), choosing
option **A** on the agent's recommendation:

- **A (chosen) — reference & delete.** Per-example shaders resolve up to
  `<repo>/ExampleNN/shader/`; the 181 byte-identical copies are deleted. Single source of
  truth, eliminates the drift class, removes 181 redundant files.
- **B (rejected) — keep copies + a drift-check gate.** Would preserve a self-contained
  `pip` wheel and the shaders-beside-the-demo readability, catching drift instead of
  eliminating it (what the original repo did with a copy script).

**The deciding factor was distribution.** `pyproject.toml` shipped the shaders in the
wheel, so A means the port no longer builds a meaningful standalone wheel — it runs from a
checkout of this repo. That was already true for assets (`find_media` needs this repo's
`Binaries/` or a hand-set `MCNOPPER_MEDIA`), so A is consistent with how the port already
works, and the maintainer accepted dropping the standalone wheel in exchange for killing
the drift class.

## What must stay copied (cannot be referenced)

Measured by `tasks/adhoc/python-port-shader-references/classify.py` (compares each copy to
its C sibling from the `#version` line down, ignoring the provenance comment and trailing
newline): **181 identical → delete**, **13 differ → keep**:

- **Example13 `tessellation.cont.glsl`** — the one deliberate port edit (`u_tessLevel`
  slider replacing the C's hard-coded levels; `tasks/reference/python-port/c-to-python-rules.md` §6).
- **Examples 48/49/50 `background.{vert,frag}.glsl` + `fullscreen.{vert,frag}.glsl`** — the
  `#version 460`→`410 core` lowering, so they differ from the C.
- **`python/src/mcnopper/_glus_shaders/*` (examples 48–52)** — git-tracked only under
  `python/`; the C-side copies live in `build/_deps/glus-src/shader/`, a **gitignored build
  artifact** (cmake fetches GLUS at configure time), so there is no committed path to point
  at. Not touched by this change (loaded via `_ibl`, not `find_shader`).

## Implementation

1. **`_common.find_shader`** gains a fallback: after the local candidates (`here`,
   `here/shader`, `here.parent/shader`), map the demo dir `exampleNN` to
   `<repo>/ExampleNN/shader/` (nearest ancestor that has that directory) and look there.
   Local-first means the kept copies (13, 48–50) win and the deleted ones fall through to
   the C sibling. The load-time `_maybe_bump_version` (420 bump) and the `#define` injection
   in 47/51/52 both find `#version` by regex, which works on the C file (no provenance
   comment, so `#version` is line 1).
2. **Delete the 181 identical copies** (`git rm`), from the classifier's list.
3. **Docs** updated to match: `tasks/reference/python-port/c-to-python-rules.md` §6, the set
   `README.md`, `python/README.md`, and the `AGENTS.md` cold-start bullet — "per-example
   shaders are referenced from `ExampleNN/shader/`; only the edited/version-lowered ones and
   `_glus_shaders/` are copied." `python/pyproject.toml` package-data is left as-is: the
   `example*/**/*.glsl` glob now matches only the 13 kept copies, which is correct.

## Verification

- `python/` `make check` imports all 52 demos (does not compile shaders).
- A static resolution pass: every `.glsl` a demo names resolves via the new `find_shader`.
- **Owed (maintainer, display):** the on-display run above — shader compilation and the
  rendered picture are only verifiable with a GPU/display.

## Adhoc

`tasks/adhoc/python-port-shader-references/classify.py` — the read-only classifier whose
output drove the deletion list. One-shot; `git rm` at archive.

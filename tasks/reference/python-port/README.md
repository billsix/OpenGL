# The Python port — reference set map (L0)

**Reference set** — durable knowledge about the Python port under
[`python/`](../../../python/) of this repository
([github.com/McNopper/OpenGL](https://github.com/McNopper/OpenGL)): the C→Python
translation rules it settled on, the GL/imgui/GLFW traps met along the way, the
modern-GL failure modes specific to Mesa + PyOpenGL, and the per-demo fidelity
verdict against the C originals that sit beside it (`Example01..52/`). Read this
first; each row links down to the doc with the detail. Updated in place, never
archived.

This knowledge was carried over from the port's development history (it was first
built inside a separate course repository and moved here) so that this repository
reads cold: everything a fresh contributor needs is in these docs or the files they
point at, with no dependency on where the port came from.

## What the port is

A Python port of all **52** C/GLUS examples in this repository, using PyOpenGL +
glfw + Dear ImGui (imgui-bundle) and rendering the same scenes. It lives in
`python/` and **reuses this repository's own files** rather than shipping copies:
`Binaries/` for assets (meshes, textures, HDRIs, glTF scenes) and each C example's
`ExampleNN/shader/*.glsl` for shaders. Only the handful of shaders that differ from
the C (`c-to-python-rules.md` §6) and the shared `glus_shaders/` are kept in the
package. The container contract, packaging, and run commands are in
[`python/README.md`](../../../python/README.md).

- Self-contained package: `python/src/mcnopper/` — `common.py` (window loop,
  shader compile, OBJ/TGA/HDR loaders, GLUS-exact primitive builders, `find_media`),
  `perlin.py`, the glTF/PBR pipeline (`gltf.py`/`gltf_gl.py`/`ibl.py`/`sh.py`
  + `glus_shaders/`), `cli.py`, and one flat `exampleNN.py` per demo
  (example13/48/49/50 stay nested for their co-located shaders).
- The C originals are right here: each demo module `python/src/mcnopper/exampleNN.py`
  (13/48/49/50 nested as `exampleNN/exampleNN.py`) has its counterpart in
  `ExampleNN/src/main.c` (+ `ExampleNN/shader/`), which the fidelity audit compares against.

## The docs in this set

| Doc | One breath |
|---|---|
| `c-to-python-rules.md` | The settled C→Python rules: the type table, imgui-widgets-not-keyboard (menubar Controls menu, F11, Esc), "set" a uniform vs "upload" a buffer, inline small fixtures, **shaders are verbatim copies of the C build's** (§8), **keyword every argument a callee accepts, and which PyOpenGL functions cannot** (§10). |
| `gl-and-imgui-gotchas.md` | The imgui/GLFW integration traps (callback chaining and Esc, the stencil-test-around-imgui clip, `import glfw` before `imgui_bundle`, the narrow-floating-window `imgui.ini` trap), the planar-shadow clip-space `w` sign, the repo-wide GL conventions, and the container/display/asset-resolution learnings (Wayland libraries, `find_media`, X11-vs-Wayland backend agreement). |
| `mesa-and-pyopengl-traps.md` | The modern-GL failure modes on Mesa + PyOpenGL that the proprietary drivers hide for you: the `contextdata` patch, `glMemoryBarrier`, fragment-shader interlock, `glClearTexImage`, the GLSL `#version`/`flat` rewrites, and the two primitive-level parity bugs (sphere winding, HDR row order). |
| `fidelity-audit.md` | Per-demo verdict (faithful / drift / gap / broken) of all the Python demos vs the C at this repository's examples, the cross-cutting root causes, and the finding → fix table. All findings are fixed; kept as the record and the latent-risk list. |
| `headless-verification.md` | What a no-display sweep can and cannot tell you, the back-buffer capture method and the X11/Wayland backend agreement `run_demo` supports via `MVP_GLFW_X11`, and the capture-timing caveat for the slow CPU demos. Method only — the mvp-era harness scripts were not carried over. |

## Verification status

Every Python demo was audited against its C original and brought to **faithful**
on the maintainer's display (see `fidelity-audit.md` for the per-demo record and
the remaining latent risks). A no-display run can confirm a demo imports and does
not crash (`make check` in `python/` imports all 52); it cannot confirm the
picture matches — that needs a display, which is the maintainer's to run
(`headless-verification.md` §1).

## Related (elsewhere in this repo)

- [`python/README.md`](../../../python/README.md) — the commands-forward run/build guide.
- [`AGENTS.md`](../../../AGENTS.md) — the repository working contract; its "Python
  port" section points here.
- [`THIRD-PARTY.md`](../../../THIRD-PARTY.md) — the port's runtime-dependency licenses.

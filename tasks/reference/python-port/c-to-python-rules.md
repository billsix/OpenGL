# C/GLUS → Python: the settled rules for the port (L2)

Up: `README.md`. The rules the Python port under `python/` settled on, translating this
repository's C/GLUS examples. **Behaviour-faithful throughout:** nothing here may change
GL call order, RNG, or what is drawn. Code paths are under `python/src/mcnopper/`.

## 1. Structure follows the C, names follow Python

Each demo keeps its C original's *shape* — GLUS's function-pointer callbacks become the
port's `DemoCallbacks` (`init`/`reshape`/`update`/`key`/`mouse`/`mouse_move`/`terminate`/
`imgui_panel`), handed to `common.run_demo` — so a reader can hold `ExampleNN/src/main.c`
and `exampleNN.py` side by side. Names are brought to Python house style:
`m_pWindow` → `window`; a demo imports the installed package directly
(`from mcnopper import common`, and the PBR demos `from mcnopper import gltf, gltf_gl as gl,
ibl`). The C↔Python mapping is kept in the framework docstrings so the side-by-side reading
still works.

## 2. The type table (C / GLUS → Python)

| C / library type | Python | Note |
|---|---|---|
| `GLuint`, `GLint`, `GLsizei` | `int` | names, locations, counts |
| `GLenum` | `GLenum = int \| Constant` | the alias threaded through `common` — PyOpenGL constants are `IntConstant`, so a bare `int` annotation on a GL enum needs a suppression; the alias avoids ~125 of them |
| `GLfloat` / `GLboolean` / `GLchar*` | `float` / `bool` / `str` | |
| `GLfloat[16]`/`[9]`/`[3]`/`[4]` | `np.ndarray` | 4×4 **row-major**, transposed at upload (`gl-and-imgui-gotchas.md` §3) |
| `GLUSshape` | `common.Shape` | from `make_sphere`/`make_cube`/`make_torus`/`make_rectangular_grid_plane` |
| `GLUSwavefront`/`GLUSgroup`/`GLUSmaterial` | `common.ObjMesh`/`ObjGroup`/`ObjMaterial` | `load_obj` keeps GLUS's group rules and reads the `.mtl`; `draw_obj_group` is the per-group `glDrawElements` |
| `GLUSimage` / TGA / HDR | `common.TgaImage` / `load_tga` / `load_hdr` | |
| `glusPerlinCreateNoise{1,2,3}D` | `perlin.create_noise_{1,2,3}d` | texel-identical, glibc `rand()` reproduced |
| a GLFW window handle | `Any` | **the one sanctioned `Any`** — glfw's stubs type it as an opaque pointer |

Two footguns: `glGen*(1)` returns an `int` but `glGen*(n>1)` returns an array — annotate
only the `(1)` form; and the per-demo `state: dict = {}` is heterogeneous by design, so do
not type it more precisely. **Typing found real bugs by counting:** `example31`'s C set
four `PointLightLocation` uniforms and the Python two; `example40`'s sphere program never
had `u_color` set. **Count the `glUniform*` calls per program against the C's
`glGetUniformLocation`s.**

## 3. Every adjustable parameter is an imgui widget; keys are accelerators at most

A C `case 'M': mode = (mode+1)%N;` becomes an `imgui.radio_button`/combo; `case 'P':
paused = !paused;` becomes a checkbox; `+`/`-` become a slider; `R: load_shaders()` becomes
a button. **Show the widget — do not write "press M to toggle" in the panel.**

**The widgets live in the menubar's Controls menu, never a floating window** (the
maintainer's call, William Emerison Six <billsix@gmail.com>, 2026-10-14: a floating panel
overlaps the drawing, and a bare `imgui.begin` saves a too-narrow size — `gl-and-imgui-gotchas.md`
§1). The menubar is `File → Quit (Esc)`, `View → Fullscreen (F11)`, `Controls → the demo's
widgets`. A McNopper demo passes `imgui_panel=` to `run_demo`, which draws it inside the
Controls menu. A menu body runs only while the menu is open, so apply a change on the spot
or store it on the demo's `state`. Headless gates set `MVP_CONTROLS_WINDOW=1`, which also
draws the body in a plain window every frame so every widget executes
(`headless-verification.md`).

## 4. "Set" a uniform; "upload" a buffer or a texture

`glUniform*` specifies program state (it moves no bytes to GPU memory); `glBufferData`/
`glTexImage*` do. So helpers are named `set_uniforms`, and "upload" is reserved for VBO
creation, texture loads, and UBO `glBufferSubData`.

## 5. Inline small static fixtures; never parse a C header at runtime

Static vertex/index tables small enough to hold (a few KB) live as Python tuples, not a
regex over a C source file that may not be present. Same for anything you would otherwise
read out of the C tree at runtime.

## 6. The shaders are the C build's; parity work is Python-side

**The port uses this repository's own C shaders, not copies of them.** A demo's
per-example shaders are read directly from the C example's `ExampleNN/shader/*.glsl` —
`common.find_shader` maps the demo `exampleNN` to `<repo>/ExampleNN/shader` (after
checking beside the demo first). This replaced 181 byte-for-byte copies that duplicated the
C shaders when the port lived in a separate repo (decided by the maintainer, William
Emerison Six <billsix@gmail.com>, 2026-10-18;
`tasks/archive/2026/10/18/python-port-reference-c-shaders.md`).
**A local copy is kept only where the shader genuinely differs from the C**, and
`find_shader`'s local-first order picks it over the C sibling:

- **Example13 `tessellation.cont.glsl`** — the one deliberate port edit: `u_tessLevel`
  replaces the C's six hard-coded `4.0` levels so the imgui slider can drive it.
- **Examples 48/49/50 `background.{vert,frag}` + `fullscreen.{vert,frag}`** — `#version
  460` lowered to `410 core`.
- **`python/src/mcnopper/glus_shaders/`** — GLUS's shared PBR/IBL shaders (the passes the C
  loads from `GLUS_SHADER_DIR`, examples 48–52), loaded via `ibl` rather than
  `find_shader`. These are committed here because the C-side copies are a build artifact
  (GLUS is fetched at configure time), and their `#version 460` is lowered to `410 core`.

**The `#version` is otherwise the C's** (`common._maybe_bump_version` raises a
`layout(binding)` shader to 420 at load; `mesa-and-pyopengl-traps.md` §6). Never
hand-reduce, "simplify", restyle or re-derive a shader; if the port needs different
behaviour, that is a Python-side change (uniforms, texture formats/units, GL state, camera).
Decided by the maintainer (William Emerison Six <billsix@gmail.com>, 2026-10-13) after a
hand-written PBR shader hid two Python-side bugs (an HDR overflow and unfiltered importance
sampling) behind shading differences.

- **Runtime splicing finds `#version` by regex, never assumes a line number.** A port that
  injects `#define`s at load (47's `GRID_N`, 51/52's `SPLAT_STRIDE`) inserts them after the
  line a `^#version` regex finds. The referenced C shaders have `#version` on line 1; a kept
  copy may carry a provenance comment above it — the regex handles both, but splicing after a
  fixed "first newline" would put `#version` out of place and GLSL would reject the file.
- Anything in a per-example shader that differs from its `ExampleNN/shader/` original, beyond
  the kept exceptions listed above, is a bug.

## 7. Keyword arguments at every call site that can take them

Every call whose callee accepts keyword arguments passes them by keyword, under one
mechanical rule (maintainer's call, William Emerison Six <billsix@gmail.com>, 2026-10-14):

- **Project APIs and keyword-capable library calls keyword every argument:** the port's own
  functions and dataclasses, PyOpenGL's Python-level convenience functions
  (`glBufferData(target=…, size=…, data=…, usage=…)`, `glVertexAttribPointer`,
  `glGetUniformLocation`, `glDeleteTextures`, …), imgui-bundle
  (`imgui.slider_float(label=…, v=…, v_min=…, v_max=…)`) and glfw.
- **Library callees keep required leading parameters positional** and keyword only those
  with a default: `np.zeros((3,), dtype=…)`, `open(path, mode="rb")`. `np.cross(a, b)` and
  `re.sub(p, r, s)` stay as they are. **Single-argument and `*`/`**`-splat calls are left
  alone.**

**What cannot take keywords, and stays positional — do not "finish the job" by hand:**
PyOpenGL 3.1.10's raw ctypes functions (`glUseProgram`, `glBindTexture`, `glUniform*f/i`,
`glEnable`, `glViewport`, `glClearColor`, `glDrawArrays`, …) and its array-converting
wrappers (`glGen*`, `glDrawElements`, `glTexImage2D/3D`, `glUniformMatrix4fv`,
`glGetIntegerv`, …) raise `TypeError` on a keyword argument. numpy ufuncs (`np.sin`,
`np.radians`), numpy scalar constructors and `math.*` take no keywords either. Classify by
probing the live GL object, never by a hand list. This is a convention, not a gate (ruff
has no rule for it).

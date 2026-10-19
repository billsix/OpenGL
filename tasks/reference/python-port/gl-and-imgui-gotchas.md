# OpenGL / imgui / GLFW gotchas (Python port) (L2)

Up: `README.md`. The GL/imgui/GLFW trap corpus for the Python port under `python/` —
each entry cost a debugging session. Check here before writing or debugging any
GL/imgui/GLFW code in the port. Modern-GL-on-Mesa failure modes (barriers, interlock,
`#version` rewrites) are the sibling `mesa-and-pyopengl-traps.md`; this doc is the
imgui/GLFW integration, the fixed-function-era shadow math, the repo-wide conventions,
and the container/display setup. Updated in place.

All code paths are under `python/src/mcnopper/`; the window loop and the shared helpers
are in `common.py` (`run_demo`, `draw_menubar`, `find_media`, `find_shader`).

---

## 1. imgui-bundle / GLFW integration

- **`GlfwRenderer(window)` installs its own GLFW key/mouse/cursor callbacks in its
  constructor and does NOT chain to any previously-installed ones.** Whatever you set
  with `glfw.set_key_callback` *before* constructing the renderer is silently
  discarded. The port's `run_demo` therefore constructs the renderer **first**, then
  installs its own `on_key`/`on_mouse_button`/`on_mouse_move` and has each one forward
  the event to the renderer's handler (`impl.keyboard_callback` /
  `mouse_button_callback` / `mouse_callback`) before running the demo's logic — so
  imgui keeps its input *and* the demo's keys fire. The symptom when this is wrong:
  **Escape does not quit and no demo key works**, while the menubar still responds
  (the renderer's own callbacks were the ones that survived). `char`/`scroll`/`resize`
  are left as the renderer set them (the port does not override those), so imgui text
  input and scrolling keep working (William Emerison Six <billsix@gmail.com>,
  2026-10-18).
- **imgui-bundle's GL backend sets up blend/cull/depth/scissor but never touches
  `GL_STENCIL_TEST`, and does not save/restore it.** A demo that leaves a restrictive
  `glStencilFunc` enabled (Example18, the stencil cap: `GL_EQUAL, 1, 1`) would
  otherwise have its whole UI stencil-rejected — the menubar draws where the stencil
  buffer is 0 and vanishes. `run_demo` disables `GL_STENCIL_TEST` around imgui's draw
  and restores it after, the same shielding it already does for the TEXTURE0 bindings
  (the renderer binds its font atlas to `GL_TEXTURE0`). This is the port's central
  version of the fix; a demo could instead disable stencil itself before returning,
  but doing it once in `run_demo` covers every stencil demo (William Emerison Six
  <billsix@gmail.com>, 2026-10-18).
- **`import glfw` must run before `import imgui_bundle` in every module that imports
  imgui (the first GLFW loaded into the process wins).** imgui-bundle ships its own
  `libglfw.so.3`, an X11-only build. If its native module loads first, pyglfw binds
  that copy, the window gets a GLX context through XWayland, and PyOpenGL — which picks
  its EGL backend purely from `WAYLAND_DISPLAY` — sees no current context. Its
  extension check then fails, so every GL 4.2+ function that `OpenGL.GL` exposes as an
  ARB alias (`glBindImageTexture`, `glMemoryBarrier`, `glTexStorage2D`,
  `glDispatchCompute`) raises `NullFunctionError` while `glGetString` still says "4.6".
  `common` imports glfw before imgui, so a demo that only imports `common`
  (`from mcnopper import common`) is covered; a module that imports `imgui_bundle` directly carries
  `import glfw  # noqa: F401` above it. The container's Makefile also exports
  `PYGLFW_LIBRARY=/usr/lib64/libglfw.so.3` (the system dual-backend build) which masks
  the problem, so a green container run is not evidence the import order is right —
  keep the rule.
- **A bare `imgui.begin(title)` floating window opens too narrow to show a slider, and
  `imgui.ini` then remembers it.** imgui auto-fits a new window once and sizes each
  slider to 65 % of the window's near-zero first-frame width, so it fits around
  collapsed sliders; the saved `imgui.ini` (written to the cwd) then pins the narrow
  size forever. The port avoids this entirely by putting every demo's widgets in the
  **menubar's Controls menu, never a floating window** (`c-to-python-rules.md` §3). If
  a floating window is ever genuinely needed, open it with `always_auto_resize` plus
  `set_next_window_size_constraints` giving a minimum width (which overrides a saved
  size).

## 2. The planar-shadow matrix and clip-space `w`

**A planar-shadow matrix is NOT sign-invariant in OpenGL.** Projectively `(p, w)` and
`(−p, −w)` are the same point, but clip testing happens *before* the perspective
divide: GL tests `−w ≤ x,y,z ≤ w`, which is unsatisfiable for `w < 0`, so **every
shadow vertex is discarded — no error, no warning, the shadow just vanishes.**

The matrix's bottom row makes every transformed vertex share the constant
`w = n·(−light.xyz)`, so with the wrong plane-normal sign that `w` is negative for the
typical ground-plane/overhead-light setup. Fix:
`sign = 1.0 if (a*lx + b*ly + c*lz) > 0.0 else -1.0`, multiply the whole 4×4. The port's
`planar_shadow_directional` / `planar_shadow_point` do this. Diagnosis pattern worth
copying: renders can be pixel-plausible, `glGetError()` clean, and the modelview matrix
correct — the smoking gun is a manual `M @ v` showing a negative `w`.

## 3. Repo-wide GL conventions for the port

Standing conventions for writing or editing port code (failure modes are §1–2 above and
`mesa-and-pyopengl-traps.md`):

- **Import order is load-bearing, not cosmetic.** `glfw` imports before `imgui_bundle`
  (§1); isort must not reorder these.
- **GL constants are not `int`.** PyOpenGL constants are `IntConstant`; annotating a GL
  enum parameter as `int` needs a checker suppression. The port threads a
  `GLenum = int | Constant` alias through `common` for this; decide a suppression once
  rather than copy-pasting `# ty: ignore` at each site.
- **`-1` is the "uniform/attribute not present" sentinel**, and **each check must be
  justified independently** — gating one uniform's use on *another* uniform's presence
  is a latent bug. Always `glGetUniformLocation` after linking, even where the C
  hard-codes a location (`mesa-and-pyopengl-traps.md` §5).
- **All 4×4s are row-major, `M @ column_vector`; the transpose happens exactly at the
  `glUniformMatrix4fv` boundary** (`GL_TRUE`). Do not "fix" that flag.
- **Winding is CCW everywhere**, and the port's primitive builders reproduce GLUS's
  vertex/index order so a demo can keep the C's `glFrontFace(GL_CW)` + culling idiom
  verbatim (`mesa-and-pyopengl-traps.md` §15). Any new geometry builder documents its
  winding.
- **Shaders are this repo's C shaders, resolved by `common.find_shader(__file__, name)`**
  — beside the demo first, then the C example's `<repo>/ExampleNN/shader/` (never cwd). The
  port references the C shaders rather than copying them; only the ones that differ keep a
  local copy (`c-to-python-rules.md` §6).
- **macOS Core Profile requires a non-zero VAO bound at all times** — `run_demo` binds a
  default VAO at startup and never calls `glBindVertexArray(0)` for the process lifetime.

## 4. Container, display, and asset resolution

The operational traps for getting the port to actually open a window and find its data,
in and out of the container (`python/README.md` has the commands):

- **The container needs the Wayland client libraries, not just the X11 ones.** glfw's
  Wayland backend `dlopen`s `libwayland-client`, `libwayland-cursor`, `libwayland-egl`
  and `libxkbcommon` at runtime; if the image installs only the `libX*` packages, a run
  under a Wayland session dies with "Failed to load libwayland-cursor" before any
  window appears. `python/Dockerfile` installs all four (William Emerison Six
  <billsix@gmail.com>, 2026-10-18).
- **Assets resolve against this repo's `Binaries/`, found relative to the package.**
  `common.find_media(name)` tries `$MCNOPPER_MEDIA` first (the container sets it to
  `/opengl/Binaries`), then walks up from `common.py`'s own location to the nearest
  ancestor holding a `Binaries/` directory — so a demo run straight from a checkout
  finds `<repo>/Binaries` with no environment variable, and a `pip install` into
  site-packages (no ancestor `Binaries/`) falls through to the packaged `media/`
  candidates as before. Do not duplicate assets under `python/` (William Emerison Six
  <billsix@gmail.com>, 2026-10-18).
- **The `.hdr` and `.bin` assets in `Binaries/` are git-lfs-tracked.** A fresh clone
  needs `git lfs pull` (or git-lfs installed at clone time) or the HDR demos (32/33)
  and the glTF demos (48–52) cannot load their data.
- **Pick GLFW's backend and PyOpenGL's backend to agree.** PyOpenGL chooses EGL when
  `WAYLAND_DISPLAY` is set and GLX when it is not; GLFW 3.4 prefers Wayland when the
  socket is present. If they disagree, PyOpenGL sees no current context and ARB-aliased
  GL 4.2+ calls raise `NullFunctionError` (§1, same mechanism). For a headless/Xvfb run,
  set `MVP_GLFW_X11=1` and unset `WAYLAND_DISPLAY`: `run_demo` then pins GLFW to X11
  (`glfw.init_hint(PLATFORM, PLATFORM_X11)`) and PyOpenGL lands on GLX to match
  (`headless-verification.md` §2).

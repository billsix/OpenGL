# Headless verification of the port (method) (L2)

Up: `README.md`. How to run the demos without a display or a real GPU, what that proves and
what it cannot. This is the **method**; the full capture harness from the port's development
history was not carried into this repository — rebuild from the description here if you need
it. `run_demo` already carries the one hook the method needs (`MVP_GLFW_X11`, §2).

## 1. What a no-display sweep can and cannot tell you

| Class | Findable headlessly? | How |
|---|---|---|
| **CRASH** — the demo raises before or during its first frames | yes | non-124 exit code + stderr tail |
| **BLANK** — a window opens and nothing is drawn | yes | ≤3 distinct colours in the capture |
| **RENDERED** — a plausible picture | yes, that it *exists* | >3 colours |
| *Correct* — matches the C original | **no** | needs the C frame side by side, or a human |

`RENDERED` means *worth a human look*, not *right*. A no-display sweep shortens the on-display
pass; it does not replace it — the fidelity verdicts in `fidelity-audit.md` all needed the
maintainer's display. The cheap check this repo ships is `make check` in `python/`, which
imports all 52 demo modules against the referenced `Binaries/` and so catches import-time and
asset-resolution failures without opening a window.

## 2. Running under Xvfb, and the X11/Wayland backend agreement

A GLFW demo has no natural exit: run it under `timeout N` and treat **rc 124 as success**;
anything else non-zero is a crash. Judge pixels, never exit codes — a blank window exits
exactly like a working one.

The load-bearing detail is that **GLFW's backend and PyOpenGL's backend must agree.** PyOpenGL
picks EGL when `WAYLAND_DISPLAY` is set and GLX when it is not; GLFW 3.4 prefers Wayland when
the socket is present. If they disagree, PyOpenGL sees no current context and every ARB-aliased
GL 4.2+ entry point (`glTexStorage2D`, `glBindImageTexture`, `glDispatchCompute`) raises
`NullFunctionError` while `glGetString` still answers "4.6 (Core Profile) Mesa". For a headless
run:

- `Xvfb :99 -screen 0 1280x800x24 &`, share `/tmp/.X11-unix` into the container, `-e DISPLAY=:99`.
- Run with **`MVP_GLFW_X11=1` and `WAYLAND_DISPLAY` unset**. `run_demo` then pins GLFW to X11
  (`glfw.init_hint(PLATFORM, PLATFORM_X11)`) so the window lives on Xvfb, and PyOpenGL lands on
  GLX to match. Without the pin GLFW takes a mounted `wayland-0` (the maintainer's own
  compositor): the window appears on their desktop, and while their screen is off the compositor
  sends no frame callbacks, so every swap blocks and the run stalls.
- `MVP_CONTROLS_WINDOW=1` also draws each demo's Controls-menu body in a plain window every
  frame, so every widget executes headlessly (`c-to-python-rules.md` §3).

## 3. When X captures are black: photograph the back buffer

On recent Mesa a GL window's pixels may not reach Xvfb at all (`import -window root` is one
colour even inside the container). The demo can still be photographed from inside the process,
just before a swap, which is also stronger evidence (no window manager in the way): wrap
`glfw.swap_buffers`, and on the chosen frame read the back buffer with `glReadPixels`.

- **Read the LAST of N frames, not the first.** imgui hides a new window for its auto-fit
  frame, so a frame-1 capture never shows a control panel.
- Keep N small (≈3): under Xvfb a swap takes seconds.
- A single-buffered path (if any) ends its frame on `glFlush`, not a swap — count a flush as a
  frame only while no swap has been seen, and read the front buffer after it.

## 4. Capture timing for the slow CPU demos

Three examples compute on the CPU and print a "please wait" line: **29** (CPU raytracer),
**33** (IBL prefilter) and **37** (CPU raymarcher). They were BLANK at an 8 s budget and
RENDERED at 70 s (37 was still computing at 70 s under llvmpipe). Re-run slow ones with a
120 s+ budget before calling them broken.

## 5. Two parity bugs a Python-only sweep cannot catch

Comparing the C frame to the Python frame found two primitive-level bugs that every affected
demo still RENDERED through (so a crash/blank sweep stayed green) — both now fixed and
documented in `mesa-and-pyopengl-traps.md` §9–10:

- **`make_sphere` was mirror-handed relative to GLUS.** Exterior spheres survived (their demos
  do not cull); the inside-out background sphere of example32, drawn with `glFrontFace(GL_CW)`,
  was culled to nothing. Flipping the front face in a scratch copy brought the backdrop back,
  which pinned it.
- **`load_hdr` filled rows top-down; GLUS stores them bottom-up.** With the backdrop restored
  the panorama was upside down, and its image-based lighting was lit from the wrong directions.
  One `np.flipud`.

The lesson: a no-display sweep proves "does not crash / puts a picture up"; **correctness is a
C-vs-Python comparison**, and the cheapest form of it is to flip one suspected thing in a
scratch copy and watch the picture change.

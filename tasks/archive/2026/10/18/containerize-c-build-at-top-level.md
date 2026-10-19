# Move the container contract to the top level; build the C in the image

**Status:** done 2026-10-18 — implemented and container-verified (`make image` builds all
52 C examples, 51 binaries, Example50 Vulkan-skipped, + installs the port, exit 0; mounted
on-entry build relinks cleanly; `make check` imports 52/52); the maintainer display-confirmed
`make run N=…` (Python/Wayland), `make shell`→`run-c N=…` (C/X11) and `mcnopper N`. The
C-vs-Wayland/GLEW constraint and the `run-c` helper are the settled outcome.
**Priority:** 4
**Difficulty:** 5

## Decisions (maintainer, William Emerison Six <billsix@gmail.com>, 2026-10-18)

1. **Replace** `python/`'s container contract with the top-level one; `python/` keeps only
   its package (`pyproject.toml`, `src/`, `requirements.txt`, `README.md`).
2. **Bake-then-rebuild-on-entry** — the Dockerfile bakes `cmake --build` from the COPYed
   source; the entrypoint rebuilds incrementally against the live mount on `make shell`
   (CMake outputs into the source tree's `Binaries/` + `lib/`, which the mount shadows, so
   the on-entry rebuild is what makes the mounted binaries current).
3. `make shell` drops into the **repo root**.
4. **Also generalize the pattern** into the C/C++ container-per-project contract in
   `runClaudeInContainer` / `runCrushInContainer` (see that repo's own task; this repo is the
   first consumer).
5. **Drop `run.sh`** — it duplicated `make run`/`make list` and went stale after the move;
   the Makefile suffices (maintainer, 2026-10-18).
6. **Build tree at `/opt/ogl-build`, OUTSIDE the `/opengl` mount** — discovered during
   verification that a build tree *under* the mount collides with the host's own `build/`
   CMakeCache (configured for the host path), which cmake refuses. Building outside the mount
   avoids the collision; CMake still emits binaries into the mounted `Binaries/`.

## BLUF

Move the container contract (`Makefile`, `Dockerfile`, `entrypoint/`, `run.sh`) from
`python/` up to the repository top level so **one** `make image` / `make shell` builds and
runs both the C examples and the Python port. The Dockerfile compiles the C sources
(`cmake`) as part of the image build, and `make shell` drops into the directory where the
built binaries and the assets live so both a C binary and a Python demo run without extra
setup. "Done" means: `make image` builds the C (all `ExampleNN`) and installs the Python
port; `make shell` lands you somewhere both `./<a C binary>` and `mcnopper <N>` run against
`Binaries/`.

## Context (read first)

- **Today there is one container contract and it is Python-only:** `python/Makefile`,
  `python/Dockerfile`, `python/entrypoint/`, `python/run.sh`. It is a Fedora + podman
  ephemeral-container template: it mounts the whole repo at `/opengl`, sets
  `MCNOPPER_MEDIA=/opengl/Binaries`, installs the Python runtime deps, and runs the port
  (`make run/list/check/shell`). It installs **no C toolchain**.
- **The C side builds standalone, outside any container** (`AGENTS.md` › Build): `cmake -S
  . -B build/ninja -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build/ninja`.
  Binaries land in `Binaries/`, which is also the runtime cwd (examples load assets by bare
  filename). Dependencies are fetched at configure time (GLFW 3.4, GLEW 2.2.0, GLUS) via
  CMake FetchContent, so the build needs **network at configure time**.
- **Both halves already share `Binaries/`:** the C examples run with it as cwd; the Python
  port reads it through `MCNOPPER_MEDIA` / `find_media`. A single container that has the C
  binaries built into `Binaries/` and the Python port installed can run either.
- **The Makefile is already nested-podman-aware** (`PODMAN_RUN_FLAGS` adds
  `--cgroups=disabled` under `NESTED_PODMAN=1`) and forwards X11/Wayland; the moved-up
  version inherits that. The Wayland client libraries are already installed
  (`tasks/reference/python-port/gl-and-imgui-gotchas.md` §4).

## Plan (subject to the open questions)

1. **Move** `python/{Makefile,Dockerfile,entrypoint/,run.sh}` to the repo top level
   (`git mv`). The mount changes from "repo at `/opengl`, package at `/opengl/python`" to
   the repo root being the build + run root. `python/pyproject.toml`, `requirements.txt`,
   `src/`, and `README.md` stay in `python/`.
2. **Add the C build toolchain to the Dockerfile:** `cmake`, `ninja-build`, `gcc-g++`
   (present), `git` (FetchContent), and the dev packages the fetched deps and the examples
   need — `glew-devel`, `glfw-devel`, `mesa-libGL-devel`, `mesa-libEGL-devel`,
   `wayland-devel`, `libxkbcommon-devel`, and the `libX*-devel` set. (The runtime `libX*` /
   `libwayland-*` packages are already there.)
3. **Build the C in the image:** after `COPY`ing the source in, run the standalone cmake
   build so `ExampleNN` binaries are produced. Decide the build/output location against the
   mount-shadowing problem (open question 2).
4. **Entry point:** `make shell` lands in the directory holding the binaries + `Binaries/`
   (open question 3), with `MCNOPPER_MEDIA` set and the Python port editable-installed (as
   now). The entrypoint keeps both builds current against the live mount (open question 2).
5. **Top-level Makefile targets:** keep the Python targets (`run`/`list`/`check` → `mcnopper`)
   and add C targets (build, and a way to run a C example). Update `python/README.md` and
   `AGENTS.md` to point at the top-level commands; the standalone `cmake` instructions in
   `AGENTS.md` stay valid for a no-container build.
6. **Verify** (nested): `make image` exits 0 and produced the C binaries; `make shell` →
   a C binary runs and `mcnopper 5` runs; `make check` still imports all 52. On-display runs
   remain the maintainer's.

## Open questions (block implementation)

1. **Does the top-level contract replace `python/`'s, or coexist?** Recommend **replace** —
   one contract at the top level that does both C and Python; `python/` keeps only its
   package (`pyproject.toml`, `src/`, `requirements.txt`, `README.md`). Keeping two
   Dockerfiles/Makefiles would drift.
2. **How to keep the baked C build from being shadowed by the live mount.** `make shell`
   mounts the live repo over the build root, so a build baked into `Binaries/` at image-build
   time disappears under the mount (the same tension the Python editable install handles by
   re-installing on entry). Recommend the **bake + rebuild-on-entry** pattern: the Dockerfile
   bakes `cmake --build` from the COPYed source (so a *pulled* image runs standalone), and the
   entrypoint runs `cmake --build` again — incremental — against the live mount on `make
   shell`, so the binaries always match the mounted source. OK to adopt that pattern?
3. **Which directory should `make shell` drop into** — the repo root (both `python/` and
   `Binaries/` reachable; run a C example with `cd Binaries && ./ExampleNN`), or `Binaries/`
   directly (C examples run from there by design, but `python/` is then two levels away)?
   Recommend the **repo root**, and document the two run commands.

## Known risks / gotchas

- **The headless GLEW/GLUS gotchas are already handled in `CMakeLists.txt`** (verified
  2026-10-18): it `PATCH_COMMAND`-patches GLUS's `glus_window_glfw.c` to accept
  `GLEW_ERROR_NO_GLX_DISPLAY` (the Mesa/GLX `glewInit()` returns 4 issue), and patches
  glew-cmake's stale `cmake_minimum_required`. So the in-container build needs no manual
  patching. If a demo still fails at `glewInit` on Mesa, that patch is where to look.
- **FetchContent needs network during `make image`** (GLFW/GLEW/GLUS). The image build already
  has network for `dnf`, so this is fine, but it makes the build non-hermetic.
- **Image size grows** (cmake + dev libraries + the fetched/compiled deps). Acceptable for a
  dev image; a lean variant is out of scope.
- **CI** (`.github/workflows/`) builds the C standalone today; moving the container contract
  up does not change that, but if CI later uses the container, the move is a prerequisite —
  note, do not wire it here.

## References

- `AGENTS.md` › Build (standalone) — the canonical cmake invocation and the `Binaries/` cwd rule.
- `python/Makefile`, `python/Dockerfile`, `python/entrypoint/` — the contract being moved.
- `tasks/reference/python-port/gl-and-imgui-gotchas.md` §4 — Wayland libs, `find_media`, the
  X11/Wayland backend agreement a run needs.

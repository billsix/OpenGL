#!/bin/env bash
# `make shell`: build the C + install the Python port against the mounted repo,
# then drop into an interactive shell at the repo root. The container has BOTH
# display backends:
#   cd python && mcnopper 5     # Python demo — Wayland (or X11), just works
#   run-c 5                     # C Example05 — forced onto X11 (see below)
# The C examples use GLUS + GLEW (GLX-only); GLFW would otherwise pick Wayland
# when WAYLAND_DISPLAY is set and GLEW could not draw, so `run-c` launches them
# with WAYLAND_DISPLAY unset (→ X11/XWayland).

export VIRTUAL_ENV_DISABLE_PROMPT=1
source /venv/bin/activate
export MCNOPPER_MEDIA="${MCNOPPER_MEDIA:-/opengl/Binaries}"
cd /opengl
loadpackages.sh

# Define the `run-c` helper in the interactive shell (sourced from ~/.bashrc).
cat > /root/.opengl-shell-rc <<'RC'
# Run a C example on X11 (GLEW needs GLX, so Wayland is unset for it). Usage: run-c 5
run-c() {
  if [ -z "$1" ]; then echo "usage: run-c <example-number 1..52>"; return 2; fi
  local n; n=$(printf '%02d' "$1")
  ( cd /opengl/Binaries && env -u WAYLAND_DISPLAY "./Example$n" )
}
RC
grep -qxF 'source /root/.opengl-shell-rc' /root/.bashrc 2>/dev/null \
    || echo 'source /root/.opengl-shell-rc' >> /root/.bashrc

echo
echo "OpenGL dev shell (repo root: /opengl). Both display backends available."
echo "  Python demo:  cd python && mcnopper <N>     (Wayland)"
echo "  C example:    run-c <N>                     (X11 — GLEW is GLX-only)"
echo

exec bash

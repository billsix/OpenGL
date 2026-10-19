#!/bin/env bash
# Prepare the mounted repo: build the C examples (incremental) and editable-install
# the Python port. Sourced/called by shell.sh on `make shell`. No `set -e`, so a
# C build failure still lets the shell open (the errors print; you debug inside).

export VIRTUAL_ENV_DISABLE_PROMPT=1
source /venv/bin/activate
export MCNOPPER_MEDIA="${MCNOPPER_MEDIA:-/opengl/Binaries}"

# Build the C examples into /opengl/Binaries. The build tree is /opt/ogl-build,
# OUTSIDE the mount (the baked image already configured + built it, so this is a
# fast incremental relink against the live source; a build tree under /opengl
# would collide with the host's own build/ cache). See the Dockerfile note.
cmake -S /opengl -B /opt/ogl-build -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build /opt/ogl-build

# Editable-install the Python port.
( cd /opengl/python && \
  uv pip install --no-deps --no-index --no-build-isolation -e . --python "$(which python)" )

#!/bin/env bash
# Default entrypoint: used by `make check`. Installs the Python port and imports
# every demo module as a no-display health check. It does NOT build the C (that
# is `make build-c` / baked in the image / done on `make shell`), so `make check`
# stays a fast Python-only gate.

export VIRTUAL_ENV_DISABLE_PROMPT=1
source /venv/bin/activate

# The demos load their assets from the repo's Binaries/ folder, mounted at
# /opengl/Binaries. Default MCNOPPER_MEDIA to it when the caller did not set it.
export MCNOPPER_MEDIA="${MCNOPPER_MEDIA:-/opengl/Binaries}"

cd /opengl/python/

# Install the package in editable mode (deps are pre-installed in the image).
uv pip install --no-deps --no-index --no-build-isolation -e . --python "$(which python)"

# Quick health check: make sure every demo module imports. Most demos are flat
# modules (mcnopper.exampleNN); the four with co-located shaders are a
# subpackage (mcnopper.exampleNN.exampleNN).
python -c "
import importlib, sys
from pathlib import Path
import mcnopper
root = Path(mcnopper.__file__).resolve().parent
errors = []
for n in range(1, 53):
    if (root / f'example{n:02d}.py').exists():
        name = f'mcnopper.example{n:02d}'
    else:
        name = f'mcnopper.example{n:02d}.example{n:02d}'
    try:
        importlib.import_module(name)
    except Exception as e:
        errors.append(name)
        print(f'FAIL {name}: {e!r}')
print(f'imported {52 - len(errors)} / 52 demos')
sys.exit(0 if not errors else 1)
"

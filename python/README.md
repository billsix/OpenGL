# McNopper OpenGL Examples — Python Port

A Python port of the 52 C/GLUS examples in this repository (Norbert Nopper's
MIT-licensed OpenGL 4 examples). The originals are C + GLUS + EGL/GLFW; this port
uses Python + PyOpenGL + glfw + Dear ImGui and renders the same scenes.

It lives in `python/` inside the C repo on purpose: the demos **reuse this repo's
own files** instead of shipping copies — `Binaries/` for assets (meshes, textures,
HDRIs, glTF scenes) and each C example's `ExampleNN/shader/*.glsl` for shaders
(resolved at runtime by `common.find_shader`). Only the shaders that differ from
the C stay in the package: Example13's one edited shader, examples 48–50's
version-lowered background/fullscreen shaders, and the shared GLUS PBR/IBL shaders
in `src/mcnopper/glus_shaders/` (which have no committed C source here). See
`../tasks/reference/python-port/c-to-python-rules.md` §6.

## Layout

```
python/
  src/mcnopper/
    common.py             framework: window/loop, shader compile, OBJ/TGA/HDR loaders, find_media
    gltf.py gltf_gl.py    glTF 2.0 + PBR (examples 48-52)
    ibl.py perlin.py sh.py
    cli.py                console-script entry point (mcnopper.cli:main)
    glus_shaders/         shared GLUS PBR/IBL shaders (verbatim)
    example01.py … example52.py  one flat module per demo (shaders read from the
                               C ../../../ExampleNN/shader/, except the few kept
                               copies that differ; example13/48/49/50 stay nested)
  pyproject.toml requirements.txt          packaging
```

The container contract (`Dockerfile`, `Makefile`, `entrypoint/`) lives at the
**repository root** — one image builds both the C examples and this port.

Assets are **not** stored here. `common.find_media(name)` resolves a filename
against `$MCNOPPER_MEDIA` first; the container sets `MCNOPPER_MEDIA=/opengl/Binaries`
so `find_media("crate.tga")`, `find_media("einstein/scene.gltf")`,
`find_media("sponza.obj")`, etc. resolve against the repo's `Binaries/`.

> The `.hdr` and `.bin` assets in `Binaries/` are git-lfs-tracked. A fresh clone
> needs `git lfs pull` (or `git lfs` installed at clone time) so the real bytes are
> present; otherwise examples 32/33 (HDR) and 48–52 (glTF `.bin`) will fail to load.

## Running (containerised — recommended)

The container contract lives at the **repository root** (one image builds both the
C examples and this Python port). From the repo root:

```bash
make image            # build the image (builds the C examples + installs the port)
make list             # list all 52 Python demos with their descriptions
make run N=5          # run Python demo 5 on your display
make run-c N=5        # run the C Example05 on your display
make shell            # shell at the repo root (C built, port installed)
make check            # import every Python demo module against Binaries/ (no display)
```

The run targets mount the repo root at `/opengl`, set
`MCNOPPER_MEDIA=/opengl/Binaries`, and forward your X11/Wayland display. Running
nested inside a podman-in-podman sandbox is handled automatically
(`NESTED_PODMAN=1` adds `--cgroups=disabled`).

## Running (directly, no container)

Install the package editable (`pip install -e .`), which also pulls the deps. The
demos import the installed `mcnopper` package, so this one step enables both the
console script and running a demo module by path. From a checkout they find this
repo's `Binaries/` on their own (resolved relative to the package), so no setup is
needed:

```bash
pip install -e .                      # editable install: console script + deps
mcnopper 5                            # by number
mcnopper list                         # list all demos
python src/mcnopper/example05.py      # or run a demo module by path

# Optional: point the port at a Binaries/ elsewhere (overrides the above):
export MCNOPPER_MEDIA=/path/to/Binaries
```

Each demo is a standalone script. The menubar's Controls menu (File → Quit, View →
Fullscreen F11) exposes the rotation / material / light parameters that the C
versions drove from keyboard shortcuts.

## Dependencies

Python 3.10+, PyOpenGL, glfw, numpy, imgui (optional but recommended), and —
for the glTF/PBR demos (48–52) — pygltflib and Pillow. See `requirements.txt`
and the repo's `THIRD-PARTY.md` for licenses.

## Relationship to the C examples

- Per-example shaders are the C originals, read from `ExampleNN/shader/` at runtime;
  pass structure, FBO layouts and binding indices match (that is why they can be shared).
- `load_obj` collapses an OBJ into a single attribute-array draw (visual parity for
  the single-material OBJs bundled here).
- ImGui replaces the C examples' keyboard shortcuts.
- A few CPU-heavy demos (29 raytracer, 37 raymarcher) render at lower resolution
  because Python is slower than C; tweak the constants at the top of those scripts.

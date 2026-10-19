"""Shared helpers for the McNopper OpenGL port.

This module replaces the C-side GLUS framework with a glfw + PyOpenGL +
imgui stack while preserving the per-example init/reshape/update/terminate
shape that makes the original demos legible.

Conventions baked into this file (from prior porting experience):

* Always create and bind a default VAO right after context creation; macOS
  Core Profile rejects VAO 0.
* Pass ``validate=False`` to ``shaders.compileProgram`` so multi-sampler
  programs don't fail Apple's link-time validator before the app sets units.
* Auto-bump ``#version`` to 420 if shader source uses ``layout(binding=...)``;
  Mesa is strict about that combination.
* Resolve every asset path through ``find_media`` so a script run from any
  directory can find the bundled ``media/`` tree.
* Keep imgui as the primary control surface for every demo (the panel
  callback draws inside the menubar's Controls menu); keys are an
  optional accelerator only.
"""

from __future__ import annotations

import ctypes
import math
import os
import re
import struct
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np

# Convenience aliases for common 3-vector / 4-vector parameter types.
Vec3 = Union[Sequence[float], np.ndarray]
Vec4 = Union[Sequence[float], np.ndarray]

# --- PyOpenGL pointer-storage workaround ----------------------------------
# Recent PyOpenGL silently ignores STORE_POINTERS=False (and prints a notice
# if you try to set it). Skip the flag entirely; instead patch
# contextdata.getContext to return a constant id so per-context dictionaries
# don't pile up references to short-lived ctypes arrays.
import OpenGL

OpenGL.ERROR_CHECKING = True

from OpenGL import (
    GL,
    contextdata,
)
from OpenGL.constant import Constant
from OpenGL.GL import shaders

#: PyOpenGL's GL constants are ``Constant`` instances, not plain ``int``s,
#: so parameters taking a GL enum accept either (house pattern; mirrors
#: ``mvpvisualization/_pipeline.py``).
GLenum = int | Constant

_orig_get_context = contextdata.getContext
contextdata.getContext = lambda context=None: 0  # ty: ignore[invalid-assignment]  # deliberate monkey-patch (see note above)

import glfw

HAS_IMGUI: bool = False
imgui = None  # type: ignore[assignment]
GlfwRenderer = None  # type: ignore[assignment]
try:
    from imgui_bundle import imgui as _imgui
    from imgui_bundle.python_backends.glfw_backend import (
        GlfwRenderer as _GlfwRenderer,
    )

    imgui = _imgui
    GlfwRenderer = _GlfwRenderer
    HAS_IMGUI = True
except Exception as _imgui_err:  # noqa: BLE001
    print(
        f"[McNopper] imgui_bundle unavailable, panels will be hidden: {_imgui_err}"
    )


# --- Path resolution ------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent


def find_media(filename: str) -> str:
    """Locate a media file shipped under ``mcnopper/media/``.

    Resolution order:
      1. ``$MCNOPPER_MEDIA``                (env override)
      2. ``<repo>/Binaries``                (in-repo checkout, found by
         walking up from this file to the nearest ancestor with a
         ``Binaries/`` dir -- so demos run from a checkout with no env var)
      3. ``<script>/media``                 (dev-mode same-dir layout)
      4. ``<script>/../media``              (per-example sibling layout)
      5. ``<this file>/media``              (in-tree canonical layout)
      6. ``importlib.resources("mcnopper")/media``  (post-pip-install)
    """
    candidates: list[Path] = []
    env = os.environ.get("MCNOPPER_MEDIA")
    if env:
        candidates.append(Path(env))
    # In-repo dev layout: the C repo's Binaries/ sits above this package
    # (<repo>/python/src/mcnopper/_common.py -> <repo>/Binaries). Walk up to
    # the nearest ancestor that has a Binaries/ dir so the demos find assets
    # whether run in-container or straight from a checkout, no env var needed.
    # After a pip install into site-packages no ancestor has one, so this adds
    # nothing and resolution falls through to the media/ candidates below.
    for ancestor in (_THIS_DIR, *_THIS_DIR.parents):
        repo_binaries: Path = ancestor / "Binaries"
        if repo_binaries.is_dir():
            candidates.append(repo_binaries)
            break
    try:
        caller = Path(sys.argv[0]).resolve().parent
        candidates.append(caller / "media")
        candidates.append(caller.parent / "media")
    except Exception:
        pass
    candidates.append(_THIS_DIR / "media")
    try:
        import importlib.resources as _resources

        pkg_media = _resources.files("mcnopper").joinpath("media")
        # importlib.resources returns a Traversable; coerce to Path for our use.
        candidates.append(Path(str(pkg_media)))
    except Exception:
        pass

    for base in candidates:
        p = base / filename
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"media/{filename} not found; tried: {[str(c) for c in candidates]}"
    )


def find_shader(script_file: str, filename: str) -> str:
    """Locate a per-example shader file.

    Resolution order:
      1. beside the demo (``here``, ``here/shader``, ``here/../shader``)
      2. this repo's C example shaders: ``exampleNN`` -> ``<repo>/ExampleNN/shader``

    The port references each C example's shaders directly rather than keeping a
    copy; only shaders that differ from the C (Example13's ``u_tessLevel`` edit,
    48-50's version-lowered background/fullscreen) keep a local copy, which the
    local-first order above picks over the C sibling. Shaders travelling with the
    package (``glus_shaders/``) are loaded elsewhere, not through here.
    """
    here: Path = Path(script_file).resolve().parent
    bases: list[Path] = [here, here / "shader", here.parent / "shader"]
    # Map demo ``exampleNN`` -> the C example's ``ExampleNN/shader`` in this
    # repo, found by walking up to the ancestor that holds it. Use the script's
    # own stem (``exampleNN``) so this works whether the demo is a flat
    # ``exampleNN.py`` or the nested ``exampleNN/exampleNN.py``.
    stem: str = Path(script_file).stem
    if stem.startswith("example"):
        c_dir_name: str = "Example" + stem[len("example"):]
        for ancestor in here.parents:
            c_shader_dir: Path = ancestor / c_dir_name / "shader"
            if c_shader_dir.is_dir():
                bases.append(c_shader_dir)
                break
    for base in bases:
        p: Path = base / filename
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"shader {filename!r} not found near {script_file}")


# --- Shader / program helpers --------------------------------------------

_LAYOUT_BINDING = re.compile(
    r"layout\s*\(\s*[^)]*binding\s*=", flags=re.IGNORECASE
)


def _maybe_bump_version(src: str) -> str:
    if not _LAYOUT_BINDING.search(src):
        return src
    m = re.search(r"#version\s+(\d+)\s*([a-z]*)", src)
    if not m:
        return src
    ver: int = int(m.group(1))
    if ver >= 420:
        return src
    profile = m.group(2) or "core"
    return src[: m.start()] + f"#version 420 {profile}" + src[m.end() :]


def compile_program(
    vert_src: str | None = None,
    frag_src: str | None = None,
    tesc_src: str | None = None,
    tese_src: str | None = None,
    geom_src: str | None = None,
    comp_src: str | None = None,
    tf_varyings: list[str] | None = None,
    tf_mode: GLenum = GL.GL_INTERLEAVED_ATTRIBS,
) -> int:
    """Compile + link a program. Returns the GL program id."""
    objs = []
    if vert_src is not None:
        objs.append(
            shaders.compileShader(
                _maybe_bump_version(vert_src), GL.GL_VERTEX_SHADER
            )
        )
    if tesc_src is not None:
        objs.append(
            shaders.compileShader(
                _maybe_bump_version(tesc_src), GL.GL_TESS_CONTROL_SHADER
            )
        )
    if tese_src is not None:
        objs.append(
            shaders.compileShader(
                _maybe_bump_version(tese_src), GL.GL_TESS_EVALUATION_SHADER
            )
        )
    if geom_src is not None:
        objs.append(
            shaders.compileShader(
                _maybe_bump_version(geom_src), GL.GL_GEOMETRY_SHADER
            )
        )
    if frag_src is not None:
        objs.append(
            shaders.compileShader(
                _maybe_bump_version(frag_src), GL.GL_FRAGMENT_SHADER
            )
        )
    if comp_src is not None:
        objs.append(
            shaders.compileShader(
                _maybe_bump_version(comp_src), GL.GL_COMPUTE_SHADER
            )
        )

    if tf_varyings:
        prog: int = GL.glCreateProgram()
        for o in objs:
            GL.glAttachShader(prog, o)
        n = len(tf_varyings)
        # PyOpenGL's wrapper expects POINTER(POINTER(c_char)). A bare
        # `(c_char_p * n)` array carries the right contents but ctypes
        # rejects it as the wrong type; cast through.
        encoded = [v.encode("utf-8") for v in tf_varyings]
        arr = (ctypes.c_char_p * n)(*encoded)
        ptr = ctypes.cast(arr, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
        GL.glTransformFeedbackVaryings(prog, n, ptr, tf_mode)
        # Keep `arr` alive until the link consumes it.
        _keepalive_arr = arr
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            log = GL.glGetProgramInfoLog(prog).decode(errors="replace")
            raise RuntimeError(f"link error: {log}")
        for o in objs:
            GL.glDeleteShader(o)
        return prog

    return shaders.compileProgram(*objs, validate=False)


def compile_program_files(script_file: str, **kw: Any) -> int:
    """Like compile_program, but each kw is a filename relative to the demo dir.

    Usage: compile_program_files(__file__, vert="phong.vert.glsl", frag="phong.frag.glsl")
    """
    src_kw = {}
    for stage in ("vert", "frag", "tesc", "tese", "geom", "comp"):
        fn = kw.pop(stage, None)
        if fn is not None:
            with open(find_shader(script_file=script_file, filename=fn)) as f:
                src_kw[f"{stage}_src"] = f.read()
    # The splats carry disjoint keys (``*_src`` strings from the loop above,
    # everything else forwarded untouched), but ty types ``src_kw`` as
    # ``dict[str, str]`` and cannot rule out a string landing on ``tf_mode``
    # / ``tf_varyings``.
    return compile_program(**src_kw, **kw)  # ty: ignore[invalid-argument-type]


def uniform(prog: int, name: str) -> int:
    return GL.glGetUniformLocation(program=prog, name=name)


def attrib(prog: int, name: str) -> int:
    return GL.glGetAttribLocation(program=prog, name=name)


# --- TGA loader -----------------------------------------------------------
# Supports uncompressed and RLE TGAs (types 2, 3, 10, 11), 8/16/24/32 bit,
# with and without alpha. Returns (width, height, channels, bytes-as-rgba).


@dataclass
class TgaImage:
    width: int
    height: int
    channels: int  # 1 (grey), 3 (rgb), 4 (rgba)
    # tightly packed; bottom-up if origin bit not set. The BGR->RGB swizzle
    # produces bytes; the grey/8-bit path leaves the decoded bytearray as-is.
    pixels: bytes | bytearray


def load_tga(filename: str) -> TgaImage:
    with open(filename, mode="rb") as f:
        data = f.read()
    id_len = data[0]
    cmap_type = data[1]
    image_type = data[2]
    cmap_len = struct.unpack_from("<H", data, offset=5)[0]
    cmap_size = data[7]
    width = struct.unpack_from("<H", data, offset=12)[0]
    height = struct.unpack_from("<H", data, offset=14)[0]
    bpp = data[16]
    image_desc = data[17]

    offset = 18 + id_len + cmap_type * cmap_len * (cmap_size // 8)
    pixel_size = bpp // 8

    is_rle = image_type in (10, 11)
    is_grey = image_type in (3, 11)

    pix_bytes = bytearray()
    if is_rle:
        target = width * height * pixel_size
        i = offset
        while len(pix_bytes) < target:
            head = data[i]
            i += 1
            count = (head & 0x7F) + 1
            if head & 0x80:
                pixel = data[i : i + pixel_size]
                i += pixel_size
                pix_bytes.extend(pixel * count)
            else:
                pix_bytes.extend(data[i : i + pixel_size * count])
                i += pixel_size * count
    else:
        pix_bytes.extend(data[offset : offset + width * height * pixel_size])

    if pixel_size in (3, 4):
        # BGR(A) -> RGB(A)
        arr: np.ndarray = (
            np.frombuffer(bytes(pix_bytes), dtype=np.uint8)
            .reshape(-1, pixel_size)
            .copy()
        )
        arr[:, [0, 2]] = arr[:, [2, 0]]
        pix_bytes = arr.tobytes()

    channels = 1 if is_grey else pixel_size
    # Match GLUS's `glusImageLoadTga`: upload TGA bytes as-is. The McNopper
    # assets are authored bottom-up (image_descriptor bit 5 = 0), which is
    # the same convention OpenGL uses for `glTexImage2D` data, so no flip
    # is needed for 2D textures.
    return TgaImage(
        width=width, height=height, channels=channels, pixels=pix_bytes
    )


def upload_tga_2d(
    filename: str,
    internal: GLenum | None = None,
    mipmaps: bool = True,
    wrap: GLenum = GL.GL_REPEAT,
    min_filter: GLenum = GL.GL_LINEAR_MIPMAP_LINEAR,
    mag_filter: GLenum = GL.GL_LINEAR,
) -> int:
    img = load_tga(filename)
    fmt = {1: GL.GL_RED, 3: GL.GL_RGB, 4: GL.GL_RGBA}[img.channels]
    if internal is None:
        internal = {1: GL.GL_R8, 3: GL.GL_RGB8, 4: GL.GL_RGBA8}[img.channels]
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        internal,
        img.width,
        img.height,
        0,
        fmt,
        GL.GL_UNSIGNED_BYTE,
        img.pixels,
    )
    if not mipmaps:
        if min_filter == GL.GL_LINEAR_MIPMAP_LINEAR:
            min_filter = GL.GL_LINEAR
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, min_filter)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, mag_filter)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap)
    if mipmaps:
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tex


def upload_cubemap_tgas(faces: list[str]) -> int:
    """faces: [+X, -X, +Y, -Y, +Z, -Z]"""
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, tex)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    targets = [
        GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        GL.GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        GL.GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
        GL.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        GL.GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
        GL.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
    ]
    for tgt, fn in zip(targets, faces):
        img = load_tga(fn)
        fmt = {1: GL.GL_RED, 3: GL.GL_RGB, 4: GL.GL_RGBA}[img.channels]
        internal = {1: GL.GL_R8, 3: GL.GL_RGB8, 4: GL.GL_RGBA8}[img.channels]
        GL.glTexImage2D(
            tgt,
            0,
            internal,
            img.width,
            img.height,
            0,
            fmt,
            GL.GL_UNSIGNED_BYTE,
            img.pixels,
        )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE
    )
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
    return tex


# --- Radiance .HDR loader -------------------------------------------------


def load_hdr(filename: str) -> tuple[int, int, np.ndarray]:
    """Load a Radiance .hdr file. Returns (width, height, float32 RGB array)."""
    with open(filename, mode="rb") as f:
        data = f.read()
    nl = data.index(b"\n")
    sig = data[:nl]
    assert sig.startswith(b"#?"), "not an HDR file"
    i = nl + 1
    fmt = "rgb"
    while True:
        nl2 = data.index(b"\n", i)
        line = data[i:nl2]
        i = nl2 + 1
        if not line:
            break
        if line.upper().startswith(b"FORMAT="):
            if b"XYZE" in line.upper():
                fmt = "xyz"
    res_nl = data.index(b"\n", i)
    res = data[i:res_nl].decode()
    i = res_nl + 1
    parts = res.split()
    assert len(parts) == 4
    if parts[0] == "-Y":
        height, width = int(parts[1]), int(parts[3])
    else:
        width, height = int(parts[1]), int(parts[3])

    pixels: np.ndarray = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        if i + 4 > len(data):
            break
        a, b, c, d = data[i], data[i + 1], data[i + 2], data[i + 3]
        if a == 2 and b == 2 and (c & 0x80) == 0:
            scanline_w = (c << 8) | d
            i += 4
            scan: np.ndarray = np.zeros((4, scanline_w), dtype=np.uint8)
            for ch in range(4):
                x = 0
                while x < scanline_w:
                    run = data[i]
                    i += 1
                    if run > 128:
                        n = run - 128
                        v = data[i]
                        i += 1
                        scan[ch, x : x + n] = v
                        x += n
                    else:
                        scan[ch, x : x + run] = np.frombuffer(
                            data[i : i + run], dtype=np.uint8
                        )
                        i += run
                        x += run
            r, g, b_, e = scan
            mant = e.astype(np.int32) - 128
            scale: np.ndarray = np.where(
                e == 0, 0.0, np.ldexp(1.0, mant - 8)
            ).astype(np.float32)
            pixels[y, :, 0] = r * scale
            pixels[y, :, 1] = g * scale
            pixels[y, :, 2] = b_ * scale
        else:
            # uncompressed scanline
            for x in range(width):
                r, g, b_, e = data[i], data[i + 1], data[i + 2], data[i + 3]
                i += 4
                if e == 0:
                    pixels[y, x] = 0
                else:
                    s = math.ldexp(1.0, e - 128 - 8)
                    pixels[y, x] = (r * s, g * s, b_ * s)
    # GLUS's glusImageLoadHdr stores the file's FIRST scanline in the LAST row
    # (it starts at y = height - 1 and walks down): the Radiance "-Y +X" file is
    # top-down, GL's texture origin is bottom-left. Without this flip every HDR
    # panorama was upside down -- found 2026-09-09 when example32's restored
    # backdrop showed the Venice courtyard floor at the top of the frame.
    return width, height, np.ascontiguousarray(np.flipud(pixels))


def upload_hdr_2d(filename: str) -> int:
    """Load a Radiance .hdr file and upload it as a float RGB texture."""
    w, h, px = load_hdr(filename)
    return upload_hdr_pixels(w=w, h=h, px=px)


def upload_hdr_pixels(w: int, h: int, px: np.ndarray) -> int:
    """Upload a ``(h, w, 3)`` float32 array as a GL_RGB32F texture."""
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    # 32-bit float: a 16-bit float texel saturates to infinity above 65504,
    # and outdoor HDR suns exceed that (sunny_rose_garden_4k.hdr peaks at
    # 77824); one infinite texel then poisons every IBL map filtered from it.
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F, w, h, 0, GL.GL_RGB, GL.GL_FLOAT, px
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tex


# --- Wavefront OBJ loader -------------------------------------------------


@dataclass
class ObjMaterial:
    """One ``newmtl`` block of a ``.mtl`` file, GLUS's ``GLUSmaterial``.

    Colours are RGBA with alpha 1; defaults are GLUS's
    (``glusWavefrontInitMaterial``: black, shininess 0, transparency 1).
    Texture filenames are resolved against the ``.mtl`` file's directory, so
    they can be passed straight to ``upload_tga_2d``; an empty string means
    the material has no such map. The ``*_texture_name`` fields are the GL
    texture objects a demo creates for the maps, 0 until it does (the C demos
    store theirs in the material too).
    """

    name: str
    emissive: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    ambient: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    diffuse: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    specular: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    shininess: float = 0.0
    transparency: float = 1.0
    reflection: bool = False
    refraction: bool = False
    index_of_refraction: float = 1.0
    emissive_texture: str = ""
    ambient_texture: str = ""
    diffuse_texture: str = ""
    specular_texture: str = ""
    transparency_texture: str = ""
    bump_texture: str = ""
    emissive_texture_name: int = 0
    ambient_texture_name: int = 0
    diffuse_texture_name: int = 0
    specular_texture_name: int = 0
    transparency_texture_name: int = 0
    bump_texture_name: int = 0


@dataclass
class ObjGroup:
    """A ``g``/``o``/``usemtl`` run of faces: GLUS's ``GLUSgroup``.

    ``first``/``count`` delimit the group's triangles in ``ObjMesh.indices``
    (``indices[first:first + count]``); ``material`` is the ``.mtl`` entry
    ``material_name`` resolved to, or ``None`` when no material of that name
    was loaded.
    """

    name: str
    material_name: str = ""
    material: ObjMaterial | None = None
    first: int = 0
    count: int = 0


@dataclass
class ObjMesh:
    positions: np.ndarray  # (N,4) float32, w=1
    normals: np.ndarray  # (N,3) float32
    texcoords: np.ndarray  # (N,2) float32
    indices: np.ndarray  # (M,)  uint32, triangles
    groups: list[ObjGroup] = field(default_factory=list)
    materials: dict[str, ObjMaterial] = field(default_factory=dict)


def _read_color4(parts: list[str]) -> tuple[float, float, float, float]:
    return (float(parts[1]), float(parts[2]), float(parts[3]), 1.0)


def load_mtl(filename: str, materials: dict[str, ObjMaterial]) -> None:
    """GLUS's ``glusWavefrontLoadMaterial``: parse a ``.mtl`` into ``materials``.

    Keywords are matched case-insensitively, as GLUS lower-cases them (the
    chess ``.mtl`` files write ``kd``/``map_kd``); lines before the first
    ``newmtl`` are skipped; ``d`` and ``Tr`` both set ``transparency``, the
    later one winning; ``illum`` sets only the reflection/refraction flags.
    Map filenames are resolved against the ``.mtl`` file's directory.
    """
    mtl_dir: str = os.path.dirname(filename)
    current: ObjMaterial | None = None
    with open(filename) as f:
        for raw in f:
            parts: list[str] = raw.split()
            if not parts or parts[0].startswith("#"):
                continue
            key: str = parts[0].lower()
            if key == "newmtl":
                current = ObjMaterial(name=parts[1] if len(parts) > 1 else "")
                materials[current.name] = current
                continue
            if current is None:
                continue
            tex: str = os.path.join(mtl_dir, parts[1]) if len(parts) > 1 else ""
            match key:
                case "ke":
                    current.emissive = _read_color4(parts)
                case "ka":
                    current.ambient = _read_color4(parts)
                case "kd":
                    current.diffuse = _read_color4(parts)
                case "ks":
                    current.specular = _read_color4(parts)
                case "ns":
                    current.shininess = float(parts[1])
                case "d" | "tr":
                    current.transparency = float(parts[1])
                case "ni":
                    current.index_of_refraction = float(parts[1])
                case "map_ke":
                    current.emissive_texture = tex
                case "map_ka":
                    current.ambient_texture = tex
                case "map_kd":
                    current.diffuse_texture = tex
                case "map_ks":
                    current.specular_texture = tex
                case "map_d" | "map_tr":
                    current.transparency_texture = tex
                case "map_bump" | "bump":
                    current.bump_texture = tex
                case "illum":
                    illum: int = int(parts[1])
                    current.reflection = illum in (3, 4, 5, 6, 7, 8, 9)
                    current.refraction = illum in (6, 7)
                case _:
                    pass  # Tf, sharpness, decal, disp, ...: GLUS ignores them


def load_obj(filename: str) -> ObjMesh:
    """GLUS's ``glusWavefrontLoad``: positions, normals, texcoords, groups.

    Faces are fan-triangulated; vertices are shared between faces that use
    the same ``v/vt/vn`` triple (GLUS emits one vertex per face corner, which
    draws the same picture). Group bookkeeping follows GLUS: ``g`` and ``o``
    start a group; ``usemtl`` names the current group's material, or starts
    a new group named after the material when the current one already has
    one; faces before the first group are credited to the first group. The
    ``mtllib`` file is looked up next to the ``.obj``.
    """
    v: list[list[float]] = []
    vn: list[list[float]] = []
    vt: list[list[float]] = []
    faces: list[tuple[tuple[int, int, int], ...]] = []
    groups: list[ObjGroup] = []
    materials: dict[str, ObjMaterial] = {}
    # Triangles emitted so far when each group began (GLUS's ``counter``).
    group_starts: list[int] = []
    obj_dir: str = os.path.dirname(filename)
    with open(filename) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            tag = parts[0]
            if tag == "v":
                v.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "vn":
                vn.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "vt":
                vt.append([float(parts[1]), float(parts[2])])
            elif tag == "f":
                face = []
                for t in parts[1:]:
                    bits = t.split("/")
                    pi: int = int(bits[0]) - 1
                    ti: int = (
                        int(bits[1]) - 1 if len(bits) > 1 and bits[1] else -1
                    )
                    ni: int = (
                        int(bits[2]) - 1 if len(bits) > 2 and bits[2] else -1
                    )
                    face.append((pi, ti, ni))
                # fan triangulation
                for k in range(1, len(face) - 1):
                    faces.append((face[0], face[k], face[k + 1]))
            elif tag == "mtllib":
                if len(parts) > 1:
                    load_mtl(
                        filename=os.path.join(obj_dir, parts[1]),
                        materials=materials,
                    )
            elif tag == "usemtl":
                name: str = parts[1] if len(parts) > 1 else ""
                if not groups or groups[-1].material_name:
                    groups.append(ObjGroup(name=name))
                    group_starts.append(len(faces) if len(groups) > 1 else 0)
                groups[-1].material_name = name
            elif tag in ("g", "o"):
                groups.append(ObjGroup(name=parts[1] if len(parts) > 1 else ""))
                group_starts.append(len(faces) if len(groups) > 1 else 0)
    group_starts.append(len(faces))
    for g, start, stop in zip(groups, group_starts, group_starts[1:]):
        g.first = 3 * start
        g.count = 3 * (stop - start)
        g.material = materials.get(g.material_name)

    out_pos, out_nor, out_uv = [], [], []
    cache = {}
    indices = []
    for tri in faces:
        for vi, ti, ni in tri:
            key = (vi, ti, ni)
            j = cache.get(key)
            if j is None:
                j = len(out_pos)
                cache[key] = j
                out_pos.append(v[vi])
                out_nor.append(
                    vn[ni] if ni >= 0 and ni < len(vn) else [0.0, 0.0, 1.0]
                )
                out_uv.append(
                    vt[ti] if ti >= 0 and ti < len(vt) else [0.0, 0.0]
                )
            indices.append(j)

    pos3: np.ndarray = np.asarray(out_pos, dtype=np.float32)
    pos4 = np.hstack([pos3, np.ones((len(pos3), 1), dtype=np.float32)])
    return ObjMesh(
        positions=pos4,
        normals=np.asarray(out_nor, dtype=np.float32),
        texcoords=np.asarray(out_uv, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.uint32),
        groups=groups,
        materials=materials,
    )


def upload_material_diffuse_textures(
    materials: dict[str, ObjMaterial], mipmaps: bool = True
) -> None:
    """The C demos' material loop: one repeating GL texture per material with
    a ``map_Kd``, stored in ``diffuse_texture_name`` (trilinear mipmapped,
    or plain ``GL_LINEAR`` when ``mipmaps`` is off, as 45 does)."""
    for material in materials.values():
        if material.diffuse_texture:
            material.diffuse_texture_name = upload_tga_2d(
                filename=material.diffuse_texture, mipmaps=mipmaps
            )


def draw_obj_group(group: ObjGroup, instances: int = 0) -> None:
    """``glDrawElements`` of one group's triangles from the bound index VBO
    holding ``ObjMesh.indices`` (instanced when ``instances`` > 0)."""
    offset = ctypes.c_void_p(group.first * 4)
    if instances:
        GL.glDrawElementsInstanced(
            GL.GL_TRIANGLES, group.count, GL.GL_UNSIGNED_INT, offset, instances
        )
    else:
        GL.glDrawElements(
            GL.GL_TRIANGLES, group.count, GL.GL_UNSIGNED_INT, offset
        )


# --- Shape generators (sphere / plane / cube / cylinder / etc.) ---------


@dataclass
class Shape:
    vertices: np.ndarray  # (N,4) float32
    normals: np.ndarray  # (N,3) float32
    texcoords: np.ndarray  # (N,2) float32
    tangents: np.ndarray | None = None
    bitangents: np.ndarray | None = None
    indices: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.uint32)
    )


def make_sphere(radius: float = 0.5, slices: int = 32) -> Shape:
    """GLUS's ``glusShapeCreateSpheref``, vertex for vertex and index for index.

    Faithfulness here is not cosmetic. The C demos draw this sphere from the
    INSIDE for panoramic backgrounds (``example32``) with the GLUS idiom
    ``glFrontFace(GL_CW)`` and culling on -- so the *effective winding* must be
    GLUS's. The previous port swapped x and z in the parametrization AND emitted
    the two triangles of each quad in a different order, and the combination was
    mirror-handed: every inside-out sphere was culled to nothing (the
    2026-09-09 C-vs-Python frame comparison showed example32's backdrop black
    where the C shows the Venice courtyard; flipping the front face brought it
    back). Exterior spheres never showed it because their demos do not cull.
    """
    stacks = slices // 2
    verts, nors, uvs = [], [], []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        sp, cp = math.sin(phi), math.cos(phi)
        for j in range(slices + 1):
            theta = 2.0 * math.pi * j / slices
            st, ct = math.sin(theta), math.cos(theta)
            # GLUS: x = sin(lat) * sin(ang), z = sin(lat) * cos(ang)
            x, y, z = sp * st, cp, sp * ct
            verts.append([radius * x, radius * y, radius * z, 1.0])
            nors.append([x, y, z])
            uvs.append([j / slices, 1.0 - i / stacks])
    idx = []
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            # GLUS: (a, b, b+1), (a, b+1, a+1)
            idx.extend([a, b, b + 1, a, b + 1, a + 1])
    return Shape(
        vertices=np.asarray(verts, dtype=np.float32),
        normals=np.asarray(nors, dtype=np.float32),
        texcoords=np.asarray(uvs, dtype=np.float32),
        indices=np.asarray(idx, dtype=np.uint32),
    )


def make_plane(half: float = 1.0) -> Shape:
    """Mirror glusShapeCreatePlanef: X-Y plane, normal +Z.

    Vertex order matches GLUS so demos that rely on a specific quad
    layout (Example 13's tess patch indices [0,1,3,2]) work directly.
    """
    v = np.array(
        [
            [-half, -half, 0, 1],  # 0: bottom-left
            [half, -half, 0, 1],  # 1: bottom-right
            [-half, half, 0, 1],  # 2: top-left
            [half, half, 0, 1],  # 3: top-right
        ],
        dtype=np.float32,
    )
    n = np.array([[0, 0, 1]] * 4, dtype=np.float32)
    t = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    i = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
    return Shape(vertices=v, normals=n, texcoords=t, indices=i)


def make_rectangular_plane(half_x: float, half_y: float) -> Shape:
    v = np.array(
        [
            [-half_x, -half_y, 0, 1],
            [half_x, -half_y, 0, 1],
            [half_x, half_y, 0, 1],
            [-half_x, half_y, 0, 1],
        ],
        dtype=np.float32,
    )
    n = np.array([[0, 0, 1]] * 4, dtype=np.float32)
    t = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    i = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return Shape(vertices=v, normals=n, texcoords=t, indices=i)


def make_rectangular_grid_plane(
    horizontal_extend: float, vertical_extend: float, rows: int, columns: int
) -> Shape:
    """GLUS's ``glusShapeCreateRectangularGridPlanef`` in triangles mode.

    ``horizontal_extend`` and ``vertical_extend`` are the plane's FULL width
    and height (GLUS semantics): the plane spans ``+-extend/2`` in the X-Y
    plane with normal +Z and tangent +X. Vertices run left to right, top row
    first (``t`` = 1 at the top), in GLUS's order, so index-based callers see
    the same layout as the C.
    """
    verts, nors, tans, uvs = [], [], [], []
    for i in range((rows + 1) * (columns + 1)):
        x: float = (i % (columns + 1)) / columns
        y: float = 1.0 - (i // (columns + 1)) / rows
        verts.append(
            [
                horizontal_extend * (x - 0.5),
                vertical_extend * (y - 0.5),
                0.0,
                1.0,
            ]
        )
        nors.append([0.0, 0.0, 1.0])
        tans.append([1.0, 0.0, 0.0])
        uvs.append([x, y])
    idx: list[int] = []
    for i in range(rows * columns):
        c: int = i % columns
        r: int = i // columns
        idx.extend(
            [
                c + r * (columns + 1),
                c + (r + 1) * (columns + 1),
                (c + 1) + (r + 1) * (columns + 1),
                (c + 1) + (r + 1) * (columns + 1),
                (c + 1) + r * (columns + 1),
                c + r * (columns + 1),
            ]
        )
    return Shape(
        vertices=np.asarray(verts, dtype=np.float32),
        normals=np.asarray(nors, dtype=np.float32),
        texcoords=np.asarray(uvs, dtype=np.float32),
        tangents=np.asarray(tans, dtype=np.float32),
        indices=np.asarray(idx, dtype=np.uint32),
    )


# glusShapeCreateCubef's 24-vertex tables, one row per face vertex:
# (unit position, normal, tangent, texcoord). Faces in GLUS order: -Y, +Y,
# -Z, +Z, -X, +X. The texcoord table is what makes the crate in example06
# land the right way up on every face.
_CUBE_FACES: tuple[
    tuple[tuple[float, float, float], tuple[float, float, float]], ...
] = (
    ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
    ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)),
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
)
_CUBE_VERTICES: tuple[tuple[float, float, float], ...] = (
    (-1, -1, -1), (-1, -1, 1), (1, -1, 1), (1, -1, -1),
    (-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1),
    (-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1),
    (-1, -1, 1), (-1, 1, 1), (1, 1, 1), (1, -1, 1),
    (-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1),
    (1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1),
)  # fmt: skip
_CUBE_TEXCOORDS: tuple[tuple[float, float], ...] = (
    (0, 0), (0, 1), (1, 1), (1, 0),
    (0, 1), (0, 0), (1, 0), (1, 1),
    (1, 0), (1, 1), (0, 1), (0, 0),
    (0, 0), (0, 1), (1, 1), (1, 0),
    (0, 0), (1, 0), (1, 1), (0, 1),
    (1, 0), (0, 0), (0, 1), (1, 1),
)  # fmt: skip
_CUBE_INDICES: tuple[int, ...] = (
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 15, 14, 12, 14, 13,
    16, 17, 18, 16, 18, 19,
    20, 23, 22, 20, 22, 21,
)  # fmt: skip


def make_cube(half: float = 0.5) -> Shape:
    """GLUS's ``glusShapeCreateCubef``: a ``+-half`` cube, 24 vertices."""
    verts: list[list[float]] = [
        [half * x, half * y, half * z, 1.0] for x, y, z in _CUBE_VERTICES
    ]
    nors: list[list[float]] = [
        list(_CUBE_FACES[i // 4][0]) for i in range(len(_CUBE_VERTICES))
    ]
    tans: list[list[float]] = [
        list(_CUBE_FACES[i // 4][1]) for i in range(len(_CUBE_VERTICES))
    ]
    return Shape(
        vertices=np.asarray(verts, dtype=np.float32),
        normals=np.asarray(nors, dtype=np.float32),
        texcoords=np.asarray(_CUBE_TEXCOORDS, dtype=np.float32),
        tangents=np.asarray(tans, dtype=np.float32),
        indices=np.asarray(_CUBE_INDICES, dtype=np.uint32),
    )


def make_torus(
    inner_radius: float,
    outer_radius: float,
    slices: int = 24,
    stacks: int = 24,
) -> Shape:
    """GLUS's ``glusShapeCreateTorusf``, vertex for vertex and index for index.

    ``inner_radius`` is the radius of the hole and ``outer_radius`` the radius
    of the whole torus (GLUS semantics): the tube radius is
    ``(outer - inner) / 2`` and the centre-line radius ``outer - tube``. The
    torus lies in the X-Y plane; ``slices`` run around the centre line and
    ``stacks`` around the tube.
    """
    torus_radius: float = (outer_radius - inner_radius) / 2.0
    center_radius: float = outer_radius - torus_radius
    verts, nors, tans, uvs = [], [], [], []
    for side in range(slices + 1):
        s: float = side / slices
        cs, ss = math.cos(2.0 * math.pi * s), math.sin(2.0 * math.pi * s)
        for face in range(stacks + 1):
            t: float = face / stacks
            ct, st = math.cos(2.0 * math.pi * t), math.sin(2.0 * math.pi * t)
            verts.append(
                [
                    (center_radius + torus_radius * ct) * cs,
                    (center_radius + torus_radius * ct) * ss,
                    torus_radius * st,
                    1.0,
                ]
            )
            nors.append([cs * ct, ss * ct, st])
            uvs.append([s, t])
            # GLUS: Rz(360 * s) applied to (0, 1, 0).
            tans.append([-ss, cs, 0.0])
    idx: list[int] = []
    for side in range(slices):
        for face in range(stacks):
            v0: int = side * (stacks + 1) + face
            v1: int = (side + 1) * (stacks + 1) + face
            v2: int = (side + 1) * (stacks + 1) + face + 1
            v3: int = side * (stacks + 1) + face + 1
            idx.extend([v0, v1, v2, v0, v2, v3])
    return Shape(
        vertices=np.asarray(verts, dtype=np.float32),
        normals=np.asarray(nors, dtype=np.float32),
        texcoords=np.asarray(uvs, dtype=np.float32),
        tangents=np.asarray(tans, dtype=np.float32),
        indices=np.asarray(idx, dtype=np.uint32),
    )


def make_cylinder(half_height: float, radius: float, slices: int = 32) -> Shape:
    verts, nors, uvs = [], [], []
    # side
    for j in range(slices + 1):
        t = j / slices
        a = t * 2 * math.pi
        ca, sa = math.cos(a), math.sin(a)
        verts.append([radius * ca, -half_height, radius * sa, 1])
        nors.append([ca, 0, sa])
        uvs.append([t, 0])
        verts.append([radius * ca, half_height, radius * sa, 1])
        nors.append([ca, 0, sa])
        uvs.append([t, 1])
    side_idx = []
    for j in range(slices):
        a = 2 * j
        side_idx.extend([a, a + 1, a + 2, a + 2, a + 1, a + 3])
    # caps
    base_top: int = len(verts)
    verts.append([0, half_height, 0, 1])
    nors.append([0, 1, 0])
    uvs.append([0.5, 0.5])
    for j in range(slices + 1):
        a = (j / slices) * 2 * math.pi
        verts.append(
            [radius * math.cos(a), half_height, radius * math.sin(a), 1]
        )
        nors.append([0, 1, 0])
        uvs.append([0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a)])
    cap_top_idx = []
    for j in range(slices):
        cap_top_idx.extend([base_top, base_top + 1 + j, base_top + 2 + j])

    base_bot: int = len(verts)
    verts.append([0, -half_height, 0, 1])
    nors.append([0, -1, 0])
    uvs.append([0.5, 0.5])
    for j in range(slices + 1):
        a = (j / slices) * 2 * math.pi
        verts.append(
            [radius * math.cos(a), -half_height, radius * math.sin(a), 1]
        )
        nors.append([0, -1, 0])
        uvs.append([0.5 + 0.5 * math.cos(a), 0.5 - 0.5 * math.sin(a)])
    cap_bot_idx = []
    for j in range(slices):
        cap_bot_idx.extend([base_bot, base_bot + 2 + j, base_bot + 1 + j])

    idx = side_idx + cap_top_idx + cap_bot_idx
    return Shape(
        vertices=np.asarray(verts, dtype=np.float32),
        normals=np.asarray(nors, dtype=np.float32),
        texcoords=np.asarray(uvs, dtype=np.float32),
        indices=np.asarray(idx, dtype=np.uint32),
    )


def make_cone(half_height: float, radius: float, slices: int = 32) -> Shape:
    verts, nors, uvs = [], [], []
    for j in range(slices + 1):
        t = j / slices
        a = t * 2 * math.pi
        ca, sa = math.cos(a), math.sin(a)
        nx, ny, nz = ca, radius / (2 * half_height), sa
        ln: float = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / ln, ny / ln, nz / ln
        verts.append([0, half_height, 0, 1])
        nors.append([nx, ny, nz])
        uvs.append([t, 1])
        verts.append([radius * ca, -half_height, radius * sa, 1])
        nors.append([nx, ny, nz])
        uvs.append([t, 0])
    side_idx = []
    for j in range(slices):
        a = 2 * j
        side_idx.extend([a, a + 1, a + 3])
    base = len(verts)
    verts.append([0, -half_height, 0, 1])
    nors.append([0, -1, 0])
    uvs.append([0.5, 0.5])
    for j in range(slices + 1):
        a = (j / slices) * 2 * math.pi
        verts.append(
            [radius * math.cos(a), -half_height, radius * math.sin(a), 1]
        )
        nors.append([0, -1, 0])
        uvs.append([0.5 + 0.5 * math.cos(a), 0.5 - 0.5 * math.sin(a)])
    cap_idx = []
    for j in range(slices):
        cap_idx.extend([base, base + 2 + j, base + 1 + j])
    idx = side_idx + cap_idx
    return Shape(
        vertices=np.asarray(verts, dtype=np.float32),
        normals=np.asarray(nors, dtype=np.float32),
        texcoords=np.asarray(uvs, dtype=np.float32),
        indices=np.asarray(idx, dtype=np.uint32),
    )


def make_disc(radius: float = 0.5, slices: int = 32) -> Shape:
    verts, nors, uvs = [], [], []
    verts.append([0, 0, 0, 1])
    nors.append([0, 0, 1])
    uvs.append([0.5, 0.5])
    for j in range(slices + 1):
        a = (j / slices) * 2 * math.pi
        verts.append([radius * math.cos(a), radius * math.sin(a), 0, 1])
        nors.append([0, 0, 1])
        uvs.append([0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a)])
    idx = []
    for j in range(slices):
        idx.extend([0, j + 1, j + 2])
    return Shape(
        vertices=np.asarray(verts, dtype=np.float32),
        normals=np.asarray(nors, dtype=np.float32),
        texcoords=np.asarray(uvs, dtype=np.float32),
        indices=np.asarray(idx, dtype=np.uint32),
    )


def make_dome(radius: float = 0.5, slices: int = 32) -> Shape:
    """Hemisphere (top half of sphere)."""
    stacks = slices // 4
    verts, nors, uvs = [], [], []
    for i in range(stacks + 1):
        phi = (math.pi / 2) * i / stacks
        sp, cp = math.sin(phi), math.cos(phi)
        for j in range(slices + 1):
            theta = 2.0 * math.pi * j / slices
            st, ct = math.sin(theta), math.cos(theta)
            x, y, z = sp * ct, cp, sp * st
            verts.append([radius * x, radius * y, radius * z, 1.0])
            nors.append([x, y, z])
            uvs.append([j / slices, 1.0 - i / stacks])
    idx = []
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            idx.extend([a, b, a + 1, a + 1, b, b + 1])
    return Shape(
        vertices=np.asarray(verts, dtype=np.float32),
        normals=np.asarray(nors, dtype=np.float32),
        texcoords=np.asarray(uvs, dtype=np.float32),
        indices=np.asarray(idx, dtype=np.uint32),
    )


def calculate_tangents(shape: Shape) -> Shape:
    """Add tangents and bitangents using positions/normals/uvs and indices."""
    verts = shape.vertices[:, :3]
    uvs = shape.texcoords
    idx = shape.indices.reshape(-1, 3)
    tan = np.zeros((len(verts), 3), dtype=np.float32)
    bit = np.zeros((len(verts), 3), dtype=np.float32)
    for i0, i1, i2 in idx:
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]
        u0, u1, u2 = uvs[i0], uvs[i1], uvs[i2]
        e1, e2 = v1 - v0, v2 - v0
        d1, d2 = u1 - u0, u2 - u0
        denom = d1[0] * d2[1] - d2[0] * d1[1]
        if abs(denom) < 1e-8:
            continue
        f = 1.0 / denom
        t = f * (d2[1] * e1 - d1[1] * e2)
        b = f * (-d2[0] * e1 + d1[0] * e2)
        for j in (i0, i1, i2):
            tan[j] += t
            bit[j] += b
    # orthonormalize against normal
    n = shape.normals
    tan = tan - n * (n * tan).sum(axis=1, keepdims=True)
    norms = np.linalg.norm(tan, axis=1, keepdims=True)
    tan = np.where(
        norms > 1e-8, tan / np.maximum(norms, 1e-8), np.array([1.0, 0.0, 0.0])
    )
    bit = np.cross(n, tan)
    shape.tangents = tan.astype(np.float32)
    shape.bitangents = bit.astype(np.float32)
    return shape


# --- Matrix utilities (column-major, OpenGL-style) ------------------------


def identity4() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def perspective(
    fovy_deg: float, aspect: float, near: float, far: float
) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def ortho(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m


def frustum(
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float,
    far: float,
) -> np.ndarray:
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 2.0 * near / (right - left)
    m[1, 1] = 2.0 * near / (top - bottom)
    m[0, 2] = (right + left) / (right - left)
    m[1, 2] = (top + bottom) / (top - bottom)
    m[2, 2] = -(far + near) / (far - near)
    m[2, 3] = -(2.0 * far * near) / (far - near)
    m[3, 2] = -1.0
    return m


def look_at(eye: Vec3, center: Vec3, up: Vec3) -> np.ndarray:
    eye: np.ndarray = np.asarray(eye, dtype=np.float32)
    center: np.ndarray = np.asarray(center, dtype=np.float32)
    up: np.ndarray = np.asarray(up, dtype=np.float32)
    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def scale_m(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = x
    m[1, 1] = y
    m[2, 2] = z
    return m


def rotate_x(deg: float) -> np.ndarray:
    r: float = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    m = np.eye(4, dtype=np.float32)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def rotate_y(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def rotate_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return m


def rotate_axis(deg: float, axis: Vec3) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    one_minus_c = 1 - c
    m = np.array(
        [
            [
                c + x * x * one_minus_c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
                0,
            ],
            [
                y * x * one_minus_c + z * s,
                c + y * y * one_minus_c,
                y * z * one_minus_c - x * s,
                0,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                c + z * z * one_minus_c,
                0,
            ],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return m


def normal_matrix3(view_or_modelview: np.ndarray) -> np.ndarray:
    """Top-left 3x3 of inverse-transpose. For rigid-body, just the 3x3."""
    n = np.linalg.inv(view_or_modelview[:3, :3]).T.astype(np.float32)
    return n


def planar_shadow_directional(plane: Vec4, light_dir: Vec3) -> np.ndarray:
    """plane: (a,b,c,d) with ax+by+cz+d=0; light_dir: (x,y,z)."""
    a, b, c, d = plane
    lx, ly, lz = light_dir
    dot = a * lx + b * ly + c * lz
    m = np.array(
        [
            [dot - lx * a, -lx * b, -lx * c, -lx * d],
            [-ly * a, dot - ly * b, -ly * c, -ly * d],
            [-lz * a, -lz * b, dot - lz * c, -lz * d],
            [0, 0, 0, dot],
        ],
        dtype=np.float32,
    )
    return m


def planar_shadow_point(plane: Vec4, light_pos: Vec3) -> np.ndarray:
    a, b, c, d = plane
    lx, ly, lz = light_pos
    dot = a * lx + b * ly + c * lz + d
    m = np.array(
        [
            [dot - lx * a, -lx * b, -lx * c, -lx * d],
            [-ly * a, dot - ly * b, -ly * c, -ly * d],
            [-lz * a, -lz * b, dot - lz * c, -lz * d],
            [-a, -b, -c, dot - d],
        ],
        dtype=np.float32,
    )
    return m


def upload_matrix(loc: int, m: np.ndarray) -> None:
    """OpenGL is column-major; pass transpose flag GL_TRUE for our row-major arrays."""
    GL.glUniformMatrix4fv(
        loc, 1, GL.GL_TRUE, np.ascontiguousarray(m, dtype=np.float32)
    )


def upload_matrix3(loc: int, m: np.ndarray) -> None:
    GL.glUniformMatrix3fv(
        loc, 1, GL.GL_TRUE, np.ascontiguousarray(m, dtype=np.float32)
    )


# --- VAO / VBO helpers ----------------------------------------------------


def make_vbo(
    data: np.ndarray,
    target: GLenum = GL.GL_ARRAY_BUFFER,
    usage: GLenum = GL.GL_STATIC_DRAW,
) -> int:
    buf: int = GL.glGenBuffers(1)
    GL.glBindBuffer(target, buf)
    arr = np.ascontiguousarray(data)
    GL.glBufferData(target=target, size=arr.nbytes, data=arr, usage=usage)
    GL.glBindBuffer(target, 0)
    return buf


def attach_attrib(
    prog: int,
    vao: int,
    vbo: int,
    name: str,
    size: int,
    gl_type: GLenum = GL.GL_FLOAT,
    stride: int = 0,
    offset: int = 0,
) -> int:
    """Bind a vertex attribute by attribute name. Returns the location, or -1 if not present."""
    loc: int = GL.glGetAttribLocation(program=prog, name=name)
    if loc < 0:
        return -1
    GL.glBindVertexArray(vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glVertexAttribPointer(
        index=loc,
        size=size,
        type=gl_type,
        normalized=GL.GL_FALSE,
        stride=stride,
        pointer=ctypes.c_void_p(offset),
    )
    GL.glEnableVertexAttribArray(loc)
    return loc


# --- Demo runner ----------------------------------------------------------


@dataclass
class DemoCallbacks:
    init: Callable[[], bool] | None = None
    reshape: Callable[[int, int], None] | None = None
    update: Callable[[float], bool] | None = None
    terminate: Callable[[], None] | None = None
    key: Callable[[bool, int], None] | None = None
    mouse: Callable[[bool, int, int, int], None] | None = None
    mouse_move: Callable[[int, int, int], None] | None = None
    imgui_panel: Callable[[], None] | None = None


@dataclass
class _WindowState:
    """Fullscreen toggle state: the windowed geometry to restore, and the
    F11 edge detector (the key is polled every frame; only the frame it
    goes down toggles)."""

    fullscreen: bool = False
    saved_x: int = 0
    saved_y: int = 0
    saved_w: int = 0
    saved_h: int = 0
    f11_down: bool = False


FULLSCREEN_KEY: int = glfw.KEY_F11
FULLSCREEN_KEY_LABEL: str = "F11"


def toggle_fullscreen(win: Any, state: _WindowState) -> None:
    """Flip between windowed and exclusive fullscreen on the primary
    monitor with ``glfw.set_window_monitor``; the way back restores the
    saved windowed geometry. The same mechanism as the SuperBible v4
    tree's ``toggle_fullscreen`` (this package does not import that
    tree)."""
    if state.fullscreen:
        glfw.set_window_monitor(
            window=win,
            monitor=None,  # ty: ignore[invalid-argument-type]  # GLFW takes NULL here; the stub says otherwise
            xpos=state.saved_x,
            ypos=state.saved_y,
            width=state.saved_w,
            height=state.saved_h,
            refresh_rate=0,
        )
        state.fullscreen = False
        return
    state.saved_x, state.saved_y = glfw.get_window_pos(window=win)
    state.saved_w, state.saved_h = glfw.get_window_size(window=win)
    monitor: Any = glfw.get_primary_monitor()
    if monitor is None:
        return
    mode: Any = glfw.get_video_mode(monitor=monitor)
    glfw.set_window_monitor(
        window=win,
        monitor=monitor,
        xpos=0,
        ypos=0,
        width=mode.size.width,
        height=mode.size.height,
        refresh_rate=mode.refresh_rate,
    )
    state.fullscreen = True


def poll_fullscreen_key(win: Any, state: _WindowState) -> None:
    """Toggle fullscreen on the frame F11 goes down (polled from the
    menubar draw, so the demo's own key callback stays untouched)."""
    down: bool = glfw.get_key(window=win, key=FULLSCREEN_KEY) == glfw.PRESS
    if down and not state.f11_down:
        toggle_fullscreen(win=win, state=state)
    state.f11_down = down


def draw_menubar(
    win: Any, state: _WindowState, panel: Callable[[], None] | None
) -> None:
    """The menubar every demo shares: File → Quit (Esc), View →
    Fullscreen (F11, also polled), and, when the demo passed an
    ``imgui_panel`` callback, a Controls menu whose body is that
    callback. ``panel`` runs only while the Controls menu is open, as
    any ``begin_menu`` body does; demos write their ``ui`` dicts there
    and read them in ``update``. With the environment variable
    ``MVP_CONTROLS_WINDOW`` set, ``panel()`` is also drawn in a plain
    window every frame so a headless run executes every widget (the
    gates' hook; never set it for a user). Call between
    ``imgui.new_frame()`` and ``imgui.render()``."""
    poll_fullscreen_key(win=win, state=state)
    if not imgui.begin_main_menu_bar():  # ty: ignore[unresolved-attribute]  # guarded by HAS_IMGUI
        return
    if imgui.begin_menu(label="File"):  # ty: ignore[unresolved-attribute]
        clicked_quit, _ = imgui.menu_item(  # ty: ignore[unresolved-attribute]
            label="Quit", shortcut="Esc", p_selected=False
        )
        if clicked_quit:
            glfw.set_window_should_close(window=win, value=True)
        imgui.end_menu()  # ty: ignore[unresolved-attribute]
    if imgui.begin_menu(label="View"):  # ty: ignore[unresolved-attribute]
        clicked_fs, _ = imgui.menu_item(  # ty: ignore[unresolved-attribute]
            label="Fullscreen",
            shortcut=FULLSCREEN_KEY_LABEL,
            p_selected=state.fullscreen,
        )
        if clicked_fs:
            toggle_fullscreen(win=win, state=state)
        imgui.end_menu()  # ty: ignore[unresolved-attribute]
    if panel is not None and imgui.begin_menu(label="Controls"):  # ty: ignore[unresolved-attribute]
        panel()
        imgui.end_menu()  # ty: ignore[unresolved-attribute]
    imgui.end_main_menu_bar()  # ty: ignore[unresolved-attribute]
    if panel is not None and os.environ.get("MVP_CONTROLS_WINDOW"):
        imgui.begin(  # ty: ignore[unresolved-attribute]
            name="Controls (verification window)",
            flags=imgui.WindowFlags_.always_auto_resize,  # ty: ignore[unresolved-attribute]
        )
        panel()
        imgui.end()  # ty: ignore[unresolved-attribute]


def run_demo(
    title: str,
    callbacks: DemoCallbacks,
    width: int = 1280,
    height: int = 720,
    gl_major: int = 3,
    gl_minor: int = 3,
    depth_bits: int = 24,
    stencil_bits: int = 0,
    samples: int = 0,
) -> None:
    """Create a glfw window, init imgui, run the demo's update loop."""
    if os.environ.get("MVP_GLFW_X11"):
        # The headless gate's pin (headless-verification.md §3): the
        # window goes to the Xvfb DISPLAY, not a mounted Wayland socket.
        glfw.init_hint(hint=glfw.PLATFORM, value=glfw.PLATFORM_X11)
    if not glfw.init():
        raise RuntimeError("glfw init failed")

    glfw.window_hint(hint=glfw.CONTEXT_VERSION_MAJOR, value=gl_major)
    glfw.window_hint(hint=glfw.CONTEXT_VERSION_MINOR, value=gl_minor)
    glfw.window_hint(hint=glfw.OPENGL_PROFILE, value=glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(hint=glfw.OPENGL_FORWARD_COMPAT, value=True)
    glfw.window_hint(hint=glfw.DEPTH_BITS, value=depth_bits)
    glfw.window_hint(hint=glfw.STENCIL_BITS, value=stencil_bits)
    if samples:
        glfw.window_hint(hint=glfw.SAMPLES, value=samples)

    win = glfw.create_window(
        width=width, height=height, title=title, monitor=None, share=None
    )
    if not win:
        glfw.terminate()
        raise RuntimeError("glfw window creation failed")
    glfw.make_context_current(win)
    glfw.swap_interval(1)

    # macOS Core Profile requires a default VAO bound at all times.
    default_vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(default_vao)
    win_state: _WindowState = _WindowState()

    last_mouse: list[int] = [0, 0]
    mouse_buttons: list[bool] = [False, False, False]

    # Create the imgui renderer FIRST. imgui_bundle's GlfwRenderer installs
    # its own glfw callbacks in __init__ and does NOT chain to any previously
    # installed ones, so callbacks set before it are silently discarded. We
    # install ours AFTER and forward each event to the renderer's handler, so
    # imgui keeps its input and our keys/Esc/mouse logic also run.
    # ``imgui``/``GlfwRenderer`` are None only when the import failed, and
    # HAS_IMGUI is False exactly then — a correlation ty cannot see, hence
    # the ignores here and in the render block below.
    if HAS_IMGUI:
        imgui.create_context()  # ty: ignore[unresolved-attribute]
        impl = GlfwRenderer(win)  # ty: ignore[call-non-callable]
    else:
        impl = None

    def on_key(
        window: Any, key: int, scancode: int, action: int, mods: int
    ) -> None:
        if impl is not None:
            impl.keyboard_callback(window, key, scancode, action, mods)
        if callbacks.key:
            if action == glfw.PRESS:
                callbacks.key(True, key)
            elif action == glfw.RELEASE:
                callbacks.key(False, key)
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window=window, value=True)

    def on_mouse_button(
        window: Any, button: int, action: int, mods: int
    ) -> None:
        if impl is not None:
            impl.mouse_button_callback(window, button, action, mods)
        x, y = glfw.get_cursor_pos(window)
        pressed = action == glfw.PRESS
        if 0 <= button < 3:
            mouse_buttons[button] = pressed
        if callbacks.mouse:
            callbacks.mouse(pressed, button, int(x), int(y))

    def on_mouse_move(window: Any, x: float, y: float) -> None:
        if impl is not None:
            impl.mouse_callback(window, x, y)
        last_mouse[0] = int(x)
        last_mouse[1] = int(y)
        if callbacks.mouse_move:
            mask = sum((1 << i) for i, b in enumerate(mouse_buttons) if b)
            callbacks.mouse_move(mask, int(x), int(y))

    # Our callbacks override imgui's for key/mouse-button/cursor-pos (we
    # forward to them above); char/scroll/resize stay as the renderer set
    # them, so imgui text input and scrolling still work.
    glfw.set_key_callback(window=win, cbfun=on_key)
    glfw.set_mouse_button_callback(window=win, cbfun=on_mouse_button)
    glfw.set_cursor_pos_callback(window=win, cbfun=on_mouse_move)

    if callbacks.init is not None:
        if callbacks.init() is False:
            raise RuntimeError("demo init returned False")

    fbw, fbh = glfw.get_framebuffer_size(win)
    if callbacks.reshape:
        callbacks.reshape(fbw, fbh)

    last_size = (fbw, fbh)
    last_time = glfw.get_time()

    while not glfw.window_should_close(win):
        glfw.poll_events()
        if impl is not None:
            impl.process_inputs()

        fbw, fbh = glfw.get_framebuffer_size(win)
        if (fbw, fbh) != last_size and callbacks.reshape:
            callbacks.reshape(fbw, fbh)
            last_size = (fbw, fbh)

        now: float = glfw.get_time()
        dt = now - last_time
        last_time = now

        # Match /mvp's working pattern: scene first, imgui frame on top.
        cont = True
        if callbacks.update:
            cont = callbacks.update(dt)

        if HAS_IMGUI:
            # Save bindings on TEXTURE0 around imgui's render so the demo's
            # init-time bindings don't get clobbered. ImGui's renderer binds
            # its font atlas to GL_TEXTURE0/GL_TEXTURE_2D; we also save the
            # CUBE_MAP and 3D bindings on the same unit to be safe.
            GL.glActiveTexture(GL.GL_TEXTURE0)
            saved_t0_2d = GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D)
            saved_t0_cube = GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_CUBE_MAP)
            saved_t0_3d = GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_3D)

            # imgui_bundle's GL backend sets up blend/cull/depth/scissor but
            # leaves GL_STENCIL_TEST untouched (and doesn't save/restore it).
            # A demo that left a restrictive glStencilFunc enabled (e.g. the
            # stencil-cap example) would otherwise stencil-reject the whole
            # UI, so disable it around imgui's draw and restore it after.
            saved_stencil = GL.glIsEnabled(GL.GL_STENCIL_TEST)
            GL.glDisable(GL.GL_STENCIL_TEST)

            imgui.new_frame()  # ty: ignore[unresolved-attribute]  # guarded by HAS_IMGUI (see init note)
            draw_menubar(win=win, state=win_state, panel=callbacks.imgui_panel)
            imgui.render()  # ty: ignore[unresolved-attribute]  # guarded by HAS_IMGUI (see init note)
            impl.render(imgui.get_draw_data())  # ty: ignore[unresolved-attribute]  # guarded by HAS_IMGUI (see init note)

            if saved_stencil:
                GL.glEnable(GL.GL_STENCIL_TEST)

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, int(saved_t0_2d))
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, int(saved_t0_cube))
            GL.glBindTexture(GL.GL_TEXTURE_3D, int(saved_t0_3d))

        glfw.swap_buffers(win)
        if cont is False:
            break

    if callbacks.terminate:
        callbacks.terminate()

    if impl is not None:
        impl.shutdown()
    GL.glBindVertexArray(0)
    GL.glDeleteVertexArrays(1, [default_vao])
    glfw.destroy_window(win)
    glfw.terminate()

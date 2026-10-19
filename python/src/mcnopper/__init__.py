"""McNopper — Python port of Norbert Nopper's OpenGL 4 examples.

Original C source: https://github.com/McNopper/OpenGL (52 examples at v3.2.7,
all ported).

This package bundles 52 demos, the GLUS-equivalent helper module
``_common``, all GLSL shaders, and the full ``media/`` asset tree
(textures, OBJ meshes, HDR environment maps).

Run a demo from the CLI::

    mcnopper 13                # launches example13 (tessellation)
    mcnopper list              # prints all 52 demos with descriptions

Or programmatically::

    from mcnopper import run, demo_count
    run(13)
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["demo_count", "demo_titles", "run"]

import runpy as _runpy

demo_count: int = 52


def demo_titles() -> dict[int, str]:
    """Return ``{1: "first-line of example01 docstring", ...}`` for all 52 demos.

    Reads the docstring from the file's AST so this works without OpenGL or
    GLFW installed — useful for ``mcnopper list`` in CI / packaging contexts.
    """
    import ast
    from pathlib import Path

    pkg_root: Path = Path(__file__).resolve().parent
    titles: dict[int, str] = {}
    for n in range(1, demo_count + 1):
        # Most demos are flat modules; the four with co-located shaders are a
        # subpackage (exampleNN/exampleNN.py).
        flat: Path = pkg_root / f"example{n:02d}.py"
        path: Path = (
            flat
            if flat.exists()
            else pkg_root / f"example{n:02d}" / f"example{n:02d}.py"
        )
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            doc = ast.get_docstring(tree) or ""
            first = doc.strip().splitlines()
            titles[n] = first[0] if first else f"Example {n}"
        except Exception as e:  # noqa: BLE001
            titles[n] = f"(unreadable: {e})"
    return titles


def run(n: int) -> None:
    """Launch demo number ``n`` (1..52)."""
    if not 1 <= n <= demo_count:
        raise ValueError(f"demo number must be 1..{demo_count}, got {n}")
    from pathlib import Path

    pkg_root = Path(__file__).resolve().parent
    # Flat module (most demos), or the exampleNN/exampleNN.py subpackage for
    # the four demos that carry co-located shaders.
    if (pkg_root / f"example{n:02d}.py").exists():
        mod_name = f"mcnopper.example{n:02d}"
    else:
        mod_name = f"mcnopper.example{n:02d}.example{n:02d}"
    _runpy.run_module(mod_name, run_name="__main__")

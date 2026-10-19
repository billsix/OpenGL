#!/usr/bin/env python3
"""Classify each per-example Python shader as identical-to-C (safe to delete and
reference up to ExampleNN/shader/) or differing (a port edit / version-lowering
that must stay copied).

Read-only: prints two lists and counts. The repo root is found relative to this
script (tasks/adhoc/<slug>/classify.py -> three parents up), so it has no
container-absolute paths. Comparison ignores the provenance comment block above
``#version`` and any trailing-whitespace/newline difference, by comparing each
file from its first ``#version`` to the rstripped end.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
PKG_DIR: Path = REPO_ROOT / "python" / "src" / "mcnopper"


def body_from_version(text: str) -> str:
    """Return the shader text from its first ``#version`` line, rstripped."""
    idx: int = text.find("#version")
    return (text[idx:] if idx >= 0 else text).rstrip()


def c_sibling(py_shader: Path) -> Path | None:
    """Map python/src/mcnopper/exampleNN/foo.glsl -> <repo>/ExampleNN/shader/foo.glsl."""
    dirname: str = py_shader.parent.name  # e.g. "example05"
    if not dirname.startswith("example"):
        return None
    num: str = dirname[len("example"):]
    sib: Path = REPO_ROOT / f"Example{num}" / "shader" / py_shader.name
    return sib if sib.exists() else None


def main() -> None:
    identical: list[tuple[Path, Path]] = []
    differ: list[tuple[Path, Path]] = []
    no_sibling: list[Path] = []
    for py_shader in sorted(PKG_DIR.glob("example*/*.glsl")):
        sib: Path | None = c_sibling(py_shader)
        if sib is None:
            no_sibling.append(py_shader)
            continue
        py_body: str = body_from_version(py_shader.read_text(encoding="utf-8", errors="replace"))
        c_body: str = body_from_version(sib.read_text(encoding="utf-8", errors="replace"))
        rel: Path = py_shader.relative_to(REPO_ROOT)
        if py_body == c_body:
            identical.append((rel, sib.relative_to(REPO_ROOT)))
        else:
            differ.append((rel, sib.relative_to(REPO_ROOT)))

    print(f"IDENTICAL (delete, reference C sibling): {len(identical)}")
    for rel, _sib in identical:
        print(f"  DEL {rel}")
    print(f"\nDIFFER (keep copy): {len(differ)}")
    for rel, _sib in differ:
        print(f"  KEEP {rel}")
    print(f"\nNO C SIBLING (keep copy): {len(no_sibling)}")
    for rel in no_sibling:
        print(f"  KEEP {rel.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Append ``# ty: ignore[unsupported-operator]`` to GL bitmask-OR lines.

``ty`` does not know PyOpenGL's ``Constant.__or__``, so every
``GL.GL_X_BIT | GL.GL_Y_BIT`` reports ``unsupported-operator`` ("both operands
have type Constant") even though it works at runtime. The sibling ports already
carry this ignore on such lines; McNopper (ported before the ty gate existed)
did not. This adds it.

Content-matched (a ``GL.GL_<name>`` immediately followed by ``|``), not by line
number, and idempotent: a line that already has a ``ty: ignore`` is skipped, and
comment-only lines (and GLSL inside triple-quoted strings, which never contain
``GL.GL_``) are not matched. Re-run and re-grep should show zero further
changes. Repo-relative, so it runs from any checkout.
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]
PKG_DIR: pathlib.Path = REPO_ROOT / "python" / "src" / "mcnopper"
# A GL constant used as the left operand of a bitwise OR (``GL_X | GL_Y``) or
# an addition (``GL_..._POSITIVE_X + face`` cubemap face offsets) -- both hit
# the same untyped-``Constant``-operator false positive.
BITMASK_OR: re.Pattern[str] = re.compile(r"GL\.GL_\w+\s*[|+]")
IGNORE: str = "  # ty: ignore[unsupported-operator]"


def main() -> int:
    changed: int = 0
    for path in sorted(PKG_DIR.rglob("*.py")):
        text: str = path.read_text(encoding="utf-8")
        out_lines: list[str] = []
        for raw in text.splitlines(keepends=True):
            body: str = raw.rstrip("\n")
            already: bool = "ty: ignore" in body
            is_comment: bool = body.lstrip().startswith("#")
            if BITMASK_OR.search(body) and not already and not is_comment:
                newline: str = "\n" if raw.endswith("\n") else ""
                out_lines.append(body + IGNORE + newline)
                changed += 1
            else:
                out_lines.append(raw)
        new_text: str = "".join(out_lines)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
    print(f"added {changed} ty:ignore[unsupported-operator] line(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

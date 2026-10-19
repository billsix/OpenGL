#!/usr/bin/env python3
"""Rewrite McNopper demo scripts to import the installed `mcnopper` package.

The demos used a ``sys.path`` bootstrap plus ``import _common as common`` /
``from _common import GL`` (resolved only at runtime, hence the ``# noqa: E402``
and ``# ty: ignore[unresolved-import]``). Now that the shared modules are the
installed ``mcnopper`` package with no leading underscore, this codemod makes
the demos import it normally:

  * drops the ``sys.path.insert(..., parent.parent)`` line and its comment,
  * ``import _X as X`` -> ``from mcnopper import X``,
  * ``from _X import Y`` -> ``from mcnopper.X import Y``.

Processes both the flat demos (``src/mcnopper/example*.py``) and the four
shader-bearing demos kept nested (``src/mcnopper/example*/example*.py``). Run
``ruff check --fix`` after to drop the now-unused ``sys`` / ``Path`` imports.

Usage (from the python/ dir)::

    python ../tasks/adhoc/pipify-port/rewrite_demo_imports.py
"""

from __future__ import annotations

import pathlib
import re

PKG: str = "mcnopper"
# Only the shared modules — never `__future__` or other dunders.
_MOD: str = r"(gltf_gl|gltf|common|ibl|perlin|sh)"


def main() -> int:
    root: pathlib.Path = pathlib.Path("src/mcnopper")
    demos: list[pathlib.Path] = sorted(root.glob("example*.py")) + sorted(
        root.glob("example*/example*.py")
    )
    changed: int = 0
    for f in demos:
        out: list[str] = []
        touched: bool = False
        for ln in f.read_text().splitlines():
            s: str = ln.strip()
            if s.startswith("sys.path.insert(0, str(Path(__file__)"):
                touched = True
                continue
            if s.startswith("# ty can't see the sys.path insert"):
                touched = True
                continue
            m = re.match(rf"^(\s*)import _{_MOD} as (\w+)(\s+#.*)?$", ln)
            if m:
                out.append(f"{m.group(1)}from {PKG} import {m.group(3)}")
                touched = True
                continue
            m = re.match(rf"^(\s*)import _{_MOD}(\s+#.*)?$", ln)
            if m:
                out.append(f"{m.group(1)}from {PKG} import {m.group(2)}")
                touched = True
                continue
            m = re.match(rf"^(\s*)from _{_MOD} import (.+?)(\s+#.*)?$", ln)
            if m:
                out.append(
                    f"{m.group(1)}from {PKG}.{m.group(2)} import {m.group(3)}"
                )
                touched = True
                continue
            out.append(ln)
        if touched:
            f.write_text("\n".join(out) + "\n")
            changed += 1
    print(f"rewrote {changed} demos")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

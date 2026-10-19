"""CLI entry point. Wired into ``[project.scripts]`` as ``mcnopper``.

Usage::

    mcnopper             # prints help
    mcnopper list        # lists all 52 demos
    mcnopper 13          # runs example13
    mcnopper 31          # runs example31
"""

from __future__ import annotations

import sys

from . import demo_count, demo_titles, run


def main(argv: list[str] | None = None) -> int:
    args: list[str] = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return 0
    if args[0] == "list":
        _print_list()
        return 0
    try:
        n: int = int(args[0])
    except ValueError:
        print(f"unrecognised argument: {args[0]!r}", file=sys.stderr)
        _print_help()
        return 2
    if not 1 <= n <= demo_count:
        print(f"demo number must be 1..{demo_count}, got {n}", file=sys.stderr)
        return 2
    run(n)
    return 0


def _print_help() -> None:
    print("Usage:")
    print("  mcnopper              # show this help")
    print(f"  mcnopper list         # list all {demo_count} demos")
    print(f"  mcnopper <N>          # run demo number N (1..{demo_count})")


def _print_list() -> None:
    titles: dict[int, str] = demo_titles()
    for n in sorted(titles):
        print(f"  {n:2d}  {titles[n]}")


if __name__ == "__main__":
    raise SystemExit(main())

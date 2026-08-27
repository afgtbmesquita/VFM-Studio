"""``python -m vfmstudio`` entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vfmstudio", description="VFM Studio")
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Open a .vfmproj.json (or .vfm) on startup.",
    )
    args = parser.parse_args(argv)
    from vfmstudio.app import run_app

    return run_app(project_path=args.project)


if __name__ == "__main__":
    sys.exit(main())

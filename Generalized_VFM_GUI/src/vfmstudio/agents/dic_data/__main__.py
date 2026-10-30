"""Scripted DIC data report: everything the data agent measures, without the GUI or the LLM.

    python -m vfmstudio.agents.dic_data <files or test folders>... [--out DIR] [--list]
                                        [--export-mat DIR]

Each folder is one test. A folder is used only when it contains both the raw
exports and synchronized_DIC_machine_data.csv, with a machine row for each DIC frame.
Pass the machine file yourself when you pass individual exports. Raw files are
only read. Writes facts.json, report.md and figures/ to --out (default: a new
temp folder). Exit codes: 0 done, 2 refused, 1 other failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


def _print_overview(groups) -> None:
    from vfmstudio.agents.dic_data.intake import format_load, frame_match

    for group in groups:
        folders = ", ".join(str(folder) for folder in group.folders)
        print(f"\n== {group.name}  ({len(group.files)} file(s); {folders})")
        match = frame_match(group)
        machine = group.companions.sync_file.name if group.companions.sync_file else "not submitted"
        print(f"   machine data: {machine}  ({'matched' if match.ready else 'not matched'})")
        print(f"   {'step':>4}  {'file':<22} {'DIC':>6} {'campaign':>8} {'time [s]':>9}  {'facets':>7}  format / load")
        for index, info in enumerate(group.files, start=1):
            step = time = None
            if info.dic_number is not None:
                step = group.companions.step_of_dic.get(info.dic_number)
                time = group.companions.time_of_dic.get(info.dic_number)
            status = ("ok  " + format_load(info, group.companions)) if info.ok else "REFUSED: " + "; ".join(info.problems)
            print(
                f"   {index:>4}  {info.path.name:<22} {info.dic_number if info.dic_number is not None else '':>6} "
                f"{step if step is not None else '':>8} {f'{time:.1f}' if time is not None else '':>9}  "
                f"{info.n_facets:>7}  {status}"
            )


def _utf8_console() -> None:
    """Report text has Greek symbols (θ, κ, σ); Windows consoles default to a code page without them."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _utf8_console()
    parser = argparse.ArgumentParser(
        prog="python -m vfmstudio.agents.dic_data",
        description="Convert raw Vic-3D exports to solver data and write the measured DIC report (no LLM).",
    )
    parser.add_argument("paths", nargs="+", help="raw exp-*.csv files and/or test folders")
    parser.add_argument("--out", type=Path, help="output folder (default: new folder under the temp dir)")
    parser.add_argument("--list", action="store_true", help="only show the file overview and format check")
    parser.add_argument("--export-mat", type=Path, metavar="DIR",
                        help="also write one Layout A .mat per step under DIR (never into raw-data folders)")
    args = parser.parse_args(argv)

    from vfmstudio.platform.scipy_wdac import prepare_scipy_native_imports

    prepare_scipy_native_imports()
    from vfmstudio.agents.dic_data.analysis import DataRefused
    from vfmstudio.agents.dic_data.intake import prepare_submission
    from vfmstudio.agents.dic_data.session import export_solver_mats, run_analysis, validate_groups

    missing = [path for path in args.paths if not Path(path).exists()]
    if missing:
        print("Not found: " + ", ".join(missing), file=sys.stderr)
        return 1
    prepared = prepare_submission(args.paths)
    groups = prepared.groups
    _print_overview(groups)
    reasons = prepared.refusal_reasons() + validate_groups(groups)
    if reasons:
        print("\nRefused: the DIC exports and the machine data are not both submitted and matched. "
              "Nothing was analysed.",
              file=sys.stderr)
        for reason in reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 2
    if args.list:
        return 0
    try:
        run = run_analysis(groups, progress=lambda text: print(f"  {text}", file=sys.stderr), workdir=args.out)
        written = export_solver_mats(run, args.export_mat) if args.export_mat else []
    except DataRefused as exc:
        print("\nRefused:", file=sys.stderr)
        for reason in exc.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 2
    print(f"\nReport:  {run.report_path}")
    print(f"Facts:   {run.facts_path}")
    print(f"Figures: {run.workdir / 'figures'}")
    if written:
        print(f"Solver .mat files: {len(written)} under {args.export_mat}")
    for analysis in run.analyses:
        checks = analysis.facts.get("automatic_checks", [])
        print(f"{analysis.name}: {len(checks)} automatic check(s)")
        for check in checks:
            print(f"  - {check['check']}: {check['observation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

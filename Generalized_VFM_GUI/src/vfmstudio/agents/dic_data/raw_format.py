"""The raw Vic-3D facet export the campaign produces, and a strict check against it.

Every raw export in the campaign (``exp-<frame>_<camera>.csv``) carries exactly
these 33 columns in this order. A file that differs is refused, never coerced.
Files are opened read-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vfmstudio.io.csv_import import long_path

RAW_DIC_COLUMNS = (
    "X", "Y", "Z", "U", "V", "W",
    "exx", "eyy", "exy", "e1", "e2", "gamma",
    "cxx", "cyy", "cxy", "c1", "c2", "gamma_c",
    "sigma", "x", "y", "u", "v", "q", "r", "q_ref", "r_ref",
    "SIGMA_X", "SIGMA_Y", "SIGMA_Z", "Xp", "Yp", "Zp",
)
RAW_NAME = re.compile(r"^exp-(\d+)_(\d+)\.csv$", re.IGNORECASE)
MIN_FACETS = 50
_COUNT_CHUNK = 1 << 20


@dataclass
class RawFileInfo:
    path: Path
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dic_number: Optional[int] = None
    n_facets: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems


def _clean(name: str) -> str:
    return name.strip().strip('"').strip()


def _count_data_rows(path: Path) -> int:
    lines = 0
    tail = b""
    with open(long_path(path), "rb") as handle:
        while chunk := handle.read(_COUNT_CHUNK):
            lines += chunk.count(b"\n")
            tail = (tail + chunk)[-64:]
    if tail and not tail.endswith(b"\n"):
        lines += 1
    blank_tail = tail[len(tail.rstrip(b"\r\n \t")):].count(b"\n")
    lines -= max(0, blank_tail - 1)
    return max(0, lines - 1)


def inspect_raw_file(path: Path | str) -> RawFileInfo:
    """Header, first-row and size check of one file against :data:`RAW_DIC_COLUMNS`."""
    path = Path(path)
    info = RawFileInfo(path=path)
    match = RAW_NAME.match(path.name)
    if match:
        info.dic_number = int(match.group(1))
    else:
        info.warnings.append("name is not exp-<frame>_<camera>.csv, so the DIC frame number is unknown")
    if path.suffix.lower() != ".csv":
        info.problems.append("not a .csv file")
        return info
    try:
        with open(long_path(path), "r", encoding="utf-8-sig", errors="replace") as handle:
            header = handle.readline()
            first = handle.readline()
    except OSError as exc:
        info.problems.append(f"cannot be read ({exc.strerror or exc})")
        return info
    columns = tuple(_clean(name) for name in header.split(","))
    if columns != RAW_DIC_COLUMNS:
        missing = [name for name in RAW_DIC_COLUMNS if name not in columns]
        extra = [name for name in columns if name not in RAW_DIC_COLUMNS]
        if missing or extra:
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing[:8]) + ("…" if len(missing) > 8 else ""))
            if extra:
                detail.append("unexpected " + ", ".join(extra[:8]) + ("…" if len(extra) > 8 else ""))
            info.problems.append(
                f"columns differ from the raw Vic-3D export ({'; '.join(detail)})"
            )
        else:
            info.problems.append("columns are the raw Vic-3D set but in a different order")
        return info
    values = first.strip().split(",")
    if len(values) != len(RAW_DIC_COLUMNS):
        info.problems.append(
            f"first data row has {len(values)} values, expected {len(RAW_DIC_COLUMNS)}"
        )
        return info
    try:
        [float(value) for value in values]
    except ValueError:
        info.problems.append("first data row is not numeric")
        return info
    info.n_facets = _count_data_rows(path)
    if info.n_facets < MIN_FACETS:
        info.problems.append(f"only {info.n_facets} facets (at least {MIN_FACETS} expected)")
    return info


def describe_expected_format() -> str:
    return (
        "raw Vic-3D facet export: one row per facet with the 33 columns "
        + ", ".join(RAW_DIC_COLUMNS)
    )

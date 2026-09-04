"""Convert Vic-3D DIC CSVs plus synchronized machine CSVs into Layout A MAT files."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import savemat

from vfmstudio.core.document import LoadStepModel


def natural_path_key(path: Path | str):
    """Sort key so ``step2`` / ``step09`` / ``step10`` order numerically."""
    name = Path(path).name
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", name)]


def path_identity(path: Path | str) -> str:
    candidate = Path(path)
    try:
        if candidate.exists():
            return str(candidate.resolve())
    except OSError:
        pass
    return str(candidate)

KEY_STRIDE = 1_000_000

# Vic-3D (Correlated Solutions) → Kirchhoff plate theory (solver Voigt rows).
#
# Curvature.  Vic-3D reports
#     cxx = +½ ∂²w/∂x² ,  cyy = +½ ∂²w/∂y² ,  cxy = −∂²w/∂x∂y
# The solver stores
#     κ_xx = −∂²w/∂x² ,  κ_yy = −∂²w/∂y² ,  κ_xy = −2 ∂²w/∂x∂y  (engineering twist)
# so SK = (−2 cxx, −2 cyy, +2 cxy).
#
# Strain.  Vic-3D ``exy`` is tensor shear; the solver uses engineering shear
#     ε_xx = exx ,  ε_yy = eyy ,  γ_xy = 2 exy
# so SE = (exx, eyy, 2 exy).
#
# Uncheck the dialog option to copy the CSV columns unchanged (already plate theory).
CURVATURE_FACTORS = (-2.0, -2.0, +2.0)
STRAIN_FACTORS = (1.0, 1.0, 2.0)

VIC3D_CONVENTION_NOTE = (
    "Vic-3D → plate theory: κₓₓ = −2 cxx, κᵧᵧ = −2 cyy, κₓᵧ = +2 cxy "
    "(Vic-3D cxx = ½ ∂²w/∂x², cxy = −∂²w/∂x∂y; solver κₓₓ = −∂²w/∂x², "
    "κₓᵧ = −2 ∂²w/∂x∂y). Strains: εₓₓ = exx, εᵧᵧ = eyy, γₓᵧ = 2 exy "
    "(Vic-3D exy is tensor shear)."
)

JOIN_AUTO = "auto"
JOIN_DIC_NUMBER = "dic_file_number"
JOIN_DIC_NAME = "dic_file_name"
JOIN_ROW = "row_index"

UNIT_AUTO = "auto"
UNIT_NMM = "nmm"
UNIT_NM = "nm"
UNIT_N = "n"
UNIT_CUSTOM = "custom"

UNIT_SCALES = {
    UNIT_NMM: 1.0,
    UNIT_NM: 1000.0,
    UNIT_N: 1.0,
}

REQUIRED_COORD = ("X", "Y")
OPTIONAL_XYZUVW = ("Z", "U", "V", "W")
STRAIN_COLUMNS = ("exx", "eyy", "exy")
CURVATURE_COLUMNS = ("cxx", "cyy", "cxy")
FACET_COLUMNS = ("x", "y")

_SUGGEST_LOAD = re.compile(r"(^|_)d?T[xyz](_|$)", re.IGNORECASE)
_EXP_NUMBER = re.compile(r"exp-(\d+)", re.IGNORECASE)
_NMM = re.compile(r"(^|_)Nmm(_|$)|Nmm", re.IGNORECASE)
_NM = re.compile(r"(^|_)Nm(_|$)", re.IGNORECASE)


class CsvImportError(ValueError):
    """User-facing failure while reading or converting CSV load steps."""


@dataclass
class LoadColumn:
    source_file: Path
    source_column: str
    import_as: str
    unit: str = UNIT_AUTO
    scale: float = 1.0
    sign: float = 1.0
    used: bool = False

    def resolved_scale(self) -> float:
        if self.unit == UNIT_CUSTOM:
            return float(self.scale)
        if self.unit == UNIT_AUTO:
            return infer_unit_scale(self.source_column)
        return float(UNIT_SCALES.get(self.unit, 1.0))


@dataclass
class StepMatch:
    dic_path: Path
    label: str
    row_by_source: Dict[str, Optional[int]] = field(default_factory=dict)

    def row_for(self, source: Path | str) -> Optional[int]:
        return self.row_by_source.get(path_identity(Path(source)))

    def is_ready(self, columns: Sequence[LoadColumn]) -> bool:
        used = [item for item in columns if item.used]
        if not used:
            return True
        return all(self.row_for(item.source_file) is not None for item in used)


def long_path(path: Path | str) -> str:
    """Windows long-path prefix for deep Google Drive paths."""
    text = os.path.abspath(str(path))
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        return "\\\\?\\" + text
    return text


def _clean(name: str) -> str:
    return str(name).strip().strip('"').strip()


def read_csv_columns(path: Path | str) -> List[str]:
    frame = pd.read_csv(long_path(path), nrows=0, skipinitialspace=True)
    return [_clean(name) for name in frame.columns]


def read_load_csv(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(long_path(path), skipinitialspace=True)
    frame.columns = [_clean(name) for name in frame.columns]
    return frame


def is_vic3d_csv(path: Path | str) -> bool:
    columns = set(read_csv_columns(path))
    return "X" in columns and "cxx" in columns


def numeric_columns(frame: pd.DataFrame) -> List[str]:
    names: List[str] = []
    for name in frame.columns:
        series = pd.to_numeric(frame[name], errors="coerce")
        if series.notna().any():
            names.append(str(name))
    return names


def suggest_load_column(name: str) -> bool:
    text = str(name).strip()
    if text.lower() in {"time_s", "time", "dic_file_number"}:
        return False
    return bool(_SUGGEST_LOAD.search(text))


def default_import_name(column: str) -> str:
    name = str(column)
    if name.startswith("Machine_"):
        name = name[len("Machine_") :]
    name = re.sub(r"_+(Nmm|Nm|N)_*$", "", name, flags=re.IGNORECASE)
    name = name.strip("_")
    return name or str(column)


def infer_unit_scale(column: str) -> float:
    text = str(column).replace("·", "").replace(".", "")
    if _NMM.search(text):
        return 1.0
    if _NM.search(text) or re.search(r"Nm", text, re.IGNORECASE):
        return 1000.0
    return 1.0


def dic_file_number(path: Path | str) -> Optional[int]:
    stem = Path(path).stem
    match = _EXP_NUMBER.search(stem)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"(\d+)", stem)
    if not numbers:
        return None
    if len(numbers) >= 2 and len(numbers[-1]) <= 1:
        return int(numbers[-2])
    return int(numbers[-1])


def default_output_folder(
    source_path: Optional[Path],
    dic_paths: Sequence[Path | str],
) -> Path:
    if source_path is not None:
        path = Path(source_path)
        return path.parent / f"{path.stem}_mat"
    if dic_paths:
        return Path(dic_paths[0]).parent / "vfm_steps"
    return Path("vfm_steps")


def collect_load_columns(
    load_paths: Sequence[Path | str],
    frames: Optional[Mapping[str, pd.DataFrame]] = None,
) -> List[LoadColumn]:
    """One row per numeric column, pre-checking likely moment channels."""
    columns: List[LoadColumn] = []
    seen_names: Dict[str, int] = {}
    for path in load_paths:
        path = Path(path)
        frame = (
            frames[path_identity(path)]
            if frames is not None and path_identity(path) in frames
            else read_load_csv(path)
        )
        for name in numeric_columns(frame):
            import_as = default_import_name(name)
            count = seen_names.get(import_as, 0)
            seen_names[import_as] = count + 1
            if count:
                import_as = f"{import_as}_{count + 1}"
            columns.append(
                LoadColumn(
                    source_file=path,
                    source_column=name,
                    import_as=import_as,
                    unit=UNIT_AUTO,
                    scale=infer_unit_scale(name),
                    sign=1.0,
                    used=suggest_load_column(name),
                )
            )
    return columns


def available_join_keys(frames: Sequence[pd.DataFrame]) -> List[str]:
    keys = [JOIN_AUTO]
    if any("DIC_file_number" in frame.columns for frame in frames):
        keys.append(JOIN_DIC_NUMBER)
    name_columns = ("DIC_file_name", "DIC_file", "DIC_file_Name")
    if any(any(name in frame.columns for name in name_columns) for frame in frames):
        keys.append(JOIN_DIC_NAME)
    keys.append(JOIN_ROW)
    return keys


def _name_column(frame: pd.DataFrame) -> Optional[str]:
    for name in ("DIC_file_name", "DIC_file", "DIC_file_Name"):
        if name in frame.columns:
            return name
    return None


def _match_row(
    dic_path: Path,
    frame: pd.DataFrame,
    join_key: str,
    dic_index: int,
    n_dic: int,
) -> Optional[int]:
    key = join_key
    if key == JOIN_AUTO:
        if "DIC_file_number" in frame.columns and dic_file_number(dic_path) is not None:
            key = JOIN_DIC_NUMBER
        elif _name_column(frame) is not None:
            key = JOIN_DIC_NAME
        elif len(frame) == n_dic:
            key = JOIN_ROW
        else:
            return None
    if key == JOIN_DIC_NUMBER:
        if "DIC_file_number" not in frame.columns:
            return None
        number = dic_file_number(dic_path)
        if number is None:
            return None
        series = pd.to_numeric(frame["DIC_file_number"], errors="coerce")
        hits = np.flatnonzero(series.to_numpy() == number)
        return int(hits[0]) if hits.size else None
    if key == JOIN_DIC_NAME:
        column = _name_column(frame)
        if column is None:
            return None
        target = {Path(dic_path).name.lower(), Path(dic_path).stem.lower()}
        for index, value in enumerate(frame[column].astype(str).tolist()):
            cell = Path(str(value).strip())
            if cell.name.lower() in target or cell.stem.lower() in target:
                return int(index)
        return None
    if key == JOIN_ROW:
        if dic_index < len(frame):
            return int(dic_index)
        return None
    return None


def match_dic_steps(
    dic_paths: Sequence[Path | str],
    load_frames: Mapping[str, pd.DataFrame],
    *,
    join_key: str = JOIN_AUTO,
) -> List[StepMatch]:
    paths = sorted((Path(item) for item in dic_paths), key=natural_path_key)
    n_dic = len(paths)
    matches: List[StepMatch] = []
    for index, path in enumerate(paths):
        rows: Dict[str, Optional[int]] = {}
        for identity, frame in load_frames.items():
            rows[identity] = _match_row(path, frame, join_key, index, n_dic)
        matches.append(
            StepMatch(dic_path=path, label=path.stem, row_by_source=rows)
        )
    return matches


def force_preview(
    match: StepMatch,
    columns: Sequence[LoadColumn],
    load_frames: Mapping[str, pd.DataFrame],
) -> Dict[str, Optional[float]]:
    preview: Dict[str, Optional[float]] = {}
    for item in columns:
        if not item.used:
            continue
        row = match.row_for(item.source_file)
        identity = path_identity(item.source_file)
        frame = load_frames.get(identity)
        if row is None or frame is None or item.source_column not in frame.columns:
            preview[item.import_as] = None
            continue
        preview[item.import_as] = _force_value(frame, row, item)
    return preview


def _force_value(frame: pd.DataFrame, row: int, column: LoadColumn) -> Optional[float]:
    if row < 0 or row >= len(frame):
        return None
    raw = pd.to_numeric(frame.iloc[row][column.source_column], errors="coerce")
    if pd.isna(raw):
        return None
    value = float(column.sign) * float(column.resolved_scale()) * float(raw)
    if not np.isfinite(value):
        return None
    return value


def format_match_row(
    match: StepMatch,
    load_frames: Mapping[str, pd.DataFrame],
    columns: Sequence[LoadColumn],
) -> str:
    used = [item for item in columns if item.used]
    sources = { path_identity(item.source_file) for item in used } if used else set(
        match.row_by_source
    )
    if not sources:
        return "kinematics only"
    parts: List[str] = []
    for identity in sources:
        row = match.row_by_source.get(identity)
        if row is None:
            parts.append("unmatched")
            continue
        frame = load_frames.get(identity)
        extra = ""
        if frame is not None:
            if "DIC_file_number" in frame.columns and row < len(frame):
                extra = f", DIC {frame.iloc[row]['DIC_file_number']}"
            elif "Time_s" in frame.columns and row < len(frame):
                extra = f", t={frame.iloc[row]['Time_s']}"
        parts.append(f"row {row + 1}{extra}")
    unique = list(dict.fromkeys(parts))
    return "; ".join(unique)


def _load_dic_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(long_path(path), skipinitialspace=True)
    frame.columns = [_clean(name) for name in frame.columns]
    missing = [name for name in REQUIRED_COORD if name not in frame.columns]
    if missing:
        raise CsvImportError(
            f"{path.name} is not a Vic-3D facet cloud: missing column(s) {missing}."
        )
    if not any(name in frame.columns for name in CURVATURE_COLUMNS) and not any(
        name in frame.columns for name in STRAIN_COLUMNS
    ):
        raise CsvImportError(
            f"{path.name}: missing both curvature ({', '.join(CURVATURE_COLUMNS)}) "
            f"and strain ({', '.join(STRAIN_COLUMNS)}) columns."
        )
    keep = [
        name
        for name in (
            *REQUIRED_COORD,
            *OPTIONAL_XYZUVW,
            *STRAIN_COLUMNS,
            *CURVATURE_COLUMNS,
            *FACET_COLUMNS,
        )
        if name in frame.columns
    ]
    frame = frame.loc[:, keep].copy()
    if all(name in frame.columns for name in FACET_COLUMNS):
        frame = frame.dropna(subset=list(FACET_COLUMNS))
        frame["key"] = (
            frame["x"].astype(np.int64) * KEY_STRIDE + frame["y"].astype(np.int64)
        )
        if frame["key"].duplicated().any():
            raise CsvImportError(f"{path.name}: duplicate facet pixel keys.")
    return frame


def _common_keys(frames: Sequence[pd.DataFrame]) -> np.ndarray:
    common = None
    for frame in frames:
        if "key" not in frame.columns:
            raise CsvImportError(
                "Facet pixel columns x, y are required to keep only facets "
                "present in every DIC file."
            )
        keys = set(frame["key"].tolist())
        common = keys if common is None else (common & keys)
    if not common:
        raise CsvImportError("No facet is correlated in every selected DIC file.")
    return np.array(sorted(common), dtype=np.int64)


def _aligned_frames(
    frames: Sequence[pd.DataFrame],
    *,
    intersect: bool,
) -> List[pd.DataFrame]:
    if not intersect:
        return [frame.reset_index(drop=True) for frame in frames]
    keys = _common_keys(frames)
    aligned: List[pd.DataFrame] = []
    for frame in frames:
        aligned.append(frame.set_index("key").loc[keys].reset_index())
    return aligned


def _field_array(frame: pd.DataFrame, name: str) -> Optional[np.ndarray]:
    if name not in frame.columns:
        return None
    return frame[name].to_numpy(dtype=float)


def _convert_kinematics(
    frame: pd.DataFrame,
    *,
    vic3d_conventions: bool,
) -> Dict[str, np.ndarray]:
    """Map Vic-3D CSV columns onto Layout A SK/SE grids.

    When ``vic3d_conventions`` is true, apply :data:`CURVATURE_FACTORS` and
    :data:`STRAIN_FACTORS` (see ``VIC3D_CONVENTION_NOTE``).  Otherwise the
    CSV values are copied as already plate-theory κ and engineering strain.
    """
    x = _field_array(frame, "X")
    y = _field_array(frame, "Y")
    if x is None or y is None:
        raise CsvImportError("DIC file is missing X or Y.")
    payload: Dict[str, np.ndarray] = {
        "nodes": np.column_stack([x, y]),
        "X": x,
        "Y": y,
    }
    for name in OPTIONAL_XYZUVW:
        values = _field_array(frame, name)
        if values is not None:
            payload[name] = values
    curv = [_field_array(frame, name) for name in CURVATURE_COLUMNS]
    if all(item is not None for item in curv):
        factors = CURVATURE_FACTORS if vic3d_conventions else (1.0, 1.0, 1.0)
        payload["SK1grid"] = factors[0] * curv[0]
        payload["SK2grid"] = factors[1] * curv[1]
        payload["SK3grid"] = factors[2] * curv[2]
    strain = [_field_array(frame, name) for name in STRAIN_COLUMNS]
    if all(item is not None for item in strain):
        factors = STRAIN_FACTORS if vic3d_conventions else (1.0, 1.0, 1.0)
        payload["SE1grid"] = factors[0] * strain[0]
        payload["SE2grid"] = factors[1] * strain[1]
        payload["SE3grid"] = factors[2] * strain[2]
    if "SK1grid" not in payload and "SE1grid" not in payload:
        raise CsvImportError(
            "Need a complete curvature trio (cxx, cyy, cxy) or strain trio "
            "(exx, eyy, exy) to write a measurement MAT file."
        )
    nodes = payload["nodes"]
    if nodes.shape[0] < 2:
        raise CsvImportError(
            "Need at least two facets to write a measurement MAT file."
        )
    x_span = float(nodes[:, 0].max() - nodes[:, 0].min())
    y_span = float(nodes[:, 1].max() - nodes[:, 1].min())
    if x_span <= 0.0 or y_span <= 0.0:
        raise CsvImportError(
            "The imported point cloud has no area (all points are colinear "
            "with a constant X or Y). Check the DIC files and facet filter."
        )
    return payload


def _force_vector(
    match: StepMatch,
    columns: Sequence[LoadColumn],
    load_frames: Mapping[str, pd.DataFrame],
) -> Tuple[Optional[np.ndarray], List[str]]:
    used = [item for item in columns if item.used]
    if not used:
        return None, []
    values: List[float] = []
    labels: List[str] = []
    for item in used:
        row = match.row_for(item.source_file)
        identity = path_identity(item.source_file)
        frame = load_frames.get(identity)
        if row is None or frame is None:
            raise CsvImportError(
                f"{match.dic_path.name}: no matching row in {Path(item.source_file).name}."
            )
        value = _force_value(frame, row, item)
        if value is None:
            raise CsvImportError(
                f"{match.dic_path.name}: {item.source_column} is not numeric "
                f"in {Path(item.source_file).name} row {row + 1}."
            )
        values.append(value)
        labels.append(item.import_as)
    return np.asarray(values, dtype=np.float64), labels


def write_step_mat(
    destination: Path,
    payload: Mapping[str, np.ndarray],
    force: Optional[np.ndarray] = None,
    force_labels: Optional[Sequence[str]] = None,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = {key: np.asarray(value) for key, value in payload.items()}
    if force is not None:
        data["F"] = np.atleast_1d(np.asarray(force, dtype=np.float64)).ravel()
        if force_labels:
            data["F_labels"] = np.array(list(force_labels), dtype=object)
    savemat(long_path(destination), data, do_compression=True)
    return destination


def convert_csv_steps(
    matches: Sequence[StepMatch],
    load_frames: Mapping[str, pd.DataFrame],
    columns: Sequence[LoadColumn],
    output_folder: Path | str,
    *,
    vic3d_conventions: bool = True,
    intersect_facets: bool = True,
    skip: Optional[Sequence[Path | str]] = None,
) -> List[LoadStepModel]:
    """Write one Layout A MAT file per DIC CSV and return load-step models."""
    if not matches:
        return []
    used = [item for item in columns if item.used]
    for match in matches:
        if not match.is_ready(used):
            raise CsvImportError(
                f"{match.dic_path.name}: could not match this DIC file to the "
                "selected load-table columns."
            )
    raw_frames = [_load_dic_frame(match.dic_path) for match in matches]
    frames = _aligned_frames(raw_frames, intersect=intersect_facets)
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    skip_ids = {path_identity(Path(item)) for item in (skip or [])}
    steps: List[LoadStepModel] = []
    for match, frame in zip(matches, frames):
        payload = _convert_kinematics(frame, vic3d_conventions=vic3d_conventions)
        force, labels = _force_vector(match, columns, load_frames)
        mat_path = folder / f"{match.dic_path.stem}.mat"
        write_step_mat(mat_path, payload, force=force, force_labels=labels)
        identity = path_identity(mat_path)
        if identity in skip_ids:
            continue
        steps.append(
            LoadStepModel(data_file=mat_path, label=match.label or mat_path.stem)
        )
    return steps


def load_frames_from_paths(paths: Iterable[Path | str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for path in paths:
        path = Path(path)
        frames[path_identity(path)] = read_load_csv(path)
    return frames

# =============================================================================
#  bc_gui / bc_model.py  (plate)
# -----------------------------------------------------------------------------
#  Internal data model for the plate GUI boundary-condition editor.
#
#  The model mirrors the Python schema understood by
#  ``plate/vfm_data_io.load_boundary_conditions`` so that anything the GUI
#  writes can be read back by the solver without any extra translation step.
#  The solver looks at three top-level lists:
#
#      PRESCRIBED  ->  homogeneous Dirichlet     (selected_dof = 0)
#      LINKING     ->  multi-node tie            (follower - reference = 0)
#      EXTERNAL    ->  external-work measurement group
#
#  ``BcDocument.write_py`` emits a clean, commented Python module with the
#  three lists (one dict per entry, same key names the solver expects), plus
#  optional ``DESCRIPTION`` and ``LAST_UPDATED`` metadata strings.
#  ``BcDocument.from_py_file`` re-imports that module via
#  ``importlib.util`` -- the same mechanism the solver uses -- so a hand-
#  edited file round-trips without information loss.
#
#  Key differences relative to the Q_Matrix membrane bc_model:
#    * ``SUPPORTED_DOFS`` is widened to all six plate DOFs.
#    * Persistence uses Python module files (``.py``) instead of JSON.
#    * A ``.json`` legacy loader is also exposed so that previously-saved
#      Q_Matrix-style JSON files can pre-populate the editor on first open.
# =============================================================================

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# DOF labels supported by the plate solver.  Order matches the offsets used
# by ``plate/vfm_mesh._resolve_dof_component``.
SUPPORTED_DOFS: tuple[str, ...] = ("U", "V", "W", "W_X", "W_Y", "W_XY")

# Categorical role of each DOF.  The mesh canvas uses this to pick the
# right arrow style (single-headed for translational forces/displacements,
# double-headed for moments/rotations) so that translational and rotational
# constraints cannot be visually confused.
TRANSLATIONAL_DOFS: tuple[str, ...] = ("U", "V", "W")
ROTATIONAL_DOFS: tuple[str, ...] = ("W_X", "W_Y", "W_XY")

# Per-analysis subset of DOF labels.  The GUI dropdown uses these to keep
# the user from authoring constraints on DOFs the active analysis does not
# expose; the solver mesh also rejects them at resolve time.
ANALYSIS_TYPE_DOFS: dict[str, tuple[str, ...]] = {
    "in_plane":     ("U", "V"),
    "out_of_plane": ("W", "W_X", "W_Y", "W_XY"),
    "both":         ("U", "V", "W", "W_X", "W_Y", "W_XY"),
}


def allowed_dofs_for(analysis_type: str | None) -> tuple[str, ...]:
    """Return the tuple of DOF labels allowed for the given analysis type.

    When ``analysis_type`` is ``None`` we return the full plate set so
    legacy files (without an explicit ``ANALYSIS_TYPE``) keep loading; the
    caller is expected to surface a separate compatibility warning in that
    case.
    """
    if analysis_type is None:
        return SUPPORTED_DOFS
    key = str(analysis_type).strip().lower()
    if key not in ANALYSIS_TYPE_DOFS:
        raise ValueError(
            f"Unsupported analysis_type {analysis_type!r}; expected one of "
            f"{tuple(ANALYSIS_TYPE_DOFS.keys())}."
        )
    return ANALYSIS_TYPE_DOFS[key]

# Edge labels accepted by ``plate/vfm_mesh._select_edge_nodes``.
SUPPORTED_EDGES: tuple[str, ...] = ("left", "right", "bottom", "top")

# Load types accepted by the plate external-work assembly.
SUPPORTED_LOAD_TYPES: tuple[str, ...] = ("concentrated", "distributed")

# Sort directions accepted by ``plate/vfm_mesh._sort_nodes``.
SUPPORTED_SORT_DIRECTIONS: tuple[str, ...] = ("", "x", "y")

# Internal category labels.  Strings everywhere so they can be passed
# around as plain Python without extra plumbing.
CATEGORY_PRESCRIBED: str = "prescribed"
CATEGORY_LINKING: str = "linking"
CATEGORY_EXTERNAL: str = "external"

ALL_CATEGORIES: tuple[str, ...] = (
    CATEGORY_PRESCRIBED,
    CATEGORY_LINKING,
    CATEGORY_EXTERNAL,
)


# =============================================================================
#  Validation helpers
# =============================================================================

def _validate_dof(dof: str) -> str:
    key = str(dof).strip().upper()
    if key not in SUPPORTED_DOFS:
        raise ValueError(
            f"Unsupported DOF label {dof!r}; expected one of {SUPPORTED_DOFS}."
        )
    return key


def _validate_edge(edge: Optional[str]) -> Optional[str]:
    if edge is None or edge == "":
        return None
    key = str(edge).strip().lower()
    if key not in SUPPORTED_EDGES:
        raise ValueError(
            f"Unsupported edge {edge!r}; expected one of {SUPPORTED_EDGES}."
        )
    return key


def _validate_sort_direction(sort_direction: Optional[str]) -> Optional[str]:
    if sort_direction is None or sort_direction == "":
        return None
    key = str(sort_direction).strip().lower()
    if key not in ("x", "y"):
        raise ValueError(
            f"Unsupported sortDirection {sort_direction!r}; expected 'x' or 'y'."
        )
    return key


def _validate_load_type(load_type: str) -> str:
    key = str(load_type).strip().lower()
    if key not in SUPPORTED_LOAD_TYPES:
        raise ValueError(
            f"Unsupported loadType {load_type!r}; "
            f"expected one of {SUPPORTED_LOAD_TYPES}."
        )
    return key


def _validate_external_value(value: Any) -> Optional[float]:
    """Coerce a user-supplied force / moment value to ``float`` or ``None``.

    ``None`` and the empty string both map to ``None`` (meaning "not set
    -- fall back to the MAT-file F field at solve time").  Any other
    value must be a finite real number.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(
                f"Applied force / moment value must be a finite number "
                f"or blank; got {text!r}."
            ) from exc
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Applied force / moment value must be a finite number or "
            f"blank; got {value!r}."
        ) from exc
    if not (out == out):                          # NaN check
        raise ValueError(
            "Applied force / moment value must be a finite number; "
            "got NaN."
        )
    if out in (float("inf"), float("-inf")):
        raise ValueError(
            "Applied force / moment value must be a finite number; "
            f"got {out!r}."
        )
    return out


def _normalise_nodes(nodes: Optional[Sequence[int]]) -> Optional[List[int]]:
    if nodes is None:
        return None
    out = sorted({int(n) for n in nodes})
    return out if out else None


def dof_kind(dof: str) -> str:
    """Return ``'translational'`` or ``'rotational'`` for a plate DOF label.

    The mesh canvas uses this classification to pick single- vs double-
    headed arrow glyphs.
    """
    key = str(dof).strip().upper()
    if key in TRANSLATIONAL_DOFS:
        return "translational"
    if key in ROTATIONAL_DOFS:
        return "rotational"
    raise ValueError(f"Unknown DOF label {dof!r}.")


# =============================================================================
#  Entry dataclasses
# -----------------------------------------------------------------------------
#  Each entry stores both the user-facing fields (label, DOF, ...) and the
#  selection itself -- either as an explicit ``nodes`` list or as a named
#  ``edge`` shortcut.  ``selection_mode`` records the user's preference so
#  the GUI can re-display the entry the same way on the next launch.
# =============================================================================

@dataclass
class _BaseEntry:
    """Common fields shared by all three categories."""

    label: str
    dof: str
    edge: Optional[str] = None
    nodes: Optional[List[int]] = None
    selection_mode: str = "nodes"

    def __post_init__(self) -> None:
        self.label = str(self.label).strip() or "<unlabeled>"
        self.dof = _validate_dof(self.dof)
        self.edge = _validate_edge(self.edge)
        self.nodes = _normalise_nodes(self.nodes)
        if self.edge is None and not self.nodes:
            raise ValueError(
                f"Entry {self.label!r} has no selection (need 'edge' or 'nodes')."
            )
        mode = str(self.selection_mode).strip().lower()
        if mode not in ("nodes", "edge"):
            mode = "edge" if self.edge is not None else "nodes"
        self.selection_mode = mode

    def _common_payload(self) -> Dict[str, Any]:
        """Return the shared serialisation payload (label, dof, selector)."""
        out: Dict[str, Any] = {"label": self.label, "dof": self.dof}
        if self.selection_mode == "edge" and self.edge is not None:
            out["edge"] = self.edge
        elif self.nodes:
            out["nodes"] = list(self.nodes)
        elif self.edge is not None:
            out["edge"] = self.edge
        return out

    def to_python_dict(self) -> Dict[str, Any]:
        """Return the Python-dict representation written to the .py file."""
        raise NotImplementedError


@dataclass
class PrescribedEntry(_BaseEntry):
    """Homogeneous Dirichlet constraint on the selected DOF."""

    value: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if float(self.value) != 0.0:
            raise ValueError(
                f"Prescribed value must be 0.0 (homogeneous); "
                f"entry {self.label!r} has value={self.value!r}."
            )
        self.value = 0.0

    def to_python_dict(self) -> Dict[str, Any]:
        payload = self._common_payload()
        payload["value"] = 0.0
        return payload


@dataclass
class LinkingEntry(_BaseEntry):
    """Multi-node tie on the selected DOF (homogeneous: follower - ref = 0)."""

    value: float = 0.0
    sort_direction: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if float(self.value) != 0.0:
            raise ValueError(
                f"Linking offset must be 0.0 (homogeneous); "
                f"entry {self.label!r} has value={self.value!r}."
            )
        self.value = 0.0
        self.sort_direction = _validate_sort_direction(self.sort_direction)

    def to_python_dict(self) -> Dict[str, Any]:
        payload = self._common_payload()
        if self.sort_direction:
            payload["sortDirection"] = self.sort_direction
        payload["value"] = 0.0
        return payload


@dataclass
class ExternalEntry(_BaseEntry):
    """External-work measurement group.

    ``value`` is the applied force or moment magnitude associated with
    this group.  It is OPTIONAL: when the MAT file already provides an
    ``F`` array the solver uses that and ignores this field; when the
    MAT file does not provide ``F`` (the new Loading_Data workflow),
    every EXTERNAL entry must declare a numeric ``value`` so the
    identification can pair the virtual displacement with a real load.
    The GUI exposes this as an "Applied force / moment value" line edit.
    """

    load_type: str = "concentrated"
    sort_direction: Optional[str] = None
    value: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.load_type = _validate_load_type(self.load_type)
        self.sort_direction = _validate_sort_direction(self.sort_direction)
        self.value = _validate_external_value(self.value)

    def to_python_dict(self) -> Dict[str, Any]:
        payload = self._common_payload()
        payload["loadType"] = self.load_type
        if self.sort_direction:
            payload["sortDirection"] = self.sort_direction
        if self.value is not None:
            payload["value"] = float(self.value)
        return payload


# =============================================================================
#  Top-level container
# =============================================================================

@dataclass
class BcDocument:
    """In-memory mirror of the boundary-condition Python file."""

    prescribed: List[PrescribedEntry] = field(default_factory=list)
    linking: List[LinkingEntry] = field(default_factory=list)
    external: List[ExternalEntry] = field(default_factory=list)
    mesh_settings: Optional[Any] = None
    solver_settings: Optional[Any] = None
    analysis_type: Optional[str] = None
    description: str = "Generated by bc_gui (plate)"
    last_updated: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    source_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.source_path is not None:
            self.source_path = Path(self.source_path)
        self.solver_settings = coerce_solver_settings(self.solver_settings)
        if self.analysis_type is not None:
            self.analysis_type = str(self.analysis_type).strip().lower() or None

    def incompatible_entries(self,
                             analysis_type: str) -> List[tuple]:
        """Return [(category, index, entry)] for entries with disallowed DOFs.

        Used by the GUI to surface a warning when loading a BC file that
        was authored under a different analysis type.
        """
        allowed = set(allowed_dofs_for(analysis_type))
        out: List[tuple] = []
        for category, entries in (
            (CATEGORY_PRESCRIBED, self.prescribed),
            (CATEGORY_LINKING,    self.linking),
            (CATEGORY_EXTERNAL,   self.external),
        ):
            for idx, entry in enumerate(entries):
                if entry.dof not in allowed:
                    out.append((category, idx, entry))
        return out

    # ------------------------------------------------------------------
    #  Mutation helpers
    # ------------------------------------------------------------------
    def add(self, category: str, entry: _BaseEntry) -> None:
        bucket = self._bucket(category)
        bucket.append(entry)

    def replace(self, category: str, index: int, entry: _BaseEntry) -> None:
        bucket = self._bucket(category)
        if not 0 <= index < len(bucket):
            raise IndexError(
                f"replace: index {index} out of range for category {category!r} "
                f"(size {len(bucket)})."
            )
        bucket[index] = entry

    def remove(self, category: str, index: int) -> None:
        bucket = self._bucket(category)
        if not 0 <= index < len(bucket):
            raise IndexError(
                f"remove: index {index} out of range for category {category!r} "
                f"(size {len(bucket)})."
            )
        bucket.pop(index)

    def count(self, category: str) -> int:
        return len(self._bucket(category))

    def entries(self, category: str) -> List[_BaseEntry]:
        return list(self._bucket(category))

    def _bucket(self, category: str) -> List[_BaseEntry]:
        key = str(category).strip().lower()
        if key == CATEGORY_PRESCRIBED:
            return self.prescribed  # type: ignore[return-value]
        if key == CATEGORY_LINKING:
            return self.linking  # type: ignore[return-value]
        if key == CATEGORY_EXTERNAL:
            return self.external  # type: ignore[return-value]
        raise ValueError(
            f"Unknown BC category {category!r}; expected one of {ALL_CATEGORIES}."
        )

    # ------------------------------------------------------------------
    #  Python-module serialisation
    # ------------------------------------------------------------------
    def write_py(self,
                 output_path: Path,
                 keep_backup: bool = True) -> Path:
        """Atomically write the document as a Python module.

        The output file mirrors the structure of
        ``boundary_conditions_plate_example.py``: top-level ``DESCRIPTION``
        and ``LAST_UPDATED`` strings followed by the three lists
        ``PRESCRIBED``, ``LINKING``, ``EXTERNAL`` that the solver reads.

        If ``keep_backup`` is true and ``output_path`` already exists, the
        previous content is copied to ``<stem>.bak.py`` first.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if keep_backup and output_path.is_file():
            backup_path = output_path.with_suffix(".bak.py")
            try:
                backup_path.write_bytes(output_path.read_bytes())
            except OSError:
                pass

        self.last_updated = datetime.now().strftime("%Y-%m-%d")
        text = self._render_python_module()
        output_path.write_text(text, encoding="utf-8")
        self.source_path = output_path
        return output_path

    def _render_python_module(self) -> str:
        """Build the textual content of the generated Python module."""
        lines: List[str] = []
        lines.append("# =============================================================================")
        lines.append("#  Auto-generated by plate/bc_gui (mesh BC editor).")
        lines.append("# -----------------------------------------------------------------------------")
        lines.append("#  This file is consumed by plate/vfm_data_io.load_boundary_conditions.")
        lines.append("#  You may edit it by hand; the GUI's Load button re-reads it.")
        lines.append("#")
        lines.append("#  Valid DOF labels: U, V, W, W_X, W_Y, W_XY")
        lines.append("#  Valid edges    : left, right, bottom, top")
        lines.append("#  Prescribed / linking values must stay homogeneous (0.0).")
        lines.append("#  MESH_SETTINGS is optional metadata used by the GUI/run_vfm to")
        lines.append("#  restore the exact virtual mesh before applying node-based BCs.")
        lines.append("#  SOLVER_SETTINGS stores GUI solver-tab values so the next load/run")
        lines.append("#  can restore the same identification settings.")
        lines.append("# =============================================================================")
        lines.append("")
        lines.append(f"DESCRIPTION = {self.description!r}")
        lines.append(f"LAST_UPDATED = {self.last_updated!r}")
        if self.analysis_type is not None:
            lines.append(f"ANALYSIS_TYPE = {self.analysis_type!r}")
        else:
            lines.append("ANALYSIS_TYPE = None")
        lines.append("MESH_SETTINGS = " + _format_value(
            mesh_settings_to_payload(self.mesh_settings)
        ))
        lines.append("SOLVER_SETTINGS = " + _format_value(
            solver_settings_to_payload(self.solver_settings)
        ))
        lines.append("")
        lines.append("")
        lines.extend(self._render_list("PRESCRIBED", self.prescribed))
        lines.append("")
        lines.extend(self._render_list("LINKING", self.linking))
        lines.append("")
        lines.extend(self._render_list("EXTERNAL", self.external))
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_list(var_name: str,
                     entries: Sequence[_BaseEntry]) -> List[str]:
        """Render one of the three top-level lists as Python source lines."""
        out: List[str] = []
        if not entries:
            out.append(f"{var_name} = []")
            return out
        out.append(f"{var_name} = [")
        for entry in entries:
            payload = entry.to_python_dict()
            out.append("    {")
            for key, value in payload.items():
                out.append(f"        {key!r}: {_format_value(value)},")
            out.append("    },")
        out.append("]")
        return out

    # ------------------------------------------------------------------
    #  Construction from existing files
    # ------------------------------------------------------------------
    @classmethod
    def from_payload(cls,
                     prescribed: Sequence[Dict[str, Any]],
                     linking: Sequence[Dict[str, Any]],
                     external: Sequence[Dict[str, Any]],
                     mesh_settings: Optional[Any] = None,
                     solver_settings: Optional[Any] = None,
                     analysis_type: Optional[str] = None,
                     description: str = "Generated by bc_gui (plate)",
                     last_updated: Optional[str] = None) -> "BcDocument":
        """Build a document from the three raw dict lists."""
        if last_updated is None:
            last_updated = datetime.now().strftime("%Y-%m-%d")
        return cls(
            prescribed=[_prescribed_from_dict(d) for d in prescribed],
            linking=[_linking_from_dict(d) for d in linking],
            external=[_external_from_dict(d) for d in external],
            mesh_settings=coerce_mesh_settings(mesh_settings),
            solver_settings=coerce_solver_settings(solver_settings),
            analysis_type=analysis_type,
            description=str(description),
            last_updated=str(last_updated),
        )

    @classmethod
    def from_py_file(cls, path: Path) -> "BcDocument":
        """Import a Python BC module and pull its three top-level lists.

        Uses ``importlib.util.spec_from_file_location`` with a unique
        module name (suffixed by the source path stem) so successive
        loads do not reuse a stale ``sys.modules`` entry.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"BC file not found: {path}")
        if path.suffix.lower() != ".py":
            raise ValueError(
                f"from_py_file expects a .py file; got {path.name}."
            )

        module_name = f"_bc_gui_doc_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import BC file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        missing = [
            name for name in ("PRESCRIBED", "LINKING", "EXTERNAL")
            if not hasattr(module, name)
        ]
        if missing:
            raise AttributeError(
                f"BC file {path.name} is missing required variable(s): "
                f"{missing}.  Expected PRESCRIBED, LINKING, EXTERNAL."
            )

        prescribed = list(getattr(module, "PRESCRIBED") or [])
        linking = list(getattr(module, "LINKING") or [])
        external = list(getattr(module, "EXTERNAL") or [])

        description = str(getattr(module, "DESCRIPTION", "Generated by bc_gui (plate)"))
        last_updated = str(getattr(module, "LAST_UPDATED",
                                   datetime.now().strftime("%Y-%m-%d")))
        mesh_settings = getattr(module, "MESH_SETTINGS", None)
        solver_settings = getattr(module, "SOLVER_SETTINGS", None)
        raw_analysis = getattr(module, "ANALYSIS_TYPE", None)
        analysis_type: Optional[str] = None
        if raw_analysis is not None:
            analysis_type = str(raw_analysis).strip().lower() or None

        doc = cls.from_payload(
            prescribed=prescribed,
            linking=linking,
            external=external,
            mesh_settings=mesh_settings,
            solver_settings=solver_settings,
            analysis_type=analysis_type,
            description=description,
            last_updated=last_updated,
        )
        doc.source_path = path
        return doc

    @classmethod
    def from_json_file(cls, path: Path) -> "BcDocument":
        """Read a legacy Q_Matrix-style JSON file (used only as a seed).

        The plate solver itself does NOT accept JSON; this loader exists so
        that an older saved JSON can pre-fill the editor when the user
        provides it as the initial BC file.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"BC file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        metadata = dict(payload.get("metadata") or {})
        description = str(metadata.get("description") or "Generated by bc_gui (plate)")
        last_updated = str(metadata.get("lastUpdated")
                           or datetime.now().strftime("%Y-%m-%d"))

        raw_analysis = metadata.get("analysisType") or metadata.get("analysis_type")
        analysis_type = (str(raw_analysis).strip().lower() or None) \
            if raw_analysis is not None else None

        doc = cls.from_payload(
            prescribed=list(payload.get("prescribed") or []),
            linking=list(payload.get("linking") or []),
            external=list(payload.get("external") or []),
            mesh_settings=(metadata.get("meshSettings")
                           or metadata.get("mesh_settings")),
            solver_settings=(metadata.get("solverSettings")
                             or metadata.get("solver_settings")),
            analysis_type=analysis_type,
            description=description,
            last_updated=last_updated,
        )
        doc.source_path = path
        return doc

    @classmethod
    def from_any_file(cls, path: Path) -> "BcDocument":
        """Convenience loader: dispatch on the file extension.

        ``.py``   -> ``from_py_file``
        ``.json`` -> ``from_json_file``  (legacy seed only)
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".py":
            return cls.from_py_file(path)
        if suffix == ".json":
            return cls.from_json_file(path)
        raise ValueError(
            f"Unsupported BC file extension {suffix!r}; "
            "expected '.py' or '.json'."
        )


# =============================================================================
#  Python-literal formatting helpers
# -----------------------------------------------------------------------------
#  Used by ``BcDocument._render_python_module``.  We keep the output
#  deterministic (sorted keys via insertion order, fixed float formatting)
#  so two consecutive saves of the same in-memory document produce byte-
#  identical files.
# =============================================================================

def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        # Keep the canonical ``0.0`` representation for the homogeneous
        # constraint constant; otherwise fall back to the Python repr.
        if value == 0.0:
            return "0.0"
        return repr(float(value))
    if isinstance(value, int):
        return repr(int(value))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        parts = [
            f"{_format_value(k)}: {_format_value(v)}"
            for k, v in value.items()
        ]
        return "{" + ", ".join(parts) + "}"
    if value is None:
        return "None"
    return repr(value)


def mesh_settings_to_payload(settings: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Convert ``MeshSettings`` into plain Python metadata for persistence."""
    if settings is None:
        return None
    bounds = getattr(settings, "bounds", None)
    if bounds is None:
        return None
    payload = {
        "number_of_elements_x": int(getattr(settings, "number_of_elements_x")),
        "number_of_elements_y": int(getattr(settings, "number_of_elements_y")),
        "bounds": {
            "x_min": float(getattr(bounds, "x_min")),
            "x_max": float(getattr(bounds, "x_max")),
            "y_min": float(getattr(bounds, "y_min")),
            "y_max": float(getattr(bounds, "y_max")),
        },
    }
    x_nodes = getattr(settings, "x_nodes", None)
    y_nodes = getattr(settings, "y_nodes", None)
    if x_nodes is not None and y_nodes is not None:
        payload["x_nodes"] = [float(x) for x in x_nodes]
        payload["y_nodes"] = [float(y) for y in y_nodes]
    return payload


def solver_settings_to_payload(settings: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Return plain Python solver-tab metadata for persistence."""
    coerced = coerce_solver_settings(settings)
    if coerced is None:
        return None
    return dict(coerced)


def coerce_mesh_settings(payload: Optional[Any]) -> Optional[Any]:
    """Return a ``MeshSettings`` instance from saved metadata, if present."""
    if payload is None:
        return None
    if hasattr(payload, "bounds"):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(
            "MESH_SETTINGS must be a dict with element counts and bounds."
        )

    bounds_payload = payload.get("bounds")
    if not isinstance(bounds_payload, dict):
        raise TypeError("MESH_SETTINGS['bounds'] must be a dict.")

    from vfm_user_inputs import Bounds, MeshSettings

    x_nodes = payload.get("x_nodes")
    y_nodes = payload.get("y_nodes")

    return MeshSettings(
        number_of_elements_x=int(payload["number_of_elements_x"]),
        number_of_elements_y=int(payload["number_of_elements_y"]),
        bounds=Bounds(
            x_min=float(bounds_payload["x_min"]),
            x_max=float(bounds_payload["x_max"]),
            y_min=float(bounds_payload["y_min"]),
            y_max=float(bounds_payload["y_max"]),
        ),
        x_nodes=([float(x) for x in x_nodes]
                 if x_nodes is not None else None),
        y_nodes=([float(y) for y in y_nodes]
                 if y_nodes is not None else None),
    )


def coerce_solver_settings(payload: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Validate and canonicalise saved solver-tab metadata."""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("SOLVER_SETTINGS must be a dict or None.")

    aliases = {
        "sample_area_total": ("sample_area_total", "sampleAreaTotal"),
        "max_iterations": ("max_iterations", "maxIterations"),
        "convergence_tolerance": (
            "convergence_tolerance",
            "convergenceTolerance",
        ),
    }

    def _first_present(names: tuple[str, ...]) -> tuple[bool, Any]:
        for name in names:
            if name in payload:
                return True, payload[name]
        return False, None

    out: Dict[str, Any] = {}

    found, raw = _first_present(aliases["sample_area_total"])
    if found:
        if raw is None:
            out["sample_area_total"] = None
        else:
            value = float(raw)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "SOLVER_SETTINGS['sample_area_total'] must be a "
                    "positive finite number or None."
                )
            out["sample_area_total"] = value

    found, raw = _first_present(aliases["max_iterations"])
    if found and raw is not None:
        value = int(raw)
        if value < 1:
            raise ValueError(
                "SOLVER_SETTINGS['max_iterations'] must be >= 1."
            )
        out["max_iterations"] = value

    found, raw = _first_present(aliases["convergence_tolerance"])
    if found and raw is not None:
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                "SOLVER_SETTINGS['convergence_tolerance'] must be a "
                "positive finite number."
            )
        out["convergence_tolerance"] = value

    return out or None


# =============================================================================
#  Dict -> entry helpers
# -----------------------------------------------------------------------------
#  Defensive about missing keys so a hand-edited file still parses as long
#  as the solver itself could have parsed it.
# =============================================================================

def _shared_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    edge_value = d.get("edge")
    nodes_value = d.get("nodes")

    edge: Optional[str] = None
    if isinstance(edge_value, str) and edge_value.strip():
        edge = edge_value
    nodes: Optional[List[int]] = None
    if nodes_value:
        nodes = [int(n) for n in nodes_value]

    selection_mode = "edge" if edge is not None and not nodes else "nodes"

    return {
        "label": str(d.get("label", "<unlabeled>")),
        "dof": str(d.get("dof", "U")),
        "edge": edge,
        "nodes": nodes,
        "selection_mode": selection_mode,
    }


def _prescribed_from_dict(d: Dict[str, Any]) -> PrescribedEntry:
    return PrescribedEntry(**_shared_kwargs(d), value=float(d.get("value", 0.0)))


def _linking_from_dict(d: Dict[str, Any]) -> LinkingEntry:
    return LinkingEntry(
        **_shared_kwargs(d),
        value=float(d.get("value", 0.0)),
        sort_direction=d.get("sortDirection"),
    )


def _external_from_dict(d: Dict[str, Any]) -> ExternalEntry:
    raw_value = d.get("value", None)
    return ExternalEntry(
        **_shared_kwargs(d),
        load_type=str(d.get("loadType", "concentrated")),
        sort_direction=d.get("sortDirection"),
        value=raw_value,
    )

# VFM Studio

CAE-style desktop application for the **virtual fields method** on Kirchhoff–Love
plates. This package is the GUI and session layer. Identification is performed
by the untouched sibling solver in `../Generalized VFM/`.

```text
python -m vfmstudio
python -m vfmstudio --project "../Generalized VFM/projects/campaign.vfmproj.json"
```

## Layout

| Path | Role |
|---|---|
| `src/vfmstudio/core/` | Qt-free document, solver bridge, jobs |
| `src/vfmstudio/gui/` | PySide6 main window, docks, viewport |
| `src/vfmstudio/io/` | `.vfmproj.json`, HDF5 `.vfm`, VTK export |
| `tests/` | bridge, document, GUI smoke |

The GUI never stores mechanics state. Widgets edit a Pydantic session
document; `vfm_project.run_project` is the only identification entry point.

## Install

```text
cd Generalized_VFM_GUI
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Requires the sibling folder `Generalized VFM` next to this one (or set
`GVFM_CORE_PATH`).

## Architecture

```text
PySide6  →  SessionDocument  →  vfm_project.run_project
QtInteractor (PyVista/VTK)     PyQtGraph plots
```

See `THIRD_PARTY_LICENSES.md` for the dependency license inventory.

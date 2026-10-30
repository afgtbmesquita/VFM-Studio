# VFM Studio

CAE-style desktop application for the **virtual fields method** on Kirchhoff–Love
plates. This package is the GUI and session layer. Identification is performed
by the untouched sibling solver in `../Generalized VFM/`.

```text
python -m vfmstudio
python -m vfmstudio --project "../Generalized VFM/projects/campaign.vfmproj.json"
python -m vfmstudio --project "../Generalized VFM/projects/samples_ab_all.vfmproj.json"
```

## Layout

| Path | Role |
|---|---|
| `src/vfmstudio/core/` | Qt-free document, solver bridge, jobs |
| `src/vfmstudio/gui/` | PySide6 main window, docks, viewport |
| `src/vfmstudio/io/` | `.vfmproj.json`, HDF5 `.vfm`, VTK export |
| `src/vfmstudio/agents/` | Qt-free Claude Code CLI runner and agent registry (Assistant panel) |
| `src/vfmstudio/agents/dic_data/` | DIC data agent: raw Vic-3D format gate, intake/ordering, solver-data build, noise-vs-model and read-quality analysis, figures, scripted report (`report.py`) and reply audit (`audit.py`); read-only on raw data. Run without GUI or LLM: `python -m vfmstudio.agents.dic_data <folders> --out DIR`. Export also writes a `data_agent/` record next to the `.mat` files. |
| `src/vfmstudio/gui/data_agent_window.py` | Window opened by "Use data agent": arrange files into tests, analyse, inspect solver data, export `.mat` |
| `src/vfmstudio/agents/model_build/` | Model build agent: reads imported load steps and the DIC data agent's record. Boundary-condition reference is `bc_library.json` in this folder (scheme cards + adopt/decline reward). The agent never opens `.vfm` files. Offline rebuild: `python -m vfmstudio.agents.model_build.precedents ROOT` |
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

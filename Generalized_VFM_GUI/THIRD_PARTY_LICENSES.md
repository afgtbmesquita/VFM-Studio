# Third-party licenses

VFM Studio links against the following packages. This file is the license
inventory for the application. It is **not** legal advice. PySide6 in
particular is LGPLv3; a commercial Qt license is a later product decision,
not a v1 requirement.

The sibling solver in `../Generalized VFM/` is **not** copied into this tree.
It is imported at runtime from that folder.

| Package | Constraint | SPDX | Use in VFM Studio | Project |
|---|---|---|---|---|
| PySide6 | >=6.5 | LGPL-3.0-or-later OR GPL-3.0-or-later OR LicenseRef-Qt-Commercial | Desktop shell (widgets, docks, menus) | https://doc.qt.io/qtforpython-6/ |
| superqt | >=0.6 | BSD-3-Clause | Range sliders, extra Qt widgets | https://github.com/pyapp-kit/superqt |
| PyVista | >=0.43 | MIT | Mesh / field visualization API | https://docs.pyvista.org/ |
| pyvistaqt | >=0.11 | MIT | Embed PyVista plotter in Qt | https://github.com/pyvista/pyvistaqt |
| VTK | >=9.2 | BSD-3-Clause | GPU scientific visualization backend | https://vtk.org/ |
| PyQtGraph | >=0.13 | MIT | Interactive XY plots and heatmaps | https://www.pyqtgraph.org/ |
| NumPy | >=1.24 | BSD-3-Clause | Arrays | https://numpy.org/ |
| SciPy | >=1.10 | BSD-3-Clause | Linear algebra, `.mat` I/O | https://scipy.org/ |
| scikit-fem | >=8.0 | BSD-3-Clause | Required by the sibling plate mesh (`skfem`) | https://github.com/kinnala/scikit-fem |
| xarray | >=2023.1 | Apache-2.0 | Labeled DIC / kinematic fields | https://xarray.dev/ |
| Pydantic | >=2.0 | MIT | Session document validation | https://docs.pydantic.dev/ |
| Pint | >=0.22 | BSD-3-Clause | Unit registry (mm, N, N·mm) | https://pint.readthedocs.io/ |
| h5py | >=3.8 | BSD-3-Clause | Native `.vfm` (HDF5) project container | https://www.h5py.org/ |
| meshio | >=5.3 | MIT | Mesh import/export | https://github.com/nschloe/meshio |
| tifffile | >=2023.1 | BSD-3-Clause | TIFF image I/O | https://github.com/cgohlke/tifffile |
| Pillow | >=10.0 | HPND | Standard image I/O | https://python-pillow.org/ |
| Shapely | >=2.0 | BSD-3-Clause | 2-D ROI geometry | https://shapely.readthedocs.io/ |
| Matplotlib | >=3.7 | LicenseRef-PSF-Matplotlib | Publication figure export only | https://matplotlib.org/ |
| pytest | >=7.4 | MIT | Tests (dev) | https://pytest.org/ |
| pytest-qt | >=4.2 | MIT | Qt widget tests (dev) | https://pytest-qt.readthedocs.io/ |
| Ruff | >=0.4 | MIT | Linting (dev) | https://docs.astral.sh/ruff/ |
| mypy | >=1.8 | MIT | Type checking (dev) | https://mypy-lang.org/ |
| PyInstaller | >=6.0 | GPL-2.0-or-later WITH Bootloader-exception | Frozen Windows build (packaging only; not imported by the app) | https://pyinstaller.org/ |

## Notes

- **PySide6 / Qt.** Dynamic linking via the official PySide6 wheels. LGPL
  obligations (dynamic link, ability to relink against a modified Qt) apply
  if you distribute a binary. Do not add GPL libraries to this application.
- **No Gmsh** is bundled, because of GPL implications.
- **Matplotlib** is used only to export publication figures, not as the
  interactive viewport.
- Exact license texts of each package are those shipped with the installed
  distribution (`LICENSE`, `LICENSES/`, or metadata).

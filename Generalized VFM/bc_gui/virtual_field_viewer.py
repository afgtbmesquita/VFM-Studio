# =============================================================================
#  bc_gui / virtual_field_viewer.py
# -----------------------------------------------------------------------------
#  Post-run 3D viewer for optimal virtual fields.
#
#  For each constitutive parameter k and iteration, plots the mid-surface in
#  the current configuration **x = X + u** (Cartesian x, y, z): reference
#  (X, Y) from the mesh plus in-plane (u, v) and/or transverse w according to
#  ``analysis_type`` (see ``vfm_virtual_field_post.nodal_deformed_xyz``).
#
#  The strain probe lists only the rows the active analysis exposes: three
#  membrane components in ``in_plane``, three curvature components in
#  ``out_of_plane``, all six in ``both``.
# =============================================================================

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


# Strain-probe row labels per analysis type (matches the per-mode B matrix).
_STRAIN_ROW_LABELS = {
    "in_plane":     ("εxx", "εyy", "γxy"),
    "out_of_plane": ("κxx", "κyy", "κxy"),
    "both":         ("εxx", "εyy", "γxy", "κxx", "κyy", "κxy"),
}


class VirtualFieldViewerWindow(QMainWindow):
    """3D deformed mid-surface **x = X + u** per optimal virtual field (parameter)."""

    def __init__(self, mesh_pkg, identification_result) -> None:
        super().__init__()
        self.setWindowTitle("Plate VFM — optimal virtual fields (3D)")
        self.resize(1280, 820)

        self._mesh_pkg = mesh_pkg
        self._ident = identification_result
        self._labels = list(identification_result.constitutive_model.parameter_labels)
        self._n_param = len(self._labels)
        self._history = identification_result.optimal_nodal_history
        self._n_iter = len(self._history)

        analysis_type = getattr(mesh_pkg, "analysis_type", "both")
        self._analysis_type = analysis_type
        self._strain_row_labels: Tuple[str, ...] = _STRAIN_ROW_LABELS.get(
            analysis_type, _STRAIN_ROW_LABELS["both"]
        )

        n_panels = self._n_param

        root = QWidget()
        self.setCentralWidget(root)
        h = QHBoxLayout(root)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.West)
        for it in range(self._n_iter):
            self._tabs.addTab(QWidget(), f"Iteration {it + 1}")
        self._tabs.setCurrentIndex(max(0, self._n_iter - 1))
        self._tabs.currentChanged.connect(self._on_tab_changed)
        h.addWidget(self._tabs, 0)

        right_split = QSplitter(Qt.Vertical)
        plot_host = QWidget()
        pv = QVBoxLayout(plot_host)
        pv.setContentsMargins(0, 0, 0, 0)

        ncols = int(math.ceil(math.sqrt(n_panels)))
        nrows = int(math.ceil(n_panels / ncols))
        # constrained_layout + multiple Axes3D triggers a matplotlib bug
        # (Bbox.union gets None from get_tightbbox during Qt draw).
        self._fig = Figure(figsize=(10, 7), constrained_layout=False)
        self._axes_flat: List = []
        for i in range(nrows * ncols):
            ax = self._fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            if i >= n_panels:
                ax.set_visible(False)
            else:
                self._axes_flat.append(ax)
        self._fig.subplots_adjust(wspace=0.15, hspace=0.22, top=0.93)

        self._canvas = FigureCanvasQTAgg(self._fig)
        pv.addWidget(self._canvas, 1)
        self._nav = NavigationToolbar2QT(self._canvas, plot_host)
        pv.addWidget(self._nav)

        pick_row = QHBoxLayout()
        self._pick_btn = QPushButton("Pick node")
        self._pick_btn.setCheckable(True)
        self._pick_btn.setToolTip(
            "When on, click near a node on any 3D subplot to select it.  "
            "Strains in the probe correspond to the virtual field for the "
            "parameter of that subplot.  Coordinates x, y, z are the current "
            "configuration (reference mid-plane plus that field's displacement)."
        )
        self._pick_btn.toggled.connect(self._on_toggle_pick)
        pick_row.addWidget(self._pick_btn)
        self._diagnosis_btn = QPushButton("Identifiability Diagnosis")
        self._diagnosis_btn.setToolTip(
            "Open H-matrix, singular-value, determinant, CID, and "
            "correlation-index diagnostics for the completed identification."
        )
        self._diagnosis_btn.clicked.connect(self._open_identifiability_diagnosis)
        pick_row.addWidget(self._diagnosis_btn)
        pick_row.addStretch(1)
        pv.addLayout(pick_row)

        bottom = QWidget()
        bh = QHBoxLayout(bottom)
        bh.addStretch(1)
        probe = QGroupBox("Probe")
        pv_probe = QVBoxLayout(probe)
        self._probe_label = QLabel(
            "Turn on “Pick node”, then click a point on a surface.\n"
            "Use the matplotlib toolbar to rotate / pan / zoom."
        )
        self._probe_label.setWordWrap(True)
        self._probe_label.setMinimumWidth(340)
        pv_probe.addWidget(self._probe_label)
        bh.addWidget(probe)
        right_split.addWidget(plot_host)
        right_split.addWidget(bottom)
        right_split.setStretchFactor(0, 10)
        right_split.setStretchFactor(1, 0)
        h.addWidget(right_split, 1)

        self._pick_enabled = False
        self._pick_scatters: List[Optional[object]] = [None] * n_panels
        self._hl_artists: List[List[object]] = [[] for _ in range(n_panels)]
        self._selected_node: Optional[int] = None
        self._probe_panel_idx: int = 0

        self._canvas.mpl_connect("pick_event", self._on_pick)
        self._current_it = self._tabs.currentIndex()
        self._refresh_plots()

    def _open_identifiability_diagnosis(self) -> None:
        from .identifiability_viewer import launch_identifiability_diagnosis

        launch_identifiability_diagnosis(self._ident)

    def _on_toggle_pick(self, checked: bool) -> None:
        self._pick_enabled = bool(checked)
        for sc in self._pick_scatters:
            if sc is not None:
                sc.set_picker(12.0 if self._pick_enabled else False)
        self._canvas.draw_idle()

    def _on_tab_changed(self, index: int) -> None:
        self._current_it = int(index)
        self._selected_node = None
        self._probe_panel_idx = 0
        self._clear_highlights()
        self._refresh_plots()

    def _clear_highlights(self) -> None:
        for idx, artists in enumerate(self._hl_artists):
            for art in artists:
                try:
                    art.remove()
                except Exception:
                    pass
            self._hl_artists[idx] = []
        self._probe_label.setText(
            "Turn on “Pick node”, then click a point on a surface.\n"
            "Use the matplotlib toolbar to rotate / pan / zoom."
        )

    def _scatter_index_to_panel(self, event) -> int:
        art = getattr(event, "artist", None)
        for idx, sc in enumerate(self._pick_scatters):
            if sc is not None and art is sc:
                return idx
        return 0

    def _on_pick(self, event) -> None:
        if not self._pick_enabled:
            return
        if event.ind is None or len(event.ind) == 0:
            return
        nid = int(event.ind[0])
        self._selected_node = nid
        self._probe_panel_idx = self._scatter_index_to_panel(event)
        self._update_probe_and_highlights()

    def _refresh_plots(self) -> None:
        from vfm_virtual_field_post import (
            nodal_deformed_xyz,
            nodal_field,
            structured_surface_grids,
            structured_surface_grids_from_nodal_xyz,
        )

        u_list = self._history[self._current_it]

        self._pick_scatters = [None] * self._n_param
        self._clear_highlights()

        for idx, ax in enumerate(self._axes_flat):
            ax.clear()
            k = idx
            cx, cy, cz = nodal_deformed_xyz(self._mesh_pkg, u_list[k])
            X, Y, Z = structured_surface_grids_from_nodal_xyz(
                self._mesh_pkg, cx, cy, cz
            )
            if self._analysis_type == "in_plane":
                uu = nodal_field(self._mesh_pkg, u_list[k], "U")
                vv = nodal_field(self._mesh_pkg, u_list[k], "V")
                mag_n = np.hypot(uu, vv)
                _, _, mag_grid = structured_surface_grids(self._mesh_pkg, mag_n)
                corners = np.stack([
                    mag_grid[:-1, :-1],
                    mag_grid[1:, :-1],
                    mag_grid[1:, 1:],
                    mag_grid[:-1, 1:],
                ], axis=0)
                quad_mean = np.nanmean(corners, axis=0)
                mn = float(np.nanmin(quad_mean))
                mx = float(np.nanmax(quad_mean))
                if (not np.isfinite(mn)) or (not np.isfinite(mx)) or mx - mn <= 0.0:
                    rgba = cm.viridis(0.5)
                    fc = np.broadcast_to(rgba, quad_mean.shape + (4,)).copy()
                else:
                    norm = mcolors.Normalize(vmin=mn, vmax=mx)
                    fc = cm.viridis(norm(quad_mean))
                ax.plot_surface(
                    X,
                    Y,
                    Z,
                    facecolors=fc,
                    rstride=1,
                    cstride=1,
                    linewidth=0,
                    antialiased=True,
                    alpha=0.92,
                    shade=False,
                )
            else:
                ax.plot_surface(
                    X,
                    Y,
                    Z,
                    cmap="viridis",
                    linewidth=0,
                    antialiased=True,
                    alpha=0.92,
                )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(self._labels[k])

            sc = ax.scatter(
                cx,
                cy,
                cz,
                s=80.0,
                alpha=0.0,
                depthshade=False,
                picker=(12.0 if self._pick_enabled else False),
            )
            self._pick_scatters[idx] = sc

        if self._selected_node is not None:
            self._update_probe_and_highlights()
        self._canvas.draw_idle()

    def _update_probe_and_highlights(self) -> None:
        nid = self._selected_node
        if nid is None:
            return

        from vfm_virtual_field_post import (
            nodal_deformed_xyz,
            nodal_virtual_generalized_strains,
        )

        u_list = self._history[self._current_it]
        Xr = float(self._mesh_pkg.node_coords[nid, 0])
        Yr = float(self._mesh_pkg.node_coords[nid, 1])
        panel_idx = int(np.clip(self._probe_panel_idx, 0, self._n_param - 1))
        k = panel_idx
        xd, yd, zd = nodal_deformed_xyz(self._mesh_pkg, u_list[k])
        x = float(xd[nid])
        y = float(yd[nid])
        z = float(zd[nid])
        ek = nodal_virtual_generalized_strains(self._mesh_pkg, u_list[k])[nid]

        lines = [
            f"Node {nid}  (virtual field: {self._labels[k]})",
            f"x = {x:.6g}  (current)",
            f"y = {y:.6g}  (current)",
            f"z = {z:.6g}  (current)",
            f"X = {Xr:.6g}, Y = {Yr:.6g}  (reference mid-plane)",
            "",
            "Virtual strains (nodal patch average)",
        ]
        for label, value in zip(self._strain_row_labels, ek):
            lines.append(f"{label} = {value:.6g}")
        self._probe_label.setText("\n".join(lines) + "\n")

        for idx, ax in enumerate(self._axes_flat):
            for art in self._hl_artists[idx]:
                try:
                    art.remove()
                except Exception:
                    pass
            self._hl_artists[idx] = []
            kk = idx
            cx, cy, cz = nodal_deformed_xyz(self._mesh_pkg, u_list[kk])
            hl = ax.scatter(
                [float(cx[nid])],
                [float(cy[nid])],
                [float(cz[nid])],
                color="red",
                s=120.0,
                depthshade=False,
                edgecolors="darkred",
                linewidths=1.0,
            )
            self._hl_artists[idx].append(hl)

        self._canvas.draw_idle()


def launch_virtual_field_viewer(mesh_pkg, identification_result) -> None:
    """Open the viewer modally (blocks until the window is closed)."""
    import sys
    from PySide6.QtCore import QEventLoop
    from PySide6.QtWidgets import QApplication

    existing_app = QApplication.instance()
    owns_app = existing_app is None
    app = existing_app if existing_app is not None else QApplication(sys.argv)
    if owns_app:
        app.setQuitOnLastWindowClosed(True)

    window = VirtualFieldViewerWindow(mesh_pkg, identification_result)
    window.setAttribute(Qt.WA_DeleteOnClose, True)
    window.show()
    window.raise_()
    window.activateWindow()

    if owns_app:
        window.destroyed.connect(app.quit)
        app.exec()
    else:
        loop = QEventLoop()
        window.destroyed.connect(loop.quit)
        loop.exec()

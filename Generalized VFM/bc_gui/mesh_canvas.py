# =============================================================================
#  bc_gui / mesh_canvas.py  (plate)
# -----------------------------------------------------------------------------
#  Interactive mesh viewer for the plate-GUI BC editor.
#
#  The canvas renders the virtual Q1 mesh produced by ``vfm_mesh.build_mesh``
#  (node coordinates + 4-corner element table) and supports three exclusive
#  selection modes:
#
#      * "navigate" -- pan and zoom only, no picking
#      * "nodes"    -- click or rectangle-drag to select nodes
#      * "elements" -- click or rectangle-drag to select elements
#
#  Selection modifiers:
#      * Plain click / drag      -> replaces the current selection
#      * Shift / Ctrl click/drag -> toggles items within the new pick
#
#  Two boolean overlays can be turned on independently:
#      * show_node_ids
#      * show_element_ids
#
#  When a strain / curvature contour is active, the main window can enable
#  ``MeshCanvas.set_probe_mode(True)`` so left-clicks place read-outs at the
#  nearest measurement node; right-click still removes probe markers.
#
#  Plate-specific BC overlay rules (the visual encoding the user asked for):
#      * Forces / translational displacements (U, V, W)
#          - U: single-headed arrow along +x
#          - V: single-headed arrow along +y
#          - W: out-of-page marker (circle + center dot) -- the transverse
#               DOF is perpendicular to the canvas
#      * Moments / rotations — axis pairing matches
#        ``Forward_ABD_solution/plate_postprocess._BC_GLYPH_TABLE`` (w_x → y,
#        w_y → x in the x–y plane).  Drawing style: **two** single-headed arrows
#        rooted at the node, **same direction** along that axis (+y for Wx,
#        +x for Wy), **different lengths** — never a bidirectional double-head
#        (rotation is not drawn as ± opposing arrows).
#      * W_XY (w_xy): same prescribed / external colours as rotations, drawn
#        as a small disk (twist DOF).
#      * Colour map (in-plane glyphs):
#          - orange / red for prescribed / external U, V, W
#          - purple / dark green for prescribed / external W_X, W_Y, W_XY
#            (Wxy uses the same hue as the twin-arrow rotation glyph)
#          - purple for linking (tied) nodes
#
#  Selection changes emit ``selection_changed`` (Qt signal) so the main
#  window can refresh the status bar and the BC dialogs.
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.tri import Triangulation
from matplotlib.widgets import RectangleSelector

# We only import MeshPackage for typing; the canvas itself just reads the
# arrays so a duck-typed object with the same fields also works.
try:
    from vfm_mesh import MeshPackage  # type: ignore
except Exception:  # pragma: no cover -- import path varies at runtime
    MeshPackage = object  # type: ignore[misc, assignment]


# Visual constants kept at module scope so the main window can re-use them
# in its legend if needed.
NODE_BASE_SIZE = 30.0
NODE_SELECTED_SIZE = 90.0
LINKING_NODE_SIZE = NODE_BASE_SIZE
ELEMENT_FACE_DEFAULT_ALPHA = 0.0      # transparent face by default
ELEMENT_FACE_SELECTED_ALPHA = 0.45    # filled when selected

# Boundary-condition overlay colours (Abaqus-inspired).  Used both on the
# canvas and in the bottom-center legend.
COLOR_PRESCRIBED = "#ff8800"          # orange  -- prescribed U, V, W
COLOR_EXTERNAL = "#d62728"            # red     -- external U, V, W
# Rotational W_X / W_Y: match ``plate_postprocess._BC_COLOR_ROT`` for prescribed;
# external uses a distinct green (forward 3-D plot only shows Dirichlet BCs).
BC_ROT_PRESCRIBED = "#9467bd"
BC_ROT_EXTERNAL = "#0a5230"
COLOR_LINKING = "#7f0eaa"             # purple  -- linking (tied) nodes
COLOR_SELECTION = "#00b4d8"           # cyan    -- transient picking highlight
COLOR_MEASUREMENTS = "#bdbdbd"        # light grey -- experimental point cloud
COLOR_PROBE_MARKER = "#ffc107"        # amber square -- contour read-out markers
# Material-frame axes.  Deliberately outside the BC palette above -- these say
# which way the PLIES run, not what is constrained, and the two layers overlap
# on screen.  Teal/brown stay distinguishable under deuteranopia and
# protanopia, where a green/orange pair would collapse.
COLOR_MATERIAL_AXIS_1 = "#0b7285"     # teal  -- material 1-axis (fibre)
COLOR_MATERIAL_AXIS_2 = "#8c5a1a"     # brown -- material 2-axis


def _bc_glyph_color(category: str, dof: str) -> str:
    """Return the overlay colour for a BC glyph (match Forward_ABD where set)."""
    key = str(dof).strip().upper()
    if key in ("W_X", "W_Y", "W_XY"):
        return BC_ROT_PRESCRIBED if category == "prescribed" else BC_ROT_EXTERNAL
    if category == "prescribed":
        return COLOR_PRESCRIBED
    return COLOR_EXTERNAL


# =============================================================================
#  Data-overlay identifiers
# -----------------------------------------------------------------------------
#  The "Show measurement points" toolbar dropdown in the main window emits one
#  of these identifiers via ``MeshCanvas.set_active_overlay``.  They are kept
#  at module scope so the main window can reference them by name without
#  re-defining them locally.
#
#  Field overlays render a filled tri-contour of the corresponding row of
#  ``PointMeasurements.strains``.  Row order depends on the active analysis:
#    in_plane     -> [Eps1, Eps2, Eps6]
#    out_of_plane -> [K1, K2, K6]
#    both         -> [Eps1, Eps2, Eps6, K1, K2, K6]
#  The canvas resolves row indices from the mesh's ``analysis_type``.
# =============================================================================

OVERLAY_NONE = ""
OVERLAY_POINTS = "points"
OVERLAY_EPS_X = "eps_x"
OVERLAY_EPS_Y = "eps_y"
OVERLAY_EPS_XY = "eps_xy"
OVERLAY_K_X = "k_x"
OVERLAY_K_Y = "k_y"
OVERLAY_K_XY = "k_xy"

# Map: overlay key -> (strain row in ``PointMeasurements.strains``,
#                      mathtext colorbar label, matplotlib colormap name).
#
# Units:
#   * Strains (eps_xx, eps_yy, gamma_xy) are dimensionless -> "[-]".
#   * Curvatures (kappa_*) have units of inverse length; the actual length
#     unit comes from the input MAT file's nodes array, so the label uses
#     a generic "[1/L]" placeholder rather than hard-coding mm vs. m.
_FIELD_TABLE: Dict[str, Dict[str, Any]] = {
    OVERLAY_EPS_X:  {"row": 0,
                     "label": r"$\varepsilon_{xx}$  [-]",
                     "probe_plain": "ε_xx [-]",
                     "cmap": "RdBu_r"},
    OVERLAY_EPS_Y:  {"row": 1,
                     "label": r"$\varepsilon_{yy}$  [-]",
                     "probe_plain": "ε_yy [-]",
                     "cmap": "RdBu_r"},
    OVERLAY_EPS_XY: {"row": 2,
                     "label": r"$\gamma_{xy}$  [-]",
                     "probe_plain": "γ_xy [-]",
                     "cmap": "RdBu_r"},
    OVERLAY_K_X:    {"row": 3,
                     "label": r"$\kappa_{xx}$  [1/L]",
                     "probe_plain": "κ_xx [1/L]",
                     "cmap": "PuOr_r"},
    OVERLAY_K_Y:    {"row": 4,
                     "label": r"$\kappa_{yy}$  [1/L]",
                     "probe_plain": "κ_yy [1/L]",
                     "cmap": "PuOr_r"},
    OVERLAY_K_XY:   {"row": 5,
                     "label": r"$\kappa_{xy}$  [1/L]",
                     "probe_plain": "κ_xy [1/L]",
                     "cmap": "PuOr_r"},
}

_FIELD_ROWS_BY_ANALYSIS: Dict[str, Dict[str, int]] = {
    "in_plane": {
        OVERLAY_EPS_X: 0,
        OVERLAY_EPS_Y: 1,
        OVERLAY_EPS_XY: 2,
    },
    "out_of_plane": {
        OVERLAY_K_X: 0,
        OVERLAY_K_Y: 1,
        OVERLAY_K_XY: 2,
    },
    "both": {
        OVERLAY_EPS_X: 0,
        OVERLAY_EPS_Y: 1,
        OVERLAY_EPS_XY: 2,
        OVERLAY_K_X: 3,
        OVERLAY_K_Y: 4,
        OVERLAY_K_XY: 5,
    },
}

_STRAIN_ROWS_DESCRIPTION_BY_ANALYSIS: Dict[str, str] = {
    "in_plane": "(3, N) array (Eps1, Eps2, Eps6 rows)",
    "out_of_plane": "(3, N) array (K1, K2, K6 rows)",
    "both": "(6, N) array (Eps1, Eps2, Eps6, K1, K2, K6 rows)",
}

# Pixel radius (screen space) for snapping a contour probe click to the
# nearest measurement node, and for hit-testing probe markers on delete.
_PROBE_SNAP_PIXELS = 18.0
_PROBE_DELETE_PIXELS = 14.0


def _is_field_overlay(name: str) -> bool:
    """Return True if ``name`` selects a tri-contour overlay (not just dots)."""
    return name in _FIELD_TABLE


# =============================================================================
#  Triangulation tuning constants
# -----------------------------------------------------------------------------
#  These three values control how aggressively ``_filtered_triangulation``
#  prunes triangles that span empty regions in the measurement cloud (the
#  voids between dog-bone tabs and the gauge area, the L-shaped notches
#  inside a concave specimen outline, etc.).  All thresholds are scaled by
#  the median nearest-neighbour distance ``h`` of the cloud, so they work
#  on any unit system and any grid pitch without per-dataset tuning.
#
#      * ``_FIELD_EDGE_FACTOR``      -- max edge length / h.  Catches
#                                       "long" bridges across wide voids
#                                       (e.g. several grid cells).
#      * ``_FIELD_NN_RADIUS_FACTOR`` -- radius / h of the local
#                                       neighbourhood used by the
#                                       common-neighbour test.  Just
#                                       above sqrt(2) so the 8 orthogonal
#                                       + diagonal grid neighbours all
#                                       count, with a small margin for
#                                       slightly irregular grids.
#      * ``_FIELD_MIN_COMMON``       -- min number of shared
#                                       neighbours required between the
#                                       two endpoints of a triangle
#                                       edge for that edge to be
#                                       considered "interior data".
#                                       2 is the natural count for any
#                                       edge inside a regular Delaunay
#                                       quad; L-corner bridge edges only
#                                       reach 1, so the mask catches
#                                       them.
# =============================================================================
_FIELD_EDGE_FACTOR = 2.0
_FIELD_NN_RADIUS_FACTOR = 1.6
_FIELD_MIN_COMMON = 2


def _filtered_triangulation(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    edge_factor: float = _FIELD_EDGE_FACTOR,
    radius_factor: float = _FIELD_NN_RADIUS_FACTOR,
    min_common: int = _FIELD_MIN_COMMON,
) -> Triangulation:
    """Delaunay triangulation with void-spanning triangles masked out.

    ``matplotlib.pyplot.tricontourf`` builds a Delaunay triangulation
    over every input point, which by construction connects each point to
    its nearest neighbours -- *even when those neighbours are on the
    other side of a physical void* (e.g. the empty region between the
    tabs and the central gauge area of a dog-bone specimen, or across
    an L-shaped notch at the inside corner of a concave outline).  The
    resulting contour then visually "bridges" those gaps and shows
    coloured strain where the experiment has no data.

    This helper combines two filters that share a single length scale
    ``h`` (the median nearest-neighbour distance of the input cloud):

      1.  **Edge-length filter** -- any triangle whose longest edge is
          more than ``edge_factor * h`` is masked.  This catches
          "obvious" bridges across multi-cell voids.

      2.  **Common-neighbour filter** -- for each remaining triangle, we
          count, for each of its three edges, how many input points lie
          inside the ``radius_factor * h`` neighbourhood of BOTH
          endpoints.  In a regular grid every interior edge has at
          least two such common neighbours; an L-corner bridge edge has
          only one (the corner point itself) and any longer bridge has
          zero.  A triangle is masked as soon as one of its edges has
          fewer than ``min_common`` shared neighbours.

    The second filter is the one that catches the subtle "right
    triangle riding over the inside corner of a dog-bone notch" case,
    which is geometrically indistinguishable from a valid interior
    right triangle on a regular grid.

    Parameters
    ----------
    xs, ys
        1-D arrays of point coordinates (already cleaned of NaN).
    edge_factor
        Multiplier on ``h`` for the max-edge filter (default ``2.0``).
    radius_factor
        Multiplier on ``h`` for the local neighbourhood radius used by
        the common-neighbour filter.  Defaults to ``1.6`` -- comfortably
        above ``sqrt(2)`` so all 8 orthogonal + diagonal grid
        neighbours are picked up.
    min_common
        Minimum number of common neighbours per edge (default ``2``).

    Returns
    -------
    matplotlib.tri.Triangulation
        Ready to hand straight to ``Axes.tricontourf``.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    triang = Triangulation(xs, ys)
    tris = triang.triangles                          # (n_tri, 3)
    n_tri = int(tris.shape[0])
    if n_tri == 0:
        return triang

    points = np.column_stack([xs, ys])
    n_pts = int(points.shape[0])
    if n_pts < 4:
        return triang

    # Local length scale: median nearest-neighbour distance.  cKDTree is
    # the cheapest way to get this for ~10^5-point clouds; SciPy is
    # already a dependency (the MAT-file loader uses ``scipy.io``).
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    nn_distances, _ = tree.query(points, k=2)
    h = float(np.median(nn_distances[:, 1]))
    if h <= 0.0:
        return triang

    unique_x = np.unique(xs).size
    unique_y = np.unique(ys).size
    grid_axis_scale = max(4.0 * np.sqrt(n_pts), 16.0)
    is_scattered_cloud = (
        unique_x > grid_axis_scale and unique_y > grid_axis_scale
    )

    # -------- Filter 1: max-edge length ------------------------------
    p0 = tris[:, 0]
    p1 = tris[:, 1]
    p2 = tris[:, 2]
    e01 = np.hypot(xs[p1] - xs[p0], ys[p1] - ys[p0])
    e12 = np.hypot(xs[p2] - xs[p1], ys[p2] - ys[p1])
    e20 = np.hypot(xs[p0] - xs[p2], ys[p0] - ys[p2])
    max_edge = np.maximum(np.maximum(e01, e12), e20)
    edge_threshold = edge_factor * h
    edge_mask = max_edge > edge_threshold

    if is_scattered_cloud:
        # The common-neighbour test below is tuned for axis-aligned structured
        # measurement grids.  A genuinely scattered cloud can have thousands of
        # unique x/y coordinates; applying the grid-specific test there masks
        # many valid interior triangles and makes the contour look fragmented.
        if edge_mask.any():
            triang.set_mask(edge_mask)
        return triang

    # -------- Filter 2: common-neighbour count -----------------------
    # Build the per-point neighbour lists ONCE and convert to sets so
    # the per-triangle intersection below is O(k) on average.
    radius = radius_factor * h
    nbr_lists = tree.query_ball_point(points, r=radius)
    nbr_sets: List[set] = [set(nb) - {i} for i, nb in enumerate(nbr_lists)]

    corner_mask = np.zeros(n_tri, dtype=bool)
    for i in range(n_tri):
        if edge_mask[i]:
            continue                                  # already going to mask
        a = int(p0[i])
        b = int(p1[i])
        c = int(p2[i])
        na = nbr_sets[a]
        nb = nbr_sets[b]
        nc = nbr_sets[c]
        if (len(na & nb) < min_common
                or len(nb & nc) < min_common
                or len(nc & na) < min_common):
            corner_mask[i] = True

    mask = edge_mask | corner_mask
    if mask.all():
        # Filter is more aggressive than the data supports (very sparse
        # or irregular cloud).  Falling back to the un-masked
        # triangulation is better than showing nothing.
        return triang
    triang.set_mask(mask)
    return triang


# =============================================================================
#  DOF -> glyph mapping
# -----------------------------------------------------------------------------
#  Lookup table consumed by ``_refresh_bc_overlay``.  Each entry returns a
#  ``GlyphSpec`` tuple ``(kind, direction)`` where:
#      kind        : "single"        -- single-headed arrow (translational)
#                    "rot_pair"      -- two ``-|>`` arrows from the node in the
#                                       **same** direction along ``direction``,
#                                       with two different lengths (rotation
#                                       about the axis normal to the plane slice).
#                    "marker_mix"    -- small disk for W_xy (twist) DOF.
#                    "out_of_page"   -- circle-with-center-dot (W transverse)
#      direction   : unit (dx, dy) for ``single`` / ``rot_pair``; ignored
#                    for ``marker_mix`` / ``out_of_page``.
#
#  Axis pairing matches ``Forward_ABD_solution/plate_postprocess._BC_GLYPH_TABLE``
#  projected onto the x–y plane: ``w_x`` uses axis ``y``, ``w_y`` uses axis ``x``.
# =============================================================================

_DOF_GLYPH_TABLE: Dict[str, Tuple[str, Tuple[float, float]]] = {
    "U":    ("single",      (1.0, 0.0)),
    "V":    ("single",      (0.0, 1.0)),
    "W":    ("out_of_page", (0.0, 0.0)),
    "W_X":  ("rot_pair",    (0.0, 1.0)),   # forward ``w_x`` -> axis y in-plane
    "W_Y":  ("rot_pair",    (1.0, 0.0)),   # forward ``w_y`` -> axis x in-plane
    "W_XY": ("marker_mix",  (0.0, 0.0)),
}

_DOF_SHORT_LABEL: Dict[str, str] = {
    "U": "U", "V": "V", "W": "W",
    "W_X": "Wx", "W_Y": "Wy", "W_XY": "Wxy",
}


def _dof_glyph_spec(dof: str) -> Tuple[str, Tuple[float, float]]:
    """Return the (kind, direction) glyph spec for a plate DOF label."""
    key = str(dof).strip().upper()
    return _DOF_GLYPH_TABLE.get(key, ("single", (1.0, 0.0)))


def _dof_short_label(dof: str) -> str:
    """Compact label used to annotate linking markers next to each node."""
    key = str(dof).strip().upper()
    return _DOF_SHORT_LABEL.get(key, key)


class MeshCanvas(FigureCanvasQTAgg):
    """Embedded matplotlib canvas with node/element picking (plate)."""

    selection_changed = Signal()

    def __init__(self, mesh_pkg=None, parent=None, *,
                 analysis_type: str = "both") -> None:
        figure = Figure(figsize=(6.0, 5.5))
        # Manual layout: reserve a strip at the bottom of the figure for the
        # horizontal colorbar (shown only when a strain / curvature overlay
        # is active) AND the multi-row legend that already lives "south" of
        # the plot.  ``tight_layout`` would fight this every time the axes
        # rebuild, so we drive the geometry by hand instead.
        figure.subplots_adjust(
            top=0.95, bottom=0.32, left=0.10, right=0.95,
        )
        super().__init__(figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)

        self._mesh = mesh_pkg
        if mesh_pkg is not None:
            self._analysis_type = getattr(mesh_pkg, "analysis_type",
                                          analysis_type)
        else:
            self._analysis_type = str(analysis_type).strip().lower()

        # Selection state ---------------------------------------------------
        self._selected_nodes: Set[int] = set()
        self._selected_elements: Set[int] = set()
        self._mode: str = "nodes"

        # Overlay toggles ---------------------------------------------------
        self._show_node_ids: bool = False
        self._show_element_ids: bool = False

        # Boundary-condition overlay state ---------------------------------
        self._bc_document = None
        self._bc_artists: List[Any] = []
        #  Material-frame overlay: the two arrows showing which way the plies
        #  run relative to the frame the fields and mesh are expressed in.
        self._material_orientation_deg = 0.0
        self._show_material_frame = True
        self._frame_artists: List[Any] = []
        self._show_bc_decorations: bool = True

        # Data-overlay state (measurement-point cloud OR strain/curvature
        # tri-contour).  Only ONE may be active at a time -- the toolbar
        # dropdown enforces that via a QActionGroup.  ``_measurement_points``
        # / ``_measurement_strains`` cache the experimental data so the
        # canvas can switch fields without re-reading the MAT file.
        self._measurement_points: Optional[np.ndarray] = None
        self._measurement_strains: Optional[np.ndarray] = None
        self._active_overlay: str = OVERLAY_NONE
        self._measurement_artist = None    # dot scatter for OVERLAY_POINTS
        self._field_artist = None          # QuadContourSet for field overlays
        self._colorbar = None              # matplotlib.colorbar.Colorbar

        # Interactive read-outs on strain / curvature contours -------------
        # Each entry is ``(annotation, marker_line)``.  Cleared whenever
        # the data overlay changes (dropdown switch, contour field change,
        # measurement reload, mesh rebuild).
        self._contour_probe_artists: List[Tuple[Any, Any]] = []
        # (screen_x, screen_y, data_x, data_y) at button-press for probe clicks
        self._probe_press_pos: Optional[Tuple[float, float, float, float]] = (
            None)
        self._cid_probe_release: Optional[int] = None
        # When True (and a field contour is active), left-clicks place probe
        # markers instead of driving the rectangle marquee / BC pick path.
        self._probe_mode: bool = False

        # Plot handles (created in _build_plot) -----------------------------
        self._ax = figure.add_subplot(111)
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")

        # Rectangle (figure-relative coords) used to spawn the horizontal
        # colorbar axes south of the plot whenever a field overlay is
        # shown.  ``_cax`` is ``None`` while no colorbar is on screen.
        self._cax_rect = (0.22, 0.235, 0.60, 0.025)
        self._cax = None

        self._element_collection: Optional[PolyCollection] = None
        self._node_scatter = None
        self._selected_node_scatter = None
        self._manual_node_preview_scatter = None
        self._node_id_texts: list = []
        self._element_id_texts: list = []
        self._measurement_count_texts: list = []
        self._measurement_counts: Optional[np.ndarray] = None
        self._legend = None

        # Selectors ---------------------------------------------------------
        self._rect_selector: Optional[RectangleSelector] = None

        # Cached rectangle used to (re)create the horizontal colorbar axes
        # whenever a field overlay is shown.  We don't keep a long-lived
        # ``_cax`` because ``Colorbar.remove()`` detaches its axes from
        # the figure, so we just create a fresh axes on every field switch.

        self._build_plot()
        self._build_legend()
        self._install_rect_selector()
        self._cid_click = self.mpl_connect("button_press_event",
                                           self._on_button_press)

    # ------------------------------------------------------------------
    #  Qt size hint
    # ------------------------------------------------------------------
    def sizeHint(self) -> QSize:        # noqa: N802 -- Qt naming
        return QSize(720, 560)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 -- Qt naming
        return QSize(360, 280)

    # ==================================================================
    #  Plot construction
    # ==================================================================
    def _build_plot(self) -> None:
        if self._mesh is None:
            self._build_empty_plot()
            return

        coords = np.asarray(self._mesh.node_coords)        # (n_nodes, 2)
        elem = np.asarray(self._mesh.element_nodes)         # (n_elem, 4)

        polygons = coords[elem]                             # (n_elem, 4, 2)
        edge_color = (0.35, 0.35, 0.35, 1.0)
        face_default = (0.85, 0.88, 0.92, ELEMENT_FACE_DEFAULT_ALPHA)
        self._element_collection = PolyCollection(
            polygons,
            facecolors=[face_default] * len(polygons),
            edgecolors=edge_color,
            linewidths=0.8,
            zorder=1,
        )
        self._ax.add_collection(self._element_collection)

        self._node_scatter = self._ax.scatter(
            coords[:, 0], coords[:, 1],
            s=NODE_BASE_SIZE,
            facecolors="#1f77b4",
            edgecolors="#0b3b66",
            linewidths=0.6,
            zorder=3,
        )

        # Highlight scatter -- must sit above ALL bc-overlay artists.
        self._selected_node_scatter = self._ax.scatter(
            [], [],
            s=NODE_SELECTED_SIZE,
            facecolors=COLOR_SELECTION,
            edgecolors="#003a4a",
            linewidths=1.0,
            zorder=8,
        )

        # Selection count text -- bottom-left corner, hidden when empty.
        self._selection_count_text = self._ax.text(
            0.01, 0.01,
            "",
            transform=self._ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#003a4a",
            zorder=11,
            visible=False,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#003a4a",
                alpha=0.75,
                linewidth=0.8,
            ),
        )

        self._apply_view_limits()

        self.draw_idle()

    def _build_empty_plot(self) -> None:
        """Axes with no virtual mesh yet (measurement overlays only)."""
        self._element_collection = None
        self._node_scatter = None
        self._selected_node_scatter = self._ax.scatter(
            [], [],
            s=NODE_SELECTED_SIZE,
            facecolors=COLOR_SELECTION,
            edgecolors="#003a4a",
            linewidths=1.0,
            zorder=8,
        )
        self._selection_count_text = self._ax.text(
            0.01, 0.01,
            "",
            transform=self._ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#003a4a",
            zorder=11,
            visible=False,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#003a4a",
                alpha=0.75,
                linewidth=0.8,
            ),
        )
        self._apply_view_limits()
        self.draw_idle()

    def _apply_view_limits(self) -> None:
        """Set the axis limits to fit the mesh, plus any active data overlay.

        When no data overlay is active, the view is sized to the mesh
        bounding box plus a 5% padding.  When the dot cloud or a strain /
        curvature contour is shown, the view is expanded to the union of
        the mesh box and the measurement-points box so the user can
        actually see the data -- the original "Show measurement points
        does nothing" bug was caused by the axes being clamped to the
        mesh and clipping any cloud that extended past it.
        """
        boxes = []
        if self._mesh is not None:
            coords = np.asarray(self._mesh.node_coords)
            boxes.append(coords)
        if (self._active_overlay != OVERLAY_NONE
                and self._measurement_points is not None
                and self._measurement_points.size > 0):
            boxes.append(np.asarray(self._measurement_points))
        if self._manual_node_preview_scatter is not None:
            offsets = self._manual_node_preview_scatter.get_offsets()
            if offsets is not None and len(offsets) > 0:
                boxes.append(np.asarray(offsets, dtype=float))

        if boxes:
            all_pts = np.vstack(boxes)
            x_min = float(all_pts[:, 0].min())
            x_max = float(all_pts[:, 0].max())
            y_min = float(all_pts[:, 1].min())
            y_max = float(all_pts[:, 1].max())
        else:
            x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0

        x_pad = 0.05 * max(x_max - x_min, 1.0e-6)
        y_pad = 0.05 * max(y_max - y_min, 1.0e-6)
        self._ax.set_xlim(x_min - x_pad, x_max + x_pad)
        self._ax.set_ylim(y_min - y_pad, y_max + y_pad)

    def _install_rect_selector(self) -> None:
        try:
            self._rect_selector = RectangleSelector(
                self._ax,
                onselect=self._on_rectangle_select,
                useblit=True,
                button=[1],
                minspanx=1.0,
                minspany=1.0,
                spancoords="pixels",
                interactive=False,
                props=dict(facecolor="#ffd166", edgecolor="#b08000",
                           alpha=0.25, fill=True),
            )
        except TypeError:                    # pragma: no cover
            self._rect_selector = RectangleSelector(
                self._ax,
                self._on_rectangle_select,
                useblit=True,
                button=[1],
                minspanx=1.0,
                minspany=1.0,
                spancoords="pixels",
                interactive=False,
            )
        self._rect_selector.set_active(self._rect_selector_should_be_active())

    def _rect_selector_should_be_active(self) -> bool:
        """Marquee selection is off in navigate mode and while contour probing."""
        if self._mode == "navigate":
            return False
        if self._probe_mode and _is_field_overlay(self._active_overlay):
            return False
        return True

    def _sync_rect_selector_active(self) -> None:
        if self._rect_selector is not None:
            self._rect_selector.set_active(self._rect_selector_should_be_active())

    # ==================================================================
    #  Public API
    # ==================================================================
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        key = str(mode).strip().lower()
        if key not in ("navigate", "nodes", "elements"):
            raise ValueError(f"Unknown selection mode: {mode!r}")
        if key == self._mode:
            return
        self._mode = key
        self._sync_rect_selector_active()
        self.draw_idle()

    def set_show_node_ids(self, show: bool) -> None:
        self._show_node_ids = bool(show)
        self._refresh_node_id_overlay()

    def set_show_element_ids(self, show: bool) -> None:
        self._show_element_ids = bool(show)
        self._refresh_element_id_overlay()

    def set_show_bc_decorations(self, show: bool) -> None:
        """Toggle the orange/red/purple BC overlay layer."""
        self._show_bc_decorations = bool(show)
        self._refresh_bc_overlay()

    def set_bc_document(self, document) -> None:
        """Tell the canvas which ``BcDocument`` to visualise."""
        self._bc_document = document
        self._refresh_bc_overlay()

    def set_material_frame_angle(self, angle_deg: float) -> None:
        """Set the material 1-axis direction, in degrees CCW from mesh +x."""
        self._material_orientation_deg = float(angle_deg)
        self._refresh_material_frame_overlay()

    def set_show_material_frame(self, show: bool) -> None:
        """Toggle the material-frame arrow layer."""
        self._show_material_frame = bool(show)
        self._refresh_material_frame_overlay()

    def set_manual_node_preview(self, points) -> None:
        """Show temporary manual-node dots while the dialog is open."""
        self.clear_manual_node_preview()
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            return
        pts = pts.reshape((-1, 2))
        self._manual_node_preview_scatter = self._ax.scatter(
            pts[:, 0], pts[:, 1],
            s=46.0,
            facecolors="#ffb703",
            edgecolors="#7a4f00",
            linewidths=0.9,
            zorder=9,
        )
        self._apply_view_limits()
        self.draw_idle()

    def clear_manual_node_preview(self) -> None:
        if self._manual_node_preview_scatter is not None:
            try:
                self._manual_node_preview_scatter.remove()
            except Exception:                    # pragma: no cover -- defensive
                pass
            self._manual_node_preview_scatter = None
            self._apply_view_limits()
            self.draw_idle()

    def set_mesh(self, mesh_pkg) -> None:
        """Replace the visualised mesh and rebuild every artist.

        Used by the live virtual-mesh editor: the user changes nx / ny /
        bounds in the right-dock panel, the parent rebuilds a fresh
        ``MeshPackage`` via :func:`vfm_mesh.build_mesh`, and this method
        swaps it in.

        All selection state is dropped because node and element IDs are
        not stable under a mesh rebuild.  Overlay toggles (node IDs,
        element IDs, BC decorations, measurement-point cloud) are
        preserved so the user does not lose UI state.
        """
        self._mesh = mesh_pkg
        if mesh_pkg is not None:
            self._analysis_type = getattr(mesh_pkg, "analysis_type",
                                          self._analysis_type)

        # Drop contour read-outs immediately -- node IDs changed and the
        # annotations belong to the previous mesh session anyway.
        self._clear_contour_probes()

        # Clear ALL existing artists so the matplotlib axes start clean.
        for art_list in (self._node_id_texts, self._element_id_texts,
                         self._measurement_count_texts,
                         self._bc_artists,
                         self._frame_artists):
            for art in art_list:
                try:
                    art.remove()
                except Exception:                # pragma: no cover -- defensive
                    pass
        self._node_id_texts = []
        self._element_id_texts = []
        self._measurement_count_texts = []
        self._measurement_counts = None
        self._bc_artists = []
        self._frame_artists = []

        for art in (self._element_collection,
                    self._node_scatter,
                    self._selected_node_scatter,
                    self._measurement_artist,
                    self._manual_node_preview_scatter):
            if art is None:
                continue
            try:
                art.remove()
            except Exception:                    # pragma: no cover -- defensive
                pass
        self._element_collection = None
        self._node_scatter = None
        self._selected_node_scatter = None
        self._measurement_artist = None
        self._manual_node_preview_scatter = None

        # The field tri-contour and its colorbar reference the matplotlib
        # axes we are about to rebuild; tear them down before
        # ``_build_plot`` so the new artists start from a clean slate.
        self._remove_field_artist()
        self._remove_colorbar()

        # The selection-count text is anchored in axes-fraction space and
        # can survive the rebuild; remove it so ``_build_plot`` can
        # recreate it without leaving a stale duplicate behind.
        if self._selection_count_text is not None:
            try:
                self._selection_count_text.remove()
            except Exception:                    # pragma: no cover -- defensive
                pass
            self._selection_count_text = None

        # Selection state is no longer meaningful (the old node / element
        # IDs do not match the new mesh).  Clear it before emitting the
        # signal so listeners see consistent state.
        had_selection = (len(self._selected_nodes) > 0
                         or len(self._selected_elements) > 0)
        self._selected_nodes = set()
        self._selected_elements = set()

        self._build_plot()
        self._install_rect_selector()

        # Reapply all the overlay toggles the user had set.
        if self._show_node_ids:
            self._refresh_node_id_overlay()
        if self._show_element_ids:
            self._refresh_element_id_overlay()
        # BC overlay always refreshes; it handles ``_show_bc_decorations``
        # and ``_bc_document is None`` internally.
        self._refresh_bc_overlay()
        self._refresh_material_frame_overlay()
        # Re-draw whichever data overlay (dot cloud / strain contour / ...)
        # was active before the rebuild.  The helper internally resizes
        # the view to fit the new mesh AND any visible measurement data.
        self._refresh_active_overlay()

        # Legend depends on which artists exist; rebuild last so it ends
        # up on top of the freshly-drawn mesh.
        self._build_legend()

        if had_selection:
            self.selection_changed.emit()

        self.draw_idle()

    def refresh_bc_overlay(self) -> None:
        """Public alias kept for callers that already hold the document."""
        self._refresh_bc_overlay()

    def set_measurement_points(self, points) -> None:
        """Provide the experimental point cloud to overlay on the mesh.

        The points must be (N, 2).  Calling this with ``None`` clears the
        cached cloud (and drops the currently-active data overlay, since
        every overlay variant -- dot scatter and tri-contours alike --
        depends on the point coordinates).
        """
        if points is None:
            self._measurement_points = None
        else:
            arr = np.asarray(points, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(
                    "Measurement points must be an (N, 2) array; "
                    f"got shape {arr.shape}."
                )
            self._measurement_points = arr
        self._refresh_active_overlay()
        #  With no mesh the cloud is what the axes are anchored on, so the
        #  overlay only becomes drawable once the points arrive.
        if self._mesh is None:
            self._refresh_material_frame_overlay()

    def set_measurement_strains(self, strains) -> None:
        """Provide analysis-aware strain/curvature samples for contours."""
        if strains is None:
            self._measurement_strains = None
        else:
            arr = np.asarray(strains, dtype=float)
            expected_rows = 6 if self._analysis_type == "both" else 3
            row_desc = _STRAIN_ROWS_DESCRIPTION_BY_ANALYSIS.get(
                self._analysis_type,
                f"({expected_rows}, N) array",
            )
            if arr.ndim != 2 or arr.shape[0] != expected_rows:
                raise ValueError(
                    f"Measurement strains must be a {row_desc} for "
                    f"analysis_type='{self._analysis_type}'; got shape "
                    f"{arr.shape}."
                )
            if (self._measurement_points is not None
                    and arr.shape[1] != self._measurement_points.shape[0]):
                raise ValueError(
                    "Measurement strains column count must match the number "
                    f"of points; got {arr.shape[1]} columns vs. "
                    f"{self._measurement_points.shape[0]} points."
                )
            self._measurement_strains = arr
        # Re-render so a field overlay that was waiting on data shows up.
        self._refresh_active_overlay()

    def _field_row_for_active_overlay(self) -> Optional[int]:
        """Return the strain row for the active overlay in this analysis mode."""
        rows = _FIELD_ROWS_BY_ANALYSIS.get(self._analysis_type, {})
        row = rows.get(self._active_overlay)
        if row is None:
            return None
        return int(row)

    def set_element_measurement_counts(self, counts: Optional[np.ndarray]) -> None:
        """Show per-element counts of mapped measurement points, or hide them.

        ``counts`` must have length ``mesh_pkg.n_elements`` when provided.
        Passing ``None`` removes the numeric overlay from the canvas.
        """
        if counts is None:
            self._measurement_counts = None
        else:
            if self._mesh is None:
                self._measurement_counts = None
                self._refresh_measurement_count_overlay()
                return
            arr = np.asarray(counts, dtype=int).ravel()
            n_el = int(self._mesh.n_elements)
            if arr.size != n_el:
                raise ValueError(
                    "Measurement count vector must have length "
                    f"{n_el} (one entry per element); got {arr.size}."
                )
            self._measurement_counts = arr
        self._refresh_measurement_count_overlay()

    def set_active_overlay(self, name: str) -> None:
        """Pick which data overlay (if any) is drawn on top of the mesh.

        Valid identifiers are exposed at module scope:
        ``OVERLAY_NONE``, ``OVERLAY_POINTS``, ``OVERLAY_EPS_X``,
        ``OVERLAY_EPS_Y``, ``OVERLAY_EPS_XY``, ``OVERLAY_K_X``,
        ``OVERLAY_K_Y``, ``OVERLAY_K_XY``.  Unknown values are coerced
        to ``OVERLAY_NONE`` rather than raising, so a stale identifier
        in the main window cannot crash the canvas.
        """
        key = "" if name is None else str(name)
        if key != OVERLAY_NONE and key != OVERLAY_POINTS and not _is_field_overlay(key):
            key = OVERLAY_NONE
        if key == self._active_overlay:
            return
        self._active_overlay = key
        if not _is_field_overlay(key):
            self._probe_mode = False
        self._refresh_active_overlay()
        self._sync_rect_selector_active()

    def active_overlay(self) -> str:
        return self._active_overlay

    def set_probe_mode(self, active: bool) -> None:
        """Enable/disable left-click read-outs on the active field contour.

        When ``active`` is True but the current overlay is not a strain /
        curvature field, the request is ignored (probe stays off).
        """
        want = bool(active) and _is_field_overlay(self._active_overlay)
        if self._probe_mode == want:
            return
        self._probe_mode = want
        self._sync_rect_selector_active()
        self.draw_idle()

    def probe_mode(self) -> bool:
        return self._probe_mode

    def set_show_measurement_points(self, show: bool) -> None:
        """Backwards-compatible wrapper around ``set_active_overlay``."""
        self.set_active_overlay(OVERLAY_POINTS if show else OVERLAY_NONE)

    def has_measurement_points(self) -> bool:
        return self._measurement_points is not None

    def has_measurement_strains(self) -> bool:
        return self._measurement_strains is not None

    def selected_nodes(self) -> list[int]:
        return sorted(self._selected_nodes)

    def selected_elements(self) -> list[int]:
        return sorted(self._selected_elements)

    def set_selected_nodes(self, node_ids) -> None:
        if self._mesh is None:
            self._selected_nodes.clear()
            self._refresh_selection_visuals()
            self.selection_changed.emit()
            return
        self._selected_nodes = {int(n) for n in node_ids
                                if 0 <= int(n) < int(self._mesh.n_nodes)}
        self._refresh_selection_visuals()
        self.selection_changed.emit()

    def set_selected_elements(self, element_ids) -> None:
        if self._mesh is None:
            self._selected_elements.clear()
            self._refresh_selection_visuals()
            self.selection_changed.emit()
            return
        self._selected_elements = {int(e) for e in element_ids
                                   if 0 <= int(e) < int(self._mesh.n_elements)}
        self._refresh_selection_visuals()
        self.selection_changed.emit()

    def clear_selection(self) -> None:
        changed = False
        if self._selected_nodes or self._selected_elements:
            self._selected_nodes.clear()
            self._selected_elements.clear()
            self._refresh_selection_visuals()
            self.selection_changed.emit()
            changed = True
        if self._contour_probe_artists:
            self._clear_contour_probes()
            changed = True
        if changed:
            self.draw_idle()

    def expand_elements_to_nodes(self) -> None:
        """Add the corner nodes of every selected element to the node set."""
        if self._mesh is None or not self._selected_elements:
            return
        elem = np.asarray(self._mesh.element_nodes)
        added = elem[list(self._selected_elements)].ravel().tolist()
        self._selected_nodes.update(int(n) for n in added)
        self._refresh_selection_visuals()
        self.selection_changed.emit()

    # ==================================================================
    #  Selection event handlers
    # ==================================================================
    def _on_rectangle_select(self, eclick, erelease) -> None:
        if self._mesh is None:
            return
        if self._mode == "navigate":
            return

        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        if None in (x0, x1, y0, y1):
            return

        modifiers = self._modifiers_from_event(eclick)
        toggle = bool(modifiers & (Qt.ShiftModifier | Qt.ControlModifier))

        if self._mode == "nodes":
            picked = self._nodes_in_box(x0, x1, y0, y1)
            self._apply_node_pick(picked, toggle=toggle)
        elif self._mode == "elements":
            picked = self._elements_in_box(x0, x1, y0, y1)
            self._apply_element_pick(picked, toggle=toggle)

    def _on_button_press(self, event) -> None:
        if event.inaxes is not self._ax:
            return

        # Right-click on data axes: delete nearest contour probe marker when a
        # strain / curvature field is displayed (works in every toolbar mode).
        if event.button == 3:
            if (_is_field_overlay(self._active_overlay)
                    and event.xdata is not None
                    and event.ydata is not None):
                self._remove_nearest_contour_probe(event)
                self.draw_idle()
            return

        if event.button != 1:
            return

        # Left-click contour probe: only while ``Probe`` toolbar mode is on
        # (``set_probe_mode(True)``) and a field contour is displayed.
        if (_is_field_overlay(self._active_overlay) and self._probe_mode):
            if event.xdata is None or event.ydata is None:
                return
            old_cid = getattr(self, "_cid_probe_release", None)
            if old_cid is not None:
                try:
                    self.mpl_disconnect(old_cid)
                except Exception:        # pragma: no cover -- defensive
                    pass
            self._probe_press_pos = (
                float(event.x), float(event.y),
                float(event.xdata), float(event.ydata),
            )
            self._cid_probe_release = self.mpl_connect(
                "button_release_event", self._on_probe_button_release)
            return

        if self._mode == "navigate":
            return
        if self._mesh is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        self._press_pos = (event.x, event.y, event.xdata, event.ydata,
                           self._modifiers_from_event(event))

        self._cid_release = self.mpl_connect(
            "button_release_event", self._on_button_release
        )

    def _on_button_release(self, event) -> None:
        cid = getattr(self, "_cid_release", None)
        if cid is not None:
            self.mpl_disconnect(cid)
            self._cid_release = None

        press = getattr(self, "_press_pos", None)
        if press is None:
            return
        self._press_pos = None

        if event.button != 1 or event.inaxes is not self._ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        px, py, _, _, modifiers = press
        dx = event.x - px
        dy = event.y - py
        click_threshold_pixels = 4.0
        if (dx * dx + dy * dy) > click_threshold_pixels ** 2:
            return  # treated as a drag by the RectangleSelector

        toggle = bool(modifiers & (Qt.ShiftModifier | Qt.ControlModifier))

        if self._mode == "nodes":
            node_id = self._nearest_node(event.xdata, event.ydata)
            if node_id is None:
                return
            self._apply_node_pick([node_id], toggle=toggle)
        elif self._mode == "elements":
            elem_id = self._element_under_point(event.xdata, event.ydata)
            if elem_id is None:
                return
            self._apply_element_pick([elem_id], toggle=toggle)

    @staticmethod
    def _modifiers_from_event(event):
        """Return the Qt keyboard modifier flags attached to a matplotlib event."""
        gui_event = getattr(event, "guiEvent", None)
        if gui_event is None:
            return Qt.NoModifier
        try:
            return gui_event.modifiers()
        except Exception:                       # pragma: no cover -- defensive
            return Qt.NoModifier

    # ==================================================================
    #  Hit testing
    # ==================================================================
    def _nodes_in_box(self, x0, x1, y0, y1) -> list[int]:
        if self._mesh is None:
            return []
        coords = np.asarray(self._mesh.node_coords)
        mask = ((coords[:, 0] >= x0) & (coords[:, 0] <= x1)
                & (coords[:, 1] >= y0) & (coords[:, 1] <= y1))
        return np.where(mask)[0].tolist()

    def _elements_in_box(self, x0, x1, y0, y1) -> list[int]:
        if self._mesh is None:
            return []
        coords = np.asarray(self._mesh.node_coords)
        elem = np.asarray(self._mesh.element_nodes)
        polygons = coords[elem]                       # (n_elem, 4, 2)
        inside = ((polygons[..., 0] >= x0) & (polygons[..., 0] <= x1)
                  & (polygons[..., 1] >= y0) & (polygons[..., 1] <= y1))
        mask = inside.all(axis=1)
        return np.where(mask)[0].tolist()

    def _nearest_node(self, x: float, y: float) -> Optional[int]:
        if self._mesh is None:
            return None
        coords = np.asarray(self._mesh.node_coords)
        if coords.size == 0:
            return None
        trans = self._ax.transData
        pix = trans.transform(coords)
        target = trans.transform(np.array([[x, y]]))
        d2 = ((pix - target) ** 2).sum(axis=1)
        idx = int(np.argmin(d2))
        if d2[idx] > 12.0 ** 2:
            return None
        return idx

    def _element_under_point(self, x: float, y: float) -> Optional[int]:
        if self._mesh is None:
            return None
        coords = np.asarray(self._mesh.node_coords)
        elem = np.asarray(self._mesh.element_nodes)
        for element_id, corners in enumerate(elem):
            xs = coords[corners, 0]
            ys = coords[corners, 1]
            if (xs.min() <= x <= xs.max()
                    and ys.min() <= y <= ys.max()):
                return int(element_id)
        return None

    # ==================================================================
    #  Selection bookkeeping
    # ==================================================================
    def _apply_node_pick(self, picked, toggle: bool) -> None:
        picked_set = {int(n) for n in picked}
        if toggle:
            self._selected_nodes ^= picked_set
        else:
            self._selected_nodes = picked_set
        self._refresh_selection_visuals()
        self.selection_changed.emit()

    def _apply_element_pick(self, picked, toggle: bool) -> None:
        picked_set = {int(e) for e in picked}
        if toggle:
            self._selected_elements ^= picked_set
        else:
            self._selected_elements = picked_set
        self._refresh_selection_visuals()
        self.selection_changed.emit()

    def _refresh_selection_visuals(self) -> None:
        if self._mesh is None:
            if self._selected_node_scatter is not None:
                self._selected_node_scatter.set_offsets(np.empty((0, 2)))
            if self._selection_count_text is not None:
                self._selection_count_text.set_visible(False)
            self.draw_idle()
            return

        coords = np.asarray(self._mesh.node_coords)

        if self._selected_nodes:
            selected_ids = sorted(self._selected_nodes)
            self._selected_node_scatter.set_offsets(coords[selected_ids])
        else:
            self._selected_node_scatter.set_offsets(np.empty((0, 2)))

        if self._element_collection is not None:
            n_elem = int(self._mesh.n_elements)
            colors = np.zeros((n_elem, 4), dtype=float)
            colors[:, :3] = np.array([0.85, 0.88, 0.92])
            colors[:, 3] = ELEMENT_FACE_DEFAULT_ALPHA
            if self._selected_elements:
                sel = np.fromiter(self._selected_elements, dtype=int)
                colors[sel, :3] = np.array([0.0, 0.706, 0.847])
                colors[sel, 3] = ELEMENT_FACE_SELECTED_ALPHA
            self._element_collection.set_facecolors(colors)

        n_nodes = len(self._selected_nodes)
        if n_nodes == 0:
            self._selection_count_text.set_visible(False)
        else:
            label = "1 node selected" if n_nodes == 1 else f"{n_nodes} nodes selected"
            self._selection_count_text.set_text(label)
            self._selection_count_text.set_visible(True)

        self.draw_idle()

    # ==================================================================
    #  Overlays
    # ==================================================================
    def _refresh_node_id_overlay(self) -> None:
        for text in self._node_id_texts:
            text.remove()
        self._node_id_texts = []
        if self._show_node_ids:
            if self._mesh is None:
                self.draw_idle()
                return
            coords = np.asarray(self._mesh.node_coords)
            for node_id, (x, y) in enumerate(coords):
                text = self._ax.annotate(
                    str(node_id),
                    xy=(x, y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color="#0b3b66",
                    zorder=5,
                )
                self._node_id_texts.append(text)
        self.draw_idle()

    def _refresh_element_id_overlay(self) -> None:
        for text in self._element_id_texts:
            text.remove()
        self._element_id_texts = []
        if self._show_element_ids:
            if self._mesh is None:
                self.draw_idle()
                return
            coords = np.asarray(self._mesh.node_coords)
            elem = np.asarray(self._mesh.element_nodes)
            for element_id, corners in enumerate(elem):
                cx = float(coords[corners, 0].mean())
                cy = float(coords[corners, 1].mean())
                text = self._ax.annotate(
                    str(element_id),
                    xy=(cx, cy),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#7a3d05",
                    zorder=5,
                )
                self._element_id_texts.append(text)
        self.draw_idle()

    def _refresh_measurement_count_overlay(self) -> None:
        for text in self._measurement_count_texts:
            try:
                text.remove()
            except Exception:                # pragma: no cover -- defensive
                pass
        self._measurement_count_texts = []
        if self._measurement_counts is None:
            self.draw_idle()
            return
        if self._mesh is None:
            self.draw_idle()
            return
        coords = np.asarray(self._mesh.node_coords)
        elem = np.asarray(self._mesh.element_nodes)
        for element_id, corners in enumerate(elem):
            cx = float(coords[corners, 0].mean())
            cy = float(coords[corners, 1].mean())
            text = self._ax.annotate(
                str(int(self._measurement_counts[element_id])),
                xy=(cx, cy),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#0d5c2e",
                zorder=6,
            )
            self._measurement_count_texts.append(text)
        self.draw_idle()

    # ==================================================================
    #  Boundary-condition overlay
    # ------------------------------------------------------------------
    #  Renders three kinds of decoration on top of the mesh:
    #      * Prescribed (Dirichlet) -- orange glyph keyed to the DOF.
    #      * External work / load   -- red    glyph keyed to the DOF.
    #      * Linking                -- purple node marker + small DOF
    #                                   label, since the tie is between
    #                                   many DOFs and does not have a
    #                                   single direction.
    #
    #  DOF-specific glyphs:
    #      * Translational (U, V) -- single-headed FancyArrowPatch.
    #      * Transverse   (W)     -- Circle + center dot (out-of-page
    #                                 standard symbol).
    #      * Rotational (W_X, W_Y) -- two same-sense single-headed arrows from
    #                                 the node along the axis (+y for Wx, +x for Wy).
    #      * Twist (W_XY)          -- small disk (``marker_mix``), same colours
    #                                 as the rotation glyphs.
    #
    #  Edge-based entries are expanded to the corresponding node set using
    #  the same tolerance logic as ``plate/vfm_mesh._select_edge_nodes`` so
    #  the GUI agrees with the solver.
    # ==================================================================
    def _build_legend(self) -> None:
        """(Re)build the static legend at the bottom-center of the axes."""
        if self._legend is not None:
            try:
                self._legend.remove()
            except Exception:                # pragma: no cover -- defensive
                pass
            self._legend = None

        legend_handles = [
            Line2D(
                [0], [0],
                marker=r"$\rightarrow$",
                color=COLOR_PRESCRIBED,
                linestyle="None",
                markersize=14,
                markeredgewidth=2,
                label="Prescribed force / disp (U, V)",
            ),
            Line2D(
                [0], [0],
                marker=r"$\rightarrow$",
                color=COLOR_EXTERNAL,
                linestyle="None",
                markersize=14,
                markeredgewidth=2,
                label="External force / disp (U, V)",
            ),
            Line2D(
                [0], [0],
                marker=r"$\odot$",
                color=COLOR_PRESCRIBED,
                linestyle="None",
                markersize=14,
                markeredgewidth=2,
                label="Prescribed transverse (W)",
            ),
            Line2D(
                [0], [0],
                marker=r"$\odot$",
                color=COLOR_EXTERNAL,
                linestyle="None",
                markersize=14,
                markeredgewidth=2,
                label="External transverse (W)",
            ),
        ]
        _leg_pres_rot_arrow = Line2D(
            [0], [0],
            marker=r"$\rightarrow$",
            color=BC_ROT_PRESCRIBED,
            linestyle="None",
            markersize=15,
            markeredgewidth=1.8,
            markeredgecolor=BC_ROT_PRESCRIBED,
        )
        _leg_pres_rot_twist = Line2D(
            [0], [0],
            marker="o",
            color=BC_ROT_PRESCRIBED,
            linestyle="None",
            markersize=8,
            markeredgecolor="#5a3080",
            markeredgewidth=0.85,
        )
        _leg_ext_rot_arrow = Line2D(
            [0], [0],
            marker=r"$\rightarrow$",
            color=BC_ROT_EXTERNAL,
            linestyle="None",
            markersize=15,
            markeredgewidth=1.8,
            markeredgecolor=BC_ROT_EXTERNAL,
        )
        _leg_ext_rot_twist = Line2D(
            [0], [0],
            marker="o",
            color=BC_ROT_EXTERNAL,
            linestyle="None",
            markersize=8,
            markeredgecolor="#063c1f",
            markeredgewidth=0.85,
        )
        legend_handles.append(
            (_leg_pres_rot_arrow, _leg_pres_rot_twist)
        )
        legend_handles.append(
            (_leg_ext_rot_arrow, _leg_ext_rot_twist)
        )
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color=COLOR_LINKING,
                linestyle="None",
                markersize=9,
                markeredgecolor="#3a004a",
                label="Linking (tied nodes)",
            ),
        )

        legend_labels = [h.get_label() for h in legend_handles[:4]]
        legend_labels.extend([
            "Prescribed rotation (Wx, Wy, Wxy)",
            "External rotation (Wx, Wy, Wxy)",
        ])
        legend_labels.append(legend_handles[-1].get_label())
        if (self._active_overlay == OVERLAY_POINTS
                and self._measurement_points is not None):
            _meas = Line2D(
                [0], [0],
                marker="o",
                color=COLOR_MEASUREMENTS,
                linestyle="None",
                markersize=7,
                markeredgecolor="#888888",
                alpha=0.6,
                label="Measurement points",
            )
            legend_handles.append(_meas)
            legend_labels.append(_meas.get_label())

        if self._show_material_frame and self._frame_artists:
            _axis_1 = Line2D([0], [0], color=COLOR_MATERIAL_AXIS_1,
                             linewidth=2.0)
            _axis_2 = Line2D([0], [0], color=COLOR_MATERIAL_AXIS_2,
                             linewidth=2.0)
            legend_handles.append((_axis_1, _axis_2))
            legend_labels.append(
                f"Material frame 1/2 ({self._material_orientation_deg:+.1f} deg)")

        # Wrap the legend into two rows so the 7+ entries do not overflow
        # the bottom of the axes when the window is narrow.  The legend
        # itself is anchored OUTSIDE the plotting region (south of the
        # axes, in figure coordinates) together with the colorbar -- see
        # ``__init__`` for the matching ``subplots_adjust`` margins.
        n_cols = 4 if len(legend_handles) >= 4 else len(legend_handles)
        self._legend = self._ax.legend(
            legend_handles,
            legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            bbox_transform=self.figure.transFigure,
            ncol=n_cols,
            fontsize=8,
            framealpha=0.9,
            handletextpad=0.4,
            columnspacing=1.0,
        )
        if self._legend is not None:
            self._legend.set_zorder(10)

    # ------------------------------------------------------------------
    #  Data-overlay dispatch
    # ------------------------------------------------------------------
    def _refresh_active_overlay(self) -> None:
        """Synchronise dot scatter + field contour with ``_active_overlay``.

        Centralises the bookkeeping so that switching dropdown options on
        the main window always ends up in a consistent state: at most one
        of (dots, tri-contour) is on screen, the colorbar follows the
        contour, the axes view fits the visible data, and the legend is
        rebuilt last to reflect whichever overlay is now live.

        Contour probe markers (left-click read-outs) are cleared on every
        refresh so stale annotations never survive a contour / view change.
        """
        self._clear_contour_probes()
        self._refresh_dot_overlay()
        self._refresh_field_overlay()
        self._apply_view_limits()
        self._build_legend()
        self.draw_idle()

    def _clear_contour_probes(self) -> None:
        """Remove every strain/curvature read-out marker and drop the
        pending left-button release hook used to pair probe clicks."""
        cid = getattr(self, "_cid_probe_release", None)
        if cid is not None:
            try:
                self.mpl_disconnect(cid)
            except Exception:                # pragma: no cover -- defensive
                pass
            self._cid_probe_release = None
        self._probe_press_pos = None

        for pair in self._contour_probe_artists:
            for art in pair:
                try:
                    art.remove()
                except Exception:            # pragma: no cover -- defensive
                    pass
        self._contour_probe_artists.clear()

    def _nearest_measurement_probe_index(self,
                                         x: float, y: float) -> Optional[int]:
        """Index into ``measurement_points`` for the finite sample nearest
        the click (screen-space snap radius)."""
        if (not _is_field_overlay(self._active_overlay)
                or self._measurement_points is None
                or self._measurement_strains is None):
            return None
        row = self._field_row_for_active_overlay()
        if row is None or row >= self._measurement_strains.shape[0]:
            return None
        xs = np.asarray(self._measurement_points[:, 0], dtype=float)
        ys = np.asarray(self._measurement_points[:, 1], dtype=float)
        values = np.asarray(self._measurement_strains[row], dtype=float)
        if xs.size != values.size:
            return None
        finite = np.isfinite(values) & np.isfinite(xs) & np.isfinite(ys)
        if not finite.any():
            return None
        coords = np.column_stack([xs[finite], ys[finite]])
        trans = self._ax.transData
        pix = trans.transform(coords)
        target = trans.transform(np.array([[x, y]]))
        d2 = ((pix - target) ** 2).sum(axis=1)
        i_loc = int(np.argmin(d2))
        if d2[i_loc] > _PROBE_SNAP_PIXELS ** 2:
            return None
        return int(np.flatnonzero(finite)[i_loc])

    def _add_contour_probe(self, x: float, y: float) -> None:
        """Place a read-out annotation at the nearest measurement node."""
        idx = self._nearest_measurement_probe_index(x, y)
        if idx is None:
            return
        meta = _FIELD_TABLE[self._active_overlay]
        row = self._field_row_for_active_overlay()
        if row is None:
            return
        xn = float(self._measurement_points[idx, 0])
        yn = float(self._measurement_points[idx, 1])
        val = float(self._measurement_strains[row, idx])
        plain = str(meta.get("probe_plain", "value"))

        text = self._ax.annotate(
            f"#{idx}\n({xn:g}, {yn:g})\n{plain}: {val:.6g}",
            xy=(xn, yn),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=8,
            color="#1a1a1a",
            zorder=12,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="#fffacd",
                edgecolor="#333333",
                alpha=0.92,
            ),
        )
        (marker_line,) = self._ax.plot(
            [xn], [yn],
            marker="s",
            markersize=5.5,
            markerfacecolor=COLOR_PROBE_MARKER,
            markeredgecolor="#1a1a1a",
            markeredgewidth=0.8,
            linestyle="None",
            zorder=12,
        )
        self._contour_probe_artists.append((text, marker_line))

    def _remove_nearest_contour_probe(self, event) -> None:
        """Delete the probe marker whose anchor is closest to the click."""
        if (event.xdata is None or event.ydata is None
                or not self._contour_probe_artists):
            return
        trans = self._ax.transData
        target = trans.transform(np.array([[event.xdata, event.ydata]]))
        best_i: Optional[int] = None
        best_d2 = float("inf")
        for i, (_ann, mline) in enumerate(self._contour_probe_artists):
            xy = mline.get_xydata()
            if xy.size == 0:
                continue
            xn, yn = float(xy[0, 0]), float(xy[0, 1])
            pix = trans.transform(np.array([[xn, yn]]))
            d2 = float(((pix - target) ** 2).sum())
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        thr = _PROBE_DELETE_PIXELS ** 2
        if best_i is None or best_d2 > thr:
            return
        pair = self._contour_probe_artists.pop(best_i)
        for art in pair:
            try:
                art.remove()
            except Exception:                # pragma: no cover -- defensive
                pass

    def _on_probe_button_release(self, event) -> None:
        """Finish a contour probe click (paired with ``_on_button_press``)."""
        cid = getattr(self, "_cid_probe_release", None)
        if cid is not None:
            self.mpl_disconnect(cid)
            self._cid_probe_release = None

        press = getattr(self, "_probe_press_pos", None)
        self._probe_press_pos = None
        if press is None:
            return

        if event.button != 1:
            return

        px, py, x_press, y_press = press
        dx = event.x - px
        dy = event.y - py
        click_threshold_pixels = 4.0
        if (dx * dx + dy * dy) > click_threshold_pixels ** 2:
            return

        # Prefer release coordinates on the data axes; if the cursor left
        # the axes (common with toolbar pan), fall back to the press data.
        if (event.inaxes is self._ax
                and event.xdata is not None
                and event.ydata is not None):
            self._add_contour_probe(float(event.xdata), float(event.ydata))
        else:
            self._add_contour_probe(float(x_press), float(y_press))
        self.draw_idle()

    def _refresh_dot_overlay(self) -> None:
        """Show the experimental point cloud iff ``OVERLAY_POINTS`` is active."""
        if self._measurement_artist is not None:
            try:
                self._measurement_artist.remove()
            except Exception:                # pragma: no cover -- defensive
                pass
            self._measurement_artist = None

        if (self._active_overlay == OVERLAY_POINTS
                and self._measurement_points is not None):
            xs = self._measurement_points[:, 0]
            ys = self._measurement_points[:, 1]
            # zorder=2 keeps the dots above the element wireframe (zorder=1)
            # but under the node scatter (zorder=3), so the cloud is always
            # legible without occluding the picking glyphs.
            (artist,) = self._ax.plot(
                xs, ys,
                marker=".",
                linestyle="None",
                color=COLOR_MEASUREMENTS,
                markersize=4,
                markeredgecolor="none",
                alpha=0.7,
                zorder=2,
                rasterized=True,
            )
            self._measurement_artist = artist

    def _refresh_field_overlay(self) -> None:
        """Show the filled tri-contour for the active strain / curvature.

        Does nothing -- and tears down any previous field artist + the
        colorbar -- when the active overlay is not a field, or when the
        cached measurement data is incompatible (missing strains, fewer
        than three points, all-NaN, ...).
        """
        self._remove_field_artist()

        if (not _is_field_overlay(self._active_overlay)
                or self._measurement_points is None
                or self._measurement_strains is None):
            self._remove_colorbar()
            return

        meta = _FIELD_TABLE[self._active_overlay]
        row = self._field_row_for_active_overlay()
        if row is None or row >= self._measurement_strains.shape[0]:
            self._remove_colorbar()
            return

        xs = np.asarray(self._measurement_points[:, 0], dtype=float)
        ys = np.asarray(self._measurement_points[:, 1], dtype=float)
        values = np.asarray(self._measurement_strains[row], dtype=float)

        if xs.size != values.size:
            # Defensive: shape was validated in the setter, but a later
            # ``set_mesh`` could not in theory reorder things; skip the
            # draw rather than raise from inside an event callback.
            self._remove_colorbar()
            return

        finite = np.isfinite(values) & np.isfinite(xs) & np.isfinite(ys)
        if finite.sum() < 3:
            self._remove_colorbar()
            return

        xs_f = xs[finite]
        ys_f = ys[finite]
        v_f = values[finite]

        vmin = float(v_f.min())
        vmax = float(v_f.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            self._remove_colorbar()
            return
        if vmax - vmin <= 0.0:
            # Constant field: nudge the range so tricontourf produces a
            # single band and the colorbar still renders sensible ticks.
            eps = max(abs(vmax), 1.0) * 1.0e-9
            vmin -= eps
            vmax += eps

        # Build a Delaunay triangulation in which triangles spanning an
        # empty region of the cloud (e.g. the gap between the dog-bone
        # tabs and the central sample) are hidden.  Without this filter,
        # tricontourf would happily connect every point to its closest
        # neighbour across that void and falsely paint coloured strain
        # in an area where the experiment actually has no data.
        try:
            triang = _filtered_triangulation(xs_f, ys_f)
            contour = self._ax.tricontourf(
                triang, v_f,
                levels=20,
                cmap=meta["cmap"],
                vmin=vmin, vmax=vmax,
                extend="both",
                zorder=0,
            )
        except Exception as exc:                # pragma: no cover -- defensive
            print(f"[bc_gui] could not draw '{self._active_overlay}' "
                  f"contour: {exc}")
            self._remove_colorbar()
            return

        self._field_artist = contour
        self._show_colorbar(contour, str(meta["label"]))

    def _remove_field_artist(self) -> None:
        """Detach any previous tri-contour from the axes."""
        if self._field_artist is None:
            return
        try:
            # Matplotlib >= 3.7 supports ``QuadContourSet.remove()``.
            self._field_artist.remove()
        except (AttributeError, NotImplementedError):       # pragma: no cover
            try:
                for coll in self._field_artist.collections:
                    coll.remove()
            except Exception:
                pass
        self._field_artist = None

    def _show_colorbar(self, mappable, label: str) -> None:
        """Spawn a fresh horizontal colorbar axes south of the plot.

        ``Colorbar.remove()`` detaches its host axes from the figure, so
        we create a brand-new axes every time rather than trying to
        recycle one across field switches.  The position is locked to
        ``_cax_rect`` (figure coords) so the layout stays stable.
        """
        self._remove_colorbar()
        self._cax = self.figure.add_axes(self._cax_rect)
        self._colorbar = self.figure.colorbar(
            mappable,
            cax=self._cax,
            orientation="horizontal",
        )
        self._colorbar.set_label(label, fontsize=9)
        self._colorbar.ax.tick_params(labelsize=8)

    def _remove_colorbar(self) -> None:
        """Detach the field colorbar (and its host axes) from the figure."""
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:                    # pragma: no cover -- defensive
                pass
            self._colorbar = None
        if self._cax is not None:
            try:
                self._cax.remove()
            except Exception:                    # pragma: no cover -- defensive
                pass
            self._cax = None

    def _material_frame_bounds(self):
        """(x_min, x_max, y_min, y_max) to anchor the axes on, or ``None``.

        Prefers the virtual mesh; falls back to the measurement point cloud so
        the overlay is present before a mesh has been applied.
        """
        if self._mesh is not None:
            return (float(getattr(self._mesh, "x_min", 0.0)),
                    float(getattr(self._mesh, "x_max", 1.0)),
                    float(getattr(self._mesh, "y_min", 0.0)),
                    float(getattr(self._mesh, "y_max", 1.0)))

        points = self._measurement_points
        if points is None:
            return None
        points = np.asarray(points, dtype=float).reshape((-1, 2))
        if points.size == 0:
            return None
        return (float(points[:, 0].min()), float(points[:, 0].max()),
                float(points[:, 1].min()), float(points[:, 1].max()))

    def _refresh_material_frame_overlay(self) -> None:
        """Draw the material 1- and 2-axes over the middle of the plate.

        This is the frame ``theta`` is expressed in.  The mesh, the measured
        fields and the boundary conditions are all in the CANVAS frame; the
        angle drawn here is what the constitutive matrix gets rotated by, so
        the arrows are the direct visual answer to "which way do the plies run
        relative to what I am looking at".
        """
        for art in self._frame_artists:
            try:
                art.remove()
            except Exception:                    # pragma: no cover -- defensive
                pass
        self._frame_artists = []

        if not self._show_material_frame:
            self.draw_idle()
            return

        #  Anchor on the mesh when there is one, otherwise on the measurement
        #  cloud.  The BC editor is launched from the experiment dialog with
        #  NO mesh -- the user picks nodes against the point cloud first -- so
        #  a mesh-only overlay would be invisible during exactly the step this
        #  is meant to inform.
        bounds = self._material_frame_bounds()
        if bounds is None:
            self.draw_idle()
            return
        x_min, x_max, y_min, y_max = bounds
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)

        #  Same sizing rule the BC glyphs use, so the two layers read as one
        #  drawing rather than competing for scale.
        length = max(0.9 * min(x_max - x_min, y_max - y_min) * 0.5, 1.0e-9)

        phi = np.deg2rad(self._material_orientation_deg)
        axes = (
            (np.cos(phi), np.sin(phi), COLOR_MATERIAL_AXIS_1, "1"),
            (-np.sin(phi), np.cos(phi), COLOR_MATERIAL_AXIS_2, "2"),
        )
        for ux, uy, color, name in axes:
            head = (cx + length * ux, cy + length * uy)
            arrow = FancyArrowPatch(
                (cx, cy), head,
                arrowstyle="-|>", mutation_scale=16,
                color=color, linewidth=2.0, zorder=7,
                alpha=0.9,
            )
            self._ax.add_patch(arrow)
            self._frame_artists.append(arrow)

            label = self._ax.annotate(
                name,
                xy=head, xytext=(6 * ux, 6 * uy),
                textcoords="offset points",
                color=color, fontsize=10, fontweight="bold",
                ha="center", va="center", zorder=8,
            )
            self._frame_artists.append(label)

        marker = self._ax.scatter([cx], [cy], s=26.0,
                                  facecolors="none",
                                  edgecolors=COLOR_MATERIAL_AXIS_1,
                                  linewidths=1.2, zorder=7)
        self._frame_artists.append(marker)

        self.draw_idle()

    def _refresh_bc_overlay(self) -> None:
        # Remove the previous decorations.
        for art in self._bc_artists:
            try:
                art.remove()
            except Exception:                # pragma: no cover -- defensive
                pass
        self._bc_artists = []

        if (self._bc_document is None) or (not self._show_bc_decorations):
            self.draw_idle()
            return

        if self._mesh is None:
            self.draw_idle()
            return

        coords = np.asarray(self._mesh.node_coords)

        dx = float(getattr(self._mesh, "dx", 1.0))
        dy = float(getattr(self._mesh, "dy", 1.0))
        arrow_length = max(0.4 * min(dx, dy), 1.0e-9)

        # ---- Prescribed: orange glyphs ---------------------------------
        for entry in self._bc_document.entries("prescribed"):
            kind, direction = _dof_glyph_spec(entry.dof)
            for node_id in self._entry_node_ids(entry):
                x, y = coords[node_id]
                self._draw_dof_glyph(
                    kind, direction, x, y,
                    arrow_length=arrow_length,
                    color=_bc_glyph_color("prescribed", entry.dof),
                    category="prescribed",
                )

        # ---- External: red glyphs --------------------------------------
        for entry in self._bc_document.entries("external"):
            kind, direction = _dof_glyph_spec(entry.dof)
            for node_id in self._entry_node_ids(entry):
                x, y = coords[node_id]
                self._draw_dof_glyph(
                    kind, direction, x, y,
                    arrow_length=arrow_length,
                    color=_bc_glyph_color("external", entry.dof),
                    category="external",
                )

        # ---- Linking: purple node markers + small DOF labels -----------
        # Group by DOF first so scatter is called once per DOF (cheaper for
        # very large linking groups).
        linking_by_dof: Dict[str, List[Tuple[int, float, float]]] = {}
        for entry in self._bc_document.entries("linking"):
            bucket = linking_by_dof.setdefault(entry.dof, [])
            for node_id in self._entry_node_ids(entry):
                bucket.append((int(node_id),
                               float(coords[node_id, 0]),
                               float(coords[node_id, 1])))

        for dof, items in linking_by_dof.items():
            if not items:
                continue
            xs = [item[1] for item in items]
            ys = [item[2] for item in items]
            scatter = self._ax.scatter(
                xs, ys,
                s=LINKING_NODE_SIZE,
                facecolors=COLOR_LINKING,
                edgecolors="#3a004a",
                linewidths=1.0,
                zorder=5,
            )
            self._bc_artists.append(scatter)
            short = _dof_short_label(dof)
            for _node_id, x, y in items:
                annot = self._ax.annotate(
                    short,
                    xy=(x, y),
                    xytext=(4, -10),
                    textcoords="offset points",
                    fontsize=7,
                    color="#3a004a",
                    zorder=6,
                )
                self._bc_artists.append(annot)

        self.draw_idle()

    def _draw_dof_glyph(self,
                        kind: str,
                        direction: Tuple[float, float],
                        x: float,
                        y: float,
                        *,
                        arrow_length: float,
                        color: str,
                        category: str) -> None:
        """Draw one boundary-condition glyph at the node location.

        Parameters
        ----------
        kind
            ``"single"``, ``"rot_pair"``, ``"marker_mix"``, or
            ``"out_of_page"`` -- see ``_DOF_GLYPH_TABLE``.
        direction
            (dx, dy) unit direction along which the arrow is drawn
            (ignored for the out-of-page marker).
        x, y
            Node coordinates.
        arrow_length
            Total visual length of the arrow (or outer diameter for the
            out-of-page marker).
        color
            Glyph colour (depends on category and DOF; see ``_bc_glyph_color``).
        category
            ``"prescribed"`` or ``"external"`` -- selects the
            ``tail->head`` (prescribed) vs ``head->tail`` (external)
            orientation for single-headed arrows.  Ignored otherwise.
        """
        if kind == "single":
            if category == "prescribed":
                tail = (x - arrow_length * direction[0],
                        y - arrow_length * direction[1])
                head = (x, y)
            else:                          # external
                tail = (x, y)
                head = (x + arrow_length * direction[0],
                        y + arrow_length * direction[1])
            arrow = FancyArrowPatch(
                tail, head,
                arrowstyle="-|>",
                mutation_scale=14,
                color=color,
                linewidth=1.8 if category == "external" else 1.6,
                zorder=6,
            )
            self._ax.add_patch(arrow)
            self._bc_artists.append(arrow)
            return

        if kind == "rot_pair":
            # Two single-headed arrows from the node in the **same** direction
            # along (dx, dy), with different lengths — encodes a rotational DOF
            # without bidirectional / opposing arrowheads.
            dx = float(direction[0])
            dy = float(direction[1])
            norm = float(np.hypot(dx, dy))
            if norm < 1.0e-12:
                return
            dx /= norm
            dy /= norm
            L = float(arrow_length)
            # Longer arrow drawn first (under), shorter on top so both heads show.
            frac_short = 0.62
            frac_long = 0.82
            lw = 1.75 if category == "external" else 1.55
            ms = 11.0 if category == "external" else 10.0
            for z, frac in ((6, frac_long), (7, frac_short)):
                tail = (x, y)
                head = (x + frac * L * dx, y + frac * L * dy)
                arrow = FancyArrowPatch(
                    tail, head,
                    arrowstyle="-|>",
                    mutation_scale=ms,
                    color=color,
                    linewidth=lw,
                    zorder=z,
                )
                self._ax.add_patch(arrow)
                self._bc_artists.append(arrow)
            return

        if kind == "marker_mix":
            # Same disk idiom as ``plate_postprocess`` for ``w_xy``; edge tone
            # matches prescribed vs external rotation hue.
            ec = "#5a3080" if category == "prescribed" else "#063c1f"
            sc = self._ax.scatter(
                [x], [y],
                s=52.0,
                c=[color],
                edgecolors=ec,
                linewidths=0.65,
                zorder=6,
            )
            self._bc_artists.append(sc)
            return

        if kind == "out_of_page":
            # Outer circle (no fill) + inner dot at the node centre.
            outer_radius = 0.20 * arrow_length
            inner_radius = 0.055 * arrow_length
            outer = Circle(
                (x, y),
                radius=outer_radius,
                fill=False,
                edgecolor=color,
                linewidth=1.8 if category == "external" else 1.6,
                zorder=6,
            )
            inner = Circle(
                (x, y),
                radius=inner_radius,
                fill=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0.0,
                zorder=6,
            )
            self._ax.add_patch(outer)
            self._ax.add_patch(inner)
            self._bc_artists.extend([outer, inner])
            return

        # Unknown kind -- silently skip rather than raise so a hand-edited
        # BC file with an unsupported DOF does not crash the GUI.

    def _entry_node_ids(self, entry) -> list[int]:
        """Resolve the entry's selection (``edge`` or ``nodes``) into ids."""
        edge = getattr(entry, "edge", None)
        mode = getattr(entry, "selection_mode", "nodes")
        if mode == "edge" and edge is not None:
            return self._edge_node_indices(edge)
        node_ids = list(getattr(entry, "nodes", None) or [])
        if self._mesh is None:
            return []
        n_nodes = int(getattr(self._mesh, "n_nodes", 0))
        # During multi-test switching the canvas briefly has the new mesh
        # before the matching BC document is installed.  Ignore stale node
        # ids from the previous document so redraws cannot crash.
        return [
            int(node_id)
            for node_id in node_ids
            if 0 <= int(node_id) < n_nodes
        ]

    def _edge_node_indices(self, edge_name: str) -> list[int]:
        """Same tolerance logic as ``plate/vfm_mesh._select_edge_nodes``."""
        if self._mesh is None:
            return []
        coords = np.asarray(self._mesh.node_coords)
        tolerance = max(
            1.0e-12,
            1.0e-10 * max(
                abs(float(self._mesh.x_min)), abs(float(self._mesh.x_max)),
                abs(float(self._mesh.y_min)), abs(float(self._mesh.y_max)),
                1.0,
            ),
        )
        edge = str(edge_name).strip().lower()
        if edge == "left":
            mask = np.abs(coords[:, 0] - float(self._mesh.x_min)) <= tolerance
        elif edge == "right":
            mask = np.abs(coords[:, 0] - float(self._mesh.x_max)) <= tolerance
        elif edge == "bottom":
            mask = np.abs(coords[:, 1] - float(self._mesh.y_min)) <= tolerance
        elif edge == "top":
            mask = np.abs(coords[:, 1] - float(self._mesh.y_max)) <= tolerance
        else:
            return []
        return np.where(mask)[0].tolist()


# =============================================================================
#  Convenience: matplotlib's default Qt navigation toolbar.
# =============================================================================

MatplotlibToolbar = NavigationToolbar2QT

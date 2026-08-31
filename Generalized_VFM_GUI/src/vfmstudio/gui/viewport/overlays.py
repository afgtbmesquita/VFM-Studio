"""Mesh, BC, material-axis, and CID overlays for the PyVista viewport."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from vfmstudio.core import mesh_convert
from vfmstudio.core.fields import conjugate_moment_plain, field_math_title
from vfmstudio.core.document import ExperimentModel
from vfmstudio.gui.session import Selection


BC_COLORS = {
    "prescribed": "#E8A444",
    "linking": "#C0392B",
    "external": "#1baf7a",
}
HIGHLIGHT = "#F5C518"
PRESCRIBED_CONE_SCALE = 0.5
_ISO_SHEAR_X = 0.45
_ISO_SHEAR_Y = 0.20
_ARC_THETA_EPS = 0.12
_ARC_SAMPLES = 36

_EDGE_INWARD = {
    "left": np.array([1.0, 0.0, 0.0]),
    "right": np.array([-1.0, 0.0, 0.0]),
    "bottom": np.array([0.0, 1.0, 0.0]),
    "top": np.array([0.0, -1.0, 0.0]),
}


def unstructured_quad_grid(pv, mesh_pkg: Any, z: Optional[np.ndarray] = None):
    xyz = mesh_convert.mesh_xyz(mesh_pkg)
    if z is not None:
        xyz = xyz.copy()
        xyz[:, 2] = np.asarray(z, dtype=float).ravel()[: xyz.shape[0]]
    quads = mesh_convert.mesh_quads(mesh_pkg)
    cells, celltypes = mesh_convert.vtk_quad_cells(quads)
    celltypes = np.full(quads.shape[0], pv.CellType.QUAD, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes, xyz)


MESH_EDGE_COLOR = "#1A1A1A"
MESH_EDGE_WIDTH = 3.5


def add_mesh_wireframe(plotter, pv, mesh_pkg: Any, name: str = "mesh") -> None:
    grid = unstructured_quad_grid(pv, mesh_pkg)
    kwargs = dict(
        style="wireframe",
        color=MESH_EDGE_COLOR,
        line_width=MESH_EDGE_WIDTH,
        lighting=False,
        name=name,
    )
    try:
        plotter.add_mesh(grid, ambient=1.0, **kwargs)
    except TypeError:
        plotter.add_mesh(grid, **kwargs)


def remove_named(plotter, *names: str) -> None:
    for name in names:
        try:
            plotter.remove_actor(name)
        except Exception:
            pass
    try:
        plotter.remove_scalar_bar()
    except Exception:
        pass


def set_named_visible(plotter, name: str, visible: bool) -> None:
    try:
        actors = getattr(plotter, "actors", None) or {}
        actor = actors.get(name)
        if actor is not None:
            actor.SetVisibility(1 if visible else 0)
    except Exception:
        pass


def _add_styled_scalar_bar(plotter, title: str) -> None:
    """Vertical colorbar in the top-left, thin strip with padded labels."""
    try:
        bar = plotter.add_scalar_bar(
            title=title,
            n_labels=5,
            fmt="%.2e",
            vertical=True,
            color="black",
            title_font_size=14,
            label_font_size=9,
            width=0.08,
            height=0.36,
            position_x=0.03,
            position_y=0.58,
            interactive=False,
            fill=False,
            render=False,
            unconstrained_font_size=False,
        )
    except Exception:
        return
    if bar is None:
        return
    try:
        bar.SetTextPad(10)
        bar.SetVerticalTitleSeparation(16)
        bar.SetBarRatio(0.18)
    except Exception:
        pass
    for prop_name in ("GetTitleTextProperty", "GetLabelTextProperty"):
        getter = getattr(bar, prop_name, None)
        if getter is None:
            continue
        try:
            prop = getter()
            prop.SetLineOffset(3)
        except Exception:
            pass


def add_field_contour(
    plotter,
    pv,
    pts: np.ndarray,
    values: np.ndarray,
    scalar_name: str,
    *,
    deformed: bool = False,
) -> None:
    xyz = np.asarray(pts, dtype=float)
    scalars = np.asarray(values, dtype=float).ravel()
    n = min(xyz.shape[0], scalars.size)
    xyz = xyz[:n]
    scalars = scalars[:n]
    cloud = pv.PolyData(xyz)
    cloud[scalar_name] = scalars
    try:
        surf = cloud.delaunay_2d()
        if scalar_name not in surf.array_names:
            surf[scalar_name] = cloud[scalar_name]
    except Exception:
        surf = cloud
    finite = scalars[np.isfinite(scalars)]
    clim = None
    if finite.size:
        lo, hi = np.percentile(finite, [2.0, 98.0])
        limit = float(max(abs(lo), abs(hi), 1e-12))
        clim = [-limit, limit]
    kwargs = dict(
        scalars=scalar_name,
        cmap="RdBu_r",
        show_edges=False,
        lighting=bool(deformed),
        smooth_shading=bool(deformed),
        name="contour",
        show_scalar_bar=False,
    )
    if deformed:
        kwargs.update(ambient=0.40, diffuse=0.60, specular=0.18, specular_power=12)
    if clim is not None:
        kwargs["clim"] = clim
    try:
        plotter.add_mesh(surf, **kwargs)
    except TypeError:
        kwargs.pop("lighting", None)
        kwargs.pop("smooth_shading", None)
        kwargs.pop("ambient", None)
        kwargs.pop("diffuse", None)
        kwargs.pop("specular", None)
        kwargs.pop("specular_power", None)
        plotter.add_mesh(surf, **kwargs)
    _add_styled_scalar_bar(plotter, field_math_title(scalar_name))
    return surf


def set_mesh_points(mesh, pts: np.ndarray) -> bool:
    """Replace point coordinates on an existing PolyData / grid (no retessellation)."""
    if mesh is None:
        return False
    xyz = np.asarray(pts, dtype=float)
    try:
        n = int(mesh.n_points)
    except Exception:
        return False
    if xyz.ndim != 2 or xyz.shape[0] < n or xyz.shape[1] < 3:
        return False
    try:
        mesh.points = xyz[:n, :3]
    except Exception:
        return False
    return True


def add_viewport_caption(plotter, text: str) -> None:
    """Plain corner annotation — text only, no billboard or frame."""
    if plotter is None:
        return
    try:
        plotter.remove_actor("view_caption")
    except Exception:
        pass
    body = str(text or "").strip()
    if not body:
        return
    kwargs = dict(
        position="lower_right",
        font_size=9,
        color="#1a1a1a",
        name="view_caption",
        shadow=False,
    )
    actor = None
    try:
        actor = plotter.add_text(body, font="arial", **kwargs)
    except TypeError:
        try:
            actor = plotter.add_text(body, **kwargs)
        except TypeError:
            kwargs.pop("shadow", None)
            kwargs.pop("name", None)
            try:
                actor = plotter.add_text(body, **kwargs)
            except Exception:
                return
    _clear_text_backdrop(actor)


def _clear_text_backdrop(actor) -> None:
    """Strip VTK text frames/fills so the caption is letters only."""
    if actor is None:
        return
    props = []
    getter = getattr(actor, "GetTextProperty", None)
    if callable(getter):
        try:
            props.append(getter())
        except Exception:
            pass
    inner = getattr(actor, "GetTextActor", None)
    if callable(inner):
        try:
            text_actor = inner()
            get_prop = getattr(text_actor, "GetTextProperty", None)
            if callable(get_prop):
                props.append(get_prop())
        except Exception:
            pass
    for prop in props:
        if prop is None:
            continue
        for method in ("FrameOff", "ShadowOff"):
            fn = getattr(prop, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        for method, value in (
            ("SetBackgroundOpacity", 0.0),
            ("SetFrame", 0),
        ):
            fn = getattr(prop, method, None)
            if callable(fn):
                try:
                    fn(value)
                except Exception:
                    pass


def add_dic_points(
    plotter,
    pv,
    pts: np.ndarray,
    color: str = "#B8B8B8",
    values: Optional[np.ndarray] = None,
    scalar_name: str = "field",
):
    cloud = pv.PolyData(np.asarray(pts, dtype=float))
    kwargs = dict(
        point_size=8,
        render_points_as_spheres=True,
        lighting=False,
        name="dic",
    )
    if values is not None:
        scalars = np.asarray(values, dtype=float).ravel()[: cloud.n_points]
        cloud[scalar_name] = scalars
        finite = scalars[np.isfinite(scalars)]
        kwargs.update(scalars=scalar_name, cmap="coolwarm", show_scalar_bar=True)
        if finite.size:
            lo, hi = np.percentile(finite, [2.0, 98.0])
            if hi > lo:
                kwargs["clim"] = [float(lo), float(hi)]
    else:
        kwargs.update(color=color, show_scalar_bar=False)
    try:
        plotter.add_mesh(cloud, **kwargs)
    except TypeError:
        kwargs.pop("lighting", None)
        plotter.add_mesh(cloud, **kwargs)
    return cloud


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.size < 3:
        vector = np.array([vector[0] if vector.size > 0 else 0.0,
                           vector[1] if vector.size > 1 else 0.0, 0.0])
    else:
        vector = vector[:3]
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        return np.array([0.0, 0.0, 1.0])
    return vector / norm


def _merge(pv, parts):
    geom = None
    for part in parts:
        if part is None:
            continue
        geom = part if geom is None else geom.merge(part)
    return geom


# Kirchhoff DOF → Abaqus axis.  W is U3 (z); W_X / W_Y are rotations.
_DOF_AXIS = {
    "W": np.array([0.0, 0.0, 1.0]),
    "U": np.array([1.0, 0.0, 0.0]),
    "V": np.array([0.0, 1.0, 0.0]),
    "W_X": np.array([1.0, 0.0, 0.0]),
    "W_Y": np.array([0.0, 1.0, 0.0]),
    "W_XY": np.array([0.0, 0.0, 1.0]),
}
_ANGULAR_DOFS = {"W_X", "W_Y", "W_XY"}


def dof_direction(dof: str) -> np.ndarray:
    key = str(dof or "W").strip().upper().replace(" ", "_")
    return _DOF_AXIS.get(key, np.array([0.0, 0.0, 1.0])).copy()


def is_angular_dof(dof: str) -> bool:
    key = str(dof or "").strip().upper().replace(" ", "_")
    return key in _ANGULAR_DOFS


def _dof_key(dof: str) -> str:
    return str(dof or "").strip().upper().replace(" ", "_")


def moment_plane_normal(dof: str, edge: Optional[str] = None) -> np.ndarray:
    """In-plane axis ``n`` spanning the Kirchhoff moment-arc plane with +Z.

    The arc lives in the plane of the rotation the traction produces, not in
    a DOF-named plane.  ``W_X`` is ``∂w/∂x`` (rotation about Y → Z–X): that
    is ``M_xx`` on left/right and the twisting ``M_xy`` on top/bottom.
    ``W_Y`` is the Z–Y arc: ``M_yy`` on top/bottom, ``M_xy`` on left/right.
    ``W_XY`` follows the edge tangent (top/bottom → X, left/right → Y).
    """
    key = _dof_key(dof)
    side = str(edge or "").strip().lower()
    if key == "W_X":
        return np.array([1.0, 0.0, 0.0])
    if key == "W_Y":
        return np.array([0.0, 1.0, 0.0])
    if key == "W_XY":
        if side in ("left", "right"):
            return np.array([0.0, 1.0, 0.0])
        return np.array([1.0, 0.0, 0.0])
    axis = dof_direction(key)
    planar = np.array([axis[0], axis[1], 0.0])
    if float(np.linalg.norm(planar)) <= 1.0e-15:
        return np.array([1.0, 0.0, 0.0])
    return _unit(planar)


def _as_xyz(point, z: float = 0.0) -> np.ndarray:
    point = np.asarray(point, dtype=float).reshape(-1)
    if point.size >= 3:
        return point[:3].copy()
    return np.array(
        [
            point[0] if point.size > 0 else 0.0,
            point[1] if point.size > 1 else 0.0,
            z,
        ]
    )


def moment_label_anchor(
    origin,
    plane_axis,
    radius: float,
    specimen_center=None,
) -> np.ndarray:
    """Place the EVW label above the arc crest, out of the moment plane.

    The previous offset was ``origin + 0.85 R * inward``. On a top-edge
    ``M_yy`` glyph the inward is −Y, which lies in the Z–Y arc plane, so the
    green text sat on the green tube. From a camera looking along Y the same
    inward offset also projects a top-edge ``M_xy`` label onto the Z–X arc.
    """
    origin = _as_xyz(origin)
    planar = _as_xyz(plane_axis)
    planar[2] = 0.0
    n = (
        _unit(planar)
        if float(np.linalg.norm(planar)) > 1.0e-15
        else np.array([1.0, 0.0, 0.0])
    )
    z_hat = np.array([0.0, 0.0, 1.0])
    sideways = np.cross(z_hat, n)
    if float(np.linalg.norm(sideways)) <= 1.0e-15:
        sideways = np.array([1.0, 0.0, 0.0])
    else:
        sideways = _unit(sideways)
    if specimen_center is not None:
        to_center = _as_xyz(specimen_center) - origin
        to_center[2] = 0.0
        if float(np.dot(sideways, to_center)) < 0.0:
            sideways = -sideways
    radius = float(radius)
    return origin + 1.28 * radius * z_hat + 0.82 * radius * sideways


def _median_edge_z(surface_xyz, edge: str, mesh_pkg: Any) -> float:
    """Median deformed Z of DIC points that sit on a named mesh edge."""
    pts = np.asarray(surface_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 3:
        return 0.0
    xyz = mesh_convert.mesh_xyz(mesh_pkg)
    x_min, x_max = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    y_min, y_max = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    span_x = max(x_max - x_min, 1.0e-9)
    span_y = max(y_max - y_min, 1.0e-9)
    band = 0.08
    key = str(edge or "").strip().lower()
    if key == "top":
        mask = pts[:, 1] >= y_max - band * span_y
    elif key == "bottom":
        mask = pts[:, 1] <= y_min + band * span_y
    elif key == "left":
        mask = pts[:, 0] <= x_min + band * span_x
    elif key == "right":
        mask = pts[:, 0] >= x_max - band * span_x
    else:
        mask = np.ones(pts.shape[0], dtype=bool)
    z = pts[mask, 2] if mask.any() else pts[:, 2]
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite))


def moment_arc_points(
    origin,
    plane_axis,
    radius: float,
    n_samples: int = _ARC_SAMPLES,
) -> np.ndarray:
    """Semicircle in the ``n``–Z plane, bulge on +Z, arrowhead at +n.

    ``p(θ) = origin + R (cos θ n + sin θ e_z)`` with θ running from
    ``π−ε`` down to ``ε``, so the tangent at the crest takes +Z toward +n
    (CLT positive ``M_αα`` / positive ``κ_αα``).
    """
    origin = np.asarray(origin, dtype=float).reshape(-1)
    if origin.size < 3:
        origin = np.array(
            [
                origin[0] if origin.size > 0 else 0.0,
                origin[1] if origin.size > 1 else 0.0,
                0.0,
            ]
        )
    else:
        origin = origin[:3].copy()
    planar = np.asarray(plane_axis, dtype=float).reshape(-1)
    if planar.size < 3:
        planar = np.array(
            [planar[0] if planar.size > 0 else 1.0,
             planar[1] if planar.size > 1 else 0.0, 0.0]
        )
    planar[2] = 0.0
    n = _unit(planar) if float(np.linalg.norm(planar)) > 1.0e-15 else np.array(
        [1.0, 0.0, 0.0]
    )
    z = np.array([0.0, 0.0, 1.0])
    theta = np.linspace(np.pi - _ARC_THETA_EPS, _ARC_THETA_EPS, int(max(n_samples, 3)))
    radius = float(radius)
    return origin[None, :] + radius * (
        np.cos(theta)[:, None] * n[None, :] + np.sin(theta)[:, None] * z[None, :]
    )


def _isometric_xy(origin_xy, offset_xyz) -> np.ndarray:
    """Top-view shear used by Z-pyramids: z contributes to both x and y."""
    origin_xy = np.asarray(origin_xy, dtype=float).reshape(-1)[:2]
    offset_xyz = np.asarray(offset_xyz, dtype=float).reshape(-1)
    if offset_xyz.size < 3:
        offset_xyz = np.array(
            [
                offset_xyz[0] if offset_xyz.size > 0 else 0.0,
                offset_xyz[1] if offset_xyz.size > 1 else 0.0,
                0.0,
            ]
        )
    return origin_xy + np.array(
        [
            offset_xyz[0] + _ISO_SHEAR_X * offset_xyz[2],
            offset_xyz[1] + _ISO_SHEAR_Y * offset_xyz[2],
        ]
    )


def display_axis(dof: str, inward: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit axis for the glyph. In-plane symbols point inward when that is defined."""
    axis = dof_direction(dof)
    if abs(axis[2]) > 0.7:
        return np.array([0.0, 0.0, 1.0])
    if inward is not None:
        inward = _unit(inward)
        if float(np.dot(axis, inward)) < -1.0e-9:
            axis = -axis
    return _unit(axis)


def _pin_cone(pv, tip: np.ndarray, direction: np.ndarray, height: float, radius: float):
    """Abaqus pin: cone axis along ``direction``, apex at the node."""
    direction = _unit(direction)
    tip = np.asarray(tip, dtype=float)
    if abs(direction[2]) > 0.7:
        # Keep ±Z cones on that side of the plate (not collapsed into XY).
        body = np.array([0.0, 0.0, 1.0 if direction[2] > 0 else -1.0])
        center = tip + 0.50 * height * body
        return pv.Cone(
            center=center,
            direction=-body,
            height=height,
            radius=radius,
            resolution=20,
        )
    center = tip - 0.50 * height * direction
    return pv.Cone(
        center=center,
        direction=direction,
        height=height,
        radius=radius,
        resolution=20,
    )


def _double_cone(pv, node: np.ndarray, axis: np.ndarray, height: float, radius: float):
    """Abaqus rotation BC: two cones along ±axis, tips at the node."""
    return _merge(
        pv,
        [
            _pin_cone(pv, node, axis, height, radius),
            _pin_cone(pv, node, -axis, height, radius),
        ],
    )


def _force_arrow(pv, origin: np.ndarray, direction: np.ndarray, length: float):
    direction = _unit(direction)
    origin = np.asarray(origin, dtype=float)
    shaft_h = 0.70 * length
    head_h = 0.30 * length
    shaft = pv.Cylinder(
        center=origin + 0.50 * shaft_h * direction,
        direction=direction,
        radius=0.055 * length,
        height=shaft_h,
        resolution=18,
    )
    head = pv.Cone(
        center=origin + (shaft_h + 0.50 * head_h) * direction,
        direction=direction,
        height=head_h,
        radius=0.16 * length,
        resolution=20,
    )
    return _merge(pv, [shaft, head])


def _moment_arc(pv, origin: np.ndarray, plane_axis: np.ndarray, radius: float):
    """Kirchhoff moment: tube along a +Z semicircle in the n–Z plane, head at +n."""
    pts = moment_arc_points(origin, plane_axis, radius)
    n_pts = int(pts.shape[0])
    if n_pts < 2:
        return None
    poly = pv.PolyData(pts)
    poly.lines = np.hstack([[n_pts], np.arange(n_pts, dtype=np.int64)]).astype(np.int64)
    try:
        shaft = poly.tube(radius=0.08 * float(radius), n_sides=16, capping=True)
    except TypeError:
        shaft = poly.tube(radius=0.08 * float(radius))
    tangent = _unit(pts[-1] - pts[-2])
    head_h = 0.28 * float(radius)
    head = pv.Cone(
        center=pts[-1] + 0.50 * head_h * tangent,
        direction=tangent,
        height=head_h,
        radius=0.17 * float(radius),
        resolution=20,
    )
    return _merge(pv, [shaft, head])


def _glyph_scale(mesh_pkg: Any) -> float:
    xyz = mesh_convert.mesh_xyz(mesh_pkg)
    span = float(max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), 1.0))
    return 0.055 * span


def _subsample(xyz: np.ndarray, limit: int = 5) -> np.ndarray:
    if xyz.shape[0] <= limit:
        return xyz
    idx = np.linspace(0, xyz.shape[0] - 1, limit).astype(int)
    return xyz[idx]


def _remove_bc_actors(plotter, experiment: ExperimentModel | None = None) -> None:
    names = []
    actors = getattr(plotter, "actors", None) or {}
    for name in list(actors):
        if str(name).startswith("bc-"):
            names.append(name)
    if not names and experiment is not None:
        for kind, entries in (
            ("prescribed", experiment.prescribed),
            ("linking", experiment.linking),
            ("external", experiment.external),
        ):
            for index in range(len(entries) + 1):
                names.append(f"bc-{kind}-{index}")
    for name in names:
        try:
            plotter.remove_actor(name)
        except Exception:
            pass


def add_bc_glyphs(
    plotter,
    pv,
    experiment: ExperimentModel,
    mesh_pkg: Any,
    selection: Optional[Selection] = None,
    surface_xyz: Optional[np.ndarray] = None,
) -> None:
    """Abaqus BC symbols oriented along each constrained / loaded DOF."""
    scale = _glyph_scale(mesh_pkg)
    selected = None
    if selection is not None and selection.kind == "bc":
        selected = (selection.bc_kind, selection.bc_index)
    mesh_xyz = mesh_convert.mesh_xyz(mesh_pkg)
    specimen_center = mesh_xyz.mean(axis=0) if mesh_xyz.size else np.zeros(3)
    for kind, entries in (
        ("prescribed", experiment.prescribed),
        ("linking", experiment.linking),
        ("external", experiment.external),
    ):
        for index, entry in enumerate(entries):
            if not entry.edge:
                continue
            xyz = mesh_convert.edge_node_xyz(mesh_pkg, entry.edge)
            if xyz.size == 0:
                continue
            if surface_xyz is not None:
                xyz = np.asarray(xyz, dtype=float).copy()
                xyz[:, 2] = _median_edge_z(surface_xyz, entry.edge, mesh_pkg)
            pts = _subsample(xyz)
            inward = _EDGE_INWARD.get(str(entry.edge).lower(), np.array([0.0, 0.0, 1.0]))
            axis = display_axis(entry.dof, inward)
            active = selected == (kind, index)
            color = HIGHLIGHT if active else BC_COLORS[kind]
            mag = scale * (1.35 if active else 1.0)
            name = f"bc-{kind}-{index}"
            if kind == "external" and (entry.load_type or "distributed") == "concentrated":
                pts = pts.mean(axis=0, keepdims=True)
            try:
                geom = None
                if kind == "prescribed":
                    cone_mag = PRESCRIBED_CONE_SCALE * mag
                    if is_angular_dof(entry.dof):
                        parts = [
                            _double_cone(pv, pt, axis, 0.85 * cone_mag, 0.26 * cone_mag)
                            for pt in pts
                        ]
                    else:
                        parts = [_pin_cone(pv, pt, axis, cone_mag, 0.30 * cone_mag) for pt in pts]
                    geom = _merge(pv, parts)
                elif kind == "linking":
                    plotter.add_mesh(
                        pv.PolyData(pts),
                        color=color,
                        point_size=16 if active else 11,
                        render_points_as_spheres=True,
                        name=name,
                        reset_camera=False,
                    )
                    continue
                else:
                    if is_angular_dof(entry.dof):
                        plane_n = moment_plane_normal(entry.dof, entry.edge)
                        radius = 0.90 * mag
                        parts = [_moment_arc(pv, pt, plane_n, radius) for pt in pts]
                    else:
                        parts = [_force_arrow(pv, pt, axis, mag * 1.8) for pt in pts]
                    geom = _merge(pv, parts)
                if geom is not None:
                    plotter.add_mesh(
                        geom,
                        color=color,
                        name=name,
                        reset_camera=False,
                        smooth_shading=True,
                    )
                if kind == "external" and is_angular_dof(entry.dof):
                    label = conjugate_moment_plain(entry.dof, entry.edge)
                    origin = np.asarray(pts, dtype=float).mean(axis=0)
                    plane_n = moment_plane_normal(entry.dof, entry.edge)
                    radius = 0.90 * mag
                    anchor = moment_label_anchor(
                        origin, plane_n, radius, specimen_center=specimen_center
                    )
                    _add_moment_point_label(
                        plotter, anchor, label, name=f"{name}-label"
                    )
            except Exception:
                pass


def _darken_hex(color: str, factor: float = 0.72) -> str:
    token = str(color).lstrip("#")
    if len(token) != 6:
        return str(color)
    rgb = np.array([int(token[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)
    rgb = np.clip(rgb * factor, 0, 255).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def plane_glyph_direction(axis: np.ndarray) -> np.ndarray:
    """XY drawing direction for a 3D glyph axis.

    In-plane DOFs stay on X/Y. A Z-axis glyph is sheared so it is visible in
    the top view without coinciding with the specimen X or Y axes.
    """
    axis = _unit(axis)
    if abs(axis[2]) > 0.7:
        sign = 1.0 if axis[2] >= 0 else -1.0
        return _unit(sign * np.array([0.42, 0.18, 0.0]))[:2]
    xy = np.array([axis[0], axis[1]], dtype=float)
    if float(np.linalg.norm(xy)) <= 1.0e-15:
        return np.array([0.42, 0.18])
    return _unit(xy)[:2]


def _edge_sample_xy(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    edge: str,
    count: int = 5,
    concentrated: bool = False,
) -> np.ndarray:
    xs = np.linspace(x_min, x_max, count)
    ys = np.linspace(y_min, y_max, count)
    key = str(edge).strip().lower()
    if key == "top":
        pts = np.column_stack([xs, np.full(count, y_max)])
    elif key == "bottom":
        pts = np.column_stack([xs, np.full(count, y_min)])
    elif key == "left":
        pts = np.column_stack([np.full(count, x_min), ys])
    elif key == "right":
        pts = np.column_stack([np.full(count, x_max), ys])
    else:
        return np.zeros((0, 2))
    if concentrated:
        return pts.mean(axis=0, keepdims=True)
    return pts


def _triangle_2d(axis, tip, direction, size: float, color: str) -> None:
    from matplotlib.patches import Polygon

    direction = np.asarray(direction, dtype=float).reshape(-1)[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-15:
        return
    direction = direction / norm
    perp = np.array([-direction[1], direction[0]])
    tip = np.asarray(tip, dtype=float).reshape(-1)[:2]
    base = tip - size * direction
    verts = np.vstack(
        [tip, base + 0.42 * size * perp, base - 0.42 * size * perp]
    )
    axis.add_patch(
        Polygon(
            verts,
            closed=True,
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.45,
            zorder=8,
            clip_on=False,
        )
    )


def _z_pyramid_2d(axis, origin, size: float, color: str, sign: float = 1.0) -> None:
    """Two-face pyramid along ±Z so a top-view plot still reads as out-of-plane."""
    from matplotlib.patches import Polygon

    origin = np.asarray(origin, dtype=float).reshape(-1)[:2]
    height = float(size)
    radius = 0.38 * height
    z = sign * height

    def _project(x: float, y: float, z_value: float) -> np.ndarray:
        return origin + np.array(
            [x + _ISO_SHEAR_X * z_value, y + _ISO_SHEAR_Y * z_value]
        )

    apex = _project(0.0, 0.0, 0.0)
    b1 = _project(radius, 0.0, z)
    b2 = _project(-0.5 * radius, 0.866 * radius, z)
    b3 = _project(-0.5 * radius, -0.866 * radius, z)
    for verts, shade in (
        (np.vstack([apex, b1, b2]), color),
        (np.vstack([apex, b1, b3]), _darken_hex(color)),
    ):
        axis.add_patch(
            Polygon(
                verts,
                closed=True,
                facecolor=shade,
                edgecolor="#333333",
                linewidth=0.4,
                zorder=8,
                clip_on=False,
            )
        )


def _arrow_2d(axis, origin, direction, length: float, color: str, head_scale: float = 1.0) -> None:
    from matplotlib.patches import FancyArrow

    direction = np.asarray(direction, dtype=float).reshape(-1)[:2]
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-15 or length <= 0:
        return
    direction = direction / norm
    origin = np.asarray(origin, dtype=float).reshape(-1)[:2]
    axis.add_patch(
        FancyArrow(
            origin[0],
            origin[1],
            length * direction[0],
            length * direction[1],
            width=0.055 * length * head_scale,
            head_width=0.26 * length * head_scale,
            head_length=0.30 * length * head_scale,
            length_includes_head=True,
            color=color,
            zorder=8,
            clip_on=False,
        )
    )


def _moment_arc_2d(axis, origin, plane_axis, radius: float, color: str) -> None:
    """Isometric projection of the Kirchhoff n–Z moment arc (same as 3D)."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    pts = moment_arc_points(origin, plane_axis, radius)
    origin_xy = np.asarray(origin, dtype=float).reshape(-1)[:2]
    projected = np.vstack(
        [_isometric_xy(origin_xy, pt - np.array([origin_xy[0], origin_xy[1], 0.0])) for pt in pts]
    )
    if projected.shape[0] < 2:
        return
    verts = projected
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    axis.add_patch(
        PathPatch(
            Path(verts, codes),
            facecolor="none",
            edgecolor=color,
            linewidth=1.6,
            capstyle="round",
            joinstyle="round",
            zorder=8,
            clip_on=False,
        )
    )
    tangent = projected[-1] - projected[-2]
    _triangle_2d(axis, projected[-1], tangent, 0.32 * float(radius), color)


def add_bc_symbols_2d(
    axis,
    experiment: ExperimentModel,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Draw Abaqus BC/load glyphs on a 2-D specimen map (after limits are locked)."""
    from matplotlib.patches import Circle

    span = float(max(x_max - x_min, y_max - y_min, 1.0))
    pin = 0.055 * span
    cone_pin = PRESCRIBED_CONE_SCALE * pin
    arrow = 0.11 * span
    for kind, entries in (
        ("prescribed", experiment.prescribed),
        ("linking", experiment.linking),
        ("external", experiment.external),
    ):
        color = BC_COLORS[kind]
        for entry in entries:
            if not entry.edge:
                continue
            concentrated = kind == "external" and (entry.load_type or "distributed") == "concentrated"
            pts = _edge_sample_xy(
                x_min, x_max, y_min, y_max, entry.edge, concentrated=concentrated
            )
            if pts.size == 0:
                continue
            inward = _EDGE_INWARD.get(str(entry.edge).lower(), np.array([0.0, 0.0, 1.0]))
            axis3 = display_axis(entry.dof, inward)
            along_z = abs(axis3[2]) > 0.7
            plane = plane_glyph_direction(axis3)
            for pt in pts:
                if kind == "linking":
                    axis.add_patch(
                        Circle(
                            pt,
                            0.28 * pin,
                            facecolor=color,
                            edgecolor="#333333",
                            linewidth=0.4,
                            zorder=8,
                            clip_on=False,
                        )
                    )
                    continue
                if kind == "prescribed":
                    if is_angular_dof(entry.dof):
                        if along_z:
                            _z_pyramid_2d(axis, pt, 0.85 * cone_pin, color, sign=1.0)
                            _z_pyramid_2d(axis, pt, 0.85 * cone_pin, color, sign=-1.0)
                        else:
                            _triangle_2d(axis, pt, plane, 0.90 * cone_pin, color)
                            _triangle_2d(axis, pt, -plane, 0.90 * cone_pin, color)
                    elif along_z:
                        _z_pyramid_2d(axis, pt, cone_pin, color, sign=1.0)
                    else:
                        _triangle_2d(axis, pt, plane, cone_pin, color)
                    continue
                if is_angular_dof(entry.dof):
                    plane_n = moment_plane_normal(entry.dof, entry.edge)
                    _moment_arc_2d(axis, pt, plane_n, 0.90 * pin, color)
                else:
                    _arrow_2d(axis, pt, plane, arrow, color)
            if kind == "external" and is_angular_dof(entry.dof):
                mid = np.asarray(pts, dtype=float).mean(axis=0)
                label = conjugate_moment_plain(entry.dof, entry.edge)
                plane_n = moment_plane_normal(entry.dof, entry.edge)
                radius = 0.90 * pin
                specimen_center = np.array(
                    [0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.0]
                )
                anchor3 = moment_label_anchor(
                    [mid[0], mid[1], 0.0],
                    plane_n,
                    radius,
                    specimen_center=specimen_center,
                )
                anchor_xy = _isometric_xy(mid, anchor3 - np.array([mid[0], mid[1], 0.0]))
                axis.text(
                    anchor_xy[0],
                    anchor_xy[1],
                    label,
                    color="#0B3D2E",
                    fontsize=9,
                    ha="center",
                    va="center",
                    zorder=9,
                    clip_on=False,
                    bbox={
                        "boxstyle": "round,pad=0.22",
                        "facecolor": "white",
                        "edgecolor": color,
                        "linewidth": 0.7,
                        "alpha": 0.94,
                    },
                )


def update_bc_glyphs(
    plotter,
    pv,
    experiment: ExperimentModel,
    mesh_pkg: Any,
    selection: Optional[Selection] = None,
    surface_xyz: Optional[np.ndarray] = None,
) -> None:
    _remove_bc_actors(plotter, experiment)
    add_bc_glyphs(
        plotter, pv, experiment, mesh_pkg, selection, surface_xyz=surface_xyz
    )


def _add_moment_point_label(plotter, anchor, label: str, *, name: str) -> None:
    """Dark text on a white billboard so the green Myy/Mxy label is readable."""
    kwargs = dict(
        text_color="#0B3D2E",
        font_size=14,
        fill_shape=True,
        shape_color="white",
        shape_opacity=0.94,
        always_visible=True,
        justification_horizontal="center",
        show_points=False,
        name=name,
        reset_camera=False,
        render=False,
    )
    try:
        plotter.add_point_labels([anchor], [label], **kwargs)
    except TypeError:
        fallback = dict(
            text_color="#0B3D2E",
            font_size=14,
            fill_shape=True,
            shape="rounded_rect",
            name=name,
        )
        try:
            plotter.add_point_labels([anchor], [label], **fallback)
        except TypeError:
            plotter.add_point_labels(
                [anchor], [label], text_color="#0B3D2E", font_size=14, name=name
            )


def add_material_axes(plotter, experiment: ExperimentModel, mesh_pkg: Any) -> None:
    xyz = mesh_convert.mesh_xyz(mesh_pkg)
    span = float(max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), 1.0))
    axes = mesh_convert.material_axes(
        mesh_pkg, experiment.material_orientation_deg, scale=0.14 * span
    )
    origin = axes["origin"]
    pad = 0.045 * span
    for vector, color, label in (
        (axes["axis_1"], "#9B1B8C", "1"),
        (axes["axis_2"], "#00796B", "2"),
    ):
        plotter.add_arrows(origin[None, :], vector[None, :], mag=1.0, color=color)
        length = float(np.linalg.norm(vector)) + 1e-12
        direction = vector / length
        label_pt = origin + vector + pad * direction
        plotter.add_point_labels(
            [label_pt],
            [label],
            text_color=color,
            font_size=11,
            fill_shape=False,
            shape=None,
            always_visible=True,
            justification_horizontal="center",
            show_points=False,
            margin=8,
            bold=True,
            reset_camera=False,
            render=False,
        )


def add_global_axes(plotter, pv, mesh_pkg: Any) -> None:
    """Fixed XYZ triad at the model origin (Abaqus-style datum)."""
    xyz = mesh_convert.mesh_xyz(mesh_pkg)
    origin = np.array([float(np.min(xyz[:, 0])), float(np.min(xyz[:, 1])), 0.0])
    length = 0.08 * float(max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), 1.0))
    for vec, color, label in (
        (np.array([length, 0.0, 0.0]), "#C0392B", "X"),
        (np.array([0.0, length, 0.0]), "#1baf7a", "Y"),
        (np.array([0.0, 0.0, length]), "#2a78d6", "Z"),
    ):
        plotter.add_arrows(origin[None, :], vec[None, :], mag=1.0, color=color)
        plotter.add_point_labels(
            [origin + vec], [label], text_color=color, font_size=14, fill_shape=False, shape=None
        )


def _nice_ticks(lo: float, hi: float, n: int = 6) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([lo, hi], dtype=float)
    raw = (hi - lo) / max(n - 1, 1)
    exponent = np.floor(np.log10(raw)) if raw > 0 else 0.0
    mag = 10.0 ** exponent
    residual = raw / mag if mag > 0 else 1.0
    if residual < 1.5:
        step = 1.0 * mag
    elif residual < 3.0:
        step = 2.0 * mag
    elif residual < 7.0:
        step = 5.0 * mag
    else:
        step = 10.0 * mag
    start = np.floor(lo / step) * step
    stop = np.ceil(hi / step) * step
    ticks = np.arange(start, stop + 0.5 * step, step)
    ticks = ticks[(ticks >= lo - 1.0e-9 * step) & (ticks <= hi + 1.0e-9 * step)]
    if ticks.size == 0 or abs(ticks[0] - lo) > 1.0e-6 * max(abs(lo), 1.0):
        ticks = np.insert(ticks, 0, lo)
    if abs(ticks[-1] - hi) > 1.0e-6 * max(abs(hi), 1.0):
        ticks = np.append(ticks, hi)
    return np.unique(np.round(ticks, 12))


def _format_tick(value: float, step: float) -> str:
    if not np.isfinite(value):
        return ""
    if step >= 1.0:
        return f"{value:.0f}"
    if step >= 0.1:
        return f"{value:.1f}"
    if step >= 0.01:
        return f"{value:.2f}"
    return f"{value:.3g}"


def specimen_xy_extents(mesh_pkg: Any, dic_xyz: Optional[np.ndarray] = None):
    xyz = mesh_convert.mesh_xyz(mesh_pkg)
    x_min = float(np.min(xyz[:, 0]))
    x_max = float(np.max(xyz[:, 0]))
    y_min = float(np.min(xyz[:, 1]))
    y_max = float(np.max(xyz[:, 1]))
    if dic_xyz is not None:
        cloud = np.asarray(dic_xyz, dtype=float)
        if cloud.ndim == 2 and cloud.shape[0] > 0:
            finite = np.isfinite(cloud[:, 0]) & np.isfinite(cloud[:, 1])
            if np.any(finite):
                x_min = min(x_min, float(np.min(cloud[finite, 0])))
                x_max = max(x_max, float(np.max(cloud[finite, 0])))
                y_min = min(y_min, float(np.min(cloud[finite, 1])))
                y_max = max(y_max, float(np.max(cloud[finite, 1])))
    return x_min, x_max, y_min, y_max


_XY_AXES_NAMES = (
    "xy_axes",
    "xy_axes_ticks",
    "xy_axes_x_labels",
    "xy_axes_y_labels",
    "xy_axes_titles",
)


def remove_xy_axes(plotter) -> None:
    for name in _XY_AXES_NAMES:
        try:
            plotter.remove_actor(name)
        except Exception:
            pass
    if hasattr(plotter, "remove_bounds_axes"):
        try:
            plotter.remove_bounds_axes()
        except Exception:
            pass


def add_specimen_xy_axes(
    plotter,
    pv,
    mesh_pkg: Any,
    dic_xyz: Optional[np.ndarray] = None,
) -> None:
    """Numbered X and Y axes spanning the full specimen footprint."""
    remove_xy_axes(plotter)
    x_min, x_max, y_min, y_max = specimen_xy_extents(mesh_pkg, dic_xyz)
    span_x = max(x_max - x_min, 1.0e-9)
    span_y = max(y_max - y_min, 1.0e-9)
    span = max(span_x, span_y)
    tick_len = 0.018 * span
    z = 0.0
    axis_pts = np.array(
        [[x_min, y_min, z], [x_max, y_min, z], [x_min, y_min, z], [x_min, y_max, z]],
        dtype=float,
    )
    axis_lines = np.hstack(
        [[2, 0, 1], [2, 2, 3]]
    ).astype(np.int64)
    axes = pv.PolyData(axis_pts)
    axes.lines = axis_lines
    plotter.add_mesh(
        axes,
        color="#222222",
        line_width=2,
        name="xy_axes",
        reset_camera=False,
        pickable=False,
    )

    x_ticks = _nice_ticks(x_min, x_max)
    y_ticks = _nice_ticks(y_min, y_max)
    x_step = float(np.min(np.diff(x_ticks))) if x_ticks.size > 1 else span_x
    y_step = float(np.min(np.diff(y_ticks))) if y_ticks.size > 1 else span_y

    tick_pts = []
    tick_conn = []
    for value in x_ticks:
        a = len(tick_pts)
        tick_pts.append([float(value), y_min, z])
        tick_pts.append([float(value), y_min - tick_len, z])
        tick_conn.extend([2, a, a + 1])
    for value in y_ticks:
        a = len(tick_pts)
        tick_pts.append([x_min, float(value), z])
        tick_pts.append([x_min - tick_len, float(value), z])
        tick_conn.extend([2, a, a + 1])
    if tick_pts:
        ticks = pv.PolyData(np.asarray(tick_pts, dtype=float))
        ticks.lines = np.asarray(tick_conn, dtype=np.int64)
        plotter.add_mesh(
            ticks,
            color="#222222",
            line_width=1.5,
            name="xy_axes_ticks",
            reset_camera=False,
            pickable=False,
        )

    label_kwargs = dict(
        font_size=11,
        text_color="#222222",
        fill_shape=False,
        shape=None,
        always_visible=True,
        show_points=False,
        reset_camera=False,
        render=False,
    )
    x_label_pts = [[float(v), y_min - 2.4 * tick_len, z] for v in x_ticks]
    x_labels = [_format_tick(float(v), x_step) for v in x_ticks]
    y_label_pts = [[x_min - 2.6 * tick_len, float(v), z] for v in y_ticks]
    y_labels = [_format_tick(float(v), y_step) for v in y_ticks]
    try:
        plotter.add_point_labels(
            x_label_pts, x_labels, name="xy_axes_x_labels", **label_kwargs
        )
        plotter.add_point_labels(
            y_label_pts, y_labels, name="xy_axes_y_labels", **label_kwargs
        )
        plotter.add_point_labels(
            [
                [0.5 * (x_min + x_max), y_min - 5.2 * tick_len, z],
                [x_min - 5.4 * tick_len, 0.5 * (y_min + y_max), z],
            ],
            ["X (mm)", "Y (mm)"],
            name="xy_axes_titles",
            **label_kwargs,
        )
    except TypeError:
        label_kwargs.pop("name", None)
        plotter.add_point_labels(x_label_pts, x_labels, **label_kwargs)
        plotter.add_point_labels(y_label_pts, y_labels, **label_kwargs)


_COUNT_NAMES = ("element_counts", "element_counts_empty")


def remove_element_point_counts(plotter) -> None:
    for name in _COUNT_NAMES:
        try:
            plotter.remove_actor(name)
        except Exception:
            pass


def add_element_point_counts(
    plotter,
    mesh_pkg: Any,
    counts,
    *,
    max_labels: int = mesh_convert.MAX_ELEMENT_COUNT_LABELS,
) -> int:
    """Draw per-element DIC counts at element centroids. Empty cells in red."""
    remove_element_point_counts(plotter)
    arr = np.asarray(counts, dtype=np.int64).ravel()
    centroids = mesh_convert.element_centroids(mesh_pkg)
    n = min(int(centroids.shape[0]), int(arr.size))
    if n == 0 or n > int(max_labels):
        return 0
    xyz = mesh_convert.mesh_xyz(mesh_pkg)
    span = float(max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), 1.0))
    lift = 0.004 * span
    pts = centroids[:n].copy()
    pts[:, 2] = lift
    empty = arr[:n] == 0
    filled = ~empty
    label_kwargs = dict(
        font_size=13,
        fill_shape=False,
        shape=None,
        always_visible=True,
        show_points=False,
        reset_camera=False,
        render=False,
        bold=True,
        justification_horizontal="center",
        justification_vertical="center",
    )

    def _add(mask, color: str, name: str) -> None:
        if not np.any(mask):
            return
        labels = [str(int(v)) for v in arr[:n][mask]]
        points = pts[mask]
        try:
            plotter.add_point_labels(
                points, labels, name=name, text_color=color, **label_kwargs
            )
        except TypeError:
            fallback = dict(label_kwargs)
            fallback.pop("justification_horizontal", None)
            fallback.pop("justification_vertical", None)
            fallback.pop("bold", None)
            try:
                plotter.add_point_labels(
                    points, labels, name=name, text_color=color, **fallback
                )
            except TypeError:
                plotter.add_point_labels(points, labels, text_color=color, **fallback)

    _add(filled, "#0d5c2e", "element_counts")
    _add(empty, "#C0392B", "element_counts_empty")
    return n


def add_cid_cloud(plotter, pv, identifiability: Any) -> int:
    """Overlay the first-case CID map if identifiability results exist."""
    if identifiability is None:
        return 0
    cases = getattr(identifiability, "cases", None) or []
    if not cases:
        return 0
    case = cases[0]
    points = np.asarray(getattr(case, "points"), dtype=float)
    cid = np.asarray(getattr(case, "CID"), dtype=float)
    if cid.ndim > 1:
        cid = cid[:, 0]
    cid = cid.ravel()
    if points.ndim != 2 or points.shape[0] == 0:
        return 0
    xyz = points if points.shape[1] >= 3 else np.column_stack(
        [points[:, 0], points[:, 1], np.zeros(points.shape[0])]
    )
    cloud = pv.PolyData(xyz[:, :3])
    n = min(xyz.shape[0], cid.size)
    cloud["CID"] = cid[:n]
    plotter.add_mesh(
        cloud,
        scalars="CID",
        cmap="inferno",
        point_size=8,
        render_points_as_spheres=True,
        name="cid",
        show_scalar_bar=False,
    )
    _add_styled_scalar_bar(plotter, field_math_title("CID"))
    return n

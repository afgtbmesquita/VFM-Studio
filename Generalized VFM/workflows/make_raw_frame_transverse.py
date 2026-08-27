# =============================================================================
#  Generalized VFM / make_raw_frame_transverse.py
# -----------------------------------------------------------------------------
#  Undo the transverse export's +90 deg field rotation, so the campaign can be
#  run the OTHER way round: fields in the frame they were measured in, and the
#  constitutive matrix rotated instead.
#
#  WHY THIS EXISTS
#  ---------------------------------------------------------------------------
#  SCRIPT_export_transverse_ML_dataset.m rotates the transverse coupon's fields
#  +90 deg in-plane (X' = Y, Y' = -X) at export time, so that both coupons hand
#  the solver curvatures in one common material frame.  That works for 90 deg,
#  where the rotation is a signed permutation of channels, but it does not
#  extend: at 45 deg a rotation MIXES kappa_xx, kappa_yy and kappa_xy, so every
#  channel's factor-of-two and sign convention would have to be exactly right or
#  all three are contaminated.
#
#  The alternative is to leave the measurement alone and rotate D.  To prove the
#  two are equivalent we need the transverse data in its raw frame -- and since
#  the export's rotation is a signed permutation, its inverse is EXACT and can
#  be applied here without re-running MATLAB or touching the .h5.
#
#      forward (MATLAB, at export)          inverse (here)
#      X'   = Y_raw                         x_raw = -y'
#      Y'   = -X_raw                        y_raw = +x'
#      cxx' = cyy_raw                       SK1_raw = SK2'
#      cyy' = cxx_raw                       SK2_raw = SK1'
#      cxy' = -cxy_raw                      SK3_raw = -SK3'
#
#  The applied moment vector F is NOT touched.  It was never rotated in the
#  first place (the machine wrench ships raw, and the export identifies the dTy
#  CHANNEL with a plate moment component), and the boundary-condition mirror
#  below maps the loaded edges so that F keeps pairing with the same physical
#  cuts in the same order.
#
#  NOTHING IN Transverse_bending/VFM_export/ IS MODIFIED.  Output goes to a
#  sibling VFM_export_raw_frame/ directory, so route (A) stays runnable as the
#  reference that route (B) is checked against.
#
#      python -m workflows.make_raw_frame_transverse
#
#  The script re-applies the forward rotation to its own output and refuses to
#  leave the files on disk unless the round trip is bit-identical.
# =============================================================================

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from workflows.campaign_common import TRANS_DIR, TRANS_RAW_DIR

BC_SOURCE = TRANS_DIR / "BC_Transverse.py"
BC_TARGET_NAME = "BC_Transverse_raw.py"


# =============================================================================
#  The inverse rotation
# =============================================================================

def unrotate_nodes(nodes: np.ndarray) -> np.ndarray:
    """Fixed-frame node coordinates back to the raw measurement frame."""
    nodes = np.asarray(nodes, dtype=float)
    raw = np.empty_like(nodes)
    raw[:, 0] = -nodes[:, 1]     # x_raw = -y'
    raw[:, 1] = +nodes[:, 0]     # y_raw = +x'
    return raw


def unrotate_curvatures(k_xx, k_yy, k_xy):
    """Fixed-frame curvature triplet back to the raw measurement frame.

    A symmetric second-order tensor under a 90 deg frame rotation: the normal
    components swap and the shear negates.  The map is an involution, which is
    what makes the round-trip self-check below meaningful.
    """
    return (np.asarray(k_yy, dtype=float).copy(),
            np.asarray(k_xx, dtype=float).copy(),
            -np.asarray(k_xy, dtype=float))


def rotate_nodes_forward(nodes: np.ndarray) -> np.ndarray:
    """The export's own +90 deg map, used only to verify the round trip."""
    nodes = np.asarray(nodes, dtype=float)
    out = np.empty_like(nodes)
    out[:, 0] = +nodes[:, 1]     # X' = Y_raw
    out[:, 1] = -nodes[:, 0]     # Y' = -X_raw
    return out


# =============================================================================
#  Bounds and the boundary-condition mirror
# =============================================================================

def unrotate_bounds(bounds: dict) -> dict:
    """Mesh bounds under x_raw = -y', y_raw = +x'."""
    return {
        "x_min": -float(bounds["y_max"]),
        "x_max": -float(bounds["y_min"]),
        "y_min": +float(bounds["x_min"]),
        "y_max": +float(bounds["x_max"]),
    }


#  Under x_raw = -y', y_raw = +x' the four named edges map as follows.  Read it
#  as "the edge called K in the fixed frame is called EDGE_MAP[K] in the raw
#  frame": the loaded cuts (right/left) become top/bottom, and the traction-free
#  lateral edges (top/bottom) become left/right.
EDGE_MAP = {"right": "top", "left": "bottom", "top": "left", "bottom": "right"}

#  d/dx' = d/dy_raw and d/dy' = -d/dx_raw, so the two first-derivative DOFs
#  exchange.  Every constraint using them is homogeneous (value 0) and every
#  EXTERNAL group is driven by the .mat F, so the sign carried by the map is
#  immaterial -- but the RENAMING is not.
DOF_MAP = {"W": "W", "W_X": "W_Y", "W_Y": "W_X", "W_XY": "W_XY"}

#  A loaded edge that ran along y' runs along x_raw, and vice versa.
SORT_MAP = {"x": "y", "y": "x"}


def mirror_bc_entry(entry: dict) -> dict:
    """One PRESCRIBED / EXTERNAL dict rewritten for the raw frame."""
    out = dict(entry)
    if "edge" in out:
        out["edge"] = EDGE_MAP[str(out["edge"])]
    if "dof" in out:
        out["dof"] = DOF_MAP[str(out["dof"])]
    if out.get("sortDirection") in SORT_MAP:
        out["sortDirection"] = SORT_MAP[out["sortDirection"]]
    if "label" in out:
        out["label"] = (f"{out['label']}  [raw frame: was "
                        f"{entry.get('edge', '?')} / {entry.get('dof', '?')}]")
    return out


def _format_entries(entries) -> str:
    return "\n".join(f"    {entry!r}," for entry in entries)


def write_raw_bc_file(target: Path) -> dict:
    """Generate BC_Transverse_raw.py from the shipped BC_Transverse.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(BC_SOURCE.stem, BC_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    mesh = dict(module.MESH_SETTINGS)
    raw_mesh = {
        #  The element counts are given in the SOLVER's x/y, so they transpose
        #  with the frame just like the bounds do.
        "number_of_elements_x": int(mesh["number_of_elements_y"]),
        "number_of_elements_y": int(mesh["number_of_elements_x"]),
        "bounds": unrotate_bounds(mesh["bounds"]),
    }
    solver = dict(module.SOLVER_SETTINGS)   # frame-invariant (a bbox area)

    prescribed = [mirror_bc_entry(e) for e in module.PRESCRIBED]
    linking = [mirror_bc_entry(e) for e in module.LINKING]
    external = [mirror_bc_entry(e) for e in module.EXTERNAL]

    text = f'''# =============================================================================
#  BC_Transverse_raw.py -- GENERATED, do not edit by hand.
#
#  Written by Generalized VFM/make_raw_frame_transverse.py from
#  {BC_SOURCE.name}, mirrored out of the fixed material frame and back into
#  the frame the transverse coupon was actually measured in.
#
#      x_raw = -y_fixed      y_raw = +x_fixed
#      right/left (loaded cuts)   ->  top/bottom
#      top/bottom (free edges)    ->  left/right
#      W_X <-> W_Y                    sortDirection x <-> y
#
#  Used with material_orientation_deg = 90.0, i.e. the material 1-axis lies at
#  +90 deg from this frame's +x.  The D matrix is rotated by that angle instead
#  of the fields; see vfm_constitutive.rotate_constitutive_model.
#
#  The EXTERNAL entries keep their original ORDER, so the .mat 'F' vector still
#  pairs with the same two physical cuts.
# =============================================================================

MESH_SETTINGS = {raw_mesh!r}

SOLVER_SETTINGS = {solver!r}

PRESCRIBED = [
{_format_entries(prescribed)}
]

LINKING = [
{_format_entries(linking)}
]

EXTERNAL = [
{_format_entries(external)}
]
'''
    target.write_text(text, encoding="utf-8")
    return raw_mesh


# =============================================================================
#  Driver
# =============================================================================

def main() -> int:
    sources = sorted((TRANS_DIR / "mat").glob("Transverse_step*.mat"))
    if not sources:
        raise SystemExit(
            f"No transverse .mat found in {TRANS_DIR / 'mat'}. "
            "Run Transverse_bending/VFM_export/export_vfm_dataset.py first.")

    out_mat = TRANS_RAW_DIR / "mat"
    if TRANS_RAW_DIR.exists():
        shutil.rmtree(TRANS_RAW_DIR)
    out_mat.mkdir(parents=True)

    print(f"[raw-frame] undoing the +90 deg export rotation on "
          f"{len(sources)} load step(s)")
    print(f"[raw-frame]   source {TRANS_DIR / 'mat'}")
    print(f"[raw-frame]   target {out_mat}")

    worst_round_trip = 0.0
    for path in sources:
        data = loadmat(path, squeeze_me=True)
        nodes = np.asarray(data["nodes"], dtype=float)
        k_xx = np.asarray(data["SK1grid"], dtype=float)
        k_yy = np.asarray(data["SK2grid"], dtype=float)
        k_xy = np.asarray(data["SK3grid"], dtype=float)

        raw_nodes = unrotate_nodes(nodes)
        raw_xx, raw_yy, raw_xy = unrotate_curvatures(k_xx, k_yy, k_xy)

        #  Round trip: re-apply the export's own forward map and demand the
        #  original back exactly.  A signed permutation has no round-off, so
        #  anything but 0.0 here means the maps disagree.
        back_nodes = rotate_nodes_forward(raw_nodes)
        back_xx, back_yy, back_xy = unrotate_curvatures(raw_xx, raw_yy, raw_xy)
        errors = [
            np.abs(back_nodes - nodes).max(),
            np.abs(back_xx - k_xx).max(),
            np.abs(back_yy - k_yy).max(),
            np.abs(back_xy - k_xy).max(),
        ]
        worst_round_trip = max(worst_round_trip, float(max(errors)))

        target = out_mat / path.name.replace("Transverse_step",
                                             "Transverse_raw_step")
        savemat(target, {
            "nodes": raw_nodes,
            "SK1grid": raw_xx,
            "SK2grid": raw_yy,
            "SK3grid": raw_xy,
            #  Untouched on purpose: the wrench was never rotated, and the BC
            #  mirror keeps the group order, so F still pairs with the same
            #  physical cuts.
            "F": np.asarray(data["F"], dtype=float),
        })

    if worst_round_trip != 0.0:
        shutil.rmtree(TRANS_RAW_DIR)
        raise SystemExit(
            f"Round trip is not exact (worst |error| = {worst_round_trip:.3e}). "
            "The forward and inverse rotations disagree; nothing was written.")
    print(f"[raw-frame] round trip through the forward map: "
          f"{worst_round_trip:.1e} (exact, as a signed permutation must be)")

    raw_mesh = write_raw_bc_file(TRANS_RAW_DIR / BC_TARGET_NAME)
    print(f"[raw-frame] wrote {BC_TARGET_NAME}")
    print(f"[raw-frame]   bounds x [{raw_mesh['bounds']['x_min']:.6f}, "
          f"{raw_mesh['bounds']['x_max']:.6f}]  "
          f"y [{raw_mesh['bounds']['y_min']:.6f}, "
          f"{raw_mesh['bounds']['y_max']:.6f}]")

    #  The point of the exercise: the coupon's length must now run along y,
    #  like the longitudinal coupon's, because we removed the rotation that
    #  had put it along x.
    check = loadmat(sorted(out_mat.glob("*.mat"))[-1], squeeze_me=True)
    n = np.asarray(check["nodes"], dtype=float)
    span_x = n[:, 0].max() - n[:, 0].min()
    span_y = n[:, 1].max() - n[:, 1].min()
    print(f"[raw-frame] raw-frame spans: x {span_x:.3f} mm, y {span_y:.3f} mm "
          f"-> length along {'y' if span_y > span_x else 'x'}")
    if span_y <= span_x:
        raise SystemExit(
            "The raw-frame window is still longer in x than in y. The shipped "
            ".mat do not look pre-rotated; check the export.")

    print("[raw-frame] done. Run validate_rotation.py to compare the routes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

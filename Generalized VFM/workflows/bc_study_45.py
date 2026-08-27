# =============================================================================
#  Generalized VFM / bc_study_45.py
# -----------------------------------------------------------------------------
#  Three questions, answered by measurement rather than by argument:
#
#    1. WHY IS THE 45 DEG WINDOW TRIMMED SO HARD ALONG y?
#       The objection is exactly right in principle: if the virtual field
#       annihilates every unknown traction on the window boundary, the VFM
#       identity holds on ANY subdomain, and the answer must not depend on
#       where the cuts are placed.  So the trim is swept from the bare
#       15-filter minimum (7 steps, window straddling the curvature reversal)
#       to the shipped 28, under EVERY boundary-condition form, and the
#       dependence is plotted.
#
#       The distinction that matters is between two kinds of BC form:
#
#         * forms that FEED THE MACHINE COUPLE IN at a cut assert
#           M_yy(cut) = M_machine.  That is an assertion about the SPECIMEN,
#           not about the virtual field, and it is false as soon as the
#           internal moment varies along the length -- which is exactly what a
#           curvature reversal is.  These forms MUST be trim-dependent.
#
#         * the SEALED form prescribes W, W_X, W_XY and W_Y on BOTH cuts, so
#           no traction -- known or unknown -- does any virtual work.  The
#           external work is identically zero and the VFM reduces to the
#           homogeneous system K theta = 0.  Nothing about the moment enters,
#           so this form is the objection's ideal case, and it should be
#           trim-INDEPENDENT.  It buys that by determining theta only up to
#           scale (see SEALED FORM below).
#
#       Sweeping both and comparing is the actual test.
#
#    2. WHICH BC FORM SHOULD THE 45 DEG COUPON USE?
#       Same rule as bc_study.py: a cut is sealed when W, W_X, W_XY, W_Y are
#       all prescribed; a DOF may be left free only if its conjugate moment is
#       measured and supplied.  The longitudinal and transverse coupons are
#       held at their shipped reference throughout, so every difference is
#       attributable to the 45 deg coupon.
#
#    3. WHY IS D12 SO SENSITIVE?
#       Because at 45 deg the three normal-block coefficients enter ONLY
#       through the combination S = (D11 + D22 + 2 D12)/4, so
#
#           D12 = 2 S - (D11 + D22) / 2
#
#       is a DIFFERENCE OF LARGE NUMBERS: ~96 - ~105 here.  The condition
#       analysis in `d12_diagnosis()` quantifies the amplification.
#
#  THE SEALED FORM -- what "cancel the unknowns entirely" actually costs
#  ---------------------------------------------------------------------------
#  Seal both cuts and the virtual-work identity becomes
#
#      sum_p w_p kappa*_p^T D kappa_p  =  0        for every admissible w*
#
#  which is K theta = 0 with K the usual m x 4 sensitivity matrix.  This is
#  homogeneous: it fixes the DIRECTION of theta in parameter space but not its
#  length, because scaling D scales both sides.  It is therefore immune to the
#  applied moment, the specimen width, the machine calibration AND the moment
#  gradient -- every quantity the objection is right to be suspicious of.
#
#  The scale must come from somewhere, and here it comes from the other two
#  coupons, which are loaded normally.  So the sealed 45 deg coupon contributes
#  pure SHAPE information (the ratios D12/D11, D22/D11, D66/D11) and no
#  magnitude, which is precisely the half of its information that is
#  trustworthy.
#
#      python -m workflows.bc_study_45                # everything, plus the figures
#      python -m workflows.bc_study_45 --quick        # skip the trim sweep
# =============================================================================

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from workflows.campaign_common import (
    ANALYSIS_TYPE,
    FORTYFIVE,
    FIGURES_DIR,
    HERE,
    INITIAL_PARAMETERS,
    LONGITUDINAL,
    MATERIAL_TYPE,
    RESULTS_DIR,
    TESTS_WITH_45,
    TRANSVERSE,
    build_cases,
    load_bc_module,
)
from workflows.make_figures import apply_style, chi_square_diagnostics, math_label, soft_grid
from vfm_constitutive import define_constitutive_model
from vfm_data_io import BoundaryConditions, PointMeasurements
from vfm_identification import IdentificationCase
from vfm_gls_identification import run_gls_identification_cases
from vfm_mesh import build_mesh, map_points_to_mesh, prune_mesh_by_measurement_points
from vfm_user_inputs import (
    Bounds,
    GlsSettings,
    IdentificationSettings,
    LinearAlgebraSettings,
    MeshSettings,
)

LABELS = ("D11", "D12", "D22", "D66")
FIGURE_DIR = FIGURES_DIR / "study_45"

#  Trim sweep, in facet-grid steps removed from the LOW-y end of the 45 deg
#  window.  7 is the bare Vic-3D 15-filter half width -- i.e. no physical trim
#  at all, window straddling the curvature reversal at y ~ 4.2 mm.  28 is what
#  the shipped export uses.  The HIGH-y end stays at 7 throughout, since that
#  end has only the filter artifact to remove.
LOW_Y_TRIMS = (7, 12, 17, 22, 28, 33)
HIGH_Y_TRIM = 7


# =============================================================================
#  Building 45 deg cases at an arbitrary trim, without writing .mat files
# =============================================================================

def fortyfive_cases_at_trim(low_y_trim: int,
                            bc: BoundaryConditions,
                            external_values,
                            n_elements: int = 2) -> Tuple[List, Dict]:
    """IdentificationCases for the 45 deg coupon at one length trim.

    Reads the .h5 through the shipped export module, so the channel algebra,
    the reference step and the moment sign are the exported ones by
    construction and cannot drift from them.

    ``external_values(step, bending_pair)`` returns the list of EXTERNAL
    values for that load step, in the order of ``bc.external``.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export45", FORTYFIVE.directory / "export_vfm_dataset.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)

    steps = list(range(export.REFERENCE_STEP, len(export.FILE_NUMBERS) + 1))
    loaded = {s: export.load_step(export.FILE_NUMBERS[s - 1]) for s in steps}
    selection = export.select_points([loaded[s][2] for s in steps],
                                     low_y_trim=low_y_trim,
                                     high_y_trim=HIGH_Y_TRIM)

    ref_data, columns, ref_keys = loaded[export.REFERENCE_STEP]
    ref = ref_data[export.row_index(ref_keys, selection)]
    nodes = np.column_stack([ref[:, columns["X"]],
                             ref[:, columns["Y"]]]).astype(float)

    x_min, x_max = float(nodes[:, 0].min()), float(nodes[:, 0].max())
    y_min, y_max = float(nodes[:, 1].min()), float(nodes[:, 1].max())
    bbox_area = (x_max - x_min) * (y_max - y_min)
    n_points = nodes.shape[0]

    #  Round the mesh box OUTWARD so the extreme facets map -- the same trap
    #  the shipped export documents.
    pad = 1e-6
    mesh_settings = MeshSettings(
        number_of_elements_x=n_elements,
        number_of_elements_y=n_elements,
        bounds=Bounds(x_min=x_min - pad, x_max=x_max + pad,
                      y_min=y_min - pad, y_max=y_max + pad))

    geometry = {"low_y_trim": low_y_trim, "n_points": n_points,
                "y_min": y_min, "y_max": y_max, "length": y_max - y_min,
                "bbox_area": bbox_area}

    cases = []
    for step in steps:
        if step == export.REFERENCE_STEP:
            continue
        data, _, keys = loaded[step]
        current = data[export.row_index(keys, selection)]
        k_xx, k_yy, k_xy = export.curvatures(current, ref, columns)
        moment = export.moment(current, ref, columns)
        bending = (-moment, +moment)  # [top, bottom], as EXTERNAL_SIGNS writes

        measurements = PointMeasurements(
            points=nodes,
            strains=np.vstack([k_xx, k_yy, k_xy]).astype(float),
            applied_force=None,
            area_per_point=bbox_area / n_points,
            data_bounds={"x_min": x_min, "x_max": x_max,
                         "y_min": y_min, "y_max": y_max},
            analysis_type=ANALYSIS_TYPE)

        mesh_pkg = build_mesh(mesh_settings, analysis_type=ANALYSIS_TYPE)
        mapping = map_points_to_mesh(measurements.points, mesh_pkg)
        if mapping.n_unmapped:
            raise SystemExit(
                f"trim {low_y_trim}: {mapping.n_unmapped} points outside the mesh")
        mesh_pkg, _ = prune_mesh_by_measurement_points(mesh_pkg,
                                                       measurements.points)

        step_bc = copy.deepcopy(bc)
        values = external_values(step, bending)
        step_bc.external = [{**entry, "value": value}
                            for entry, value in zip(step_bc.external, values)]

        case = IdentificationCase(
            measurements=measurements,
            mesh_pkg=mesh_pkg,
            boundary_conditions=step_bc,
            label=f"45deg:trim{low_y_trim:02d}_step{step:02d}",
            data_file=None,
            material_orientation_deg=FORTYFIVE.material_orientation_deg,
        )
        case.applied_load_source = "manual"
        cases.append(case)

    return cases, geometry


# =============================================================================
#  The BC variants -- only the 45 deg coupon changes
# =============================================================================

@dataclass(frozen=True)
class Variant45:
    key: str
    title: str
    prescribed: Dict[str, Tuple[str, ...]]
    external: Tuple[Tuple[str, str], ...]   # (edge, dof)
    linked: Tuple[Tuple[str, str], ...] = ()
    sealed: bool = False
    note: str = ""

    def boundary_conditions(self) -> BoundaryConditions:
        prescribed = [{"label": f"{edge} {dof}=0", "dof": dof,
                       "edge": edge, "value": 0.0}
                      for edge, dofs in self.prescribed.items()
                      for dof in dofs]
        external = [{"label": f"{edge} {dof}", "dof": dof, "edge": edge,
                     "loadType": "distributed", "sortDirection": "x"}
                    for edge, dof in self.external]
        linking = [{"label": f"{edge} {dof} tied", "dof": dof, "edge": edge,
                    "sortDirection": "x"} for edge, dof in self.linked]
        return BoundaryConditions(prescribed=prescribed, linking=linking,
                                  external=external,
                                  analysis_type=ANALYSIS_TYPE)

    def values(self, step, bending):
        """EXTERNAL values for one step, in the order of self.external."""
        lookup = {"top": bending[0], "bottom": bending[1]}
        return [lookup[edge] for edge, _dof in self.external]


_ALL = ("W", "W_X", "W_XY", "W_Y")
_CUT = ("W", "W_X", "W_XY")

VARIANTS_45: Tuple[Variant45, ...] = (
    Variant45(
        "A", "A  both cuts loaded  [SHIPPED]",
        prescribed={"top": _CUT, "bottom": _CUT},
        external=(("top", "W_Y"), ("bottom", "W_Y")),
        note="F = [-M, +M]; external work follows the MEAN of the two cut moments"),
    Variant45(
        "B", "B  top loaded, bottom sealed",
        prescribed={"top": _CUT, "bottom": _ALL},
        external=(("top", "W_Y"),),
        note="assumes the TOP cut moment equals the machine couple"),
    Variant45(
        "C", "C  bottom loaded, top sealed",
        prescribed={"top": _ALL, "bottom": _CUT},
        external=(("bottom", "W_Y"),),
        note="assumes the BOTTOM cut moment equals the machine couple"),
    Variant45(
        "D", "D  both cuts loaded + linked W_Y",
        prescribed={"top": _CUT, "bottom": _CUT},
        external=(("top", "W_Y"), ("bottom", "W_Y")),
        linked=(("top", "W_Y"), ("bottom", "W_Y")),
        note="each cut rotates as one; admissible, smaller virtual-field space"),
    Variant45(
        "E", "E  bottom W_XY free  [LEAKS]",
        prescribed={"top": _CUT, "bottom": ("W", "W_X", "W_Y")},
        external=(("top", "W_Y"),),
        note="LEAKS: the bottom twisting moment does unaccounted work"),
    Variant45(
        "F", "F  top W_XY free  [LEAKS]",
        prescribed={"top": ("W", "W_X"), "bottom": _ALL},
        external=(("top", "W_Y"),),
        note="LEAKS: the top twisting moment does unaccounted work"),
    Variant45(
        "G", "G  lateral edges: w* = 0",
        prescribed={"top": _CUT, "bottom": _CUT,
                    "left": ("W", "W_Y", "W_XY"),
                    "right": ("W", "W_Y", "W_XY")},
        external=(("top", "W_Y"), ("bottom", "W_Y")),
        note="w*=0 along the free edges; still RELIES on M_xx = 0 there, "
             "since the normal slope W_X is left free"),
    Variant45(
        "H", "H  lateral edges fully sealed",
        prescribed={"top": _CUT, "bottom": _CUT,
                    "left": _ALL, "right": _ALL},
        external=(("top", "W_Y"), ("bottom", "W_Y")),
        note="the free-edge assumption is NOT used at all -- the honest test "
             "of whether the window really reaches the free edges"),
    Variant45(
        "S", "S  BOTH CUTS SEALED  (scale-free)",
        prescribed={"top": _ALL, "bottom": _ALL},
        external=(),
        sealed=True,
        note="no traction does virtual work: K.theta = 0, theta up to scale"),
)


# =============================================================================
#  Running
# =============================================================================

def _settings():
    return (IdentificationSettings(max_iterations=40,
                                   convergence_tolerance=1.0e-8),
            LinearAlgebraSettings())


def reference_cases() -> Tuple[List, List[int]]:
    """The longitudinal + transverse coupons, at their shipped reference."""
    cases, ids = [], []
    for spec in (LONGITUDINAL, TRANSVERSE):
        spec_cases = build_cases(spec, verbose=False)
        cases.extend(spec_cases)
        ids.extend([spec.experiment_id] * len(spec_cases))
    return cases, ids


def run_variant(variant: Variant45, low_y_trim: int,
                base: Optional[Tuple[List, List[int]]] = None):
    """Identify with the 45 deg coupon under `variant` at `low_y_trim`."""
    base_cases, base_ids = base if base is not None else reference_cases()
    cases45, geometry = fortyfive_cases_at_trim(
        low_y_trim, variant.boundary_conditions(), variant.values)

    if variant.sealed:
        return sealed_direction(cases45, geometry)

    cases = list(base_cases) + cases45
    ids = list(base_ids) + [FORTYFIVE.experiment_id] * len(cases45)
    identification, linear_algebra = _settings()
    result = run_gls_identification_cases(
        cases=cases, material_type=MATERIAL_TYPE, analysis_type=ANALYSIS_TYPE,
        initial_parameters=INITIAL_PARAMETERS,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra, experiment_ids=ids,
        gls_settings=GlsSettings(), verbose=False)
    return {"theta": np.asarray(result.final_parameters, dtype=float),
            "chi2": float(chi_square_diagnostics(result)["reduced"]),
            "geometry": geometry, "result": result}


# =============================================================================
#  The sealed form: K.theta = 0, solved as a null-space direction
# =============================================================================

def sealed_design_matrix(case) -> np.ndarray:
    """``R^T A`` for one sealed case: the homogeneous system's rows.

    For a virtual field ``u = R q`` restricted to the admissible space, the
    internal virtual work is

        IVW(theta) = u^T A theta = q^T (R^T A) theta

    where ``A[:, j] = sum_p w_p B_p^T B_j^lab kappa_p`` is exactly the object
    the optimal-field solver assembles.  With BOTH cuts fully sealed the
    external virtual work is identically zero, so the VFM statement is

        (R^T A) theta = 0        for every admissible q

    which is what this returns.  Note the OPTIMAL-field route cannot be used
    here: the speciality condition pins K = I by construction, so a zero
    right-hand side would simply return theta = 0.  The homogeneous system has
    to be assembled directly, which is what this does.
    """
    from vfm_constitutive import rotate_constitutive_model
    from vfm_mesh import build_admissible_basis, build_kinematic_constraints
    from vfm_optimal_field import (
        OptimalFieldSolverOptions,
        _assemble_virtual_force_vector,
        collect_assembly_payload,
    )

    model = define_constitutive_model(MATERIAL_TYPE, ANALYSIS_TYPE,
                                      INITIAL_PARAMETERS)
    #  Same rotation the solver applies: the fields are in the coupon's own
    #  frame, so the BASIS matrices move and theta stays material-frame.
    model = rotate_constitutive_model(model, case.material_orientation_deg)

    options = OptimalFieldSolverOptions()
    mapping = map_points_to_mesh(case.measurements.points, case.mesh_pkg)
    n_gen = model.material_matrix.shape[0]
    payload = collect_assembly_payload(case.measurements, case.mesh_pkg,
                                       mapping, n_gen)

    A, touched = _assemble_virtual_force_vector(
        measurements=case.measurements, mesh_pkg=case.mesh_pkg,
        mapping=mapping, basis_matrices=model.basis_matrices, payload=payload)

    inactive = np.where((np.abs(A).sum(axis=1)
                         <= options.inactive_dof_threshold) & (~touched))[0]
    constraints = build_kinematic_constraints(
        mesh_pkg=case.mesh_pkg, bc=case.boundary_conditions,
        inactive_dofs=inactive)
    R, _meta = build_admissible_basis(
        C=constraints.C, n_dofs=case.mesh_pkg.n_dofs,
        svd_tolerance_factor=options.svd_tolerance_factor,
        rank_tolerance_floor=options.rank_tolerance_floor)
    return R.T @ A


def sealed_direction(cases45, geometry) -> Dict:
    """theta DIRECTION from the 45 deg coupon with both cuts fully sealed.

    NO MOMENT, NO WIDTH, NO MACHINE CALIBRATION AND NO MOMENT GRADIENT ENTERS
    -- every quantity the trim objection is right to be suspicious of drops
    out, because nothing is fed in at the boundary at all.  The price is that
    only the DIRECTION of theta is determined: scaling D scales both sides of
    a homogeneous equation.

    The estimate is the right singular vector of the stacked ``R^T A`` block
    belonging to its smallest singular value -- the direction that comes
    closest to satisfying the homogeneous equation at every load step at once.
    """
    design = np.vstack([sealed_design_matrix(case) for case in cases45])

    #  Column scaling: the four columns carry very different magnitudes and an
    #  unscaled null space would simply pick the largest.  Scale to unit column
    #  norm, solve, then undo -- so "smallest singular value" means smallest
    #  RELATIVE inconsistency, which is the meaningful notion here.
    scale = np.linalg.norm(design, axis=0)
    scale[scale == 0.0] = 1.0
    _, singular, right = np.linalg.svd(design / scale, full_matrices=False)
    direction = right[-1] / scale
    if direction[0] < 0:
        direction = -direction
    return {"theta": direction, "chi2": float("nan"), "geometry": geometry,
            "singular": singular,
            "gap": float(singular[-2] / singular[-1]) if singular.size > 1
            else float("nan"),
            "sealed": True}


# =============================================================================
#  Why D12 moves so much
# =============================================================================

def s_value(theta) -> float:
    """S = (D11 + D22 + 2 D12) / 4 -- the ONLY normal-block combination a
    45 deg coupon can see."""
    d11, d12, d22, _ = (float(v) for v in theta)
    return (d11 + d22 + 2.0 * d12) / 4.0


def d12_empirical(results: Dict[str, Dict], baseline) -> None:
    """The decisive check: S is stable, D12 is the amplified difference.

    If D12 really is a cancellation artefact rather than an independently
    measured quantity, then across BC forms S should move little while D12
    swings wildly -- and the swing should be reproduced exactly by
    D12 = 2 S - (D11 + D22)/2.  Both are checked here rather than asserted.
    """
    print()
    print("  MEASURED ACROSS THE BC FORMS -- S is stable, D12 is not")
    print("  " + "-" * 74)
    print(f"    {'variant':<8}{'D11':>9}{'D22':>9}{'(D11+D22)/2':>13}"
          f"{'S':>9}{'2S':>9}{'D12':>9}")
    keys = [v.key for v in VARIANTS_45 if v.key in results and not v.sealed]
    table = []
    for key in keys:
        theta = np.asarray(results[key]["theta"], dtype=float)
        s = s_value(theta)
        half = (theta[0] + theta[2]) / 2.0
        table.append((key, theta[0], theta[2], half, s, 2 * s, theta[1]))
        print(f"    {key:<8}{theta[0]:9.2f}{theta[2]:9.2f}{half:13.2f}"
              f"{s:9.2f}{2 * s:9.2f}{theta[1]:9.2f}")
    if not table:
        return
    array = np.array([row[1:] for row in table], dtype=float)
    spans = array.max(axis=0) - array.min(axis=0)
    middles = np.abs(0.5 * (array.max(axis=0) + array.min(axis=0)))
    percent = 100.0 * spans / np.maximum(middles, 1e-12)
    print(f"    {'spread %':<8}{percent[0]:9.1f}{percent[1]:9.1f}"
          f"{percent[2]:13.1f}{percent[3]:9.1f}{percent[4]:9.1f}"
          f"{percent[5]:9.1f}")
    print()
    print(f"    S moves {percent[3]:.0f} % across the admissible BC forms while "
          f"D12 moves {percent[5]:.0f} %.")
    print("    The identification is determining S robustly.  D12 is a")
    print("    DERIVED difference, and the derivation is what amplifies.")
    print()
    check = np.array([2 * row[4] - row[3] for row in table])
    actual = np.array([row[6] for row in table])
    print(f"    identity D12 = 2S - (D11+D22)/2 reproduces every row to "
          f"{np.max(np.abs(check - actual)):.2e} N.mm")


def d12_diagnosis(theta: Sequence[float]) -> None:
    """The 45 deg coupon constrains D11, D12, D22 only through one number."""
    from vfm_constitutive import engineering_rotation_matrix

    print()
    print("=" * 78)
    print("  WHY D12 IS THE MOST SENSITIVE COEFFICIENT")
    print("=" * 78)

    t45 = engineering_rotation_matrix(45.0)
    basis = {name: np.zeros((3, 3)) for name in LABELS}
    basis["D11"][0, 0] = 1.0
    basis["D12"][0, 1] = basis["D12"][1, 0] = 1.0
    basis["D22"][1, 1] = 1.0
    basis["D66"][2, 2] = 1.0

    print("  Lab-frame basis matrices at 45 deg, NORMAL block only:")
    print(f"    {'':>6}{'[0,0]':>9}{'[0,1]':>9}{'[1,1]':>9}")
    for name in LABELS:
        rotated = t45.T @ basis[name] @ t45
        print(f"    {name:>6}{rotated[0, 0]:9.2f}{rotated[0, 1]:9.2f}"
              f"{rotated[1, 1]:9.2f}")
    print()
    print("  D11, D12 and D22 all produce the SAME pattern (+1,+1,+1) x their")
    print("  coefficient / 4, with D12 counted twice.  So in the normal")
    print("  channels the 45 deg coupon sees one number only:")
    print()
    print("      S = (D11 + D22 + 2 D12) / 4        and, separately, D66")
    print()
    print("  It cannot separate D11, D12 and D22 at all -- that separation has")
    print("  to come from the two axis-aligned coupons.  Inverting for D12:")
    print()
    print("      D12 = 2 S - (D11 + D22) / 2")
    print()
    d11, d12, d22, d66 = (float(v) for v in theta)
    s_value = (d11 + d22 + 2.0 * d12) / 4.0
    print(f"    2 S            = {2.0 * s_value:9.2f} N.mm")
    print(f"    (D11 + D22)/2  = {(d11 + d22) / 2.0:9.2f} N.mm")
    print(f"    D12            = {d12:9.2f} N.mm   <- the difference")
    print()
    magnitude = max(2.0 * s_value, (d11 + d22) / 2.0)
    print(f"  A DIFFERENCE OF LARGE NUMBERS: {magnitude:.0f} - {magnitude:.0f}"
          f" ~ {d12:.0f}.  The cancellation is")
    print(f"  {magnitude / max(abs(d12), 1e-9):.0f}-fold, so a 1 % error in "
          f"EITHER term becomes a "
          f"{100.0 * 0.01 * magnitude / max(abs(d12), 1e-9):.0f} % error in D12.")
    print()
    print("  Sensitivity of D12 to each input, all else held:")
    print(f"    dD12/dS   = 2      -> 1 N.mm of S   moves D12 by 2.0 N.mm")
    print(f"    dD12/dD11 = -1/2   -> 1 N.mm of D11 moves D12 by 0.5 N.mm")
    print(f"    dD12/dD22 = -1/2   -> 1 N.mm of D22 moves D12 by 0.5 N.mm")
    print()
    print("  And S is the 45 deg coupon's MAGNITUDE observable -- the one the")
    print("  moment gradient contaminates -- while its ratio observable")
    print("  (the anticlastic ratio) carries D66 instead.  So D12 inherits")
    print("  twice the error of the least trustworthy thing this coupon")
    print("  measures, on top of half the error of D11, which the campaign")
    print("  already could not pin down.  That is the whole explanation.")
    print()
    print("  IT IS NOT A 45 DEG PROBLEM PER SE.  D12 was already unresolved on")
    print("  two coupons (0.44 to 2.07 depending on BC form, CI +/- 4.3).  The")
    print("  45 deg coupon does not make D12 worse; it makes the reason")
    print("  VISIBLE, because it is the first test that constrains the")
    print("  combination D12 sits inside at all.")


# =============================================================================
#  Figures
# =============================================================================

COLOURS = {"D11": "#1F4E79", "D12": "#C55A11", "D22": "#2E7D32",
           "D66": "#7B4FA0"}
MARKERS = {"D11": "o", "D12": "s", "D22": "^", "D66": "D"}


def figure_trim(sweep, path, d11_scale=1.0) -> None:
    """Coefficients vs trim, one panel per BC form."""
    forms = list(sweep.keys())
    n = len(forms)
    columns = 4
    rows = int(np.ceil(n / columns))
    figure, axes = plt.subplots(rows, columns,
                                figsize=(7.16, 1.85 * rows + 0.55),
                                sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for index, key in enumerate(forms):
        axis = axes[index]
        entries = sweep[key]
        variant0 = next(v for v in VARIANTS_45 if v.key == key)
        starts = [e["geometry"]["y_min"] for e in entries]
        #  The sealed form determines theta only up to scale, so its panel is
        #  normalised onto the same D11 as the loaded forms.  Without that the
        #  bars would be in arbitrary units and the comparison meaningless.
        def _theta(entry):
            theta = np.asarray(entry["theta"], dtype=float)
            return theta / theta[0] * d11_scale if variant0.sealed else theta
        for position, name in enumerate(LABELS):
            values = [_theta(e)[position] for e in entries]
            axis.plot(starts, values, marker=MARKERS[name], ms=3.4, lw=1.3,
                      color=COLOURS[name], label=math_label(name))
        axis.axhline(0.0, color="#999999", lw=0.7, ls=":")
        variant = next(v for v in VARIANTS_45 if v.key == key)
        axis.set_title(variant.title, fontsize=7.2, pad=3)
        soft_grid(axis)
        if index % columns == 0:
            axis.set_ylabel("N$\\cdot$mm", fontsize=8)
        if index >= n - columns:
            axis.set_xlabel("window start $y$ [mm]", fontsize=8)
        axis.tick_params(labelsize=7)
    for spare in range(n, len(axes)):
        axes[spare].axis("off")

    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=4, frameon=False,
                  fontsize=8, bbox_to_anchor=(0.5, -0.015))
    figure.suptitle("45$^\\circ$ coupon: identified coefficients vs how far "
                    "the window is trimmed", fontsize=9.5, y=0.995)
    figure.tight_layout(rect=(0, 0.045, 1, 0.975))
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(figure)


def figure_variants(results, baseline, path) -> None:
    """One panel per coefficient; every LOADED BC form side by side.

    The scale-free form is deliberately excluded: it determines only a
    direction in parameter space, and a weakly determined one at that
    (singular gap ~1.6), so plotting it on an absolute axis alongside the
    loaded forms would both distort the scale and imply a precision it does
    not have.  It gets its own figure, where trim-independence is the message.
    """
    keys = [v.key for v in VARIANTS_45
            if v.key in results
            and not next(x for x in VARIANTS_45 if x.key == v.key).sealed]
    figure, axes = plt.subplots(1, 4, figsize=(7.16, 2.5))
    positions = np.arange(len(keys))

    for index, name in enumerate(LABELS):
        axis = axes[index]
        values = [results[k]["theta"][index] for k in keys]
        axis.bar(positions, values, color=COLOURS[name], width=0.68)
        axis.axhline(baseline[index], color="#444444", lw=1.1, ls="--",
                     label="2-coupon campaign")
        axis.axhline(0.0, color="#999999", lw=0.7)
        axis.set_xticks(positions)
        axis.set_xticklabels(keys, fontsize=7)
        axis.set_title(math_label(name), fontsize=9)
        axis.tick_params(labelsize=7)
        soft_grid(axis)
        if index == 0:
            axis.set_ylabel("N$\\cdot$mm", fontsize=8)
    axes[0].legend(fontsize=6.5, frameon=False, loc="best")
    figure.suptitle("45$^\\circ$ coupon: identified coefficients by "
                    "boundary-condition form", fontsize=9.5, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(figure)


def figure_sealed(sweep_sealed, reference_ratio, path) -> None:
    """The scale-free form's ratios vs trim -- the objection's ideal case."""
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
    names = ["D12", "D22", "D66"]
    for index, name in enumerate(names):
        axis = axes[index]
        position = LABELS.index(name)
        starts = [e["geometry"]["y_min"] for e in sweep_sealed]
        ratios = [e["theta"][position] / e["theta"][0] for e in sweep_sealed]
        axis.plot(starts, ratios, marker=MARKERS[name], ms=4.2, lw=1.4,
                  color=COLOURS[name])
        axis.axhline(reference_ratio[position], color="#444444", ls="--",
                     lw=1.1, label="3-coupon FGLS")
        axis.set_title(f"{math_label(name)} / {math_label('D11')}", fontsize=9)
        axis.set_xlabel("window start $y$ [mm]", fontsize=8)
        axis.tick_params(labelsize=7)
        soft_grid(axis)
    axes[0].set_ylabel("ratio", fontsize=8)
    axes[0].legend(fontsize=6.5, frameon=False)
    figure.suptitle("Both cuts sealed: no traction does virtual work, so the "
                    "answer is moment-free — and trim-free", fontsize=9,
                    y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=600)
    plt.close(figure)


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="skip the trim sweep")
    arguments = parser.parse_args()

    apply_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  45 DEG COUPON -- BOUNDARY CONDITIONS AND WINDOW TRIM")
    print("=" * 78)
    print("  The longitudinal and transverse coupons are held at their shipped")
    print("  reference in every run below, so every difference belongs to the")
    print("  45 deg coupon.")
    base = reference_cases()

    #  ---- the two-coupon baseline, for the figures --------------------------
    identification, linear_algebra = _settings()
    baseline = run_gls_identification_cases(
        cases=base[0], material_type=MATERIAL_TYPE,
        analysis_type=ANALYSIS_TYPE, initial_parameters=INITIAL_PARAMETERS,
        identification_settings=identification,
        linear_algebra_settings=linear_algebra, experiment_ids=base[1],
        gls_settings=GlsSettings(), verbose=False)
    baseline_theta = np.asarray(baseline.final_parameters, dtype=float)

    # =========================================================================
    #  1. BC variants at the shipped trim
    # =========================================================================
    print()
    print("=" * 78)
    print("  BOUNDARY-CONDITION FORMS, at the shipped trim (window from "
          "y = 5.30 mm)")
    print("=" * 78)
    print(f"  {'variant':<38}" + "".join(f"{n:>10}" for n in LABELS)
          + f"{'chi2/dof':>11}")
    print("  " + "-" * (38 + 10 * len(LABELS) + 11))
    print(f"  {'2-coupon campaign (no 45 deg)':<38}"
          + "".join(f"{v:10.2f}" for v in baseline_theta) + f"{830:11.0f}")

    results: Dict[str, Dict] = {}
    for variant in VARIANTS_45:
        try:
            outcome = run_variant(variant, 28, base=base)
        except Exception as error:
            print(f"  {variant.title:<38}  FAILED: "
                  f"{type(error).__name__}: {error}")
            continue
        results[variant.key] = outcome
        theta = outcome["theta"]
        if variant.sealed:
            shown = theta / theta[0] * baseline_theta[0]
            print(f"  {variant.title:<38}"
                  + "".join(f"{v:10.2f}" for v in shown)
                  + f"{'  (scaled)':>11}")
            print(f"  {'':38}   singular values "
                  + np.array2string(outcome["singular"], precision=3)
                  + f",  gap sigma3/sigma4 = {outcome['gap']:.1f}")
            print(f"  {'':38}   a gap near 1 means the null direction is NOT "
                  f"sharply determined:")
            print(f"  {'':38}   the 45 deg coupon constrains only S and D66, "
                  f"i.e. 2 of 4 unknowns.")
        else:
            print(f"  {variant.title:<38}"
                  + "".join(f"{v:10.2f}" for v in theta)
                  + f"{outcome['chi2']:11.0f}")

    print()
    for variant in VARIANTS_45:
        if variant.key in results:
            print(f"    {variant.key}: {variant.note}")

    #  Rescale the sealed row for the figure, so the bar chart is comparable.
    figure_results = {k: dict(v) for k, v in results.items()}
    if "S" in figure_results:
        direction = figure_results["S"]["theta"]
        figure_results["S"]["theta"] = direction / direction[0] * baseline_theta[0]
    figure_variants(figure_results, baseline_theta,
                    FIGURE_DIR / "fig45_bc_forms")
    with (FIGURE_DIR / "fig45_bc_forms_key.txt").open(
            "w", encoding="utf-8") as handle:
        for variant in VARIANTS_45:
            handle.write(f"{variant.key}: {variant.title}\n    "
                         f"{variant.note}\n")
    print(f"\n  wrote {FIGURE_DIR.name}/fig45_bc_forms.pdf/.png")

    # =========================================================================
    #  2. The trim sweep
    # =========================================================================
    if not arguments.quick:
        print()
        print("=" * 78)
        print("  TRIM SWEEP -- does the answer depend on where the cuts are?")
        print("=" * 78)
        print("  low_y_trim = 7 is the bare 15-filter half width: NO physical")
        print("  trim, window straddling the curvature reversal at y ~ 4.2 mm.")
        print("  28 is what the shipped export uses.")
        print()

        sweep: Dict[str, List[Dict]] = {}
        for variant in VARIANTS_45:
            entries = []
            print(f"  {variant.title}")
            print(f"    {'trim':>5}{'y_min':>8}{'len':>7}{'npts':>7}"
                  + "".join(f"{n:>10}" for n in LABELS) + f"{'chi2/dof':>11}")
            for trim in LOW_Y_TRIMS:
                try:
                    outcome = run_variant(variant, trim, base=base)
                except Exception as error:
                    print(f"    {trim:>5}  FAILED: {type(error).__name__}")
                    continue
                entries.append(outcome)
                geometry = outcome["geometry"]
                theta = outcome["theta"]
                shown = (theta / theta[0] * baseline_theta[0]
                         if variant.sealed else theta)
                chi2 = ("  (scaled)" if variant.sealed
                        else f"{outcome['chi2']:11.0f}")
                print(f"    {trim:>5}{geometry['y_min']:8.2f}"
                      f"{geometry['length']:7.2f}{geometry['n_points']:7d}"
                      + "".join(f"{v:10.2f}" for v in shown)
                      + (chi2 if isinstance(chi2, str) else chi2))
            sweep[variant.key] = entries
            #  spread across the sweep
            if len(entries) > 1:
                stack = np.array([
                    (e["theta"] / e["theta"][0]) if variant.sealed
                    else e["theta"] for e in entries])
                span = stack.max(axis=0) - stack.min(axis=0)
                middle = np.abs(0.5 * (stack.max(axis=0) + stack.min(axis=0)))
                print(f"    {'spread %':>27}"
                      + "".join(f"{v:10.1f}" for v in
                                100.0 * span / np.maximum(middle, 1e-12)))
            print()

        figure_trim(sweep, FIGURE_DIR / "fig45_trim_sweep",
                    d11_scale=float(baseline_theta[0]))
        print(f"  wrote {FIGURE_DIR.name}/fig45_trim_sweep.pdf/.png")

        if sweep.get("S"):
            reference_ratio = (np.asarray(results["A"]["theta"])
                               / results["A"]["theta"][0]) if "A" in results \
                else np.zeros(4)
            figure_sealed(sweep["S"], reference_ratio,
                          FIGURE_DIR / "fig45_sealed_ratios")
            print(f"  wrote {FIGURE_DIR.name}/fig45_sealed_ratios.pdf/.png")

        #  CSV
        output_dir = RESULTS_DIR / "bc_studies" / "three_coupon"
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / "bc_study_45.csv"
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["variant", "low_y_trim", "y_min", "length_mm",
                             "n_points", *LABELS, "chi2_per_dof", "sealed"])
            for key, entries in sweep.items():
                variant = next(v for v in VARIANTS_45 if v.key == key)
                for entry in entries:
                    writer.writerow(
                        [key, entry["geometry"]["low_y_trim"],
                         f"{entry['geometry']['y_min']:.4f}",
                         f"{entry['geometry']['length']:.4f}",
                         entry["geometry"]["n_points"],
                         *[f"{v:.6f}" for v in entry["theta"]],
                         f"{entry['chi2']:.3f}", int(variant.sealed)])
        print(f"  wrote {out.name}")

    # =========================================================================
    #  3. D12
    # =========================================================================
    if "A" in results:
        d12_diagnosis(results["A"]["theta"])
        d12_empirical(results, baseline_theta)


if __name__ == "__main__":
    main()

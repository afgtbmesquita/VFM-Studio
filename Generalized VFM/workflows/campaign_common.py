# =============================================================================
#  Generalized VFM / campaign_common.py
# -----------------------------------------------------------------------------
#  Shared setup for the longitudinal + transverse bending campaign.  Both the
#  real run (run_campaign_gls.py) and the known-answer benchmark
#  (validate_gls.py) build their cases through here, so the two cannot drift
#  apart in geometry, boundary conditions or sample area.
#
#  Each test contributes its 9 load steps as ONE experiment, which is the
#  grouping the GLS needs: load steps of one test share a dense cross-time
#  covariance block, separate tests are assumed independent.
#
#  ORIENTATION -- THE THING THAT MUST NOT BE WRONG
#  ---------------------------------------------------------------------------
#  Stacking is only meaningful if both datasets express their curvatures and
#  moments in ONE material frame.  They do, by construction:
#
#  The MATERIAL 1-AXIS RUNS ALONG THE LONGITUDINAL COUPON'S LENGTH.
#
#    * The LONGITUDINAL coupon bends along the material 1-direction, so it
#      identifies D11.  Its length runs along the lab +y axis, so the material
#      1-axis sits at +90 deg from lab +x: material_orientation_deg = 90.
#    * The TRANSVERSE coupon was cut at 90 deg to it, so it bends along the
#      material 2-direction and identifies D22.  In its own raw frame its
#      length also runs along +y, and the material 1-axis is perpendicular to
#      that, i.e. along +x: material_orientation_deg = 0.
#    * The pre-rotated TRANSVERSE export shares the longitudinal coupon's
#      frame by construction, so it carries the same 90 deg.
#
#  Note 180 deg and 0 deg are the SAME rotation of D (T(180) = I), which is why
#  shifting the whole frame by +90 leaves the raw transverse coupon at 0.
#
#  Trace the material direction to check the rotation lands the right way up.
#  Call A the plate direction that ran along the LONGITUDINAL coupon's length.
#  In the transverse coupon A runs across its width, i.e. along X_raw.  The
#  rotation sets X' = Y_raw and Y' = -X_raw, so X_raw = -Y': A ends up along
#  the fixed frame's y-axis -- the same axis it occupies in the longitudinal
#  test.  Both datasets therefore put A along y, so D22 is the A-direction
#  stiffness and D11 the other.
#
#  assert_orientation() verifies this from the shipped .mat rather than
#  trusting it, because a re-export that lost or double-applied the rotation is
#  the one silent failure that would corrupt the whole campaign.
#
#  TWO ROUTES TO THE SAME ANSWER
#  ---------------------------------------------------------------------------
#  Rotating the fields is not the only way to put both coupons in one material
#  frame, and it does not extend to a coupon cut at 45 deg, where the rotation
#  MIXES kappa_xx, kappa_yy and kappa_xy instead of merely permuting them.  The
#  alternative leaves every measurement in the frame it was taken in and rotates
#  the CONSTITUTIVE MATRIX instead:
#
#      (A)  TRANSVERSE      fields pre-rotated  +  D unrotated   (angle 0)
#      (B)  TRANSVERSE_RAW  fields raw          +  D rotated     (angle 90)
#
#  Internal virtual work is a frame-invariant scalar, so the two give the same
#  theta -- and doing BOTH would rotate twice.  validate_rotation.py runs the
#  transverse experiment each way and requires them to agree.
#
#  Nothing in Longitudinal_bending/ or Transverse_bending/VFM_export/ is
#  modified; route (B) reads a separate generated directory.
# =============================================================================

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from scipy.io import loadmat

from vfm_data_io import load_boundary_conditions, load_point_measurements
from vfm_identification import IdentificationCase
from vfm_mesh import (
    build_mesh,
    map_points_to_mesh,
    prune_mesh_by_measurement_points,
    subset_measurements_to_mapped_mesh,
)
from vfm_user_inputs import Bounds, MeshSettings

# Stable repository locations.  Workflow modules live one level below the
# solver root.  Coupon exports live in Pre_Conference_Studies (or, in the
# original layout, as siblings of Generalized VFM).  Walk up from this file
# so the extra VFM Studio folder does not break the campaign.
HERE = Path(__file__).resolve().parent.parent


def _find_campaign() -> Path:
    """Folder that contains Longitudinal_bending/, Transverse_bending/, ..."""
    marker = Path("Longitudinal_bending") / "VFM_export"
    for parent in Path(__file__).resolve().parents:
        nested = parent / "Pre_Conference_Studies"
        if (nested / marker).is_dir():
            return nested
        if (parent / marker).is_dir():
            return parent
    return HERE.parent.parent / "Pre_Conference_Studies"


CAMPAIGN = _find_campaign()
RESULTS_DIR = HERE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PROJECTS_DIR = HERE / "projects"

LONG_DIR = CAMPAIGN / "Longitudinal_bending" / "VFM_export"
TRANS_DIR = CAMPAIGN / "Transverse_bending" / "VFM_export"
#  Raw-frame twin of TRANS_DIR, generated by make_raw_frame_transverse.py.  The
#  shipped export is never modified.
TRANS_RAW_DIR = CAMPAIGN / "Transverse_bending" / "VFM_export_raw_frame"
#  The off-axis coupon.  Route (B) ONLY -- there is no pre-rotated twin and
#  there cannot be one, because a 45 deg rotation MIXES the curvature channels
#  instead of permuting them.  See FORTYFIVE below.
FORTYFIVE_DIR = CAMPAIGN / "45deg_bending" / "VFM_export"
#  The TWIST coupon.  Route (B) as well, and the only test in the campaign that
#  is not loaded in bending: its machine channel is dTx and its dominant
#  curvature is kappa_xy.  See TWIST below.
TWIST_DIR = CAMPAIGN / "Twist" / "VFM_export"

ANALYSIS_TYPE = "out_of_plane"
MATERIAL_TYPE = "orthotropic"

#  Micro-CT laminate thickness, longitudinal coupon.  Used only for the derived
#  engineering-constant readout; the transverse coupon was never scanned.
THICKNESS_MM = 0.2564

#  Starting point for the outer loop.  Near the two measured values so the
#  benchmark exercises the solver in the regime the real data occupies.
INITIAL_PARAMETERS = [65.0, 15.0, 175.0, 40.0]

#  Reference numbers from the earlier single-test and stacked runs.  Used only
#  for the side-by-side comparison in the report; they are not inputs.
REFERENCE_RESULTS = {
    "longitudinal alone (OLS)": [4.130392, 0.515324, 173.033537, -9.924532],
    "transverse alone (OLS)": [63.410597, 0.117825, -0.966637, -2.660349],
    "stacked naive (OLS)": [32.273549, -0.358909, 85.455622, -7.864845],
    "stacked inverse-variance": [60.326214, -0.162497, 178.725066, -6.474808],
}


@dataclass(frozen=True)
class TestSpec:
    """One physical test: where its data, BC file and .mat files live.

    ``length_axis`` describes the frame the .mat are actually IN, so
    assert_orientation() can check the shipped data rather than trust it.
    ``material_orientation_deg`` says how that frame relates to the material
    frame: the angle from the data's +x axis to the material 1-axis, CCW
    positive.  Pre-rotated data is already in the material frame and carries
    0.0; raw-frame data carries the coupon's real cut angle.

    ``min_curvature_dominance`` is the factor by which the length-conjugate
    curvature must exceed the other one in assert_orientation().  It is 5.0 for
    an AXIS-ALIGNED coupon, where anything less would mean the data are not in
    the frame this campaign assumes.  An OFF-AXIS coupon legitimately fails
    that test: at 45 deg the anticlastic ratio is -(S-D66)/(S+D66), which for
    any physical S, D66 > 0 lies strictly inside (-1, 1) and here measures
    -0.38, so the dominance is ~2.6x BY CONSTRUCTION and is not a symptom of a
    wrong frame.  Lowering it is a statement about the mechanics, not a
    loosened check -- ``max_anticlastic_ratio`` replaces the lost strictness
    with a bound that IS physical.

    ``max_anticlastic_ratio`` bounds |kappa_other / kappa_dominant|.  For an
    off-axis coupon with traction-free lateral edges that ratio equals
    (S-D66)/(S+D66), so requiring |ratio| < 1 is exactly the statement that S
    and D66 are both positive -- a positive-definiteness requirement on D,
    checked against the measured fields rather than assumed.  Left at +inf for
    the axis-aligned coupons, where it is vacuous.

    ``dominant_channel`` says WHICH curvature channel the loading drives, and
    therefore which one assert_orientation() should be checking.

      'normal'  (default)  a BENDING test.  The dominant channel is the normal
                curvature conjugate to ``length_axis`` -- kappa_yy for a
                y-length coupon, kappa_xx for an x-length one -- and the other
                normal curvature is the anticlastic response.  This is the case
                for all three bending coupons and the default preserves their
                behaviour exactly.
      'twist'   a TWIST test.  The dominant channel is kappa_xy and BOTH normal
                curvatures are secondary.  Without this branch the check would
                compare kappa_yy against kappa_xx, find a healthy-looking ratio,
                and cheerfully certify a twist test as a bending test -- which
                on the twist coupon it does: kappa_yy is 4.5x kappa_xx there,
                while both are dwarfed by a kappa_xy that the check never looks
                at.  A green light for the wrong reason is worse than no check.
    """
    name: str
    directory: Path
    bc_file: Path
    mat_glob: str
    length_axis: str        # 'y' longitudinal, 'x' transverse (fixed frame)
    identifies: str         # the parameter this test alone constrains
    experiment_id: int
    material_orientation_deg: float = 0.0
    min_curvature_dominance: float = 5.0
    max_anticlastic_ratio: float = float("inf")
    dominant_channel: str = "normal"   # 'normal' (bending) | 'twist'


LONGITUDINAL = TestSpec(
    name="longitudinal",
    directory=LONG_DIR,
    bc_file=LONG_DIR / "BC_Sample_6.py",
    mat_glob="Sample_6_step*.mat",
    length_axis="y",
    identifies="D11",
    experiment_id=1,
    #  The material 1-axis runs ALONG this coupon's length, which is the lab
    #  +y axis -- hence +90 deg.  So the direction this coupon bends in is
    #  material direction 1, and it identifies D11.
    material_orientation_deg=90.0,
)

#  The shipped transverse export, whose fields MATLAB already rotated +90 deg
#  into the material frame.  Route (A).  Kept as the reference the raw-frame
#  route is validated against; see validate_rotation.py.
TRANSVERSE = TestSpec(
    name="transverse",
    directory=TRANS_DIR,
    bc_file=TRANS_DIR / "BC_Transverse.py",
    mat_glob="Transverse_step*.mat",
    length_axis="x",
    identifies="D22",
    experiment_id=2,
    #  The export rotated these fields into the LONGITUDINAL coupon's frame, so
    #  they share its orientation.  This coupon bends across the material
    #  1-direction, i.e. along material 2 -- it identifies D22.
    material_orientation_deg=90.0,
)

#  The same nine load steps with the export's rotation undone, so the fields sit
#  in the frame the coupon was actually measured in.  Route (B): the material
#  1-axis is at +90 deg from this frame's +x, and D is rotated instead of the
#  fields.  Built by make_raw_frame_transverse.py.
TRANSVERSE_RAW = TestSpec(
    name="transverse-raw",
    directory=TRANS_RAW_DIR,
    bc_file=TRANS_RAW_DIR / "BC_Transverse_raw.py",
    mat_glob="Transverse_raw_step*.mat",
    length_axis="y",
    identifies="D22",
    experiment_id=2,
    #  This coupon's length is along its own +y, and the material 1-axis is
    #  PERPENDICULAR to it (it is the 90 deg cut), i.e. along +x -- hence 0.
    #  It therefore bends in material direction 2 and identifies D22.
    material_orientation_deg=0.0,
)

#  The 45 deg coupon.  Its fields are in the SPECIMEN frame and were never
#  rotated, so route (B) is the ONLY route open to it -- which is why the
#  rotate-D machinery had to exist before this test could be added at all.
#
#  ORIENTATION.  Its length runs along its own +y, i.e. 90 deg from +x, and the
#  material 1-axis lies at 45 deg to the length, so the 1-axis sits at
#  90 - 45 = 45 deg from +x.
#
#  WHY 45 AND NOT 135.  Both are geometrically possible; the coupon could have
#  been cut with either sense.  Everything EVEN in the angle is identical for
#  the two -- the effective bending stiffness, the anticlastic ratio, and hence
#  D66 -- because T(135) differs from T(45) only in the sign of its third row
#  and column.  The ONLY discriminator is the sign of the bend-twist coupling,
#  and the measured kappa_xy/kappa_yy is negative at every one of the nine load
#  steps (-0.009 to -0.047), which is the sign T(45) predicts for D66 > 0.
#
#  That evidence is WEAK on its own, because the fixture restrains the twist
#  and suppresses the ratio ~15x below what a free coupon would show.  It does
#  not need to be strong: run_campaign_45.py --orientation-check identifies the
#  whole campaign BOTH ways and reports the difference, which is what actually
#  settles whether the ambiguity matters for the reported theta.
FORTYFIVE = TestSpec(
    name="45deg",
    directory=FORTYFIVE_DIR,
    bc_file=FORTYFIVE_DIR / "BC_FortyFive.py",
    mat_glob="FortyFive_step*.mat",
    length_axis="y",
    identifies="D66",
    experiment_id=3,
    material_orientation_deg=45.0,
    #  Off-axis: the dominance is ~2.6x by construction, not by accident.
    min_curvature_dominance=2.0,
    #  |kappa_xx/kappa_yy| = (S-D66)/(S+D66) < 1 iff S, D66 > 0.
    max_anticlastic_ratio=1.0,
)

#  The TWIST coupon -- the only test in the campaign that is not bending.
#
#  WHAT IT IS.  An edge twist of a rectangular coupon: the commanded axis is
#  DeltaTheta_x (clean 2 deg increments), the responding wrench channel is dTx,
#  and the dominant curvature is kappa_xy at 5.4x the next largest.  dTy stays
#  at 18 % of dTx and dTz, though larger, is a torque about the plate NORMAL
#  and does no work against an out-of-plane virtual field.
#
#  WHY IT WAS ADDED.  It is the first DIRECT measurement of D66 in the
#  campaign.  The two-coupon campaign returns D66 = -6 to -10 N mm -- i.e.
#  nothing.  The 45 deg coupon recovers D66 only INDIRECTLY, from the
#  anticlastic ratio, because the yoke suppresses its twist ~15x.  This coupon
#  loads the twist channel head-on, and its weaknesses are largely DISJOINT
#  from the 45 deg route's: the 45 deg answer is a ratio of curvatures in which
#  the moment cancels (so it is free of the width, the machine calibration and
#  the moment gradient, but depends on the 45/135 cut sense), while this one
#  scales as 1/width but does not care about the cut angle at all.  Comparing
#  them is a real test of the campaign -- see studies/twist_four_coupon.
#
#  ORIENTATION.  The material cut angle was NEVER RECORDED for this coupon
#  (material_angle_deg is NaN in its .h5).  0.0 is carried, matching the
#  transverse cut.  That choice costs less than it looks: D66 is INVARIANT
#  under 0 <-> 90 -- T(90) permutes 1 <-> 2 and fixes D66, with D16 = D26 = 0
#  either way -- so the coupon's primary contribution does not depend on it.
#  Only the coupling of the secondary kappa_yy (-0.19 of kappa_xy) into
#  D12/D22 does.  studies/twist_three_coupon identifies it BOTH ways and
#  reports the spread, exactly as run_campaign_45.py --orientation-check does
#  for the 45/135 ambiguity.
#
#  ONLY TWO LOAD STEPS.  Deliberate.  Beyond step 4 the twist field stops being
#  self-similar: the kappa_xy zero crossing migrates 5 mm along the specimen
#  across the ramp, so a window fixed in space measures a different part of the
#  coupon at each step.  The FGLS cross-time block for experiment 4 is
#  therefore 2 x 2 -- thin, and stated as such wherever it is used.
TWIST = TestSpec(
    name="twist",
    directory=TWIST_DIR,
    bc_file=TWIST_DIR / "BC_Twist.py",
    mat_glob="Twist_step*.mat",
    length_axis="y",
    identifies="D66",
    experiment_id=4,
    material_orientation_deg=0.0,
    #  Not a bending test: the check must look at kappa_xy, not kappa_yy.
    dominant_channel="twist",
    #  Measured 5.4x over max(|kappa_xx|, |kappa_yy|) at the analysed step.
    #  3.0 leaves headroom without making the check vacuous.
    min_curvature_dominance=3.0,
    #  |kappa_yy / kappa_xy| = 0.19 measured.  1.0 is the same
    #  positive-definiteness statement the 45 deg coupon carries: a secondary
    #  channel larger than the driven one would mean the loading, the frame or
    #  the moment sign is wrong.
    max_anticlastic_ratio=1.0,
)

TESTS = (LONGITUDINAL, TRANSVERSE)
TESTS_RAW_FRAME = (LONGITUDINAL, TRANSVERSE_RAW)
#  The three-coupon campaign.  TESTS is deliberately left untouched so every
#  pre-existing run, figure and project reproduces bit-identically.
#
#  TESTS_WITH_45      route (A) for the transverse coupon: its fields were
#                     pre-rotated +90 deg at MATLAB export time, so in the .mat
#                     its length lies along +x.
#  TESTS_WITH_45_RAW  route (B) THROUGHOUT: every coupon in the frame it was
#                     actually measured in, and ONLY D rotated.  All three
#                     coupons then sit the way they physically sat in the
#                     fixture -- length along +y, aligned with the machine's y
#                     axis -- and the material 1-axis is at 90 deg
#                     (longitudinal), 0 deg (transverse, cut across it) and
#                     45 deg (off-axis).
#
#  The two give the SAME theta -- internal virtual work is a frame-invariant
#  scalar -- and validate_rotation.py asserts it.  PREFER _RAW: it is the only
#  one whose geometry is the physical geometry, which matters as soon as
#  anything is plotted, and it is the route the 45 deg coupon must use anyway
#  (a 45 deg rotation mixes the curvature channels instead of permuting them).
TESTS_WITH_45 = (LONGITUDINAL, TRANSVERSE, FORTYFIVE)
TESTS_WITH_45_RAW = (LONGITUDINAL, TRANSVERSE_RAW, FORTYFIVE)

#  The twist campaigns.  Every tuple ABOVE is left exactly as it was so that
#  every pre-existing run, figure and project reproduces bit-identically; these
#  are additions, not edits.
#
#  TESTS_WITH_TWIST[_RAW]     the two bending coupons plus the twist coupon.
#                             studies/twist_three_coupon.
#  TESTS_WITH_45_TWIST[_RAW]  all four.  studies/twist_four_coupon, where the
#                             two independent D66 routes meet.
#
#  PREFER the _RAW variants, for the reason stated above: route (B) throughout
#  puts every coupon in the frame it was actually measured in.  The twist
#  coupon's fields were never rotated, so it is route (B) by construction and
#  mixing it with a route (A) transverse coupon would leave the campaign half
#  in one convention and half in the other.  The two give the same theta --
#  internal virtual work is a frame-invariant scalar -- and validate_rotation.py
#  asserts it; _RAW is the one whose geometry is the physical geometry.
TESTS_WITH_TWIST = (LONGITUDINAL, TRANSVERSE, TWIST)
TESTS_WITH_TWIST_RAW = (LONGITUDINAL, TRANSVERSE_RAW, TWIST)
TESTS_WITH_45_TWIST = (LONGITUDINAL, TRANSVERSE, FORTYFIVE, TWIST)
TESTS_WITH_45_TWIST_RAW = (LONGITUDINAL, TRANSVERSE_RAW, FORTYFIVE, TWIST)


# =============================================================================
#  Boundary-condition form -- which cuts do external virtual work
# -----------------------------------------------------------------------------
#  BC_BOTH_CUTS   w*, the tangential slope and the twist are zero on BOTH
#                 loaded cuts, so the shear and the twisting moment do no
#                 virtual work -- but the normal rotation is FREE at both ends,
#                 so the bending moment works at both.  Two EXTERNAL groups
#                 carrying F = [-M, +M].
#
#  BC_TOP_ONLY    as above, PLUS the normal rotation prescribed to zero on the
#                 BOTTOM cut, so that cut's UNKNOWN reaction moment does no
#                 virtual work either.  Only the TOP region does external work:
#                 ONE EXTERNAL group carrying that cut's measured moment as a
#                 per-step value.
#
#  BC_TOP_ONLY annihilates EVERY unknown traction, which is what the VFM
#  requires when one end is an uninstrumented reaction.  See the README for the
#  side-by-side on real data.
# =============================================================================

BC_BOTH_CUTS = "both-cuts"
BC_TOP_ONLY = "top-only"
BC_FORMS = (BC_BOTH_CUTS, BC_TOP_ONLY)

#  Both coupons are loaded on their top/bottom cuts once each is in its own
#  measurement frame, so the normal rotation there is W_Y for both.
_NORMAL_ROTATION_DOF = "W_Y"


def load_bc_module(bc_file: Path):
    """Import a BC file so MESH_SETTINGS / SOLVER_SETTINGS are readable."""
    spec = importlib.util.spec_from_file_location(bc_file.stem, bc_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mesh_settings_for(bc_module) -> MeshSettings:
    bounds = bc_module.MESH_SETTINGS["bounds"]
    return MeshSettings(
        number_of_elements_x=bc_module.MESH_SETTINGS["number_of_elements_x"],
        number_of_elements_y=bc_module.MESH_SETTINGS["number_of_elements_y"],
        bounds=Bounds(**bounds),
    )


def mat_files(test: TestSpec) -> List[Path]:
    paths = sorted((test.directory / "mat").glob(test.mat_glob))
    if not paths:
        raise SystemExit(
            f"No .mat files for the {test.name} test in "
            f"{test.directory / 'mat'}; run its export_vfm_dataset.py first."
        )
    return paths


def assert_orientation(test: TestSpec, verbose: bool = True) -> dict:
    """Verify from the data that this test sits where the frame says it does.

    GEOMETRY   the measurement window must be longer along the declared length
               axis than across it.
    MECHANICS  the dominant mean curvature must be the one the LOADING drives,
               and by a wide margin.

               For a BENDING test (dominant_channel='normal', the default) that
               is the normal curvature conjugate to the length axis -- kappa_yy
               (SK2grid) for a y-length test, kappa_xx (SK1grid) for an x-length
               one -- and the competitor is the other normal curvature.

               For a TWIST test (dominant_channel='twist') it is kappa_xy
               (SK3grid), and the competitor is whichever NORMAL curvature is
               larger.  That branch is not cosmetic: run on the twist coupon,
               the bending form of this check would compare kappa_yy against
               kappa_xx, find 4.5x, and pass -- certifying a twist test as a
               y-bending test while never once looking at the channel that
               actually carries the load.

               The twist coupon's window is very nearly square (8.90 x 8.07 mm),
               so the GEOMETRY test above is nearly a tie for it.  That is fine:
               it is there to catch a transposed frame, and 8.07 > 8.90 would be
               a 90 deg error, not a rounding one.
    """
    last = mat_files(test)[-1]
    data = loadmat(last, squeeze_me=True)
    nodes = np.asarray(data["nodes"], dtype=float)
    span_x = float(nodes[:, 0].max() - nodes[:, 0].min())
    span_y = float(nodes[:, 1].max() - nodes[:, 1].min())
    k_xx = float(np.mean(np.asarray(data["SK1grid"], dtype=float)))
    k_yy = float(np.mean(np.asarray(data["SK2grid"], dtype=float)))
    k_xy = float(np.mean(np.asarray(data["SK3grid"], dtype=float)))

    long_is_x = test.length_axis == "x"
    geometry_ok = (span_x > span_y) if long_is_x else (span_y > span_x)
    is_twist = test.dominant_channel == "twist"
    if is_twist:
        dominant = k_xy
        other = k_yy if abs(k_yy) >= abs(k_xx) else k_xx
        dominant_name, other_name = "xy", ("yy" if abs(k_yy) >= abs(k_xx)
                                           else "xx")
    else:
        dominant, other = (k_xx, k_yy) if long_is_x else (k_yy, k_xx)
        dominant_name = "xx" if long_is_x else "yy"
        other_name = "yy" if long_is_x else "xx"
    ratio = abs(other) / abs(dominant)
    mechanics_ok = abs(dominant) > test.min_curvature_dominance * abs(other)
    #  Off-axis coupons only -- a positive-definiteness statement, not a frame
    #  statement.  See TestSpec.max_anticlastic_ratio.
    anticlastic_ok = ratio < test.max_anticlastic_ratio

    if verbose:
        print(f"  {test.name:<13} span x {span_x:6.2f} mm, y {span_y:6.2f} mm  "
              f"-> length along {test.length_axis}  "
              f"[{'OK' if geometry_ok else 'FAIL'}]")
        #  The bending branch prints EXACTLY what it always printed, so every
        #  captured *_output.txt in results/ still diffs clean.  Only the twist
        #  branch is new.
        if is_twist:
            print(f"  {'':<13} mean kappa_xx {k_xx:+.3e}, kappa_yy {k_yy:+.3e}, "
                  f"kappa_xy {k_xy:+.3e}  -> dominant |{dominant_name}| is "
                  f"{abs(dominant) / abs(other):5.1f}x |{other_name}| "
                  f"(need > {test.min_curvature_dominance:.1f}x)  "
                  f"[{'OK' if mechanics_ok else 'FAIL'}]")
            print(f"  {'':<13} secondary-bending ratio {-ratio:+.4f}  -> |.| < "
                  f"{test.max_anticlastic_ratio:.1f} is required for the "
                  f"response to be twist-dominated  "
                  f"[{'OK' if anticlastic_ok else 'FAIL'}]")
        else:
            print(f"  {'':<13} mean kappa_xx {k_xx:+.3e}, kappa_yy {k_yy:+.3e}  "
                  f"-> dominant |{dominant_name}| is "
                  f"{abs(dominant) / abs(other):5.1f}x the other "
                  f"(need > {test.min_curvature_dominance:.1f}x)  "
                  f"[{'OK' if mechanics_ok else 'FAIL'}]")
            if np.isfinite(test.max_anticlastic_ratio):
                print(f"  {'':<13} anticlastic ratio {-ratio:+.4f}  -> |.| < "
                      f"{test.max_anticlastic_ratio:.1f} is required for S, "
                      f"D66 > 0 at {test.material_orientation_deg:.0f} deg  "
                      f"[{'OK' if anticlastic_ok else 'FAIL'}]")

    if not geometry_ok:
        raise SystemExit(
            f"ORIENTATION CHECK FAILED for the {test.name} test: its length "
            f"should run along {test.length_axis}, but the window spans "
            f"{span_x:.2f} mm in x and {span_y:.2f} mm in y.  The .mat are not "
            "in the frame this campaign assumes."
        )
    if not mechanics_ok:
        expected = ("kappa_xy (this is declared a TWIST test)" if is_twist
                    else "the one conjugate to its length axis")
        raise SystemExit(
            f"ORIENTATION CHECK FAILED for the {test.name} test: the dominant "
            f"curvature is not {expected} "
            f"(kappa_xx = {k_xx:.3e}, kappa_yy = {k_yy:.3e}, "
            f"kappa_xy = {k_xy:.3e})."
        )
    if not anticlastic_ok:
        if is_twist:
            raise SystemExit(
                f"ORIENTATION CHECK FAILED for the {test.name} test: the "
                f"secondary-bending ratio |kappa_normal/kappa_xy| = "
                f"{ratio:.3f} is not below {test.max_anticlastic_ratio:.1f}.  "
                f"A normal curvature larger than the driven twist means this "
                f"is not the twist-dominated response the BC file assumes -- "
                f"check the window, the machine channel (dTx, not dTy) or the "
                f"curvature convention."
            )
        raise SystemExit(
            f"ORIENTATION CHECK FAILED for the {test.name} test: the "
            f"anticlastic ratio |kappa_other/kappa_dominant| = {ratio:.3f} is "
            f"not below {test.max_anticlastic_ratio:.1f}.  At "
            f"{test.material_orientation_deg:.0f} deg with traction-free "
            f"lateral edges that ratio is (S-D66)/(S+D66), so this says D is "
            f"NOT positive definite -- the frame, the moment sign or the "
            f"curvature convention is wrong."
        )

    return {"span_x": span_x, "span_y": span_y, "k_xx": k_xx, "k_yy": k_yy,
            "k_xy": k_xy, "dominant_channel": test.dominant_channel,
            "anticlastic_ratio": -ratio}


def build_cases(test: TestSpec,
                paths: Optional[Sequence[Path]] = None,
                verbose: bool = True,
                bc_form: str = BC_BOTH_CUTS) -> List[IdentificationCase]:
    """IdentificationCases for one test, each with that test's own BC + mesh.

    ``bc_form`` selects which loaded cuts do external work -- see BC_FORMS.
    """
    if bc_form not in BC_FORMS:
        raise ValueError(f"bc_form must be one of {BC_FORMS}; got {bc_form!r}.")
    bc_module = load_bc_module(test.bc_file)
    bc = load_boundary_conditions(test.bc_file)
    mesh_settings = mesh_settings_for(bc_module)
    sample_area = float(bc_module.SOLVER_SETTINGS["sample_area_total"])

    cases = []
    for path in (paths if paths is not None else mat_files(test)):
        measurements = load_point_measurements(
            path, analysis_type=ANALYSIS_TYPE, total_sample_area=sample_area)
        mesh_pkg = build_mesh(mesh_settings, analysis_type=ANALYSIS_TYPE)

        mapping = map_points_to_mesh(measurements.points, mesh_pkg)
        if mapping.n_unmapped:
            n_before = int(mapping.mapped_mask.size)
            print(
                f"  {path.name}: {mapping.n_unmapped} of {n_before} "
                "measurement points fall outside the virtual mesh and will "
                "be excluded from the identification."
            )
            measurements, mapping = subset_measurements_to_mapped_mesh(
                measurements, mesh_pkg, mapping)

        mesh_pkg, _ = prune_mesh_by_measurement_points(
            mesh_pkg, measurements.points)
        cases.append(IdentificationCase(
            measurements=measurements,
            mesh_pkg=mesh_pkg,
            boundary_conditions=bc,
            label=f"{test.name}:{path.stem}",
            data_file=path,
            material_orientation_deg=test.material_orientation_deg,
        ))
        if bc_form == BC_TOP_ONLY:
            cases[-1] = apply_top_only_form(cases[-1])

    if verbose:
        print(f"  {test.name:<13} {len(cases)} load steps, "
              f"{cases[0].measurements.points.shape[0]} points each, "
              f"BC = {test.bc_file.name}, "
              f"mesh {mesh_settings.number_of_elements_x}x"
              f"{mesh_settings.number_of_elements_y}, "
              f"area {sample_area:.2f} mm^2, "
              f"material frame {test.material_orientation_deg:+.1f} deg, "
              f"BC form {bc_form} "
              f"({len(cases[0].boundary_conditions.external)} external "
              f"group(s))")
    return cases


def build_campaign(tests: Sequence[TestSpec] = TESTS,
                   paths_by_test=None,
                   verbose: bool = True,
                   bc_form: str = BC_BOTH_CUTS):
    """All cases plus the experiment_ids the GLS needs.

    Returns (cases, experiment_ids, counts_by_test_name).
    """
    cases: List[IdentificationCase] = []
    experiment_ids: List[int] = []
    counts = {}
    for test in tests:
        paths = None if paths_by_test is None else paths_by_test.get(test.name)
        test_cases = build_cases(test, paths=paths, verbose=verbose,
                                 bc_form=bc_form)
        cases.extend(test_cases)
        experiment_ids.extend([test.experiment_id] * len(test_cases))
        counts[test.name] = len(test_cases)
    return cases, experiment_ids, counts


def apply_top_only_form(case):
    """Annihilate the bottom cut's unknown moment; load the top cut only.

    Returns a new IdentificationCase; the input is not modified.
    """
    import copy
    import dataclasses

    bc = copy.deepcopy(case.boundary_conditions)
    bc.prescribed = list(bc.prescribed) + [{
        "label": "bottom  dw*/dn = 0  (unknown reaction moment)",
        "dof": _NORMAL_ROTATION_DOF,
        "edge": "bottom",
        "value": 0.0,
    }]

    force = np.asarray(loadmat(case.data_file, squeeze_me=True)["F"],
                       dtype=float).ravel()
    kept = None
    for index, entry in enumerate(bc.external):
        if str(entry.get("edge")) == "top":
            kept = {**dict(entry), "value": float(force[index])}
            break
    if kept is None:
        raise SystemExit(
            f"{case.label}: no EXTERNAL group on the top edge to keep; the "
            f"top-only form assumes the shipped two-group layout.")
    bc.external = [kept]

    variant = dataclasses.replace(case)
    variant.boundary_conditions = bc
    #  The surviving moment is now declared per step, so the table must win
    #  over the .mat F, which still carries two entries.
    variant.applied_load_source = "manual"
    return variant


# =============================================================================
#  Reporting helpers
# =============================================================================

def report_parameters(labels, values, title: str, truth=None,
                      std=None) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)
    for index, (label, value) in enumerate(zip(labels, values)):
        line = f"  {label:>6} = {value:12.4f} N.mm"
        if std is not None:
            line += f" +/- {std[index]:8.4f}"
        if truth is not None:
            true_value = dict(
                zip(["D11", "D12", "D22", "D66"], truth)).get(label)
            if true_value is not None:
                error = 100.0 * abs(value - true_value) / abs(true_value)
                line += f"   (true {true_value:8.2f},  err {error:7.1f} %)"
        print(line)


def derived_constants(labels, values) -> None:
    lookup = dict(zip(labels, values))
    d11 = lookup.get("D11")
    d12 = lookup.get("D12")
    d22 = lookup.get("D22", lookup.get("D11"))
    if not (d11 and d22 and d11 > 0 and d22 > 0):
        return
    #  Orthotropic plate: D11 = E1 t^3 / (12 (1 - nu12 nu21)), and
    #  nu12 nu21 = D12^2 / (D11 D22).
    coupling = (d12 ** 2) / (d11 * d22) if d12 else 0.0
    thickness_cubed = THICKNESS_MM ** 3
    e1 = 12.0 * d11 * (1.0 - coupling) / thickness_cubed / 1000.0
    e2 = 12.0 * d22 * (1.0 - coupling) / thickness_cubed / 1000.0
    nu12 = d12 / d22 if d22 else float("nan")
    print(f"  derived at t = {THICKNESS_MM} mm (micro-CT, longitudinal coupon):")
    print(f"    E1_flex = {e1:6.1f} GPa   E2_flex = {e2:6.1f} GPa   "
          f"nu12 = {nu12:.4f}   nu12*nu21 = {coupling:.4f}")

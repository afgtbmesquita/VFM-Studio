# =============================================================================
#  plate / vfm_user_inputs.py
# -----------------------------------------------------------------------------
#  Plate-specific user inputs.  This is the ONLY file you need to edit to
#  change the inputs of a plate VFM run.
#
#  The plate VFM run is parameterised along two orthogonal axes:
#
#    * ``material_type``  -- constitutive law.  ``'iso'`` is supported for
#                            all analysis types.  ``'orthotropic'`` and
#                            ``'anisotropic'`` are supported for
#                            ``'in_plane'`` and ``'out_of_plane'`` only.
#
#    * ``analysis_type``  -- physics scope.  One of:
#
#         'in_plane'      -- A matrix only (3x3).  Solves for in-plane
#                            stiffness; mesh has 2 DOFs/node ``[u, v]``;
#                            Lagrangian Q1 shape functions only.
#                            Iso unknowns: ``[A11, A12]``.
#                            Orthotropic unknowns:
#                            ``[A11, A12, A22, A66]``.
#                            Anisotropic unknowns:
#                            ``[A11, A12, A22, A66, A16, A26]``.
#
#         'out_of_plane'  -- D matrix only (3x3).  Solves for bending
#                            stiffness; mesh has 4 DOFs/node
#                            ``[w, w_x, w_y, w_xy]``; BFS Hermite-16 shape
#                            functions only.  Iso unknowns: ``[D11, D12]``.
#                            Orthotropic unknowns:
#                            ``[D11, D12, D22, D66]``.
#                            Anisotropic unknowns:
#                            ``[D11, D12, D22, D66, D16, D26]``.
#
#         'both'          -- Full 6x6 ABD.  Original behaviour: mesh has
#                            6 DOFs/node ``[u, v, w, w_x, w_y, w_xy]`` with
#                            composite Q1 + BFS elements.  Iso unknowns:
#                            ``[A11, A12, D11, D12]``.  Orthotropic
#                            and anisotropic ``both`` are intentionally
#                            not supported.
#
#  ``initial_parameters`` must have the right length for the selected
#  (material_type, analysis_type) pair; the dataclass validates this in
#  ``__post_init__``.
#
#  ``data_files`` points to synthetic placeholders that do NOT exist on
#  disk by default -- replace with your real plate dataset.  The data
#  loader will raise a focused error if the required strain / curvature
#  fields for the selected analysis type are missing.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
#  Analysis-type bookkeeping
# -----------------------------------------------------------------------------
#  Single source of truth for the per-analysis DOF count.  Used by the mesh
#  builder, the GUI, and the parameter-length validation below.
# =============================================================================

ANALYSIS_TYPES: Tuple[str, ...] = ("in_plane", "out_of_plane", "both")

DOF_PER_NODE_BY_ANALYSIS: Dict[str, int] = {
    "in_plane":     2,
    "out_of_plane": 4,
    "both":         6,
}

# Parameter labels (and therefore expected count) per
# (material_type, analysis_type) pair.  Single source of truth for the
# run-setup GUI and ``UserInputs.__post_init__`` length checks.
PARAMETER_LABELS_BY_MODEL: Dict[Tuple[str, str], Tuple[str, ...]] = {
    ("iso", "in_plane"):     ("A11", "A12"),
    ("iso", "out_of_plane"): ("D11", "D12"),
    ("iso", "both"):         ("A11", "A12", "D11", "D12"),
    ("orthotropic", "in_plane"):     ("A11", "A12", "A22", "A66"),
    ("orthotropic", "out_of_plane"): ("D11", "D12", "D22", "D66"),
    ("anisotropic", "in_plane"):     ("A11", "A12", "A22", "A66", "A16", "A26"),
    ("anisotropic", "out_of_plane"): ("D11", "D12", "D22", "D66", "D16", "D26"),
}


def dof_per_node_for_analysis(analysis_type: str) -> int:
    """Return the number of DOFs per node for the given analysis type."""
    key = str(analysis_type).strip().lower()
    if key not in DOF_PER_NODE_BY_ANALYSIS:
        raise ValueError(
            f"Unsupported analysis_type {analysis_type!r}; expected one of "
            f"{ANALYSIS_TYPES}."
        )
    return DOF_PER_NODE_BY_ANALYSIS[key]


def parameter_labels_for(material_type: str,
                         analysis_type: str) -> Tuple[str, ...]:
    """Return the stiffness-coefficient labels for a supported model pair."""
    material_key = str(material_type).strip().lower()
    analysis_key = str(analysis_type).strip().lower()
    key = (material_key, analysis_key)
    if key not in PARAMETER_LABELS_BY_MODEL:
        raise ValueError(
            f"Unsupported (material_type, analysis_type) combination "
            f"{key!r}.  Currently supported pairs: "
            f"{sorted(PARAMETER_LABELS_BY_MODEL.keys())}."
        )
    return PARAMETER_LABELS_BY_MODEL[key]


# =============================================================================
#  Domain definition (helper data containers)
# =============================================================================

@dataclass
class Bounds:
    """Rectangular domain extents, in the same length units as the data file."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class MeshSettings:
    """Structured Q4 mesh definition.

    ``x_nodes`` / ``y_nodes`` are optional explicit grid coordinates for a
    nonuniform structured mesh.  When omitted, the builder uses uniform
    spacing from ``number_of_elements_*`` and ``bounds``.
    """
    number_of_elements_x: int
    number_of_elements_y: int
    bounds: Bounds
    x_nodes: Optional[List[float]] = None
    y_nodes: Optional[List[float]] = None


@dataclass
class IdentificationSettings:
    """Outer loop convergence settings for the explicit ``K theta = U`` solve."""
    max_iterations: int = 10
    convergence_tolerance: float = 1.0e-4
    solution_scale: float = 1.0


@dataclass
class LinearAlgebraSettings:
    """Numerical tolerances for the rank-revealing SVD null-space computation."""
    svd_tolerance_factor: float = 100.0
    rank_tolerance_floor: float = 1.0e-12


@dataclass
class GlsSettings:
    """Settings for the Generalized Least Squares identification.

    basis_degree
        Total degree of the 2-D polynomial used to separate the smooth
        physical field from the measurement fluctuation when estimating
        Gamma.  It is NOT part of the VFM and never touches the virtual
        fields.  Too low leaks real mechanics into Gamma; too high fits the
        noise and underestimates it.  Sweep it -- if the identified
        parameters move, the noise model is carrying the result.
    weighting
        'gls'             full block-dense C, cross-time correlation kept
        'block_diagonal'  per-step covariance only, cross-time dropped
        'diagonal'        also drop cross-parameter terms
        'ols'             C = I, i.e. ordinary least squares
        The last three exist so the estimator can be shown to reduce to the
        simpler ones; 'gls' is the method.

        This is the SINGLE-AXIS control: it sets one rule that is applied both
        to the load steps of a test and, through the resulting information
        matrix, to the weighting between tests.  Those two jobs are not the
        same job -- see load_step_weighting / experiment_weighting below.
    load_step_weighting, experiment_weighting
        The TWO-AXIS control.  Both default to None, which keeps the legacy
        single-axis path above bit-for-bit.  Setting either one switches the
        solver to the explicit two-stage estimator:

            stage 1   theta_i = combine the load steps of experiment i
            stage 2   theta   = combine the per-experiment theta_i

        Why the split exists.  Because H_t = I, every load step already yields
        a complete standalone estimate of all parameters, so the whole
        identification is "combine N estimates of the same vector".  Load
        steps of one test are repeat measurements of the SAME specimen and are
        0.96-0.999 correlated, so a GLS combination of them is a DIFFERENCING
        operation whose leverage rests entirely on a cross-step covariance
        estimated from the same data.  Separate tests are independent and
        carry genuinely different information, so a GLS combination of THEM is
        ordinary inverse-variance weighting on solid ground.  One rule cannot
        be right for both.

        load_step_weighting  : 'gls' | 'block_diagonal' | 'diagonal' | 'ols'
            How the load steps of ONE experiment are combined.  'ols' is a
            plain average, which is the conservative choice and is guaranteed
            to land inside the range of the per-step estimates.
        experiment_weighting : 'gls' | 'diagonal' | 'ols'
            How the per-experiment estimates are combined.  'gls' uses each
            experiment's full covariance (recommended).  'diagonal' keeps only
            per-parameter variances.  'ols' weights every experiment equally.
            'block_diagonal' is NOT offered here: C is already block diagonal
            between experiments, so it would be a synonym for 'gls'.

        Unset axes fall back to `weighting` (load step) and 'gls'
        (experiment); the resolved pair is echoed in verbose output.

        WITH ONE LOAD STEP PER EXPERIMENT BOTH AXES ARE IRRELEVANT for stage
        1 -- there is nothing to combine -- and every load_step_weighting
        gives the identical answer.

        REPORTED UNCERTAINTY.  With experiment_weighting='gls' the eq. (20)
        parameter covariance is exact.  With 'diagonal' or 'ols' the weights
        are not the inverse of the covariance they are applied to, so eq. (20)
        is no longer a covariance -- the same caveat that already applies to
        weighting='diagonal' / 'ols'.  Quote the jackknife instead.
    spatial_correction
        OFF BY DEFAULT, AND OFF FOR EVERY REPORTED RESULT.  The measured
        spatial correlation of the residual field is reported as a DIAGNOSTIC
        (see figD_spatial_whiteness) but is never folded into C.

        Turning it on multiplies C by kappa, the measured number of
        effectively correlated points, and changes Gamma's normaliser from
        N - d to N - kappa*d.  That is an APPROXIMATION to eq. (1): it assumes
        the virtual fields are constant over the correlation neighbourhood so
        the double spatial sum factorises, and it needs a plateau chosen from a
        block-size ladder.  Both are judgement calls, and a covariance the
        derivation does not contain has no business setting a published error
        bar.  The honest repair is to evaluate eq. (1) itself; until then the
        solver stays loyal to eq. (3) and the misfit is REPORTED rather than
        absorbed.

        Left available only as a labelled sensitivity check.
    covariance_tolerance
        Convergence threshold on ||C^(k+1) - C^(k)||_F / ||C^(k)||_F, eq. (19).
        Checked in addition to the parameter criterion.
    """
    basis_degree: int = 6
    weighting: str = "gls"
    spatial_correction: bool = False
    covariance_tolerance: float = 1.0e-4
    load_step_weighting: Optional[str] = None
    experiment_weighting: Optional[str] = None

    #  Allowed values per axis.  'block_diagonal' is deliberately absent from
    #  the experiment axis: C is already block diagonal between experiments,
    #  so it would be an alias of 'gls' and only invite confusion.
    LOAD_STEP_CHOICES = ("gls", "block_diagonal", "diagonal", "ols")
    EXPERIMENT_CHOICES = ("gls", "diagonal", "ols")

    def __post_init__(self) -> None:
        self.weighting = str(self.weighting).strip().lower()
        allowed = ("gls", "block_diagonal", "diagonal", "ols")
        if self.weighting not in allowed:
            raise ValueError(
                f"Unsupported GLS weighting {self.weighting!r}; expected one "
                f"of {allowed}."
            )
        self.basis_degree = int(self.basis_degree)
        if self.basis_degree < 0:
            raise ValueError(
                f"basis_degree must be >= 0; got {self.basis_degree}.")

        def _clean(value, field_name, choices):
            if value is None:
                return None
            text = str(value).strip().lower()
            if not text or text == "auto":
                return None
            if text not in choices:
                extra = ""
                if field_name == "experiment_weighting" \
                        and text == "block_diagonal":
                    extra = ("  C is already block diagonal between "
                             "experiments, so 'block_diagonal' would mean the "
                             "same thing as 'gls' here; use 'gls'.")
                raise ValueError(
                    f"Unsupported {field_name} {text!r}; expected one of "
                    f"{choices}.{extra}"
                )
            return text

        self.load_step_weighting = _clean(
            self.load_step_weighting, "load_step_weighting",
            self.LOAD_STEP_CHOICES)
        self.experiment_weighting = _clean(
            self.experiment_weighting, "experiment_weighting",
            self.EXPERIMENT_CHOICES)

    @property
    def two_stage(self) -> bool:
        """True when the explicit two-stage estimator should be used.

        False keeps the legacy single-`weighting` path, which stays
        bit-for-bit unchanged.
        """
        return (self.load_step_weighting is not None
                or self.experiment_weighting is not None)

    def resolved_stages(self) -> "tuple[str, str]":
        """(load_step_weighting, experiment_weighting) with fallbacks applied.

        An unset load-step axis falls back to `weighting`; an unset experiment
        axis falls back to 'gls', which is the defensible default between
        statistically independent experiments.
        """
        load_step = self.load_step_weighting or self.weighting
        experiment = self.experiment_weighting or "gls"
        if experiment not in self.EXPERIMENT_CHOICES:
            #  Reachable only when `weighting` was 'block_diagonal' and the
            #  experiment axis was left unset; 'gls' is its meaning here.
            experiment = "gls"
        return load_step, experiment


@dataclass
class UserInputs:
    """Top-level inputs for the PLATE VFM inverse-solution run.

    Fields
    ------
    data_files
        Ordered list of experimental MAT files.  Each file represents one
        test with the same material parameters and analysis type.
    boundary_conditions_file
        After ``run_vfm.py`` finishes the BC editor step, this is set to the
        saved or loaded ``.py`` file used for the run.  ``None`` in
        ``build_user_inputs()`` means no file on disk yet (the editor starts
        empty).
    material_type
        Constitutive law identifier.  ``'iso'``, ``'orthotropic'``, or
        ``'anisotropic'``.  Orthotropic and anisotropic are supported only
        for ``'in_plane'`` and ``'out_of_plane'`` analyses.
    analysis_type
        Physics scope: ``'in_plane'``, ``'out_of_plane'``, or ``'both'``.
        Drives ``n_dof_per_node``, the constitutive matrix size, the
        measured-strain layout, and the set of allowed BC DOF labels.
    initial_parameters
        Initial guess for the unknown stiffness coefficients.  Length must
        match the (material_type, analysis_type) pair:
            iso + in_plane     -> 2  [A11, A12]
            iso + out_of_plane -> 2  [D11, D12]
            iso + both         -> 4  [A11, A12, D11, D12]
            orthotropic + in_plane     -> 4  [A11, A12, A22, A66]
            orthotropic + out_of_plane -> 4  [D11, D12, D22, D66]
            anisotropic + in_plane     -> 6  [A11, A12, A22, A66, A16, A26]
            anisotropic + out_of_plane -> 6  [D11, D12, D22, D66, D16, D26]
    mesh
        Mesh definition (element counts + bounds).
    experiment_ids
        One integer per entry of ``data_files`` saying which physical
        experiment that file belongs to.  Files sharing an id are LOAD STEPS
        of one test: the GLS gives them a dense cross-time covariance block,
        and they must share a virtual mesh, a BC file and a measurement ROI.
        Different ids are treated as statistically independent experiments,
        which is what makes the global covariance block diagonal.  Defaults to
        every file in experiment 1.
    identification
        Outer-loop tolerances.
    linear_algebra
        SVD tolerances for the admissible-space construction.
    gls
        Generalized-least-squares settings (noise basis, weighting mode,
        spatial correction).
    verbose
        Print per-iteration progress to stdout.
    """
    data_files: List[Path]
    boundary_conditions_file: Optional[Path]
    material_type: str
    analysis_type: str
    initial_parameters: List[float]
    mesh: MeshSettings
    experiment_ids: Optional[List[int]] = None
    identification: IdentificationSettings = field(
        default_factory=IdentificationSettings
    )
    linear_algebra: LinearAlgebraSettings = field(
        default_factory=LinearAlgebraSettings
    )
    gls: GlsSettings = field(default_factory=GlsSettings)
    verbose: bool = True

    def __post_init__(self) -> None:
        self.data_files = [Path(p) for p in self.data_files]
        if not self.data_files:
            raise ValueError(
                "data_files must contain at least one measurement file."
            )

        if self.experiment_ids is None:
            self.experiment_ids = [1] * len(self.data_files)
        else:
            self.experiment_ids = [int(v) for v in self.experiment_ids]
            if len(self.experiment_ids) != len(self.data_files):
                raise ValueError(
                    f"experiment_ids has {len(self.experiment_ids)} entries "
                    f"but there are {len(self.data_files)} data files."
                )

        self.material_type = str(self.material_type).strip().lower()
        self.analysis_type = str(self.analysis_type).strip().lower()

        if self.analysis_type not in ANALYSIS_TYPES:
            raise ValueError(
                f"Unsupported analysis_type {self.analysis_type!r}; "
                f"expected one of {ANALYSIS_TYPES}."
            )

        key = (self.material_type, self.analysis_type)
        if key not in PARAMETER_LABELS_BY_MODEL:
            raise ValueError(
                f"Unsupported (material_type, analysis_type) combination "
                f"{key!r}.  Currently supported pairs: "
                f"{sorted(PARAMETER_LABELS_BY_MODEL.keys())}."
            )

        expected_n = len(PARAMETER_LABELS_BY_MODEL[key])
        actual_n = len(self.initial_parameters)
        if actual_n != expected_n:
            raise ValueError(
                f"initial_parameters has length {actual_n} but "
                f"(material_type='{self.material_type}', "
                f"analysis_type='{self.analysis_type}') requires "
                f"{expected_n} parameters."
            )

    @property
    def n_dof_per_node(self) -> int:
        """Number of DOFs per node consistent with ``analysis_type``."""
        return dof_per_node_for_analysis(self.analysis_type)

    @property
    def data_file(self) -> Path:
        """Legacy single-test alias: return the first measurement file."""
        return self.data_files[0]


# =============================================================================
#  Default factory
# =============================================================================

_THIS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent.parent.parent


def _pre_conference_export(coupon: str) -> Path:
    """Locate a coupon's VFM_export after the solver moved under VFM Studio."""
    marker = Path(coupon) / "VFM_export"
    for parent in _THIS_DIR.parents:
        nested = parent / "Pre_Conference_Studies" / marker
        if nested.is_dir():
            return nested
        sibling = parent / marker
        if sibling.is_dir():
            return sibling
    return _THIS_DIR.parent.parent / "Pre_Conference_Studies" / marker


def build_user_inputs() -> UserInputs:
    """Return the default plate ``UserInputs`` instance.

    Default analysis is ``'both'`` (full ABD identification) so existing
    setups keep working without changes.  Switch ``analysis_type`` to
    ``'in_plane'`` or ``'out_of_plane'`` -- and shrink
    ``initial_parameters`` accordingly -- to focus on the membrane or
    bending sub-problem.
    """
    # --- Project paths -----------------------------------------------------
    #  Seed the GUI with the campaign's longitudinal bending test: its nine
    #  .mat are nine LOAD STEPS of one experiment, so they all take
    #  experiment_id 1 and share a dense cross-time noise covariance.  The
    #  demo Loading_Data/ set that used to be seeded here is not part of this
    #  folder -- it belonged to the general-purpose solver, not to this
    #  campaign.  Add the transverse steps as experiment 2 in the run-setup
    #  dialog to reproduce run_campaign_gls.py from the GUI.
    _long_dir = _pre_conference_export("Longitudinal_bending")
    data_files = sorted((_long_dir / "mat").glob("Sample_6_step*.mat"))
    experiment_ids = [1] * len(data_files)
    if not data_files:                       # keep the dialog usable anyway
        data_files = [_long_dir / "mat" / "Sample_6_step03.mat"]
        experiment_ids = [1]

    # Boundary conditions come from the BC GUI (load or create, then run).
    # For this campaign, load ../Longitudinal_bending/VFM_export/BC_Sample_6.py.

    # --- Constitutive model + physics scope -------------------------------
    # iso + in_plane -> 2 values [A11, A12]
    # iso + out_of_plane -> 2 values [D11, D12]
    # iso + both -> 4 values [A11, A12, D11, D12]
    # orthotropic + in_plane -> 4 values [A11, A12, A22, A66]
    # orthotropic + out_of_plane -> 4 values [D11, D12, D22, D66]
    # anisotropic + in_plane -> 6 values [A11, A12, A22, A66, A16, A26]
    # anisotropic + out_of_plane -> 6 values [D11, D12, D22, D66, D16, D26]
    material_type = "orthotropic"
    analysis_type = "out_of_plane"
    initial_parameters = [65.0, 15.0, 175.0, 40.0]   # D11, D12, D22, D66

    # --- Baseline mesh -----------------------------------------------------
    #  Seeded to the longitudinal coupon's measurement window (see
    #  BC_Sample_6.py MESH_SETTINGS, which run_vfm.py rebuilds from anyway).
    #  2x2 is not a free choice: that test's validate_setup.py benchmark shows
    #  accuracy DEGRADES under refinement, because the internal virtual work is
    #  a plain point sum that cannot resolve element-scale virtual fields.
    mesh = MeshSettings(
        number_of_elements_x=2,
        number_of_elements_y=2,
        bounds=Bounds(x_min=0.892091, x_max=9.613590,
                      y_min=-9.966410, y_max=6.565020),
    )

    return UserInputs(
        data_files=data_files,
        boundary_conditions_file=None,
        material_type=material_type,
        analysis_type=analysis_type,
        initial_parameters=initial_parameters,
        mesh=mesh,
        experiment_ids=experiment_ids,
        identification=IdentificationSettings(
            max_iterations=10,
            convergence_tolerance=1.0e-4,
            solution_scale=1.0,
        ),
        linear_algebra=LinearAlgebraSettings(
            svd_tolerance_factor=100.0,
            rank_tolerance_floor=1.0e-12,
        ),
        verbose=True,
    )

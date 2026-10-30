# =============================================================================
#  Generalized VFM / vfm_gls_identification.py
# -----------------------------------------------------------------------------
#  Feasible Generalized Least Squares identification, with the equation-error
#  covariance inside the convergence loop alongside the parameters.
#
#  WHAT THIS REPLACES
#  ---------------------------------------------------------------------------
#  run_identification_cases stacks every case with equal weight:
#
#      [K_1; K_2; ...] theta = [U_1; U_2; ...]        theta = lstsq(K, U)
#
#  Pierron's speciality condition forces K_i = I exactly for every case, so
#  that least-squares degenerates to the plain arithmetic mean of the per-case
#  estimates -- a case carrying no information about a parameter still votes
#  with full weight.  The stacked-VFM folder fixed this with per-parameter
#  inverse-variance weights 1/(sigma_i eta_ik)^2.  That is the right idea but
#  three things are left on the table:
#
#      * the weights are scalars per parameter -- no cross-parameter coupling;
#      * sigma_i is ASSUMED equal between tests rather than measured;
#      * the load steps of one test are treated as independent estimates.
#
#  The last one is false on this campaign: the measured residual correlation
#  between load steps is rho = 0.6-0.9, so nine steps carry closer to one or
#  two steps' worth of independent information.
#
#  WHAT THIS DOES INSTEAD
#  ---------------------------------------------------------------------------
#      theta^(k+1) = argmin  || L^-1 (H_G theta - b_G) ||^2,   C_G = L L^T
#
#  with C_G = blockdiag(C_1, ..., C_NE): DENSE across the load steps of one
#  experiment, BLOCK-INDEPENDENT between physically separate experiments.  Each
#  C_i is rebuilt from the current theta and the current virtual fields at
#  every iteration, so the covariance converges along with the parameters.
#
#  Because H_i is still the identity, the estimator is a MATRIX-weighted mean
#  of the per-load-step estimates -- the honest generalisation of the scalar
#  inverse-variance mean.  H is nevertheless assembled from the actual blocks
#  and never hardcoded, so the code stays correct if a future formulation
#  produces m > p rows.
#
#  THE REDUCTION LADDER -- how this is proved faithful
#  ---------------------------------------------------------------------------
#      weighting='ols'         -> reproduces run_identification_cases
#      weighting='diagonal'    -> reproduces the inverse-variance pooling of
#         with Gamma = sigma^2 dt I    stacked_identification.py, up to the dA
#                                      factor that estimator omits (below)
#      weighting='block_diagonal' -> keeps per-step, per-component covariance
#                                    but drops cross-time correlation
#      weighting='gls'         -> the full estimator
#
#  THE dA FACTOR.  _assemble_virtual_stiffness_matrix uses the integration
#  weight w_p to the FIRST power, so eta^2 = u^T G u is missing one factor of
#  dA relative to a true variance.  The covariance built here carries w_p^2,
#  i.e. eq. (4)'s dA.  Since dA differs slightly between experiments
#  (w_long/w_trans = 1.0149 on this campaign) this is a real, if small,
#  reweighting -- and the GLS form is the correct one.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.linalg import LinAlgError, cholesky, solve_triangular

from vfm_constitutive import (
    ConstitutiveModel,
    define_constitutive_model,
    rotate_constitutive_model,
)
from vfm_identification import (
    IdentificationCase,
    IdentificationResult,
    IterationRecord,
    _resolve_applied_force,
)
from vfm_mesh import build_external_work_matrix, map_points_to_mesh
from vfm_noise_covariance import (
    KinematicNoiseCovariance,
    assert_common_roi,
    build_gram_tensor,
    covariance_diagnostics,
    equation_covariance,
    estimate_kinematic_covariance,
)
from vfm_optimal_field import (
    OptimalFieldSolverOptions,
    collect_assembly_payload,
    solve_optimal_virtual_fields,
)
from vfm_user_inputs import (
    GlsSettings,
    IdentificationSettings,
    LinearAlgebraSettings,
)


# =============================================================================
#  Result containers
# =============================================================================

@dataclass
class GlsIterationDiagnostics:
    """Everything about iteration k that the convergence figures need.

    The parameters and the covariance move TOGETHER in FGLS, so both are
    recorded at every step rather than only at convergence.

    correlation_logdet
        log10 |R| of the equation-error correlation matrix
        R = D^-1/2 C_G D^-1/2.  For a correlation matrix |R| lies in (0, 1],
        equalling 1 only if every VFM equation is uncorrelated with every
        other.  It is therefore a single scalar reading of how much redundancy
        the stacked system carries: the more negative, the more the 72
        equations are really saying the same thing a few times over.
    """
    iteration: int
    parameters: np.ndarray
    parameter_covariance: np.ndarray
    parameter_std: np.ndarray
    covariance_logdet: float
    correlation_logdet: float
    delta_q: float
    covariance_delta: float


@dataclass
class ExperimentReport:
    """Per-experiment diagnostics, for the printed report and the README."""
    experiment_id: int
    label: str
    n_steps: int
    n_points: int
    area_per_point: float
    noise: Optional[KinematicNoiseCovariance] = None
    covariance: Optional[np.ndarray] = None
    covariance_diagnostics: Dict[str, float] = field(default_factory=dict)
    step_labels: List[str] = field(default_factory=list)
    step_estimates: Optional[np.ndarray] = None   # (T, p) per-step U_t
    #  Two-stage mode only (None otherwise).  reduced_estimate is this
    #  experiment's own theta_i after its load steps were combined, and
    #  reduced_covariance is its covariance V_i -- the pair that stage 2
    #  actually weights.  Reporting them is the point of the two-stage form:
    #  every coupon has a quotable number a reviewer can check.
    reduced_estimate: Optional[np.ndarray] = None       # (p,)
    reduced_covariance: Optional[np.ndarray] = None     # (p, p)


@dataclass
class GlsStepLinearization:
    """Frozen data needed to reverse one load-step block of the global H."""
    case_index: int
    experiment_id: int
    step_label: str
    row_start: int
    row_stop: int
    nodal_fields: np.ndarray
    element_dofs_all: np.ndarray
    B_all: np.ndarray
    basis_matrices: np.ndarray
    mapped_mask: np.ndarray
    n_measurement_points: int
    area_per_point: float
    element_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    local_coordinates: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=float)
    )
    measured_strains: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    flat_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    n_virtual_dofs: int = 0


@dataclass
class GlsLinearization:
    """Converged FGLS system with covariance and virtual fields frozen.

    The topology optimizer deliberately does not differentiate the covariance
    estimate or the optimal-field solves.  It differentiates the actual
    stacked design matrix at this snapshot.  ``vjp_measurements`` is the public
    reverse operation from dJ/dH to the generalized strains/curvatures supplied
    by each input ``IdentificationCase``.
    """
    design_matrix: np.ndarray
    rhs_vector: np.ndarray
    covariance: np.ndarray
    covariance_cholesky: np.ndarray
    parameters: np.ndarray
    steps: List[GlsStepLinearization] = field(default_factory=list)
    #  False in two-stage mode.  The aggregate system above is exact and
    #  everything built only from it (identifiability determinants, the
    #  parameter covariance, reassemble_design_matrix) stays valid; what is
    #  gone is the per-LOAD-STEP row map, because the two-stage design has
    #  one block per EXPERIMENT.  vjp_measurements refuses rather than
    #  returning an empty list, which would look like a zero gradient.
    per_step_map_available: bool = True

    def reassemble_design_matrix(
        self, measurements: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Rebuild H for new generalized strains with all fields frozen.

        ``measurements[case_index]`` has shape ``(n_generalised, n_points)``.
        This is the forward counterpart of :meth:`vjp_measurements` and is
        intended for local finite-difference checks of the topology adjoint.
        """
        if not self.steps:
            return np.asarray(self.design_matrix, dtype=float).copy()
        rebuilt = np.zeros_like(self.design_matrix, dtype=float)
        for step in self.steps:
            if step.case_index >= len(measurements):
                raise ValueError(
                    f"No measurements supplied for case {step.case_index}."
                )
            values = np.asarray(measurements[step.case_index], dtype=float)
            expected = (step.B_all.shape[1], step.n_measurement_points)
            if values.shape != expected:
                raise ValueError(
                    f"Case {step.case_index} measurements have shape "
                    f"{values.shape}; expected {expected}."
                )
            mapped = values[:, step.mapped_mask]
            local_a = step.area_per_point * np.einsum(
                "pai,jab,bp->pij",
                step.B_all,
                step.basis_matrices,
                mapped,
            )
            n_dofs = step.nodal_fields.shape[0]
            global_a = np.zeros(
                (n_dofs, self.design_matrix.shape[1]), dtype=float
            )
            np.add.at(global_a, step.element_dofs_all, local_a)
            rebuilt[step.row_start:step.row_stop, :] = \
                step.nodal_fields.T @ global_a
        return rebuilt

    def identify_with_frozen_fields(
        self, measurements: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Solve the fixed-covariance GLS system for perturbed measurements."""
        h_matrix = self.reassemble_design_matrix(measurements)
        whitened_h = np.linalg.solve(self.covariance_cholesky, h_matrix)
        whitened_b = np.linalg.solve(
            self.covariance_cholesky, self.rhs_vector
        )
        normal = whitened_h.T @ whitened_h
        rhs = whitened_h.T @ whitened_b
        return np.linalg.solve(normal, rhs)

    def vjp_measurements(self, design_gradient: np.ndarray) -> List[np.ndarray]:
        gradient = np.asarray(design_gradient, dtype=float)
        if gradient.shape != self.design_matrix.shape:
            raise ValueError(
                f"design_gradient has shape {gradient.shape}; expected "
                f"{self.design_matrix.shape}."
            )
        if not self.per_step_map_available:
            raise RuntimeError(
                "This linearization has no per-load-step row map, so the "
                "measurement adjoint cannot be formed.  It was produced in "
                "TWO-STAGE mode (load_step_weighting / experiment_weighting), "
                "where each experiment contributes ONE block to the design "
                "matrix instead of one per load step.  The aggregate system "
                "(design_matrix, covariance, parameters) is exact and may be "
                "used; for per-measurement gradients re-run with the legacy "
                "single-axis `weighting`."
            )
        if not self.steps:
            return []
        n_cases = 1 + max(step.case_index for step in self.steps)
        output: List[Optional[np.ndarray]] = [None] * n_cases
        for step in self.steps:
            block = gradient[step.row_start:step.row_stop, :]
            # H_t = U_t^T A_t, so dJ/dA_t = U_t dJ/dH_t.
            gradient_a = step.nodal_fields @ block
            local_gradient_a = gradient_a[step.element_dofs_all]
            mapped_gradient = step.area_per_point * np.einsum(
                "pij,pai,jab->bp",
                local_gradient_a,
                step.B_all,
                step.basis_matrices,
            )
            n_generalised = step.B_all.shape[1]
            full = np.zeros(
                (n_generalised, step.n_measurement_points), dtype=float
            )
            full[:, step.mapped_mask] = mapped_gradient
            output[step.case_index] = full
        if any(value is None for value in output):
            raise RuntimeError("The FGLS linearization is missing an input case.")
        return [np.asarray(value) for value in output]


@dataclass
class GlsIdentificationResult(IdentificationResult):
    """IdentificationResult plus everything the GLS layer adds.

    Subclasses IdentificationResult so run_vfm.py, the virtual-field viewer and
    the identifiability window keep working untouched.
    """
    weighting: str = "gls"
    iteration_diagnostics: List[GlsIterationDiagnostics] = field(
        default_factory=list)
    experiments: List[ExperimentReport] = field(default_factory=list)
    global_covariance: Optional[np.ndarray] = None
    parameter_covariance: Optional[np.ndarray] = None
    covariance_delta: float = float("nan")
    gls_settings: Optional[GlsSettings] = None
    linearization: Optional[GlsLinearization] = None

    @property
    def parameter_std(self) -> Optional[np.ndarray]:
        """Model-based first-order standard errors, eq. (20).

        OPTIMISTIC BY CONSTRUCTION.  Excludes the uncertainty in Gamma_hat
        (estimated from the same finite data) and the higher-order effect of
        data-dependent virtual fields.  Quote the jackknife alongside it.
        """
        if self.parameter_covariance is None:
            return None
        return np.sqrt(np.clip(np.diag(self.parameter_covariance), 0.0, None))


# =============================================================================
#  Per-experiment preparation -- everything that does not depend on theta
# =============================================================================

class _PreparedExperiment:
    """One experiment: T load steps sharing a mesh, a BC file and an ROI."""

    def __init__(self,
                 experiment_id: int,
                 cases: Sequence[IdentificationCase],
                 analysis_type: str,
                 n_generalised: int,
                 gls_settings: GlsSettings,
                 noise_override: Optional[KinematicNoiseCovariance] = None,
                 verbose: bool = True):
        self.experiment_id = experiment_id
        self.cases = list(cases)
        self.n_steps = len(self.cases)
        self.step_labels = [c.label or f"step {i + 1}"
                            for i, c in enumerate(self.cases)]
        self.label = _common_prefix(self.step_labels) or f"experiment {experiment_id}"

        first = self.cases[0]
        self.mesh_pkg = first.mesh_pkg
        self.boundary_conditions = first.boundary_conditions
        self.area_per_point = float(first.measurements.area_per_point)
        #  Every field this experiment owns -- kinematics, mesh, Gamma, the
        #  applied load -- lives in ITS OWN measurement frame.  This is the
        #  angle from that frame to the material frame; the constitutive matrix
        #  is rotated by it rather than the fields.
        self.material_orientation_deg = float(first.material_orientation_deg)

        self._check_homogeneous(analysis_type)

        # The point index must mean the same material location at every step,
        # or Gamma_{ts} = R_t^T R_s is meaningless.
        assert_common_roi([c.measurements.points for c in self.cases],
                          label=self.label)

        self.mapping = map_points_to_mesh(first.measurements.points,
                                          self.mesh_pkg)
        self.mapped_mask = self.mapping.element_ids >= 0
        self.n_points = int(self.mapped_mask.sum())
        if self.n_points == 0:
            raise ValueError(
                f"{self.label}: no measurement point maps into the virtual "
                "mesh.")

        # B(x_p) and its DOF scatter map are step-independent, so build the
        # payload once and swap only the measured strains per step.
        self._base_payload = collect_assembly_payload(
            first.measurements, self.mesh_pkg, self.mapping, n_generalised)
        self.gram_tensor = build_gram_tensor(
            self._base_payload, self.area_per_point, self.mesh_pkg.n_dofs)

        #  The external-work OPERATOR is topology, so it is shared -- that is
        #  exactly what makes these steps one experiment.  The applied load is
        #  the one thing that varies per step, so it must be resolved against
        #  each step's OWN BoundaryConditions: reading it off
        #  self.boundary_conditions (step 0's) would silently apply the first
        #  step's load to the entire ramp.
        self.external_work_matrix, _ = build_external_work_matrix(
            mesh_pkg=self.mesh_pkg, bc=self.boundary_conditions)
        self.applied_forces = [
            _resolve_applied_force(
                measurements=c.measurements,
                boundary_conditions=c.boundary_conditions,
                n_external_groups=self.external_work_matrix.shape[1],
                applied_load_source=c.applied_load_source)
            for c in self.cases
        ]

        # Gamma is estimated ONCE, outside the theta loop: it does not depend
        # on the material properties.  A caller-supplied Gamma (validate_gls.py
        # injecting a known covariance) bypasses the estimation entirely, which
        # is what isolates the propagation from the estimation when testing.
        if noise_override is not None:
            if noise_override.n_steps != self.n_steps:
                raise ValueError(
                    f"{self.label}: noise override has "
                    f"{noise_override.n_steps} load steps but the experiment "
                    f"has {self.n_steps}.")
            self.noise = noise_override
        else:
            strain_stack = np.stack([
                c.measurements.strains[:, self.mapped_mask].T
                for c in self.cases])
            self.noise = estimate_kinematic_covariance(
                points=first.measurements.points[self.mapped_mask],
                strain_stack=strain_stack,
                basis_degree=gls_settings.basis_degree,
                spatial_correction=gls_settings.spatial_correction,
            )

        if verbose and noise_override is not None:
            print(f"[VFM-GLS] experiment {experiment_id} ({self.label}): "
                  f"{self.n_steps} load steps, {self.n_points} points, "
                  f"Gamma SUPPLIED by caller (estimation bypassed)")
        elif verbose:
            correlation = self.noise.correlation
            print(f"[VFM-GLS] experiment {experiment_id} ({self.label}): "
                  f"{self.n_steps} load steps, {self.n_points} points, "
                  f"dA = {self.area_per_point:.6f} mm^2")
            print(f"[VFM-GLS]   Gamma {self.noise.gamma.shape[0]}x"
                  f"{self.noise.gamma.shape[0]} from a degree-"
                  f"{gls_settings.basis_degree} basis "
                  f"(d = {correlation.n_basis})")
            print(f"[VFM-GLS]   residual rho at the nearest neighbour "
                  f"({correlation.point_spacing:.3f} mm) = "
                  f"{correlation.nearest_rho:.3f}"
                  f"  [1.0 would mean pure smoothing, 0.0 spatially white]")
            if gls_settings.spatial_correction:
                print(f"[VFM-GLS]   spatial correction ON (NON-DEFAULT, "
                      f"sensitivity check only): kappa = "
                      f"{correlation.kappa:.1f}, nu = {correlation.nu:.0f} "
                      f"(eq. 3 nu = {correlation.nu_white:.0f}) -> C inflated "
                      f"x{correlation.covariance_inflation:.1f}")
            else:
                print(f"[VFM-GLS]   eq. (3) verbatim: kappa = 1. The measured "
                      f"kappa is {correlation.kappa:.1f}, reported as a "
                      f"diagnostic only.")
                print("[VFM-GLS]   => C is the covariance UNDER THE MODEL, "
                      "not the variability of the data.")

    def _check_homogeneous(self, analysis_type: str) -> None:
        """All load steps of one experiment must share mesh, BC and ROI."""
        wanted = str(analysis_type).strip().lower()
        reference = self.cases[0]
        for index, case in enumerate(self.cases):
            if case.mesh_pkg.analysis_type != wanted:
                raise ValueError(
                    f"{self.label} step {index}: mesh analysis_type "
                    f"{case.mesh_pkg.analysis_type!r} does not match run "
                    f"analysis_type {wanted!r}.")
            if case.measurements.analysis_type != wanted:
                raise ValueError(
                    f"{self.label} step {index}: measurement analysis_type "
                    f"{case.measurements.analysis_type!r} does not match run "
                    f"analysis_type {wanted!r}.")
            if case.mesh_pkg.n_dofs != reference.mesh_pkg.n_dofs:
                raise ValueError(
                    f"{self.label} step {index}: mesh has "
                    f"{case.mesh_pkg.n_dofs} DOFs but step 0 has "
                    f"{reference.mesh_pkg.n_dofs}.  Load steps grouped into "
                    "one experiment must share a virtual mesh.")
            if not np.array_equal(case.mesh_pkg.node_coords,
                                  reference.mesh_pkg.node_coords):
                raise ValueError(
                    f"{self.label} step {index}: virtual mesh node "
                    "coordinates differ from step 0.  Load steps grouped into "
                    "one experiment must share a virtual mesh.")
            if len(case.boundary_conditions.external) != \
                    len(reference.boundary_conditions.external):
                raise ValueError(
                    f"{self.label} step {index}: "
                    f"{len(case.boundary_conditions.external)} EXTERNAL "
                    f"group(s) against step 0's "
                    f"{len(reference.boundary_conditions.external)}.  The "
                    "external-work operator is shared across an experiment, "
                    "so only the load MAGNITUDES may vary between steps.")
            if str(case.applied_load_source) != \
                    str(reference.applied_load_source):
                raise ValueError(
                    f"{self.label} step {index}: applied-load source "
                    f"{case.applied_load_source!r} differs from step 0's "
                    f"{reference.applied_load_source!r}.  One experiment "
                    "takes its loads from one place.")
            if float(case.material_orientation_deg) != \
                    float(reference.material_orientation_deg):
                raise ValueError(
                    f"{self.label} step {index}: material orientation "
                    f"{case.material_orientation_deg} deg differs from step 0 "
                    f"({reference.material_orientation_deg} deg).  Load steps "
                    "grouped into one experiment are the same coupon in the "
                    "same fixture and must share a frame.")

    def model_in_test_frame(self,
                            model: ConstitutiveModel) -> ConstitutiveModel:
        """``model`` rotated from the material frame into this experiment's.

        The single conversion point.  It MUST be used for both the virtual-field
        solve and the covariance propagation: Gamma is estimated from this
        experiment's measured curvatures in its own frame, so pairing it with an
        unrotated D is a silent wrong answer rather than an error.
        """
        return rotate_constitutive_model(model, self.material_orientation_deg)

    def payload_for(self, step: int):
        """Base payload with this load step's measured strains swapped in."""
        from dataclasses import replace
        strains = self.cases[step].measurements.strains[:, self.mapped_mask]
        return replace(self._base_payload, measured_strains=strains)

    def solve_step_blocks(self,
                          model: ConstitutiveModel,
                          options: OptimalFieldSolverOptions):
        """Per load step: the virtual fields U_t, and the blocks H_t and b_t.

        U_t has one COLUMN per VFM equation, so E*_{t,p} = B_p U_t.  Nothing
        assumes U_t == U_s: the virtual fields are re-solved at every step from
        that step's own measured field.
        """
        n_parameters = model.number_of_parameters
        nodal_fields: List[np.ndarray] = []
        payloads: List = []
        blocks_h: List[np.ndarray] = []
        blocks_b: List[np.ndarray] = []
        field_results = None

        for step in range(self.n_steps):
            payload = self.payload_for(step)
            results = solve_optimal_virtual_fields(
                measurements=self.cases[step].measurements,
                mesh_pkg=self.mesh_pkg,
                boundary_conditions=self.boundary_conditions,
                constitutive_model=model,
                options=options,
                mapping=self.mapping,
                payload=payload,
            )
            if field_results is None:
                field_results = results

            a_matrix = results[0].A
            nodal = np.column_stack([r.nodal_variables_k for r in results])
            nodal_fields.append(nodal)
            payloads.append(payload)
            blocks_h.append(nodal.T @ a_matrix)

            displacement = self.external_work_matrix.T @ nodal
            blocks_b.append(
                np.asarray(displacement).T @ self.applied_forces[step]
                if displacement.ndim == 2
                else np.zeros(n_parameters))

        self.last_nodal_fields = nodal_fields
        self.last_payloads = payloads
        return nodal_fields, blocks_h, blocks_b, field_results


def _common_prefix(labels: Sequence[str]) -> str:
    """Shared leading token of the per-step labels, used to name the group."""
    if not labels:
        return ""
    head = labels[0].split(":")[0]
    return head if all(l.split(":")[0] == head for l in labels) else ""


# =============================================================================
#  Public entry point
# =============================================================================

def run_gls_identification_cases(
    cases: Sequence[IdentificationCase],
    material_type: str,
    analysis_type: str,
    initial_parameters,
    identification_settings: IdentificationSettings,
    linear_algebra_settings: LinearAlgebraSettings,
    experiment_ids: Optional[Sequence[int]] = None,
    gls_settings: Optional[GlsSettings] = None,
    noise_overrides: Optional[Dict[int, KinematicNoiseCovariance]] = None,
    verbose: bool = True,
) -> GlsIdentificationResult:
    """FGLS identification over experiments, each holding several load steps.

    experiment_ids
        One integer per case saying which physical experiment it belongs to.
        Cases sharing an id are load steps of ONE test: they get a dense
        cross-time covariance block and must share a mesh, a BC file and an
        ROI.  Different ids are assumed to have statistically independent
        measurement noise, which is what makes C_global block diagonal.
        Defaults to putting every case in one experiment.
    """
    cases = list(cases)
    if not cases:
        raise ValueError(
            "run_gls_identification_cases requires at least one case.")

    if gls_settings is None:
        gls_settings = GlsSettings()

    if experiment_ids is None:
        experiment_ids = [1] * len(cases)
    experiment_ids = [int(v) for v in experiment_ids]
    if len(experiment_ids) != len(cases):
        raise ValueError(
            f"experiment_ids has {len(experiment_ids)} entries for "
            f"{len(cases)} cases.")

    initial_parameters = np.asarray(
        initial_parameters, dtype=float).ravel().copy()
    current_parameters = initial_parameters.copy()

    options = OptimalFieldSolverOptions(
        svd_tolerance_factor=linear_algebra_settings.svd_tolerance_factor,
        rank_tolerance_floor=linear_algebra_settings.rank_tolerance_floor,
    )

    probe = define_constitutive_model(
        material_type=material_type, analysis_type=analysis_type,
        parameters=current_parameters)
    n_generalised = probe.n_generalised_strain
    n_parameters = probe.number_of_parameters

    # ---- group cases into experiments, preserving first-seen order --------
    order: List[int] = []
    grouped: Dict[int, List[IdentificationCase]] = {}
    for case, exp_id in zip(cases, experiment_ids):
        if exp_id not in grouped:
            grouped[exp_id] = []
            order.append(exp_id)
        grouped[exp_id].append(case)

    two_stage = gls_settings.two_stage
    load_step_rule, experiment_rule = gls_settings.resolved_stages()

    if verbose:
        if two_stage:
            print(f"[VFM-GLS] {len(cases)} load steps in {len(order)} "
                  f"experiment(s); TWO-STAGE weighting: "
                  f"load steps = {load_step_rule!r}, "
                  f"experiments = {experiment_rule!r}")
            print("[VFM-GLS]   stage 1 collapses each experiment to one "
                  "estimate; stage 2 combines the experiments.")
            if experiment_rule != "gls":
                print("[VFM-GLS]   NOTE: experiment_weighting is not 'gls', "
                      "so the eq. (20) parameter covariance is NOT a "
                      "covariance -- quote the jackknife instead.")
        else:
            print(f"[VFM-GLS] {len(cases)} load steps in {len(order)} "
                  f"experiment(s); weighting = {gls_settings.weighting!r}")

    experiments = [
        _PreparedExperiment(
            experiment_id=exp_id,
            cases=grouped[exp_id],
            analysis_type=analysis_type,
            n_generalised=n_generalised,
            gls_settings=gls_settings,
            noise_override=(None if noise_overrides is None
                            else noise_overrides.get(exp_id)),
            verbose=verbose,
        )
        for exp_id in order
    ]

    if (gls_settings.spatial_correction and len(experiments) > 1 and verbose
            and noise_overrides is None):
        inflations = [e.noise.correlation.covariance_inflation
                      for e in experiments]
        print(f"[VFM-GLS] spatial-correction inflation per experiment: "
              + ", ".join(f"{v:.1f}" for v in inflations)
              + f"  -> spread {max(inflations) / min(inflations):.3f}x")
        print("[VFM-GLS]   only the SPREAD moves theta; a common factor "
              "cancels because every H_i is the identity.")

    history: List[IterationRecord] = []
    iteration_diagnostics: List[GlsIterationDiagnostics] = []
    optimal_nodal_history: List[List[np.ndarray]] = []
    converged = False
    iterations = 0
    previous_covariance: Optional[np.ndarray] = None
    covariance_delta = float("nan")

    design = rhs = global_covariance = None
    parameter_covariance = None
    last_field_results: List = []
    constitutive_model = probe

    for iteration in range(1, identification_settings.max_iterations + 1):
        iterations = iteration
        constitutive_model = define_constitutive_model(
            material_type=material_type, analysis_type=analysis_type,
            parameters=current_parameters)

        design_blocks: List[np.ndarray] = []
        rhs_blocks: List[np.ndarray] = []
        covariance_blocks: List[np.ndarray] = []

        for index, experiment in enumerate(experiments):
            #  One rotation, used for BOTH the virtual fields and the
            #  covariance.  Splitting these two would pair a material-frame D
            #  with a test-frame Gamma and corrupt the weighting silently.
            experiment_model = experiment.model_in_test_frame(
                constitutive_model)
            experiment.last_model = experiment_model
            nodal_fields, blocks_h, blocks_b, field_results = \
                experiment.solve_step_blocks(experiment_model, options)
            if index == 0:
                last_field_results = field_results
                optimal_nodal_history.append(
                    [r.nodal_variables_k.copy() for r in field_results])

            #  In two-stage mode this must be the FULL dense covariance,
            #  whatever `weighting` says.  _reduce_experiment degrades it
            #  itself to build the load-step WEIGHTS, but it also needs the
            #  undegraded C to compute V_i = L C L', the true covariance of
            #  the reduced estimate that stage 2 weights by.  Passing a
            #  pre-degraded C here would make V_i a covariance of a noise
            #  model nobody asked for -- with weighting='ols' it would be the
            #  identity, and every experiment would end up weighted by L L'
            #  instead of by what its data actually supports.  So the legacy
            #  `weighting` field is deliberately ignored once either two-stage
            #  axis is set; the GUI greys it out to match.
            covariance = equation_covariance(
                gram_tensor=experiment.gram_tensor,
                material_matrix=experiment_model.material_matrix,
                noise=experiment.noise,
                nodal_fields=nodal_fields,
                weighting=("gls" if two_stage else gls_settings.weighting),
            )
            experiment.last_covariance = covariance
            experiment.last_estimates = np.vstack(blocks_b)

            experiment.last_design = np.vstack(blocks_h)
            experiment.last_rhs = np.concatenate(blocks_b)
            design_blocks.append(experiment.last_design)
            rhs_blocks.append(experiment.last_rhs)
            covariance_blocks.append(covariance)

        if two_stage:
            #  Stage 1: each experiment collapses to one estimate + its
            #  covariance.  Stage 2 then combines those, so the global system
            #  has one p x p block per EXPERIMENT rather than one per load
            #  step -- which is exactly what "average within, weight across"
            #  means, written out.
            reduced_estimates = []
            reduced_covariances = []
            for experiment, exp_design, exp_rhs, exp_cov in zip(
                    experiments, design_blocks, rhs_blocks, covariance_blocks):
                estimate, reduced_cov = _reduce_experiment(
                    exp_design, exp_rhs, exp_cov,
                    experiment.n_steps, load_step_rule)
                experiment.last_reduced_estimate = estimate
                experiment.last_reduced_covariance = reduced_cov
                reduced_estimates.append(estimate)
                reduced_covariances.append(reduced_cov)

            weight_blocks = _experiment_weight_covariance(
                reduced_covariances, experiment_rule)
            for experiment, block in zip(experiments, weight_blocks):
                experiment.last_weight_covariance = block

            identity_block = np.eye(n_parameters)
            design = np.vstack([identity_block] * len(experiments))
            rhs = np.concatenate(reduced_estimates)
            global_covariance = _block_diagonal(weight_blocks)
        else:
            design = np.vstack(design_blocks)
            rhs = np.concatenate(rhs_blocks)
            global_covariance = _block_diagonal(covariance_blocks)

        whitened_design, whitened_rhs = _whiten(
            design, rhs, global_covariance)
        raw_parameters, *_ = np.linalg.lstsq(
            whitened_design, whitened_rhs, rcond=None)
        next_parameters = (
            identification_settings.solution_scale * raw_parameters)

        residual_norm = float(np.linalg.norm(
            whitened_design @ raw_parameters - whitened_rhs))
        condition_number = float(np.linalg.cond(whitened_design))
        identity = np.eye(n_parameters)
        identity_deviation = float(np.linalg.norm(
            design[:n_parameters, :] - identity, ord="fro"))

        if previous_covariance is not None and \
                previous_covariance.shape == global_covariance.shape:
            scale = np.linalg.norm(previous_covariance, ord="fro")
            covariance_delta = float(
                np.linalg.norm(global_covariance - previous_covariance,
                               ord="fro") / max(scale, np.finfo(float).tiny))
        previous_covariance = global_covariance.copy()

        delta_q = float(np.sum(
            (current_parameters - next_parameters) ** 2
            / np.maximum(next_parameters ** 2, np.finfo(float).eps)))

        history.append(IterationRecord(
            iteration=iteration,
            parameters_before=current_parameters.copy(),
            parameters_after=next_parameters.copy(),
            delta_q=delta_q,
            residual_norm=residual_norm,
            condition_number=condition_number,
            identity_deviation=identity_deviation,
        ))

        iteration_covariance = _parameter_covariance(whitened_design)
        iteration_diagnostics.append(GlsIterationDiagnostics(
            iteration=iteration,
            parameters=next_parameters.copy(),
            parameter_covariance=iteration_covariance,
            parameter_std=np.sqrt(np.clip(
                np.diag(iteration_covariance), 0.0, None)),
            covariance_logdet=_signed_logdet(global_covariance),
            correlation_logdet=_signed_logdet(
                _to_correlation(global_covariance)),
            delta_q=delta_q,
            covariance_delta=covariance_delta,
        ))

        if verbose:
            print(f"[VFM-GLS] iter {iteration:2d}: "
                  + np.array2string(next_parameters, precision=4)
                  + f"  deltaQ = {delta_q:.3e}"
                  + (f"  deltaC = {covariance_delta:.3e}"
                     if np.isfinite(covariance_delta) else ""))

        current_parameters = next_parameters

        parameter_converged = delta_q <= identification_settings.convergence_tolerance
        covariance_converged = (
            not np.isfinite(covariance_delta)
            or covariance_delta <= gls_settings.covariance_tolerance)
        if parameter_converged and covariance_converged:
            converged = True
            break

    constitutive_model = define_constitutive_model(
        material_type=material_type, analysis_type=analysis_type,
        parameters=current_parameters)

    # ---- eq. (20): model-based first-order parameter covariance -----------
    whitened_design, _ = _whiten(design, rhs, global_covariance)
    parameter_covariance = _parameter_covariance(whitened_design)

    reports = []
    for experiment in experiments:
        covariance = getattr(experiment, "last_covariance", None)
        reports.append(ExperimentReport(
            experiment_id=experiment.experiment_id,
            label=experiment.label,
            n_steps=experiment.n_steps,
            n_points=experiment.n_points,
            area_per_point=experiment.area_per_point,
            noise=experiment.noise,
            covariance=covariance,
            covariance_diagnostics=(covariance_diagnostics(covariance)
                                    if covariance is not None else {}),
            step_labels=experiment.step_labels,
            step_estimates=getattr(experiment, "last_estimates", None),
            reduced_estimate=getattr(
                experiment, "last_reduced_estimate", None),
            reduced_covariance=getattr(
                experiment, "last_reduced_covariance", None),
        ))

    # Public frozen snapshot for topology/experimental-design adjoints.  The
    # Cholesky factor is the same whitening operation used by the final solve.
    #
    # IN TWO-STAGE MODE the aggregate system below is still exact and is
    # returned, but WITHOUT the per-step list: GlsStepLinearization addresses
    # the design matrix by LOAD-STEP row ranges (row_start/row_stop), and the
    # two-stage design has one block per EXPERIMENT, so those ranges do not
    # exist.  Handing back indices into a differently-shaped matrix would
    # corrupt any adjoint built on them, so `steps` is empty and
    # `per_step_map_available` is False -- vjp_measurements then raises
    # instead of returning an empty list that would read as a zero gradient.
    # Consumers that only need the aggregate system (identifiability
    # determinants, parameter covariance) keep working unchanged.
    linearization = None
    case_index_by_identity = {id(case): index for index, case in enumerate(cases)}
    linearization_steps: List[GlsStepLinearization] = []
    row_offset = 0
    for experiment in experiments:
        for local_step, case in enumerate(experiment.cases):
            payload = experiment.last_payloads[local_step]
            linearization_steps.append(GlsStepLinearization(
                case_index=case_index_by_identity[id(case)],
                experiment_id=experiment.experiment_id,
                step_label=experiment.step_labels[local_step],
                row_start=row_offset,
                row_stop=row_offset + n_parameters,
                nodal_fields=experiment.last_nodal_fields[local_step].copy(),
                element_dofs_all=payload.element_dofs_all.copy(),
                B_all=payload.B_all.copy(),
                basis_matrices=experiment.last_model.basis_matrices.copy(),
                mapped_mask=experiment.mapping.mapped_mask.copy(),
                n_measurement_points=int(case.measurements.points.shape[0]),
                area_per_point=float(case.measurements.area_per_point),
                element_ids=experiment.mapping.element_ids.copy(),
                local_coordinates=experiment.mapping.local_xy.copy(),
                measured_strains=payload.measured_strains.copy(),
                flat_indices=payload.flat_idx.copy(),
                n_virtual_dofs=int(payload.n_dofs),
            ))
            row_offset += n_parameters
    linearization = GlsLinearization(
        design_matrix=design.copy(),
        rhs_vector=rhs.copy(),
        covariance=global_covariance.copy(),
        covariance_cholesky=cholesky(global_covariance, lower=True),
        parameters=current_parameters.copy(),
        steps=([] if two_stage else linearization_steps),
        per_step_map_available=not two_stage,
    )
    if two_stage and verbose:
        print("[VFM-GLS]   linearization: aggregate system only; the "
              "per-load-step adjoint map does not exist in two-stage mode.")

    return GlsIdentificationResult(
        initial_parameters=initial_parameters,
        final_parameters=current_parameters,
        converged=converged,
        iterations_completed=iterations,
        history=history,
        optimal_field_results=last_field_results,
        optimal_nodal_history=optimal_nodal_history,
        K_matrix=design,
        U_vector=rhs,
        #  The case_* fields describe EXPERIMENTS here, not individual load
        #  steps: one stacked H block and one stacked b vector each.  The
        #  per-experiment equation covariances live in `experiments`, and the
        #  18 individual load-step cases in `diagnostic_cases`.
        case_K_matrices=[getattr(e, "last_design", None) for e in experiments],
        case_U_vectors=[getattr(e, "last_rhs", None) for e in experiments],
        case_labels=[e.label for e in experiments],
        case_data_files=[c.data_file for c in cases],
        diagnostic_cases=cases,
        constitutive_model=constitutive_model,
        weighting=gls_settings.weighting,
        iteration_diagnostics=iteration_diagnostics,
        experiments=reports,
        global_covariance=global_covariance,
        parameter_covariance=parameter_covariance,
        covariance_delta=covariance_delta,
        gls_settings=gls_settings,
        linearization=linearization,
    )


# =============================================================================
#  Linear algebra helpers
# =============================================================================

def _load_step_weight_covariance(covariance: np.ndarray,
                                 rule: str,
                                 n_steps: int) -> np.ndarray:
    """The covariance whose inverse is used to WEIGHT the load steps.

    This is not the covariance of the data -- that is always the full
    ``covariance`` argument, and it is what stage 2 propagates.  This is only
    the matrix that decides the relative weights inside one experiment, so
    dropping structure here is a modelling choice, not an approximation of
    the data.

        'gls'             the full dense C (cross-step correlation kept)
        'block_diagonal'  only the t == s blocks
        'diagonal'        only the diagonal entries
        'ols'             the identity -> every load step weighted equally
    """
    if rule == "ols":
        return np.eye(covariance.shape[0])
    if rule == "gls":
        return covariance
    if rule == "diagonal":
        return np.diag(np.diag(covariance))
    if rule == "block_diagonal":
        size = covariance.shape[0] // n_steps
        out = np.zeros_like(covariance)
        for t in range(n_steps):
            lo, hi = t * size, (t + 1) * size
            out[lo:hi, lo:hi] = covariance[lo:hi, lo:hi]
        return out
    raise ValueError(f"unknown load_step_weighting {rule!r}")


def _reduce_experiment(design: np.ndarray,
                       rhs: np.ndarray,
                       covariance: np.ndarray,
                       n_steps: int,
                       load_step_rule: str):
    """Collapse ONE experiment's load steps to a single estimate.

    Stage 1 of the two-stage estimator.  With the weighting matrix
    ``P = C_w^-1`` chosen by ``load_step_rule``:

        L       = (H' P H)^-1 H' P              (p x T*m)
        theta_i = L b                           the experiment's estimate
        V_i     = L C L'                        its covariance

    ``V_i`` uses the FULL measured covariance C, never ``C_w``: the weights
    may ignore the cross-step correlation, but the uncertainty of the result
    may not.  When ``load_step_rule == 'gls'`` the two coincide and this
    reduces to the textbook GLS block, ``V_i = (H' C^-1 H)^-1``.

    Nothing here assumes ``H = I``; the speciality condition makes it so in
    practice, but the algebra does not depend on it.
    """
    weight_cov = _load_step_weight_covariance(
        covariance, load_step_rule, n_steps)
    #  X = P H, formed by a solve so P is never inverted explicitly.
    if load_step_rule == "ols":
        scatter = design
    else:
        try:
            scatter = np.linalg.solve(weight_cov, design)
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                "The load-step weighting matrix is singular for "
                f"load_step_weighting={load_step_rule!r}.  With 'gls' this "
                "usually means two load steps carry almost the same "
                "information; try 'ols' on the load-step axis, a lower "
                "gls.basis_degree, or drop a redundant load step."
            ) from error
    information = design.T @ scatter
    reducer = np.linalg.solve(information, scatter.T)      # L
    estimate = reducer @ rhs
    reduced_cov = reducer @ covariance @ reducer.T
    return estimate, 0.5 * (reduced_cov + reduced_cov.T)


def _experiment_weight_covariance(reduced: Sequence[np.ndarray],
                                  rule: str) -> List[np.ndarray]:
    """Stage 2: what each experiment's covariance block becomes.

        'gls'       V_i itself -> inverse-covariance weighting, eq. (20) exact
        'diagonal'  diag(V_i)  -> per-parameter inverse-variance only
        'ols'       a common multiple of I -> every experiment counts equally

    For 'diagonal' and 'ols' the weights are no longer the inverse of the
    covariance they act on, so the eq. (20) parameter covariance stops being
    a covariance.  That is the same caveat the legacy 'diagonal' / 'ols'
    weightings already carry; the scale for 'ols' is chosen as the mean trace
    so the whitened system stays well scaled rather than to make eq. (20)
    meaningful.
    """
    if rule == "gls":
        return [np.array(v, dtype=float) for v in reduced]
    if rule == "diagonal":
        return [np.diag(np.diag(v)) for v in reduced]
    if rule == "ols":
        size = reduced[0].shape[0]
        scale = float(np.mean([np.trace(v) / v.shape[0] for v in reduced]))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return [scale * np.eye(size) for _ in reduced]
    raise ValueError(f"unknown experiment_weighting {rule!r}")


def _block_diagonal(blocks: Sequence[np.ndarray]) -> np.ndarray:
    """blockdiag(C_1, ..., C_NE) -- eq. (13), cross-experiment independence."""
    sizes = [b.shape[0] for b in blocks]
    total = int(sum(sizes))
    out = np.zeros((total, total), dtype=float)
    offset = 0
    for block, size in zip(blocks, sizes):
        out[offset:offset + size, offset:offset + size] = block
        offset += size
    return out


def _to_correlation(covariance: np.ndarray) -> np.ndarray:
    """D^-1/2 C D^-1/2 -- strips the scales, leaves the redundancy."""
    scale = np.sqrt(np.clip(np.diag(covariance), np.finfo(float).tiny, None))
    return covariance / np.outer(scale, scale)


def _signed_logdet(matrix: np.ndarray) -> float:
    """log10 |M|, or -inf if singular.  Never forms the determinant itself.

    A 72x72 determinant underflows float64 long before it means anything, so
    everything downstream works in log10.
    """
    sign, log_determinant = np.linalg.slogdet(matrix)
    if sign <= 0:
        return float("-inf")
    return float(log_determinant / np.log(10.0))


def _parameter_covariance(whitened_design: np.ndarray) -> np.ndarray:
    """(H^T C^-1 H)^-1 from the whitened design, via QR -- eq. (20).

    Formed from R of the QR rather than by inverting the normal matrix, which
    squares the condition number.
    """
    n_parameters = whitened_design.shape[1]
    _, upper = np.linalg.qr(whitened_design)
    try:
        inverse_upper = np.linalg.inv(upper)
    except np.linalg.LinAlgError:
        return np.full((n_parameters, n_parameters), np.nan)
    return inverse_upper @ inverse_upper.T


def _whiten(design: np.ndarray,
            rhs: np.ndarray,
            covariance: np.ndarray):
    """L^-1 H and L^-1 b via Cholesky -- eq. (16).  C^-1 is never formed.

    A Cholesky failure is reported with the eigenvalue spectrum rather than
    silently regularised: a non-positive-definite C means the noise model or
    the load-step set needs looking at, and ridging it would hand GLS enormous
    weights in directions the data does not support.
    """
    try:
        factor = cholesky(covariance, lower=True)
    except (LinAlgError, np.linalg.LinAlgError) as error:
        diagnostics = covariance_diagnostics(covariance)
        raise np.linalg.LinAlgError(
            "The GLS equation-error covariance C is not positive definite, so "
            "it cannot be Cholesky-factorised.\n"
            f"  min eigenvalue = {diagnostics['min_eigenvalue']:.6e}\n"
            f"  max eigenvalue = {diagnostics['max_eigenvalue']:.6e}\n"
            f"  asymmetry      = {diagnostics['asymmetry']:.6e}\n"
            "Investigate rather than regularise: near-zero eigenvalues usually "
            "mean two load steps carry almost the same information, or that "
            "the smoothing basis absorbed the signal at some step.  Try a "
            "lower gls.basis_degree, or drop a redundant load step."
        ) from error

    return (solve_triangular(factor, design, lower=True),
            solve_triangular(factor, rhs, lower=True))


# =============================================================================
#  Uncertainty that eq. (20) cannot see
# =============================================================================

def jackknife_parameters(
    cases: Sequence[IdentificationCase],
    experiment_ids: Sequence[int],
    material_type: str,
    analysis_type: str,
    initial_parameters,
    identification_settings: IdentificationSettings,
    linear_algebra_settings: LinearAlgebraSettings,
    gls_settings: Optional[GlsSettings] = None,
    verbose: bool = False,
):
    """Leave-one-load-step-out jackknife over the whole pipeline.

    Gamma is re-estimated inside every replicate, so this propagates the
    uncertainty of the noise model itself -- exactly what eq. (20) omits.
    Returns (estimates (n_replicates, p), mean, jackknife_std).

    Experiments that would be left with fewer than two load steps are skipped,
    since Gamma_{ts} needs at least a 2x2 time block to mean anything.
    """
    cases = list(cases)
    experiment_ids = [int(v) for v in experiment_ids]
    counts: Dict[int, int] = {}
    for exp_id in experiment_ids:
        counts[exp_id] = counts.get(exp_id, 0) + 1

    estimates = []
    for drop in range(len(cases)):
        if counts[experiment_ids[drop]] <= 2:
            continue
        kept = [i for i in range(len(cases)) if i != drop]
        result = run_gls_identification_cases(
            cases=[cases[i] for i in kept],
            material_type=material_type,
            analysis_type=analysis_type,
            initial_parameters=initial_parameters,
            identification_settings=identification_settings,
            linear_algebra_settings=linear_algebra_settings,
            experiment_ids=[experiment_ids[i] for i in kept],
            gls_settings=gls_settings,
            verbose=False,
        )
        estimates.append(result.final_parameters)
        if verbose:
            print(f"    jackknife drop {cases[drop].label or drop}: "
                  + np.array2string(result.final_parameters, precision=3))

    if not estimates:
        raise ValueError(
            "Jackknife needs at least one experiment with 3+ load steps.")

    estimates = np.vstack(estimates)
    n_replicates = estimates.shape[0]
    mean = estimates.mean(axis=0)
    #  Delete-one jackknife: std = sqrt((n-1)/n * sum (x_i - xbar)^2)
    spread = np.sqrt((n_replicates - 1) / n_replicates
                     * ((estimates - mean) ** 2).sum(axis=0))
    return estimates, mean, spread

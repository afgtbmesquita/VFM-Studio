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
        blocks_h: List[np.ndarray] = []
        blocks_b: List[np.ndarray] = []
        field_results = None

        for step in range(self.n_steps):
            results = solve_optimal_virtual_fields(
                measurements=self.cases[step].measurements,
                mesh_pkg=self.mesh_pkg,
                boundary_conditions=self.boundary_conditions,
                constitutive_model=model,
                options=options,
                mapping=self.mapping,
                payload=self.payload_for(step),
            )
            if field_results is None:
                field_results = results

            a_matrix = results[0].A
            nodal = np.column_stack([r.nodal_variables_k for r in results])
            nodal_fields.append(nodal)
            blocks_h.append(nodal.T @ a_matrix)

            displacement = self.external_work_matrix.T @ nodal
            blocks_b.append(
                np.asarray(displacement).T @ self.applied_forces[step]
                if displacement.ndim == 2
                else np.zeros(n_parameters))

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

    if verbose:
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
            nodal_fields, blocks_h, blocks_b, field_results = \
                experiment.solve_step_blocks(experiment_model, options)
            if index == 0:
                last_field_results = field_results
                optimal_nodal_history.append(
                    [r.nodal_variables_k.copy() for r in field_results])

            covariance = equation_covariance(
                gram_tensor=experiment.gram_tensor,
                material_matrix=experiment_model.material_matrix,
                noise=experiment.noise,
                nodal_fields=nodal_fields,
                weighting=gls_settings.weighting,
            )
            experiment.last_covariance = covariance
            experiment.last_estimates = np.vstack(blocks_b)

            experiment.last_design = np.vstack(blocks_h)
            experiment.last_rhs = np.concatenate(blocks_b)
            design_blocks.append(experiment.last_design)
            rhs_blocks.append(experiment.last_rhs)
            covariance_blocks.append(covariance)

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
        ))

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
    )


# =============================================================================
#  Linear algebra helpers
# =============================================================================

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

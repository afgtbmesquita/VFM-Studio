# =============================================================================
#  Generalized VFM / vfm_noise_covariance.py
# -----------------------------------------------------------------------------
#  The statistical half of the FGLS identification.  Four objects, and they must
#  never be conflated:
#
#      Gamma_i        covariance of the KINEMATIC MEASUREMENT NOISE, including
#                     correlation between load steps.     (n_g*T, n_g*T)
#      C_i(theta)     covariance of the VFM EQUATION ERRORS for experiment i,
#                     obtained by propagating Gamma_i through the actual
#                     step-dependent virtual fields.       (m*T, m*T)
#      C_global       blockdiag(C_1, ..., C_NE)
#      Cov(theta_hat) covariance of the identified parameters
#
#  This module builds the first two.  vfm_gls_identification.py does the rest.
#
#  WHERE THE NOISE ESTIMATE COMES FROM
#  ---------------------------------------------------------------------------
#  There are no repeated frames and no repeated tests, so the noise covariance
#  cannot be estimated by replication.  It is estimated from the SPATIAL
#  ENSEMBLE instead: fit a smooth spatial basis to the measured generalised
#  kinematics at each load step, and treat what the basis cannot represent as
#  the measurement fluctuation.
#
#      R_{i,t} = (I - P_i) Y_{i,t} = M_i Y_{i,t}
#      Z_i     = [R_{i,1} ... R_{i,T}]                (N_i, n_g*T)
#      Gamma_i = Z_i^T Z_i / nu_i                     (n_g*T, n_g*T)
#
#  Gamma_i >= 0 automatically because it is a Gram matrix -- which is the reason
#  for forming the whole thing as Z^T Z rather than assembling unrelated
#  pairwise blocks.
#
#  WHAT Gamma ACTUALLY CONTAINS -- READ THIS BEFORE QUOTING ANYTHING
#  ---------------------------------------------------------------------------
#  R_{i,t} is "everything the smooth basis cannot represent".  That is
#  measurement noise PLUS any real mechanical structure too fine for the basis
#  (lack of fit).  On this campaign the split is visible in the data: the
#  residual level correlates with the applied load at 0.88-0.99, but sigma/F
#  falls by ~4x across the load steps.  So it is a noise FLOOR plus a modest
#  load-proportional component, not one or the other.  Gamma is therefore an
#  EFFECTIVE ERROR covariance.  For GLS weighting that is arguably the right
#  object -- it is the error the estimator actually has to fight -- but it is
#  not a pure DIC noise measurement and must not be reported as one.
#
#  THE SPATIAL-WHITE ASSUMPTION, AND WHY IT IS CORRECTED HERE
#  ---------------------------------------------------------------------------
#  The derivation assumes E[eta_p eta_q^T] = delta_pq Gamma, which collapses a
#  spatial double integral to a single one.  On this data that assumption is
#  strongly violated: the residual correlation at ONE grid step (0.2 mm) is
#  rho = 0.87-0.97, because Vic-3D curvatures are differentiated and smoothed
#  fields.  Two separate consequences, and they are not the same size:
#
#    1. Gamma_hat = Z^T Z / (N - d) is BIASED.  The correct normaliser is
#           nu = tr(M P M) = tr(P M) = N - tr(Q^T P Q)
#       where P is the spatial correlation matrix.  Under whiteness P = I and
#       this reduces to tr(M) = N - d exactly.
#
#    2. The propagation eq. (3) is WRONG.  The honest object is the double sum
#           C_ts = sum_p sum_q w_p w_q rho_pq B_tp^T K Gamma_ts K^T B_sq
#       Because rho decays over a few grid steps while the virtual fields vary
#       over the specimen, B_sq ~ B_sp across the correlation neighbourhood and
#       the double sum factorises to
#           C_ts ~ sum_p w_p^2 kappa_p B_tp^T K Gamma_ts K^T B_sp
#       with kappa_p = sum_q rho_pq, the number of effectively correlated
#       points.  kappa = 1 recovers eq. (3) exactly.
#
#  EFFECT ON THE ANSWER vs EFFECT ON THE ERROR BAR -- these differ enormously:
#
#    * On theta: the speciality condition makes every H_i the identity, so a
#      covariance factor COMMON to all experiments cancels exactly.  Only the
#      RATIO between experiments survives.  Both tests here share one Vic-3D
#      setup and facet size, so the shift is small -- but it is printed, not
#      assumed.
#    * On Cov(theta): the factor does NOT cancel.  Uncorrected, eq. (20) is
#      optimistic by a large factor.  Corrected, it is still optimistic, because
#      it cannot see the uncertainty in Gamma_hat itself.  This is why the
#      driver also reports a leave-one-load-step-out jackknife.
#
#  Both paths are always computed and reported side by side.  `spatial_
#  correction=False` reproduces the derivation verbatim.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree


# =============================================================================
#  Smoothing basis -- used ONLY to separate smooth field from fluctuation.
#  It is not part of the VFM and never touches the identification.
# =============================================================================

def polynomial_basis(points: np.ndarray, degree: int) -> np.ndarray:
    """Total-degree 2-D polynomial basis on normalised coordinates.

    Returns ``(N, d)`` with ``d = (degree+1)(degree+2)/2``.  Coordinates are
    centred and scaled to O(1) so that high degrees stay conditioned.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be (N, 2); got {points.shape}.")
    if degree < 0:
        raise ValueError(f"degree must be >= 0; got {degree}.")

    columns = []
    for axis in range(2):
        values = points[:, axis]
        span = float(np.ptp(values))
        if span <= 0.0:
            raise ValueError(
                f"Measurement points are degenerate along axis {axis}; "
                "cannot build a smoothing basis."
            )
        columns.append((values - values.mean()) / span)
    xs, ys = columns

    basis = [
        (xs ** i) * (ys ** j)
        for i in range(degree + 1)
        for j in range(degree + 1 - i)
    ]
    return np.column_stack(basis)


def _orthonormal_basis(points: np.ndarray, degree: int) -> np.ndarray:
    """QR-orthonormalised smoothing basis Q, so that P = Q Q^T."""
    phi = polynomial_basis(points, degree)
    n_points, n_basis = phi.shape
    if n_basis >= n_points:
        raise ValueError(
            f"Smoothing basis has {n_basis} functions for only {n_points} "
            f"measurement points; reduce basis_degree (currently {degree})."
        )
    q, _ = np.linalg.qr(phi)
    return q


# =============================================================================
#  Spatial correlation of the residual field
# =============================================================================

@dataclass
class SpatialCorrelation:
    """Measured spatial correlation of the residual field, and its two factors.

    kappa               mean_p sum_q rho_pq -- the number of effectively
                        correlated points.  1.0 means spatially white.
    nu                  N - kappa*d, the unbiased normaliser for Z^T Z
    nu_white            N - d, the normaliser under spatial whiteness
    nearest_rho         residual correlation at the nearest-neighbour lag
    point_spacing       median nearest-neighbour distance, mm
    equivalent_length   h*sqrt(kappa/2pi): the isotropic exponential
                        correlation length that would give this kappa.  A
                        REPORTING convenience only -- nothing uses it, and the
                        real correlation here is measurably anisotropic.
    lags / rho_of_lag   correlogram by neighbour rank, for the record
    kappa_ladder        (block count, mean block size, kappa) per block scale
    """
    kappa: float
    nu: float
    nu_white: float
    nearest_rho: float
    point_spacing: float
    equivalent_length: float
    n_points: int
    n_basis: int
    lags: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rho_of_lag: np.ndarray = field(default_factory=lambda: np.zeros(0))
    kappa_ladder: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    @property
    def covariance_inflation(self) -> float:
        """Net factor applied to C relative to the spatial-white formula.

        C_corrected / C_white = kappa * nu_white / nu.  Only the RATIO of this
        number between experiments moves theta; the common part cancels.
        """
        return float(self.kappa * self.nu_white / self.nu)


def _standardise(residuals: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance per (load step, kinematic component) slice.

    Legitimate under the homogeneity assumption already made in eq. (2), and it
    is what lets the load steps and components be pooled into one correlation
    estimate.
    """
    standard = residuals - residuals.mean(axis=1, keepdims=True)
    scale = standard.std(axis=1, keepdims=True)
    scale[scale <= 0.0] = 1.0
    return standard / scale


def _kappa_by_block_variance(points: np.ndarray,
                             standard: np.ndarray,
                             blocks_per_axis: Sequence[int],
                             min_points_per_block: int = 8,
                             min_blocks: int = 25):
    """Effective correlated points, by spatial block averaging.

    For a block B of n_B points,

        E[ n_B * mean_B(eta)^2 ] = (1/n_B) sum_{p,q in B} rho_pq  ->  kappa

    once the block is large compared with the correlation length.  Averaging
    n_B * blockmean^2 over blocks handles the unequal block populations of an
    irregular DIC cloud automatically.

    Why this and not a fitted rho(r): the measurement cloud here is NOT a
    regular lattice (3126 distinct x for 3486 points) and the residual
    correlation is measurably anisotropic (rho ~ 0.73 along x against ~ 0.89
    along y), so a binned isotropic variogram fit is fragile and sensitive to
    bin-edge placement.  Block averaging needs neither isotropy nor a lattice,
    and it estimates exactly the sum_q rho_pq that the propagation needs.

    kappa(block size) rises while blocks are smaller than the correlation
    length and falls again once blocks approach the ROI, because the smoothing
    projector has already removed the large-scale modes.  The plateau between
    the two is the estimate, so this returns the maximum over the ladder.
    """
    x, y = points[:, 0], points[:, 1]
    span_x, span_y = np.ptp(x) * (1.0 + 1.0e-9), np.ptp(y) * (1.0 + 1.0e-9)
    n_steps, _, n_components = standard.shape

    ladder = []
    for grid in blocks_per_axis:
        bin_x = np.clip(((x - x.min()) / span_x * grid).astype(int), 0, grid - 1)
        bin_y = np.clip(((y - y.min()) / span_y * grid).astype(int), 0, grid - 1)
        block_id = bin_x * grid + bin_y

        counts = np.bincount(block_id, minlength=grid * grid)
        usable = counts >= min_points_per_block
        if int(usable.sum()) < min_blocks:
            continue

        totals = np.zeros((grid * grid, n_steps, n_components), dtype=float)
        np.add.at(totals, block_id, standard.transpose(1, 0, 2))
        block_means = totals[usable] / counts[usable][:, None, None]
        kappa = float((counts[usable][:, None, None] * block_means ** 2).mean())
        ladder.append((float(usable.sum()), float(counts[usable].mean()), kappa))

    if not ladder:
        raise ValueError(
            "Could not form enough spatial blocks to estimate the residual "
            "correlation.  The measurement cloud is too small for the "
            "spatial correction; run with spatial_correction=False."
        )

    ladder = np.array(ladder, dtype=float)
    return float(max(ladder[:, 2].max(), 1.0)), ladder


def measure_spatial_correlation(points: np.ndarray,
                                residuals: np.ndarray,
                                basis_q: np.ndarray,
                                n_neighbour_ranks: int = 8,
                                blocks_per_axis: Sequence[int] = (
                                    24, 20, 16, 12, 10, 8, 6),
                                ) -> SpatialCorrelation:
    """Measure the residual spatial correlation and derive kappa and nu.

    residuals   (T, N, n_g) residual field, one slab per load step
    basis_q     (N, d) orthonormal smoothing basis, so M = I - Q Q^T

    Two things come out:

        kappa = mean_p sum_q rho_pq          scales C   (eq. 3 -> eq. 1)
        nu    = tr(P M) ~ N - kappa*d        normalises Z^T Z

    The nu approximation uses tr(Q^T P Q) ~ kappa*d, which holds because the
    smoothing basis functions are by construction the smoothest modes on the
    ROI, so P acts on them as multiplication by kappa.  Both reduce to the
    spatially-white values at kappa = 1.
    """
    points = np.asarray(points, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    n_points = points.shape[0]
    n_basis = basis_q.shape[1]

    tree = cKDTree(points)
    ranks = min(n_neighbour_ranks + 1, n_points)
    nn_distance, nn_index = tree.query(points, k=ranks)
    spacing = float(np.median(nn_distance[:, 1]))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Could not determine a positive point spacing.")

    standard = _standardise(residuals)

    # ---- correlogram by neighbour RANK (no binning, no lattice assumed) ---
    lags = nn_distance[:, 1:ranks].mean(axis=0)
    rho_of_lag = np.array([
        float((standard * standard[:, nn_index[:, j], :]).mean())
        for j in range(1, ranks)
    ])
    nearest_rho = float(rho_of_lag[0]) if rho_of_lag.size else float("nan")

    # ---- kappa, and from it nu -------------------------------------------
    kappa, ladder = _kappa_by_block_variance(points, standard, blocks_per_axis)

    nu = float(n_points - kappa * n_basis)
    if nu <= 0.1 * n_points:
        raise ValueError(
            f"Spatial-correlation-corrected degrees of freedom collapsed "
            f"(nu = {nu:.1f} out of N = {n_points}) because the residual is "
            f"correlated over kappa = {kappa:.1f} points and the smoothing "
            f"basis has d = {n_basis} functions.  Reduce gls.basis_degree, or "
            "run with spatial_correction=False and treat eq. (20) as a lower "
            "bound."
        )

    return SpatialCorrelation(
        kappa=kappa,
        nu=nu,
        nu_white=float(n_points - n_basis),
        nearest_rho=nearest_rho,
        point_spacing=spacing,
        equivalent_length=float(spacing * np.sqrt(kappa / (2.0 * np.pi))),
        n_points=n_points,
        n_basis=n_basis,
        lags=lags,
        rho_of_lag=rho_of_lag,
        kappa_ladder=ladder,
    )


# =============================================================================
#  Gamma:  cross-time kinematic noise covariance
# =============================================================================

@dataclass
class KinematicNoiseCovariance:
    """Gamma_i and everything needed to report how it was obtained."""
    gamma: np.ndarray                 # (n_g*T, n_g*T), step-major
    n_generalised: int
    n_steps: int
    basis_degree: int
    spatial_correction: bool
    correlation: SpatialCorrelation
    residual_scale: np.ndarray        # (T, n_g) rms residual per step/component

    def block(self, t: int, s: int) -> np.ndarray:
        """The (n_g, n_g) cross-covariance Gamma_{ts} between steps t and s."""
        g = self.n_generalised
        return self.gamma[t * g:(t + 1) * g, s * g:(s + 1) * g]

    @property
    def kappa(self) -> float:
        """Effective correlated points; 1.0 when the correction is off."""
        return self.correlation.kappa if self.spatial_correction else 1.0

    def temporal_correlation(self, component: int) -> np.ndarray:
        """(T, T) correlation across load steps for one kinematic component."""
        g = self.n_generalised
        index = [t * g + component for t in range(self.n_steps)]
        sub = self.gamma[np.ix_(index, index)]
        sd = np.sqrt(np.clip(np.diag(sub), 1.0e-300, None))
        return sub / np.outer(sd, sd)


def estimate_kinematic_covariance(points: np.ndarray,
                                  strain_stack: np.ndarray,
                                  basis_degree: int = 6,
                                  spatial_correction: bool = True
                                  ) -> KinematicNoiseCovariance:
    """Estimate Gamma_i from the spatial ensemble of the measured fields.

    points        (N, 2)        common ROI, identical across load steps
    strain_stack  (T, N, n_g)   measured generalised kinematics per load step

    The point index must refer to the same material location at every load
    step; the caller is responsible for enforcing that (see
    ``assert_common_roi``).
    """
    points = np.asarray(points, dtype=float)
    strain_stack = np.asarray(strain_stack, dtype=float)
    if strain_stack.ndim != 3:
        raise ValueError(
            f"strain_stack must be (T, N, n_g); got {strain_stack.shape}.")
    n_steps, n_points, n_gen = strain_stack.shape
    if points.shape[0] != n_points:
        raise ValueError(
            f"strain_stack has {n_points} points but points has "
            f"{points.shape[0]}.")

    basis_q = _orthonormal_basis(points, basis_degree)

    # R_{t} = M Y_{t},  M = I - Q Q^T.  Applied via Q to avoid forming M.
    residuals = strain_stack - np.einsum(
        "ij,tjc->tic", basis_q @ basis_q.T, strain_stack)

    correlation = measure_spatial_correlation(points, residuals, basis_q)
    nu = correlation.nu if spatial_correction else correlation.nu_white

    # Z = [R_1 ... R_T] with (step, component) ordering, then Gamma = Z^T Z/nu
    z_matrix = residuals.transpose(1, 0, 2).reshape(n_points, n_steps * n_gen)
    gamma = (z_matrix.T @ z_matrix) / nu
    gamma = 0.5 * (gamma + gamma.T)

    return KinematicNoiseCovariance(
        gamma=gamma,
        n_generalised=n_gen,
        n_steps=n_steps,
        basis_degree=basis_degree,
        spatial_correction=spatial_correction,
        correlation=correlation,
        residual_scale=np.sqrt((residuals ** 2).mean(axis=1)),
    )


def known_kinematic_covariance(gamma: np.ndarray,
                               n_generalised: int,
                               kappa: float = 1.0
                               ) -> KinematicNoiseCovariance:
    """Wrap a KNOWN Gamma so it can be fed straight to the propagation.

    Used by validate_gls.py to separate the two halves of the method: with a
    supplied Gamma the estimation step is bypassed entirely, so a failure can
    only come from the propagation, the assembly or the solve.
    """
    gamma = np.asarray(gamma, dtype=float)
    if gamma.ndim != 2 or gamma.shape[0] != gamma.shape[1]:
        raise ValueError(f"gamma must be square; got {gamma.shape}.")
    if gamma.shape[0] % n_generalised:
        raise ValueError(
            f"gamma has size {gamma.shape[0]}, not a multiple of "
            f"n_generalised = {n_generalised}.")
    n_steps = gamma.shape[0] // n_generalised

    return KinematicNoiseCovariance(
        gamma=0.5 * (gamma + gamma.T),
        n_generalised=n_generalised,
        n_steps=n_steps,
        basis_degree=-1,
        spatial_correction=(kappa != 1.0),
        correlation=SpatialCorrelation(
            kappa=float(kappa),
            nu=float("nan"),
            nu_white=float("nan"),
            nearest_rho=float("nan"),
            point_spacing=float("nan"),
            equivalent_length=float("nan"),
            n_points=0,
            n_basis=0,
        ),
        residual_scale=np.sqrt(np.abs(np.diag(gamma))).reshape(
            n_steps, n_generalised),
    )


def assert_common_roi(points_per_step: Sequence[np.ndarray],
                      label: str = "experiment",
                      tolerance: float = 0.0) -> np.ndarray:
    """Verify the point index means the same material location at every step.

    Gamma_{ts} = R_t^T R_s / nu is only meaningful if row p of R_t and row p of
    R_s are the same physical location.  This is the assumption that a silent
    re-export would break, so it is checked rather than trusted.
    """
    reference = np.asarray(points_per_step[0], dtype=float)
    for index, candidate in enumerate(points_per_step[1:], start=1):
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != reference.shape:
            raise ValueError(
                f"COMMON-ROI CHECK FAILED for {label}: load step {index} has "
                f"{candidate.shape[0]} points but step 0 has "
                f"{reference.shape[0]}.  The cross-time covariance "
                "Gamma_{ts} = R_t^T R_s has no meaning unless the point index "
                "refers to the same material location at every step."
            )
        deviation = float(np.abs(candidate - reference).max())
        if deviation > tolerance:
            raise ValueError(
                f"COMMON-ROI CHECK FAILED for {label}: load step {index} "
                f"point coordinates differ from step 0 by up to "
                f"{deviation:.3e} mm (tolerance {tolerance:.3e}).  Re-export "
                "the dataset on a common mask."
            )
    return reference


# =============================================================================
#  Propagation:  Gamma  ->  C
# =============================================================================

def build_gram_tensor(payload,
                      area_per_point: float,
                      n_dofs: int) -> np.ndarray:
    """The theta-independent half of the covariance propagation.

        Phi[e, f] = sum_p w_p^2 B_p[e, :]^T B_p[f, :]        (n_dofs, n_dofs)

    Note the SQUARED weight.  The solver's own G matrix
    (_assemble_virtual_stiffness_matrix) uses w_p to the first power, so
    Phi carries exactly one extra factor of dA -- which is the dA of eq. (4).
    The unit test in validate_gls.py pins this down by reproducing the
    solver's G from Phi with Gamma = I.

    Depends only on the mesh and the measurement cloud, so it is built once per
    experiment and reused at every iteration.
    """
    b_all = np.asarray(payload.B_all, dtype=float)
    dof_index = np.asarray(payload.element_dofs_all)
    n_mapped, n_gen, _ = b_all.shape

    # Scatter the element-local B onto global DOFs.
    b_global = np.zeros((n_mapped, n_gen, n_dofs), dtype=float)
    point_index = np.arange(n_mapped)[:, None]
    for component in range(n_gen):
        np.add.at(b_global,
                  (point_index, component, dof_index),
                  b_all[:, component, :])

    weight = float(area_per_point) ** 2
    return np.einsum("pei,pfj->efij", b_global, b_global) * weight


def virtual_work_gram(gram_tensor: np.ndarray,
                      material_matrix: np.ndarray,
                      gamma_block: np.ndarray) -> np.ndarray:
    """G(Gamma) = sum_p w_p^2 B_p^T K Gamma K^T B_p, as an (n_dofs, n_dofs).

    Expanded over the precomputed Phi so that the only per-iteration work is a
    small (n_g, n_g) contraction:

        Psi[e, f] = sum_cd K[e, c] K[f, d] Gamma[c, d]
        G(Gamma)  = sum_ef Psi[e, f] Phi[e, f]
    """
    psi = np.einsum("ec,fd,cd->ef",
                    material_matrix, material_matrix, gamma_block)
    return np.einsum("ef,efij->ij", psi, gram_tensor)


def equation_covariance(gram_tensor: np.ndarray,
                        material_matrix: np.ndarray,
                        noise: KinematicNoiseCovariance,
                        nodal_fields: Sequence[np.ndarray],
                        weighting: str = "gls") -> np.ndarray:
    """Assemble C_i(theta) for one experiment -- eq. (10) and eq. (11).

        C_{i,ts} = kappa * U_t^T G(Gamma_{ts}) U_s

    nodal_fields   list of T arrays (n_dofs, m); column a is the optimal
                   virtual field for equation a AT THAT LOAD STEP.  Nothing
                   here assumes E*_t == E*_s: every block is built from the
                   virtual fields actually used at steps t and s.

    weighting
        'gls'             full block-dense C (cross-time kept)
        'block_diagonal'  drop the t != s blocks
        'diagonal'        also drop cross-parameter terms within a block
        'ols'             identity -- ordinary least squares
    """
    if weighting not in ("gls", "block_diagonal", "diagonal", "ols"):
        raise ValueError(f"unknown weighting {weighting!r}")

    n_steps = len(nodal_fields)
    if n_steps != noise.n_steps:
        raise ValueError(
            f"{n_steps} load steps of virtual fields but Gamma was estimated "
            f"from {noise.n_steps}.")
    n_equations = nodal_fields[0].shape[1]
    size = n_steps * n_equations

    if weighting == "ols":
        return np.eye(size)

    kappa = noise.kappa
    covariance = np.zeros((size, size), dtype=float)
    for t in range(n_steps):
        columns = range(n_steps) if weighting == "gls" else (t,)
        for s in columns:
            gram = virtual_work_gram(
                gram_tensor, material_matrix, noise.block(t, s))
            block = nodal_fields[t].T @ gram @ nodal_fields[s]
            covariance[t * n_equations:(t + 1) * n_equations,
                       s * n_equations:(s + 1) * n_equations] = kappa * block

    covariance = 0.5 * (covariance + covariance.T)
    if weighting == "diagonal":
        covariance = np.diag(np.diag(covariance))
    return covariance


def covariance_diagnostics(covariance: np.ndarray) -> Dict[str, float]:
    """Symmetry, definiteness and conditioning of an assembled C."""
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    smallest = float(eigenvalues.min())
    largest = float(eigenvalues.max())
    return {
        "min_eigenvalue": smallest,
        "max_eigenvalue": largest,
        "condition_number": float(largest / smallest) if smallest > 0
        else float("inf"),
        "asymmetry": float(np.abs(covariance - covariance.T).max()),
        "positive_definite": bool(smallest > 0.0),
    }

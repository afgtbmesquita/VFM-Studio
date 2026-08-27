# =============================================================================
#  plate / vfm_constitutive.py
# -----------------------------------------------------------------------------
#  Plate-specific constitutive model.  Three analysis types are supported,
#  each with its own material matrix:
#
#      analysis_type == 'in_plane'
#          material_matrix  =  A   (3x3 in-plane stiffness)
#          generalised stress = [N_xx, N_yy, N_xy]^T
#          generalised strain = [eps_xx, eps_yy, gamma_xy]^T
#
#      analysis_type == 'out_of_plane'
#          material_matrix  =  D   (3x3 bending stiffness)
#          generalised stress = [M_xx, M_yy, M_xy]^T
#          generalised strain = [kappa_xx, kappa_yy, kappa_xy]^T
#
#      analysis_type == 'both'
#          material_matrix  =  ABD  (6x6 laminate stiffness)
#          generalised stress = [N_xx, N_yy, N_xy, M_xx, M_yy, M_xy]^T
#          generalised strain = [eps_xx, eps_yy, gamma_xy,
#                                kappa_xx, kappa_yy, kappa_xy]^T
#
#  For the isotropic material the three pairs of unknowns are:
#
#      iso + in_plane     -> [A11, A12]              (2 unknowns)
#      iso + out_of_plane -> [D11, D12]              (2 unknowns)
#      iso + both         -> [A11, A12, D11, D12]    (4 unknowns; B = 0)
#
#  For the orthotropic material, only partial analyses are supported:
#
#      orthotropic + in_plane     -> [A11, A12, A22, A66]
#      orthotropic + out_of_plane -> [D11, D12, D22, D66]
#
#  For the anisotropic material, only partial analyses are supported:
#
#      anisotropic + in_plane     -> [A11, A12, A22, A66, A16, A26]
#      anisotropic + out_of_plane -> [D11, D12, D22, D66, D16, D26]
#
#  Isotropic reductions:
#
#      A22 = A11        A66 = (A11 - A12) / 2
#      D22 = D11        D66 = (D11 - D12) / 2.
#
#  We expose the basis tensor
#
#      d(material_matrix) / d(theta_j)   --   shape (P, n, n)  with n in {3, 6}
#
#  so the optimal-field solver never re-derives the sensitivity by hand.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ConstitutiveModel:
    """Bundle of everything the plate VFM assembly needs about the material.

    Fields
    ------
    material_type
        Constitutive law identifier.  ``'iso'``, ``'orthotropic'``, or
        ``'anisotropic'``.
    analysis_type
        Physics scope.  One of ``'in_plane'``, ``'out_of_plane'``, ``'both'``.
    parameters
        1-D float array of length P (the number of stiffness coefficients
        for the active (material_type, analysis_type) pair).
    parameter_labels
        Human-readable names matched to ``parameters``.
    material_matrix
        The active stiffness matrix.  Shape ``(n, n)`` with
          n = 3 for in_plane (A only) or out_of_plane (D only)
          n = 6 for both (full ABD).
    basis_matrices
        Stack of constant ``(n, n)`` matrices
        ``d(material_matrix) / d(theta_k)`` of shape ``(P, n, n)`` so that
        ``material_matrix(theta) = sum_k theta_k * basis_matrices[k]``.
    orientation_deg
        Frame the matrices are expressed in, as the angle from the lab +x axis
        to the material 1-axis (counter-clockwise positive).  ``0.0`` means the
        matrices are in the material frame, which is the case for everything
        ``define_constitutive_model`` returns.  A non-zero value marks a model
        that ``rotate_constitutive_model`` has pushed into a test frame.  It is
        carried for reporting only -- nothing in the solver branches on it.
    """
    material_type: str
    analysis_type: str
    parameters: np.ndarray
    parameter_labels: List[str]
    material_matrix: np.ndarray
    basis_matrices: np.ndarray
    orientation_deg: float = 0.0

    @property
    def number_of_parameters(self) -> int:
        return self.parameters.size

    @property
    def n_generalised_strain(self) -> int:
        """Row count of the per-point strain vector for this model."""
        return int(self.material_matrix.shape[0])


# =============================================================================
#  Public factory
# =============================================================================

def define_constitutive_model(material_type: str,
                              analysis_type: str,
                              parameters) -> ConstitutiveModel:
    """Build a plate ``ConstitutiveModel`` from the (material, analysis) pair.

    Parameters
    ----------
    material_type : str
        ``'iso'``, ``'orthotropic'``, or ``'anisotropic'``.
    analysis_type : str
        One of ``'in_plane'``, ``'out_of_plane'``, ``'both'``.
    parameters : array-like
        Length-P sequence of stiffness coefficients.  Expected length depends
        on the (material_type, analysis_type) pair:
            iso + in_plane     -> 2  [A11, A12]
            iso + out_of_plane -> 2  [D11, D12]
            iso + both         -> 4  [A11, A12, D11, D12]
            orthotropic + in_plane     -> 4  [A11, A12, A22, A66]
            orthotropic + out_of_plane -> 4  [D11, D12, D22, D66]
            anisotropic + in_plane     -> 6  [A11, A12, A22, A66, A16, A26]
            anisotropic + out_of_plane -> 6  [D11, D12, D22, D66, D16, D26]
    """
    material_key = str(material_type).strip().lower()
    analysis_key = str(analysis_type).strip().lower()
    parameters = np.asarray(parameters, dtype=float).reshape(-1).copy()

    if material_key == "iso":
        if analysis_key == "in_plane":
            return _build_iso_inplane(parameters)
        if analysis_key == "out_of_plane":
            return _build_iso_outofplane(parameters)
        if analysis_key == "both":
            return _build_iso_both(parameters)
        raise ValueError(
            f"Unsupported analysis_type {analysis_type!r}; expected one of "
            "'in_plane', 'out_of_plane', 'both'."
        )

    if material_key == "orthotropic":
        if analysis_key == "in_plane":
            return _build_orthotropic_inplane(parameters)
        if analysis_key == "out_of_plane":
            return _build_orthotropic_outofplane(parameters)
        if analysis_key == "both":
            raise ValueError(
                "Orthotropic material is supported only for analysis_type "
                "'in_plane' or 'out_of_plane'; 'both' is intentionally not "
                "supported."
            )
        raise ValueError(
            f"Unsupported analysis_type {analysis_type!r}; expected one of "
            "'in_plane', 'out_of_plane', 'both'."
        )

    if material_key == "anisotropic":
        if analysis_key == "in_plane":
            return _build_anisotropic_inplane(parameters)
        if analysis_key == "out_of_plane":
            return _build_anisotropic_outofplane(parameters)
        if analysis_key == "both":
            raise ValueError(
                "Anisotropic material is supported only for analysis_type "
                "'in_plane' or 'out_of_plane'; 'both' is intentionally not "
                "supported."
            )
        raise ValueError(
            f"Unsupported analysis_type {analysis_type!r}; expected one of "
            "'in_plane', 'out_of_plane', 'both'."
        )

    raise ValueError(
        f"Unsupported material_type {material_type!r}.  Supported material "
        "types are 'iso', 'orthotropic', and 'anisotropic'."
    )


# =============================================================================
#  Frame rotation
# -----------------------------------------------------------------------------
#  The identified parameters theta live in the MATERIAL frame.  A coupon cut at
#  an angle measures its fields in its own TEST frame, and the two are
#  reconciled in exactly one of two ways:
#
#      (A)  rotate the measured fields (and mesh, and wrench) into the material
#           frame, and use the material-frame D unchanged;
#      (B)  leave every measurement in its own test frame and rotate D into
#           that frame instead.
#
#  Doing both would rotate twice.  This module provides (B).
#
#  Internal virtual work is a frame-invariant scalar,
#
#      kappa*_lab^T D_lab kappa_lab  =  kappa*_mat^T D_mat kappa_mat
#
#  and with  kappa_mat = T(phi) kappa_lab  that forces
#
#      D_lab = T(phi)^T D_mat T(phi).
#
#  THE FACTOR OF TWO.  Row 3 of the generalised strain is the ENGINEERING
#  shear/twist -- vfm_mesh builds B[:, 2, :] = -2 * N_xy, and vfm_data_io reads
#  a kappa_xy that already carries the same factor.  The transformation that
#  acts on an engineering-shear vector is therefore NOT the plain tensor
#  rotation T_sigma, but the Reuter-adjusted
#
#      T = R T_sigma R^-1,     R = diag(1, 1, 2)
#
#        T(phi) = [[   c^2,   s^2,     c s   ],
#                  [   s^2,   c^2,    -c s   ],
#                  [-2 c s, 2 c s, c^2 - s^2 ]]
#
#  Using T_sigma on both sides instead is wrong by factors of two in the
#  16 / 26 / 66 entries.  Two known cases pin the convention down, and
#  validate_rotation.py asserts both:
#
#      phi = 90 deg  ->  D11 <-> D22 swap, D12 and D66 unchanged, D16 = D26 = 0
#      phi = 45 deg  ->  D_lab[0,0] = (D11 + D22 + 2 D12) / 4 + D66
#                        D_lab[2,2] = (D11 + D22 - 2 D12) / 4
#                        D_lab[0,2] = (D11 - D22) / 4
#
#  Because D_lab is LINEAR in the entries of D_mat, the rotated basis matrices
#  are still constant,
#
#      basis_lab_k = T^T basis_mat_k T,
#
#  so  D(theta) = sum_k theta_k basis_k  survives the rotation untouched and
#  the whole downstream solver is unaffected.  That is the property that makes
#  route (B) cheap, and _check_linear_decomposition re-proves it every call.
#
#  SIGN CONVENTION.  ``angle_deg`` is the angle from the lab (test-frame) +x
#  axis to the material 1-axis, counter-clockwise positive.
# =============================================================================

def engineering_rotation_matrix(angle_deg: float,
                                n_generalised: int = 3) -> np.ndarray:
    """``T(phi)`` mapping a lab-frame generalised strain to the material frame.

    Parameters
    ----------
    angle_deg : float
        Angle from the lab +x axis to the material 1-axis, CCW positive.
    n_generalised : int
        3 for a single 3x3 block (``in_plane`` or ``out_of_plane``), 6 for the
        stacked ABD (``both``), which gets ``blockdiag(T, T)`` -- the in-plane
        block uses engineering shear too, so it is the very same ``T``.

    Notes
    -----
    This is the Reuter-adjusted (classical-lamination-theory) transformation
    for an ENGINEERING-shear strain vector, not the plain tensor rotation.
    See the section header.
    """
    if n_generalised not in (3, 6):
        raise ValueError(
            f"n_generalised must be 3 or 6; got {n_generalised}."
        )

    phi = np.deg2rad(float(angle_deg))
    c = np.cos(phi)
    s = np.sin(phi)

    T = np.array([
        [c * c,        s * s,        c * s        ],
        [s * s,        c * c,       -c * s        ],
        [-2.0 * c * s, 2.0 * c * s,  c * c - s * s],
    ], dtype=float)

    if n_generalised == 3:
        return T

    T6 = np.zeros((6, 6), dtype=float)
    T6[:3, :3] = T
    T6[3:, 3:] = T
    return T6


def rotate_constitutive_model(model: ConstitutiveModel,
                              angle_deg: float) -> ConstitutiveModel:
    """Express ``model`` in a test frame rotated by ``angle_deg``.

    The parameter vector and its labels are NOT touched -- ``theta`` stays in
    the material frame, which is what lets several differently-oriented
    experiments share one unknown vector in the stacked/GLS identification.
    Only ``material_matrix`` and ``basis_matrices`` move.

    ``angle_deg == 0`` returns the input object unchanged, so the default path
    is provably bit-identical to a run without any rotation at all.
    """
    angle = float(angle_deg)
    if angle == 0.0:
        return model

    n_gen = model.n_generalised_strain
    T = engineering_rotation_matrix(angle, n_generalised=n_gen)

    rotated_matrix = T.T @ np.asarray(model.material_matrix, dtype=float) @ T
    rotated_matrix = 0.5 * (rotated_matrix + rotated_matrix.T)

    basis = np.asarray(model.basis_matrices, dtype=float)
    rotated_basis = np.einsum("ai,kab,bj->kij", T, basis, T)
    rotated_basis = 0.5 * (rotated_basis
                           + np.transpose(rotated_basis, (0, 2, 1)))

    #  The whole point of route (B): rotation is linear in D, so the rotated
    #  basis is still a valid constant decomposition of the rotated matrix.
    _check_linear_decomposition(
        model.parameters, rotated_basis, rotated_matrix,
        f"rotate_constitutive_model(angle_deg={angle!r})")

    return ConstitutiveModel(
        material_type=model.material_type,
        analysis_type=model.analysis_type,
        parameters=model.parameters,
        parameter_labels=list(model.parameter_labels),
        material_matrix=rotated_matrix,
        basis_matrices=rotated_basis,
        orientation_deg=angle,
    )


# =============================================================================
#  Builder: isotropic, in-plane only  (A matrix, 3x3, 2 unknowns)
# -----------------------------------------------------------------------------
#  Layout of the 3x3 A matrix:
#
#      A = [[ A11   A12    0  ],
#           [ A12   A22    0  ],
#           [  0     0    A66 ]]
#
#  with the isotropic reductions  A22 = A11,  A66 = (A11 - A12)/2.
# =============================================================================

def _build_iso_inplane(parameters: np.ndarray) -> ConstitutiveModel:
    if parameters.size != 2:
        raise ValueError(
            "Isotropic in-plane model expects 2 parameters [A11, A12]; "
            f"got {parameters.size}."
        )

    a11, a12 = parameters
    a22 = a11
    a66 = 0.5 * (a11 - a12)

    A = np.array([
        [a11, a12, 0.0],
        [a12, a22, 0.0],
        [0.0, 0.0, a66],
    ], dtype=float)

    basis_matrices = np.zeros((2, 3, 3), dtype=float)
    # dA / dA11
    basis_matrices[0, 0, 0] = 1.0
    basis_matrices[0, 1, 1] = 1.0    # A22 = A11
    basis_matrices[0, 2, 2] = 0.5    # A66 = (A11 - A12)/2 -> +1/2 on A11
    # dA / dA12
    basis_matrices[1, 0, 1] = 1.0
    basis_matrices[1, 1, 0] = 1.0
    basis_matrices[1, 2, 2] = -0.5   # A66 -> -1/2 on A12

    _check_linear_decomposition(parameters, basis_matrices, A,
                                "_build_iso_inplane")

    return ConstitutiveModel(
        material_type="iso",
        analysis_type="in_plane",
        parameters=parameters,
        parameter_labels=["A11", "A12"],
        material_matrix=A,
        basis_matrices=basis_matrices,
    )


# =============================================================================
#  Builder: isotropic, out-of-plane only  (D matrix, 3x3, 2 unknowns)
# -----------------------------------------------------------------------------
#  Layout of the 3x3 D matrix:
#
#      D = [[ D11   D12    0  ],
#           [ D12   D22    0  ],
#           [  0     0    D66 ]]
#
#  with the isotropic reductions  D22 = D11,  D66 = (D11 - D12)/2.
# =============================================================================

def _build_iso_outofplane(parameters: np.ndarray) -> ConstitutiveModel:
    if parameters.size != 2:
        raise ValueError(
            "Isotropic out-of-plane model expects 2 parameters [D11, D12]; "
            f"got {parameters.size}."
        )

    d11, d12 = parameters
    d22 = d11
    d66 = 0.5 * (d11 - d12)

    D = np.array([
        [d11, d12, 0.0],
        [d12, d22, 0.0],
        [0.0, 0.0, d66],
    ], dtype=float)

    basis_matrices = np.zeros((2, 3, 3), dtype=float)
    # dD / dD11
    basis_matrices[0, 0, 0] = 1.0
    basis_matrices[0, 1, 1] = 1.0
    basis_matrices[0, 2, 2] = 0.5
    # dD / dD12
    basis_matrices[1, 0, 1] = 1.0
    basis_matrices[1, 1, 0] = 1.0
    basis_matrices[1, 2, 2] = -0.5

    _check_linear_decomposition(parameters, basis_matrices, D,
                                "_build_iso_outofplane")

    return ConstitutiveModel(
        material_type="iso",
        analysis_type="out_of_plane",
        parameters=parameters,
        parameter_labels=["D11", "D12"],
        material_matrix=D,
        basis_matrices=basis_matrices,
    )


# =============================================================================
#  Builder: orthotropic, in-plane only  (A matrix, 3x3, 4 unknowns)
# -----------------------------------------------------------------------------
#  Layout of the 3x3 A matrix:
#
#      A = [[ A11   A12    0  ],
#           [ A12   A22    0  ],
#           [  0     0    A66 ]]
#
#  All four entries are independent.  No A16 / A26 terms are included.
# =============================================================================

def _build_orthotropic_inplane(parameters: np.ndarray) -> ConstitutiveModel:
    if parameters.size != 4:
        raise ValueError(
            "Orthotropic in-plane model expects 4 parameters "
            f"[A11, A12, A22, A66]; got {parameters.size}."
        )

    a11, a12, a22, a66 = parameters

    A = np.array([
        [a11, a12, 0.0],
        [a12, a22, 0.0],
        [0.0, 0.0, a66],
    ], dtype=float)

    basis_matrices = np.zeros((4, 3, 3), dtype=float)
    # dA / dA11
    basis_matrices[0, 0, 0] = 1.0
    # dA / dA12
    basis_matrices[1, 0, 1] = 1.0
    basis_matrices[1, 1, 0] = 1.0
    # dA / dA22
    basis_matrices[2, 1, 1] = 1.0
    # dA / dA66
    basis_matrices[3, 2, 2] = 1.0

    _check_linear_decomposition(parameters, basis_matrices, A,
                                "_build_orthotropic_inplane")

    return ConstitutiveModel(
        material_type="orthotropic",
        analysis_type="in_plane",
        parameters=parameters,
        parameter_labels=["A11", "A12", "A22", "A66"],
        material_matrix=A,
        basis_matrices=basis_matrices,
    )


# =============================================================================
#  Builder: orthotropic, out-of-plane only  (D matrix, 3x3, 4 unknowns)
# -----------------------------------------------------------------------------
#  Layout of the 3x3 D matrix:
#
#      D = [[ D11   D12    0  ],
#           [ D12   D22    0  ],
#           [  0     0    D66 ]]
#
#  All four entries are independent.  No D16 / D26 terms are included.
# =============================================================================

def _build_orthotropic_outofplane(parameters: np.ndarray) -> ConstitutiveModel:
    if parameters.size != 4:
        raise ValueError(
            "Orthotropic out-of-plane model expects 4 parameters "
            f"[D11, D12, D22, D66]; got {parameters.size}."
        )

    d11, d12, d22, d66 = parameters

    D = np.array([
        [d11, d12, 0.0],
        [d12, d22, 0.0],
        [0.0, 0.0, d66],
    ], dtype=float)

    basis_matrices = np.zeros((4, 3, 3), dtype=float)
    # dD / dD11
    basis_matrices[0, 0, 0] = 1.0
    # dD / dD12
    basis_matrices[1, 0, 1] = 1.0
    basis_matrices[1, 1, 0] = 1.0
    # dD / dD22
    basis_matrices[2, 1, 1] = 1.0
    # dD / dD66
    basis_matrices[3, 2, 2] = 1.0

    _check_linear_decomposition(parameters, basis_matrices, D,
                                "_build_orthotropic_outofplane")

    return ConstitutiveModel(
        material_type="orthotropic",
        analysis_type="out_of_plane",
        parameters=parameters,
        parameter_labels=["D11", "D12", "D22", "D66"],
        material_matrix=D,
        basis_matrices=basis_matrices,
    )


# =============================================================================
#  Builder: anisotropic, in-plane only  (A matrix, 3x3, 6 unknowns)
# -----------------------------------------------------------------------------
#  Layout of the 3x3 A matrix:
#
#      A = [[ A11   A12   A16 ],
#           [ A12   A22   A26 ],
#           [ A16   A26   A66 ]]
#
#  All six symmetric entries are independent.
# =============================================================================

def _build_anisotropic_inplane(parameters: np.ndarray) -> ConstitutiveModel:
    if parameters.size != 6:
        raise ValueError(
            "Anisotropic in-plane model expects 6 parameters "
            f"[A11, A12, A22, A66, A16, A26]; got {parameters.size}."
        )

    a11, a12, a22, a66, a16, a26 = parameters

    A = np.array([
        [a11, a12, a16],
        [a12, a22, a26],
        [a16, a26, a66],
    ], dtype=float)

    basis_matrices = np.zeros((6, 3, 3), dtype=float)
    # dA / dA11
    basis_matrices[0, 0, 0] = 1.0
    # dA / dA12
    basis_matrices[1, 0, 1] = 1.0
    basis_matrices[1, 1, 0] = 1.0
    # dA / dA22
    basis_matrices[2, 1, 1] = 1.0
    # dA / dA66
    basis_matrices[3, 2, 2] = 1.0
    # dA / dA16
    basis_matrices[4, 0, 2] = 1.0
    basis_matrices[4, 2, 0] = 1.0
    # dA / dA26
    basis_matrices[5, 1, 2] = 1.0
    basis_matrices[5, 2, 1] = 1.0

    _check_linear_decomposition(parameters, basis_matrices, A,
                                "_build_anisotropic_inplane")

    return ConstitutiveModel(
        material_type="anisotropic",
        analysis_type="in_plane",
        parameters=parameters,
        parameter_labels=["A11", "A12", "A22", "A66", "A16", "A26"],
        material_matrix=A,
        basis_matrices=basis_matrices,
    )


# =============================================================================
#  Builder: anisotropic, out-of-plane only  (D matrix, 3x3, 6 unknowns)
# -----------------------------------------------------------------------------
#  Layout of the 3x3 D matrix:
#
#      D = [[ D11   D12   D16 ],
#           [ D12   D22   D26 ],
#           [ D16   D26   D66 ]]
#
#  All six symmetric entries are independent.
# =============================================================================

def _build_anisotropic_outofplane(parameters: np.ndarray) -> ConstitutiveModel:
    if parameters.size != 6:
        raise ValueError(
            "Anisotropic out-of-plane model expects 6 parameters "
            f"[D11, D12, D22, D66, D16, D26]; got {parameters.size}."
        )

    d11, d12, d22, d66, d16, d26 = parameters

    D = np.array([
        [d11, d12, d16],
        [d12, d22, d26],
        [d16, d26, d66],
    ], dtype=float)

    basis_matrices = np.zeros((6, 3, 3), dtype=float)
    # dD / dD11
    basis_matrices[0, 0, 0] = 1.0
    # dD / dD12
    basis_matrices[1, 0, 1] = 1.0
    basis_matrices[1, 1, 0] = 1.0
    # dD / dD22
    basis_matrices[2, 1, 1] = 1.0
    # dD / dD66
    basis_matrices[3, 2, 2] = 1.0
    # dD / dD16
    basis_matrices[4, 0, 2] = 1.0
    basis_matrices[4, 2, 0] = 1.0
    # dD / dD26
    basis_matrices[5, 1, 2] = 1.0
    basis_matrices[5, 2, 1] = 1.0

    _check_linear_decomposition(parameters, basis_matrices, D,
                                "_build_anisotropic_outofplane")

    return ConstitutiveModel(
        material_type="anisotropic",
        analysis_type="out_of_plane",
        parameters=parameters,
        parameter_labels=["D11", "D12", "D22", "D66", "D16", "D26"],
        material_matrix=D,
        basis_matrices=basis_matrices,
    )


# =============================================================================
#  Builder: isotropic, both  (6x6 ABD, B = 0, 4 unknowns)
# -----------------------------------------------------------------------------
#  Layout of the 6x6 ABD with B = 0:
#
#      ABD = [[ A11   A12    0  ,    0     0    0  ],
#             [ A12   A22    0  ,    0     0    0  ],
#             [  0     0    A66 ,    0     0    0  ],
#             ----------------------------------------
#             [  0     0     0  ,   D11   D12   0  ],
#             [  0     0     0  ,   D12   D22   0  ],
#             [  0     0     0  ,    0     0   D66 ]]
#
#  with the isotropic reductions  A22 = A11,  A66 = (A11 - A12)/2,
#  D22 = D11,  D66 = (D11 - D12)/2.
# =============================================================================

def _build_iso_both(parameters: np.ndarray) -> ConstitutiveModel:
    if parameters.size != 4:
        raise ValueError(
            "Isotropic both-mode model expects 4 parameters "
            f"[A11, A12, D11, D12]; got {parameters.size}."
        )

    a11, a12, d11, d12 = parameters
    a22 = a11
    a66 = 0.5 * (a11 - a12)
    d22 = d11
    d66 = 0.5 * (d11 - d12)

    A = np.array([
        [a11, a12, 0.0],
        [a12, a22, 0.0],
        [0.0, 0.0, a66],
    ], dtype=float)
    D = np.array([
        [d11, d12, 0.0],
        [d12, d22, 0.0],
        [0.0, 0.0, d66],
    ], dtype=float)

    zero_3x3 = np.zeros((3, 3), dtype=float)
    ABD = np.block([
        [A,        zero_3x3],
        [zero_3x3, D       ],
    ])

    basis_matrices = np.zeros((4, 6, 6), dtype=float)
    # dABD / dA11
    basis_matrices[0, 0, 0] = 1.0
    basis_matrices[0, 1, 1] = 1.0
    basis_matrices[0, 2, 2] = 0.5
    # dABD / dA12
    basis_matrices[1, 0, 1] = 1.0
    basis_matrices[1, 1, 0] = 1.0
    basis_matrices[1, 2, 2] = -0.5
    # dABD / dD11
    basis_matrices[2, 3, 3] = 1.0
    basis_matrices[2, 4, 4] = 1.0
    basis_matrices[2, 5, 5] = 0.5
    # dABD / dD12
    basis_matrices[3, 3, 4] = 1.0
    basis_matrices[3, 4, 3] = 1.0
    basis_matrices[3, 5, 5] = -0.5

    _check_linear_decomposition(parameters, basis_matrices, ABD,
                                "_build_iso_both")

    return ConstitutiveModel(
        material_type="iso",
        analysis_type="both",
        parameters=parameters,
        parameter_labels=["A11", "A12", "D11", "D12"],
        material_matrix=ABD,
        basis_matrices=basis_matrices,
    )


# =============================================================================
#  Linear-decomposition self-test
# =============================================================================

def _check_linear_decomposition(parameters: np.ndarray,
                                basis_matrices: np.ndarray,
                                material_matrix: np.ndarray,
                                builder_name: str) -> None:
    reconstructed = np.einsum("k,kij->ij", parameters, basis_matrices)
    if not np.allclose(reconstructed, material_matrix, atol=1.0e-12):
        raise RuntimeError(
            f"Internal error: linear decomposition does not match the "
            f"closed-form matrix in {builder_name}."
        )

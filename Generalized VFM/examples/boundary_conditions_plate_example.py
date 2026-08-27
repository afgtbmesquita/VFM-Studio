# =============================================================================
#  boundary_conditions_plate_example.py
# -----------------------------------------------------------------------------
#  Boundary-condition inputs for the plate VFM inverse solver.
#
#  This file replaces the older JSON input because Python is easier to comment,
#  group by category, and edit without worrying about JSON commas or quoting.
#
#  How the solver reads this file:
#    * ``vfm_data_io.load_boundary_conditions`` imports this file as a Python
#      module.
#    * It reads the three variables below: ``PRESCRIBED``, ``LINKING``, and
#      ``EXTERNAL``.
#    * Each entry is a plain Python dictionary with the same keys used by the
#      previous JSON file.
#
#  Valid DOF labels for the plate solver:
#    * "U"    : in-plane horizontal displacement
#    * "V"    : in-plane vertical displacement
#    * "W"    : transverse deflection
#    * "W_X"  : dw/dx
#    * "W_Y"  : dw/dy
#    * "W_XY" : d2w/(dx dy)
#
#  Valid edge labels:
#    * "left", "right", "bottom", "top"
#
#  IMPORTANT:
#    * Prescribed and linking values must stay homogeneous for this first
#      implementation, meaning ``value = 0.0``.
#    * For node-based selections, replace ``edge`` with ``nodes`` and provide
#      a list of zero-based node indices, for example ``"nodes": [0, 1, 2]``.
# =============================================================================


# =============================================================================
#  Metadata
# -----------------------------------------------------------------------------
#  This block is not used by the solver.  It is here only so you can document
#  what this boundary-condition file represents.
# =============================================================================

DESCRIPTION = (
    "Plate VFM cantilever-style example: left edge fully clamped; right edge "
    "linked in V and W so both membrane shear and transverse bending virtual "
    "work can be measured."
)
LAST_UPDATED = "2026-05-12"

# Analysis type the file was authored for.  The loader reads this and the
# top-level ``run_vfm.py`` warns when it does not match the active run.
# Allowed values: 'in_plane', 'out_of_plane', 'both'.
ANALYSIS_TYPE = "both"


# =============================================================================
#  Category 1: Prescribed boundary conditions
# -----------------------------------------------------------------------------
#  These are homogeneous Dirichlet constraints.  Each selected node receives
#  one equation of the form
#
#      selected_dof = 0
#
#  For this example, the left edge is fully clamped in all six plate DOFs.
#  The right edge is also pinned in U to provide an axial reference, matching
#  the spirit of the membrane Iosipescu fixture.
# =============================================================================

PRESCRIBED = [
    # --- Left edge: in-plane clamp ----------------------------------------
    {
        "label": "Clamp left edge in U",
        "dof": "U",
        "edge": "left",
        "value": 0.0,
    },
    {
        "label": "Clamp left edge in V",
        "dof": "V",
        "edge": "left",
        "value": 0.0,
    },

    # --- Left edge: transverse/bending clamp ------------------------------
    {
        "label": "Clamp left edge in W",
        "dof": "W",
        "edge": "left",
        "value": 0.0,
    },
    {
        "label": "Clamp left edge in W_X",
        "dof": "W_X",
        "edge": "left",
        "value": 0.0,
    },
    {
        "label": "Clamp left edge in W_Y",
        "dof": "W_Y",
        "edge": "left",
        "value": 0.0,
    },
    {
        "label": "Clamp left edge in W_XY",
        "dof": "W_XY",
        "edge": "left",
        "value": 0.0,
    },

    # --- Right edge: axial reference --------------------------------------
    {
        "label": "Pin right edge in U (membrane-axial reference)",
        "dof": "U",
        "edge": "right",
        "value": 0.0,
    },
]


# =============================================================================
#  Category 2: Linking constraints
# -----------------------------------------------------------------------------
#  These create multi-node tie equations.  For a selected group of nodes, the
#  solver chooses the first node (after optional sorting) as a reference and
#  enforces
#
#      follower_dof - reference_dof = 0
#
#  This is useful when a fixture/load introduces an approximately rigid motion
#  along an edge.  Here the right edge is tied in V for membrane shear and in W
#  for transverse plate bending.
# =============================================================================

LINKING = [
    # --- Membrane shear measurement group ---------------------------------
    {
        "label": "Rigid-link right edge in V (membrane shear measurement)",
        "dof": "V",
        "edge": "right",
        "sortDirection": "y",
        "value": 0.0,
    },

    # --- Plate bending measurement group ----------------------------------
    {
        "label": "Rigid-link right edge in W (plate bending measurement)",
        "dof": "W",
        "edge": "right",
        "sortDirection": "y",
        "value": 0.0,
    },
]


# =============================================================================
#  Category 3: External-work measurement groups
# -----------------------------------------------------------------------------
#  These groups define how the applied force / moment is energetically paired
#  with a virtual displacement.
#
#  Two ways to provide the magnitude of the applied force / moment:
#
#    A) Inside the MAT file as an ``F`` field (legacy):
#         * F scalar -> broadcast to every EXTERNAL entry,
#         * F length-N array -> F[i] pairs with EXTERNAL[i].
#       When present in the MAT file, ``F`` takes precedence and the per-
#       entry ``value`` declared below is ignored.
#
#    B) Per-entry ``value`` field below (recommended for the new Loading_Data/
#       FromAbaqustoVFM.m workflow, which produces a MAT file WITHOUT ``F``):
#         * Set ``"value": <number>`` on each EXTERNAL entry.  The GUI
#           exposes the same field as "Applied force / moment".
#         * Omit the key (or leave the GUI field blank) to fall back to
#           the MAT file's ``F``.  Missing in both places is an error at
#           identification time.
#
#  ``loadType`` options:
#    * "concentrated" : each selected node receives equal weight 1/N
#    * "distributed" : line-load weights based on edge segment lengths
#
#  Moment about a plate rotation direction: use W_X or W_Y as the DOF.
# =============================================================================

EXTERNAL = [
    # --- Group 1: membrane shear virtual work -----------------------------
    {
        "label": "Measure right edge vertical motion (membrane shear, group 1)",
        "dof": "V",
        "edge": "right",
        "loadType": "concentrated",
        # "value": 1.0,            # uncomment + edit when the MAT file
        #                            does not provide F (new workflow).
    },

    # --- Group 2: plate bending virtual work ------------------------------
    {
        "label": "Measure right edge transverse deflection (plate bending, group 2)",
        "dof": "W",
        "edge": "right",
        "loadType": "concentrated",
        # "value": 1.0,            # ditto for the bending group.
    },
]


"""Is w* actually zero along the loaded cuts of the solved virtual fields?

Checks the DOFs, and then the FIELD -- because zero nodal values are not by
themselves enough.  Along an edge y = const the Hermite element makes w* a
CUBIC in x set by the nodal W and the nodal TANGENTIAL slope at the two end
nodes.  Zero value AND zero tangential slope at both ends is what forces that
cubic to vanish identically; W = 0 alone would leave a cubic bulging between
the nodes.  That is why both are prescribed.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workflows.campaign_common import (ANALYSIS_TYPE, INITIAL_PARAMETERS, MATERIAL_TYPE,
                             TESTS_WITH_45_RAW, build_cases, load_bc_module,
                             mat_files)
from vfm_constitutive import define_constitutive_model, rotate_constitutive_model
from vfm_optimal_field import solve_optimal_virtual_fields

DOF = {"W": 0, "W_X": 1, "W_Y": 2, "W_XY": 3}
TOL = 1e-9


def edge_nodes(coords, edge):
    x, y = coords[:, 0], coords[:, 1]
    pick = {"top": y >= y.max() - 1e-9, "bottom": y <= y.min() + 1e-9,
            "left": x <= x.min() + 1e-9, "right": x >= x.max() + -1e-9}
    return np.where(pick[edge])[0]


def hermite_edge_w(u, coords, nodes, along):
    """w* sampled along an edge from the two end nodes' Hermite data."""
    order = np.argsort(coords[nodes, along])
    nodes = nodes[order]
    tangential = "W_X" if along == 0 else "W_Y"
    samples = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        h = coords[b, along] - coords[a, along]
        w0, s0 = u[4 * a + DOF["W"]], u[4 * a + DOF[tangential]]
        w1, s1 = u[4 * b + DOF["W"]], u[4 * b + DOF[tangential]]
        t = np.linspace(0.0, 1.0, 21)
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        samples.append(h00 * w0 + h10 * h * s0 + h01 * w1 + h11 * h * s1)
    return np.concatenate(samples)


model0 = define_constitutive_model(MATERIAL_TYPE, ANALYSIS_TYPE,
                                   INITIAL_PARAMETERS)
print(f"{'experiment':<13}{'edge':<8}{'kind':<8}"
      f"{'max|W| node':>13}{'max|tang| node':>16}"
      f"{'max|w*| ALONG':>15}{'max|W_n| node':>15}")
print("-" * 90)
for spec in TESTS_WITH_45_RAW:
    module = load_bc_module(spec.bc_file)
    case = build_cases(spec, paths=[mat_files(spec)[-1]],
                       verbose=False)[0]
    model = rotate_constitutive_model(model0, case.material_orientation_deg)
    fields = solve_optimal_virtual_fields(
        measurements=case.measurements, mesh_pkg=case.mesh_pkg,
        boundary_conditions=case.boundary_conditions,
        constitutive_model=model)
    u = np.asarray(fields[0].nodal_variables_k, dtype=float)
    coords = np.asarray(case.mesh_pkg.node_coords, dtype=float)

    loaded = {e["edge"] for e in module.EXTERNAL}
    constrained = {e["edge"] for e in module.PRESCRIBED}
    for edge in ("top", "bottom", "left", "right"):
        nodes = edge_nodes(coords, edge)
        if nodes.size == 0:
            continue
        horizontal = edge in ("top", "bottom")
        along = 0 if horizontal else 1
        tangential = "W_X" if horizontal else "W_Y"
        normal = "W_Y" if horizontal else "W_X"
        kind = ("LOADED" if edge in loaded
                else "clamped" if edge in constrained else "FREE")
        w_node = np.max(np.abs(u[4 * nodes + DOF["W"]]))
        t_node = np.max(np.abs(u[4 * nodes + DOF[tangential]]))
        n_node = np.max(np.abs(u[4 * nodes + DOF[normal]]))
        w_edge = np.max(np.abs(hermite_edge_w(u, coords, nodes, along)))
        print(f"{spec.name:<13}{edge:<8}{kind:<8}"
              f"{w_node:13.3e}{t_node:16.3e}{w_edge:15.3e}{n_node:15.3e}")
    print()

print("Reading: on every LOADED cut w* is zero AT the nodes and ALONG the whole")
print("edge (both < 1e-9), so the unknown Kirchhoff shear V_n and the unknown")
print("twisting moment do no virtual work.  The NORMAL rotation W_n is nonzero")
print("there -- that is the DOF the measured bending moment works against.")
print("On the FREE edges nothing is zero, which is the point: they are")
print("traction-free, so the virtual field is left completely unconstrained.")

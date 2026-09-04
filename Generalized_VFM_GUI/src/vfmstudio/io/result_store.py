"""HDF5 snapshot of identification results. Duck-typed for the GUI, not a pickle."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np


def _write_array(group, name: str, value) -> None:
    if value is None:
        return
    array = np.asarray(value)
    if array.size == 0 and array.ndim == 0:
        return
    group.create_dataset(name, data=array)


def _read_array(group, name: str):
    if name not in group:
        return None
    return np.asarray(group[name][()])


def _write_string_list(group, name: str, values: List[str]) -> None:
    encoded = [str(item).encode("utf-8") for item in values]
    if not encoded:
        return
    group.create_dataset(name, data=np.array(encoded, dtype="S"))


def _read_string_list(group, name: str) -> List[str]:
    if name not in group:
        return []
    raw = group[name][()]
    items = np.atleast_1d(raw)
    out = []
    for item in items:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def dump_result(group, result: Any) -> None:
    if result is None:
        return
    ident = group.require_group("identification")
    _write_array(ident, "final_parameters", getattr(result, "final_parameters", None))
    _write_array(ident, "initial_parameters", getattr(result, "initial_parameters", None))
    _write_array(
        ident, "parameter_covariance", getattr(result, "parameter_covariance", None)
    )
    _write_array(ident, "global_covariance", getattr(result, "global_covariance", None))
    _write_array(ident, "K_matrix", getattr(result, "K_matrix", None))
    _write_array(ident, "U_vector", getattr(result, "U_vector", None))
    if getattr(result, "converged", None) is not None:
        ident.attrs["converged"] = bool(result.converged)
    if getattr(result, "iterations_completed", None) is not None:
        ident.attrs["iterations_completed"] = int(result.iterations_completed)
    if getattr(result, "weighting", None):
        ident.attrs["weighting"] = str(result.weighting)

    model = getattr(result, "constitutive_model", None)
    labels = list(getattr(model, "parameter_labels", ()) or ())
    if labels:
        _write_string_list(ident, "parameter_labels", labels)

    diagnostics = list(getattr(result, "iteration_diagnostics", None) or [])
    if diagnostics:
        hist = ident.require_group("history")
        hist.attrs["n"] = len(diagnostics)
        for index, row in enumerate(diagnostics):
            item = hist.require_group(str(index))
            item.attrs["iteration"] = int(getattr(row, "iteration", index))
            _write_array(item, "parameters", getattr(row, "parameters", None))

    fields = list(getattr(result, "optimal_field_results", None) or [])
    if fields:
        vf = ident.require_group("virtual_fields")
        vf.attrs["n"] = len(fields)
        for index, field in enumerate(fields):
            _write_array(
                vf,
                f"nodal_{index}",
                getattr(field, "nodal_variables_k", None),
            )

    designs = list(getattr(result, "case_K_matrices", None) or [])
    observations = list(getattr(result, "case_U_vectors", None) or [])
    reports = list(getattr(result, "experiments", None) or [])
    n_cases = min(len(designs), len(observations), len(reports) or len(designs))
    if n_cases:
        cases = ident.require_group("cases")
        cases.attrs["n"] = n_cases
        for index in range(n_cases):
            item = cases.require_group(str(index))
            _write_array(item, "design", designs[index] if index < len(designs) else None)
            _write_array(
                item,
                "observation",
                observations[index] if index < len(observations) else None,
            )
            if index < len(reports):
                report = reports[index]
                item.attrs["experiment_id"] = int(
                    getattr(report, "experiment_id", index + 1)
                )
                item.attrs["label"] = str(getattr(report, "label", "") or "")
                item.attrs["n_steps"] = int(getattr(report, "n_steps", 0) or 0)
                _write_array(item, "covariance", getattr(report, "covariance", None))
                _write_array(
                    item, "step_estimates", getattr(report, "step_estimates", None)
                )
                labels_step = list(getattr(report, "step_labels", None) or [])
                if labels_step:
                    _write_string_list(item, "step_labels", labels_step)


def load_result(group) -> Optional[Any]:
    if "identification" not in group:
        return None
    ident = group["identification"]
    theta = _read_array(ident, "final_parameters")
    if theta is None:
        return None
    labels = _read_string_list(ident, "parameter_labels")
    model = SimpleNamespace(parameter_labels=labels) if labels else None

    history = []
    if "history" in ident:
        n = int(ident["history"].attrs.get("n", 0))
        for index in range(n):
            key = str(index)
            if key not in ident["history"]:
                continue
            item = ident["history"][key]
            history.append(
                SimpleNamespace(
                    iteration=int(item.attrs.get("iteration", index)),
                    parameters=_read_array(item, "parameters"),
                )
            )

    fields = []
    if "virtual_fields" in ident:
        n = int(ident["virtual_fields"].attrs.get("n", 0))
        for index in range(n):
            nodal = _read_array(ident["virtual_fields"], f"nodal_{index}")
            fields.append(SimpleNamespace(nodal_variables_k=nodal))

    designs: List = []
    observations: List = []
    reports: List = []
    if "cases" in ident:
        n = int(ident["cases"].attrs.get("n", 0))
        for index in range(n):
            key = str(index)
            if key not in ident["cases"]:
                continue
            item = ident["cases"][key]
            designs.append(_read_array(item, "design"))
            observations.append(_read_array(item, "observation"))
            reports.append(
                SimpleNamespace(
                    experiment_id=int(item.attrs.get("experiment_id", index + 1)),
                    label=str(item.attrs.get("label", "") or ""),
                    n_steps=int(item.attrs.get("n_steps", 0) or 0),
                    covariance=_read_array(item, "covariance"),
                    step_estimates=_read_array(item, "step_estimates"),
                    step_labels=_read_string_list(item, "step_labels"),
                )
            )

    return SimpleNamespace(
        final_parameters=theta,
        initial_parameters=_read_array(ident, "initial_parameters"),
        parameter_covariance=_read_array(ident, "parameter_covariance"),
        global_covariance=_read_array(ident, "global_covariance"),
        K_matrix=_read_array(ident, "K_matrix"),
        U_vector=_read_array(ident, "U_vector"),
        converged=bool(ident.attrs.get("converged", True)),
        iterations_completed=int(ident.attrs.get("iterations_completed", 0)),
        weighting=str(ident.attrs.get("weighting", "") or ""),
        constitutive_model=model,
        iteration_diagnostics=history,
        optimal_field_results=fields,
        case_K_matrices=designs,
        case_U_vectors=observations,
        experiments=reports,
    )


def dump_identifiability(group, ident: Any) -> None:
    if ident is None:
        return
    root = group.require_group("identifiability")
    aggregate = getattr(ident, "aggregate", None)
    if aggregate is not None:
        agg = root.require_group("aggregate")
        for name in ("condition_number", "determinant"):
            value = getattr(aggregate, name, None)
            if value is not None:
                agg.attrs[name] = float(value)
        if getattr(aggregate, "used_pseudoinverse", None) is not None:
            agg.attrs["used_pseudoinverse"] = bool(aggregate.used_pseudoinverse)
        _write_array(
            agg, "singular_values", getattr(aggregate, "singular_values", None)
        )
    labels = list(getattr(ident, "parameter_labels", ()) or ())
    if labels:
        _write_string_list(root, "parameter_labels", labels)
    cases = list(getattr(ident, "cases", None) or [])
    if cases:
        folder = root.require_group("cases")
        folder.attrs["n"] = len(cases)
        for index, case in enumerate(cases):
            item = folder.require_group(str(index))
            item.attrs["label"] = str(getattr(case, "label", "") or "")
            _write_array(item, "CID", getattr(case, "CID", None))
            _write_array(item, "points", getattr(case, "points", None))


def load_identifiability(group) -> Optional[Any]:
    if "identifiability" not in group:
        return None
    root = group["identifiability"]
    aggregate = None
    if "aggregate" in root:
        agg = root["aggregate"]
        aggregate = SimpleNamespace(
            condition_number=float(agg.attrs.get("condition_number", float("nan"))),
            determinant=float(agg.attrs.get("determinant", float("nan"))),
            used_pseudoinverse=bool(agg.attrs.get("used_pseudoinverse", False)),
            singular_values=_read_array(agg, "singular_values"),
        )
    cases = []
    if "cases" in root:
        n = int(root["cases"].attrs.get("n", 0))
        for index in range(n):
            key = str(index)
            if key not in root["cases"]:
                continue
            item = root["cases"][key]
            cases.append(
                SimpleNamespace(
                    label=str(item.attrs.get("label", "") or ""),
                    CID=_read_array(item, "CID"),
                    points=_read_array(item, "points"),
                )
            )
    if aggregate is None and not cases:
        return None
    return SimpleNamespace(
        aggregate=aggregate,
        cases=cases,
        parameter_labels=_read_string_list(root, "parameter_labels"),
    )


def dump_snapshot(
    handle,
    result: Any = None,
    identifiability: Any = None,
    uncertainty: Any = None,
    view_state: Optional[dict] = None,
) -> None:
    dump_result(handle, result)
    dump_identifiability(handle, identifiability)
    dump_uncertainty(handle, uncertainty)
    dump_view_state(handle, view_state)


def load_snapshot(handle) -> Tuple[Optional[Any], Optional[Any], Optional[Any], dict]:
    return (
        load_result(handle),
        load_identifiability(handle),
        load_uncertainty(handle),
        load_view_state(handle),
    )


def dump_uncertainty(handle, uncertainty: Any) -> None:
    if uncertainty is None:
        return
    import pickle

    try:
        raw = pickle.dumps(uncertainty, protocol=4)
    except Exception:
        return
    handle.create_dataset("uncertainty_pickle", data=np.frombuffer(raw, dtype=np.uint8))
    kind = getattr(uncertainty, "kind", None)
    if kind:
        handle.attrs["uncertainty_kind"] = str(kind)


def load_uncertainty(handle) -> Optional[Any]:
    if "uncertainty_pickle" not in handle:
        return None
    import pickle

    raw = np.asarray(handle["uncertainty_pickle"][()], dtype=np.uint8).tobytes()
    try:
        return pickle.loads(raw)
    except Exception:
        return None


def dump_view_state(handle, state: Optional[dict]) -> None:
    if not state:
        return
    group = handle.require_group("view_state")
    for key, value in state.items():
        if value is None:
            continue
        group.attrs[key] = value


def load_view_state(handle) -> dict:
    if "view_state" not in handle:
        return {}
    group = handle["view_state"]
    out = {}
    for key, value in group.attrs.items():
        if hasattr(value, "item"):
            value = value.item()
        out[str(key)] = value
    return out

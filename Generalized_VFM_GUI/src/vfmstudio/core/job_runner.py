"""Child-process Identify / UQ so the GUI can kill the solve."""

from __future__ import annotations

import sys
from pathlib import Path

from vfmstudio.platform.scipy_wdac import explain_native_import_error, prepare_scipy_native_imports


def _log(message: str) -> None:
    print(message, flush=True)


def run_job(kind: str, input_path: Path, output_path: Path, uq_kind: str = "CR2") -> None:
    from vfmstudio.core.jobs import (
        run_identifiability,
        run_identification,
        run_loo_refits,
        run_uncertainty,
    )
    from vfmstudio.io.hdf5_project import load_vfm_project, save_vfm

    prepare_scipy_native_imports()
    loaded = load_vfm_project(Path(input_path), bind_source=False)
    document = loaded.document
    kind = str(kind or "identify").strip().lower()
    uq_kind = str(uq_kind or loaded.view_state.get("uncertainty_kind") or "CR2")
    result = loaded.result
    ident = loaded.identifiability
    uncertainty = loaded.uncertainty

    if kind == "identify":
        _log("Building solver project and running FGLS…")
        result = run_identification(document, verbose=False)
        ident = None
        try:
            _log("Computing identifiability (H, SVD, CID)…")
            ident = run_identifiability(document, result)
        except Exception as exc:
            _log(f"Identifiability skipped: {exc}")
        uncertainty = None
        try:
            _log("Computing uncertainty quantification…")
            uncertainty = run_uncertainty(document, result, kind=uq_kind)
        except Exception as exc:
            _log(f"Uncertainty quantification skipped: {exc}")
    elif kind == "identifiability":
        if result is None:
            raise ValueError("Run identification first.")
        _log("Computing identifiability (H, SVD, CID)…")
        ident = run_identifiability(document, result)
    elif kind == "uncertainty":
        if result is None:
            raise ValueError("Run identification first.")
        _log(f"Computing uncertainty quantification ({uq_kind})…")
        uncertainty = run_uncertainty(document, result, kind=uq_kind)
    elif kind == "loo":
        if uncertainty is None:
            raise ValueError("Compute uncertainty quantification first.")
        _log("Leave-one-out FGLS refits…")
        uncertainty = run_loo_refits(
            document, uncertainty, kind=uq_kind, log=_log
        )
    else:
        raise ValueError(f"Unknown job kind {kind!r}.")

    save_vfm(
        document,
        Path(output_path),
        result=result,
        identifiability=ident,
        uncertainty=uncertainty,
        view_state=loaded.view_state,
        bind_source=False,
    )
    _log("Job finished.")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3:
        sys.stderr.write(
            "usage: python -m vfmstudio.core.job_runner "
            "KIND INPUT.vfm OUTPUT.vfm [UQ_KIND]\n"
        )
        return 2
    kind, input_path, output_path = args[0], args[1], args[2]
    uq_kind = args[3] if len(args) > 3 else "CR2"
    try:
        run_job(kind, Path(input_path), Path(output_path), uq_kind=uq_kind)
    except Exception as exc:
        sys.stderr.write(explain_native_import_error(exc) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

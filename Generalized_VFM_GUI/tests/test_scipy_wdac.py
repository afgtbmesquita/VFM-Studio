"""Windows Smart App Control can block optional SciPy native extensions."""

from __future__ import annotations

import sys

from vfmstudio.platform.scipy_wdac import (
    _TOEPLITZ_MODULE,
    blocked_extension_name,
    explain_native_import_error,
    prepare_scipy_native_imports,
    specfun_was_stubbed,
)


def test_explain_native_import_error_adds_hint():
    message = explain_native_import_error(
        ImportError(
            "DLL load failed while importing _solve_toeplitz: "
            "An Application Control policy has blocked this file."
        )
    )
    assert "Application Control" in message
    assert "Smart App Control" in message
    assert "_solve_toeplitz" in message


def test_explain_native_import_error_leaves_other_errors():
    assert explain_native_import_error(ValueError("bad window")) == "bad window"


def test_blocked_extension_name_parses_wdac_error():
    assert (
        blocked_extension_name(
            ImportError(
                "DLL load failed while importing _solve_toeplitz: "
                "An Application Control policy has blocked this file."
            )
        )
        == "_solve_toeplitz"
    )
    assert blocked_extension_name(ValueError("bad window")) is None


def test_specfun_stub_allows_ckdtree_and_gls_import():
    prepare_scipy_native_imports()
    from scipy.spatial import cKDTree
    from scipy.linalg import cholesky, solve_triangular

    assert cKDTree is not None
    assert cholesky is not None
    assert solve_triangular is not None
    import scipy.linalg._solve_toeplitz as toeplitz_mod

    assert hasattr(toeplitz_mod, "levinson")

    from vfmstudio.core.paths import ensure_core_on_path

    ensure_core_on_path()
    import vfm_gls_identification
    import vfm_noise_covariance

    assert hasattr(vfm_gls_identification, "run_gls_identification_cases")
    assert vfm_noise_covariance.cKDTree is not None


def test_blocked_toeplitz_still_imports_cholesky():
    """A stubbed _solve_toeplitz must not prevent scipy.linalg from loading."""
    import vfmstudio.platform.scipy_wdac as wdac

    wdac._install_flexible_stub(_TOEPLITZ_MODULE)
    wdac._PREPARED = False
    for key in list(sys.modules):
        if key == "scipy.linalg" or key.startswith("scipy.linalg."):
            if key not in wdac._STUBBED:
                sys.modules.pop(key, None)

    wdac.prepare_scipy_native_imports()
    from scipy.linalg import cholesky, solve_triangular
    import scipy.linalg._solve_toeplitz as toeplitz_mod

    assert cholesky is not None
    assert solve_triangular is not None
    assert callable(getattr(toeplitz_mod, "levinson", None))


def test_import_with_stubs_retries_unknown_blocked_pyd(monkeypatch):
    """A newly blocked SciPy extension is stubbed and the parent import retried."""
    import vfmstudio.platform.scipy_wdac as wdac

    real_import = __import__
    calls = {"n": 0}

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy.linalg" and not fromlist and level == 0 and calls["n"] == 0:
            calls["n"] += 1
            raise ImportError(
                "DLL load failed while importing _not_a_real_extension: "
                "An Application Control policy has blocked this file."
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    wdac._import_with_stubs("scipy.linalg")
    stub = sys.modules.get("scipy.linalg._not_a_real_extension")
    assert stub is not None
    assert callable(getattr(stub, "levinson", None))
    from scipy.linalg import cholesky

    assert cholesky is not None


def test_assemble_does_not_import_scipy_linalg_at_module_level():
    import inspect

    import vfmstudio.core.uncertainty.assemble as assemble

    source = inspect.getsource(assemble)
    assert "from scipy.linalg import" not in source.split("def _scipy_linalg")[0]
    assert hasattr(assemble, "_scipy_linalg")


def test_prepare_is_idempotent():
    prepare_scipy_native_imports()
    prepare_scipy_native_imports()
    if sys.platform == "win32":
        from scipy.spatial import cKDTree

        assert cKDTree is not None
    _ = specfun_was_stubbed()

"""Work around Windows Smart App Control blocking SciPy native extensions.

FGLS needs ``scipy.spatial.cKDTree`` and ``scipy.linalg`` (cholesky /
solve_triangular). Importing those packages also pulls optional Cython
extensions that Smart App Control / WDAC often blocks:

    DLL load failed while importing _specfun: An Application Control policy
    has blocked this file.

    DLL load failed while importing _solve_toeplitz: An Application Control
    policy has blocked this file.

Those modules are not used by the identification path. When Windows blocks an
extension, we register a stub, retry the parent import, and keep going. New
blocked SciPy ``.pyd`` files are handled the same way without a code change.
"""

from __future__ import annotations

import importlib.machinery
import os
import re
import sys
import types
from pathlib import Path

_PREPARED = False
_SPECFUN_STUBBED = False
_STUBBED: set[str] = set()

_SPECFUN_MODULE = "scipy.special._specfun"
_TOEPLITZ_MODULE = "scipy.linalg._solve_toeplitz"
_WARM_IMPORTS = ("scipy.special", "scipy.linalg", "scipy.spatial")
_MAX_STUB_RETRIES = 40
_BLOCKED_NAME_RE = re.compile(r"while importing ([A-Za-z_][A-Za-z0-9_]*)")

_WDAC_HINT = (
    "Windows Smart App Control blocked a Python extension (.pyd). "
    "abdXtract stubs optional SciPy modules that identification does not "
    "use (including _specfun and _solve_toeplitz) and retries the import. "
    "If jobs still fail, in Windows Security allow python.exe, or turn off "
    "Smart App Control for this device."
)


def specfun_was_stubbed() -> bool:
    return _SPECFUN_STUBBED


def explain_native_import_error(exc: BaseException) -> str:
    """User-facing text for DLL / Application Control import failures."""
    text = str(exc).strip() or type(exc).__name__
    lowered = text.lower()
    if "application control" in lowered or "dll load failed" in lowered:
        return f"{text}\n\n{_WDAC_HINT}"
    return text


def prepare_scipy_native_imports() -> None:
    """Make SciPy imports succeed under Windows Application Control.

    Safe to call more than once. No-op on non-Windows and when SciPy is
    missing. Called from app startup and again before loading the solver.
    """
    global _PREPARED
    if _PREPARED or sys.platform != "win32":
        return
    _add_package_dll_directories()
    _unblock_native_files()
    for name in _WARM_IMPORTS:
        _import_with_stubs(name)
    _PREPARED = True


def _scipy_package_dir() -> Path | None:
    try:
        import scipy
    except Exception:
        return None
    path = getattr(scipy, "__file__", None)
    if not path:
        return None
    return Path(path).resolve().parent


def _site_packages_root() -> Path | None:
    package = _scipy_package_dir()
    if package is None:
        return None
    return package.parent


def _add_package_dll_directories() -> None:
    add = getattr(os, "add_dll_directory", None)
    if add is None:
        return
    root = _site_packages_root()
    if root is None:
        return
    for name in ("scipy.libs", "numpy.libs"):
        folder = root / name
        if folder.is_dir():
            try:
                add(str(folder))
            except OSError:
                pass


def _unblock_native_files() -> None:
    """Clear Mark-of-the-Web on SciPy/NumPy extensions and bundled DLLs."""
    root = _site_packages_root()
    if root is None:
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.DeleteFileW.argtypes = [ctypes.c_wchar_p]
        kernel32.DeleteFileW.restype = ctypes.c_int
    except Exception:
        return
    folders = [root / "scipy", root / "numpy", root / "scipy.libs", root / "numpy.libs"]
    for folder in folders:
        if not folder.is_dir():
            continue
        for pattern in ("*.pyd", "*.dll"):
            for native in folder.rglob(pattern):
                try:
                    kernel32.DeleteFileW(f"{native}:Zone.Identifier")
                except Exception:
                    pass


def _is_wdac_block(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "application control" in text or "dll load failed" in text


def blocked_extension_name(exc: BaseException) -> str | None:
    """Return the short extension name Windows blocked, if the error says so."""
    if not _is_wdac_block(exc):
        return None
    match = _BLOCKED_NAME_RE.search(str(exc))
    return match.group(1) if match else None


def _install_flexible_stub(full_name: str) -> None:
    global _SPECFUN_STUBBED
    existing = sys.modules.get(full_name)
    if existing is not None and getattr(existing, "__file__", None):
        return
    stub = _WdacStub(full_name)
    sys.modules[full_name] = stub
    _STUBBED.add(full_name)
    if full_name == _SPECFUN_MODULE:
        _SPECFUN_STUBBED = True


def _drop_incomplete_package(package: str) -> None:
    loaded = sys.modules.get(package)
    if package == "scipy.special" and loaded is not None and hasattr(loaded, "rel_entr"):
        return
    if package == "scipy.linalg" and loaded is not None and hasattr(loaded, "cholesky"):
        return
    if package == "scipy.spatial" and loaded is not None and hasattr(loaded, "cKDTree"):
        return
    for key in list(sys.modules):
        if key == package or (key.startswith(package + ".") and key not in _STUBBED):
            sys.modules.pop(key, None)


def _import_with_stubs(name: str) -> None:
    """Import ``name``, stubbing any WDAC-blocked SciPy extension and retrying."""
    last_blocked = None
    for _ in range(_MAX_STUB_RETRIES):
        try:
            __import__(name)
            return
        except (ImportError, OSError) as exc:
            child = blocked_extension_name(exc)
            if child is None:
                raise
            full_name = child if child.startswith("scipy.") else f"{name}.{child}"
            if full_name == last_blocked:
                raise
            last_blocked = full_name
            _install_flexible_stub(full_name)
            _drop_incomplete_package(name)
    raise ImportError(f"Windows Application Control kept blocking {name}.")


class _WdacStub(types.ModuleType):
    """Stand-in for a blocked ``.pyd`` so ``from .mod import name`` can succeed."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__doc__ = (
            f"Placeholder used when Windows Application Control blocked {name}."
        )
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=False)
        spec.origin = None
        spec.has_location = False
        self.__spec__ = spec

    def __getattr__(self, item: str):
        def _blocked(*_args, **_kwargs):
            raise ImportError(
                f"{self.__name__}.{item} is unavailable because Windows "
                "Application Control blocked this SciPy extension."
            )

        _blocked.__name__ = item
        _blocked.__qualname__ = item
        setattr(self, item, _blocked)
        return _blocked

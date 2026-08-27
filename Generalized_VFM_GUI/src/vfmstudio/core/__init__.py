"""Qt-free core of VFM Studio."""

from vfmstudio.core.document import SessionDocument, empty_document
from vfmstudio.core.paths import ensure_core_on_path, resolve_core_path

__all__ = [
    "SessionDocument",
    "empty_document",
    "ensure_core_on_path",
    "resolve_core_path",
]

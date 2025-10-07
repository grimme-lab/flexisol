"""FlexiSol Evaluator package.

Holds the package version in one place. The version is sourced from the
installed distribution metadata (pyproject.toml [project].version).
"""

from typing import Final, List

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except Exception:  # pragma: no cover - fallback for very old Python
    from importlib_metadata import PackageNotFoundError, version as _pkg_version  # type: ignore

try:
    __version__: Final[str] = _pkg_version("flexisol-cli")
except PackageNotFoundError:  # when running from source without install - fallback
    __version__ = "0.1.0"

# Authors: resolved from installed metadata if available, else fallback
try:
    from importlib.metadata import metadata as _pkg_metadata
    _md = _pkg_metadata("flexisol-cli")
    _authors: List[str] = []
    if hasattr(_md, 'get_all'):
        vals = _md.get_all('Author') or []
        _authors.extend([v for v in vals if v])
    if not _authors:
        v = _md.get('Author')
        if v:
            _authors.append(v)
except Exception:
    _authors = []

if not _authors:
    _authors = ["L. Wittmann", "C. E. Selzer"]

__authors__: Final[List[str]] = _authors

__all__ = ["__version__", "__authors__"]

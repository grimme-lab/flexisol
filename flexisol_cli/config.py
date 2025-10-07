"""Configuration dataclass for FlexiSol runs (package version)."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import os
from pathlib import Path


@dataclass
class FlexisolConfig:
    root: str
    registry: str
    ref_gsolv: Optional[str]
    ref_pkab: Optional[str]
    output_dir: str
    weighting: str
    geometry: str
    sigma: Optional[float]
    abs_cutoff: Optional[float]
    registry_map: Optional[Dict[str, Dict[str, str]]] = None

    @classmethod
    def from_args(cls, args) -> "FlexisolConfig":
        """Build config from args with robust defaults.

        Avoids using the installed package path as a pseudo "repo root". Instead:
        - Root: prefer `--root`, then `FLEXISOL_ROOT`, else `<cwd>/flexisol` if present, else `<cwd>`.
        - References: prefer flags, then env vars (`FLEXISOL_REF_GSOLV`, `FLEXISOL_REF_PKAB`), then `<root>/../data/references/*` if present, else `<cwd>/data/references/*` if present, else None.
        - Output dir: prefer `--csv-dir`/`--csv`, else `<cwd>/output`.
        - Registry: packaged `registry.json` unless overridden by `--registry`.
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()

        # Root directory
        env_root = os.environ.get('FLEXISOL_ROOT')
        if getattr(args, 'root', None):
            default_root = os.path.abspath(getattr(args, 'root'))
        elif env_root:
            default_root = os.path.abspath(env_root)
        else:
            # Prefer <cwd>/flexisol if it exists; otherwise use cwd
            cand = Path(cwd) / 'flexisol'
            default_root = str(cand.resolve()) if cand.is_dir() else str(Path(cwd).resolve())

        # References
        env_ref_g = os.environ.get('FLEXISOL_REF_GSOLV')
        env_ref_p = os.environ.get('FLEXISOL_REF_PKAB')
        def _pick_ref(which: str) -> Optional[str]:
            # which in {"dgsolv-references.csv", "logkab-references.csv"}
            # try <root>/../data/references, then <cwd>/data/references
            rel = Path('data') / 'references' / which
            for base in [Path(default_root).parent, Path(cwd)]:
                candidate = (base / rel)
                if candidate.is_file():
                    return str(candidate.resolve())
            return None

        default_ref_gsolv = env_ref_g or _pick_ref('dgsolv-references.csv')
        default_ref_pkab = env_ref_p or _pick_ref('logkab-references.csv')

        # Output
        default_output_dir = getattr(args, 'csv_dir', None) or getattr(args, 'csv', None)
        if not default_output_dir:
            default_output_dir = str((Path(cwd) / 'output').resolve())

        # Registry path (packaged)
        default_registry = os.path.join(this_dir, 'registry.json')

        sigma = getattr(args, 'sigma', 3.0)
        if getattr(args, 'no_sigma', False):
            sigma = None
        abs_cutoff = getattr(args, 'abs_cutoff', 200.0)
        if getattr(args, 'no_abs_cutoff', False):
            abs_cutoff = None
        return cls(
            root=default_root,
            registry=getattr(args, 'registry', None) or default_registry,
            ref_gsolv=getattr(args, 'ref_gsolv', None) or default_ref_gsolv,
            ref_pkab=getattr(args, 'ref_pkab', None) or default_ref_pkab,
            output_dir=default_output_dir,
            weighting=getattr(args, 'weighting', 'boltzmann'),
            geometry=getattr(args, 'geometry', 'full'),
            sigma=sigma,
            abs_cutoff=abs_cutoff,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

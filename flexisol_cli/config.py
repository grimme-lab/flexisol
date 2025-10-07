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
    energies_csv: Optional[str] = None

    @classmethod
    def from_args(cls, args) -> "FlexisolConfig":
        """Build config from args with robust defaults.

        Root resolution avoids relying on the installed package location.
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()

        # Root directory (benchmark root)
        env_root = os.environ.get('FLEXISOL_ROOT')
        if getattr(args, 'root', None):
            default_root = os.path.abspath(getattr(args, 'root'))
        elif env_root:
            default_root = os.path.abspath(env_root)
        else:
            cand = Path(cwd) / 'flexisol'
            default_root = str(cand.resolve()) if cand.is_dir() else str(Path(cwd).resolve())

        # References
        env_ref_g = os.environ.get('FLEXISOL_REF_GSOLV')
        env_ref_p = os.environ.get('FLEXISOL_REF_PKAB')
        def _pick_ref(which: str) -> Optional[str]:
            rel = Path('data') / 'references' / which
            for base in [Path(default_root).parent, Path(cwd)]:
                candidate = (base / rel)
                if candidate.is_file():
                    return str(candidate.resolve())
            return None

        default_ref_gsolv = getattr(args, 'ref_gsolv', None) or env_ref_g or _pick_ref('dgsolv-references.csv')
        default_ref_pkab = getattr(args, 'ref_pkab', None) or env_ref_p or _pick_ref('logkab-references.csv')

        # Output
        default_output_dir = getattr(args, 'csv_dir', None) or getattr(args, 'csv', None)
        if not default_output_dir:
            default_output_dir = str((Path(cwd) / 'output').resolve())

        # Energies CSV (for populate)
        env_energies = os.environ.get('FLEXISOL_ENERGIES_CSV')
        if getattr(args, 'csv', None):
            default_energies_csv = os.path.abspath(getattr(args, 'csv'))
        elif env_energies:
            default_energies_csv = os.path.abspath(env_energies)
        else:
            rel = Path('data') / 'raw_energies' / 'energies.csv'
            cand = (Path(default_root).parent / rel)
            if cand.is_file():
                default_energies_csv = str(cand.resolve())
            else:
                cand2 = Path(cwd) / rel
                default_energies_csv = str(cand2.resolve()) if cand2.is_file() else None

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
            ref_gsolv=default_ref_gsolv,
            ref_pkab=default_ref_pkab,
            output_dir=default_output_dir,
            weighting=getattr(args, 'weighting', 'boltzmann'),
            geometry=getattr(args, 'geometry', 'full'),
            sigma=sigma,
            abs_cutoff=abs_cutoff,
            energies_csv=default_energies_csv,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

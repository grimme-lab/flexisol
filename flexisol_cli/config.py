"""Configuration dataclass for FlexiSol runs (package version)."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import os


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
        this_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(this_dir, os.pardir))
        default_root = os.path.join(repo_root, 'flexisol')
        default_ref_gsolv = os.path.join(repo_root, 'data', 'references', 'dgsolv-references.csv')
        default_ref_pkab = os.path.join(repo_root, 'data', 'references', 'logkab-references.csv')
        default_output_dir = os.path.join(repo_root, 'output')
        default_registry = os.path.join(this_dir, 'registry.json')

        sigma = getattr(args, 'sigma', 3.0)
        if getattr(args, 'no_sigma', False):
            sigma = None
        abs_cutoff = getattr(args, 'abs_cutoff', 200.0)
        if getattr(args, 'no_abs_cutoff', False):
            abs_cutoff = None
        return cls(
            root=getattr(args, 'root', None) or default_root,
            registry=getattr(args, 'registry', None) or default_registry,
            ref_gsolv=getattr(args, 'ref_gsolv', None) or default_ref_gsolv,
            ref_pkab=getattr(args, 'ref_pkab', None) or default_ref_pkab,
            output_dir=(getattr(args, 'csv_dir', None) or getattr(args, 'csv', None) or default_output_dir),
            weighting=getattr(args, 'weighting', 'boltzmann'),
            geometry=getattr(args, 'geometry', 'full'),
            sigma=sigma,
            abs_cutoff=abs_cutoff,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#!/usr/bin/env python3
import argparse
import os
import sys
import time
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd
import numpy as np

from .reader import get_structures, get_energies, get_boltzmann, get_minimum
from .evaluation import read_ref, eval_gsolv, eval_pkab
from .metrics import print_stats
from .config import FlexisolConfig
from . import __version__, __authors__


def _print_header():
    """Pretty-print a simple header with tool name and version."""
    title = 'FlexiSol Evaluator'
    subtitle = f'v {__version__}'
    authors_line = 'Authors: ' + ', '.join(__authors__)
    inner_w = max(len(title), len(subtitle))
    inner_w = max(inner_w, len(authors_line))
    top = '   +' + '-' * (inner_w + 2) + '+'
    mid1 = '   | ' + title.center(inner_w) + ' |'
    mid2 = '   | ' + subtitle.center(inner_w) + ' |'
    mid3 = '   | ' + authors_line.center(inner_w) + ' |'
    bot = '   +' + '-' * (inner_w + 2) + '+'
    print() ; print(top) ; print(mid1) ; print(mid2) ; print(mid3) ; print(bot) ; print()


def status_start(label: str, width: int = 25) -> float:
    """Start a timed status line with unified delimiter and return t0."""
    t0 = time.time()
    print(f"  {label.ljust(width)} ... ", end='', flush=True)
    return t0


def status_done(t0: float) -> None:
    """Finish a timed status line started with status_start."""
    print(f"done ({time.time()-t0:.2f} sec)")


def print_kv(key: str, value: object, indent: int = 2, key_width: int = 25) -> None:
    """Unified key/value printer with ' ... ' delimiter."""
    pad = ' ' * indent
    sval = '' if value is None else str(value)
    print(f"{pad}{key.ljust(key_width)} ... {sval}")


BUILTIN_METHOD_REGISTRY: Dict[str, Dict[str, str]] = {
    # Electronic methods
    "el_gfn2": {"method": "el_gfn2", "type": "el"},
    "el_gxtb": {"method": "el_gxtb", "type": "el"},
    "el_r2scan-3c": {"method": "el_r2scan-3c", "type": "el"},
    "el_wb97m-v_def2-tzvppd": {"method": "el_wb97m-v_def2-tzvppd", "type": "el"},
    # Solvation methods
    "alpb": {"method": "alpb", "type": "solv"},
    "cpcm-x": {"method": "cpcmx", "type": "solv"},
    "ddcosmo": {"method": "ddcosmo", "type": "solv"},
    "directml": {"method": "directml", "type": "solv"},
    "cigin": {"method": "cigin", "type": "solv"},
    "ese-ee-dnn": {"method": "ese-ee-dnn", "type": "solv"},
    "ese-gb-dnn": {"method": "ese-gb-dnn", "type": "solv"},
    "uese": {"method": "uese", "type": "solv"},
    "ese-pm7": {"method": "ese-pm7", "type": "solv"},
    "solv-pc": {"method": "solv-PC", "type": "solv"},
    "solv-spts": {"method": "solv-SPTS", "type": "solv"},
    "solv-sptv": {"method": "solv-SPTV", "type": "solv"},
    "cpcm": {"method": "cpcm", "type": "solv"},
    "smd": {"method": "smd", "type": "solv"},
    "dcosmors": {"method": "dcosmors", "type": "solv"},
    "opencosmors": {"method": "opencosmors", "type": "solv"},
    "cosmors": {"method": "cosmors-1601", "type": "solv"},
}

METHOD_REGISTRY: Dict[str, Dict[str, str]] = BUILTIN_METHOD_REGISTRY.copy()


def _load_registry(path: Optional[str]) -> None:
    """Load method registry JSON or fall back to built-in defaults."""
    global METHOD_REGISTRY
    if not path:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'registry.json')
    if not os.path.isfile(path):
        METHOD_REGISTRY = BUILTIN_METHOD_REGISTRY.copy()
        return
    try:
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        methods = data.get('methods', {})
        mr: Dict[str, Dict[str, str]] = {}
        for key, meta in methods.items():
            if not isinstance(meta, dict):
                continue
            mname = meta.get('method')
            mtype = meta.get('type')
            if mname and mtype in ('el', 'solv'):
                mr[key] = {'method': mname, 'type': mtype}
        METHOD_REGISTRY = mr or BUILTIN_METHOD_REGISTRY.copy()
    except Exception:
        METHOD_REGISTRY = BUILTIN_METHOD_REGISTRY.copy()


def _normalize_path(p: str, root: Optional[str]) -> str:
    """Normalize a structure path with respect to a benchmark root directory.

    Heuristics:
    - If `p` is relative: join to `root` if provided.
    - If `p` is absolute and contains "flexisol/": rebase the portion after that onto `root`.
    - If `p` is absolute, does not exist, and `root` is provided: try rebase by last three parts
      (structuremode/solvent/structure) under `root`.
    - Otherwise, return normalized `p`.
    """
    if not root:
        return os.path.normpath(p)
    if not os.path.isabs(p):
        if p.startswith("flexisol/"):
            return os.path.normpath(os.path.join(root, p[len("flexisol/"):]))
        return os.path.normpath(os.path.join(root, p))
    # Absolute path
    idx = p.find("flexisol/")
    if idx != -1:
        return os.path.normpath(os.path.join(root, p[idx + len("flexisol/") :]))
    if not os.path.isdir(p):
        # Try last 3 parts under root
        parts = p.strip(os.sep).split(os.sep)
        if len(parts) >= 3:
            candidate = os.path.join(root, *parts[-3:])
            return os.path.normpath(candidate)
    return os.path.normpath(p)


def _write_energy(path: str, energy: str, etype: Literal["el", "solv"], dry_run: bool = False) -> None:
    """Write a single energy value to `el_energy` or `solv_energy` file."""
    filename = "el_energy" if etype == "el" else "solv_energy"
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, filename)
    if dry_run:
        return
    with open(fpath, "w") as f:
        f.write(str(energy).strip() + "\n")


def cmd_populate(args: argparse.Namespace) -> int:
    """Create per-method energy files under each structure directory."""
    cfg = getattr(args, 'cfg', None) or FlexisolConfig.from_args(args)
    csv_path = getattr(args, 'csv', None) or cfg.energies_csv
    if not csv_path or not os.path.isfile(csv_path):
        print(f"Error ... energies CSV not found (got: {csv_path}). Pass --csv or set FLEXISOL_ENERGIES_CSV")
        return 2
    df = pd.read_csv(csv_path)

    csv_cols: Dict[str, Optional[str]] = {}
    missing_cols: List[str] = []
    registry = cfg.registry_map or METHOD_REGISTRY
    for key in registry.keys():
        if key in df.columns:
            csv_cols[key] = key
        else:
            csv_cols[key] = None
            missing_cols.append(key)
    if missing_cols and not args.allow_missing:
        print(f"Error: CSV missing expected columns (after aliasing): {missing_cols}")
        return 2

    n_written = 0
    n_skipped = 0
    per_method_written: Dict[str, int] = {meta["method"]: 0 for meta in registry.values()}

    print(f"Populating FlexiSol")
    print_kv("directory", cfg.root)
    print_kv("energies", csv_path)
    t0 = status_start("writing files")

    for _, row in df.iterrows():
        base_path = _normalize_path(str(row["path"]), cfg.root)
        if not os.path.isdir(base_path):
            if args.strict:
                print(f"Missing structure dir (strict) ... {base_path}")
                return 3
            else:
                if args.verbose:
                    print(f"Skip ... no dir {base_path}")
                n_skipped += 1
                continue

        for col, meta in registry.items():
            use_col = csv_cols.get(col)
            if not use_col:
                continue
            val = row[use_col]
            if pd.isna(val):
                continue
            method_dir = os.path.join(base_path, meta["method"])
            _write_energy(method_dir, str(val), meta["type"], dry_run=args.dry_run)
            n_written += 1
            per_method_written[meta["method"]] = per_method_written.get(meta["method"], 0) + 1

    status_done(t0)
    # Always print a short summary
    print_kv("summary", f"wrote {n_written} energies, skipped {n_skipped} structures")
    if args.verbose:
        print("Method coverage ... (files written)")
        for m, cnt in sorted(per_method_written.items()):
            print(f"  {m.ljust(28)} {cnt}")
    return 0


def _resolve_method(name: str, kind: Literal["el", "solv"], registry: Optional[Dict[str, Dict[str, str]]] = None) -> Optional[str]:
    """Resolve a method from user input to its folder name."""
    reg = registry or METHOD_REGISTRY
    if name in reg and reg[name]["type"] == kind:
        return reg[name]["method"]
    for _, meta in reg.items():
        if meta["type"] == kind and meta["method"] == name:
            return name
    return None


def _pick_weight(df: pd.DataFrame, methods: List[str], scheme: str) -> pd.DataFrame:
    if scheme == "boltzmann":
        return get_boltzmann(df, methods)
    if scheme == "minimum":
        return get_minimum(df, methods)
    raise ValueError(f"Unknown weighting: {scheme}")


def _apply_geometry(weighted_E: pd.DataFrame, structure_mode: str) -> pd.DataFrame:
    if structure_mode == 'full':
        pass
    elif structure_mode == 'gas':
        weighted_E = weighted_E[weighted_E['structuremode'] == 'gas']
        tmp = weighted_E.copy()
        tmp['structuremode'] = 'solv'
        weighted_E = pd.concat([weighted_E, tmp], ignore_index=True)
    elif structure_mode == 'solv':
        weighted_E = weighted_E[weighted_E['structuremode'] == 'solv']
        tmp = weighted_E.copy()
        tmp['structuremode'] = 'gas'
        weighted_E = pd.concat([weighted_E, tmp], ignore_index=True)
    else:
        raise ValueError(f'Unknown structure mode: {structure_mode}')
    return weighted_E


def _attach_solventmode(weighted_E: pd.DataFrame, baseline_method: str) -> pd.DataFrame:
    base_cols = ['name', 'benchmark', 'structuremode', 'solvent', 'charge', baseline_method]
    base_cols = [c for c in base_cols if c in weighted_E.columns]
    tmp = weighted_E[base_cols].copy()
    tmp['solventmode'] = 'gas'
    tmp = (
        tmp.groupby(['name', 'benchmark', 'structuremode', 'solventmode', 'solvent', 'charge'], as_index=False)
           .mean()
    )
    solv_part = weighted_E.copy()
    if baseline_method in solv_part.columns:
        solv_part = solv_part.drop(columns=[baseline_method])
    solv_part['solventmode'] = 'solv'
    return pd.concat([solv_part, tmp], ignore_index=True)


def _evaluate(root: str,
              ee_method: str,
              solv_methods: List[str],
              weighting: str,
              geometry: str,
              ref_gsolv: Optional[str],
              ref_pkab: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def _status_start(label: str, width: int = 25):
        print(f"  {label.ljust(width)}... ", end='', flush=True)

    def _status_done(t0: float):
        print(f"done ({time.time()-t0:.2f} sec)")
    t0 = status_start("reading structures")
    structures = get_structures(root, molbar=False)
    status_done(t0)

    t1 = status_start("reading energies")
    methods_all = [ee_method] + solv_methods
    energies = get_energies(structures, methods_all)
    status_done(t1)

    workenergies = energies.copy()
    grp_keys = ['name', 'benchmark', 'solvent', 'charge']
    if ee_method in workenergies.columns:
        mask_gas = workenergies['structuremode'] == 'gas'
        base_gas = workenergies.loc[mask_gas, [ee_method] + grp_keys].copy()
        base_gas['filled'] = base_gas.groupby(grp_keys)[ee_method].transform(
            lambda s: s.fillna(s.dropna().iloc[0]) if s.notna().any() else s
        )
        mask_solv = workenergies['structuremode'] == 'solv'
        base_solv = workenergies.loc[mask_solv, [ee_method] + grp_keys].copy()
        base_solv['filled'] = base_solv.groupby(grp_keys)[ee_method].transform(
            lambda s: s.fillna(s.dropna().iloc[0]) if s.notna().any() else s
        )
        base_filled = pd.Series(np.nan, index=workenergies.index)
        base_filled.loc[mask_gas] = base_gas['filled'].values
        base_filled.loc[mask_solv] = base_solv['filled'].values
    else:
        base_filled = pd.Series([np.nan] * len(workenergies), index=workenergies.index)
    for m in solv_methods:
        if m in workenergies.columns:
            workenergies[m] = workenergies[m] + base_filled

    t2 = status_start(f"weighting ({weighting})")
    weighted_E = _pick_weight(workenergies, methods_all, weighting)
    weighted_E['name'] = (
        weighted_E['name']
            .str.replace(r"[()']", "", regex=True)
            .str.replace(r"[ _]", "-", regex=True)
            .str.lower()
    )
    weighted_E['solvent'] = weighted_E['solvent'].str.lower()
    weighted_E = _apply_geometry(weighted_E, geometry)
    weighted_E = _attach_solventmode(weighted_E, ee_method)
    status_done(t2)

    results = pd.DataFrame()
    if ref_gsolv and os.path.exists(ref_gsolv):
        t3 = status_start("evaluating gsolv")
        gsolv_ref = read_ref(ref_gsolv)
        res = eval_gsolv(weighted_E, gsolv_ref, solv_methods, gas_method=ee_method)
        results = pd.concat([results, res]) if not res.empty else results
        status_done(t3)
    if ref_pkab and os.path.exists(ref_pkab):
        t4 = status_start("evaluating pkab")
        pkab_ref = read_ref(ref_pkab)
        res = eval_pkab(weighted_E, pkab_ref, solv_methods)
        results = pd.concat([results, res]) if not res.empty else results
        status_done(t4)

    return weighted_E, results


def cmd_evaluate_one(args: argparse.Namespace) -> int:
    cfg = getattr(args, 'cfg', None) or FlexisolConfig.from_args(args)
    ee = _resolve_method(args.electronic_energy, "el", registry=cfg.registry_map)
    se = _resolve_method(args.solvation_energy, "solv", registry=cfg.registry_map)
    if not ee:
        print(f"Unknown electronic method: {args.electronic_energy}")
        return 2
    if not se:
        print(f"Unknown solvation method: {args.solvation_energy}")
        return 2
    print(f"Working on {ee} + {se}  [weighting={cfg.weighting}, geometry={cfg.geometry}]")
    _, results = _evaluate(cfg.root, ee, [se], cfg.weighting, cfg.geometry, cfg.ref_gsolv, cfg.ref_pkab)
    n = len(results)
    out_path = args.csv
    if not out_path:
        out_dir = cfg.output_dir
        os.makedirs(out_dir, exist_ok=True)
        out_fname = f"{ee}-{cfg.geometry}-{cfg.weighting}-{se}-results.csv"
        out_path = os.path.join(out_dir, out_fname)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    results.to_csv(out_path, index=False, float_format='%.1f')
    by_dp = results['datapoint'].value_counts(dropna=False).to_dict() if 'datapoint' in results.columns else {}
    parts = [f"{k}={v}" for k, v in sorted(by_dp.items())]
    parts_str = ", ".join(parts) if parts else str(n)
    print(f"\n  results contain {n} rows ({parts_str})")
    print(f"  results written to {out_path}")
    print_stats(results, [se], sigma=cfg.sigma, abs_cutoff=cfg.abs_cutoff)
    return 0


def cmd_evaluate_all(args: argparse.Namespace) -> int:
    cfg = getattr(args, 'cfg', None) or FlexisolConfig.from_args(args)
    registry = cfg.registry_map or METHOD_REGISTRY
    ee_methods = [meta["method"] for meta in registry.values() if meta["type"] == "el"]
    solv_methods = [meta["method"] for meta in registry.values() if meta["type"] == "solv"]

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    for ee in ee_methods:
        t0 = time.time()
        print(f"Working on {ee} [weighting={cfg.weighting}, geometry={cfg.geometry}]")
        _, results = _evaluate(cfg.root, ee, solv_methods, cfg.weighting, cfg.geometry, cfg.ref_gsolv, cfg.ref_pkab)
        out = os.path.join(out_dir, f"{ee}-{cfg.geometry}-{cfg.weighting}-results.csv")
        results.to_csv(out, index=False, float_format='%.1f')
        print(f"  results written to {out}")
        by_dp = results['datapoint'].value_counts(dropna=False).to_dict() if 'datapoint' in results.columns else {}
        parts = [f"{k}={v}" for k, v in sorted(by_dp.items())]
        parts_str = ", ".join(parts) if parts else str(len(results))
        print(f"  results: {len(results)} rows ({parts_str})  (total {time.time()-t0:.2f} sec)")
        print_stats(results, solv_methods, sigma=cfg.sigma, abs_cutoff=cfg.abs_cutoff)
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FlexiSol evaluator CLI")
    sub = p.add_subparsers(dest="cmd")

    pp = sub.add_parser("populate", help="Populate method folders with energy files from CSV")
    pp.add_argument("--csv", default="data/raw_energies/energies.csv", help="Input CSV with energies")
    pp.add_argument("--root", "--benchmark-root", "--dataset-root", dest="root", default=None,
                    help="Benchmark root directory (contains structure folders)")
    pp.add_argument("--registry", default=None, help="Path to registry.json (methods + aliases)")
    pp.add_argument("--dry-run", action="store_true", help="Do not write files")
    pp.add_argument("--verbose", action="store_true")
    pp.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    pp.add_argument("--strict", action="store_true", help="Fail on missing structure directories")
    pp.add_argument("--allow-missing", action="store_true", help="Allow missing CSV columns")
    pp.set_defaults(func=cmd_populate)

    ea = sub.add_parser("evaluate-all", aliases=["all"], help="Evaluate all registry methods for a given weighting/geometry")
    ea.add_argument("--root", "--benchmark-root", "--dataset-root", dest="root", default=None,
                    help="Benchmark root directory (contains structure folders)")
    ea.add_argument("--registry", default=None, help="Path to registry.json (methods + aliases)")
    ea.add_argument("-w", "--weighting", default="boltzmann", choices=["boltzmann", "minimum"], help="Weighting scheme")
    ea.add_argument("-g", "--geometry", default="full", choices=["full", "gas", "solv"], help="Geometry selection")
    ea.add_argument("--ref-gsolv", default=None, help="Path to dgsolv reference CSV")
    ea.add_argument("--ref-pkab", default=None, help="Path to logK reference CSV")
    ea.add_argument("--csv-dir", default=None, help="Directory to write per-combination raw results CSVs (defaults to output/)")
    ea.add_argument("--sigma", type=float, default=3.0, help="Sigma threshold for outlier filtering in stats")
    ea.add_argument("--abs-cutoff", type=float, default=200.0, help="Absolute error cutoff before sigma (legacy=200)")
    ea.add_argument("--no-abs-cutoff", action="store_true", help="Disable absolute error cutoff")
    ea.add_argument("--verbose", action="store_true")
    ea.set_defaults(func=cmd_evaluate_all)

    eo = sub.add_parser("evaluate-one", aliases=["one"], help="Evaluate a specific (electronic,solvation) pair")
    eo.add_argument("--root", "--benchmark-root", "--dataset-root", dest="root", default=None,
                    help="Benchmark root directory (contains structure folders)")
    eo.add_argument("--registry", default=None, help="Path to registry.json (methods + aliases)")
    eo.add_argument("-ee", "--electronic-energy", required=True, help="Electronic method (registry key or folder name)")
    eo.add_argument("-se", "--solvation-energy", required=True, help="Solvation method (registry key or folder name)")
    eo.add_argument("-w", "--weighting", default="boltzmann", choices=["boltzmann", "minimum"], help="Weighting scheme")
    eo.add_argument("-g", "--geometry", default="full", choices=["full", "gas", "solv"], help="Geometry selection")
    eo.add_argument("--ref-gsolv", default=None, help="Path to dgsolv reference CSV")
    eo.add_argument("--ref-pkab", default=None, help="Path to logK reference CSV")
    eo.add_argument("-csv", "--csv", default=None, help="Write raw results to this CSV path (defaults to output/)")
    eo.add_argument("--limit", type=int, default=20, help="Rows to print if not saving to CSV")
    eo.add_argument("--sigma", type=float, default=3.0, help="Sigma threshold for outlier filtering in stats")
    eo.add_argument("--no-sigma", action="store_true", help="Disable sigma outlier filtering in stats")
    eo.add_argument("--abs-cutoff", type=float, default=200.0, help="Absolute error cutoff before sigma (legacy=200)")
    eo.add_argument("--no-abs-cutoff", action="store_true", help="Disable absolute error cutoff")
    eo.add_argument("--verbose", action="store_true")
    eo.set_defaults(func=cmd_evaluate_one)

    pc = sub.add_parser("config", help="Show resolved configuration")
    pc.add_argument("--show", action="store_true", help="Print resolved configuration", default=True)
    pc.set_defaults(func=cmd_config)

    return p


def _self_check(root: str, ref_gsolv: Optional[str], ref_pkab: Optional[str]) -> None:
    problems = []
    if not os.path.isdir(root):
        problems.append(f"Missing root directory: {root}")
    if ref_gsolv and not os.path.isfile(ref_gsolv):
        problems.append(f"Missing dgsolv reference CSV: {ref_gsolv}")
    if ref_pkab and not os.path.isfile(ref_pkab):
        problems.append(f"Missing logK reference CSV: {ref_pkab}")
    if problems:
        print("Self-check warnings:")
        for pmsg in problems:
            print(f" - {pmsg}")


def cmd_config(args: argparse.Namespace) -> int:
    """Print the resolved configuration (paths and options)."""
    cfg = getattr(args, 'cfg', None) or FlexisolConfig.from_args(args)
    data = cfg.to_dict()
    print("Resolved configuration ...")
    # Pretty print key: value lines in a stable order
    order = [
        'root', 'registry', 'ref_gsolv', 'ref_pkab', 'energies_csv', 'output_dir',
        'weighting', 'geometry', 'sigma', 'abs_cutoff'
    ]
    file_keys = {'registry', 'ref_gsolv', 'ref_pkab', 'energies_csv'}
    dir_keys = {'root', 'output_dir'}
    for key in order:
        val = data.get(key)
        status = ''
        if isinstance(val, str):
            try:
                if key in file_keys:
                    status = ' (ok)' if os.path.isfile(val) else ' (missing)'
                elif key in dir_keys:
                    status = ' (ok)' if os.path.isdir(val) else ' (missing)'
            except Exception:
                status = ''
        print(f"  {key.ljust(14)} ... {val}{status}")
    # Registry size hint
    reg = data.get('registry_map') or {}
    if isinstance(reg, dict) and reg:
        n_el = sum(1 for v in reg.values() if isinstance(v, dict) and v.get('type') == 'el')
        n_solv = sum(1 for v in reg.values() if isinstance(v, dict) and v.get('type') == 'solv')
        print(f"  registry_map  ... {len(reg)} entries ({n_el} el, {n_solv} solv)")
    else:
        print("  registry_map  ... (not loaded)")
    print()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    _print_header()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'cmd', None):
        parser.print_help()
        return 0
    cfg = FlexisolConfig.from_args(args)
    setattr(args, 'cfg', cfg)
    _load_registry(cfg.registry)
    cfg.registry_map = METHOD_REGISTRY.copy()
    _self_check(cfg.root, cfg.ref_gsolv, cfg.ref_pkab)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

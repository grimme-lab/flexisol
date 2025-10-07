"""Lightweight I/O and weighting utilities for FlexiSol.

This module discovers structure folders, reads per-method energies written by
the CLI or external tools, and provides simple weighting aggregations
(Boltzmann, minimum).
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

autocal = 627.5096080306  # (kcal/mol)/Eh
autokj = 2625.499639479   # (kJ/mol)/Eh
kB = 0.0083144621         # kJ/(mol*K)


def boltzmann_weight(energies, temperature=298.15):
    """Return Boltzmann-weighted energy from a 1D list/array.

    Args:
        energies (array-like): Energies (Hartree). NaNs are ignored.
        temperature (float): Temperature in Kelvin. Default 298.15 K.

    Returns:
        float: Weighted average energy (Hartree). NaN if no valid values.
    """
    energies = np.array(energies, dtype=float)
    energies = energies[~np.isnan(energies)]
    if energies.size == 0:
        return np.nan
    energies *= autokj
    beta = 1.0 / (kB * temperature)
    min_energy = energies.min()
    shifted = energies - min_energy
    weights = np.exp(-beta * shifted)
    weighted_avg = min_energy + np.sum(shifted * weights) / np.sum(weights)
    weighted_avg /= autokj
    return weighted_avg


def min_weight(energies):
    """Return the minimum energy (Hartree) ignoring NaNs."""
    energies = np.array(energies, dtype=float)
    energies = energies[~np.isnan(energies)]
    if energies.size == 0:
        return np.nan
    return float(np.min(energies))


def get_structures(benchmarkdir, molbar=False):
    """Discover structure directories under a benchmark root.

    Traverses `<benchmarkdir>/<structuremode>/<solvent>/<structure>` and builds a
    DataFrame with parsed metadata from directory names (name, charge, tautomer,
    conformer).
    """
    structures = []
    for structuremode in os.listdir(benchmarkdir):
        sm_path = os.path.join(benchmarkdir, structuremode)
        if not os.path.isdir(sm_path):
            continue
        for solvent in os.listdir(sm_path):
            so_path = os.path.join(sm_path, solvent)
            if not os.path.isdir(so_path):
                continue
            for structure in os.listdir(so_path):
                st_path = os.path.join(so_path, structure)
                if os.path.isdir(st_path):
                    structures.append(st_path)

    df = pd.DataFrame(structures, columns=['path'])
    # Portable split of last 3 path components
    def _last_parts(p: str, n: int = 3):
        parts = list(Path(p).parts)
        if len(parts) < n:
            return [None] * (n - len(parts)) + parts
        return parts[-n:]
    last3 = df['path'].map(lambda p: _last_parts(p, 3))
    tmp = pd.DataFrame(last3.tolist(), columns=['structuremode', 'solvent', 'structure'])
    df = pd.concat([df, tmp], axis=1)

    structure_parts = df['structure'].astype(str).str.extract(
        r'^(?P<name>.+?)_chrg(?P<charge>\d+)(?:_t(?P<tautomer>\d+))?(?:_c(?P<conformer>\d+))?$'
    )

    df = pd.concat([
        df[['path', 'structuremode', 'solvent']],
        structure_parts
    ], axis=1)

    df['solventmode'] = np.nan
    df['benchmark'] = 'solvl'
    return df


def get_energy(args):
    """Read a single structure's energies for a set of methods."""
    index, row, methods = args
    workpath = row['path']
    energy_dict = {}
    for m in methods:
        en = np.nan
        try:
            simple_el = os.path.join(workpath, m, 'el_energy')
            simple_solv = os.path.join(workpath, m, 'solv_energy')
            if os.path.exists(simple_el):
                with open(simple_el, 'r') as f:
                    try:
                        en = float(f.read().strip())
                    except ValueError:
                        en = np.nan
                energy_dict[m] = en
                continue
            if os.path.exists(simple_solv):
                with open(simple_solv, 'r') as f:
                    try:
                        en = float(f.read().strip())
                    except ValueError:
                        en = np.nan
                energy_dict[m] = en
                continue
        except Exception:
            en = np.nan
        energy_dict[m] = en

    return (index, energy_dict)


def get_energies(df, methods, processes=None):
    """Read energies for all structures and methods using a thread pool.

    Args:
        df: Structure DataFrame.
        methods: Method folder names to read.
        processes: Interpreted as max_workers for threads (I/O-bound).
    """
    tasks = [(index, row, methods) for index, row in df.iterrows()]
    with ThreadPoolExecutor(max_workers=processes) as ex:
        results = list(ex.map(get_energy, tasks))

    indices, energy_dicts = zip(*results)
    energy_df = pd.DataFrame(energy_dicts, index=indices)
    df = pd.concat([df, energy_df], axis=1)

    return df


def get_boltzmann(df, methods):
    """Aggregate conformer energies by Boltzmann weighting per group."""
    group_keys = ['name', 'benchmark', 'structuremode', 'solvent', 'charge']
    weighted_energies = df.groupby(group_keys)[methods].agg(lambda x: boltzmann_weight(x))
    return weighted_energies.reset_index()


def get_minimum(df, methods):
    """Aggregate conformer energies by minimum per group."""
    group_keys = ['name', 'benchmark', 'structuremode', 'solvent', 'charge']
    weighted_energies = df.groupby(group_keys)[methods].agg(lambda x: min_weight(x))
    return weighted_energies.reset_index()

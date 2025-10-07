"""Evaluation helpers for gsolv and pkab metrics."""

import numpy as np
import pandas as pd

autocal = 627.5096080306


def read_ref(filepath):
    """Read and normalize a reference CSV."""
    df = pd.read_csv(filepath)
    rename_map = {}
    lower_map = {c.lower(): c for c in df.columns}
    if 'flexisol name' in lower_map:
        rename_map[lower_map['flexisol name']] = 'name'
    if 'solvent' in lower_map:
        rename_map[lower_map['solvent']] = 'solvent'
    if 'solvents' in lower_map:
        rename_map[lower_map['solvents']] = 'solvents'
    for key in ['dG_solv / kcal/mol', 'value', 'value (log units)', r'value (\kcalpmole)', 'value (kcal/mol)']:
        if key.lower() in lower_map:
            rename_map[lower_map[key.lower()]] = 'value'
            break
    if rename_map:
        df = df.rename(columns=rename_map)

    if 'name' in df.columns:
        df['name'] = (
            df['name'].astype(str)
                      .str.replace(r"[()']", "", regex=True)
                      .str.replace(r"[ _]", "-", regex=True)
                      .str.lower()
        )
    if 'solvent' in df.columns and df['solvent'].dtype == object:
        df['solvent'] = df['solvent'].str.lower()
    if 'solvents' in df.columns and df['solvents'].dtype == object:
        df['solvents'] = df['solvents'].str.lower()
    return df


def eval_gsolv(energies, reference, methods, gas_method='el_r2scan-3c'):
    """Compute gsolv errors per method against a reference table."""
    columns = (['name', 'charge1', 'benchmark', 'solvent1', 'ref'] +
               [x for x in methods] +
               ['err_' + x for x in methods])

    results = []
    for _, ref in reference.iterrows():
        name = ref['name']
        solvent = ref['solvent']
        charges = energies[(energies['name'] == name) & (energies['solvent'] == solvent)]['charge'].unique()
        for charge in charges:
            df_solvent = energies[(energies['name'] == name) & (energies['charge'] == charge) &
                                  (energies['solvent'] == solvent) &
                                  (energies['structuremode'] == 'solv') &
                                  (energies['solventmode'] == 'solv')]
            df_gas = energies[(energies['name'] == name) & (energies['charge'] == charge) &
                              (energies['solvent'] == solvent) &
                              (energies['structuremode'] == 'gas') &
                              (energies['solventmode'] == 'gas')]
            if df_solvent.empty or df_gas.empty:
                continue

            en_a = df_solvent.iloc[0]
            en_b = df_gas.iloc[0]
            en = (en_a[methods] - en_b[gas_method]) * autocal
            if en.isnull().values.all():
                continue

            value = ref.get('value', ref.get('dG_solv / kcal/mol', np.nan))
            try:
                value = float(value)
            except Exception:
                continue
            en_arr = np.array(en, dtype=float)
            err = en_arr - value
            row = [name, charge, ref.get('benchmark', 'solvl'), solvent, value]
            row.extend(en_arr.tolist())
            row.extend(err.tolist())
            results.append(row)

    dfi = pd.DataFrame(results, columns=columns)
    dfi['datapoint'] = 'gsolv'
    return dfi


def eval_pkab(energies, reference, methods):
    """Compute pkab (log units) differences between two solvents."""
    columns = (['name', 'charge1', 'benchmark', 'solvent1', 'solvent2', 'ref'] +
               [x for x in methods] +
               ['err_' + x for x in methods])

    results = []
    for _, ref in reference.iterrows():
        name = ref['name']
        solvents = ref['solvents'].split(',')
        servers = [s.strip().lower() for s in solvents]
        if len(servers) < 2:
            continue
        solvent_a, solvent_b = servers[0], servers[1]

        charges_to_try = []
        if 'charge' in ref and pd.notna(ref['charge']):
            charges_to_try = [ref['charge']]
        else:
            available = energies[energies['name'] == name]['charge'].dropna().unique().tolist()
            if 0 in available:
                charges_to_try = [0]
            else:
                charges_to_try = available

        for charge in charges_to_try:
            en_a = energies[(energies['name'] == name) & (energies['solvent'] == solvent_a) &
                            (energies['charge'] == charge) & (energies['structuremode'] == 'solv') &
                            (energies['solventmode'] == 'solv')]
            en_b = energies[(energies['name'] == name) & (energies['solvent'] == solvent_b) &
                            (energies['charge'] == charge) & (energies['structuremode'] == 'solv') &
                            (energies['solventmode'] == 'solv')]
            if en_a.empty or en_b.empty:
                continue

            en_a = en_a[methods].iloc[0].astype(float)
            en_b = en_b[methods].iloc[0].astype(float)
            en = en_b - en_a
            if en.isnull().values.all():
                continue

            try:
                value = float(ref['value'])
            except Exception:
                continue

            en = np.array(en, dtype=float)
            factor = 2625.5 * 1000.0 / (8.314 * 298)
            with np.errstate(over='ignore', invalid='ignore'):
                logp = (en * factor) / np.log(10.0)
            err = logp - value

            row = [name, charge, ref.get('benchmark', 'solvl'), solvent_a, solvent_b, value]
            row.extend(logp.tolist())
            row.extend(err.tolist())
            results.append(row)

    dfi = pd.DataFrame(results, columns=columns)
    dfi['datapoint'] = 'pkab'
    return dfi


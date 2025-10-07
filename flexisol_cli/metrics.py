"""Error metrics and outlier handling utilities."""

from typing import List, Optional, Dict

import numpy as np
import pandas as pd


def get_average_errors(err_series: pd.Series) -> Dict[str, float]:
    arr = pd.to_numeric(err_series, errors='coerce').dropna().values
    if arr.size == 0:
        return dict(n=0, me=np.nan, mae=np.nan, rmse=np.nan, sd=np.nan, amax=np.nan)
    me = float(np.mean(arr))
    mae = float(np.mean(np.abs(arr)))
    rmse = float(np.sqrt(np.mean(arr ** 2)))
    sd = float(np.std(arr))
    amax = float(np.max(np.abs(arr)))
    return dict(n=int(arr.size), me=me, mae=mae, rmse=rmse, sd=sd, amax=amax)


def apply_abs_cutoff(results: pd.DataFrame, methods: List[str], cutoff: float) -> pd.DataFrame:
    if results.empty:
        return results
    frames = []
    for _, sub in results.groupby('datapoint', sort=False):
        sub = sub.copy()
        for m in methods:
            col = 'err_' + m
            if col not in sub.columns:
                continue
            s = pd.to_numeric(sub[col], errors='coerce')
            mask = s.abs() > cutoff
            if mask.any():
                sub.loc[mask, col] = np.nan
        frames.append(sub)
    return pd.concat(frames, axis=0, ignore_index=True)


def apply_3sigma(results: pd.DataFrame, methods: List[str], threshold: float = 3.0) -> pd.DataFrame:
    if results.empty:
        return results
    frames = []
    for _, sub in results.groupby('datapoint', sort=False):
        sub = sub.copy()
        for m in methods:
            col = 'err_' + m
            if col not in sub.columns:
                continue
            s = pd.to_numeric(sub[col], errors='coerce')
            mu = s.mean(skipna=True)
            sd = s.std(skipna=True)
            if sd == 0 or np.isnan(sd):
                continue
            z = (s - mu).abs() / sd
            mask = z > threshold
            if mask.any():
                sub.loc[mask, col] = np.nan
        frames.append(sub)
    return pd.concat(frames, axis=0, ignore_index=True)


def print_stats(results: pd.DataFrame,
                methods: List[str],
                sigma: Optional[float] = 3.0,
                abs_cutoff: Optional[float] = 200.0) -> None:
    if results.empty:
        print("  statistics: no rows")
        return
    base = results.reset_index(drop=True)
    filt = base
    if abs_cutoff is not None:
        filt = apply_abs_cutoff(filt, methods, cutoff=float(abs_cutoff))
    if sigma is not None:
        filt = apply_3sigma(filt, methods, threshold=float(sigma))

    units = {
        'gsolv': 'kcal/mol',
        'pkab': 'log units',
    }
    for dtype, sub in filt.groupby('datapoint'):
        unit = units.get(dtype, '')
        unit_str = f" ({unit})" if unit else ''
        parts = []
        if abs_cutoff is not None:
            parts.append(f"abs>{abs_cutoff:g}")
        parts.append("no-sigma" if sigma is None else f"{sigma:g}-sigma")
        note = ", ".join(parts)
        print(f"\n  statistics for {dtype}{unit_str} ({note}):")
        header = f"    {'method'.ljust(24)} {'ME':>8} {'MAE':>8} {'RMSE':>8} {'SD':>8} {'AMAX':>8} {'N':>6}"
        print(header)
        print("    " + ("-" * 24) + (" " + "-" * 8) * 5 + (" " + "-" * 6))
        for m in methods:
            col = 'err_' + m
            if col not in sub.columns:
                continue
            stats = get_average_errors(sub[col])
            def r(x):
                return 'nan' if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.2f}"
            label = m[:24].ljust(24)
            print(f"    {label} {r(stats['me']):>8} {r(stats['mae']):>8} {r(stats['rmse']):>8} {r(stats['sd']):>8} {r(stats['amax']):>8} {stats['n']:>6}")


"""Statistical process control for reject rate: p-chart with 3-sigma limits and run rules.

Two flavours are offered:

* classic p-chart: limits from the binomial standard error p*(1-p)/n_i.
* Laney p' chart: multiplies the limits by an overdispersion factor estimated from the
  moving range of the standardised points. With ~50-100k units per shift even tiny, normal
  shift-to-shift differences look 'out of control' on a classic p-chart, so p' is the
  appropriate chart for this kind of high-volume data.

Rules applied: Rule 1 = point beyond 3 sigma; Rule 2 = 8 consecutive points on the same side
of the centre line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def laney_sigma(z: pd.Series) -> float:
    """Overdispersion factor sigma_z = mean(|z_i - z_{i-1}|) / 1.128 (floored at 1)."""
    mr = z.diff().abs().dropna()
    if len(mr) == 0:
        return 1.0
    return float(max(mr.mean() / 1.128, 1.0))


def run_rule_flags(values: np.ndarray, center: float, k: int = 8) -> np.ndarray:
    """Flag points that complete / extend a run of >= k consecutive points on one side of centre."""
    side = np.sign(values - center)
    flags = np.zeros(len(values), dtype=bool)
    run = 0
    prev = 0
    for i, s in enumerate(side):
        if s != 0 and s == prev:
            run += 1
        elif s != 0:
            run = 1
        else:
            run = 0
        prev = s
        if run >= k:
            flags[i] = True
    return flags


def p_chart(df: pd.DataFrame, laney: bool = True, k_run: int = 8) -> pd.DataFrame:
    """Compute a p-chart for ONE line (df = shifts of that line). Returns the shifts with limits and flags."""
    d = df.sort_values(["date", "shift"]).reset_index(drop=True).copy()
    d = d[d["total_units"] > 0].reset_index(drop=True)
    if len(d) == 0:
        return d
    d["p"] = d["reject_units"] / d["total_units"]
    pbar = d["reject_units"].sum() / d["total_units"].sum()
    se = np.sqrt(pbar * (1 - pbar) / d["total_units"])
    z = (d["p"] - pbar) / se
    factor = laney_sigma(z) if laney else 1.0
    d["center"] = pbar
    d["ucl"] = pbar + 3 * se * factor
    d["lcl"] = (pbar - 3 * se * factor).clip(lower=0)
    d["sigma_factor"] = factor
    d["rule1"] = (d["p"] > d["ucl"]) | (d["p"] < d["lcl"])
    d["rule2"] = run_rule_flags(d["p"].to_numpy(), pbar, k=k_run)
    d["flagged"] = d["rule1"] | d["rule2"]
    d["rule"] = np.select(
        [d["rule1"] & d["rule2"], d["rule1"], d["rule2"]],
        ["Rule 1 + Rule 2", "Rule 1: beyond 3 sigma", f"Rule 2: {k_run} in a row on one side"],
        default="",
    )
    return d


def flagged_table(chart: pd.DataFrame) -> pd.DataFrame:
    cols = ["shift_id", "date", "shift_label", "sku_id", "total_units", "reject_units", "p", "center", "ucl", "rule"]
    cols = [c for c in cols if c in chart.columns]
    return chart.loc[chart["flagged"], cols].reset_index(drop=True)

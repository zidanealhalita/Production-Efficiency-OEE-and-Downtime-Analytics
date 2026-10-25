"""OEE maths, weighted aggregation and the OEE loss tree.

Definitions (match how the data was generated):
    Availability = operating_time / planned_production_time
    Performance  = total_units / (operating_time * ideal_rate)
    Quality      = good_units / total_units
    OEE          = A * P * Q

Aggregation rule: SUM numerators and denominators first, THEN divide (weighted OEE).
Averaging shift-level percentages over-weights short / low-volume shifts and gives a
different (wrong) answer, see ``naive_oee`` and the unit tests.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .config import AVAILABILITY_CATEGORIES, LOSS_CATEGORIES

REQUIRED = ["planned_production_time_min", "operating_time_min", "ideal_rate_units_per_min",
            "total_units", "good_units"]


def safe_div(num, den):
    """Division that returns NaN (not inf / error) when the denominator is 0 or NaN."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den == 0, np.nan, num / den)
    return out if out.ndim else float(out)


def _with_ideal_output(df: pd.DataFrame) -> pd.DataFrame:
    if "ideal_output" in df.columns:
        return df
    return df.assign(ideal_output=df["operating_time_min"] * df["ideal_rate_units_per_min"])


def oee_from_sums(planned: float, operating: float, ideal_output: float, total: float, good: float) -> dict:
    """OEE and its three factors from already-summed quantities (zero-division safe)."""
    a = safe_div(operating, planned)
    p = safe_div(total, ideal_output)
    q = safe_div(good, total)
    return {"availability": a, "performance": p, "quality": q, "oee": a * p * q}


def oee_summary(df: pd.DataFrame) -> dict:
    """Weighted OEE for any slice of shifts."""
    if df is None or len(df) == 0:
        return {"availability": np.nan, "performance": np.nan, "quality": np.nan, "oee": np.nan}
    d = _with_ideal_output(df)
    return oee_from_sums(
        d["planned_production_time_min"].sum(), d["operating_time_min"].sum(),
        d["ideal_output"].sum(), d["total_units"].sum(), d["good_units"].sum(),
    )


def oee_by(df: pd.DataFrame, by: Union[str, Sequence[str]]) -> pd.DataFrame:
    """Weighted A/P/Q/OEE per group (sum-then-divide). Returns one row per group."""
    d = _with_ideal_output(df)
    keys = [by] if isinstance(by, str) else list(by)
    g = d.groupby(keys, observed=True).agg(
        planned=("planned_production_time_min", "sum"),
        operating=("operating_time_min", "sum"),
        ideal_output=("ideal_output", "sum"),
        total_units=("total_units", "sum"),
        good_units=("good_units", "sum"),
        reject_units=("reject_units", "sum") if "reject_units" in d.columns else ("total_units", "sum"),
        shifts=("operating_time_min", "size"),
    ).reset_index()
    if "reject_units" not in d.columns:
        g["reject_units"] = g["total_units"] - g["good_units"]
    g["availability"] = safe_div(g["operating"], g["planned"])
    g["performance"] = safe_div(g["total_units"], g["ideal_output"])
    g["quality"] = safe_div(g["good_units"], g["total_units"])
    g["oee"] = g["availability"] * g["performance"] * g["quality"]
    g["downtime_h"] = (g["planned"] - g["operating"]) / 60.0
    return g


def naive_oee(df: pd.DataFrame) -> float:
    """The WRONG way: simple mean of shift-level OEE. Kept only to demonstrate the difference."""
    d = _with_ideal_output(df)
    a = safe_div(d["operating_time_min"], d["planned_production_time_min"])
    p = safe_div(d["total_units"], d["ideal_output"])
    q = safe_div(d["good_units"], d["total_units"])
    return float(np.nanmean(a * p * q))


def weekly_oee(df: pd.DataFrame, by: str = "line_id", rolling: int = 4, week_col: str = "week") -> pd.DataFrame:
    """Weekly weighted OEE per group plus a rolling mean of the weekly values."""
    w = oee_by(df, [by, week_col]).sort_values([by, week_col])
    w["oee_roll"] = w.groupby(by)["oee"].transform(lambda s: s.rolling(rolling, min_periods=1).mean())
    return w


def benchmark_band(oee: float) -> str:
    """Rule-of-thumb band: >=85% world class, 60-85% typical, <60% low."""
    if oee is None or (isinstance(oee, float) and np.isnan(oee)):
        return "n/a"
    if oee >= 0.85:
        return "World-class (>=85%)"
    if oee >= 0.60:
        return "Typical (60-85%)"
    return "Low (<60%)"


# ----------------------------------------------------------------------------------------------
# Loss tree
# ----------------------------------------------------------------------------------------------
def loss_tree(df: pd.DataFrame) -> pd.DataFrame:
    """OEE loss tree in minutes.

    planned time -> 7 availability losses -> minor stoppages -> reduced speed -> rejects
    -> fully productive time.

    * availability losses = minutes of the 7 availability loss categories (they sum exactly to
      planned - operating time in this dataset)
    * minor stoppages = minutes of 'Minor stoppage' events (they happen while the line runs)
    * reduced speed = (operating - total/ideal_rate) - minor stoppages  (residual speed loss)
    * quality loss = reject_units / ideal_rate
    * fully productive time = good_units / ideal_rate
    """
    def col(name):
        return float(df[name].sum()) if name in df.columns else 0.0

    planned = col("planned_production_time_min")
    operating = col("operating_time_min")
    rate_time_total = float((df["total_units"] / df["ideal_rate_units_per_min"]).sum())
    rate_time_good = float((df["good_units"] / df["ideal_rate_units_per_min"]).sum())
    minor = col("min_minor_stoppage")
    speed_total = operating - rate_time_total
    rows: List[dict] = [{"label": "Planned production time", "minutes": planned, "kind": "total"}]
    for slug in AVAILABILITY_CATEGORIES:
        rows.append({"label": LOSS_CATEGORIES[slug], "minutes": -col(f"min_{slug}"), "kind": "availability"})
    rows.append({"label": "Minor stoppage", "minutes": -minor, "kind": "performance"})
    rows.append({"label": "Reduced speed", "minutes": -(speed_total - minor), "kind": "performance"})
    rows.append({"label": "Rejects (quality loss)", "minutes": -(rate_time_total - rate_time_good), "kind": "quality"})
    rows.append({"label": "Fully productive time", "minutes": rate_time_good, "kind": "total"})
    tree = pd.DataFrame(rows)
    tree["pct_of_planned"] = safe_div(tree["minutes"].abs(), planned)
    return tree


def check_loss_tree(tree: pd.DataFrame, tol: float = 1e-6) -> float:
    """Return planned - sum(losses) - fully productive (should be ~0). Used in tests and the UI."""
    planned = tree.loc[tree["kind"] == "total", "minutes"].iloc[0]
    productive = tree.loc[tree["kind"] == "total", "minutes"].iloc[-1]
    losses = -tree.loc[tree["kind"] != "total", "minutes"].sum()
    return float(planned - losses - productive)

"""Pareto tables, drill-down helpers and MTBF / MTTR."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def pareto_table(df: pd.DataFrame, by: str, value: str = "duration_min", top_n: Optional[int] = None) -> pd.DataFrame:
    """Sorted Pareto table with share and cumulative share.

    Columns: <by>, minutes, events, share, cum_share. ``top_n`` keeps the largest items and
    folds the rest into 'Other' (still included in the cumulative maths).
    """
    if len(df) == 0:
        return pd.DataFrame(columns=[by, "minutes", "events", "share", "cum_share"])
    t = df.groupby(by, observed=True).agg(minutes=(value, "sum"), events=(value, "size")).reset_index()
    t = t.sort_values("minutes", ascending=False).reset_index(drop=True)
    if top_n is not None and len(t) > top_n:
        head, tail = t.iloc[:top_n], t.iloc[top_n:]
        other = pd.DataFrame({by: ["Other"], "minutes": [tail["minutes"].sum()], "events": [tail["events"].sum()]})
        t = pd.concat([head, other], ignore_index=True)
    total = t["minutes"].sum()
    t["share"] = t["minutes"] / total if total else np.nan
    t["cum_share"] = t["share"].cumsum()
    return t


def vital_few(table: pd.DataFrame, threshold: float = 0.8) -> int:
    """How many items are needed to reach ``threshold`` of the total (the '80% mark')."""
    if len(table) == 0:
        return 0
    return int((table["cum_share"] < threshold).sum() + 1)


def drilldown(events: pd.DataFrame, category: Optional[str] = None, equipment: Optional[str] = None) -> pd.DataFrame:
    """Return the pareto for the next level: category -> equipment -> reason."""
    if category is None:
        return pareto_table(events, "loss_category")
    sub = events[events["loss_category"] == category]
    if equipment is None:
        return pareto_table(sub, "equipment")
    return pareto_table(sub[sub["equipment"] == equipment], "reason")


def mtbf_mttr(events: pd.DataFrame, shifts: pd.DataFrame, min_events: int = 3) -> pd.DataFrame:
    """MTBF / MTTR per line x equipment from Breakdown events.

    MTBF (h) = operating hours of the line / number of breakdowns on that equipment
    MTTR (min) = mean breakdown duration
    Operating hours come from the (filtered) shifts, so the metric respects the sidebar filters.
    """
    bd = events[events["loss_category"] == "Breakdown"]
    if len(bd) == 0:
        return pd.DataFrame(columns=["line_id", "equipment", "breakdowns", "downtime_h", "mttr_min", "mtbf_h"])
    op_h = shifts.groupby("line_id")["operating_time_min"].sum() / 60.0
    t = bd.groupby(["line_id", "equipment"]).agg(
        breakdowns=("event_id", "count"),
        downtime_min=("duration_min", "sum"),
        mttr_min=("duration_min", "mean"),
    ).reset_index()
    t["downtime_h"] = t["downtime_min"] / 60.0
    t["operating_h"] = t["line_id"].map(op_h)
    t["mtbf_h"] = t["operating_h"] / t["breakdowns"]
    t = t[t["breakdowns"] >= min_events].sort_values("downtime_h", ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    return t[["rank", "line_id", "equipment", "breakdowns", "downtime_h", "mttr_min", "mtbf_h"]]


def count_vs_minutes(events: pd.DataFrame, by: str = "reason", top_n: int = 10) -> pd.DataFrame:
    """Rank by event count and by minutes side by side (frequency vs severity)."""
    g = events.groupby(by).agg(events=("duration_min", "size"), minutes=("duration_min", "sum")).reset_index()
    g["mean_min"] = g["minutes"] / g["events"]
    g["rank_by_count"] = g["events"].rank(ascending=False, method="min").astype(int)
    g["rank_by_minutes"] = g["minutes"].rank(ascending=False, method="min").astype(int)
    return g.sort_values("minutes", ascending=False).head(top_n).reset_index(drop=True)

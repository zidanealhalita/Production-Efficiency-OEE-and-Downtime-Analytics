"""Loading, joining and deriving columns. Pure pandas (no Streamlit) so it is testable.

The raw CSVs are never modified; everything derived is added to in-memory copies.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

from .config import AVAILABILITY_CATEGORIES, LOSS_CATEGORIES, RAINY_MONTHS, SLUG_BY_LABEL

FILES = {
    "shifts": "shift_production.csv",
    "events": "downtime_events.csv",
    "quality": "quality_defects.csv",
    "maint": "maintenance_log.csv",
    "lines": "line_master.csv",
    "plants": "plant_master.csv",
    "rates": "line_sku_rates.csv",
}


def load_raw(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Read all CSVs. Dates are parsed; nothing else is changed."""
    raw = {}
    for key, fn in FILES.items():
        raw[key] = pd.read_csv(os.path.join(data_dir, fn))
    raw["shifts"]["date"] = pd.to_datetime(raw["shifts"]["date"])
    raw["events"]["date"] = pd.to_datetime(raw["events"]["date"])
    raw["events"]["start_ts"] = pd.to_datetime(raw["events"]["start_ts"])
    raw["events"]["end_ts"] = pd.to_datetime(raw["events"]["end_ts"])
    raw["quality"]["date"] = pd.to_datetime(raw["quality"]["date"])
    raw["maint"]["date"] = pd.to_datetime(raw["maint"]["date"])
    return raw


def _calendar_cols(df: pd.DataFrame) -> pd.DataFrame:
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["month_label"] = df["date"].dt.strftime("%Y-%m")
    df["week"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")
    df["dow"] = df["date"].dt.dayofweek
    return df


def build_shift_table(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Shift-level table with OEE inputs, loss minutes and flags (idle-before, SKU change)."""
    sp = raw["shifts"].copy()
    ev = raw["events"]
    lines = raw["lines"].set_index("line_id")
    rates = raw["rates"]

    sp = _calendar_cols(sp)
    sp["shift_label"] = "S" + sp["shift"].astype(str)
    sp["line_name"] = sp["line_id"].map(lines["line_name"])
    sp["family"] = sp["line_id"].map(lines["family"])
    sp = sp.merge(rates[["line_id", "sku_id", "unit_weight_kg"]], on=["line_id", "sku_id"], how="left")

    # --- OEE building blocks -----------------------------------------------------------------
    sp["ideal_output"] = sp["operating_time_min"] * sp["ideal_rate_units_per_min"]
    sp["availability"] = sp["operating_time_min"] / sp["planned_production_time_min"]
    sp["performance"] = sp["total_units"] / sp["ideal_output"]
    sp["quality"] = sp["good_units"] / sp["total_units"]
    sp["oee"] = sp["availability"] * sp["performance"] * sp["quality"]

    # --- downtime minutes per loss category, per shift ---------------------------------------
    piv = (
        ev.assign(slug=ev["loss_category"].map(SLUG_BY_LABEL))
        .pivot_table(index="shift_id", columns="slug", values="duration_min", aggfunc="sum", fill_value=0)
    )
    for slug in LOSS_CATEGORIES:
        col = f"min_{slug}"
        sp[col] = sp["shift_id"].map(piv[slug]) if slug in piv.columns else 0.0
        sp[col] = sp[col].fillna(0.0)

    bd = ev[ev["loss_category"] == "Breakdown"].groupby("shift_id").agg(
        n_breakdowns=("event_id", "count"), max_breakdown_min=("duration_min", "max")
    )
    sp["n_breakdowns"] = sp["shift_id"].map(bd["n_breakdowns"]).fillna(0).astype(int)
    sp["max_breakdown_min"] = sp["shift_id"].map(bd["max_breakdown_min"]).fillna(0.0)

    # --- OEE loss tree minutes ---------------------------------------------------------------
    sp["avail_loss_min"] = sp["planned_production_time_min"] - sp["operating_time_min"]
    sp["speed_loss_min"] = sp["operating_time_min"] - sp["total_units"] / sp["ideal_rate_units_per_min"]
    sp["reduced_speed_min"] = sp["speed_loss_min"] - sp["min_minor_stoppage"]
    sp["quality_loss_min"] = sp["reject_units"] / sp["ideal_rate_units_per_min"]
    sp["fully_productive_min"] = sp["good_units"] / sp["ideal_rate_units_per_min"]

    # --- line history: previous shift idle? SKU change? ---------------------------------------
    sp = sp.sort_values(["line_id", "date", "shift"]).reset_index(drop=True)
    sp["slot"] = (sp["date"] - sp["date"].min()).dt.days * 3 + (sp["shift"] - 1)
    prev_slot = sp.groupby("line_id")["slot"].shift(1)
    # Previous shift slot of the line was NOT run (no row) -> start-up after idle.
    sp["prev_idle"] = (prev_slot.notna() & ((sp["slot"] - prev_slot) > 1))
    sp["prev_sku"] = sp.groupby("line_id")["sku_id"].shift(1)
    sp["sku_change"] = sp["prev_sku"].notna() & (sp["prev_sku"] != sp["sku_id"])
    sp["prev_idle"] = sp["prev_idle"].astype(bool)
    sp["sku_change"] = sp["sku_change"].astype(bool)
    sp["transition"] = np.where(sp["sku_change"], sp["prev_sku"].astype(str) + " > " + sp["sku_id"].astype(str), "")
    sp["rainy_season"] = sp["date"].dt.month.isin(RAINY_MONTHS)
    liquid = sp["family"] != "POWDER"
    sp["litres_eq"] = np.where(liquid, sp["good_units"] * sp["unit_weight_kg"], np.nan)  # 1 kg ~ 1 L
    return sp


def build_event_table(raw: Dict[str, pd.DataFrame], shifts: pd.DataFrame) -> pd.DataFrame:
    """Downtime events enriched with shift attributes."""
    ev = raw["events"].copy()
    ev = _calendar_cols(ev)
    ev["hour"] = ev["start_ts"].dt.hour
    attrs = shifts.set_index("shift_id")[["plant_id", "shift", "shift_label", "sku_id", "prev_idle", "sku_change", "prev_sku"]]
    ev = ev.join(attrs, on="shift_id")
    ev["slug"] = ev["loss_category"].map(SLUG_BY_LABEL)
    return ev


def build_quality_table(raw: Dict[str, pd.DataFrame], shifts: pd.DataFrame) -> pd.DataFrame:
    """Defect rows enriched with shift attributes (units produced, SKU change flag...)."""
    q = raw["quality"].copy()
    q = _calendar_cols(q)
    attrs = shifts.set_index("shift_id")[
        ["plant_id", "shift", "shift_label", "sku_id", "total_units", "prev_idle", "sku_change"]
    ]
    return q.join(attrs, on="shift_id")


def build_maintenance_table(raw: Dict[str, pd.DataFrame], events: pd.DataFrame) -> pd.DataFrame:
    """Work orders with plant and calendar columns; labour cost is added later (rate is an assumption)."""
    m = raw["maint"].copy()
    m = _calendar_cols(m)
    m["plant_id"] = m["line_id"].map(raw["lines"].set_index("line_id")["plant_id"])
    return m


def load_all(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load everything and return a dict of analysis-ready tables."""
    raw = load_raw(data_dir)
    shifts = build_shift_table(raw)
    events = build_event_table(raw, shifts)
    return {
        "shifts": shifts,
        "events": events,
        "quality": build_quality_table(raw, shifts),
        "maint": build_maintenance_table(raw, events),
        "lines": raw["lines"],
        "plants": raw["plants"],
        "rates": raw["rates"],
    }

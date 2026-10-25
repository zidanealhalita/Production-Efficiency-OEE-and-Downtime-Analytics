"""Streamlit glue: cached loading, shared sidebar filters, assumptions panel, KPI cards, footer."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from .config import ACTIONS, DATA_DIR, DEFAULT_ASSUMPTIONS, SKUS
from .data import load_all
from .oee import oee_summary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_data(show_spinner="Loading production data...")
def get_data() -> Dict[str, pd.DataFrame]:
    """Load + derive all tables once; cached across reruns and pages."""
    return load_all(os.path.join(ROOT, DATA_DIR))


@dataclass
class Ctx:
    """Everything a page needs: filtered tables, the unfiltered ones, filter state and assumptions."""
    shifts: pd.DataFrame          # filtered shifts (all filters)
    events: pd.DataFrame
    quality: pd.DataFrame
    maint: pd.DataFrame
    prev_shifts: pd.DataFrame     # same filters, immediately preceding period of equal length
    all_shifts: pd.DataFrame      # full data, no filters (page 5 / 7 baselines)
    all_events: pd.DataFrame
    all_maint: pd.DataFrame
    lines: pd.DataFrame
    assump: dict
    start: pd.Timestamp
    end: pd.Timestamp
    sel: Dict[str, list]
    filters_active: bool


def _assumptions_panel() -> dict:
    """One editable place for every assumption; the edited copy flows into all calculations."""
    a = copy.deepcopy(DEFAULT_ASSUMPTIONS)
    with st.sidebar.expander("Assumptions (editable)", expanded=False):
        st.caption("All money, CO2e and effort values are illustrative planning assumptions, not company data.")
        mt = pd.DataFrame({"sku_id": SKUS, "margin_idr_per_good_unit": [a["margin_per_unit"][s] for s in SKUS],
                           "unit_cost_idr_per_reject": [a["unit_cost"][s] for s in SKUS]})
        mt = st.data_editor(mt, hide_index=True, disabled=["sku_id"], key="ed_sku", use_container_width=True)
        a["margin_per_unit"] = dict(zip(mt["sku_id"], mt["margin_idr_per_good_unit"]))
        a["unit_cost"] = dict(zip(mt["sku_id"], mt["unit_cost_idr_per_reject"]))
        a["labour_rate_idr_per_h"] = st.number_input("Maintenance labour rate (IDR/hour)", 0, 1_000_000, int(a["labour_rate_idr_per_h"]), step=5000)
        a["grid_factor_kg_per_kwh"] = st.number_input("Grid emission factor (kg CO2e/kWh)", 0.0, 2.0, float(a["grid_factor_kg_per_kwh"]), step=0.01)
        a["idle_energy_share"] = st.slider("Energy of a stopped line vs running", 0.0, 1.0, float(a["idle_energy_share"]), 0.05)
        a["idle_water_share"] = st.slider("Water use of a stopped line vs running", 0.0, 1.0, float(a["idle_water_share"]), 0.05)
        st.markdown("**Improvement-case assumptions**")
        a["night_gap_closure"] = st.slider("Night-shift OEE gap closed", 0.0, 1.0, float(a["night_gap_closure"]), 0.05)
        a["smed_reduction"] = st.slider("SMED changeover reduction (page 7)", 0.0, 0.8, float(a["smed_reduction"]), 0.05)
        a["l06_avoidable_share"] = st.slider("L06 shortage minutes avoidable", 0.0, 1.0, float(a["l06_avoidable_share"]), 0.05)
        a["startup_fixable_share"] = st.slider("Start-up losses fixable", 0.0, 1.0, float(a["startup_fixable_share"]), 0.05)
        a["rainy_fixable_share"] = st.slider("Rainy-season utility stops fixable", 0.0, 1.0, float(a["rainy_fixable_share"]), 0.05)
        a["minor_stop_reduction"] = st.slider("Minor-stoppage minutes reduced", 0.0, 0.6, float(a["minor_stop_reduction"]), 0.05)
        et = pd.DataFrame({"action_id": list(ACTIONS), "action": [v[0] for v in ACTIONS.values()], "effort_1_easy_5_hard": [a["effort"][k] for k in ACTIONS]})
        et = st.data_editor(et, hide_index=True, disabled=["action_id", "action"], key="ed_effort", use_container_width=True,
                            column_config={"effort_1_easy_5_hard": st.column_config.NumberColumn(min_value=1, max_value=5, step=1)})
        a["effort"] = dict(zip(et["action_id"], et["effort_1_easy_5_hard"]))
    return a


def build_ctx() -> Ctx:
    """Render the shared sidebar and return the filtered context. Called once per run from app.py."""
    D = get_data()
    S, E, Q, M = D["shifts"], D["events"], D["quality"], D["maint"]
    st.sidebar.header("Filters")
    plants = sorted(S["plant_id"].unique())
    plant_names = D["plants"].set_index("plant_id")["plant_name"].to_dict()
    sel_plants = st.sidebar.multiselect("Plant", plants, default=plants, format_func=lambda p: f"{p} - {plant_names.get(p, '')}")
    line_pool = sorted(S[S["plant_id"].isin(sel_plants)]["line_id"].unique())
    line_names = D["lines"].set_index("line_id")["line_name"].to_dict()
    sel_lines = st.sidebar.multiselect("Line", line_pool, default=line_pool, format_func=lambda l: f"{l} - {line_names.get(l, '')}")
    sel_shifts = st.sidebar.multiselect("Shift", ["S1", "S2", "S3"], default=["S1", "S2", "S3"])
    sku_pool = sorted(S[S["line_id"].isin(sel_lines)]["sku_id"].unique())
    sel_skus = st.sidebar.multiselect("SKU", sku_pool, default=sku_pool)
    dmin, dmax = S["date"].min().date(), S["date"].max().date()
    dr = st.sidebar.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        start, end = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
    else:
        start, end = pd.Timestamp(dmin), pd.Timestamp(dmax)

    def apply(df: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        m = (df["date"] >= s) & (df["date"] <= e) & df["plant_id"].isin(sel_plants) & df["line_id"].isin(sel_lines)
        m &= df["shift_label"].isin(sel_shifts) & df["sku_id"].isin(sel_skus)
        return df[m]

    shifts = apply(S, start, end)
    ids = set(shifts["shift_id"])
    events = E[E["shift_id"].isin(ids)]
    quality = Q[Q["shift_id"].isin(ids)]
    span = (end - start).days + 1
    prev = apply(S, start - pd.Timedelta(days=span), start - pd.Timedelta(days=1))
    maint = M[(M["date"] >= start) & (M["date"] <= end) & M["line_id"].isin(sel_lines)]
    assump = _assumptions_panel()
    filters_active = (len(sel_shifts) < 3) or (set(sel_skus) != set(sku_pool))
    if len(shifts) == 0:
        st.warning("No shifts match the current filters. Widen the selection in the sidebar.")
        st.stop()
    return Ctx(shifts, events, quality, maint, prev, S, E, M, D["lines"], assump, start, end,
               {"plants": sel_plants, "lines": sel_lines, "shifts": sel_shifts, "skus": sel_skus}, filters_active)


def get_ctx() -> Ctx:
    return st.session_state["ctx"]


def fmt_pct(x: float, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def fmt_idr(x: float) -> str:
    """Compact IDR: 1.2 bn / 340 M."""
    a = abs(x)
    if a >= 1e9:
        return f"IDR {x / 1e9:,.2f} bn"
    if a >= 1e6:
        return f"IDR {x / 1e6:,.0f} M"
    return f"IDR {x:,.0f}"


def kpi(col, label: str, value: str, delta: Optional[float] = None, delta_fmt: str = "pts", invert: bool = False, help: Optional[str] = None):
    """KPI card with delta vs previous period. ``delta`` is in absolute units (ratio points for % KPIs)."""
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        col.metric(label, value, help=help)
        return
    txt = f"{delta * 100:+.1f} pts" if delta_fmt == "pts" else f"{delta:+,.0f}"
    col.metric(label, value, txt, delta_color="inverse" if invert else "normal", help=help)


def footer():
    st.markdown("---")
    st.caption("Synthetic data for portfolio purposes. Plant and line names are generic; they are not real Danone plants. "
               "OEE benchmark bands are a common rule of thumb, not a Danone target.")

"""Impact calculators: minutes -> units -> hours -> IDR, CO2e, resource savings, and the
ranked improvement-opportunity table.

Every money / emission number depends on ``assump`` (see ``config.DEFAULT_ASSUMPTIONS``) and is
therefore an *illustrative estimate*, not a company figure.

Core conversion: a recovered operating minute is assumed to run at the line's realised
good-units-per-operating-minute (its actual 2025 average), valued at the volume-weighted
contribution margin of the SKUs that line actually produced.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import ACTIONS, L02_AFTER, L02_PROBLEM, LOSS_CATEGORIES, RAINY_MONTHS, SLUG_BY_LABEL
from .oee import oee_summary


# ----------------------------------------------------------------------------------------------
# Value of a recovered minute
# ----------------------------------------------------------------------------------------------
def line_value_table(shifts: pd.DataFrame, assump: dict, group: str = "line_id") -> pd.DataFrame:
    """Per group: realised good units per operating minute, margin per good unit, IDR per minute."""
    margin = shifts["sku_id"].map(assump["margin_per_unit"]).astype(float)
    d = shifts.assign(_m=shifts["good_units"] * margin)
    g = d.groupby(group).agg(good=("good_units", "sum"), op=("operating_time_min", "sum"), m=("_m", "sum")).reset_index()
    g["good_per_min"] = g["good"] / g["op"]
    g["margin_per_good"] = g["m"] / g["good"]
    g["idr_per_min"] = g["good_per_min"] * g["margin_per_good"]
    return g[[group, "good_per_min", "margin_per_good", "idr_per_min"]]


def minutes_to_value(minutes_by_line: Dict[str, float], shifts: pd.DataFrame, assump: dict) -> pd.DataFrame:
    """Convert recovered minutes per line into hours, extra good units and IDR."""
    vt = line_value_table(shifts, assump).set_index("line_id")
    rows = []
    for line, mins in minutes_by_line.items():
        if line not in vt.index:
            continue
        units = mins * vt.loc[line, "good_per_min"]
        rows.append({"line_id": line, "minutes": mins, "hours": mins / 60.0, "units": units, "idr": units * vt.loc[line, "margin_per_good"]})
    return pd.DataFrame(rows, columns=["line_id", "minutes", "hours", "units", "idr"])


def annualise(value: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = max((pd.Timestamp(end) - pd.Timestamp(start)).days + 1, 1)
    return value * 365.0 / days


# ----------------------------------------------------------------------------------------------
# Changeover / SMED
# ----------------------------------------------------------------------------------------------
def changeover_events(events: pd.DataFrame) -> pd.DataFrame:
    co = events[events["loss_category"] == "Changeover"].copy()
    co["transition"] = co["prev_sku"].astype(str) + " > " + co["sku_id"].astype(str)
    return co


def changeover_stats(events: pd.DataFrame, by=("line_id",)) -> pd.DataFrame:
    co = changeover_events(events)
    if len(co) == 0:
        return pd.DataFrame(columns=list(by) + ["n", "mean", "median", "p90", "total_min"])
    g = co.groupby(list(by))["duration_min"]
    t = g.agg(n="count", mean="mean", median="median", total_min="sum").reset_index()
    t["p90"] = g.quantile(0.9).to_numpy()
    return t


def smed_whatif(shifts: pd.DataFrame, events: pd.DataFrame, reduction: float, assump: dict, start, end, line: Optional[str] = None) -> pd.DataFrame:
    """Recovered minutes / units / IDR if changeover time falls by ``reduction`` (0-1), per line, plus annualised."""
    co = events[events["loss_category"] == "Changeover"]
    if line:
        co = co[co["line_id"] == line]
    mins = (co.groupby("line_id")["duration_min"].sum() * reduction).to_dict()
    out = minutes_to_value(mins, shifts, assump)
    for c in ["minutes", "hours", "units", "idr"]:
        out[f"{c}_annual"] = annualise(out[c], start, end)
    return out


def changeover_cost_monthly(shifts: pd.DataFrame, events: pd.DataFrame, assump: dict) -> pd.DataFrame:
    """Lost contribution (IDR) from changeover minutes per month and line."""
    vt = line_value_table(shifts, assump).set_index("line_id")["idr_per_min"]
    co = events[events["loss_category"] == "Changeover"].groupby(["month", "line_id"])["duration_min"].sum().reset_index()
    co["idr"] = co["duration_min"] * co["line_id"].map(vt)
    return co


# ----------------------------------------------------------------------------------------------
# Quality & maintenance cost
# ----------------------------------------------------------------------------------------------
def copq_by(shifts: pd.DataFrame, assump: dict, by) -> pd.DataFrame:
    """Cost of poor quality = reject_units x unit cost (assumption)."""
    cost = shifts["sku_id"].map(assump["unit_cost"]).astype(float)
    d = shifts.assign(copq_idr=shifts["reject_units"] * cost)
    return d.groupby(by, observed=True).agg(reject_units=("reject_units", "sum"), copq_idr=("copq_idr", "sum")).reset_index()


def maintenance_costs(maint: pd.DataFrame, labour_rate: float) -> pd.DataFrame:
    m = maint.copy()
    m["labour_cost_idr"] = m["labor_hours"] * labour_rate
    m["total_cost_idr"] = m["labour_cost_idr"] + m["spare_parts_cost_idr"]
    return m


# ----------------------------------------------------------------------------------------------
# Sustainability
# ----------------------------------------------------------------------------------------------
def resource_intensity(shifts: pd.DataFrame, by, assump: dict) -> pd.DataFrame:
    """Energy, water, waste and CO2e per 1,000 good units (plus L per litre-equivalent where sensible)."""
    keys = [by] if isinstance(by, str) else list(by)
    g = shifts.groupby(keys, observed=True).agg(
        energy=("energy_kwh", "sum"), water=("water_l", "sum"), waste=("waste_kg", "sum"), good=("good_units", "sum"),
        litres=("litres_eq", "sum"), planned=("planned_production_time_min", "sum"), operating=("operating_time_min", "sum"),
        total=("total_units", "sum"), ideal_output=("ideal_output", "sum"), reject=("reject_units", "sum"),
    ).reset_index()
    g["kwh_per_1000"] = g["energy"] / g["good"] * 1000
    g["water_l_per_unit"] = g["water"] / g["good"]
    g["water_l_per_litre"] = np.where(g["litres"] > 0, g["water"] / g["litres"].replace(0, np.nan), np.nan)
    g["waste_kg_per_1000"] = g["waste"] / g["good"] * 1000
    g["co2e_kg_per_1000"] = g["energy"] * assump["grid_factor_kg_per_kwh"] / g["good"] * 1000
    g["oee"] = (g["operating"] / g["planned"]) * (g["total"] / g["ideal_output"]) * (g["good"] / g["total"])
    return g


def idle_energy_share_estimate(shifts: pd.DataFrame) -> pd.DataFrame:
    """Cross-check of the 'idle energy share' assumption: per line OLS energy ~ a + b*operating_min + c*stopped_min.

    c/b is the implied energy of a stopped minute relative to a running minute (clipped to [0,1]).
    """
    rows = []
    for line, g in shifts.groupby("line_id"):
        stopped = g["planned_production_time_min"] - g["operating_time_min"]
        X = np.c_[np.ones(len(g)), g["operating_time_min"], stopped, (g["shift"] == 3).astype(float)]
        y = g["energy_kwh"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append({"line_id": line, "kwh_per_running_min": beta[1], "kwh_per_stopped_min": beta[2],
                     "implied_idle_share": float(np.clip(beta[2] / beta[1], 0, 1)) if beta[1] > 0 else np.nan, "r2": r2})
    return pd.DataFrame(rows)


LEVERS = list(LOSS_CATEGORIES.values()) + ["Rejects (quality)"]


def resource_scenario(shifts: pd.DataFrame, lever: str, reduction: float, assump: dict) -> Dict[str, float]:
    """Resource savings at CONSTANT output if loss ``lever`` falls by ``reduction`` (0-1).

    Downtime levers: the recovered minutes are no longer scheduled, so the energy / water a stopped
    line burns during them is avoided (idle share assumptions). Rejects lever: fewer scrap units
    means fewer units to make for the same good output -> avoided energy, water (per unit made)
    and product waste (reject units x unit weight).
    Also reports the kWh/1,000-good-unit intensity if the recovered time is instead used to produce more.
    """
    d = shifts
    base_kwh_1000 = d["energy_kwh"].sum() / d["good_units"].sum() * 1000
    res = {"lever": lever, "recovered_min": 0.0, "avoided_kwh": 0.0, "avoided_water_l": 0.0, "avoided_waste_kg": 0.0,
           "avoided_co2e_t": 0.0, "extra_good_units": 0.0, "kwh_per_1000_before": base_kwh_1000, "kwh_per_1000_after": base_kwh_1000}
    if len(d) == 0:
        return res
    if lever == "Rejects (quality)":
        avoided_units = d["reject_units"] * reduction
        e_per_unit = d["energy_kwh"] / d["total_units"]
        w_per_unit = d["water_l"] / d["total_units"]
        res["avoided_kwh"] = float((avoided_units * e_per_unit).sum())
        res["avoided_water_l"] = float((avoided_units * w_per_unit).sum())
        res["avoided_waste_kg"] = float((avoided_units * d["unit_weight_kg"]).sum())
        res["extra_good_units"] = float(avoided_units.sum())
        res["recovered_min"] = float((avoided_units / d["ideal_rate_units_per_min"]).sum())
    else:
        slug = SLUG_BY_LABEL[lever]
        rec = d[f"min_{slug}"] * reduction
        kwh_per_op = d["energy_kwh"] / d["operating_time_min"]
        water_per_op = d["water_l"] / d["operating_time_min"]
        res["recovered_min"] = float(rec.sum())
        res["avoided_kwh"] = float((rec * assump["idle_energy_share"] * kwh_per_op).sum())
        res["avoided_water_l"] = float((rec * assump["idle_water_share"] * water_per_op).sum())
        good_per_min = d["good_units"] / d["operating_time_min"]
        res["extra_good_units"] = float((rec * good_per_min).sum())
        extra_kwh = float((rec * kwh_per_op).sum())
        res["kwh_per_1000_after"] = (d["energy_kwh"].sum() + extra_kwh) / (d["good_units"].sum() + res["extra_good_units"]) * 1000
    res["avoided_co2e_t"] = res["avoided_kwh"] * assump["grid_factor_kg_per_kwh"] / 1000.0
    return res


# ----------------------------------------------------------------------------------------------
# Improvement opportunities
# ----------------------------------------------------------------------------------------------
def _l02_filler_monthly(events: pd.DataFrame) -> pd.Series:
    e = events[(events["line_id"] == "L02") & (events["loss_category"] == "Breakdown") & (events["equipment"] == "Filler")]
    return e.groupby("month")["duration_min"].sum()


def l02_before_after(shifts: pd.DataFrame, events: pd.DataFrame) -> Dict[str, object]:
    """Numbers for the L02 Filler before / after panel (Apr-Sep vs Oct-Dec 2025)."""
    l02 = shifts[shifts["line_id"] == "L02"]
    before = l02[(l02["date"] >= L02_PROBLEM[0]) & (l02["date"] <= L02_PROBLEM[1])]
    after = l02[(l02["date"] >= L02_AFTER[0]) & (l02["date"] <= L02_AFTER[1])]
    fm = _l02_filler_monthly(events)
    fb = fm[(fm.index >= pd.Timestamp(L02_PROBLEM[0])) & (fm.index <= pd.Timestamp(L02_PROBLEM[1]))]
    fa = fm[(fm.index >= pd.Timestamp(L02_AFTER[0])) & (fm.index <= pd.Timestamp(L02_AFTER[1]))]
    return {
        "before": oee_summary(before), "after": oee_summary(after), "before_df": before, "after_df": after,
        "filler_h_per_month_before": float(fb.mean() / 60), "filler_h_per_month_after": float(fa.mean() / 60),
        "filler_h_total_before": float(fb.sum() / 60), "filler_h_total_after": float(fa.sum() / 60),
        "filler_monthly": fm,
    }


def build_opportunities(shifts: pd.DataFrame, events: pd.DataFrame, assump: dict) -> pd.DataFrame:
    """Ranked list of improvement actions with annual impact in minutes, hours, units and IDR.

    Uses the whole 2025 dataset (one year) so that 'per year' is literal. Estimates overlap partly
    (e.g. a Filler fix also lifts the night shift), so the total is an upper bound.
    """
    vt = line_value_table(shifts, assump).set_index("line_id")
    rows = []

    def add(aid, mins_by_line, evidence, basis, units_override=None):
        v = minutes_to_value(mins_by_line, shifts, assump)
        units = float(v["units"].sum()) if units_override is None else units_override["units"]
        idr = float(v["idr"].sum()) if units_override is None else units_override["idr"]
        hours = float(v["hours"].sum()) if units_override is None else units_override["hours"]
        rows.append({
            "id": aid, "action": ACTIONS[aid][0], "lines": ", ".join(sorted(mins_by_line.keys())) if mins_by_line else "",
            "hours": hours, "units": units, "idr": idr, "effort": assump["effort"].get(aid, ACTIONS[aid][1]),
            "evidence": evidence, "basis": basis,
        })

    # 1) L02 Filler: monthly Filler breakdown minutes, problem regime vs after the fix
    fm = _l02_filler_monthly(events)
    fb = fm[(fm.index >= pd.Timestamp(L02_PROBLEM[0])) & (fm.index <= pd.Timestamp(L02_PROBLEM[1]))].mean()
    fa = fm[(fm.index >= pd.Timestamp(L02_AFTER[0])) & (fm.index <= pd.Timestamp(L02_AFTER[1]))].mean()
    delta_month = max(fb - fa, 0)
    add("l02_filler", {"L02": delta_month * 12}, "Page 2 (MTBF/MTTR), Page 5 (early warning), Page 7 before/after",
        f"Filler breakdown fell from {fb/60:,.0f} to {fa/60:,.0f} h/month; run-rate x12 vs the unfixed regime")

    # 2) SMED L05
    co = events[(events["loss_category"] == "Changeover") & (events["line_id"] == "L05")]["duration_min"].sum()
    red = assump["smed_reduction"]
    add("smed_l05", {"L05": co * red}, "Page 3 (changeover analysis, SMED what-if)",
        f"L05 changeovers {co/60:,.0f} h/yr, cut by {red:.0%}")

    # 3) night shift gap (line-matched)
    mins, units_tot, hours_tot, idr_tot = {}, 0.0, 0.0, 0.0
    for line, g in shifts.groupby("line_id"):
        day, night = g[g["shift"] != 3], g[g["shift"] == 3]
        gap = oee_summary(day)["oee"] - oee_summary(night)["oee"]
        if gap <= 0 or len(night) == 0:
            continue
        cap_units = float((night["planned_production_time_min"] * night["ideal_rate_units_per_min"]).sum())
        planned_min = float(night["planned_production_time_min"].sum())
        u = gap * assump["night_gap_closure"] * cap_units
        mg = float((night["good_units"] * night["sku_id"].map(assump["margin_per_unit"])).sum() / night["good_units"].sum())
        mins[line] = gap * assump["night_gap_closure"] * planned_min
        units_tot += u
        hours_tot += mins[line] / 60
        idr_tot += u * mg
    add("night_shift", mins, "Page 3 (shift effect regression)",
        f"Close {assump['night_gap_closure']:.0%} of each line's S3 vs S1/S2 OEE gap", {"units": units_tot, "idr": idr_tot, "hours": hours_tot})

    # 4) L06 material shortage Jul-Sep
    ms = events[(events["line_id"] == "L06") & (events["loss_category"] == "Material shortage")].groupby("month")["duration_min"].sum()
    in_spike = ms.index.month.isin([7, 8, 9])
    base = ms[~in_spike].mean()
    excess = float((ms[in_spike] - base).clip(lower=0).sum())
    add("l06_material", {"L06": excess * assump["l06_avoidable_share"]}, "Page 3 (monthly shortage trend), Page 2 (Pareto)",
        f"Jul-Sep shortage minutes above the {base/60:,.1f} h/month baseline; {assump['l06_avoidable_share']:.0%} avoidable")

    # 5) start-up after idle (line-matched rates)
    mins, extra_units = {}, 0.0
    idr_extra = 0.0
    for line, g in shifts.groupby("line_id"):
        idle, run = g[g["prev_idle"]], g[~g["prev_idle"]]
        if len(idle) == 0:
            continue
        rate_run = run["min_breakdown"].sum() / (run["operating_time_min"].sum() / 60)
        exp_bd = rate_run * idle["operating_time_min"].sum() / 60
        ex_min = max(idle["min_breakdown"].sum() - exp_bd, 0) * assump["startup_fixable_share"]
        rej_run = run["reject_units"].sum() / run["total_units"].sum()
        ex_rej = max(idle["reject_units"].sum() - rej_run * idle["total_units"].sum(), 0) * assump["startup_fixable_share"]
        mins[line] = ex_min
        mg = vt.loc[line, "margin_per_good"]
        extra_units += ex_rej
        idr_extra += ex_rej * mg
    v = minutes_to_value(mins, shifts, assump)
    add("startup_idle", mins, "Page 3 (start-up-after-idle effect)",
        f"Excess breakdown minutes and rejects in the {int(shifts['prev_idle'].sum())} shifts that followed an idle shift; {assump['startup_fixable_share']:.0%} fixable",
        {"units": float(v["units"].sum()) + extra_units, "idr": float(v["idr"].sum()) + idr_extra, "hours": float(v["hours"].sum())})

    # 6) rainy season utilities
    mins = {}
    for line, g in shifts.groupby("line_id"):
        rain = g[g["date"].dt.month.isin(RAINY_MONTHS)]
        dry = g[~g["date"].dt.month.isin(RAINY_MONTHS)]
        rr = rain["min_utilities"].sum() / (rain["operating_time_min"].sum() / 60)
        rd = dry["min_utilities"].sum() / (dry["operating_time_min"].sum() / 60)
        ex = max(rr - rd, 0) * rain["operating_time_min"].sum() / 60 * assump["rainy_fixable_share"]
        if ex > 0:
            mins[line] = ex
    add("rainy_utilities", mins, "Page 3 / Page 2 (utilities losses by season)",
        f"Utility stop minutes per operating hour in Nov-Mar above Apr-Oct; {assump['rainy_fixable_share']:.0%} fixable")

    # 7) minor stoppages
    ms_min = (events[events["loss_category"] == "Minor stoppage"].groupby("line_id")["duration_min"].sum() * assump["minor_stop_reduction"]).to_dict()
    add("minor_stoppage", ms_min, "Page 2 (Pareto by category -> reason)",
        f"Minor-stoppage minutes cut by {assump['minor_stop_reduction']:.0%}")

    t = pd.DataFrame(rows).sort_values("idr", ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    return t

"""Tests for Pareto, SPC, statistics, anomaly, impact helpers."""
import math

import numpy as np
import pandas as pd

from src.anomaly import lead_time_weeks, mad_flags
from src.impact import annualise, minutes_to_value
from src.pareto import mtbf_mttr, pareto_table, vital_few
from src.spc import p_chart, run_rule_flags
from src.stats import bootstrap_ci, poisson_rate_ratio, two_proportion_z


def test_pareto_sorted_and_cumulative():
    df = pd.DataFrame({"cat": ["a", "b", "c", "a"], "duration_min": [10, 50, 5, 35]})
    t = pareto_table(df, "cat")
    assert list(t["cat"]) == ["b", "a", "c"]
    assert math.isclose(t["cum_share"].iloc[-1], 1.0)
    assert math.isclose(t["share"].iloc[0], 0.5)
    assert vital_few(t, 0.8) == 2


def test_mtbf_mttr():
    ev = pd.DataFrame({"event_id": range(4), "line_id": "L1", "equipment": "Filler", "loss_category": "Breakdown", "duration_min": [10, 20, 30, 40]})
    sh = pd.DataFrame({"line_id": ["L1"], "operating_time_min": [6000]})   # 100 operating hours
    t = mtbf_mttr(ev, sh, min_events=3).iloc[0]
    assert t["breakdowns"] == 4 and math.isclose(t["mtbf_h"], 25.0) and math.isclose(t["mttr_min"], 25.0)


def test_run_rule_flags_eight_in_a_row():
    v = np.array([1.0] * 7 + [1.0, 1.0, -1.0, 1.0])
    f = run_rule_flags(v, 0.0, k=8)
    assert list(np.where(f)[0]) == [7, 8]
    assert not run_rule_flags(np.array([1.0] * 7), 0.0).any()


def test_p_chart_flags_outlier():
    rng = np.random.default_rng(0)
    n = 10000
    rej = rng.binomial(n, 0.02, 60)
    rej[30] = 600                                          # clear special cause
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=60), "shift": 1, "total_units": n, "reject_units": rej})
    ch = p_chart(df, laney=False)
    assert ch.loc[30, "rule1"] and ch["rule1"].sum() >= 1
    assert (ch["lcl"] >= 0).all() and (ch["ucl"] > ch["center"]).all()


def test_stats_helpers():
    z = two_proportion_z(60, 1000, 30, 1000)
    assert z["diff"] > 0 and z["p_value"] < 0.01
    rr = poisson_rate_ratio(40, 100, 20, 100)
    assert math.isclose(rr["ratio"], 2.0) and rr["ci_low"] < 2.0 < rr["ci_high"]
    a, b = np.array([1.0, 2, 3, 4]), np.array([1.0, 1, 2, 2])
    c1, c2 = bootstrap_ci(a, b, n_boot=200, seed=1), bootstrap_ci(a, b, n_boot=200, seed=1)
    assert c1 == c2 and c1["ci_low"] <= c1["diff"] <= c1["ci_high"]


def test_mad_alarm_and_lead_time():
    weeks = pd.date_range("2025-01-06", periods=30, freq="W-MON")
    vals = [100 + (i % 3) * 5 for i in range(15)] + [400] * 15      # step change at week 15
    wk = pd.DataFrame({"line_id": "L1", "week": weeks, "breakdown_min": vals})
    out = mad_flags(wk, window=8, min_baseline=6, k=3)
    assert not out["mad_flag"].iloc[:15].any()
    assert out["mad_flag"].iloc[15:].all()                          # stays on: baseline excludes alarm weeks
    lt = lead_time_weeks(out, "L1")
    assert lt["first_alarm"] == weeks[15] and lt["lead_weeks"] >= 0


def test_minutes_to_value_and_annualise():
    shifts = pd.DataFrame({"line_id": ["L1", "L1"], "sku_id": ["X", "X"], "good_units": [6000, 4000], "operating_time_min": [60, 40]})
    a = {"margin_per_unit": {"X": 100.0}}
    v = minutes_to_value({"L1": 10}, shifts, a).iloc[0]
    assert math.isclose(v["units"], 1000.0) and math.isclose(v["idr"], 100000.0) and math.isclose(v["hours"], 10 / 60)
    assert math.isclose(annualise(1.0, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")), 365 / 181)

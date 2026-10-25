"""Unit tests for the OEE maths. Run with:  python -m pytest -q   (or python tests/run_tests.py)."""
import math
import os

import numpy as np
import pandas as pd

from src.oee import check_loss_tree, loss_tree, naive_oee, oee_by, oee_summary, safe_div


def _row(planned, op, rate, total, good, **extra):
    d = dict(planned_production_time_min=planned, operating_time_min=op, ideal_rate_units_per_min=rate, total_units=total, good_units=good, reject_units=total - good)
    d.update(extra)
    return d


def close(a, b, tol=1e-9):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def test_perfect_line_is_100_percent():
    df = pd.DataFrame([_row(450, 450, 100, 45000, 45000)])
    s = oee_summary(df)
    assert close(s["availability"], 1) and close(s["performance"], 1) and close(s["quality"], 1) and close(s["oee"], 1)


def test_hand_calculated_example():
    # A = 360/450 = 0.80 ; P = 54000/(360*200) = 0.75 ; Q = 52380/54000 = 0.97 ; OEE = 0.582
    df = pd.DataFrame([_row(450, 360, 200, 54000, 52380)])
    s = oee_summary(df)
    assert close(s["availability"], 0.80)
    assert close(s["performance"], 0.75)
    assert close(s["quality"], 0.97)
    assert close(s["oee"], 0.582)


def test_weighted_differs_from_naive_average():
    small = _row(100, 100, 10, 1000, 1000)      # OEE 100 %, tiny shift
    big = _row(1000, 500, 10, 2500, 2000)       # OEE 20 %, large shift
    df = pd.DataFrame([small, big])
    naive = naive_oee(df)
    weighted = oee_summary(df)["oee"]
    assert close(naive, 0.6)                    # simple mean of 1.0 and 0.2
    a, p, q = 600 / 1100, 3500 / 6000, 3000 / 3500
    assert close(weighted, a * p * q)           # sum-then-divide, ~0.2727
    assert abs(naive - weighted) > 0.3          # the naive average badly overstates OEE


def test_zero_division_guards():
    df = pd.DataFrame([_row(0, 0, 100, 0, 0)])
    s = oee_summary(df)
    assert np.isnan(s["oee"])
    assert np.isnan(oee_summary(pd.DataFrame(columns=list(_row(1, 1, 1, 1, 1).keys())))["oee"])
    assert np.isnan(safe_div(1, 0)) and safe_div(6, 3) == 2.0
    arr = safe_div(np.array([1.0, 2.0]), np.array([0.0, 4.0]))
    assert np.isnan(arr[0]) and close(arr[1], 0.5)


def test_oee_by_matches_summary_and_sums_groups():
    df = pd.DataFrame([_row(450, 400, 100, 36000, 35000, line_id="A"), _row(450, 300, 100, 28000, 27000, line_id="A"), _row(450, 450, 50, 22000, 22000, line_id="B")])
    g = oee_by(df, "line_id").set_index("line_id")
    assert close(g.loc["A", "oee"], oee_summary(df[df["line_id"] == "A"])["oee"])
    assert close(g.loc["B", "availability"], 1.0)
    assert g["good_units"].sum() == df["good_units"].sum()


def _consistent_frame():
    # planned 450; breakdown 50 + changeover 30 -> operating 370; minor stops 20; speed loss 60 (=> 40 reduced speed)
    rate = 100
    op, speed_total, rejects = 370, 60, 1000
    total = (op - speed_total) * rate           # 31,000
    good = total - rejects                      # 30,000
    return pd.DataFrame([_row(450, op, rate, total, good, min_breakdown=50, min_changeover=30, min_minor_stoppage=20)])


def test_loss_tree_reconciles():
    tree = loss_tree(_consistent_frame())
    assert abs(check_loss_tree(tree)) < 1e-6
    prod = tree["minutes"].iloc[-1]
    assert close(prod, 300.0)                   # good_units / ideal_rate
    assert close(tree.loc[tree["label"] == "Reduced speed", "minutes"].iloc[0], -40.0)
    assert close(tree.loc[tree["label"] == "Rejects (quality loss)", "minutes"].iloc[0], -10.0)


def test_real_data_reconciles_if_present():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data_production")
    if not os.path.exists(os.path.join(path, "shift_production.csv")):
        return
    from src.data import load_all
    D = load_all(path)
    s = D["shifts"]
    assert abs(check_loss_tree(loss_tree(s))) < 1e-3
    # logged availability events equal planned - operating time in every shift
    assert (s["avail_loss_min"] - s[[c for c in s.columns if c.startswith("min_") and c != "min_minor_stoppage"]].sum(axis=1)).abs().max() < 1e-6
    # weighted OEE differs from the naive mean on the real data
    assert abs(naive_oee(s) - oee_summary(s)["oee"]) > 0.001

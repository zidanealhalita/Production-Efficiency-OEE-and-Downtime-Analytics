"""Early-warning analytics: robust (MAD) alarms, IsolationForest, risk score, breakdown model."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------------------------
# Line-week table
# ----------------------------------------------------------------------------------------------
def line_week_table(shifts: pd.DataFrame, events: pd.DataFrame, full_weeks_only: bool = True,
                    data_start: Optional[pd.Timestamp] = None, data_end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """One row per line x week: breakdown minutes / count, reject rate, speed ratio (performance)."""
    bd = events[events["loss_category"] == "Breakdown"].groupby(["line_id", "week"]).agg(
        breakdown_min=("duration_min", "sum"), breakdown_n=("event_id", "count")).reset_index()
    sh = shifts.assign(ideal_output=shifts["operating_time_min"] * shifts["ideal_rate_units_per_min"])
    w = sh.groupby(["line_id", "week"]).agg(
        op_min=("operating_time_min", "sum"), total=("total_units", "sum"), reject=("reject_units", "sum"),
        ideal_output=("ideal_output", "sum")).reset_index()
    w = w.merge(bd, on=["line_id", "week"], how="left").fillna({"breakdown_min": 0, "breakdown_n": 0})
    w["reject_rate"] = w["reject"] / w["total"]
    w["speed_ratio"] = w["total"] / w["ideal_output"]
    if full_weeks_only:
        lo = data_start if data_start is not None else shifts["date"].min()
        hi = data_end if data_end is not None else shifts["date"].max()
        w = w[(w["week"] >= lo) & (w["week"] + pd.Timedelta(days=6) <= hi)]
    return w.sort_values(["line_id", "week"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# Robust threshold alarm (median + k * scaled MAD) with a frozen, non-contaminated baseline
# ----------------------------------------------------------------------------------------------
def mad_flags(wk: pd.DataFrame, value: str = "breakdown_min", window: int = 12, min_baseline: int = 8, k: float = 3.0) -> pd.DataFrame:
    """Flag weeks above ``median + k * 1.4826 * MAD`` of a trailing baseline.

    The baseline is the last ``window`` NON-flagged weeks of the same line, so a long
    problem does not 'teach' the baseline that bad is normal (otherwise the alarm would
    switch itself off after a few weeks). No flag is raised until ``min_baseline`` weeks exist.
    """
    out = []
    for _, g in wk.sort_values("week").groupby("line_id"):
        g = g.copy()
        baseline: list = []
        med, thr, flag = [], [], []
        for v in g[value].to_numpy():
            if len(baseline) >= min_baseline:
                b = np.array(baseline[-window:])
                m = np.median(b)
                mad = np.median(np.abs(b - m)) * 1.4826
                t = m + k * max(mad, 0.05 * m)  # floor stops a perfectly flat baseline from over-alarming
                f = v > t
            else:
                m, t, f = np.nan, np.nan, False
            med.append(m); thr.append(t); flag.append(bool(f))
            if not f:
                baseline.append(v)
        g["baseline_median"], g["threshold"], g["mad_flag"] = med, thr, flag
        out.append(g)
    return pd.concat(out).reset_index(drop=True) if out else wk.assign(baseline_median=np.nan, threshold=np.nan, mad_flag=False)


def isolation_flags(wk: pd.DataFrame, contamination: float = 0.10, baseline_weeks: int = 12, seed: int = 42) -> pd.DataFrame:
    """IsolationForest on [breakdown min, count, reject rate, speed ratio].

    Each feature is scaled with the median / IQR of the line's first ``baseline_weeks`` weeks (its
    'early baseline'), so a line is judged against its own normal. Note: the forest itself is fitted on
    the whole period (unsupervised, offline), unlike the MAD alarm which only looks backwards.
    """
    feats = ["breakdown_min", "breakdown_n", "reject_rate", "speed_ratio"]
    w = wk.sort_values(["line_id", "week"]).reset_index(drop=True).copy()
    Z = pd.DataFrame(0.0, index=w.index, columns=feats)
    for _, g in w.groupby("line_id"):
        b = g.head(baseline_weeks)[feats]
        med = b.median()
        iqr = (b.quantile(0.75) - b.quantile(0.25)).replace(0, np.nan)
        Z.loc[g.index] = ((g[feats] - med) / iqr).fillna(0.0)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    pred = model.fit_predict(Z)
    w["iforest_score"] = -model.score_samples(Z)
    w["iforest_flag"] = pred == -1
    return w


def lead_time_weeks(wk: pd.DataFrame, line: str, flag_col: str = "mad_flag", value: str = "breakdown_min", roll: int = 4) -> Dict[str, object]:
    """How many weeks BEFORE the worst period did the alarm first fire?

    Worst period = the week with the highest rolling-``roll``-week breakdown minutes (window end).
    The alarm episode is walked backwards from there, tolerating gaps of up to 2 unflagged weeks.
    """
    g = wk[wk["line_id"] == line].sort_values("week").reset_index(drop=True)
    if len(g) == 0:
        return {"line": line, "first_alarm": None, "worst_week": None, "lead_weeks": None}
    g["roll"] = g[value].rolling(roll, min_periods=roll).sum()
    if g["roll"].notna().sum() == 0:
        return {"line": line, "first_alarm": None, "worst_week": None, "lead_weeks": None}
    worst_idx = int(g["roll"].idxmax())
    first, gap = None, 0
    for i in range(worst_idx, -1, -1):
        if g.loc[i, flag_col]:
            first, gap = i, 0
        else:
            gap += 1
            if gap > 2:
                break
    if first is None:
        return {"line": line, "first_alarm": None, "worst_week": g.loc[worst_idx, "week"], "lead_weeks": None}
    return {"line": line, "first_alarm": g.loc[first, "week"], "worst_week": g.loc[worst_idx, "week"], "lead_weeks": int(worst_idx - first)}


# ----------------------------------------------------------------------------------------------
# Risk score watch-list
# ----------------------------------------------------------------------------------------------
def risk_score(shifts: pd.DataFrame, events: pd.DataFrame, as_of: Optional[pd.Timestamp] = None,
               recent_days: int = 28, min_events: int = 2) -> pd.DataFrame:
    """Risk score per line x equipment.

    frequency = breakdowns per 100 operating hours of the line in the last ``recent_days``
    severity  = mean breakdown duration (min) in the same window
    trend     = (recent breakdown minutes - previous window minutes) / (previous + 60), clipped to [-1, 3]
    score     = 100 * (0.4 * pct_rank(frequency) + 0.3 * pct_rank(severity) + 0.3 * pct_rank(trend))
    """
    as_of = as_of if as_of is not None else events["date"].max()
    r0 = as_of - pd.Timedelta(days=recent_days - 1)
    p0 = r0 - pd.Timedelta(days=recent_days)
    bd = events[events["loss_category"] == "Breakdown"]
    rec = bd[(bd["date"] >= r0) & (bd["date"] <= as_of)]
    prev = bd[(bd["date"] >= p0) & (bd["date"] < r0)]
    op_h = shifts[(shifts["date"] >= r0) & (shifts["date"] <= as_of)].groupby("line_id")["operating_time_min"].sum() / 60.0
    base = bd[["line_id", "equipment"]].drop_duplicates()
    r = rec.groupby(["line_id", "equipment"]).agg(n=("event_id", "count"), minutes=("duration_min", "sum"), mean_min=("duration_min", "mean")).reset_index()
    pv = prev.groupby(["line_id", "equipment"]).agg(prev_minutes=("duration_min", "sum")).reset_index()
    t = base.merge(r, on=["line_id", "equipment"], how="left").merge(pv, on=["line_id", "equipment"], how="left")
    t = t.fillna({"n": 0, "minutes": 0, "mean_min": 0, "prev_minutes": 0})
    t["frequency"] = t["n"] / t["line_id"].map(op_h).replace(0, np.nan) * 100
    t["frequency"] = t["frequency"].fillna(0)
    t["severity"] = t["mean_min"]
    t["trend"] = ((t["minutes"] - t["prev_minutes"]) / (t["prev_minutes"] + 60)).clip(-1, 3)
    for c in ["frequency", "severity", "trend"]:
        t[f"{c}_rank"] = t[c].rank(pct=True)
    t["risk_score"] = 100 * (0.4 * t["frequency_rank"] + 0.3 * t["severity_rank"] + 0.3 * t["trend_rank"])
    t = t[t["n"] >= min_events].sort_values("risk_score", ascending=False).reset_index(drop=True)
    t["as_of"] = as_of
    return t[["line_id", "equipment", "n", "minutes", "mean_min", "frequency", "trend", "risk_score", "as_of"]]


# ----------------------------------------------------------------------------------------------
# Predictive model (modest): >=1 breakdown >= 60 min in the next shift
# ----------------------------------------------------------------------------------------------
def build_model_frame(shifts: pd.DataFrame, n_hist: int = 6, long_break_min: int = 60) -> pd.DataFrame:
    """Features use ONLY shifts before t (rolling window shifted by one); target is shift t."""
    d = shifts.sort_values(["line_id", "date", "shift"]).reset_index(drop=True).copy()
    d["perf"] = d["total_units"] / (d["operating_time_min"] * d["ideal_rate_units_per_min"])
    d["rej"] = d["reject_units"] / d["total_units"]
    d["avail"] = d["operating_time_min"] / d["planned_production_time_min"]
    d["long_bd"] = (d["max_breakdown_min"] >= 120).astype(int)
    d["target"] = (d["max_breakdown_min"] >= long_break_min).astype(int)
    g = d.groupby("line_id")
    for src, name, fn in [("min_breakdown", "h_bd_min", "sum"), ("n_breakdowns", "h_bd_n", "sum"), ("max_breakdown_min", "h_bd_max", "max"),
                          ("rej", "h_rej", "mean"), ("perf", "h_perf", "mean"), ("avail", "h_avail", "mean")]:
        d[name] = g[src].transform(lambda s: s.shift(1).rolling(n_hist, min_periods=n_hist).agg(fn))
    # shifts since the last long repair (>=120 min), counted in run shifts, capped
    since = []
    for _, grp in d.groupby("line_id", sort=False):
        c, vals = 60, []
        for lb in grp["long_bd"].to_numpy():
            vals.append(c)
            c = 0 if lb else min(c + 1, 60)
        since.extend(vals)
    d["h_since_long"] = since
    d["is_night"] = (d["shift"] == 3).astype(int)
    d["prev_idle_i"] = d["prev_idle"].astype(int)
    return d.dropna(subset=["h_bd_min"]).reset_index(drop=True)


FEATURES = ["h_bd_min", "h_bd_n", "h_bd_max", "h_rej", "h_perf", "h_avail", "h_since_long", "is_night", "prev_idle_i"]


def breakdown_model(shifts: pd.DataFrame, n_hist: int = 6, threshold: float = 0.5, train_frac: float = 0.7, gap_days: int = 2, seed: int = 42):
    """Logistic regression with a strict chronological split (train earlier, test later, small gap).

    Returns (metrics dict, test frame with predicted probability, coefficient table).
    """
    d = build_model_frame(shifts, n_hist=n_hist)
    dates = np.sort(d["date"].unique())
    cut = dates[int(len(dates) * train_frac)]
    train = d[d["date"] < cut]
    test = d[d["date"] >= cut + np.timedelta64(gap_days, "D")]
    line_dum = lambda x: pd.get_dummies(x["line_id"]).reindex(columns=sorted(d["line_id"].unique()), fill_value=0).astype(float)
    Xtr = pd.concat([train[FEATURES].reset_index(drop=True), line_dum(train).reset_index(drop=True)], axis=1)
    Xte = pd.concat([test[FEATURES].reset_index(drop=True), line_dum(test).reset_index(drop=True)], axis=1)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(sc.transform(Xtr), train["target"])
    proba = clf.predict_proba(sc.transform(Xte))[:, 1]
    ytest = test["target"].to_numpy()
    pred = (proba >= threshold).astype(int)
    metrics = {
        "train_end": pd.Timestamp(cut) - pd.Timedelta(days=1), "test_start": test["date"].min(), "test_end": test["date"].max(),
        "n_train": len(train), "n_test": len(test),
        "base_rate_train": float(train["target"].mean()), "base_rate_test": float(ytest.mean()),
        "precision": float(precision_score(ytest, pred, zero_division=0)), "recall": float(recall_score(ytest, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(ytest, proba)), "pr_auc": float(average_precision_score(ytest, proba)),
        "alarm_rate": float(pred.mean()), "threshold": threshold,
    }
    metrics["lift_vs_base"] = metrics["precision"] / metrics["base_rate_test"] if metrics["base_rate_test"] else np.nan
    coef = pd.DataFrame({"feature": list(Xtr.columns), "std_coef": clf.coef_[0]}).sort_values("std_coef", key=np.abs, ascending=False)
    out = test[["shift_id", "date", "line_id", "target"]].reset_index(drop=True).assign(proba=proba, pred=pred)
    return metrics, out, coef.reset_index(drop=True)

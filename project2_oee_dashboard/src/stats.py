"""Small, dependency-light statistics helpers with plain-language outputs."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sst


def ols_hc3(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and HC3 heteroscedasticity-robust standard errors."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = X.T @ (X * ((resid / (1 - h)) ** 2)[:, None])
    cov = XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(cov))


def shift_effect_regression(shifts: pd.DataFrame, outcome: str = "oee", controls: Optional[List[str]] = None) -> pd.DataFrame:
    """OEE ~ line fixed effects + shift (S1 is the reference). Robust (HC3) standard errors.

    The shift coefficients are therefore the S2 / S3 difference vs S1 *within the same line*.
    ``controls`` (e.g. ['avg_crew_experience_years']) adds numeric covariates.
    """
    controls = controls or []
    d = shifts.dropna(subset=[outcome] + controls).copy()
    line_dummies = pd.get_dummies(d["line_id"], drop_first=False).astype(float)
    parts = [line_dummies, (d["shift_label"] == "S2").astype(float).rename("S2"), (d["shift_label"] == "S3").astype(float).rename("S3")]
    parts += [d[c].astype(float) for c in controls]
    X = pd.concat(parts, axis=1)
    beta, se = ols_hc3(X.to_numpy(), d[outcome].to_numpy())
    dof = len(d) - X.shape[1]
    out = []
    for name in ["S2", "S3"] + controls:
        i = list(X.columns).index(name)
        t = beta[i] / se[i]
        p = 2 * (1 - sst.t.cdf(abs(t), dof))
        label = f"{name} vs S1 (same line)" if name in ("S2", "S3") else f"{name} (per unit)"
        out.append({"term": label, "diff": beta[i], "ci_low": beta[i] - 1.96 * se[i], "ci_high": beta[i] + 1.96 * se[i], "p_value": p})
    return pd.DataFrame(out)


def bootstrap_ci(values_a: np.ndarray, values_b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Bootstrap CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    return {"diff": float(a.mean() - b.mean()), "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5))}


def welch_t(values_a, values_b) -> Dict[str, float]:
    t, p = sst.ttest_ind(values_a, values_b, equal_var=False, nan_policy="omit")
    return {"t": float(t), "p_value": float(p)}


def weekly_block_bootstrap_oee_diff(before: pd.DataFrame, after: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """CI for weighted-OEE(after) - weighted-OEE(before), resampling whole WEEKS.

    Resampling weeks (not single shifts) respects the fact that neighbouring shifts of a line are
    not independent. Each resample re-computes the weighted OEE from summed numerators/denominators.
    """
    def weekly_sums(d: pd.DataFrame) -> np.ndarray:
        dd = d.assign(ideal_output=d["operating_time_min"] * d["ideal_rate_units_per_min"])
        g = dd.groupby("week")[["planned_production_time_min", "operating_time_min", "ideal_output", "total_units", "good_units"]].sum()
        return g.to_numpy()

    def oee_of(s: np.ndarray) -> float:
        planned, op, ideal, total, good = s.sum(axis=0)
        return (op / planned) * (total / ideal) * (good / total)

    wb, wa = weekly_sums(before), weekly_sums(after)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sb = wb[rng.integers(0, len(wb), len(wb))]
        sa = wa[rng.integers(0, len(wa), len(wa))]
        diffs[i] = oee_of(sa) - oee_of(sb)
    return {"diff": float(oee_of(wa) - oee_of(wb)), "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)), "weeks_before": len(wb), "weeks_after": len(wa)}


def two_proportion_z(x1: float, n1: float, x2: float, n2: float) -> Dict[str, float]:
    """z-test for the difference of two rates x/n (e.g. reject rates). Plain, no continuity correction."""
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else np.nan
    return {"p1": p1, "p2": p2, "diff": p1 - p2, "z": float(z), "p_value": float(2 * (1 - sst.norm.cdf(abs(z)))) if se > 0 else np.nan}


def poisson_rate_ratio(events1: float, exposure1: float, events2: float, exposure2: float) -> Dict[str, float]:
    """Rate ratio (group 1 / group 2) with an approximate 95% CI (log scale) for event counts per exposure."""
    r1, r2 = events1 / exposure1, events2 / exposure2
    rr = r1 / r2
    se = np.sqrt(1 / max(events1, 1) + 1 / max(events2, 1))
    return {"rate1": r1, "rate2": r2, "ratio": rr, "ci_low": float(rr * np.exp(-1.96 * se)), "ci_high": float(rr * np.exp(1.96 * se))}

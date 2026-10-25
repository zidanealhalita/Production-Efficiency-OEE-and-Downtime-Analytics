"""Page 5 - Maintenance & early warning: what is failing, can we see it coming, and what do we watch next?"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.anomaly import breakdown_model, isolation_flags, lead_time_weeks, line_week_table, mad_flags, risk_score
from src.config import LINE_COLORS
from src.impact import maintenance_costs
from src.ui import fmt_idr, get_ctx
from src.viz import style

ctx = get_ctx()
A = ctx.assump
st.title("Maintenance & Early Warning")
st.caption("Decision question: where does maintenance effort go, can we spot a failing line early, and which equipment should we watch next?")

# --- Preventive vs corrective --------------------------------------------------------------------
st.subheader("Preventive vs corrective work")
M = maintenance_costs(ctx.maint, A["labour_rate_idr_per_h"])
if ctx.filters_active:
    st.caption("Work orders are filtered by plant, line and date only (shift / SKU filters do not apply to work orders).")
if len(M) == 0:
    st.info("No work orders in the current selection.")
else:
    g = M.groupby(["line_id", "wo_type"]).agg(wos=("wo_id", "count"), downtime_h=("downtime_min", lambda x: x.sum() / 60), cost=("total_cost_idr", "sum")).reset_index()
    c1, c2 = st.columns(2)
    f = px.bar(g, x="line_id", y="wos", color="wo_type", barmode="stack", color_discrete_map={"Corrective": "#D55E00", "Preventive": "#009E73"})
    corr_share = (M["wo_type"] == "Corrective").mean()
    c1.plotly_chart(style(f, f"{corr_share:.0%} of work orders are corrective (reactive)", "Line", "Work orders (count)", height=360), use_container_width=True)
    f = px.bar(g, x="line_id", y="cost", color="wo_type", barmode="stack", color_discrete_map={"Corrective": "#D55E00", "Preventive": "#009E73"})
    c2.plotly_chart(style(f, f"Corrective work costs {fmt_idr(M[M['wo_type'] == 'Corrective']['total_cost_idr'].sum())} (spare parts + labour at the assumed rate)", "Line", "Cost (IDR)", height=360), use_container_width=True)
    t = M.groupby("line_id").agg(work_orders=("wo_id", "count"), corrective=("wo_type", lambda x: (x == "Corrective").sum()), preventive=("wo_type", lambda x: (x == "Preventive").sum()),
                                 downtime_h=("downtime_min", lambda x: x.sum() / 60), spare_parts_idr=("spare_parts_cost_idr", "sum"), labour_idr=("labour_cost_idr", "sum")).reset_index()
    t["downtime_min_per_wo"] = t["downtime_h"] * 60 / t["work_orders"]
    st.dataframe(t, hide_index=True, use_container_width=True, column_config={
        "downtime_h": st.column_config.NumberColumn("Downtime (h)", format="%.0f"), "downtime_min_per_wo": st.column_config.NumberColumn("Downtime per WO (min)", format="%.0f"),
        "spare_parts_idr": st.column_config.NumberColumn("Spare parts (IDR)", format="%d"), "labour_idr": st.column_config.NumberColumn("Labour (IDR)", format="%d")})
    st.caption("Labour cost = labour hours x the hourly rate in the sidebar assumptions. Preventive work orders are few and tend to be long planned jobs, so the corrective/preventive imbalance is itself a talking point.")

# --- Anomaly detection ---------------------------------------------------------------------------
st.subheader("Early-warning alarm on weekly breakdown minutes")
S_all, E_all = ctx.all_shifts, ctx.all_events
sel_lines = ctx.sel["lines"]
c1, c2, c3 = st.columns(3)
k = c1.slider("Alarm sensitivity k (median + k x MAD)", 2.0, 5.0, 3.0, 0.5)
win = c2.slider("Baseline window (weeks)", 6, 20, 12, 1)
cont = c3.slider("IsolationForest contamination", 0.02, 0.20, 0.10, 0.01)

@st.cache_data(show_spinner="Computing weekly alarms...")
def alarms(k, win, cont):
    wk = line_week_table(S_all, E_all)
    wk = mad_flags(wk, window=win, k=k)
    return isolation_flags(wk, contamination=cont)

wk = alarms(k, win, cont)
wk = wk[wk["line_id"].isin(sel_lines)]
if len(wk) == 0:
    st.info("No complete weeks for the selected lines.")
else:
    lines_show = sorted(wk["line_id"].unique())
    fig = make_subplots(rows=len(lines_show), cols=1, shared_xaxes=True, subplot_titles=[f"Line {l}" for l in lines_show], vertical_spacing=0.04)
    for i, l in enumerate(lines_show, start=1):
        g = wk[wk["line_id"] == l]
        fig.add_trace(go.Scatter(x=g["week"], y=g["breakdown_min"], mode="lines", line=dict(color=LINE_COLORS[l], width=2), name=f"{l} breakdown min", showlegend=False), row=i, col=1)
        fig.add_trace(go.Scatter(x=g["week"], y=g["threshold"], mode="lines", line=dict(color="#111827", dash="dot", width=1), name="Threshold", showlegend=(i == 1)), row=i, col=1)
        fl = g[g["mad_flag"]]
        fig.add_trace(go.Scatter(x=fl["week"], y=fl["breakdown_min"], mode="markers", marker=dict(size=8, color="#D55E00", symbol="diamond", line=dict(color="black", width=1)), name="MAD alarm", showlegend=(i == 1)), row=i, col=1)
        fi = g[g["iforest_flag"]]
        fig.add_trace(go.Scatter(x=fi["week"], y=fi["breakdown_min"], mode="markers", marker=dict(size=11, color="rgba(0,0,0,0)", symbol="circle-open", line=dict(color="#009E73", width=2)), name="IsolationForest", showlegend=(i == 1)), row=i, col=1)
    fig.update_layout(height=190 * len(lines_show) + 120, template="plotly_white", title=dict(text="Weekly breakdown minutes per line with alarms (diamond = robust threshold, green ring = IsolationForest)", x=0), margin=dict(l=10, r=10, t=90, b=10),
                      legend=dict(orientation="h", y=1.04, x=1, xanchor="right"))
    fig.update_yaxes(title_text="min / week")
    st.plotly_chart(fig, use_container_width=True)

    lt = pd.DataFrame([{**lead_time_weeks(wk, l, "mad_flag"), "method": "Robust threshold"} for l in lines_show] + [{**lead_time_weeks(wk, l, "iforest_flag"), "method": "IsolationForest"} for l in lines_show])
    lt = lt.rename(columns={"line": "line_id"})
    tot_bd = wk.groupby("line_id")["breakdown_min"].sum()
    focus = tot_bd.idxmax()
    r = lt[(lt["line_id"] == focus) & (lt["method"] == "Robust threshold")].iloc[0]
    if pd.notna(r["lead_weeks"]):
        st.success(f"**{focus}** (most breakdown minutes): the robust-threshold alarm first fired in the week of **{pd.Timestamp(r['first_alarm']):%d %b %Y}**, "
                   f"**{int(r['lead_weeks'])} weeks before** its worst 4-week window (ending {pd.Timestamp(r['worst_week']):%d %b %Y}).")
    else:
        st.warning(f"No alarm preceded the worst window on {focus} with the current settings.")
    st.dataframe(lt[["line_id", "method", "first_alarm", "worst_week", "lead_weeks"]], hide_index=True, use_container_width=True, column_config={
        "first_alarm": st.column_config.DateColumn("First alarm (week)", format="YYYY-MM-DD"), "worst_week": st.column_config.DateColumn("Worst 4-week window ends", format="YYYY-MM-DD")})
    flagged = wk.groupby("line_id")[["mad_flag", "iforest_flag"]].sum().reset_index().rename(columns={"mad_flag": "robust_alarm_weeks", "iforest_flag": "iforest_alarm_weeks"})
    st.caption("How it works: each line is compared with the median of its own last N non-alarm weeks (so a long problem cannot become the 'new normal'); "
               "threshold = median + k x 1.4826 x MAD, with a 5% floor. Only complete weeks are used. The IsolationForest is unsupervised and fitted offline on the full period, so it is a cross-check, not a live alarm. "
               "False alarms (flagged weeks on lines without a real problem) are visible in the table below - each one costs a maintenance planner's attention.")
    st.dataframe(flagged, hide_index=True, use_container_width=True)

# --- Risk score ----------------------------------------------------------------------------------
st.subheader("Equipment watch-list (risk score)")
lo, hi = S_all["date"].min().date(), S_all["date"].max().date()
as_of = st.date_input("Score as of", value=ctx.end.date() if lo <= ctx.end.date() <= hi else hi, min_value=lo, max_value=hi)
rs = risk_score(S_all[S_all["date"] <= pd.Timestamp(as_of)], E_all[E_all["date"] <= pd.Timestamp(as_of)], as_of=pd.Timestamp(as_of))
rs = rs[rs["line_id"].isin(sel_lines)].head(15)
if len(rs):
    st.dataframe(rs[["line_id", "equipment", "n", "minutes", "mean_min", "frequency", "trend", "risk_score"]], hide_index=True, use_container_width=True, column_config={
        "n": st.column_config.NumberColumn("Breakdowns (28 d)"), "minutes": st.column_config.NumberColumn("Minutes (28 d)", format="%d"), "mean_min": st.column_config.NumberColumn("Mean duration (min)", format="%.0f"),
        "frequency": st.column_config.NumberColumn("Breakdowns / 100 op. h", format="%.1f"), "trend": st.column_config.NumberColumn("Trend vs previous 28 d", format="%.2f"),
        "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.0f")})
with st.expander("How the risk score is calculated"):
    st.markdown("""
For each line x equipment, over the **last 28 days** up to the chosen date:

- **Frequency** = breakdowns per 100 operating hours of the line.
- **Severity** = mean breakdown duration (minutes).
- **Trend** = (minutes in last 28 days - minutes in the 28 days before) / (previous minutes + 60), capped at -1 .. +3.

Each component is converted to a percentile rank across all equipment, then
`risk score = 100 x (0.4 x frequency rank + 0.3 x severity rank + 0.3 x trend rank)`.
The weights are a judgement call (frequency matters most, rising trend and severity next), not a fitted model. Equipment with fewer than 2 breakdowns in the window is excluded.
Use it as a **watch-list to prioritise inspections**, not as a failure prediction.
""")

# --- Optional model ------------------------------------------------------------------------------
st.subheader("Optional: predict a long breakdown in the next shift")
with st.expander("Logistic regression with a strict time split (modest, honest)"):
    st.markdown("Target: **at least one breakdown of 60+ min in the next shift** of a line. Features come only from the previous N shifts of that line (rolling window shifted by one shift, so no look-ahead) "
                "plus facts known at shift start (night shift, started after idle). Training = earliest 70% of dates, test = later dates after a 2-day gap.")
    thr = st.slider("Alarm threshold (probability)", 0.30, 0.80, 0.50, 0.05)
    if st.button("Train / evaluate model"):
        with st.spinner("Training..."):
            metrics, preds, coef = breakdown_model(S_all, threshold=thr)
        m = st.columns(5)
        m[0].metric("Base rate (test)", f"{metrics['base_rate_test']:.1%}")
        m[1].metric("Precision", f"{metrics['precision']:.1%}")
        m[2].metric("Recall", f"{metrics['recall']:.1%}")
        m[3].metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")
        m[4].metric("Alarms raised", f"{metrics['alarm_rate']:.0%} of shifts")
        st.write(f"Train until {metrics['train_end']:%d %b %Y} ({metrics['n_train']:,} shifts); test {metrics['test_start']:%d %b} - {metrics['test_end']:%d %b %Y} ({metrics['n_test']:,} shifts). "
                 f"Precision is {metrics['lift_vs_base']:.2f}x the base rate; PR-AUC {metrics['pr_auc']:.2f}.")
        st.dataframe(coef, hide_index=True, use_container_width=True)
        st.warning("Limits: a single shift is a hard thing to predict - most long breakdowns look random from the recent past, so precision is close to the base rate and an alarm on "
                   f"{metrics['alarm_rate']:.0%} of shifts is not actionable. The test period also follows the Filler fix, so conditions differ from training. "
                   "The weekly alarm and the risk-score watch-list are the more useful tools here; a real MES with sensor data (vibration, temperature) would be needed for genuine prediction. "
                   "A false alarm costs a technician inspection (about 0.5-1 h); a missed long breakdown costs an hour or more of lost output.")

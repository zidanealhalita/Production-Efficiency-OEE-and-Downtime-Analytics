"""Page 4 - Quality: which defects and conditions drive rejects, and which shifts are out of control?"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import LINE_COLORS, SHIFT_COLORS
from src.impact import copq_by
from src.oee import oee_by
from src.pareto import pareto_table, vital_few
from src.spc import flagged_table, p_chart
from src.stats import two_proportion_z
from src.ui import fmt_idr, fmt_pct, get_ctx
from src.viz import bar_group, pareto_chart, style

ctx = get_ctx()
S, Q, A = ctx.shifts, ctx.quality, ctx.assump
st.title("Quality")
st.caption("Decision question: which defects and operating conditions drive rejects, and where is the process out of control?")

# --- Trend ---------------------------------------------------------------------------------------
w = oee_by(S, ["line_id", "week"]).sort_values(["line_id", "week"])
w["reject_rate"] = w["reject_units"] / w["total_units"]
worst = oee_by(S, "line_id").assign(rr=lambda d: d["reject_units"] / d["total_units"]).sort_values("rr").iloc[-1]
f = px.line(w, x="week", y="reject_rate", color="line_id", color_discrete_map=LINE_COLORS)
f.update_traces(line=dict(width=1.6))
f.update_yaxes(tickformat=".1%")
st.plotly_chart(style(f, f"{worst['line_id']} has the highest reject rate ({worst['rr']:.2%}); weekly rates are otherwise stable", "Week", "Reject rate (% of units produced)", height=400), use_container_width=True)

# --- Pareto and mix ------------------------------------------------------------------------------
c1, c2 = st.columns(2)
if len(Q):
    pt = pareto_table(Q, "defect_type", value="defect_units")
    top = pt.iloc[0]
    with c1:
        st.plotly_chart(pareto_chart(pt, "defect_type", f"{top['defect_type']} is the top defect ({top['share']:.0%}); {vital_few(pt)} types reach 80%", "Defect type", "Defect units", unit="units", height=420), use_container_width=True)
    mix = Q.groupby(["line_id", "defect_type"])["defect_units"].sum().reset_index()
    f = px.bar(mix, x="line_id", y="defect_units", color="defect_type", barnorm="percent", color_discrete_sequence=px.colors.qualitative.Safe + px.colors.qualitative.Set2)
    with c2:
        st.plotly_chart(style(f, "Defect mix differs by line type: each line has its own failure modes", "Line", "Share of defect units (%)", height=420), use_container_width=True)

# --- Shift / changeover / idle -------------------------------------------------------------------
st.subheader("Reject rate by condition")
def rr_table(df, key, label):
    t = df.groupby(key).agg(rej=("reject_units", "sum"), tot=("total_units", "sum"), n=("shift_id", "count")).reset_index()
    t["reject_rate"] = t["rej"] / t["tot"]
    t["condition"] = t[key].map(label) if isinstance(label, dict) else t[key]
    return t

c1, c2, c3 = st.columns(3)
t_sh = rr_table(S, "shift_label", None)
f = px.bar(t_sh, x="shift_label", y="reject_rate", color="shift_label", color_discrete_map=SHIFT_COLORS, text=t_sh["reject_rate"].map(lambda v: f"{v:.2%}"))
f.update_yaxes(tickformat=".1%")
c1.plotly_chart(style(f, "By shift", "Shift", "Reject rate (%)", height=340, legend=False), use_container_width=True)

co_lines = S.groupby("line_id")["sku_change"].sum()
co_lines = co_lines[co_lines > 0].index
Sc = S[S["line_id"].isin(co_lines)]
t_co = rr_table(Sc, "sku_change", {True: "Changeover shift", False: "Normal shift"})
col_map = {"Changeover shift": "#D55E00", "Normal shift": "#0072B2"}
if t_co["sku_change"].nunique() == 2:
    f = px.bar(t_co, x="condition", y="reject_rate", color="condition", color_discrete_map=col_map, text=t_co["reject_rate"].map(lambda v: f"{v:.2%}"))
    f.update_yaxes(tickformat=".1%")
    c2.plotly_chart(style(f, "After changeover vs normal", "", "Reject rate (%)", height=340, legend=False), use_container_width=True)
    a_, b_ = t_co[t_co["sku_change"]].iloc[0], t_co[~t_co["sku_change"]].iloc[0]
    zc = two_proportion_z(a_["rej"], a_["tot"], b_["rej"], b_["tot"])
else:
    c2.info("Needs both changeover and normal shifts (multi-SKU lines).")
    zc = None
t_id = rr_table(S, "prev_idle", {True: "After idle shift", False: "Line was running"})
if t_id["prev_idle"].nunique() == 2:
    f = px.bar(t_id, x="condition", y="reject_rate", color="condition", color_discrete_map={"After idle shift": "#D55E00", "Line was running": "#0072B2"}, text=t_id["reject_rate"].map(lambda v: f"{v:.2%}"))
    f.update_yaxes(tickformat=".1%")
    c3.plotly_chart(style(f, "After idle vs running", "", "Reject rate (%)", height=340, legend=False), use_container_width=True)
if zc:
    st.caption(f"Changeover shifts (multi-SKU lines only) reject {zc['diff'] * 100:+.2f} pts vs normal shifts (z-test p = {zc['p_value']:.3g}). "
               "With tens of thousands of units per shift, tiny gaps are 'significant' - judge the size of the gap. Not adjusted for line mix.")

# --- SPC -----------------------------------------------------------------------------------------
st.subheader("SPC: p-chart of reject rate with 3-sigma limits")
c1, c2 = st.columns([1, 2])
line = c1.selectbox("Line", sorted(S["line_id"].unique()))
laney = c2.checkbox("Use Laney p' (adjusts for overdispersion) - recommended for high-volume data", value=True)
ch = p_chart(S[S["line_id"] == line], laney=laney)
if len(ch):
    ch["x"] = ch["date"] + pd.to_timedelta((ch["shift"] - 1) * 8, unit="h")
    fl = ch[ch["flagged"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ch["x"], y=ch["p"], mode="lines+markers", name="Reject rate", line=dict(color="#9CA3AF", width=1), marker=dict(size=4, color="#0072B2")))
    fig.add_trace(go.Scatter(x=ch["x"], y=ch["ucl"], mode="lines", name="UCL (3 sigma)", line=dict(color="#D55E00", dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=ch["x"], y=ch["lcl"], mode="lines", name="LCL (3 sigma)", line=dict(color="#D55E00", dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=ch["x"], y=ch["center"], mode="lines", name="Centre line", line=dict(color="#111827", width=1.5)))
    fig.add_trace(go.Scatter(x=fl["x"], y=fl["p"], mode="markers", name="Flagged", marker=dict(size=9, color="#D55E00", symbol="diamond", line=dict(color="black", width=1))))
    fig.update_yaxes(tickformat=".1%")
    kind = "Laney p'" if laney else "classic p-chart"
    st.plotly_chart(style(fig, f"{line} ({kind}): {len(fl)} of {len(ch)} shifts flagged ({len(fl) / len(ch):.1%})", "Shift date", "Reject rate (%)", height=430), use_container_width=True)
    m = st.columns(4)
    m[0].metric("Centre line", f"{ch['center'].iloc[0]:.2%}")
    m[1].metric("Rule 1 (beyond 3 sigma)", int(ch["rule1"].sum()))
    m[2].metric("Rule 2 (8 in a row)", int(ch["rule2"].sum()))
    m[3].metric("Sigma inflation factor", f"{ch['sigma_factor'].iloc[0]:.1f}x")
    if not laney:
        st.warning("A classic p-chart assumes shift-to-shift variation is pure binomial noise. With 50-100k units per shift, normal variation "
                   "(different SKUs, crews, conditions) is far larger, so most shifts fall outside the limits. Switch on Laney p' to see only unusual shifts.")
    else:
        st.caption(f"Laney p' widens the limits by {ch['sigma_factor'].iloc[0]:.1f}x to reflect the real shift-to-shift variation. Rules: (1) point beyond 3 sigma, (2) 8 consecutive points on one side of the centre line.")
    with st.expander("Flagged shifts table", expanded=True):
        st.dataframe(flagged_table(ch), hide_index=True, use_container_width=True, column_config={
            "p": st.column_config.NumberColumn("Reject rate", format="%.4f"), "center": st.column_config.NumberColumn("Centre", format="%.4f"), "ucl": st.column_config.NumberColumn("UCL", format="%.4f"),
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD")})

# --- COPQ ----------------------------------------------------------------------------------------
st.subheader("Cost of poor quality (COPQ)")
st.caption("COPQ = reject units x unit cost. Unit cost is not in the data, so it comes from the editable assumption table in the sidebar (illustrative values).")
cl = copq_by(S, A, "line_id")
cm = copq_by(S, A, ["month_label", "line_id"])
tot = cl["copq_idr"].sum()
c1, c2 = st.columns(2)
c1.metric("COPQ in selection", fmt_idr(tot), help="Reject units x assumed unit cost")
c1.metric("COPQ per 1,000 good units", fmt_idr(tot / S["good_units"].sum() * 1000))
f = px.bar(cl, x="line_id", y="copq_idr", color="line_id", color_discrete_map=LINE_COLORS)
c2.plotly_chart(style(f, f"{cl.sort_values('copq_idr').iloc[-1]['line_id']} carries the highest COPQ ({fmt_idr(cl['copq_idr'].max())})", "Line", "COPQ (IDR, assumption-based)", height=340, legend=False), use_container_width=True)
f = px.bar(cm, x="month_label", y="copq_idr", color="line_id", color_discrete_map=LINE_COLORS, barmode="stack")
st.plotly_chart(style(f, "COPQ by month", "Month", "COPQ (IDR, assumption-based)", height=360), use_container_width=True)

"""Page 7 - Improvement opportunities: what do we fix first, and what is it worth?"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import L02_FIX_DATE, LINE_COLORS
from src.impact import build_opportunities, l02_before_after, minutes_to_value
from src.oee import oee_by
from src.stats import weekly_block_bootstrap_oee_diff, welch_t
from src.ui import fmt_idr, fmt_pct, get_ctx
from src.viz import style

ctx = get_ctx()
A = ctx.assump
S, E = ctx.all_shifts, ctx.all_events
st.title("Improvement Opportunities")
st.caption("Decision question: what do we fix first, what is it worth, and how hard is it? "
           "This page always uses the full 2025 dataset (sidebar filters do not apply), so 'per year' is literal; assumptions in the sidebar do apply.")

opp = build_opportunities(S, E, A)

# --- Headline ------------------------------------------------------------------------------------
tot = opp[["hours", "units", "idr"]].sum()
m = st.columns(4)
m[0].metric("Opportunities", len(opp))
m[1].metric("Hours / year", f"{tot['hours']:,.0f} h", help="Recoverable operating-time equivalent")
m[2].metric("Extra good units / year", f"{tot['units'] / 1e6:,.1f} M")
m[3].metric("Contribution / year", fmt_idr(tot["idr"]), help="Simple sum; the levers overlap partly (e.g. a Filler fix also lifts night-shift OEE), so treat as an upper bound.")
top = opp.iloc[0]
st.info(f"**Start with:** {top['action']} (about {fmt_idr(top['idr'])} per year, effort {top['effort']}/5). "
        "Impact in IDR rests on the contribution-margin assumptions in the sidebar - change them and every number here updates.")

# --- Ranked table --------------------------------------------------------------------------------
st.subheader("Ranked action list")
show = opp[["rank", "action", "lines", "hours", "units", "idr", "effort", "evidence", "basis"]].copy()
st.dataframe(show, hide_index=True, use_container_width=True, column_config={
    "rank": st.column_config.NumberColumn("#", width="small"), "action": st.column_config.TextColumn("Action", width="large"),
    "hours": st.column_config.NumberColumn("Hours / yr", format="%.0f"), "units": st.column_config.NumberColumn("Extra good units / yr", format="%.0f"),
    "idr": st.column_config.NumberColumn("IDR / yr", format="%.0f"), "effort": st.column_config.NumberColumn("Effort (1-5)"),
    "evidence": st.column_config.TextColumn("Evidence (page)"), "basis": st.column_config.TextColumn("How it is estimated", width="large")})
st.download_button("Download table (CSV)", opp.to_csv(index=False).encode("utf-8"), "improvement_opportunities.csv", "text/csv")

f = px.bar(opp.sort_values("idr"), x="idr", y="action", orientation="h", color_discrete_sequence=["#0072B2"])
st.plotly_chart(style(f, f"{top['action']} is the biggest lever; the top 3 carry {opp['idr'].head(3).sum() / opp['idr'].sum():.0%} of the total", "Estimated annual contribution (IDR; assumption-based)", "", height=420, legend=False), use_container_width=True)

# --- Effort / impact quadrant --------------------------------------------------------------------
st.subheader("Effort vs impact")
q = opp.copy()
rng = np.random.default_rng(1)
q["effort_j"] = q["effort"] + rng.uniform(-0.12, 0.12, len(q))
q["idr_bn"] = q["idr"] / 1e9
x_mid, y_mid = 3.0, q["idr_bn"].median()
fig = px.scatter(q, x="effort_j", y="idr_bn", size="hours", text="id", color_discrete_sequence=["#0072B2"], size_max=45, hover_name="action", hover_data={"effort_j": False, "effort": True, "idr_bn": ":.2f", "hours": ":.0f"})
fig.update_traces(textposition="top center")
fig.add_vline(x=x_mid, line_dash="dot", line_color="#6B7280")
fig.add_hline(y=y_mid, line_dash="dot", line_color="#6B7280")
xmax, ymax = 5.6, q["idr_bn"].max() * 1.25
for txt, x, y in [("Quick wins", 1.2, ymax * 0.97), ("Big bets", 4.4, ymax * 0.97), ("Fill-ins", 1.2, 0.02), ("Reconsider", 4.4, 0.02)]:
    fig.add_annotation(x=x, y=y, text=f"<b>{txt}</b>", showarrow=False, font=dict(color="#6B7280"))
fig.update_xaxes(range=[0.5, xmax], dtick=1, title_text="Effort (1 = easy, 5 = hard; qualitative assumption, editable)")
fig.update_yaxes(range=[0, ymax])
st.plotly_chart(style(fig, "Quick wins sit top-left: high value, low effort (bubble size = hours recovered)", None, "Annual contribution (IDR bn)", height=470, legend=False), use_container_width=True)
st.caption("Effort scores are my qualitative judgement (capex, lead time, change management) and are editable in the sidebar. They are not engineering estimates.")

st.markdown("**Suggested actions**")
acts = {
    "l02_filler": "Keep the Filler overhaul standard: PM routine, spare-part kit and an early-warning alarm so a repeat is caught in weeks, not months.",
    "smed_l05": "SMED workshop on the UHT line: externalise set-up tasks (pre-stage parts, pre-sterilise), standardise the steps, track every changeover against a target.",
    "night_shift": "Staffing and skills plan for S3: balance experienced operators across shifts, cross-train, fix waiting-operator gaps and night-time support cover.",
    "l06_material": "Supplier review for powder inputs: safety stock for peak months, supplier scorecard, alternative source before Jul-Sep.",
    "startup_idle": "Start-up checklist and warm-up run after any idle shift (Sundays, holiday shutdowns) before releasing product.",
    "rainy_utilities": "Protect utilities before the rainy season: power-dip ride-through (UPS / soft starters), compressed-air buffer and drain checks.",
    "minor_stoppage": "Focus kaizen on the top-5 minor-stoppage reasons (page 2 drill-down): sensors, guides, jam points.",
}
for _, r in opp.iterrows():
    st.markdown(f"- **{r['action']}** - {acts[r['id']]}")

# --- Before / after L02 --------------------------------------------------------------------------
st.subheader("Before / after: L02 Filler fix (Apr-Sep vs Oct-Dec 2025)")
ba = l02_before_after(S, E)
b, a = ba["before"], ba["after"]
m = st.columns(4)
for col, k_, lab in zip(m, ["oee", "availability", "performance", "quality"], ["OEE", "Availability", "Performance", "Quality"]):
    col.metric(lab, fmt_pct(a[k_]), f"{(a[k_] - b[k_]) * 100:+.1f} pts vs Apr-Sep")
st.write(f"L02 OEE moved from **{fmt_pct(b['oee'])}** (Apr-Sep) to **{fmt_pct(a['oee'])}** (Oct-Dec). "
         f"Filler breakdown time fell from **{ba['filler_h_per_month_before']:,.0f} h/month** to **{ba['filler_h_per_month_after']:,.0f} h/month**.")

wb = oee_by(ba["before_df"], "week")
wa = oee_by(ba["after_df"], "week")
bs = weekly_block_bootstrap_oee_diff(ba["before_df"], ba["after_df"])
tt = welch_t(wa["oee"].to_numpy(), wb["oee"].to_numpy())
st.markdown(f"**Statistical check:** OEE difference = **{bs['diff'] * 100:+.1f} pts**, 95% bootstrap CI {bs['ci_low'] * 100:.1f} to {bs['ci_high'] * 100:.1f} pts (resampling whole weeks: {bs['weeks_before']} before, {bs['weeks_after']} after). "
            f"Welch t-test on weekly OEE: p = {tt['p_value']:.2g}. Plain words: the recovery is far larger than week-to-week noise, but this is a before/after comparison without a control line, so other changes in the same period (volume, season) could contribute. Synthetic data.")

w_all = oee_by(S[S["line_id"] == "L02"], "week").sort_values("week")
w_all["roll"] = w_all["oee"].rolling(4, min_periods=1).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=w_all["week"], y=w_all["oee"], mode="lines", name="Weekly OEE", line=dict(color="#D55E00", width=1.2), opacity=0.5))
fig.add_trace(go.Scatter(x=w_all["week"], y=w_all["roll"], mode="lines", name="4-week average", line=dict(color="#D55E00", width=3)))
fx = pd.Timestamp(L02_FIX_DATE)
fig.add_shape(type="line", x0=fx, x1=fx, y0=0, y1=1, yref="paper", line=dict(color="#111827", dash="dash"))
fig.add_annotation(x=fx, y=1, yref="paper", text="Filler fix (1 Oct)", showarrow=False, xanchor="left", yanchor="bottom")
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(style(fig, f"L02 OEE steps up by {bs['diff'] * 100:.0f} pts after the Filler fix", "Week", "OEE (%)", height=400), use_container_width=True)

dm = max(ba["filler_h_per_month_before"] - ba["filler_h_per_month_after"], 0) * 60
v = minutes_to_value({"L02": dm}, S, A).iloc[0]
c = st.columns(3)
c[0].metric("Filler hours recovered / month", f"{dm / 60:,.0f} h")
c[1].metric("Extra good units / month", f"{v['units'] / 1e3:,.0f} k")
c[2].metric("Contribution / month", fmt_idr(v["idr"]))
st.caption("Monthly value = recovered Filler breakdown minutes x L02's realised good units per operating minute x its assumed contribution margin.")

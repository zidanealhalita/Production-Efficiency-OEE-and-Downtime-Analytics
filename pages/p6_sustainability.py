"""Page 6 - Sustainability: how much energy, water and waste does each good unit cost, and how do OEE losses add to it?"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import LINE_COLORS, LOSS_CATEGORIES, SHIFT_COLORS
from src.impact import LEVERS, idle_energy_share_estimate, resource_intensity, resource_scenario
from src.oee import oee_by
from src.ui import get_ctx
from src.viz import style

ctx = get_ctx()
S, A = ctx.shifts, ctx.assump
st.title("Sustainability - Resource Efficiency")
st.caption("Decision question: what do our losses cost in energy, water and waste, and what would we save by fixing the biggest one?")
st.info(f"CO2e uses an **assumed** grid factor of {A['grid_factor_kg_per_kwh']:.2f} kg CO2e per kWh (editable in the sidebar; the dataset's own CO2e column uses 0.85). "
        "Litre-equivalent assumes 1 kg of liquid product ~ 1 litre and excludes the powder line.")

METRICS = {
    "Energy (kWh per 1,000 good units)": "kwh_per_1000",
    "Water (L per good unit)": "water_l_per_unit",
    "Water (L per litre of product)": "water_l_per_litre",
    "Waste (kg per 1,000 good units)": "waste_kg_per_1000",
    "CO2e (kg per 1,000 good units)": "co2e_kg_per_1000",
}
c1, c2 = st.columns(2)
mname = c1.selectbox("Metric", list(METRICS))
dim = c2.radio("By", ["Line", "Shift", "Month"], horizontal=True)
mcol = METRICS[mname]
key = {"Line": "line_id", "Shift": "shift_label", "Month": "month_label"}[dim]
ri = resource_intensity(S, key, A)
ri = ri.dropna(subset=[mcol])
if len(ri) == 0:
    st.info("This metric is not available for the selection (e.g. litre-equivalent needs a liquid line).")
else:
    cmap = {**LINE_COLORS, **SHIFT_COLORS}
    if dim == "Month":
        f = px.line(ri, x=key, y=mcol, markers=True, color_discrete_sequence=["#0072B2"])
    else:
        f = px.bar(ri, x=key, y=mcol, color=key, color_discrete_map=cmap, text=ri[mcol].round(1 if ri[mcol].max() > 5 else 3))
    hi = ri.sort_values(mcol).iloc[-1]
    st.plotly_chart(style(f, f"{mname.split(' (')[0]} by {dim.lower()}: highest is {hi[key]} ({hi[mcol]:,.2f})", dim, mname, height=380, legend=False), use_container_width=True)
    with st.expander("Table (all metrics)"):
        st.dataframe(ri[[key, "oee", "kwh_per_1000", "water_l_per_unit", "water_l_per_litre", "waste_kg_per_1000", "co2e_kg_per_1000"]], hide_index=True, use_container_width=True)

# --- OEE vs resource -----------------------------------------------------------------------------
st.subheader("OEE and resource use go together")
lm = resource_intensity(S, ["line_id", "month_label"], A)
f = px.scatter(lm, x="oee", y="kwh_per_1000", color="line_id", color_discrete_map=LINE_COLORS, hover_data=["month_label"])
if len(lm) > 3:
    b, a0 = np.polyfit(lm["oee"], lm["kwh_per_1000"], 1)
    xs = np.linspace(lm["oee"].min(), lm["oee"].max(), 20)
    f.add_scatter(x=xs, y=a0 + b * xs, mode="lines", line=dict(color="#111827", dash="dash"), name="Linear fit")
    corr = np.corrcoef(lm["oee"], lm["kwh_per_1000"])[0, 1]
    ttl = f"Each OEE point is worth about {abs(b) / 100:.2f} kWh per 1,000 units (r = {corr:.2f}) across line-months"
else:
    ttl = "OEE vs energy intensity (line-month)"
f.update_xaxes(tickformat=".0%")
st.plotly_chart(style(f, ttl, "OEE (%)", "Energy (kWh per 1,000 good units)", height=420), use_container_width=True)
st.caption("The overall slope mixes different line types (UHT is energy-hungry regardless of OEE). Read the within-line pattern: when a line's OEE drops, the same fixed energy is spread over fewer good units.")

# --- Where losses burn resources -----------------------------------------------------------------
st.subheader("How OEE losses turn into resource waste")
kwh_op = S["energy_kwh"] / S["operating_time_min"]
stopped = S["planned_production_time_min"] - S["operating_time_min"]
idle_kwh = (stopped * A["idle_energy_share"] * kwh_op).groupby(S["line_id"]).sum()
rej_waste = (S["reject_units"] * S["unit_weight_kg"]).groupby(S["line_id"]).sum()
co_min = S.groupby("line_id")["min_changeover"].sum()
tbl = pd.DataFrame({"idle_energy_kwh": idle_kwh, "idle_co2e_t": idle_kwh * A["grid_factor_kg_per_kwh"] / 1000, "reject_product_waste_t": rej_waste / 1000}).reset_index()
tbl["share_of_line_energy"] = tbl["idle_energy_kwh"] / S.groupby("line_id")["energy_kwh"].sum().reindex(tbl["line_id"]).to_numpy()
c1, c2 = st.columns(2)
f = px.bar(tbl, x="line_id", y="idle_energy_kwh", color="line_id", color_discrete_map=LINE_COLORS)
c1.plotly_chart(style(f, f"Energy burned while stopped: ~{tbl['idle_energy_kwh'].sum() / 1e3:,.0f} MWh ({tbl['idle_energy_kwh'].sum() * A['grid_factor_kg_per_kwh'] / 1e6:,.0f} tCO2e)", "Line", "kWh (assumption-based)", height=340, legend=False), use_container_width=True)
f = px.bar(tbl, x="line_id", y="reject_product_waste_t", color="line_id", color_discrete_map=LINE_COLORS)
c2.plotly_chart(style(f, f"Rejects scrap about {tbl['reject_product_waste_t'].sum():,.0f} t of product", "Line", "Product waste from rejects (tonnes)", height=340, legend=False), use_container_width=True)
st.caption(f"Idle energy = stopped minutes x idle share ({A['idle_energy_share']:.0%}, editable) x the line's kWh per operating minute. Reject waste = reject units x unit weight (does not include process waste).")
with st.expander("Cross-check of the idle-energy assumption from the data"):
    st.dataframe(idle_energy_share_estimate(S), hide_index=True, use_container_width=True, column_config={
        "kwh_per_running_min": st.column_config.NumberColumn(format="%.2f"), "kwh_per_stopped_min": st.column_config.NumberColumn(format="%.2f"),
        "implied_idle_share": st.column_config.NumberColumn("Implied idle share", format="%.2f"), "r2": st.column_config.NumberColumn("R2", format="%.2f")})
    st.caption("Regression of shift energy on operating and stopped minutes per line (plus a night-shift term). The implied share is the ratio of the two coefficients. It is noisy (shift energy also depends on volume and CIP), so it is only a plausibility check for the sidebar assumption.")

# --- Scenario ------------------------------------------------------------------------------------
st.subheader("What would we save if the top loss shrinks?")
loss_min = {LOSS_CATEGORIES[s_]: S[f"min_{s_}"].sum() for s_ in LOSS_CATEGORIES}
top_default = max(loss_min, key=loss_min.get)
c1, c2 = st.columns(2)
lever = c1.selectbox("Loss to reduce", LEVERS, index=LEVERS.index(top_default))
red = c2.slider("Reduction", 5, 80, 30, 5, format="%d%%") / 100
sc = resource_scenario(S, lever, red, A)
m = st.columns(5)
m[0].metric("Time recovered", f"{sc['recovered_min'] / 60:,.0f} h")
m[1].metric("Energy avoided", f"{sc['avoided_kwh'] / 1e3:,.0f} MWh")
m[2].metric("Water avoided", f"{sc['avoided_water_l'] / 1e6:,.2f} ML")
m[3].metric("Waste avoided", f"{sc['avoided_waste_kg'] / 1e3:,.0f} t")
m[4].metric("CO2e avoided", f"{sc['avoided_co2e_t']:,.0f} t")
st.write(f"Reducing **{lever}** by {red:.0%} frees {sc['recovered_min'] / 60:,.0f} hours and lets the plant make {sc['extra_good_units'] / 1e6:,.2f} M extra good units if the time is reused.")
st.caption("Method: for downtime losses, recovered minutes are no longer scheduled at constant output, so the energy and water a stopped line uses in those minutes (idle-share assumptions) is avoided. "
           "For rejects, avoided scrap units save the energy and water used to make them plus their product weight as waste. Waste is not modelled for downtime levers. "
           "Selection period only; scale to a year as needed.")

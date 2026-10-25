"""Entry point: shared sidebar filters + page navigation (Streamlit >= 1.36)."""
import streamlit as st

st.set_page_config(page_title="OEE & Downtime Analytics", page_icon=":factory:", layout="wide")

from src.ui import build_ctx, footer  # noqa: E402  (after set_page_config)

pages = [
    st.Page("pages/p1_overview.py", title="1. Plant Overview", default=True),
    st.Page("pages/p2_loss_analysis.py", title="2. OEE Loss Analysis"),
    st.Page("pages/p3_downtime_deep_dive.py", title="3. Downtime Deep Dive"),
    st.Page("pages/p4_quality.py", title="4. Quality"),
    st.Page("pages/p5_maintenance.py", title="5. Maintenance & Early Warning"),
    st.Page("pages/p6_sustainability.py", title="6. Sustainability"),
    st.Page("pages/p7_improvements.py", title="7. Improvement Opportunities"),
]
nav = st.navigation(pages)
st.session_state["ctx"] = build_ctx()
nav.run()
footer()

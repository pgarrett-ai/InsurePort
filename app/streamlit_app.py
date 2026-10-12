"""InsurePort, the Streamlit app: does portfolio insurance raise the portfolio's geometric return?

Run from the repo root: streamlit run app/streamlit_app.py
Five pages under st.navigation (Portfolio -> Diagnostics -> Candidates -> Stress lab -> Methodology); the
settings sidebar lives here so its widgets keep their values across pages.
"""

import streamlit as st

import common

st.set_page_config(page_title="InsurePort", page_icon=":material/umbrella:", layout="wide")
pages = st.navigation([
    st.Page("pages/portfolio.py", title="Portfolio", icon=":material/account_balance_wallet:", default=True),
    st.Page("pages/diagnostics.py", title="Diagnostics", icon=":material/monitor_heart:"),
    st.Page("pages/candidates.py", title="Candidates", icon=":material/shield:"),
    st.Page("pages/stress_lab.py", title="Stress lab", icon=":material/science:"),
    st.Page("pages/methodology.py", title="Methodology", icon=":material/menu_book:"),
])
common.settings_sidebar()
pages.run()

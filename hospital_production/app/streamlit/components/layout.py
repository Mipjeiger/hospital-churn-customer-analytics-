import streamlit as st

"""Layout utilities for the Streamlit hospital dashboard.

- initialize_page: sets Streamlit page config and injects the CSS theme.
- render_sidebar: builds the sidebar with navigation and filters.
"""

from pathlib import Path
import sys

# Import theme CSS string
from app.streamlit.config.theme import CSS

def initialize_page():
    """Configure page and inject global CSS."""
    st.set_page_config(
        page_title="Hospital Care Analytics",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

def render_sidebar(
    df_raw,
    nav_pages,
    manifests,
    build_model_selection_func,
    settings,
):
    """Render the Streamlit sidebar and return the selected navigation page and filter states.

    Returns a dict with keys:
        nav, date_range, services, age_range, churn_opt, choices, manifests_loaded
    """
    with st.sidebar:
        st.markdown("## 🏥 Hospital Care Analytics")
        st.markdown(
            """<div style='color:#8AB4B5; font-size:0.82rem; margin-top:-8px; margin-bottom:16px;'>
            Operational & Patient Intelligence</div>""",
            unsafe_allow_html=True,
        )
        nav = st.radio("Navigation", nav_pages, label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### Filters")

        min_d = df_raw["arrival_date"].min().date()
        max_d = df_raw["arrival_date"].max().date()
        d_range = st.date_input(
            "Arrival date range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
        )
        start_d, end_d = (
            d_range if isinstance(d_range, tuple) and len(d_range) == 2 else (min_d, max_d)
        )

        all_services = sorted(df_raw["service"].dropna().unique().tolist())
        sel_services = st.multiselect("Service", all_services, default=all_services)

        age_min, age_max = int(df_raw["age"].min()), int(df_raw["age"].max())
        age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))
        churn_opt = st.selectbox("Patient churn", ["All", "Churned", "Retained"])

        st.markdown("---")
        st.caption("Operational and analytical use only. No clinical diagnosis or medical advice.")
        st.markdown("---")
        st.markdown("### 🧠 Models")
        if manifests:
            st.caption(f"{len(manifests)}/{len(nav_pages)} models loaded.")
            st.markdown(
                "".join(
                    f'<div style="font-size:0.78rem;color:#8AB4B5;">✅ {c}</div>'
                    for c in manifests
                ),
                unsafe_allow_html=True,
            )
        else:
            st.warning("No manifests available.")
        st.markdown("### 🎛️ Model Selection")
        choices = build_model_selection_func(manifests)

    return {
        "nav": nav,
        "date_range": (start_d, end_d),
        "services": sel_services,
        "age_range": age_range,
        "churn_opt": churn_opt,
        "choices": choices,
        "manifests": manifests,
    }

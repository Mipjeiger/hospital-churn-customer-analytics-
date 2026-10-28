# Theme and style definitions for the Streamlit hospital dashboard

# Color palette
PRIMARY_COLOR = "#0F2A44"
SECONDARY_COLOR = "#0E7C7B"
ACCENT_COLOR = "#2A9D8F"
BACKGROUND_COLOR = "#F5F7FA"
SURFACE_COLOR = "#FFFFFF"
BORDER_COLOR = "#E6EBF0"
TEXT_COLOR = "#0F2A44"
MUTED_TEXT_COLOR = "#6B7C93"
STATUS_GREEN = "#0E7C7B"
STATUS_AMBER = "#E6A817"
STATUS_RED = "#D64545"
STATUS_BLUE = "#9B1C1C"

# Font and layout settings (used in CSS injected via Streamlit)
FONT_FAMILY = "Inter, sans-serif"

# CSS template injected into the app
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*='css'] {{ font-family: '{FONT_FAMILY}'; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 1rem; }}
h1 {{ font-weight: 700; color: {PRIMARY_COLOR}; letter-spacing: -0.02em; }}
h2 {{ color: {PRIMARY_COLOR}; font-weight: 700; font-size: 1.35rem; margin-top: 1.6rem; border-bottom: 1px solid #E8EEF3; padding-bottom: 0.4rem; }}
h3 {{ color: #1a3a5c; font-weight: 600; font-size: 1.05rem; margin-top: 1rem; }}
.kpi-card {{ background: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(15,42,68,0.06); }}
.kpi-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; color: {MUTED_TEXT_COLOR}; font-weight: 600; }}
.kpi-value {{ font-size: 1.55rem; font-weight: 700; color: {PRIMARY_COLOR}; margin-top: 4px; }}
.kpi-sub {{ font-size: 0.78rem; color: #8A9BB0; margin-top: 2px; }}
.section-desc {{ color: #5A6D85; font-size: 0.9rem; margin-top: -6px; margin-bottom: 12px; }}
.insight-box {{ background: #F0F7F7; border-left: 4px solid {STATUS_GREEN}; border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 0.88rem; color: #1a3a5c; }}
.insight-warn {{ background: #FFF7E6; border-left-color: {STATUS_AMBER}; }}
.insight-bad {{ background: #FFF0F0; border-left-color: {STATUS_RED}; }}
[data-testid='stSidebar'] {{ background: {PRIMARY_COLOR}; }}
[data-testid='stSidebar'] * {{ color: #E8EEF3; }}
[data-testid='stSidebar'] .stRadio label {{ color: #E8EEF3 !important; }}
hr {{ border-color: #E8EEF3; }}
</style>
"""

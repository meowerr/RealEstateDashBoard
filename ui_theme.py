# -*- coding: utf-8 -*-
"""
Egypt Real Estate Intelligence Platform (v1) - UI Theme & Design System
File: ui_theme.py

Defines the executive design tokens, Egyptian color palette, custom CSS layout constraints,
and chart styling helpers for high-density uncompressed bento cockpit views.

Palette:
1. Deep Navy:     #1B2A4A (Primary sidebar, headers, dark-mode bases)
2. Nile Teal:     #2A9D8F (Main brand/data color, primary charts, key metrics, active nav)
3. Sandstone Gold: #E9B44C (Egyptian accent, highlights, top-tier categories, badges, CTAs)
4. Coral Red:     #E85D4C (Negative indicators, price drops, alerts, warnings)
5. Soft Sage:     #8FB996 (Positive indicators, clean air, green space, appreciation)
6. Off-white:     #F7F5F2 (Clean page background)
7. Light gray:    #D9D9D6 (Borders, dividers, subtle neutral cards)
8. Charcoal:      #333333 (Crisp legible body typography)
"""

from typing import Dict, Any, List, Optional
import streamlit as st

# -----------------------------------------------------------------------------
# 1. EXACT COLOR PALETTE
# -----------------------------------------------------------------------------
DEEP_NAVY = "#1B2A4A"
NILE_TEAL = "#2A9D8F"
SANDSTONE_GOLD = "#E9B44C"
CORAL_RED = "#E85D4C"
SOFT_SAGE = "#8FB996"
OFF_WHITE = "#F7F5F2"
LIGHT_GRAY = "#D9D9D6"
CHARCOAL = "#333333"

def get_color_palette() -> Dict[str, str]:
    """Returns the exact Egyptian Executive color palette dictionary."""
    return {
        "navy": DEEP_NAVY,
        "teal": NILE_TEAL,
        "gold": SANDSTONE_GOLD,
        "coral": CORAL_RED,
        "sage": SOFT_SAGE,
        "offwhite": OFF_WHITE,
        "lightgray": LIGHT_GRAY,
        "charcoal": CHARCOAL,
    }

# Convenience color mapping dictionary
COLOR_MAP = {
    "navy": DEEP_NAVY,
    "teal": NILE_TEAL,
    "gold": SANDSTONE_GOLD,
    "coral": CORAL_RED,
    "sage": SOFT_SAGE,
    "off_white": OFF_WHITE,
    "light_gray": LIGHT_GRAY,
    "charcoal": CHARCOAL,
}

# Continuous scales for Plotly maps & heatmaps
TEAL_TO_NAVY_SCALE = [
    [0.0, SOFT_SAGE],
    [0.35, NILE_TEAL],
    [1.0, DEEP_NAVY],
]

BLUES_TEAL_SCALE = [
    [0.0, OFF_WHITE],
    [0.3, SOFT_SAGE],
    [0.65, NILE_TEAL],
    [1.0, DEEP_NAVY],
]

# Discrete categorical mapping for community and finishings
AMENITIES_COLOR_MAP = {
    "B(Compound)": NILE_TEAL,
    "C(BASICS)": DEEP_NAVY,
}

FINISHING_COLOR_MAP = {
    "superlux": DEEP_NAVY,
    "semi": NILE_TEAL,
    "extra super lux": SOFT_SAGE,
    "lux": SANDSTONE_GOLD,
    "without": LIGHT_GRAY,
}

# -----------------------------------------------------------------------------
# 2. DYNAMIC DISPLAY SCALE CALIBRATION (ZERO SCROLL MANDATE)
# -----------------------------------------------------------------------------
def get_display_heights(display_scale: Optional[str] = "Standard") -> Dict[str, int]:
    """
    Returns proportional widget heights calibrated for the given display scale
    to guarantee zero vertical scrollbars within the 100vh viewport.
    """
    scale_str = str(display_scale or "").strip()
    if scale_str.startswith("Large"):
        # 1440p / 2K High-Resolution Display (e.g. 2486 x 1340px)
        return {
            "h_grid": 525,
            "h_upper": 530,
            "h_lower": 525,
            "h_sub": 525,
            "h_table_leader": 525,
            "h_bench_table": 270,
            "h_listing_table": 525,
        }
    elif scale_str.startswith("Compact"):
        # Compact Laptop / 768p-900p Display
        return {
            "h_grid": 265,
            "h_upper": 270,
            "h_lower": 265,
            "h_sub": 265,
            "h_table_leader": 265,
            "h_bench_table": 140,
            "h_listing_table": 265,
        }
    else:
        # Standard 1080p FHD Display (e.g. 1920 x 1080px) or default/empty
        return {
            "h_grid": 350,
            "h_upper": 355,
            "h_lower": 350,
            "h_sub": 350,
            "h_table_leader": 350,
            "h_bench_table": 175,
            "h_listing_table": 350,
        }


# -----------------------------------------------------------------------------
# 3. EXECUTIVE CSS DESIGN SYSTEM & VIEWPORT LOCK
# -----------------------------------------------------------------------------
RAW_CSS_CONTENT = f"""
    /* 1. Global Viewport Lock: Strict Zero Page Scroll */
    html, body {{
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: {OFF_WHITE} !important;
        color: {CHARCOAL} !important;
    }}
    
    [data-testid="stAppViewContainer"], .main {{
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        background-color: {OFF_WHITE} !important;
    }}
    
    /* 2. Streamlit Header Minimalist Adjustment */
    header[data-testid="stHeader"] {{
        height: 2.0rem !important;
        background: transparent !important;
        z-index: 99 !important;
    }}
    
    /* 3. Main Bento Container: Edge-to-edge full viewport fill */
    .block-container {{
        padding-top: 0.35rem !important;
        padding-bottom: 0.35rem !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
        max-width: 100% !important;
        height: calc(100vh - 2.2rem) !important;
        max-height: calc(100vh - 2.2rem) !important;
        overflow: hidden !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
    }}
    
    /* 4. Deep Navy Sidebar: High-Trust Executive Left Command Pane (Modern Executive Width & Spacing) */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {{
        min-width: 280px !important;
        max-width: 300px !important;
        width: 288px !important;
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
        overflow-y: hidden !important;
        scrollbar-width: none !important;
        border-right: 1px solid rgba(233, 180, 76, 0.25) !important;
        background-color: {DEEP_NAVY} !important;
        color: {OFF_WHITE} !important;
    }}
    
    [data-testid="stSidebarContent"] {{
        overflow-y: auto !important;
        scrollbar-width: none !important;
        height: 100% !important;
        padding: 0.65rem 0.80rem !important;
        background-color: {DEEP_NAVY} !important;
        color: {OFF_WHITE} !important;
    }}

    /* Modern spacing in sidebar */
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{
        gap: 0.45rem !important;
    }}

    /* Sidebar Typography */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {{
        color: {OFF_WHITE} !important;
    }}

    [data-testid="stSidebar"] label {{
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px !important;
        color: {OFF_WHITE} !important;
        margin-bottom: 2px !important;
        line-height: 1.25 !important;
    }}

    /* Sidebar Selectbox & Multiselect */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: #121D33 !important;
        border: 1px solid #2B3F66 !important;
        border-radius: 6px !important;
        color: #FFFFFF !important;
    }}

    [data-testid="stSidebar"] div[data-baseweb="select"] svg {{
        fill: {SANDSTONE_GOLD} !important;
    }}

    [data-testid="stSidebar"] div[data-baseweb="popover"] {{
        background-color: {DEEP_NAVY} !important;
        border: 1px solid #2B3F66 !important;
    }}

    /* Multiselect Tag Badges */
    [data-testid="stSidebar"] span[data-baseweb="tag"] {{
        background-color: {NILE_TEAL} !important;
        color: #FFFFFF !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        border-radius: 4px !important;
        border: none !important;
    }}

    [data-testid="stSidebar"] span[data-baseweb="tag"] svg {{
        fill: #FFFFFF !important;
    }}

    /* Multiselect Tag Remove Button Reset */
    [data-testid="stSidebar"] span[data-baseweb="tag"] button {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 0 0 4px !important;
        color: #FFFFFF !important;
    }}

    /* Sliders inside Sidebar - Comfortable clearance */
    [data-testid="stSidebar"] .stSlider {{
        margin-top: 1px !important;
        margin-bottom: 4px !important;
    }}

    [data-testid="stSidebar"] div[data-testid="stSliderTickBar"] {{
        background-color: #2B3F66 !important;
    }}

    [data-testid="stSidebar"] div[role="slider"] {{
        background-color: {SANDSTONE_GOLD} !important;
        border: 2px solid {DEEP_NAVY} !important;
        box-shadow: 0 0 4px rgba(233, 180, 76, 0.6) !important;
    }}

    [data-testid="stSidebar"] .stSlider div[data-testid="stMarkdownContainer"] p {{
        color: {SANDSTONE_GOLD} !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
    }}

    /* Radio Group inside Sidebar (Executive Nav & Toggles) */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
        margin-bottom: 3px !important;
        color: {OFF_WHITE} !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }}

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
        background-color: rgba(42, 157, 143, 0.22) !important;
        border-color: {NILE_TEAL} !important;
    }}

    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background-color: rgba(42, 157, 143, 0.32) !important;
        border-color: #2A9D8F !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}

    /* Sidebar Buttons */
    [data-testid="stSidebar"] div[data-testid="stButton"] > button,
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {{
        width: 100% !important;
        background: linear-gradient(135deg, #2A9D8F 0%, #1B2A4A 100%) !important;
        background-color: #2A9D8F !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(233, 180, 76, 0.45) !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 0.76rem !important;
        padding: 7px 10px !important;
        letter-spacing: 0.2px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;
        transition: all 0.2s ease-in-out !important;
        white-space: nowrap !important;
    }}

    [data-testid="stSidebar"] div[data-testid="stButton"] > button *,
    [data-testid="stSidebar"] .stButton > button *,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] * {{
        color: #FFFFFF !important;
    }}

    [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover,
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {{
        background: linear-gradient(135deg, #E9B44C 0%, #D49B28 100%) !important;
        background-color: #E9B44C !important;
        color: #1B2A4A !important;
        border-color: #E9B44C !important;
        box-shadow: 0 3px 10px rgba(233, 180, 76, 0.45) !important;
        transform: translateY(-1px) !important;
    }}

    [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover *,
    [data-testid="stSidebar"] .stButton > button:hover *,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover * {{
        color: #1B2A4A !important;
    }}

    [data-testid="stSidebar"] div[data-testid="stButton"] > button:active,
    [data-testid="stSidebar"] .stButton > button:active,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:active {{
        transform: translateY(0px) !important;
    }}

    /* Sidebar Dividers */
    [data-testid="stSidebar"] hr {{
        margin: 4px 0 !important;
        border: none !important;
        border-top: 1px solid rgba(217, 217, 214, 0.2) !important;
    }}

    /* Sidebar Collapse Button Reset */
    [data-testid="stSidebarCollapseButton"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 2px !important;
        color: #E9B44C !important;
    }}

    /* Hide all scrollbars in sidebar */
    [data-testid="stSidebar"]::-webkit-scrollbar,
    [data-testid="stSidebarContent"]::-webkit-scrollbar {{
        display: none !important;
        width: 0 !important;
    }}

    /* 5. Executive Header Strip (Main Viewport) */
    .top-header-strip {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border: 1px solid {LIGHT_GRAY};
        border-radius: 8px;
        padding: 6px 14px;
        margin-bottom: 6px;
        box-shadow: 0 1px 3px rgba(27, 42, 74, 0.04);
    }}
    
    .top-title {{
        font-size: 1.18rem;
        font-weight: 800;
        color: {DEEP_NAVY};
        margin: 0;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .top-subtitle {{
        font-size: 0.75rem;
        color: {CHARCOAL};
        opacity: 0.82;
        margin: 1px 0 0 0;
    }}
    
    .pulse-pill {{
        background: {OFF_WHITE};
        color: {DEEP_NAVY};
        border: 1px solid {LIGHT_GRAY};
        font-size: 0.73rem;
        font-weight: 700;
        padding: 3px 11px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    .pulse-dot {{
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
    }}
    
    /* 6. Executive KPI Metric Card */
    .stat-card {{
        background: #ffffff;
        border: 1px solid {LIGHT_GRAY};
        border-radius: 8px;
        padding: 7px 12px;
        box-shadow: 0 1px 3px rgba(27, 42, 74, 0.03);
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-left: 4px solid {DEEP_NAVY};
        height: 100%;
        box-sizing: border-box;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }}

    .stat-card:hover {{
        box-shadow: 0 4px 12px rgba(27, 42, 74, 0.08);
    }}
    
    .stat-label {{
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: {CHARCOAL};
        opacity: 0.75;
        letter-spacing: 0.4px;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    
    .stat-value {{
        font-size: 1.35rem;
        font-weight: 800;
        color: {DEEP_NAVY};
        line-height: 1.15;
    }}
    
    .stat-sub {{
        font-size: 11px;
        color: {NILE_TEAL};
        font-weight: 600;
        margin-top: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    /* 7. Plotly Chart Polish & Anti-Flicker Architecture */
    .js-plotly-plot .plotly .modebar {{
        transform: scale(0.72);
        transform-origin: top right;
    }}

    /* Plot Grid Rows & Top Padding */
    .plot-row-spacer {{
        margin-top: 8px !important;
    }}
    .plot-container-card {{
        background: #ffffff !important;
        border: 1px solid #D9D9D6 !important;
        border-radius: 8px !important;
        padding: 8px !important;
        box-shadow: 0 1px 3px rgba(27, 42, 74, 0.03) !important;
        margin-top: 6px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }}

    /* Square plot formatting rules & executive card framing */
    .stPlotlyChart {{
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 auto !important;
        opacity: 1 !important;
        transition: none !important;
        border-radius: 8px !important;
        border: 1px solid rgba(217, 217, 214, 0.75) !important;
        background: #ffffff !important;
        box-shadow: 0 1px 3px rgba(27, 42, 74, 0.03) !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }}

    .js-plotly-plot {{
        margin: 0 auto !important;
        opacity: 1 !important;
        transition: none !important;
    }}

    /* Main Content Area Selectbox & Input Polish (Executive Clean White) */
    [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] div[data-baseweb="select"] *,
    div[data-testid="stMain"] [data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stMain"] [data-testid="stSelectbox"] div[data-baseweb="select"] *,
    [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] .react-aria-ComboBox,
    [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] div[role="group"],
    [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] input[role="combobox"],
    [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] button,
    section[data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox,
    section[data-testid="stMain"] [data-testid="stSelectbox"] div[role="group"],
    section[data-testid="stMain"] [data-testid="stSelectbox"] input[role="combobox"],
    section[data-testid="stMain"] [data-testid="stSelectbox"] button {{
        background-color: #ffffff !important;
        color: #1B2A4A !important;
    }}

    [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stMain"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] div[role="group"],
    section[data-testid="stMain"] [data-testid="stSelectbox"] div[role="group"] {{
        background-color: #ffffff !important;
        border: 1px solid #D9D9D6 !important;
        border-radius: 6px !important;
    }}

    [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] input[role="combobox"],
    section[data-testid="stMain"] [data-testid="stSelectbox"] input[role="combobox"] {{
        background-color: transparent !important;
        color: #1B2A4A !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMainBlockContainer"] div[data-testid="stTextInput"] input,
    div[data-testid="stMain"] div[data-testid="stTextInput"] input {{
        background-color: #ffffff !important;
        border: 1px solid #D9D9D6 !important;
        border-radius: 6px !important;
        color: #1B2A4A !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMainBlockContainer"] [data-testid="stSelectbox"] svg,
    div[data-testid="stMain"] [data-testid="stSelectbox"] svg,
    section[data-testid="stMain"] [data-testid="stSelectbox"] svg {{
        fill: #1B2A4A !important;
    }}

    [data-testid="stMainBlockContainer"] input::placeholder,
    div[data-testid="stMain"] input::placeholder,
    section[data-testid="stMain"] input::placeholder {{
        color: #888888 !important;
    }}

    /* Top margin to Plotly charts inside the main content for breathing room from KPI cards */
    div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]:has(.stPlotlyChart) {{
        margin-top: 4px !important;
    }}

    /* Completely eliminate Streamlit rerun flicker & stale fading */
    [data-stale="true"],
    .stApp [data-stale="true"],
    div[data-testid="stVerticalBlock"] > div[data-stale="true"],
    div[data-testid="stElementContainer"][data-stale="true"] {{
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }}

    .stat-card {{
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}

    /* Tighten block vertical spacing */
    div[data-testid="stVerticalBlock"] > div {{
        gap: 0.35rem !important;
    }}
    
    /* Table container borders */
    div[data-testid="stDataFrame"] {{
        border-radius: 7px !important;
        border: 1px solid {LIGHT_GRAY} !important;
        box-shadow: 0 1px 2px rgba(27, 42, 74, 0.02) !important;
        transition: opacity 0.2s ease !important;
    }}
"""

def get_custom_css(raw: bool = False) -> str:
    """Returns the CSS styling string, optionally wrapped in <style> tags."""
    if raw:
        return RAW_CSS_CONTENT
    return f"<style>\n{RAW_CSS_CONTENT}\n</style>"

def inject_theme_css():
    """Injects the unified global theme CSS into Streamlit."""
    st.markdown(get_custom_css(raw=False), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. REUSABLE HTML COMPONENT RENDERERS
# -----------------------------------------------------------------------------
def render_header_strip(
    title: str,
    subtitle: str,
    badge_text: str,
    badge_type: str = "sage",
    badge_dot_color: Optional[str] = None
) -> str:
    """Renders and returns the executive top header bar HTML."""
    dot_color_map = {
        "sage": SOFT_SAGE,
        "teal": NILE_TEAL,
        "gold": SANDSTONE_GOLD,
        "coral": CORAL_RED,
        "navy": DEEP_NAVY,
    }
    final_dot_color = badge_dot_color or dot_color_map.get(badge_type.lower(), SOFT_SAGE)
    
    html = f"""
    <div class="top-header-strip">
        <div>
            <h1 class="top-title" style="color: {DEEP_NAVY};">{title}</h1>
            <p class="top-subtitle" style="color: {CHARCOAL};">{subtitle}</p>
        </div>
        <div class="pulse-pill" title="Live system status: Audited across 10,000 real estate properties">
            <span class="pulse-dot" style="display:inline-block; width:7px; height:7px; background:{final_dot_color}; border-radius:50%;"></span>
            {badge_text}
        </div>
    </div>
    """
    try:
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        pass
    return html


def render_stat_card(
    label: str,
    value: str,
    subtext: str,
    border_color: str = "#2A9D8F",
    subtext_color: str = "#8FB996",
    tooltip: str = ""
) -> str:
    """Renders and returns a single high-density KPI metric card HTML."""
    html = f"""
    <div class="stat-card" title="{tooltip or label}" style="border-left: 4px solid {border_color};">
        <span class="stat-label" style="font-size: 11px;">{label}</span>
        <span class="stat-value" style="font-size: 1.35rem; font-weight: 800; color: {DEEP_NAVY};">{value}</span>
        <span class="stat-sub" style="font-size: 11px; color: {subtext_color};">{subtext}</span>
    </div>
    """
    try:
        st.markdown(html, unsafe_allow_html=True)
    except Exception:
        pass
    return html


def render_kpi_card(
    label: str,
    value: str,
    subtitle: str,
    border_color: str = DEEP_NAVY,
    sub_color: str = NILE_TEAL,
    tooltip: str = ""
) -> str:
    """Alias helper for render_stat_card with optional executive tooltip."""
    return render_stat_card(
        label=label,
        value=value,
        subtext=subtitle,
        border_color=border_color,
        subtext_color=sub_color,
        tooltip=tooltip
    )


def render_sidebar_header():
    """Renders the modern executive brand header in the sidebar."""
    html = f"""
    <div style="margin-bottom: 8px; padding: 4px 2px 8px 2px; border-bottom: 1px solid rgba(233, 180, 76, 0.25);">
        <div style="display: flex; align-items: center; gap: 8px;">
            <div style="background: rgba(42, 157, 143, 0.2); border: 1px solid rgba(42, 157, 143, 0.45); border-radius: 7px; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-size: 1.05rem; flex-shrink: 0;">
                🏛️
            </div>
            <div>
                <div style="font-size: 0.86rem; font-weight: 800; color: {SANDSTONE_GOLD}; letter-spacing: 0.3px; line-height: 1.15;">
                    EGYPT REAL ESTATE
                </div>
                <div style="font-size: 0.63rem; color: {SOFT_SAGE}; font-weight: 600; letter-spacing: 0.2px;">
                    Executive Analytics v1
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar_sample_counter(active_count: int, total_count: int, target=None):
    """Renders the active sample counter badge in the Deep Navy sidebar."""
    active_pct = (active_count / total_count * 100.0) if total_count > 0 else 0.0
    html = f"""
    <div style="background: rgba(18, 29, 51, 0.9); border: 1px solid rgba(233, 180, 76, 0.25); border-radius: 6px; padding: 5px 8px; margin-top: 6px; display: flex; align-items: center; justify-content: space-between; box-shadow: 0 1px 3px rgba(0,0,0,0.15); box-sizing: border-box;">
        <span style="font-size: 0.65rem; color: {SANDSTONE_GOLD}; font-weight: 800; text-transform: uppercase; letter-spacing: 0.3px;">
            Active Scope
        </span>
        <span style="font-size: 0.74rem; color: #FFFFFF; font-weight: 700;">
            <strong style="color: #FFFFFF;">{active_count:,}</strong>
            <span style="color: {SOFT_SAGE}; font-size: 0.66rem; font-weight: 600;"> / {total_count:,} ({active_pct:.1f}%)</span>
        </span>
    </div>
    """
    if target is not None:
        target.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. PLOTLY THEME SPECIFICATION
# -----------------------------------------------------------------------------
def get_plotly_layout_defaults() -> Dict[str, Any]:
    """Returns Plotly layout defaults configured for the Egyptian Executive palette."""
    return {
        "paper_bgcolor": "#F7F5F2",
        "plot_bgcolor": "#FFFFFF",
        "font": dict(family="Inter, -apple-system, sans-serif", color=CHARCOAL, size=10),
        "title": dict(
            font=dict(color=DEEP_NAVY, size=12, family="Inter, sans-serif", weight="bold"),
            x=0.01,
            y=0.96
        ),
        "xaxis": dict(gridcolor=LIGHT_GRAY, tickfont=dict(size=8, color=CHARCOAL)),
        "yaxis": dict(gridcolor=LIGHT_GRAY, tickfont=dict(size=8, color=CHARCOAL)),
        "margin": dict(l=25, r=25, t=35, b=25),
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
        "colorway": [DEEP_NAVY, NILE_TEAL, SANDSTONE_GOLD, SOFT_SAGE, CORAL_RED],
        "uirevision": "constant",
        "transition": dict(duration=350, easing="cubic-in-out")
    }


def apply_plotly_theme(fig, height: Optional[int] = None, title_text: str = ""):
    """Applies the executive layout defaults to a Plotly figure."""
    defaults = get_plotly_layout_defaults()
    if height is not None:
        defaults["height"] = height
    if title_text:
        defaults["title"]["text"] = title_text
    fig.update_layout(**defaults)
    return fig


def apply_chart_theme(fig, title_text: str = "", height: int = 340, margin: Optional[Dict[str, int]] = None):
    """Convenience wrapper for apply_plotly_theme."""
    fig = apply_plotly_theme(fig, height=height, title_text=title_text)
    if margin:
        fig.update_layout(margin=margin)
    return fig

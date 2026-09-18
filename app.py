# -*- coding: utf-8 -*-
"""
Egypt Real Estate Intelligence Platform (v1) - Streamlit Executive Application
File: app.py

High-Density Uncompressed Bento Architecture (Full-Viewport Fit, Zero Scroll)

DESIGN PRINCIPLES:
1. Strict Zero Page Scroll: The entire dashboard fits naturally within the 100vh viewport.
2. Left Sidebar: All global market filters, simulation controls, and navigation reside in the sidebar.
3. Instant Visual Feedback: Filter and parameter changes immediately update all visible widgets.
4. Egyptian Executive Palette:
   - Deep Navy (#1B2A4A): Sidebar, headers, card anchors
   - Nile Teal (#2A9D8F): Brand color, primary charts, key metrics, active states
   - Sandstone Gold (#E9B44C): Egyptian accent, top-tier categories, highlight badges
   - Coral Red (#E85D4C): Negative indicators, price drops, alerts
   - Soft Sage (#8FB996): Positive indicators, air quality, green space, appreciation
   - Off-white (#F7F5F2): Page background
   - Light gray (#D9D9D6): Borders & dividers
   - Charcoal (#333333): Primary legible text
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Local modular components
import importlib
import ui_theme
import cockpit_views
import geo_analytics
import financial_analytics

importlib.reload(ui_theme)
importlib.reload(cockpit_views)
importlib.reload(geo_analytics)
importlib.reload(financial_analytics)

from data_loader import load_clean_data, get_filter_options
from ui_theme import (
    DEEP_NAVY,
    NILE_TEAL,
    SANDSTONE_GOLD,
    CORAL_RED,
    SOFT_SAGE,
    OFF_WHITE,
    LIGHT_GRAY,
    CHARCOAL,
    inject_theme_css,
    render_sidebar_header,
    render_sidebar_sample_counter,
)
from cockpit_views import (
    render_cockpit_1,
    render_cockpit_2,
    render_cockpit_3,
)

# -----------------------------------------------------------------------------
# 1. STREAMLIT CONFIGURATION & VIEWPORT LOCK
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Egypt Real Estate Intelligence (v1)",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject executive CSS tokens and viewport containment (Zero Page Scroll)
inject_theme_css()

# -----------------------------------------------------------------------------
# 2. DATA INGESTION
# -----------------------------------------------------------------------------
df_master = load_clean_data()
filter_opts = get_filter_options(df_master)
total_count = len(df_master)

# -----------------------------------------------------------------------------
# 3. EXECUTIVE LEFT SIDEBAR COMMAND PANE
# -----------------------------------------------------------------------------
with st.sidebar:
    # 1. Platform Brand Header
    render_sidebar_header()

    # 2. Cockpit Navigation
    st.markdown(
        f"<p style='font-size: 0.72rem; font-weight: 800; text-transform: uppercase; color: {SANDSTONE_GOLD}; letter-spacing: 0.4px; margin: 0 0 4px 0;'>"
        "🧭 Analytics Views</p>",
        unsafe_allow_html=True
    )
    cockpit_options = [
        "1. 📊 Market Overview",
        "2. 💼 Investment & Yield",
        "3. 🏗️ Supply & Pipeline"
    ]
    current_cockpit = st.radio(
        "Analytics Views",
        cockpit_options,
        index=0,
        label_visibility="collapsed",
        help="Switch between: 1. Market Overview, 2. Investment & Yield Forecast, or 3. Supply & Pipeline.",
        key="app_cockpit_nav"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # 3. Dedicated Financial Simulation Controls (Rendered when Tab 2 is active)
    gross_yield_pct = 7.5
    opex_pct = 15.0
    equity_pct = 50.0
    horizon_years = 10

    if current_cockpit.startswith("2."):
        st.markdown(
            f"<p style='font-size: 0.72rem; font-weight: 800; text-transform: uppercase; color: {SANDSTONE_GOLD}; letter-spacing: 0.4px; margin: 0 0 4px 0;'>"
            "💼 Investment Assumptions</p>",
            unsafe_allow_html=True
        )
        c_fin1, c_fin2 = st.columns(2)
        with c_fin1:
            gross_yield_pct = st.slider(
                "Gross Yield (%)",
                min_value=3.0,
                max_value=12.0,
                value=7.5,
                step=0.5,
                help="Expected gross annual rental income as a percentage of property valuation (typical Cairo prime: 6% - 9%).",
                key="sb_gross_yield"
            )
        with c_fin2:
            opex_pct = st.slider(
                "OPEX (%)",
                min_value=5.0,
                max_value=30.0,
                value=15.0,
                step=2.5,
                help="Maintenance, service charges, management fees, and vacancy reserves (typical: 10% - 20%).",
                key="sb_opex"
            )
        c_fin3, c_fin4 = st.columns(2)
        with c_fin3:
            equity_pct = st.slider(
                "Down Pmt (%)",
                min_value=20.0,
                max_value=100.0,
                value=50.0,
                step=5.0,
                help="Initial equity capital invested versus financing.",
                key="sb_down_pmt"
            )
        with c_fin4:
            horizon_years = st.selectbox(
                "Holding Years",
                options=[5, 7, 10],
                index=2,
                help="Target investment holding duration before simulated exit valuation.",
                key="sb_horizon"
            )
        st.markdown("<hr>", unsafe_allow_html=True)

    # 4. Global Market Filters
    st.markdown(
        f"<p style='font-size: 0.72rem; font-weight: 800; text-transform: uppercase; color: {SANDSTONE_GOLD}; letter-spacing: 0.4px; margin: 0 0 4px 0;'>"
        "🎯 Market Filters</p>",
        unsafe_allow_html=True
    )

    # Price Slider (Million EGP)
    price_range = st.slider(
        "Price Range (M EGP)",
        min_value=0.5,
        max_value=30.0,
        value=(0.5, 25.0),
        step=0.5,
        help="Filter apartments by total asking price range in Million EGP.",
        key="sb_price_range"
    )

    # Macro Region & Finishing Type Filter (Side-by-side)
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        selected_regions = st.multiselect(
            "Macro Region",
            options=filter_opts['regions'],
            default=[],
            placeholder="All",
            help="Filter listings across Cairo Metropolitan, East/West Cairo corridors, Red Sea, Coastal, and New Urban communities.",
            key="sb_regions"
        )
    with c_f2:
        selected_finishing = st.multiselect(
            "Finishing",
            options=filter_opts['finishing'],
            default=[],
            placeholder="All",
            help="Filter by delivery finish specification (Superlux, Lux, Semi-finished, Core & Shell).",
            key="sb_finishing"
        )

    # Bedrooms & Community Type Filter (Side-by-side)
    c_r1, c_r2 = st.columns(2)
    with c_r1:
        rooms_choice = st.selectbox(
            "Bedrooms",
            options=["All", 1, 2, 3, 4, 5],
            index=0,
            help="Filter units by number of registered bedrooms (1 to 5+ beds).",
            key="sb_rooms"
        )
    with c_r2:
        amenity_choice = st.selectbox(
            "Community",
            options=["All", "Gated Compound (B)", "Standalone Urban (C)"],
            index=0,
            help="Select community topology: Gated Compound (B) with master amenities vs Standalone Urban (C) apartments.",
            key="sb_amenity"
        )

    # Listing Purpose & Negotiation Haircut Filter (Side-by-side)
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        type_choice = st.selectbox(
            "Purpose",
            options=["All", "Own (Residential)", "Investment"],
            index=0,
            help="Filter by listing intent: Primary Residential (Own) vs Capital Growth (Investment).",
            key="sb_type"
        )
    with c_p2:
        negotiation_discount = st.slider(
            "Ask Haircut (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=2.5,
            help="Market execution haircut: Adjusts listing portal asking prices down to estimated closed/transacted contract values (Cairo norm: 5% - 10%).",
            key="sb_negotiation_discount"
        )

    # Display Scaling Mode (Default 1080p)
    display_scale = st.selectbox(
        "Display Scaling Mode",
        options=["Standard 1080p", "Large / 2K Display", "Compact Laptop"],
        index=0,
        help="Calibrates chart heights to 1080p standard resolution with zero vertical scrollbars.",
        key="sb_display_scale"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # 6. Action Buttons: 2 Presets in Row 1 + Full-Width Reset in Row 2
    def _apply_preset_compounds():
        st.session_state["sb_amenity"] = "Gated Compound (B)"

    def _apply_preset_investment():
        st.session_state["sb_type"] = "Investment"

    def _reset_all_filters():
        st.session_state["sb_price_range"] = (0.5, 25.0)
        st.session_state["sb_regions"] = []
        st.session_state["sb_finishing"] = []
        st.session_state["sb_rooms"] = "All"
        st.session_state["sb_amenity"] = "All"
        st.session_state["sb_type"] = "All"
        st.session_state["sb_negotiation_discount"] = 5.0
        st.session_state["sb_gross_yield"] = 7.5
        st.session_state["sb_opex"] = 15.0
        st.session_state["sb_down_pmt"] = 50.0
        st.session_state["sb_horizon"] = 10

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.button(
            "🌟 Compounds",
            use_container_width=True,
            help="1-Click Preset: Filter for Gated Compounds with premium amenities.",
            key="sb_pre_comp",
            on_click=_apply_preset_compounds
        )
    with c_b2:
        st.button(
            "💼 Investment",
            use_container_width=True,
            help="1-Click Preset: Filter for high-growth Investment listings.",
            key="sb_pre_inv",
            on_click=_apply_preset_investment
        )

    st.button(
        "🔄 Reset All Filters",
        use_container_width=True,
        help="Reset all filters, simulation sliders, and view presets back to the default 10,000-listing master baseline.",
        key="sb_reset_btn",
        on_click=_reset_all_filters
    )

    # Real-Time Active Sample Counter Badge (Reserved Container)
    sample_counter_placeholder = st.container()


# -----------------------------------------------------------------------------
# 4. APPLY FILTERING TO DATASET
# -----------------------------------------------------------------------------
filtered_df = df_master.copy()

# 1. Price filter
p_min, p_max = price_range[0] * 1e6, price_range[1] * 1e6
filtered_df = filtered_df[(filtered_df['Price_Numeric'] >= p_min) & (filtered_df['Price_Numeric'] <= p_max)]

# 2. Region filter
if selected_regions:
    filtered_df = filtered_df[filtered_df['macro_region'].isin(selected_regions)]

# 3. Amenity filter
if amenity_choice == "Gated Compound (B)":
    filtered_df = filtered_df[filtered_df['Amenities'] == 'B(Compound)']
elif amenity_choice == "Standalone Urban (C)":
    filtered_df = filtered_df[filtered_df['Amenities'] == 'C(BASICS)']

# 4. Finishing filter
if selected_finishing:
    filtered_df = filtered_df[filtered_df['finishing_type'].isin(selected_finishing)]

# 5. Rooms filter
if rooms_choice != "All":
    filtered_df = filtered_df[filtered_df['Rooms'] == float(rooms_choice)]

# 6. Type filter
if type_choice == "Own (Residential)":
    filtered_df = filtered_df[filtered_df['Type'] == 'Own']
elif type_choice == "Investment":
    filtered_df = filtered_df[filtered_df['Type'] == 'Investment']

active_count = len(filtered_df)
active_pct = (active_count / total_count * 100.0) if total_count > 0 else 0.0

# Render the sidebar active sample counter badge with live counts
render_sidebar_sample_counter(active_count=active_count, total_count=total_count, target=sample_counter_placeholder)


# -----------------------------------------------------------------------------
# 5. DISPATCH TO EXECUTIVE BENTO COCKPIT
# -----------------------------------------------------------------------------
if current_cockpit.startswith("1."):
    render_cockpit_1(
        filtered_df=filtered_df,
        active_count=active_count,
        display_scale=display_scale,
        total_count=total_count,
        active_pct=active_pct
    )
elif current_cockpit.startswith("2."):
    render_cockpit_2(
        filtered_df=filtered_df,
        active_count=active_count,
        display_scale=display_scale
    )
else:
    render_cockpit_3(
        filtered_df=filtered_df,
        active_count=active_count,
        display_scale=display_scale
    )

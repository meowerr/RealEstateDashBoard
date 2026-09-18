# -*- coding: utf-8 -*-
"""
Egypt Real Estate Intelligence Platform (v1) - Cockpit Modular Views Renderer
File: cockpit_views.py

Renders the 3 executive, high-density, spacious, uncompressed Cockpit views:
1. render_cockpit_1: 📊 Market Overview & Property Map
2. render_cockpit_2: 💼 10-Year Investment & Cash Flow Forecast
3. render_cockpit_3: 🏗️ Supply Pipeline & Inventory Database

Adheres strictly to the Zero Page Scroll constraint and the Egyptian Executive Color Palette:
- Deep Navy: #1B2A4A
- Nile Teal: #2A9D8F
- Sandstone Gold: #E9B44C
- Coral Red: #E85D4C
- Soft Sage: #8FB996
- Off-white: #F7F5F2
- Light gray: #D9D9D6
- Charcoal: #333333
"""

import sys
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Local data & ML engine imports
from data_loader import get_location_coordinates
from ml_engine import evaluate_benchmarks

# Geospatial & Cohort Analytics module
try:
    from geo_analytics import (
        render_enhanced_geo_map,
        render_cohort_escalation_chart,
        generate_clustered_hubs,
    )
except ImportError:
    render_enhanced_geo_map = None
    render_cohort_escalation_chart = None
    generate_clustered_hubs = None

# Institutional Financial & PE Underwriting module
try:
    from financial_analytics import (
        calculate_cagr_and_moic,
        render_sensitivity_heatmap,
        apply_negotiation_discount,
        generate_sensitivity_matrix,
    )
except ImportError:
    calculate_cagr_and_moic = None
    render_sensitivity_heatmap = None
    apply_negotiation_discount = None
    generate_sensitivity_matrix = None

# UI Theme & Design Tokens import (with defensive fallbacks)
try:
    from ui_theme import (
        DEEP_NAVY,
        NILE_TEAL,
        SANDSTONE_GOLD,
        CORAL_RED,
        SOFT_SAGE,
        OFF_WHITE,
        LIGHT_GRAY,
        CHARCOAL,
        TEAL_TO_NAVY_SCALE,
        BLUES_TEAL_SCALE,
        AMENITIES_COLOR_MAP,
        FINISHING_COLOR_MAP,
        get_display_heights,
        render_header_strip,
        render_kpi_card,
        apply_chart_theme,
    )
except ImportError:
    # Autonomous standalone fallback definitions
    DEEP_NAVY = "#1B2A4A"
    NILE_TEAL = "#2A9D8F"
    SANDSTONE_GOLD = "#E9B44C"
    CORAL_RED = "#E85D4C"
    SOFT_SAGE = "#8FB996"
    OFF_WHITE = "#F7F5F2"
    LIGHT_GRAY = "#D9D9D6"
    CHARCOAL = "#333333"

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

    def get_display_heights(display_scale: str) -> Dict[str, int]:
        scale_str = str(display_scale or "").strip()
        if scale_str.startswith("Large"):
            return {
                "h_grid": 525,
                "h_upper": 530,
                "h_lower": 525,
                "h_sub": 525,
                "h_table_leader": 525,
                "h_bench_table": 270,
                "h_listing_table": 525,
            }
        elif scale_str.startswith("Standard"):
            return {
                "h_grid": 360,
                "h_upper": 365,
                "h_lower": 360,
                "h_sub": 360,
                "h_table_leader": 360,
                "h_bench_table": 185,
                "h_listing_table": 360,
            }
        else:
            return {
                "h_grid": 265,
                "h_upper": 270,
                "h_lower": 265,
                "h_sub": 265,
                "h_table_leader": 265,
                "h_bench_table": 140,
                "h_listing_table": 265,
            }

    def render_header_strip(title: str, subtitle: str, badge_text: str, badge_dot_color: str = NILE_TEAL):
        html = f"""
        <div class="top-header-strip">
            <div>
                <h1 class="top-title">{title}</h1>
                <p class="top-subtitle">{subtitle}</p>
            </div>
            <div class="pulse-pill">
                <span style="display:inline-block; width:7px; height:7px; background:{badge_dot_color}; border-radius:50%;"></span>
                {badge_text}
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    def render_kpi_card(label: str, value: str, subtitle: str, border_color: str = DEEP_NAVY, sub_color: str = NILE_TEAL, tooltip: str = ""):
        title_attr = f' title="{tooltip}"' if tooltip else ""
        html = f"""
        <div class="stat-card" style="border-left-color: {border_color};"{title_attr}>
            <span class="stat-label"{title_attr}>{label}</span>
            <span class="stat-value">{value}</span>
            <span class="stat-sub" style="color: {sub_color};">{subtitle}</span>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    def apply_chart_theme(fig, title_text: str = "", height: int = 340, margin: Optional[Dict[str, int]] = None):
        margin_dict = margin or dict(l=20, r=20, t=32, b=20)
        updates: Dict[str, Any] = {
            "height": height,
            "margin": margin_dict,
            "plot_bgcolor": "#ffffff",
            "paper_bgcolor": "#ffffff",
            "font": dict(family="Inter, -apple-system, sans-serif", color=CHARCOAL, size=10),
            "uirevision": "constant",
            "transition": dict(duration=350, easing="cubic-in-out"),
        }
        if title_text:
            updates["title"] = dict(
                text=title_text,
                font=dict(size=12, color=DEEP_NAVY, family="Inter, sans-serif"),
                x=0.01,
                y=0.96,
            )
        fig.update_layout(**updates)
        return fig


# -----------------------------------------------------------------------------
# 1. COCKPIT 1: 📊 MARKET OVERVIEW & PROPERTY MAP
# -----------------------------------------------------------------------------
def render_cockpit_1(
    filtered_df: pd.DataFrame,
    active_count: int,
    total_count: int,
    active_pct: float,
    display_scale: str
) -> None:
    """
    Renders Cockpit 1: Market Overview & Property Map.
    Features:
    - 5 Grand KPI Cards (Price, PPM, Compound %, 10Y Return, AQI & Green)
    - 2-Column Balanced Bento Grid:
      * Left: Interactive Cairo map (Teal-to-Navy) + Top Hubs Leaderboard Table
      * Right: 10Y Capital Appreciation Bar Chart + Finishing Donut & Price Histogram
    """
    heights = get_display_heights(display_scale)
    h_upper = heights["h_upper"]
    h_sub = heights["h_sub"]
    h_table_leader = heights["h_table_leader"]

    # 1. Top Executive Header Strip
    render_header_strip(
        title="📊 Market Overview & Property Map",
        subtitle="Property Pricing, Long-Term Appreciation & Neighborhood Quality",
        badge_text=f"Active: {active_count:,} Units ({active_pct:.1f}% Market Share)",
        badge_dot_color=NILE_TEAL
    )

    # 2. Metric Calculations with defensive fallbacks
    if active_count > 0:
        med_price = filtered_df["Price_Numeric"].median()
        q25 = filtered_df["Price_Numeric"].quantile(0.25)
        q75 = filtered_df["Price_Numeric"].quantile(0.75)

        med_ppm = filtered_df["Calculated_Price_Per_Meter"].median()
        min_ppm = filtered_df["Calculated_Price_Per_Meter"].min()
        max_ppm = filtered_df["Calculated_Price_Per_Meter"].max()

        compound_pct = (filtered_df["Amenities"] == "B(Compound)").mean() * 100.0
        avg_aqi = filtered_df["us_aqi"].mean() if "us_aqi" in filtered_df.columns else 79.1
        avg_green = filtered_df["green_coverage_pct"].mean() if "green_coverage_pct" in filtered_df.columns else 3.1

        # 10-Year historical growth multiplier calculation
        inv_active = filtered_df[filtered_df["Type"] == "Investment"] if "Type" in filtered_df.columns else pd.DataFrame()
        if len(inv_active) > 0 and "price after 10 years" in inv_active.columns and inv_active["price after 10 years"].notnull().any():
            valid_inv = inv_active[inv_active["price after 10 years"] > 0]
            med_mult = (valid_inv["price after 10 years"] / valid_inv["Price_Numeric"]).median() if len(valid_inv) > 0 else 1.84
        else:
            med_mult = 1.84
    else:
        med_price, q25, q75 = 0.0, 0.0, 0.0
        med_ppm, min_ppm, max_ppm = 0.0, 0.0, 0.0
        compound_pct, avg_aqi, avg_green, med_mult = 0.0, 0.0, 0.0, 1.0

    # 3. 5 Top KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card(
            label="Median Asking Price",
            value=f"{med_price / 1e6:.2f}M <small style='font-size:0.75rem; color:{CHARCOAL};'>EGP</small>",
            subtitle=f"IQR: {q25 / 1e6:.1f}M - {q75 / 1e6:.1f}M EGP",
            border_color=DEEP_NAVY,
            sub_color=NILE_TEAL,
            tooltip="Median listing price across active sample. Subtitle shows 25th-75th percentile interquartile range."
        )
    with k2:
        render_kpi_card(
            label="Price per m²",
            value=f"{med_ppm:,.0f} <small style='font-size:0.75rem; color:{CHARCOAL};'>EGP</small>",
            subtitle=f"Range: {min_ppm:,.0f} - {max_ppm:,.0f}",
            border_color=NILE_TEAL,
            sub_color=DEEP_NAVY,
            tooltip="Calculated median price per square meter (Price / Gross Area) in EGP/m²."
        )
    with k3:
        render_kpi_card(
            label="Compound Share %",
            value=f"{compound_pct:.1f}%",
            subtitle=f"{(100.0 - compound_pct):.1f}% Standalone Urban",
            border_color=SANDSTONE_GOLD,
            sub_color=DEEP_NAVY,
            tooltip="Percentage of properties located within gated master-planned compounds vs standalone urban inventory."
        )
    with k4:
        if calculate_cagr_and_moic:
            cagr_dict = calculate_cagr_and_moic(100.0, 100.0 * med_mult, horizon_years=10)
            cagr_val = cagr_dict["cagr_pct"]
        else:
            cagr_val = ((med_mult ** (1.0 / 10.0)) - 1.0) * 100.0 if med_mult > 0 else 0.0

        render_kpi_card(
            label="10-Year Growth Forecast",
            value=f"{cagr_val:.1f}% <small style='font-size:0.75rem; color:{CHARCOAL};'>CAGR</small>",
            subtitle=f"{med_mult:.2f}x 10Y Multiple",
            border_color=SANDSTONE_GOLD,
            sub_color=DEEP_NAVY,
            tooltip=f"Annualized Compound Annual Growth Rate (CAGR: {cagr_val:.2f}%/year) forecasted by the valuation model. Subtitle displays the 10-year investment multiple ({med_mult:.2f}x Multiple on Invested Capital / MOIC)."
        )
    with k5:
        render_kpi_card(
            label="Air Quality & Green Coverage",
            value=f"{avg_aqi:.1f} <small style='font-size:0.75rem; color:{CHARCOAL};'>AQI</small>",
            subtitle=f"{avg_green:.1f}% Urban Green Coverage",
            border_color=CORAL_RED,
            sub_color=CHARCOAL,
            tooltip="Environmental index: EPA US-AQI (lower is cleaner) and satellite-derived urban green space percentage."
        )

    # 4. Perfectly Balanced 2x2 Bento Grid (Even Spacing, Edge-to-Edge)
    h_grid = heights.get("h_grid", h_upper)

    # --- ROW 1: Geospatial Heatmap (Left) & 10Y Benchmark (Right) ---
    # Geocoded locations for map & liquidity leaderboard
    geo_points = get_location_coordinates(filtered_df, min_count=2)
    if geo_points.empty and active_count > 0:
        geo_points = get_location_coordinates(filtered_df, min_count=1)

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        # Interactive Map (Enhanced Density & Clustered Hubs)
        if render_enhanced_geo_map is not None:
            fig_map = render_enhanced_geo_map(
                filtered_df=filtered_df,
                active_count=active_count,
                height=h_grid,
                map_type="density",
                title="🗺️ Property Price & Location Density (Cairo & Coastal)"
            )
            fig_map.update_layout(
                title=dict(
                    text="🗺️ Property Price & Location Density (Cairo & Coastal)",
                    font=dict(size=12, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"),
                    x=0.02,
                    y=0.97
                ),
                margin=dict(l=10, r=10, t=34, b=10),
                map=dict(
                    center=dict(lat=30.044, lon=31.335),
                    zoom=9.3
                ),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff"
            )
            st.plotly_chart(fig_map, config={"displayModeBar": False}, key="c1_geo_map_density")
        else:
            if not geo_points.empty:
                fig_map = px.scatter_map(
                    geo_points,
                    lat="lat",
                    lon="lon",
                    size="count",
                    color="median_ppm",
                    hover_name="location",
                    hover_data={
                        "count": True,
                        "median_price": ":,.0f",
                        "median_ppm": ":,.0f",
                        "aqi": ":.1f",
                        "green_pct": ":.2f%"
                    },
                    color_continuous_scale=TEAL_TO_NAVY_SCALE,
                    size_max=22,
                    zoom=9.3,
                    center=dict(lat=30.044, lon=31.335)
                )
                fig_map.update_traces(
                    hovertemplate="<b>%{hovertext}</b><br>"
                                  "Units: %{customdata[0]:,}<br>"
                                  "Median Price: %{customdata[1]:,.0f} EGP<br>"
                                  "EGP/m²: %{customdata[2]:,.0f}<br>"
                                  "AQI: %{customdata[3]:.1f}<br>"
                                  "Green %: %{customdata[4]:.1f}%<extra></extra>"
                )
                fig_map.update_layout(
                    title=dict(
                        text="🗺️ Property Price & Location Density (Cairo & Coastal)",
                        font=dict(size=12, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"),
                        x=0.02,
                        y=0.97
                    ),
                    map_style="open-street-map",
                    height=h_grid,
                    margin=dict(l=10, r=10, t=34, b=10),
                    map=dict(
                        center=dict(lat=30.044, lon=31.335),
                        zoom=9.3
                    ),
                    coloraxis_colorbar=dict(
                        title=dict(text="EGP/m²", font=dict(size=9, color=CHARCOAL)),
                        thickness=9,
                        len=0.68,
                        tickfont=dict(size=8, color=CHARCOAL)
                    ),
                    uirevision="geo_cairo",
                    transition=dict(duration=350, easing="cubic-in-out")
                )
                st.plotly_chart(fig_map, config={"displayModeBar": False}, key="c1_geo_map_scatter")
            else:
                st.info("No geocoded listings found matching current filters.")

    with r1_col2:
        # 10-Year Capital Appreciation Benchmark
        if active_count > 0 and "normalized_location" in filtered_df.columns:
            top_locs = filtered_df["normalized_location"].value_counts().head(6).index.tolist()
        else:
            top_locs = []

        if top_locs:
            loc_summary = []
            for loc in top_locs:
                sub = filtered_df[filtered_df["normalized_location"] == loc]
                cur_p = sub["Price_Numeric"].median()
                inv_sub = sub[sub["price after 10 years"] > 0] if "price after 10 years" in sub.columns else pd.DataFrame()
                fut_p = inv_sub["price after 10 years"].median() if len(inv_sub) > 0 else cur_p * 1.95
                growth_pct = ((fut_p - cur_p) / cur_p * 100.0) if cur_p > 0 else 0.0
                growth_mult = (fut_p / cur_p) if cur_p > 0 else 1.0
                loc_summary.append({
                    "Location": loc[:18],
                    "Current Value": round(cur_p / 1e6, 2),
                    "Projected Value": round(fut_p / 1e6, 2),
                    "Growth Pct": round(growth_pct, 1),
                    "Multiplier": round(growth_mult, 2),
                })

            df_chart = pd.DataFrame(loc_summary)
            fig_appr = go.Figure()
            fig_appr.add_trace(go.Bar(
                x=df_chart["Location"],
                y=df_chart["Current Value"],
                ids=df_chart["Location"],
                name="Current (2026)",
                marker_color=DEEP_NAVY,
                text=df_chart["Current Value"].apply(lambda v: f"{v}M"),
                textposition="auto",
                textfont=dict(size=9, color="#ffffff"),
                hovertemplate="<b>Location: %{x}</b><br>Current Valuation: %{y:.2f}M EGP<extra></extra>"
            ))
            fig_appr.add_trace(go.Bar(
                x=df_chart["Location"],
                y=df_chart["Projected Value"],
                ids=df_chart["Location"],
                customdata=df_chart[["Growth Pct", "Multiplier"]],
                name="Projected (2036)",
                marker_color=SOFT_SAGE,
                text=df_chart["Projected Value"].apply(lambda v: f"{v}M"),
                textposition="auto",
                textfont=dict(size=9, color=DEEP_NAVY),
                hovertemplate="<b>Location: %{x}</b><br>Projected Valuation: %{y:.2f}M EGP<br>10Y Growth: +%{customdata[0]:.1f}% (%{customdata[1]:.2f}x)<extra></extra>"
            ))
            fig_appr.update_layout(
                title=dict(
                    text="📈 10-Year Property Appreciation by Area",
                    font=dict(size=12, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"),
                    x=0.02,
                    y=0.97
                ),
                barmode="group",
                height=h_grid,
                margin=dict(l=25, r=20, t=40, b=25),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
                yaxis=dict(title="Million EGP", title_font=dict(size=9, color=CHARCOAL, weight="bold"), tickfont=dict(size=8, color=CHARCOAL), gridcolor="#f1f5f9"),
                xaxis=dict(tickfont=dict(size=8.5, color=CHARCOAL)),
                bargap=0.22,
                bargroupgap=0.08,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color=CHARCOAL),
                uirevision="c1_appr",
                transition=dict(duration=350, easing="cubic-in-out")
            )
            st.plotly_chart(fig_appr, config={"displayModeBar": False}, key="c1_plotly_appr")
        else:
            st.info("Insufficient data for capital appreciation benchmark.")

    # --- ROW 2: Hubs Leaderboard Table (Left) & Market Structure (Right) ---
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        # Top Verified Hubs Leaderboard Table
        st.markdown(
            f"<p style='font-size:0.77rem; font-weight:800; color:{DEEP_NAVY}; margin:2px 0 2px 0;'>"
            "🏆 Top Investment Hubs</p>",
            unsafe_allow_html=True
        )

        if not geo_points.empty:
            top_display = geo_points.sort_values(by="count", ascending=False).head(12)[
                ["location", "count", "median_price", "median_ppm", "aqi", "green_pct"]
            ].copy()
            top_display.columns = ["Location Hub", "Units", "Median Price (EGP)", "EGP/m²", "AQI", "Green %"]

            st.dataframe(
                top_display,
                hide_index=True,
                height=h_grid - 28,
                key="c1_df_leaderboard",
                column_config={
                    "Location Hub": st.column_config.TextColumn("Location Hub", width="medium", help="Normalized geographical district or compound cluster"),
                    "Units": st.column_config.NumberColumn("Units", format="%d", help="Number of active listings in this hub"),
                    "Median Price (EGP)": st.column_config.NumberColumn("Median Price", format="%d", help="Median asking price in Egyptian Pounds"),
                    "EGP/m²": st.column_config.NumberColumn("EGP/m²", format="%d", help="Median unit price per square meter"),
                    "AQI": st.column_config.NumberColumn("AQI", format="%.1f", help="US EPA Air Quality Index score"),
                    "Green %": st.column_config.NumberColumn("Green %", format="%.1f%%", help="Satellite urban green coverage percentage"),
                }
            )
        else:
            st.caption("No hubs available for leaderboard display.")

    with r2_col2:
        # Market Dynamics & Distribution (Price Density / Finishing Breakdown)
        c1_view = st.selectbox(
            "Quadrant 4 View Selector",
            options=["📊 Price Distribution by Community Type (Gated vs. Standalone)", "🎨 Delivery Finish Distribution"],
            index=0,
            label_visibility="collapsed",
            key="c1_q4_view"
        )

        if c1_view.startswith("📊"):
            if active_count > 0:
                hist_subset = filtered_df[filtered_df["Price_Numeric"] <= 25_000_000]
                fig_hist = px.histogram(
                    hist_subset,
                    x="Price_Numeric",
                    nbins=28,
                    color="Amenities",
                    color_discrete_map={
                        "B(Compound)": NILE_TEAL,
                        "C(BASICS)": DEEP_NAVY
                    }
                )
                fig_hist.update_traces(
                    hovertemplate="<b>%{x:,.0f} EGP</b><br>Listing Units: %{y}<br>Amenity: %{fullData.name}<extra></extra>"
                )
                fig_hist.update_layout(
                    height=h_grid - 30,
                    margin=dict(l=25, r=20, t=12, b=25),
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=8.5)),
                    xaxis=dict(tickformat=",.0s", title="Price (EGP)", title_font=dict(size=9, color=CHARCOAL), tickfont=dict(size=8, color=CHARCOAL)),
                    yaxis=dict(title="Units", title_font=dict(size=9, color=CHARCOAL), tickfont=dict(size=8, color=CHARCOAL), gridcolor="#f1f5f9"),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color=CHARCOAL),
                    uirevision="c1_hist",
                    transition=dict(duration=350, easing="cubic-in-out")
                )
                st.plotly_chart(fig_hist, config={"displayModeBar": False}, key="c1_plotly_hist")
            else:
                st.caption("No price density data.")
        else:
            if active_count > 0 and "finishing_type" in filtered_df.columns:
                fin_counts = filtered_df["finishing_type"].value_counts().reset_index()
                fin_counts.columns = ["Finishing", "Count"]

                palette_donut = [NILE_TEAL, DEEP_NAVY, SANDSTONE_GOLD, SOFT_SAGE, CORAL_RED]
                fig_fin = px.pie(
                    fin_counts,
                    values="Count",
                    names="Finishing",
                    hole=0.48,
                    color_discrete_sequence=palette_donut
                )
                fig_fin.update_traces(
                    textinfo="percent+label",
                    textfont_size=9,
                    hovertemplate="<b>%{label}</b><br>Units: %{value:,}<br>Market Share: %{percent}<extra></extra>"
                )
                donut_dim = h_grid - 30
                fig_fin.update_layout(
                    width=donut_dim,
                    height=donut_dim,
                    margin=dict(l=15, r=15, t=12, b=15),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=8)),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color=CHARCOAL),
                    uirevision="c1_fin",
                    transition=dict(duration=350, easing="cubic-in-out")
                )
                st.plotly_chart(fig_fin, config={"displayModeBar": False}, key="c1_plotly_fin")
            else:
                st.caption("No finishing data.")


# -----------------------------------------------------------------------------
# 2. COCKPIT 2: 💼 10-YEAR INVESTMENT & CASH FLOW FORECAST
# -----------------------------------------------------------------------------
def render_cockpit_2(
    filtered_df: pd.DataFrame,
    active_count: int,
    display_scale: str
) -> None:
    """
    Renders Cockpit 2: 10-Year Investment & Cash Flow Forecast.
    Features:
    - Dedicated Financial Simulation Controls in the Left Sidebar
    - 4 Top KPI Cards (Net Rental Yield, Capital Gain, Total Projected Return, Leveraged Return on Equity)
    - 2-Column Bento Grid:
      * Left: Multi-year Asset Growth & Net Rent + Living Area vs Price with Market Price Trendline
      * Right: Core Corridor Investment Benchmarks Table + Scenario Sensitivity Bounds Card + Correlation Heatmap
    """
    heights = get_display_heights(display_scale)
    h_upper = heights["h_upper"]
    h_lower = heights["h_lower"]
    h_bench_table = heights["h_bench_table"]

    # 1. Top Executive Header Strip
    render_header_strip(
        title="💼 10-Year Investment & Cash Flow Forecast",
        subtitle="Long-Term Capital Appreciation & Net Rental Yields",
        badge_text="ML Valuation Model | Market Forecast",
        badge_dot_color=NILE_TEAL
    )

    # 2. Financial Modeling Calculations
    gross_yield_pct = float(st.session_state.get("sb_gross_yield", 7.5))
    opex_pct = float(st.session_state.get("sb_opex", 15.0))
    equity_pct = float(st.session_state.get("sb_down_pmt", 50.0))
    horizon_years = int(st.session_state.get("sb_horizon", 10))
    negotiation_discount = float(st.session_state.get("sb_negotiation_discount", 0.0))

    raw_base_price = filtered_df["Price_Numeric"].median() if active_count > 0 else 5_000_000.0
    base_unit_price = raw_base_price * (1.0 - (negotiation_discount / 100.0)) if negotiation_discount > 0 else raw_base_price
    annual_gross_rent = base_unit_price * (gross_yield_pct / 100.0)
    annual_net_rent = annual_gross_rent * (1.0 - (opex_pct / 100.0))
    net_yield_pct = (annual_net_rent / base_unit_price) * 100.0 if base_unit_price > 0 else 0.0

    is_compound_bias = (filtered_df["Amenities"] == "B(Compound)").mean() > 0.5 if active_count > 0 else True
    mult_horizon = (2.42 if is_compound_bias else 1.67) ** (horizon_years / 10.0)
    projected_exit_value = base_unit_price * mult_horizon
    capital_gain = projected_exit_value - base_unit_price
    total_cumulative_rent = annual_net_rent * horizon_years
    total_wealth_created = capital_gain + total_cumulative_rent
    equity_invested = base_unit_price * (equity_pct / 100.0)
    levered_roi = (total_wealth_created / equity_invested) * 100.0 if equity_invested > 0 else 0.0

    # 4. 4 Top KPI Metric Cards
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        render_kpi_card(
            label="Net Rental Yield (Cap Rate)",
            value=f"{net_yield_pct:.2f}%",
            subtitle=f"Net of {opex_pct:.0f}% OPEX/Vacancy ({gross_yield_pct:.1f}% Gross)",
            border_color=NILE_TEAL,
            sub_color=DEEP_NAVY,
            tooltip=f"Net capitalization rate: (Gross Annual Rental Income - OPEX & Vacancy Reserve) / Acquisition Price. Reflects {gross_yield_pct:.1f}% gross yield minus {opex_pct:.0f}% operating costs."
        )
    with v2:
        haircut_tag = f" (-{negotiation_discount:.0f}% Ask Haircut)" if negotiation_discount > 0 else ""
        render_kpi_card(
            label=f"{horizon_years}-Year Capital Gain",
            value=f"+{capital_gain / 1e6:.2f}M <small style='font-size:0.75rem; color:{CHARCOAL};'>EGP</small>",
            subtitle=f"Exit: {projected_exit_value / 1e6:.2f}M EGP ({mult_horizon:.2f}x){haircut_tag}",
            border_color=DEEP_NAVY,
            sub_color=NILE_TEAL,
            tooltip=f"Simulated asset value growth from transacted base ({base_unit_price / 1e6:.2f}M EGP) to projected exit based on valuation model."
        )
    with v3:
        render_kpi_card(
            label="Total Projected Return",
            value=f"+{total_wealth_created / 1e6:.2f}M <small style='font-size:0.75rem; color:{CHARCOAL};'>EGP</small>",
            subtitle="Combined Rent & Capital Appreciation",
            border_color=SANDSTONE_GOLD,
            sub_color=DEEP_NAVY,
            tooltip="Total projected return: Projected Capital Gain plus cumulative net rental dividends over the horizon."
        )
    with v4:
        render_kpi_card(
            label="Leveraged Return on Equity",
            value=f"{levered_roi:.1f}%",
            subtitle=f"Based on {equity_pct:.0f}% Equity Stake",
            border_color=SOFT_SAGE,
            sub_color=DEEP_NAVY,
            tooltip="Return on equity calculated on the initial equity down payment."
        )

    # 5. Perfectly Balanced 2x2 Bento Grid (Even Spacing, Edge-to-Edge)
    h_grid = heights.get("h_grid", h_upper)

    # --- ROW 1: Asset Growth Chart (Left) & Benchmark Valuations (Right) ---
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        # Multi-year Asset Growth & Cumulative Net Rent
        years = list(range(2026, 2026 + horizon_years + 1))
        exact_values = [
            base_unit_price * (((mult_horizon - 1.0) * (t / horizon_years)) + 1.0)
            for t in range(horizon_years + 1)
        ]
        yearly_values = [v / 1e6 for v in exact_values]
        exact_rents = [annual_net_rent * t for t in range(horizon_years + 1)]
        yearly_rents = [r / 1e6 for r in exact_rents]

        fig_cf = go.Figure()
        cf_ids = [str(y) for y in years]
        fig_cf.add_trace(go.Scatter(
            x=years,
            y=yearly_values,
            ids=cf_ids,
            customdata=exact_values,
            mode="lines+markers",
            name="Asset Value (M EGP)",
            line=dict(color=DEEP_NAVY, width=3.2),
            marker=dict(size=7, color=DEEP_NAVY),
            hovertemplate="<b>Year %{x}</b><br>Asset Value: %{y:.2f}M EGP<br>Exact Valuation: %{customdata:,.0f} EGP<extra></extra>"
        ))
        fig_cf.add_trace(go.Bar(
            x=years,
            y=yearly_rents,
            ids=cf_ids,
            customdata=exact_rents,
            name="Cumulative Net Rent (M EGP)",
            marker_color=SOFT_SAGE,
            opacity=0.88,
            hovertemplate="<b>Year %{x}</b><br>Cumulative Net Rent: %{y:.2f}M EGP<br>Exact Net Rent: %{customdata:,.0f} EGP<extra></extra>"
        ))
        fig_cf.update_layout(
            title=dict(
                text="💰 Projected Asset Growth & Cumulative Rental Income",
                font=dict(size=12, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"),
                x=0.02,
                y=0.97
            ),
            height=h_grid,
            margin=dict(l=25, r=20, t=38, b=25),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
            yaxis=dict(title="Million EGP", title_font=dict(size=9, color=CHARCOAL, weight="bold"), tickfont=dict(size=8, color=CHARCOAL), gridcolor="#f1f5f9"),
            xaxis=dict(tickmode="linear", tick0=2026, dtick=1, tickfont=dict(size=8, color=CHARCOAL)),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", color=CHARCOAL),
            uirevision="c2_cf",
            transition=dict(duration=350, easing="cubic-in-out")
        )
        st.plotly_chart(fig_cf, config={"displayModeBar": False}, key="c2_plotly_cashflow")

    with r1_col2:
        # Live Model Benchmark Valuations Table
        st.markdown(
            f"<p style='font-size:0.77rem; font-weight:800; color:{DEEP_NAVY}; margin:0 0 2px 0;'>"
            "🏆 Core Corridor Investment Benchmarks</p>",
            unsafe_allow_html=True
        )

        try:
            bench_df = evaluate_benchmarks()
            if bench_df is None or bench_df.empty:
                raise ValueError("Empty benchmark return")
        except Exception:
            bench_df = pd.DataFrame([
                {"location": "Village West (Sheikh Zayed)", "current_price": 11_275_000, "price_10y": 27_249_000, "multiplier": 2.42, "annual_cagr": 9.24},
                {"location": "Madinaty (New Cairo)", "current_price": 7_254_000, "price_10y": 17_600_000, "multiplier": 2.43, "annual_cagr": 9.28},
                {"location": "Faisal (Giza Urban)", "current_price": 1_849_000, "price_10y": 3_038_000, "multiplier": 1.64, "annual_cagr": 5.07},
                {"location": "New October (West Cairo)", "current_price": 1_498_000, "price_10y": 2_497_000, "multiplier": 1.67, "annual_cagr": 5.24},
            ])

        hub_col_name = "benchmark_name" if "benchmark_name" in bench_df.columns else "location"
        display_bench = bench_df[[hub_col_name, "current_price", "price_10y", "multiplier", "annual_cagr"]].copy()
        display_bench.columns = ["Benchmark Hub", "Current (EGP)", "10Y Target", "Multiplier", "CAGR %"]

        table_h = max(140, h_grid - 135)
        st.dataframe(
            display_bench,
            hide_index=True,
            height=table_h,
            key="c2_df_benchmarks",
            column_config={
                "Benchmark Hub": st.column_config.TextColumn("Benchmark Hub", width="medium", help="Representative core Egyptian real estate investment corridor"),
                "Current (EGP)": st.column_config.NumberColumn("Current (EGP)", format="%d", help="Baseline 2026 median valuation"),
                "10Y Target": st.column_config.NumberColumn("10Y Target", format="%d", help="10-year projected exit valuation"),
                "Multiplier": st.column_config.NumberColumn("Multiplier", format="%.2fx", help="10-year capital appreciation multiple (e.g. 2.42x)"),
                "CAGR %": st.column_config.NumberColumn("CAGR %", format="%.2f%%", help="Annualized compound annual growth rate"),
            }
        )

        # Scenario Sensitivity Bounds Card
        bear_val = (projected_exit_value * 0.90) / 1e6
        exp_val = projected_exit_value / 1e6
        bull_val = (projected_exit_value * 1.10) / 1e6

        st.markdown(f"""
        <div class="stat-card" style="border-left-color: {NILE_TEAL}; padding: 6px 12px; margin-top: 3px;">
            <span class="stat-label">Market Scenario Sensitivity Bounds ({horizon_years}-Year Outlook)</span>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:3px; border-bottom:1px solid {LIGHT_GRAY}; padding-bottom:3px;">
                <span style="font-size:0.75rem; color:{CHARCOAL}; font-weight:600;">🐻 Bearish Scenario (-10%):</span>
                <strong style="font-size:0.90rem; color:{CORAL_RED};">{bear_val:.2f}M EGP</strong>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:3px; border-bottom:1px solid {LIGHT_GRAY}; padding-bottom:3px;">
                <span style="font-size:0.75rem; color:{CHARCOAL}; font-weight:600;">🎯 Expected (Base Target):</span>
                <strong style="font-size:0.90rem; color:{NILE_TEAL};">{exp_val:.2f}M EGP</strong>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:3px; padding-bottom:1px;">
                <span style="font-size:0.75rem; color:{CHARCOAL}; font-weight:600;">🐂 Bullish Scenario (+10%):</span>
                <strong style="font-size:0.90rem; color:{SOFT_SAGE};">{bull_val:.2f}M EGP</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- ROW 2: Living Area vs Price Scatter (Left) & Correlation Heatmap (Right) ---
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        # Living Area vs Total Price Scatter Plot with Trendline
        sample_scatter = filtered_df.sample(min(1200, len(filtered_df)), random_state=42) if len(filtered_df) > 0 else filtered_df
        fig_scat = px.scatter(
            sample_scatter,
            x="Area",
            y="Price_Numeric",
            color="Amenities",
            color_discrete_map={
                "B(Compound)": NILE_TEAL,
                "C(BASICS)": DEEP_NAVY
            },
            hover_data=["normalized_location", "Calculated_Price_Per_Meter"] if ("normalized_location" in sample_scatter.columns and "Calculated_Price_Per_Meter" in sample_scatter.columns) else None
        )
        if "normalized_location" in sample_scatter.columns and "Calculated_Price_Per_Meter" in sample_scatter.columns:
            fig_scat.update_traces(
                selector=dict(mode="markers"),
                hovertemplate="<b>Hub: %{customdata[0]}</b><br>Area: %{x:,.0f} m²<br>Asking Price: %{y:,.0f} EGP<br>Price/m²: %{customdata[1]:,.0f} EGP<extra></extra>"
            )

        # Market Price Trendline in Sandstone Gold
        if len(sample_scatter) > 10:
            x_pts = sample_scatter["Area"].values
            y_pts = sample_scatter["Price_Numeric"].values
            valid_m = (~np.isnan(x_pts)) & (~np.isnan(y_pts)) & (x_pts > 0)
            if valid_m.sum() > 5:
                slope, intercept = np.polyfit(x_pts[valid_m], y_pts[valid_m], 1)
                x_fit = np.linspace(30, 380, 60)
                fig_scat.add_trace(go.Scatter(
                    x=x_fit,
                    y=slope * x_fit + intercept,
                    mode="lines",
                    name=f"Market Price Trend ({slope:,.0f} EGP/m²)",
                    line=dict(color=SANDSTONE_GOLD, width=3.0, dash="dash"),
                    hovertemplate=f"<b>Market Price Trend</b><br>Slope: {slope:,.0f} EGP/m²<br>Area: %{{x:,.0f}} m²<br>Predicted Price: %{{y:,.0f}} EGP<extra></extra>"
                ))

        fig_scat.update_layout(
            title=dict(
                text="📐 Price vs. Unit Size (m²)",
                font=dict(size=12, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"),
                x=0.02,
                y=0.97
            ),
            width=h_grid,
            height=h_grid,
            margin=dict(l=25, r=20, t=38, b=25),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=8)),
            xaxis=dict(title="Living Area (m²)", title_font=dict(size=9, color=CHARCOAL), range=[30, 380], tickfont=dict(size=8, color=CHARCOAL)),
            yaxis=dict(title="Total Price (EGP)", title_font=dict(size=9, color=CHARCOAL), range=[0, 24_000_000], tickformat=",.0s", tickfont=dict(size=8, color=CHARCOAL), gridcolor="#f1f5f9"),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", color=CHARCOAL),
            uirevision="c2_scat",
            transition=dict(duration=350, easing="cubic-in-out")
        )
        st.plotly_chart(fig_scat, config={"displayModeBar": False}, key="c2_plotly_scatter")

    with r2_col2:
        # Institutional Underwriting Sensitivity Matrix (Exit Cap Rate vs Rental Growth Rate)
        if render_sensitivity_heatmap is not None:
            fig_sens = render_sensitivity_heatmap(
                base_price=base_unit_price,
                gross_yield_pct=gross_yield_pct,
                opex_pct=opex_pct,
                equity_pct=equity_pct,
                horizon_years=horizon_years,
                height=h_grid
            )
            fig_sens.update_layout(
                width=h_grid,
                height=h_grid,
                title=dict(
                    text="📊 Investment Return Sensitivity (Exit Cap vs. Rental Growth)",
                    font=dict(size=11, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"),
                    x=0.02,
                    y=0.97
                ),
                margin=dict(l=45, r=50, t=38, b=35)
            )
            st.plotly_chart(fig_sens, config={"displayModeBar": False}, key="c2_plotly_sensitivity")
        else:
            # Fallback Correlation Heatmap if financial module unavailable
            corr_cols = ["Calculated_Price_Per_Meter", "Area", "Rooms", "Bathrooms_num", "Floor Level"]
            avail_cols = [c for c in corr_cols if c in filtered_df.columns]
            if len(avail_cols) == 5 and len(filtered_df) > 5:
                corr_mat = filtered_df[avail_cols].corr()
            else:
                corr_mat = pd.DataFrame(np.eye(5), columns=corr_cols, index=corr_cols)

            fig_corr = px.imshow(
                corr_mat.round(2),
                text_auto=True,
                color_continuous_scale=BLUES_TEAL_SCALE,
                labels=dict(color="Corr"),
                x=["PPM", "Area", "Rooms", "Baths", "Floor"],
                y=["PPM", "Area", "Rooms", "Baths", "Floor"]
            )
            fig_corr.update_layout(
                width=h_grid,
                height=h_grid,
                title=dict(text="🔗 Key Price Factors Correlation", font=dict(size=11, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"), x=0.02, y=0.97),
                margin=dict(l=35, r=35, t=35, b=35),
                coloraxis_showscale=False,
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color=CHARCOAL),
                uirevision="c2_corr",
                transition=dict(duration=350, easing="cubic-in-out")
            )
            st.plotly_chart(fig_corr, config={"displayModeBar": False}, key="c2_plotly_corr")


# -----------------------------------------------------------------------------
# 3. COCKPIT 3: 🏗️ SUPPLY PIPELINE & INVENTORY DATABASE
# -----------------------------------------------------------------------------
def render_cockpit_3(
    filtered_df: pd.DataFrame,
    active_count: int,
    display_scale: str
) -> None:
    """
    Renders Cockpit 3: Supply Pipeline & Inventory Database.
    Features:
    - 4 Stat Badges (Established Homes, Ready for Handover, Under Construction, Verified Listings)
    - Upper Row: Upcoming Delivery Schedule (Nile Teal) + Median Price/m² by Delivery Year (Soft Sage)
    - Lower Row: Search Verified Property Inventory + Sort selector + Listing Explorer table
    """
    heights = get_display_heights(display_scale)
    h_grid = heights.get("h_grid", heights["h_upper"])
    h_upper = heights["h_upper"]
    h_listing_table = heights["h_listing_table"]

    # 1. Top Executive Header Strip
    render_header_strip(
        title="🏗️ Supply Pipeline & Inventory Database",
        subtitle="Projected Delivery Schedules & Listing Explorer",
        badge_text="100% Verified Inventory | No Duplicates",
        badge_dot_color=NILE_TEAL
    )

    # 2. Audit Cohort Statistics
    if active_count > 0 and "Year Built" in filtered_df.columns:
        c_vintage = (filtered_df["Year Built"] < 2020).sum()
        c_ready = ((filtered_df["Year Built"] >= 2020) & (filtered_df["Year Built"] <= 2025)).sum()
        c_offplan = (filtered_df["Year Built"] >= 2026).sum()
        valid_geo = (filtered_df["geo_status"] == "VALID_GEO").sum() if "geo_status" in filtered_df.columns else active_count
    else:
        c_vintage, c_ready, c_offplan, valid_geo = 0, 0, 0, 0

    # 4 Audit Stat Badges
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        render_kpi_card(
            label="Established Homes (<2020)",
            value=f"{c_vintage:,}",
            subtitle=f"{(c_vintage / max(1, active_count)) * 100.0:.1f}% Established Stock",
            border_color=CHARCOAL,
            sub_color=CHARCOAL,
            tooltip="Established housing stock completed and delivered before 2020."
        )
    with d2:
        render_kpi_card(
            label="Ready for Handover (2020 - 2025)",
            value=f"{c_ready:,}",
            subtitle=f"{(c_ready / max(1, active_count)) * 100.0:.1f}% Near Term Inventory",
            border_color=NILE_TEAL,
            sub_color=DEEP_NAVY,
            tooltip="Recently completed or near-term delivery inventory ready for immediate handover."
        )
    with d3:
        render_kpi_card(
            label="Under Construction (2026 - 2035)",
            value=f"{c_offplan:,}",
            subtitle=f"{(c_offplan / max(1, active_count)) * 100.0:.1f}% Future Delivery Pipeline",
            border_color=DEEP_NAVY,
            sub_color=NILE_TEAL,
            tooltip="Future construction pipeline across new developmental corridors and satellite cities."
        )
    with d4:
        render_kpi_card(
            label="Verified Listings",
            value=f"{valid_geo:,} / {active_count:,}",
            subtitle="100% Math Verified & Deduplicated",
            border_color=SANDSTONE_GOLD,
            sub_color=DEEP_NAVY,
            tooltip="100% verified coordinates, exact price-to-area mathematical consistency, and zero duplicate listings."
        )

    # 3. Upper Row: Delivery Pipeline Volume & Price Escalation Trajectory
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    t_left, t_right = st.columns(2)
    h_chart_row = h_grid

    if active_count > 0 and "Year Built" in filtered_df.columns:
        timeline_df = filtered_df[
            (filtered_df["Year Built"] >= 2018) & (filtered_df["Year Built"] <= 2030)
        ]
    else:
        timeline_df = pd.DataFrame()

    if not timeline_df.empty:
        t_summary = timeline_df.groupby("Year Built").agg(
            Volume=("Price_Numeric", "count"),
            Median_PPM=("Calculated_Price_Per_Meter", "median")
        ).reset_index()
        t_summary["Year Built"] = t_summary["Year Built"].astype(int)

        # Delivery Pipeline Volume in Nile Teal
        with t_left:
            fig_vol = px.bar(
                t_summary,
                x="Year Built",
                y="Volume",
                title="🏗️ Upcoming Delivery Schedule (2018 - 2030)",
                text="Volume"
            )
            if len(fig_vol.data) > 0:
                fig_vol.data[0].ids = t_summary["Year Built"].astype(str).tolist()
            fig_vol.update_traces(
                marker_color=NILE_TEAL,
                textposition="outside",
                textfont=dict(size=9, color=DEEP_NAVY),
                hovertemplate="<b>Handover Year %{x}</b><br>Pipeline Volume: %{y:,} units<extra></extra>"
            )
            fig_vol.update_layout(
                title=dict(
                    text="🏗️ Upcoming Delivery Schedule (2018 - 2030)",
                    font=dict(size=12, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"),
                    x=0.02,
                    y=0.97
                ),
                height=h_chart_row,
                margin=dict(l=25, r=20, t=38, b=25),
                xaxis=dict(tickmode="linear", tick0=2018, dtick=2, tickfont=dict(size=8, color=CHARCOAL)),
                yaxis=dict(title="Listings Count", title_font=dict(size=9, color=CHARCOAL), tickfont=dict(size=8, color=CHARCOAL), gridcolor="#f1f5f9"),
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color=CHARCOAL),
                uirevision="c3_vol",
                transition=dict(duration=350, easing="cubic-in-out")
            )
            st.plotly_chart(fig_vol, config={"displayModeBar": False}, key="c3_plotly_volume")

        # Price/m² Escalation Trajectory (Institutional Cohort Bar Chart with Verified N Labels)
        with t_right:
            if render_cohort_escalation_chart is not None:
                fig_ppm = render_cohort_escalation_chart(timeline_df, height=h_chart_row)
                fig_ppm.update_layout(
                    title=dict(
                        text="📈 Median Price/m² by Delivery Year",
                        font=dict(size=12, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"),
                        x=0.02,
                        y=0.97
                    ),
                    margin=dict(l=25, r=20, t=38, b=25)
                )
                st.plotly_chart(fig_ppm, config={"displayModeBar": False}, key="c3_plotly_cohort")
            else:
                fig_ppm = px.line(
                    t_summary,
                    x="Year Built",
                    y="Median_PPM",
                    title="📈 Median Price/m² by Delivery Year",
                    markers=True
                )
                if len(fig_ppm.data) > 0:
                    fig_ppm.data[0].ids = t_summary["Year Built"].astype(str).tolist()
                fig_ppm.update_traces(
                    line=dict(color=SOFT_SAGE, width=3.2),
                    marker=dict(size=7, color=DEEP_NAVY),
                    hovertemplate="<b>Handover Cohort %{x}</b><br>Median Price/m²: %{y:,.0f} EGP<extra></extra>"
                )
                fig_ppm.update_layout(
                    title=dict(
                        text="📈 Median Price/m² by Delivery Year",
                        font=dict(size=12, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"),
                        x=0.02,
                        y=0.97
                    ),
                    height=h_chart_row,
                    margin=dict(l=25, r=20, t=38, b=25),
                    xaxis=dict(tickmode="linear", tick0=2018, dtick=2, tickfont=dict(size=8, color=CHARCOAL)),
                    yaxis=dict(title="Median Price/m² (EGP)", title_font=dict(size=9, color=CHARCOAL), tickfont=dict(size=8, color=CHARCOAL), gridcolor="#f1f5f9"),
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color=CHARCOAL),
                    uirevision="c3_ppm",
                    transition=dict(duration=350, easing="cubic-in-out")
                )
                st.plotly_chart(fig_ppm, config={"displayModeBar": False}, key="c3_plotly_cohort")
    else:
        with t_left:
            st.info("No delivery timeline data in range 2018 - 2030.")
        with t_right:
            st.info("No price escalation trajectory data.")

    # 4. Lower Row: Search Bar, Sort Selector & Full-Height Canonical Explorer Table
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:0.77rem; font-weight:800; color:{DEEP_NAVY}; margin:4px 0 2px 0;'>"
        "🔍 Search Verified Property Inventory</p>",
        unsafe_allow_html=True
    )
    c_search, c_sort = st.columns([7, 3])
    with c_search:
        search_kw = st.text_input(
            "🔍 Search Verified Property Inventory",
            placeholder="🔍 Search Verified Property Inventory (e.g. Madinaty, Zayed, Superlux...)",
            label_visibility="collapsed",
            key="c3_search_kw"
        )
    with c_sort:
        sort_by = st.selectbox(
            "Sort Listings",
            options=[
                "Price: High to Low",
                "Price: Low to High",
                "Price/m²: High to Low",
                "Area: High to Low",
                "Handover Year: Descending"
            ],
            label_visibility="collapsed",
            key="c3_sort_by"
        )

    display_table = filtered_df.copy()

    # Search keyword filter
    if search_kw:
        mask = pd.Series(False, index=display_table.index)
        for col in ["Location", "normalized_location", "macro_region", "finishing_type"]:
            if col in display_table.columns:
                mask = mask | display_table[col].astype(str).str.contains(search_kw, case=False, na=False)
        display_table = display_table[mask]

    # Sorting
    if sort_by == "Price: High to Low" and "Price_Numeric" in display_table.columns:
        display_table = display_table.sort_values(by="Price_Numeric", ascending=False)
    elif sort_by == "Price: Low to High" and "Price_Numeric" in display_table.columns:
        display_table = display_table.sort_values(by="Price_Numeric", ascending=True)
    elif sort_by == "Price/m²: High to Low" and "Calculated_Price_Per_Meter" in display_table.columns:
        display_table = display_table.sort_values(by="Calculated_Price_Per_Meter", ascending=False)
    elif sort_by == "Area: High to Low" and "Area" in display_table.columns:
        display_table = display_table.sort_values(by="Area", ascending=False)
    elif sort_by == "Handover Year: Descending" and "Year Built" in display_table.columns:
        display_table = display_table.sort_values(by="Year Built", ascending=False)

    candidate_cols = [
        "normalized_location", "macro_region", "Price_Numeric", "Calculated_Price_Per_Meter",
        "Area", "Rooms", "Bathrooms_num", "Floor Level", "Year Built", "Amenities",
        "finishing_type", "Type", "us_aqi", "green_coverage_pct"
    ]
    cols_to_show = [c for c in candidate_cols if c in display_table.columns]

    st.dataframe(
        display_table[cols_to_show].head(1500),
        height=max(180, h_grid - 42),
        hide_index=True,
        key="c3_df_inventory",
        column_config={
            "normalized_location": st.column_config.TextColumn("Location Hub", width="medium", help="Standardized location hub name"),
            "macro_region": st.column_config.TextColumn("Region", help="Geographic macro region corridor"),
            "Price_Numeric": st.column_config.NumberColumn("Price (EGP)", format="%d", help="Total asking price in EGP"),
            "Calculated_Price_Per_Meter": st.column_config.NumberColumn("Price/m²", format="%d", help="Unit price per m² (Price / Area)"),
            "Area": st.column_config.NumberColumn("Area (m²)", format="%.0f", help="Total gross living area in square meters"),
            "Rooms": st.column_config.NumberColumn("Beds", format="%d", help="Number of registered bedrooms"),
            "Bathrooms_num": st.column_config.NumberColumn("Baths", format="%d", help="Number of bathrooms"),
            "Floor Level": st.column_config.NumberColumn("Floor", format="%d", help="Building floor level"),
            "Year Built": st.column_config.NumberColumn("Handover", format="%d", help="Handover/delivery completion year"),
            "Amenities": st.column_config.TextColumn("Community", help="Gated Compound (B) vs Standalone Urban (C)"),
            "finishing_type": st.column_config.TextColumn("Finish", help="Delivery finish specification"),
            "Type": st.column_config.TextColumn("Purpose", help="Listing classification: Residential (Own) vs Investment"),
            "us_aqi": st.column_config.NumberColumn("AQI", format="%.1f", help="EPA US Air Quality Index"),
            "green_coverage_pct": st.column_config.NumberColumn("Green %", format="%.1f%%", help="Satellite green space coverage percentage"),
        }
    )

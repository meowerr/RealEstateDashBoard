# -*- coding: utf-8 -*-
"""
Egypt Real Estate Intelligence Platform (v1) - Geospatial & Cohort Analytics Module
File: geo_analytics.py

Solves institutional visualization challenges:
1. Spatial overplotting in dense hubs (e.g. New Cairo, 5th Settlement, Sheikh Zayed)
   via regular hexagonal/square spatial grid binning (~1.5km cells) and continuous
   dynamic density heatmaps without marker collision.
2. Cohort whipsawing/erratic trends after 2026 caused by thin off-plan sample sizes (N)
   via volume-verified bar visualization with clear sample count annotations.

Egyptian Executive Palette:
- Deep Navy:      #1B2A4A (Primary structural borders, headers, baseline off-plan)
- Nile Teal:      #2A9D8F (Primary brand metric, ready delivery 2020-2025)
- Sandstone Gold: #E9B44C (Egyptian accent, far-future off-plan 2029+)
- Coral Red:      #E85D4C (High-density valuation peaks, warnings)
- Soft Sage:      #8FB996 (Baseline density, positive environmental quality)
- Off-White:      #F7F5F2 (Surface backgrounds, canvas contrast)
- Light Gray:     #D9D9D6 (Dividers, borders, subtle axis grids)
- Charcoal:       #333333 (Crisp typography, vintage <2020 cohort)
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. COLOR TOKENS & GEO COLOR SCALES
# -----------------------------------------------------------------------------
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
    )
except ImportError:
    DEEP_NAVY = "#1B2A4A"
    NILE_TEAL = "#2A9D8F"
    SANDSTONE_GOLD = "#E9B44C"
    CORAL_RED = "#E85D4C"
    SOFT_SAGE = "#8FB996"
    OFF_WHITE = "#F7F5F2"
    LIGHT_GRAY = "#D9D9D6"
    CHARCOAL = "#333333"

# Smooth 4-stop Egyptian Geo Continuous Color Scale:
# Soft Sage (#8FB996) -> Nile Teal (#2A9D8F) -> Sandstone Gold (#E9B44C) -> Coral Red (#E85D4C)
EGYPTIAN_GEO_HEATMAP_SCALE = [
    [0.0, SOFT_SAGE],        # Baseline valuation / low density
    [0.33, NILE_TEAL],       # Institutional prime core
    [0.66, SANDSTONE_GOLD],   # High-demand appreciation tier
    [1.0, CORAL_RED],        # Peak pricing & density concentration
]

DEFAULT_MAP_CENTER = dict(lat=30.044, lon=31.335)
DEFAULT_MAP_ZOOM = 9.3


# -----------------------------------------------------------------------------
# 2. SPATIAL CLUSTERING & BINNING ENGINE
# -----------------------------------------------------------------------------
def generate_clustered_hubs(
    df: Optional[pd.DataFrame] = None,
    grid_deg: float = 0.015
) -> pd.DataFrame:
    """
    Spatially bins geographic coordinates into ~1.5km grid cells (grid_deg ≈ 0.015°),
    collapsing thousands of severely overlapping points into coherent, non-overlapping
    institutional hubs with comprehensive statistical aggregations.

    Args:
        df: Input listing DataFrame containing coordinates and listing attributes.
        grid_deg: Spatial binning resolution in degrees (0.015 deg ≈ 1.66 km at Cairo lat).

    Returns:
        pd.DataFrame: Aggregated spatial clusters with columns:
            ['location', 'primary_hub', 'hub_name', 'lat', 'lon', 'latitude', 'longitude',
             'count', 'unit_count', 'median_price', 'median_ppm', 'Calculated_Price_Per_Meter',
             'aqi', 'green_pct', 'macro_region', 'centroid_lat', 'centroid_lon']
    """
    expected_cols = [
        "location", "primary_hub", "hub_name", "lat", "lon",
        "latitude", "longitude", "count", "unit_count",
        "median_price", "median_ppm", "Calculated_Price_Per_Meter",
        "aqi", "green_pct", "macro_region", "centroid_lat", "centroid_lon"
    ]

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=expected_cols)

    # 1. Resolve coordinate columns
    lat_col = next((c for c in ["lat", "latitude", "Lat", "Latitude"] if c in df.columns), None)
    lon_col = next((c for c in ["lon", "longitude", "Lon", "Longitude", "lng"] if c in df.columns), None)

    if not lat_col or not lon_col:
        return pd.DataFrame(columns=expected_cols)

    # 2. Filter to valid, finite coordinates
    valid_mask = df[lat_col].notnull() & df[lon_col].notnull() & np.isfinite(df[lat_col]) & np.isfinite(df[lon_col])
    if "geo_status" in df.columns:
        # Prefer VALID_GEO when available, but fallback if no VALID_GEO exist
        valid_geo_mask = (df["geo_status"] == "VALID_GEO") & valid_mask
        if valid_geo_mask.any():
            valid_mask = valid_geo_mask

    valid_df = df[valid_mask].copy()
    if valid_df.empty:
        return pd.DataFrame(columns=expected_cols)

    # 3. Resolve metric columns
    price_col = next((c for c in ["Price_Numeric", "price_numeric", "Price", "price"] if c in valid_df.columns), None)
    ppm_col = next((c for c in ["Calculated_Price_Per_Meter", "price_per_meter", "Price_Per_Meter", "median_ppm", "ppm"] if c in valid_df.columns), None)
    loc_col = next((c for c in ["normalized_location", "Location", "location", "hub_name", "compound"] if c in valid_df.columns), None)
    aqi_col = next((c for c in ["us_aqi", "aqi", "AQI"] if c in valid_df.columns), None)
    green_col = next((c for c in ["green_coverage_pct", "green_pct", "green_space_pct"] if c in valid_df.columns), None)
    macro_col = next((c for c in ["macro_region", "region"] if c in valid_df.columns), None)

    # Synthesize fallback columns if missing
    if price_col is None and ppm_col is not None and "Area" in valid_df.columns:
        valid_df["_synth_price"] = valid_df[ppm_col] * valid_df["Area"]
        price_col = "_synth_price"
    elif price_col is None:
        valid_df["_synth_price"] = 1.0
        price_col = "_synth_price"

    if ppm_col is None and price_col is not None and "Area" in valid_df.columns:
        valid_df["_synth_ppm"] = valid_df[price_col] / valid_df["Area"].replace(0, np.nan)
        ppm_col = "_synth_ppm"
    elif ppm_col is None:
        valid_df["_synth_ppm"] = valid_df[price_col]
        ppm_col = "_synth_ppm"

    if aqi_col is None:
        valid_df["_synth_aqi"] = 78.0
        aqi_col = "_synth_aqi"

    if green_col is None:
        valid_df["_synth_green"] = 5.0
        green_col = "_synth_green"

    if loc_col is None:
        valid_df["_synth_loc"] = "Urban Cluster"
        loc_col = "_synth_loc"

    # 4. Spatial Grid Binning
    # Using integer division to prevent floating-point key mismatch
    valid_df["_bin_lat"] = np.round(valid_df[lat_col] / grid_deg).astype(int)
    valid_df["_bin_lon"] = np.round(valid_df[lon_col] / grid_deg).astype(int)
    valid_df["_grid_lat"] = valid_df["_bin_lat"] * grid_deg
    valid_df["_grid_lon"] = valid_df["_bin_lon"] * grid_deg

    def _mode_or_first(s: pd.Series) -> str:
        clean = s.dropna()
        if clean.empty:
            return "Metropolitan Hub"
        m = clean.mode()
        return str(m.iloc[0]) if not m.empty else str(clean.iloc[0])

    # 5. Grouped Aggregations
    agg_dict = {
        "count": (price_col, "count"),
        "median_price": (price_col, "median"),
        "median_ppm": (ppm_col, "median"),
        "aqi": (aqi_col, "median"),
        "green_pct": (green_col, "median"),
        "primary_hub": (loc_col, _mode_or_first),
        "lat": ("_grid_lat", "first"),
        "lon": ("_grid_lon", "first"),
        "centroid_lat": (lat_col, "mean"),
        "centroid_lon": (lon_col, "mean"),
    }

    if macro_col:
        agg_dict["macro_region"] = (macro_col, _mode_or_first)
    else:
        agg_dict["macro_region"] = ("_grid_lat", lambda _: "Greater Cairo Metropolitan")

    grouped = valid_df.groupby(["_bin_lat", "_bin_lon"]).agg(**agg_dict).reset_index(drop=True)

    # 6. Standardize Aliases & Types
    grouped["location"] = grouped["primary_hub"]
    grouped["hub_name"] = grouped["primary_hub"]
    grouped["unit_count"] = grouped["count"]
    grouped["Calculated_Price_Per_Meter"] = grouped["median_ppm"]
    grouped["latitude"] = grouped["lat"]
    grouped["longitude"] = grouped["lon"]

    grouped.sort_values(by="count", ascending=False, inplace=True)
    grouped.reset_index(drop=True, inplace=True)

    return grouped


# -----------------------------------------------------------------------------
# 3. ENHANCED GEOSPATIAL MAP RENDERER
# -----------------------------------------------------------------------------
def render_enhanced_geo_map(
    filtered_df: pd.DataFrame,
    active_count: int,
    height: int,
    map_type: str = "density",
    title: Optional[str] = None
) -> go.Figure:
    """
    Renders an institutional geospatial map resolving point overplotting in East Cairo / New Cairo.

    Modes:
    - 'density': Uses px.density_map with z='Calculated_Price_Per_Meter', radius 15-18,
      and smooth 4-stop Egyptian palette (Soft Sage -> Nile Teal -> Sandstone Gold -> Coral Red).
      Completely eliminates point collisions by visualizing true capital density surfaces.
    - 'cluster': Uses generate_clustered_hubs spatially aggregated into ~1.5km grid cells,
      scaling bubble size with verified unit volume and color-coding by median EGP/m² with rich hover data.

    Args:
        filtered_df: Cleaned listings matching active cockpit filters.
        active_count: Total active listing count.
        height: Widget display height in pixels.
        map_type: 'density' or 'cluster' (default: 'density').

    Returns:
        go.Figure: Interactive Plotly Map ready for Streamlit rendering.
    """
    # Defensive Empty / No-data State Handling
    if (
        filtered_df is None
        or not isinstance(filtered_df, pd.DataFrame)
        or filtered_df.empty
        or active_count <= 0
    ):
        fig = go.Figure(go.Scattermap(lat=[], lon=[]))
        fig.update_layout(
            title=dict(
                text="🗺️ Geospatial Pricing & Density Surface (No Listings Matching Filter)",
                font=dict(size=12, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"),
                x=0.01,
                y=0.97
            ),
            map=dict(
                style="open-street-map",
                center=DEFAULT_MAP_CENTER,
                zoom=DEFAULT_MAP_ZOOM
            ),
            height=height,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", color=CHARCOAL),
            annotations=[
                dict(
                    text="No geocoded listings match the active filter criteria.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=13, color=CHARCOAL, family="Inter, sans-serif")
                )
            ]
        )
        return fig

    # Resolve coordinates
    lat_col = next((c for c in ["lat", "latitude", "Lat"] if c in filtered_df.columns), None)
    lon_col = next((c for c in ["lon", "longitude", "Lon"] if c in filtered_df.columns), None)

    if not lat_col or not lon_col:
        fig = go.Figure(go.Scattermap(lat=[], lon=[]))
        fig.update_layout(
            map=dict(style="open-street-map", center=DEFAULT_MAP_CENTER, zoom=DEFAULT_MAP_ZOOM),
            height=height,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        return fig

    # Filter to geocoded points
    valid_mask = filtered_df[lat_col].notnull() & filtered_df[lon_col].notnull()
    if "geo_status" in filtered_df.columns:
        valid_geo = (filtered_df["geo_status"] == "VALID_GEO") & valid_mask
        if valid_geo.any():
            valid_mask = valid_geo

    geo_data = filtered_df[valid_mask].copy()
    if geo_data.empty:
        # Fallback to all non-null coordinates
        geo_data = filtered_df[filtered_df[lat_col].notnull() & filtered_df[lon_col].notnull()].copy()

    if geo_data.empty:
        fig = go.Figure(go.Scattermap(lat=[], lon=[]))
        fig.update_layout(
            map=dict(style="open-street-map", center=DEFAULT_MAP_CENTER, zoom=DEFAULT_MAP_ZOOM),
            height=height,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        return fig

    # Standardize column names for Plotly Express
    if lat_col != "lat":
        geo_data["lat"] = geo_data[lat_col]
    if lon_col != "lon":
        geo_data["lon"] = geo_data[lon_col]

    # Resolve metric column
    z_col = next(
        (c for c in ["Calculated_Price_Per_Meter", "Price_Per_Meter", "median_ppm", "Price_Numeric"] if c in geo_data.columns),
        None
    )
    if z_col is None:
        geo_data["Calculated_Price_Per_Meter"] = 35000.0
        z_col = "Calculated_Price_Per_Meter"

    normalized_map_type = str(map_type or "density").strip().lower()

    if normalized_map_type == "cluster":
        # -------------------------------------------------------------
        # MODE B: Spatially Binned Clustered Hubs
        # -------------------------------------------------------------
        clustered_df = generate_clustered_hubs(geo_data, grid_deg=0.015)

        if clustered_df.empty:
            clustered_df = geo_data.copy()
            clustered_df["count"] = 1
            clustered_df["median_price"] = clustered_df.get("Price_Numeric", 3500000)
            clustered_df["median_ppm"] = clustered_df[z_col]
            clustered_df["aqi"] = clustered_df.get("us_aqi", 78.0)
            clustered_df["green_pct"] = clustered_df.get("green_coverage_pct", 5.0)
            clustered_df["location"] = clustered_df.get("normalized_location", "Hub")

        fig_map = px.scatter_map(
            clustered_df,
            lat="lat",
            lon="lon",
            size="count",
            color="median_ppm",
            hover_name="location",
            hover_data=["count", "median_price", "median_ppm", "aqi", "green_pct"],
            color_continuous_scale=EGYPTIAN_GEO_HEATMAP_SCALE,
            size_max=26,
            zoom=DEFAULT_MAP_ZOOM,
            center=DEFAULT_MAP_CENTER
        )

        fig_map.update_traces(
            hovertemplate="<b>Hub Cluster: %{hovertext}</b><br>"
                          "Units: <b>%{customdata[0]:,}</b><br>"
                          "Median Price: <b>%{customdata[1]:,.0f} EGP</b><br>"
                          "EGP/m²: <b>%{customdata[2]:,.0f}</b><br>"
                          "Median AQI: <b>%{customdata[3]:.1f}</b><br>"
                          "Green Space: <b>%{customdata[4]:.1f}%</b><extra></extra>",
            marker=dict(opacity=0.88)
        )

        title_text = "🗺️ Geospatial Clustered Hubs & Pricing (Greater Cairo & Coast)"

    else:
        # -------------------------------------------------------------
        # MODE A: Dynamic Density Heatmap (px.density_map)
        # -------------------------------------------------------------
        fig_map = px.density_map(
            geo_data,
            lat="lat",
            lon="lon",
            z=z_col,
            radius=16,
            zoom=DEFAULT_MAP_ZOOM,
            center=DEFAULT_MAP_CENTER,
            color_continuous_scale=EGYPTIAN_GEO_HEATMAP_SCALE
        )

        fig_map.update_traces(
            hovertemplate="<b>Density & Capital Concentration</b><br>"
                          "Weighted Benchmark: <b>%{z:,.0f} EGP/m²</b><br>"
                          "Coordinates: (%{lat:.3f}, %{lon:.3f})<extra></extra>"
        )

        title_text = "🗺️ Geospatial Pricing Concentration & Density Surface (Greater Cairo)"

    # Common Executive Layout
    final_title = title if title is not None else title_text
    fig_map.update_layout(
        title=dict(
            text=final_title,
            font=dict(size=12, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"),
            x=0.01,
            y=0.97
        ),
        map_style="open-street-map",
        height=height,
        margin=dict(l=0, r=0, t=32, b=0),
        coloraxis_colorbar=dict(
            title=dict(text="EGP/m²", font=dict(size=9, color=CHARCOAL, family="Inter, sans-serif")),
            thickness=10,
            len=0.68,
            tickfont=dict(size=8, color=CHARCOAL, family="Inter, sans-serif"),
            tickformat=",.0f"
        ),
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color=CHARCOAL),
        uirevision="geo_cairo",
        transition=dict(duration=350, easing="cubic-in-out")
    )

    return fig_map


# -----------------------------------------------------------------------------
# 4. HANDOVER COHORT ESCALATION TRAJECTORY CHART (TAB 3)
# -----------------------------------------------------------------------------
def render_cohort_escalation_chart(
    timeline_df: pd.DataFrame,
    height: int,
    show_iqr: bool = False,
    title: Optional[str] = None
) -> go.Figure:
    """
    Renders an institutional bar chart with explicit sample size (N) labels and
    cohort color-coding, completely eliminating erratic line chart whipsawing caused
    by small luxury off-plan cohorts after 2026.

    Cohort Classification & Coloring:
    - Vintage (< 2020):            Charcoal (#333333)
    - Ready Delivery (2020-2025):  Nile Teal (#2A9D8F)
    - Off-Plan Core (2026-2028):   Deep Navy (#1B2A4A)
    - Far-Future Off-Plan (2029+): Sandstone Gold (#E9B44C)

    Args:
        timeline_df: Filtered listings containing 'Year Built' (or handover completion year).
        height: Display height in pixels.
        show_iqr: Whether to render IQR error bars (Q25 to Q75 bounds).
        title: Optional custom chart title.

    Returns:
        go.Figure: Bar chart displaying median price/m² with sample count (N) annotations.
    """
    title_dict = dict(
        text=title or "📈 Median Price/m² by Delivery Year",
        font=dict(size=12, color=DEEP_NAVY, family="Inter, sans-serif", weight="bold"),
        x=0.01,
        y=0.97
    )

    # Defensive check for empty or invalid DataFrame
    if (
        timeline_df is None
        or not isinstance(timeline_df, pd.DataFrame)
        or timeline_df.empty
    ):
        fig = go.Figure()
        fig.update_layout(
            title=title_dict,
            height=height,
            margin=dict(l=20, r=20, t=36, b=20),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", color=CHARCOAL),
            annotations=[
                dict(
                    text="No handover cohort listings available for the active filters.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=13, color=CHARCOAL, family="Inter, sans-serif")
                )
            ]
        )
        return fig

    # Resolve Year Built column
    year_col = next((c for c in ["Year Built", "year_built", "Handover", "Year", "year"] if c in timeline_df.columns), None)
    if not year_col:
        fig = go.Figure()
        fig.update_layout(title=title_dict, height=height)
        return fig

    # Resolve metric columns
    price_col = next((c for c in ["Price_Numeric", "Price", "price"] if c in timeline_df.columns), None)
    ppm_col = next((c for c in ["Calculated_Price_Per_Meter", "Price_Per_Meter", "median_ppm", "ppm"] if c in timeline_df.columns), None)

    working_df = timeline_df[timeline_df[year_col].notnull()].copy()
    if working_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title_dict, height=height)
        return fig

    # Filter out anomalous / corrupted year values (restrict to plausible cohorts e.g. 1990 - 2035)
    working_df = working_df[(working_df[year_col] >= 1990) & (working_df[year_col] <= 2035)]
    if working_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title_dict, height=height)
        return fig

    if price_col is None:
        working_df["_synth_price"] = 1.0
        price_col = "_synth_price"

    if ppm_col is None:
        if "Area" in working_df.columns and price_col != "_synth_price":
            working_df["_synth_ppm"] = working_df[price_col] / working_df["Area"].replace(0, np.nan)
        else:
            working_df["_synth_ppm"] = 35000.0
        ppm_col = "_synth_ppm"

    # Aggregations by Year Built
    t_summary = working_df.groupby(year_col).agg(
        Volume=(price_col, "count"),
        Median_PPM=(ppm_col, "median"),
        Q25_PPM=(ppm_col, lambda s: float(s.quantile(0.25))),
        Q75_PPM=(ppm_col, lambda s: float(s.quantile(0.75)))
    ).reset_index()

    t_summary[year_col] = t_summary[year_col].astype(int)
    t_summary.sort_values(by=year_col, inplace=True)
    t_summary.reset_index(drop=True, inplace=True)

    classifications: List[str] = []
    bar_colors: List[str] = []
    bar_texts: List[str] = []
    customdata: List[List[Any]] = []

    for _, row in t_summary.iterrows():
        yr = int(row[year_col])
        ppm = float(row["Median_PPM"])
        vol = int(row["Volume"])
        q25 = float(row["Q25_PPM"])
        q75 = float(row["Q75_PPM"])

        # Institutional Cohort Classification & Color Palette
        if yr < 2020:
            c_name = "Vintage (<2020)"
            c_color = CHARCOAL          # #333333
        elif yr <= 2025:
            c_name = "Ready (2020 - 2025)"
            c_color = NILE_TEAL         # #2A9D8F
        elif yr <= 2028:
            c_name = "Off-Plan (2026 - 2028)"
            c_color = DEEP_NAVY         # #1B2A4A
        else:
            # 2029+ Far Future Off-Plan (Highlight thin sample size / speculative variance)
            c_name = "Off-Plan Far-Future (2029+)"
            c_color = SANDSTONE_GOLD    # #E9B44C

        classifications.append(c_name)
        bar_colors.append(c_color)

        # Exact requested bar text format:
        # f"{ppm:,.0f} EGP<br><span style='font-size:8px; opacity:0.85;'>N={vol}</span>"
        bar_label = f"{ppm:,.0f} EGP<br><span style='font-size:8px; opacity:0.85;'>N={vol}</span>"
        bar_texts.append(bar_label)

        # customdata for rich hover template
        customdata.append([c_name, vol, q25, q75])

    # Construct Bar Trace
    bar_trace = go.Bar(
        x=t_summary[year_col],
        y=t_summary["Median_PPM"],
        ids=t_summary[year_col].astype(str).tolist(),
        text=bar_texts,
        textposition="outside",
        textfont=dict(size=9, color=DEEP_NAVY, family="Inter, sans-serif"),
        marker=dict(
            color=bar_colors,
            line=dict(color=OFF_WHITE, width=1)
        ),
        customdata=customdata,
        hovertemplate="<b>Handover Cohort %{x}</b><br>"
                      "Cohort Classification: <b>%{customdata[0]}</b><br>"
                      "Median Unit Price: <b>%{y:,.0f} EGP/m²</b><br>"
                      "Verified Pipeline Volume: <b>%{customdata[1]:,} units</b><extra></extra>",
        name="Delivery Years"
    )

    # Optional Interquartile Band / Error Bars
    if show_iqr:
        err_plus = [max(0.0, float(q75 - med)) for med, q75 in zip(t_summary["Median_PPM"], t_summary["Q75_PPM"])]
        err_minus = [max(0.0, float(med - q25)) for med, q25 in zip(t_summary["Median_PPM"], t_summary["Q25_PPM"])]
        bar_trace.error_y = dict(
            type="data",
            symmetric=False,
            array=err_plus,
            arrayminus=err_minus,
            color=DEEP_NAVY,
            thickness=1.5,
            width=5
        )

    fig = go.Figure(data=[bar_trace])

    # Layout Calibration with Headroom for Outside Text
    max_ppm = float(t_summary["Median_PPM"].max()) if not t_summary.empty else 60000.0
    y_upper_bound = max_ppm * 1.22  # Extra 22% headroom so N= labels never clip

    min_yr = int(t_summary[year_col].min())
    max_yr = int(t_summary[year_col].max())
    dtick_step = 1 if (max_yr - min_yr) <= 14 else 2

    fig.update_layout(
        title=title_dict,
        height=height,
        margin=dict(l=20, r=20, t=36, b=20),
        xaxis=dict(
            title=dict(text="Delivery Year", font=dict(size=10, color=CHARCOAL, family="Inter, sans-serif")),
            tickmode="linear",
            tick0=min_yr,
            dtick=dtick_step,
            tickfont=dict(size=8, color=CHARCOAL, family="Inter, sans-serif"),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="Median Price/m² (EGP)", font=dict(size=10, color=CHARCOAL, family="Inter, sans-serif")),
            tickfont=dict(size=8, color=CHARCOAL, family="Inter, sans-serif"),
            gridcolor="#f1f5f9",
            tickformat=",.0f",
            range=[0, y_upper_bound]
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color=CHARCOAL),
        showlegend=False,
        uirevision="cohort_chart",
        transition=dict(duration=350, easing="cubic-in-out")
    )

    return fig

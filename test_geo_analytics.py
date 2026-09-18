# -*- coding: utf-8 -*-
"""
Test Suite for geo_analytics.py
File: test_geo_analytics.py

Verifies:
1. Palette tokens and Egyptian executive colors.
2. generate_clustered_hubs aggregates 10,000 listings into non-overlapping spatial clusters.
3. render_enhanced_geo_map returns valid Plotly figures for both 'density' and 'cluster' modes.
4. render_cohort_escalation_chart correctly computes volumes, formats N= sample size labels,
   color-codes cohorts, and returns a valid Plotly bar figure.
5. Defensive edge case handling (empty dataframes, None inputs, missing columns).
"""

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import pdist

import geo_analytics
from data_loader import load_clean_data


def test_palette_tokens():
    print("[1/5] Testing Egyptian Executive Palette tokens & geo color scales...")
    expected_tokens = {
        "Deep Navy": (geo_analytics.DEEP_NAVY, "#1B2A4A"),
        "Nile Teal": (geo_analytics.NILE_TEAL, "#2A9D8F"),
        "Sandstone Gold": (geo_analytics.SANDSTONE_GOLD, "#E9B44C"),
        "Coral Red": (geo_analytics.CORAL_RED, "#E85D4C"),
        "Soft Sage": (geo_analytics.SOFT_SAGE, "#8FB996"),
        "Off-White": (geo_analytics.OFF_WHITE, "#F7F5F2"),
        "Light Gray": (geo_analytics.LIGHT_GRAY, "#D9D9D6"),
        "Charcoal": (geo_analytics.CHARCOAL, "#333333"),
    }

    for name, (actual, expected) in expected_tokens.items():
        assert actual.upper() == expected.upper(), f"Token mismatch for {name}: {actual} != {expected}"
        print(f"  ✓ {name}: {actual}")

    # Verify smooth 4-stop continuous color scale
    scale = geo_analytics.EGYPTIAN_GEO_HEATMAP_SCALE
    assert len(scale) == 4, f"Expected 4 scale stops, got {len(scale)}"
    assert scale[0][1].upper() == geo_analytics.SOFT_SAGE.upper()
    assert scale[1][1].upper() == geo_analytics.NILE_TEAL.upper()
    assert scale[2][1].upper() == geo_analytics.SANDSTONE_GOLD.upper()
    assert scale[3][1].upper() == geo_analytics.CORAL_RED.upper()
    print("  ✓ EGYPTIAN_GEO_HEATMAP_SCALE verified: Soft Sage -> Nile Teal -> Sandstone Gold -> Coral Red.")


def test_clustered_hubs_aggregation():
    print("\n[2/5] Testing generate_clustered_hubs on 10,000 listing dataset...")
    df = load_clean_data()
    assert len(df) == 10000, f"Expected 10,000 listings, found {len(df)}"

    valid_geo_count = int(((df["geo_status"] == "VALID_GEO") & df["lat"].notnull() & df["lon"].notnull()).sum())
    print(f"  ✓ Total listings: {len(df):,} | Valid geocoded listings: {valid_geo_count:,}")

    grid_deg = 0.015
    clustered_df = geo_analytics.generate_clustered_hubs(df, grid_deg=grid_deg)

    assert isinstance(clustered_df, pd.DataFrame), "Result must be a pd.DataFrame"
    assert not clustered_df.empty, "Clustered dataframe must not be empty"

    # Required columns
    required_cols = [
        "location", "primary_hub", "lat", "lon", "count",
        "median_price", "median_ppm", "aqi", "green_pct"
    ]
    for col in required_cols:
        assert col in clustered_df.columns, f"Missing required column: {col}"
    print(f"  ✓ All required columns present: {required_cols}")

    # Volume preservation: all valid geocoded units must be accounted for
    total_clustered_units = int(clustered_df["count"].sum())
    assert total_clustered_units == valid_geo_count, (
        f"Preservation mismatch: {total_clustered_units} clustered vs {valid_geo_count} input units"
    )
    print(f"  ✓ 100% volume preserved: {total_clustered_units:,} units across {len(clustered_df)} spatial hubs.")

    # Spatial binning non-overlapping verification
    # 1. Coordinate duplicates check
    has_dups = clustered_df.duplicated(subset=["lat", "lon"]).any()
    assert not has_dups, "Spatial clusters must have unique grid coordinate centers (lat, lon)"

    # 2. Distance check: any two distinct cluster centers on the grid must be separated by >= grid_deg
    coords = clustered_df[["lat", "lon"]].values
    if len(coords) > 1:
        # Chebyshev / Infinity norm: max(|lat1 - lat2|, |lon1 - lon2|)
        distances_chebyshev = pdist(coords, metric="chebyshev")
        min_chebyshev = distances_chebyshev.min()
        assert min_chebyshev >= (grid_deg - 1e-9), (
            f"Cluster collision detected! Min distance {min_chebyshev} < grid_deg {grid_deg}"
        )
        print(f"  ✓ Non-overlapping verified: Minimum Chebyshev distance = {min_chebyshev:.4f}° >= {grid_deg}°")

    # Value sanity check
    assert (clustered_df["count"] > 0).all(), "All clusters must have positive unit count"
    assert (clustered_df["median_price"] > 0).all(), "All clusters must have positive median price"
    assert (clustered_df["median_ppm"] > 0).all(), "All clusters must have positive median EGP/m²"
    assert (clustered_df["aqi"] > 0).all(), "All clusters must have valid AQI"
    assert (clustered_df["green_pct"] >= 0).all(), "All clusters must have valid green %"
    print("  ✓ Statistical values (Price, PPM, AQI, Green %) verified within valid ranges.")


def test_clustered_hubs_edge_cases():
    print("\n[3/5] Testing generate_clustered_hubs edge cases (empty, None, synthetic)...")
    # Empty and None
    empty_res = geo_analytics.generate_clustered_hubs(pd.DataFrame())
    assert isinstance(empty_res, pd.DataFrame) and empty_res.empty
    none_res = geo_analytics.generate_clustered_hubs(None)
    assert isinstance(none_res, pd.DataFrame) and none_res.empty
    print("  ✓ None and empty DataFrames handled safely without errors.")

    # Synthetic clustered test: 5 tightly grouped points in New Cairo
    tight_df = pd.DataFrame({
        "lat": [30.0410, 30.0412, 30.0415, 30.0411, 30.0413],
        "lon": [31.3350, 31.3352, 31.3351, 31.3353, 31.3354],
        "Price_Numeric": [5000000, 6000000, 5500000, 5200000, 5800000],
        "Calculated_Price_Per_Meter": [30000, 32000, 31000, 29000, 33000],
        "normalized_location": ["New Cairo", "New Cairo", "New Cairo", "New Cairo", "New Cairo"],
        "us_aqi": [75.0, 76.0, 75.0, 74.0, 75.0],
        "green_coverage_pct": [12.0, 14.0, 13.0, 12.5, 13.5]
    })
    tight_clusters = geo_analytics.generate_clustered_hubs(tight_df, grid_deg=0.015)
    assert len(tight_clusters) == 1, f"Expected 1 merged cluster, got {len(tight_clusters)}"
    assert tight_clusters.iloc[0]["count"] == 5
    assert tight_clusters.iloc[0]["median_price"] == 5500000.0
    assert tight_clusters.iloc[0]["median_ppm"] == 31000.0
    assert tight_clusters.iloc[0]["primary_hub"] == "New Cairo"
    print("  ✓ Synthetic tightly-spaced listings successfully consolidated into 1 single cluster.")


def test_render_enhanced_geo_map():
    print("\n[4/5] Testing render_enhanced_geo_map for 'density' and 'cluster' modes...")
    df = load_clean_data()

    # 1. Density mode
    fig_density = geo_analytics.render_enhanced_geo_map(
        filtered_df=df,
        active_count=len(df),
        height=525,
        map_type="density"
    )
    assert isinstance(fig_density, go.Figure), "Must return a plotly go.Figure"
    assert len(fig_density.data) > 0, "Figure must contain at least one trace"
    assert fig_density.data[0].type == "densitymap", f"Expected densitymap trace, got {fig_density.data[0].type}"
    assert fig_density.layout.height == 525
    assert "Geospatial" in fig_density.layout.title.text
    print(f"  ✓ Density Mode verified: trace type = '{fig_density.data[0].type}', height = {fig_density.layout.height}px")

    # 2. Cluster mode
    fig_cluster = geo_analytics.render_enhanced_geo_map(
        filtered_df=df,
        active_count=len(df),
        height=525,
        map_type="cluster"
    )
    assert isinstance(fig_cluster, go.Figure), "Must return a plotly go.Figure"
    assert len(fig_cluster.data) > 0, "Figure must contain at least one trace"
    assert fig_cluster.data[0].type == "scattermap", f"Expected scattermap trace, got {fig_cluster.data[0].type}"
    assert fig_cluster.layout.height == 525

    # Check hovertemplate and customdata fields
    trace = fig_cluster.data[0]
    hovertemplate = trace.hovertemplate
    assert "Hub Cluster:" in hovertemplate
    assert "Units:" in hovertemplate
    assert "Median Price:" in hovertemplate
    assert "EGP/m²:" in hovertemplate
    assert "Median AQI:" in hovertemplate
    assert "Green Space:" in hovertemplate
    assert trace.customdata is not None and len(trace.customdata) > 0
    assert len(trace.customdata[0]) >= 5, "customdata must contain at least 5 attributes for hover"
    print("  ✓ Cluster Mode verified: scattermap with rich hover data (Hub, Units, Price, PPM, AQI, Green %).")

    # 3. Empty input handling
    fig_empty = geo_analytics.render_enhanced_geo_map(
        filtered_df=pd.DataFrame(),
        active_count=0,
        height=525,
        map_type="density"
    )
    assert isinstance(fig_empty, go.Figure)
    assert fig_empty.layout.height == 525
    print("  ✓ Empty dataset handled gracefully with valid Plotly Figure returned.")


def test_cohort_escalation_chart():
    print("\n[5/5] Testing render_cohort_escalation_chart for Volume labels, colors & hover...")
    df = load_clean_data()
    timeline_df = df[(df["Year Built"] >= 2018) & (df["Year Built"] <= 2030)].copy()
    assert not timeline_df.empty

    fig = geo_analytics.render_cohort_escalation_chart(timeline_df, height=480, show_iqr=False)
    assert isinstance(fig, go.Figure), "Must return a plotly go.Figure"
    assert len(fig.data) == 1, f"Expected 1 bar trace, got {len(fig.data)}"

    bar_trace = fig.data[0]
    assert bar_trace.type == "bar", f"Expected Bar trace, got {bar_trace.type}"
    assert len(bar_trace.x) == len(timeline_df["Year Built"].unique())

    # Title check
    expected_title = "📈 Median Price/m² by Delivery Year"
    assert fig.layout.title.text == expected_title, f"Title mismatch: '{fig.layout.title.text}'"
    print(f"  ✓ Title verified: '{expected_title}'")

    # Check sample size (N) labels and format:
    # f"{ppm:,.0f} EGP<br><span style='font-size:8px; opacity:0.85;'>N={vol}</span>"
    bar_texts = bar_trace.text
    assert bar_texts is not None and len(bar_texts) == len(bar_trace.x)
    for txt in bar_texts:
        assert "EGP" in txt, f"Text missing 'EGP': {txt}"
        assert "N=" in txt, f"Text missing sample count 'N=': {txt}"
        assert "<span style='font-size:8px; opacity:0.85;'>" in txt, f"Text missing styling: {txt}"
    print(f"  ✓ Sample count labels verified across all {len(bar_texts)} cohorts (e.g. '{bar_texts[0]}').")

    # Check Color Coding
    # - Vintage (<2020): Charcoal #333333
    # - Ready (2020 - 2025): Nile Teal #2A9D8F
    # - Off-Plan (2026 - 2028): Deep Navy #1B2A4A
    # - Far-Future (2029+): Sandstone Gold #E9B44C
    colors = bar_trace.marker.color
    years = [int(y) for y in bar_trace.x]
    for yr, col in zip(years, colors):
        if yr < 2020:
            assert col == geo_analytics.CHARCOAL, f"Cohort {yr} expected Charcoal, got {col}"
        elif yr <= 2025:
            assert col == geo_analytics.NILE_TEAL, f"Cohort {yr} expected Nile Teal, got {col}"
        elif yr <= 2028:
            assert col == geo_analytics.DEEP_NAVY, f"Cohort {yr} expected Deep Navy, got {col}"
        else:
            assert col == geo_analytics.SANDSTONE_GOLD, f"Cohort {yr} expected Sandstone Gold, got {col}"
    print("  ✓ Cohort color-coding verified: Vintage (Charcoal), Ready (Nile Teal), Off-Plan (Navy), Far-Future (Gold).")

    # Check hovertemplate and customdata
    hovertemplate = bar_trace.hovertemplate
    assert "Handover Cohort %{x}" in hovertemplate
    assert "Cohort Classification: <b>%{customdata[0]}</b>" in hovertemplate
    assert "Median Unit Price: <b>%{y:,.0f} EGP/m²</b>" in hovertemplate
    assert "Verified Pipeline Volume: <b>%{customdata[1]:,} units</b>" in hovertemplate

    customdata = bar_trace.customdata
    assert len(customdata) == len(years)
    for cd, yr in zip(customdata, years):
        c_name, vol, q25, q75 = cd
        assert isinstance(c_name, str) and len(c_name) > 0
        assert isinstance(vol, (int, np.integer)) and vol > 0
        assert q25 <= q75
    print("  ✓ Rich hovertemplate and customdata successfully verified.")

    # Test show_iqr=True
    fig_iqr = geo_analytics.render_cohort_escalation_chart(timeline_df, height=480, show_iqr=True)
    assert fig_iqr.data[0].error_y is not None
    assert len(fig_iqr.data[0].error_y.array) == len(years)
    print("  ✓ Interquartile band (show_iqr=True) correctly calculates and attaches error bounds.")

    # Test empty dataframe
    fig_empty = geo_analytics.render_cohort_escalation_chart(pd.DataFrame(), height=480)
    assert isinstance(fig_empty, go.Figure)
    assert fig_empty.layout.title.text == expected_title
    print("  ✓ Empty timeline handled safely with valid Figure.")


def main():
    print("=" * 70)
    print("RUNNING GEOSPATIAL & COHORT ANALYTICS VERIFICATION TEST SUITE")
    print("=" * 70)

    test_palette_tokens()
    test_clustered_hubs_aggregation()
    test_clustered_hubs_edge_cases()
    test_render_enhanced_geo_map()
    test_cohort_escalation_chart()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    main()

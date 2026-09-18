# -*- coding: utf-8 -*-
"""
Test and Verification Suite for ui_theme.py
Validates:
1. Exact color palette hex codes and dictionary keys.
2. Zero-scroll CSS generation and strict styling rules.
3. render_header_strip HTML generation across all badge types.
4. render_stat_card KPI card generation, styles, and overrides.
5. get_plotly_layout_defaults structure, grid colors, and font specs.
6. Module compilation with zero errors.
"""

import sys
import py_compile
from typing import Dict, Any

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import ui_theme


def test_color_palette():
    print("\n[TEST 1/5] Verifying Color Palette Requirements...")
    palette = ui_theme.get_color_palette()
    assert isinstance(palette, dict), "Palette must be a dictionary"
    
    expected_colors = {
        "navy": "#1B2A4A",
        "teal": "#2A9D8F",
        "gold": "#E9B44C",
        "coral": "#E85D4C",
        "sage": "#8FB996",
        "offwhite": "#F7F5F2",
        "lightgray": "#D9D9D6",
        "charcoal": "#333333",
    }
    
    for key, expected_hex in expected_colors.items():
        assert key in palette, f"Missing key '{key}' in get_color_palette()"
        assert palette[key].upper() == expected_hex.upper(), (
            f"Color mismatch for '{key}': expected {expected_hex}, got {palette[key]}"
        )
        print(f"  ✓ {key.ljust(10)}: {palette[key]} (Exact Match)")

    print("  -> PASSED: All 8 exact palette colors verified successfully.")


def test_custom_css():
    print("\n[TEST 2/5] Verifying get_custom_css() & Zero-Scroll Mandate...")
    css_wrapped = ui_theme.get_custom_css(raw=False)
    css_raw = ui_theme.get_custom_css(raw=True)
    
    assert css_wrapped.startswith("<style>") and css_wrapped.endswith("</style>"), (
        "get_custom_css(raw=False) must be enclosed in <style> tags"
    )
    assert "<style>" not in css_raw, "get_custom_css(raw=True) must not contain <style> tags"

    # Strict Zero-Scroll Viewport checks
    assert "100vh !important" in css_raw, "CSS must enforce 100vh height"
    assert "overflow: hidden !important" in css_raw, "CSS must enforce overflow: hidden"
    assert "#F7F5F2" in css_raw, "CSS must set page background to off-white #F7F5F2"

    # Sidebar checks
    assert "#1B2A4A" in css_raw, "CSS must style sidebar with Deep Navy #1B2A4A"
    assert "#E9B44C" in css_raw, "CSS must style golden accents (#E9B44C)"
    assert "#2A9D8F" in css_raw, "CSS must style Nile Teal active radio buttons (#2A9D8F)"

    # Specific Sidebar Button selectors and resets
    assert '[data-testid="stSidebar"] div[data-testid="stButton"] > button' in css_raw, (
        "CSS must specifically target sidebar buttons inside stButton"
    )
    assert 'linear-gradient(135deg, #2A9D8F 0%, #1B2A4A 100%)' in css_raw, (
        "CSS must set teal-to-navy gradient on sidebar buttons"
    )
    assert '[data-testid="stSidebarCollapseButton"]' in css_raw, (
        "CSS must reset stSidebarCollapseButton"
    )
    assert '[data-testid="stSidebar"] span[data-baseweb="tag"] button' in css_raw, (
        "CSS must reset multiselect tag remove buttons"
    )
    assert '[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked)' in css_raw, (
        "CSS must target checked radio button label"
    )
    assert '[data-testid="stSidebar"] button {' not in css_raw, (
        "CSS must not use generic [data-testid='stSidebar'] button selector"
    )

    # Executive card checks
    assert "#D9D9D6" in css_raw, "CSS must style borders with light gray #D9D9D6"
    assert "#333333" in css_raw, "CSS must style body typography with Charcoal #333333"
    assert ".stat-card:hover" in css_raw, "CSS must specify hover elevation for stat-card"
    assert "box-shadow: 0 4px 12px rgba(27, 42, 74, 0.08);" in css_raw, (
        "CSS must specify subtle hover box-shadow for stat-card:hover"
    )

    # Plot formatting & padding checks
    assert ".plot-row-spacer" in css_raw, "CSS must define .plot-row-spacer"
    assert "margin-top: 8px !important;" in css_raw, "CSS must set margin-top: 8px on plot spacers"
    assert ".plot-container-card" in css_raw, "CSS must define .plot-container-card"
    assert "border: 1px solid #D9D9D6 !important;" in css_raw, "CSS must style plot container card border"
    assert "justify-content: center !important;" in css_raw, "CSS must center plot container card"

    # Square plot centering & margin
    assert ".stPlotlyChart" in css_raw, "CSS must style .stPlotlyChart"
    assert "margin: 0 auto !important;" in css_raw, "CSS must center plots with auto margins"
    assert ':has(.stPlotlyChart)' in css_raw, "CSS must add top margin for Plotly charts in column containers"
    assert "margin-top: 4px !important;" in css_raw, "CSS must set 4px top margin on Plotly chart element containers"

    # Anti-flicker architecture
    assert '[data-stale="true"]' in css_raw, "CSS must target data-stale attribute"
    assert "opacity: 1 !important;" in css_raw, "CSS must enforce opacity 1 for anti-flicker"
    assert "transition: none !important;" in css_raw, "CSS must disable transitions for anti-flicker"

    print("  ✓ Zero-scroll 100vh & overflow:hidden enforced")
    print("  ✓ Off-white background (#F7F5F2) set")
    print("  ✓ Deep Navy (#1B2A4A) sidebar with Sandstone Gold (#E9B44C) & Nile Teal (#2A9D8F) verified")
    print("  ✓ Specific sidebar button gradient and collapse/tag button resets confirmed")
    print("  ✓ Executive card borders (#D9D9D6), body text (#333333), and card hover elevation confirmed")
    print("  ✓ Plot container cards, spacer padding, and square plot centering confirmed")
    print("  ✓ Anti-flicker architecture and stale-state overrides confirmed")
    print("  -> PASSED: Custom CSS fulfills all requirements.")


def test_header_strip():
    print("\n[TEST 3/5] Verifying render_header_strip()...")
    title = "Market Command Cockpit"
    subtitle = "Real-time Geospatial & Valuation Metrics"
    badge_text = "Active: 10,000 Units"

    badge_types = ["sage", "teal", "gold", "coral", "navy"]
    for b_type in badge_types:
        html = ui_theme.render_header_strip(title, subtitle, badge_text, badge_type=b_type)
        assert isinstance(html, str), "Header strip must return a string"
        assert title in html, "Title missing in header strip HTML"
        assert subtitle in html, "Subtitle missing in header strip HTML"
        assert badge_text in html, "Badge text missing in header strip HTML"
        assert "#1B2A4A" in html, "Deep Navy title color missing in header strip"
        assert "#333333" in html, "Charcoal subtitle color missing in header strip"
        assert "pulse-pill" in html, "pulse-pill class missing in header strip"
        assert "pulse-dot" in html, "pulse-dot class missing in header strip"
        assert 'title="Live system status: Audited across 10,000 real estate properties"' in html, (
            "pulse-pill title attribute missing or incorrect"
        )
        print(f"  ✓ badge_type='{b_type}' rendered correctly with status tooltip")

    # Fallback check
    fallback_html = ui_theme.render_header_strip(title, subtitle, badge_text, badge_type="unknown")
    assert "pulse-pill" in fallback_html, "Fallback badge styling must render gracefully"
    print("  ✓ Fallback badge handling verified")
    print("  -> PASSED: render_header_strip fully verified.")


def test_stat_card():
    print("\n[TEST 4/5] Verifying render_stat_card()...")
    card_html = ui_theme.render_stat_card(
        label="Median Unit Price",
        value="5.85M <small>EGP</small>",
        subtext="IQR: 3.2M - 9.1M",
        border_color="#2A9D8F",
        subtext_color="#8FB996"
    )
    assert isinstance(card_html, str), "Stat card must return an HTML string"
    assert "stat-card" in card_html, "stat-card class missing"
    assert "MEDIAN UNIT PRICE" in card_html.upper(), "Label missing"
    assert "5.85M <small>EGP</small>" in card_html, "Value missing"
    assert "IQR: 3.2M - 9.1M" in card_html, "Subtext missing"

    # Specific styling checks
    assert "border-left: 4px solid #2A9D8F" in card_html, "4px left border requirement missing"
    assert "font-size: 11px" in card_html, "11px label/subtext requirement missing"
    assert "font-size: 1.35rem" in card_html, "1.35rem value requirement missing"
    assert "font-weight: 800" in card_html, "bold value requirement missing"
    assert "color: #8FB996" in card_html, "subtext color missing"
    assert "color: #1B2A4A" in card_html, "Deep navy value color missing"

    # Tooltip checks
    assert 'title="Median Unit Price"' in card_html, "Default fallback to label title missing"

    card_custom_tooltip = ui_theme.render_stat_card(
        label="Average PPSQM",
        value="35,000",
        subtext="+12% YoY",
        tooltip="Price per square meter calculated from verified listings"
    )
    assert 'title="Price per square meter calculated from verified listings"' in card_custom_tooltip, (
        "Custom tooltip title attribute missing"
    )

    kpi_card_html = ui_theme.render_kpi_card(
        label="Yield Rate",
        value="8.5%",
        subtitle="Gross annual",
        tooltip="Estimated rental yield based on local compound averages"
    )
    assert 'title="Estimated rental yield based on local compound averages"' in kpi_card_html, (
        "render_kpi_card tooltip forwarding failed"
    )

    print("  ✓ 4px left border confirmed")
    print("  ✓ Uppercase 11px label confirmed")
    print("  ✓ Bold 1.35rem value in Deep Navy confirmed")
    print("  ✓ 11px colored subtext confirmed")
    print("  ✓ Default and custom tooltip title attributes confirmed")
    print("  ✓ render_kpi_card tooltip forwarding confirmed")
    print("  -> PASSED: render_stat_card fully verified.")


def test_plotly_defaults():
    print("\n[TEST 5/5] Verifying get_plotly_layout_defaults()...")
    layout = ui_theme.get_plotly_layout_defaults()
    assert isinstance(layout, dict), "Layout defaults must be a dict"

    # Background checks
    assert layout.get("paper_bgcolor") in ["rgba(0,0,0,0)", "#F7F5F2", "rgba(247, 245, 242, 0)"], (
        "Background must be transparent or off-white"
    )
    assert layout.get("plot_bgcolor") in ["rgba(0,0,0,0)", "#FFFFFF", "#F7F5F2"], (
        "Plot bgcolor must be transparent, white, or off-white"
    )

    # Font checks
    font = layout.get("font", {})
    assert "Inter" in font.get("family", ""), "Font family must include Inter"
    assert font.get("color") == "#333333", "Font color must be Charcoal (#333333)"

    # Title checks
    title_font = layout.get("title", {}).get("font", {})
    assert title_font.get("color") == "#1B2A4A", "Title color must be Deep Navy (#1B2A4A)"

    # Gridlines checks
    xaxis = layout.get("xaxis", {})
    yaxis = layout.get("yaxis", {})
    assert xaxis.get("gridcolor") == "#D9D9D6", "X-axis gridcolor must be Light gray (#D9D9D6)"
    assert yaxis.get("gridcolor") == "#D9D9D6", "Y-axis gridcolor must be Light gray (#D9D9D6)"

    # Margins and Legend
    assert "margin" in layout, "Margin specification missing"
    assert "legend" in layout, "Legend specification missing"
    assert layout["legend"].get("orientation") == "h", "Legend orientation should be horizontal"

    # Colorway sequence
    assert "colorway" in layout, "Colorway sequence missing"
    assert "#1B2A4A" in layout["colorway"], "Deep Navy missing from colorway"
    assert "#2A9D8F" in layout["colorway"], "Nile Teal missing from colorway"

    print("  ✓ Transparent/off-white background confirmed")
    print("  ✓ Inter font family with Charcoal text (#333333) confirmed")
    print("  ✓ Light gray gridlines (#D9D9D6) confirmed on X and Y axes")
    print("  ✓ Deep Navy title (#1B2A4A) and horizontal legend confirmed")
    print("  ✓ Executive colorway sequence verified")

    # Verify apply_plotly_theme if plotly is available
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[1, 2, 3], y=[10, 20, 30]))
        fig = ui_theme.apply_plotly_theme(fig, height=400)
        assert fig.layout.height == 400
        print("  ✓ apply_plotly_theme helper function successfully tested with go.Figure")
    except ImportError:
        print("  ! Plotly library not installed in test environment, skipping Figure object check")

    print("  -> PASSED: get_plotly_layout_defaults fully verified.")


def test_display_heights():
    print("\n[TEST 6/6] Verifying get_display_heights() 1080p Calibration & Scalings...")
    
    expected_1080p = {
        "h_grid": 350,
        "h_upper": 355,
        "h_lower": 350,
        "h_sub": 350,
        "h_table_leader": 350,
        "h_bench_table": 175,
        "h_listing_table": 350,
    }
    
    # 1. Standard scale inputs and empty/default
    for scale in ["Standard", "Standard 1080p", "Standard FHD", "", None]:
        h = ui_theme.get_display_heights(scale)
        assert h == expected_1080p, f"get_display_heights('{scale}') mismatch: {h} != {expected_1080p}"
    
    # Default parameter call without arguments
    h_default = ui_theme.get_display_heights()
    assert h_default == expected_1080p, f"get_display_heights() default mismatch: {h_default} != {expected_1080p}"
    print("  ✓ Calibrated 1080p heights verified for Standard, empty, and default invocations")

    # 2. Large display scale
    h_large = ui_theme.get_display_heights("Large / 2K Display (Current)")
    assert all(k in h_large for k in expected_1080p), "Large scale missing required keys"
    assert h_large["h_upper"] > h_large["h_lower"], "Large scale h_upper must be greater than h_lower"
    assert h_large["h_upper"] == 530 and h_large["h_lower"] == 525, "Large scale heights mismatch"
    print("  ✓ Large / 2K Display heights verified")

    # 3. Compact laptop scale
    h_compact = ui_theme.get_display_heights("Compact Laptop")
    assert all(k in h_compact for k in expected_1080p), "Compact scale missing required keys"
    assert h_compact["h_upper"] > h_compact["h_lower"], "Compact scale h_upper must be greater than h_lower"
    assert h_compact["h_upper"] == 270 and h_compact["h_lower"] == 265, "Compact scale heights mismatch"
    print("  ✓ Compact Laptop heights verified")

    print("  -> PASSED: get_display_heights fully verified.")


def main():
    print("=" * 75)
    print("  🚀 RUNNING VERIFICATION SUITE FOR ui_theme.py")
    print("=" * 75)

    # 1. Bytecode compilation check
    print("\n[Step 0] Bytecode compiling ui_theme.py...")
    py_compile.compile("ui_theme.py", doraise=True)
    print("  ✓ Bytecode compilation successful with 0 syntax errors.")

    test_color_palette()
    test_custom_css()
    test_header_strip()
    test_stat_card()
    test_plotly_defaults()
    test_display_heights()

    print("\n" + "=" * 75)
    print("  🎉 ALL TESTS PASSED SUCCESSFULLY! (6/6 SUITES)")
    print("=" * 75)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Test Suite for cockpit_views.py and ui_theme.py
Verifies:
- All required exports and signatures exist
- Exact color palette hex values
- Execution of chart creation and data operations with real dataset
- No missing imports or runtime syntax errors
"""

import sys
import inspect
import pandas as pd
import numpy as np

def test_imports_and_palette():
    print("[1/4] Testing imports and color palette constants...")
    import ui_theme
    import cockpit_views

    expected_colors = {
        "Deep Navy": ("#1B2A4A", ui_theme.DEEP_NAVY, cockpit_views.DEEP_NAVY),
        "Nile Teal": ("#2A9D8F", ui_theme.NILE_TEAL, cockpit_views.NILE_TEAL),
        "Sandstone Gold": ("#E9B44C", ui_theme.SANDSTONE_GOLD, cockpit_views.SANDSTONE_GOLD),
        "Coral Red": ("#E85D4C", ui_theme.CORAL_RED, cockpit_views.CORAL_RED),
        "Soft Sage": ("#8FB996", ui_theme.SOFT_SAGE, cockpit_views.SOFT_SAGE),
        "Off-white": ("#F7F5F2", ui_theme.OFF_WHITE, cockpit_views.OFF_WHITE),
        "Light gray": ("#D9D9D6", ui_theme.LIGHT_GRAY, cockpit_views.LIGHT_GRAY),
        "Charcoal": ("#333333", ui_theme.CHARCOAL, cockpit_views.CHARCOAL),
    }

    for name, (expected, theme_val, views_val) in expected_colors.items():
        assert theme_val.upper() == expected.upper(), f"ui_theme {name} mismatch: {theme_val} != {expected}"
        assert views_val.upper() == expected.upper(), f"cockpit_views {name} mismatch: {views_val} != {expected}"
        print(f"  ✓ {name}: {expected}")

    print("  ✓ Color palette verified successfully.")


def test_function_signatures():
    print("\n[2/4] Testing cockpit function signatures...")
    import cockpit_views

    # Cockpit 1: (filtered_df, active_count, total_count, active_pct, display_scale)
    sig1 = inspect.signature(cockpit_views.render_cockpit_1)
    params1 = list(sig1.parameters.keys())
    expected1 = ["filtered_df", "active_count", "total_count", "active_pct", "display_scale"]
    assert params1 == expected1, f"render_cockpit_1 params mismatch: {params1} != {expected1}"
    print(f"  ✓ render_cockpit_1 params: {params1}")

    # Cockpit 2: (filtered_df, active_count, display_scale)
    sig2 = inspect.signature(cockpit_views.render_cockpit_2)
    params2 = list(sig2.parameters.keys())
    expected2 = ["filtered_df", "active_count", "display_scale"]
    assert params2 == expected2, f"render_cockpit_2 params mismatch: {params2} != {expected2}"
    print(f"  ✓ render_cockpit_2 params: {params2}")

    # Cockpit 3: (filtered_df, active_count, display_scale)
    sig3 = inspect.signature(cockpit_views.render_cockpit_3)
    params3 = list(sig3.parameters.keys())
    expected3 = ["filtered_df", "active_count", "display_scale"]
    assert params3 == expected3, f"render_cockpit_3 params mismatch: {params3} != {expected3}"
    print(f"  ✓ render_cockpit_3 params: {params3}")


def test_height_scaling():
    print("\n[3/4] Testing display height scaling calculation...")
    from ui_theme import get_display_heights

    scales = ["Large / 2K Display (Current)", "Standard 1080p", "Compact Laptop"]
    for scale in scales:
        h = get_display_heights(scale)
        assert all(k in h for k in ["h_upper", "h_lower", "h_sub", "h_table_leader", "h_bench_table", "h_listing_table"])
        assert h["h_upper"] > h["h_lower"]
        print(f"  ✓ Scale '{scale}': upper={h['h_upper']}px, lower={h['h_lower']}px, listing_table={h['h_listing_table']}px")


def test_live_data_execution():
    print("\n[4/4] Testing live data execution with real dataset...")
    from data_loader import load_clean_data
    import cockpit_views
    import streamlit as st

    df = load_clean_data()
    assert len(df) == 10000, f"Expected 10,000 rows, got {len(df)}"
    print(f"  ✓ Loaded clean data: {len(df):,} listings")

    # Mock streamlit functions that render in bare python without active streamlit server
    class MockStreamlitContext:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    class MockStreamlitColumn:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def markdown(self, *a, **kw): pass
        def plotly_chart(self, *a, **kw): pass
        def dataframe(self, *a, **kw): pass
        def slider(self, *a, **kw): return kw.get("value", 5.0)
        def selectbox(self, *a, **kw):
            opts = kw.get("options", [1])
            idx = kw.get("index", 0)
            return opts[idx] if idx < len(opts) else opts[0]
        def text_input(self, *a, **kw): return ""
        def info(self, *a, **kw): pass
        def caption(self, *a, **kw): pass

    orig_columns = st.columns
    orig_markdown = st.markdown
    orig_plotly_chart = st.plotly_chart
    orig_dataframe = st.dataframe
    orig_slider = st.slider
    orig_selectbox = st.selectbox
    orig_text_input = st.text_input
    orig_info = st.info
    orig_caption = st.caption

    try:
        st.columns = lambda n: [MockStreamlitColumn() for _ in range(n if isinstance(n, int) else len(n))]
        st.markdown = lambda *a, **kw: None
        st.plotly_chart = lambda *a, **kw: None
        st.dataframe = lambda *a, **kw: None
        st.slider = lambda *a, **kw: kw.get("value", 5.0)
        st.selectbox = lambda *a, **kw: kw.get("options", [1])[kw.get("index", 0)]
        st.text_input = lambda *a, **kw: ""
        st.info = lambda *a, **kw: None
        st.caption = lambda *a, **kw: None

        # Execute Cockpit 1
        cockpit_views.render_cockpit_1(
            filtered_df=df,
            active_count=len(df),
            total_count=len(df),
            active_pct=100.0,
            display_scale="Standard 1080p"
        )
        print("  ✓ render_cockpit_1 executed without exceptions.")

        # Execute Cockpit 2
        cockpit_views.render_cockpit_2(
            filtered_df=df,
            active_count=len(df),
            display_scale="Standard 1080p"
        )
        print("  ✓ render_cockpit_2 executed without exceptions.")

        # Execute Cockpit 3
        cockpit_views.render_cockpit_3(
            filtered_df=df,
            active_count=len(df),
            display_scale="Standard 1080p"
        )
        print("  ✓ render_cockpit_3 executed without exceptions.")

    finally:
        st.columns = orig_columns
        st.markdown = orig_markdown
        st.plotly_chart = orig_plotly_chart
        st.dataframe = orig_dataframe
        st.slider = orig_slider
        st.selectbox = orig_selectbox
        st.text_input = orig_text_input
        st.info = orig_info
        st.caption = orig_caption

    print("\nALL COCKPIT VIEWS TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_imports_and_palette()
    test_function_signatures()
    test_height_scaling()
    test_live_data_execution()

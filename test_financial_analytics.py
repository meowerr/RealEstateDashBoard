# -*- coding: utf-8 -*-
"""
Egypt Real Estate Intelligence Platform (v1) - Test Suite for Institutional Financial Analytics
File: test_financial_analytics.py

Comprehensive unit tests verifying:
1. CAGR vs. MOIC Discrepancy & Conversion:
   - 1.84x MOIC over 10-year horizon equals ~6.29% CAGR.
   - Nominal gain and mathematical boundary checks.
2. Robust Exact DCF Internal Rate of Return (IRR):
   - High-precision Newton-Raphson solver benchmarks.
   - Multi-year bullet and irregular DCF cash flows.
   - Percentage vs. decimal output formatting.
3. Transaction Negotiation Haircut:
   - Portal asking price discounting to execution levels.
4. Institutional 2-Way Underwriting Sensitivity Matrix:
   - Matrix shape, non-emptiness, and mathematical validity.
   - Monotonicity across Exit Cap Rates and Rental Growth Rates.
   - Leveraged vs. unleveraged capital structure dynamics.
   - Custom cap rate and growth grid flexibility.
5. Executive Plotly Heatmap Visualization:
   - Valid Plotly Figure generation with exact Egyptian Executive Palette tokens.
   - Cell annotations: <b>{irr:.1f}%</b><br><span style='font-size:9px;'>{moic:.2f}x</span>.
   - Rich custom hovertemplate formatting and axis labels.
"""

import sys
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Ensure UTF-8 output encoding in Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from financial_analytics import (
    calculate_cagr_and_moic,
    compute_exact_irr,
    apply_negotiation_discount,
    generate_sensitivity_matrix,
    render_sensitivity_heatmap,
    DEEP_NAVY,
    NILE_TEAL,
    SANDSTONE_GOLD,
    SOFT_SAGE,
    OFF_WHITE,
    CHARCOAL,
    EGYPTIAN_SENSITIVITY_SCALE,
)


def test_cagr_vs_moic_discrepancy():
    """Verifies that 1.84x MOIC over 10 years equals ~6.29% CAGR."""
    print("\n[TEST 1/5] Testing CAGR vs. MOIC Discrepancy & Exact Conversion...")

    # Benchmark: 1.84x MOIC over 10 years
    initial_price = 100.0
    terminal_price = 184.0
    horizon = 10

    res = calculate_cagr_and_moic(
        price_initial=initial_price,
        price_terminal=terminal_price,
        horizon_years=horizon
    )

    # 1. MOIC verification
    assert res["moic"] == 1.84, f"Expected MOIC 1.84, got {res['moic']}"
    print(f"  ✓ Verified MOIC: {res['moic']:.2f}x")

    # 2. CAGR verification: (1.84)^(1/10) - 1 = 6.28574... -> 6.29%
    assert abs(res["cagr_pct"] - 6.29) < 0.02, f"Expected CAGR ~6.29%, got {res['cagr_pct']}%"
    print(f"  ✓ Verified 10Y CAGR: {res['cagr_pct']:.2f}% (Matches expected ~6.29%)")

    # 3. Nominal gain verification
    assert res["nominal_gain"] == 84.0, f"Expected nominal gain 84.0, got {res['nominal_gain']}"
    print(f"  ✓ Verified Nominal Gain: {res['nominal_gain']:.2f}")

    # 4. Realistic 10-year property valuation test (e.g. 10M to 24.2M EGP)
    prop_res = calculate_cagr_and_moic(10_000_000.0, 24_200_000.0, 10)
    assert prop_res["moic"] == 2.42, f"Expected MOIC 2.42, got {prop_res['moic']}"
    assert abs(prop_res["cagr_pct"] - 9.24) < 0.05, f"Expected CAGR ~9.24%, got {prop_res['cagr_pct']}%"
    assert prop_res["nominal_gain"] == 14_200_000.0
    print(f"  ✓ Verified Property Benchmark: 10M -> 24.2M EGP => {prop_res['moic']}x MOIC, {prop_res['cagr_pct']}% CAGR")

    # 5. Boundary & error handling
    try:
        calculate_cagr_and_moic(-100.0, 200.0, 10)
        assert False, "Should have raised ValueError for negative price_initial"
    except ValueError:
        print("  ✓ Correctly rejected negative price_initial")

    try:
        calculate_cagr_and_moic(100.0, 200.0, 0)
        assert False, "Should have raised ValueError for zero horizon_years"
    except ValueError:
        print("  ✓ Correctly rejected zero horizon_years")


def test_compute_exact_irr_benchmarks():
    """Verifies Newton-Raphson exact IRR against known DCF benchmarks."""
    print("\n[TEST 2/5] Testing Newton-Raphson Exact DCF IRR Solver...")

    # Benchmark 1: 1-Year 10% gain (Outflow -100, Inflow +110)
    cfs_1 = [-100.0, 110.0]
    irr_1_pct = compute_exact_irr(cfs_1, as_pct=True)
    irr_1_dec = compute_exact_irr(cfs_1, as_pct=False)
    assert abs(irr_1_pct - 10.0) < 1e-3, f"Expected IRR 10.0%, got {irr_1_pct}%"
    assert abs(irr_1_dec - 0.10) < 1e-5, f"Expected IRR 0.10, got {irr_1_dec}"
    print(f"  ✓ Benchmark 1 [-100, 110]: {irr_1_pct:.2f}% (decimal: {irr_1_dec:.4f})")

    # Benchmark 2: 3-Year 8% Coupon Bond (Outflow -100, Coupon 8, 8, Exit 108)
    cfs_2 = [-100.0, 8.0, 8.0, 108.0]
    irr_2 = compute_exact_irr(cfs_2, as_pct=True)
    assert abs(irr_2 - 8.0) < 1e-3, f"Expected IRR 8.0%, got {irr_2}%"
    print(f"  ✓ Benchmark 2 [-100, 8, 8, 108]: {irr_2:.2f}%")

    # Benchmark 3: Irregular DCF series [-1000, 300, 400, 400, 300]
    cfs_3 = [-1000.0, 300.0, 400.0, 400.0, 300.0]
    irr_3_pct = compute_exact_irr(cfs_3, as_pct=True)
    r_dec = irr_3_pct / 100.0
    # Verify NPV at solved r is virtually zero
    t_vals = np.arange(len(cfs_3))
    npv_at_irr = np.sum(np.array(cfs_3) / ((1.0 + r_dec) ** t_vals))
    assert abs(npv_at_irr) < 1e-3, f"NPV at solved IRR should be ~0, got {npv_at_irr}"
    assert abs(irr_3_pct - 14.895) < 0.05, f"Expected IRR ~14.90%, got {irr_3_pct}%"
    print(f"  ✓ Benchmark 3 Irregular DCF: {irr_3_pct:.2f}% (Verified NPV(r) = {npv_at_irr:.6f})")

    # Benchmark 4: Zero Return [-100, 100]
    cfs_zero = [-100.0, 100.0]
    irr_zero = compute_exact_irr(cfs_zero, as_pct=True)
    assert abs(irr_zero - 0.0) < 1e-3, f"Expected IRR 0.0%, got {irr_zero}%"
    print(f"  ✓ Benchmark 4 Zero Return: {irr_zero:.2f}%")

    # Benchmark 5: Single sign / no sign change edge case (all positive or all negative)
    assert compute_exact_irr([100.0, 200.0]) == 0.0
    assert compute_exact_irr([-100.0, -200.0]) == 0.0
    print("  ✓ Handled no sign-change edge cases gracefully.")


def test_apply_negotiation_discount():
    """Verifies portal asking price negotiation haircut calculations."""
    print("\n[TEST 3/5] Testing Negotiation Haircut Factor Calculation...")

    asking_price = 10_000_000.0

    # 1. Default 7.5% discount
    exec_price_default = apply_negotiation_discount(asking_price)
    expected_default = 10_000_000.0 * (1.0 - 0.075)
    assert exec_price_default == expected_default, f"Expected {expected_default}, got {exec_price_default}"
    print(f"  ✓ Asking 10.0M with default 7.5% discount -> {exec_price_default:,.0f} EGP")

    # 2. Custom 10.0% discount
    exec_price_10 = apply_negotiation_discount(asking_price, discount_pct=10.0)
    assert exec_price_10 == 9_000_000.0, f"Expected 9M, got {exec_price_10}"
    print(f"  ✓ Asking 10.0M with 10.0% discount -> {exec_price_10:,.0f} EGP")

    # 3. Decimal fraction discount input (0.075)
    exec_price_dec = apply_negotiation_discount(asking_price, discount_pct=0.075)
    assert exec_price_dec == 9_250_000.0, f"Expected 9.25M, got {exec_price_dec}"
    print(f"  ✓ Asking 10.0M with decimal 0.075 input -> {exec_price_dec:,.0f} EGP")

    # 4. Zero discount
    exec_price_zero = apply_negotiation_discount(asking_price, discount_pct=0.0)
    assert exec_price_zero == 10_000_000.0, f"Expected 10M, got {exec_price_zero}"
    print(f"  ✓ Asking 10.0M with 0.0% discount -> {exec_price_zero:,.0f} EGP")

    # 5. Error handling
    try:
        apply_negotiation_discount(-1_000_000.0)
        assert False, "Should have rejected negative asking price"
    except ValueError:
        print("  ✓ Correctly rejected negative asking price")


def test_generate_sensitivity_matrix():
    """Verifies institutional 2-way sensitivity matrix generation, dimensions, and financial math."""
    print("\n[TEST 4/5] Testing 2-Way Sensitivity Matrix Dimensions & Financial Logic...")

    base_price = 10_000_000.0
    gross_yield = 7.0
    opex = 15.0
    equity_pct = 40.0
    horizon = 10

    matrix = generate_sensitivity_matrix(
        base_price=base_price,
        gross_yield_pct=gross_yield,
        opex_pct=opex,
        equity_pct=equity_pct,
        horizon_years=horizon,
    )

    # 1. Output structure & keys
    required_keys = [
        "irr_matrix", "moic_matrix", "exit_val_matrix", "exit_val_m_matrix",
        "exit_cap_rates", "rental_growth_rates", "df_irr", "df_moic", "df_exit_val",
        "initial_equity", "initial_debt", "remaining_debt", "initial_net_rent", "horizon_years"
    ]
    for k in required_keys:
        assert k in matrix, f"Missing required key '{k}' in matrix output"
    print("  ✓ Verified presence of all required institutional keys.")

    # 2. Dimensions check (5 Exit Caps x 5 Rental Growths)
    caps = matrix["exit_cap_rates"]
    growths = matrix["rental_growth_rates"]
    assert len(caps) == 5, f"Expected 5 cap rates, got {len(caps)}"
    assert len(growths) == 5, f"Expected 5 growth rates, got {len(growths)}"
    assert matrix["irr_matrix"].shape == (5, 5), f"Expected (5, 5) shape, got {matrix['irr_matrix'].shape}"
    assert matrix["moic_matrix"].shape == (5, 5), f"Expected (5, 5) shape, got {matrix['moic_matrix'].shape}"
    assert matrix["exit_val_matrix"].shape == (5, 5)
    print(f"  ✓ Verified matrix dimensions: {matrix['irr_matrix'].shape} ({len(caps)} caps x {len(growths)} growths)")

    # 3. Capital structure checks
    assert matrix["initial_equity"] == 4_000_000.0, f"Expected 4M equity, got {matrix['initial_equity']}"
    assert matrix["initial_debt"] == 6_000_000.0, f"Expected 6M debt, got {matrix['initial_debt']}"
    assert matrix["remaining_debt"] == 6_000_000.0, f"Expected 6M remaining debt, got {matrix['remaining_debt']}"
    # Stabilized Year 0 Net Rent: 10M * 7% * (1 - 15%) = 595,000 EGP
    assert matrix["initial_net_rent"] == 595_000.0, f"Expected 595k net rent, got {matrix['initial_net_rent']}"
    print(f"  ✓ Capital Structure: Equity={matrix['initial_equity']:,.0f}, Debt={matrix['initial_debt']:,.0f}, Net Rent={matrix['initial_net_rent']:,.0f}")

    # 4. Strict Financial Monotonicity Verification:
    # A. Rental Growth Monotonicity: For a fixed exit cap rate, higher rental growth MUST yield higher IRR & MOIC
    irr_mat = matrix["irr_matrix"]
    moic_mat = matrix["moic_matrix"]
    exit_val_mat = matrix["exit_val_matrix"]

    for i in range(len(caps)):
        for j in range(len(growths) - 1):
            assert irr_mat[i, j + 1] > irr_mat[i, j], (
                f"Row {i}: IRR should strictly increase with growth, but {irr_mat[i, j+1]} <= {irr_mat[i, j]}"
            )
            assert moic_mat[i, j + 1] > moic_mat[i, j], (
                f"Row {i}: MOIC should strictly increase with growth, but {moic_mat[i, j+1]} <= {moic_mat[i, j]}"
            )
            assert exit_val_mat[i, j + 1] > exit_val_mat[i, j], (
                f"Row {i}: Exit valuation should strictly increase with growth"
            )
    print("  ✓ Strict Monotonicity: Increasing Rental Growth strictly increases IRR, MOIC, and Exit Valuation.")

    # B. Cap Rate Compression Monotonicity: For a fixed rental growth rate, lower exit cap rate MUST yield higher Exit Valuation, IRR & MOIC
    for j in range(len(growths)):
        for i in range(len(caps) - 1):
            # caps are in ascending order [5.5, 6.5, 7.5, 8.5, 9.5]
            # so caps[i] < caps[i+1], meaning cap[i] is lower (compressed) -> higher valuation & return
            assert irr_mat[i, j] > irr_mat[i + 1, j], (
                f"Col {j}: Compressed cap rate should yield higher IRR, but {irr_mat[i, j]} <= {irr_mat[i+1, j]}"
            )
            assert moic_mat[i, j] > moic_mat[i + 1, j], (
                f"Col {j}: Compressed cap rate should yield higher MOIC, but {moic_mat[i, j]} <= {moic_mat[i+1, j]}"
            )
            assert exit_val_mat[i, j] > exit_val_mat[i + 1, j], (
                f"Col {j}: Compressed cap rate should yield higher Exit Valuation"
            )
    print("  ✓ Strict Monotonicity: Cap Rate Compression strictly increases Exit Valuation, IRR, and MOIC.")

    # 5. Leveraged vs. Unleveraged Verification
    unleveraged = generate_sensitivity_matrix(
        base_price=base_price,
        gross_yield_pct=gross_yield,
        opex_pct=opex,
        equity_pct=100.0,
        horizon_years=horizon,
    )
    assert unleveraged["initial_debt"] == 0.0
    assert unleveraged["remaining_debt"] == 0.0
    # Positive leverage: Leveraged MOIC (40% equity) should exceed unleveraged MOIC (100% equity)
    assert matrix["moic_matrix"][2, 2] > unleveraged["moic_matrix"][2, 2], "Leveraged MOIC should exceed unleveraged MOIC"
    print(f"  ✓ Leverage Effect: Base case MOIC increases from {unleveraged['moic_matrix'][2, 2]:.2f}x (unleveraged) to {matrix['moic_matrix'][2, 2]:.2f}x (leveraged).")

    # 6. Custom grid flexibility
    custom_caps = [6.0, 7.0, 8.0]
    custom_growths = [4.0, 8.0]
    custom_mat = generate_sensitivity_matrix(
        base_price=base_price,
        exit_cap_rates=custom_caps,
        rental_growth_rates=custom_growths
    )
    assert custom_mat["irr_matrix"].shape == (3, 2), f"Expected (3, 2), got {custom_mat['irr_matrix'].shape}"
    print("  ✓ Custom grid dimensions verified (3x2).")


def test_render_sensitivity_heatmap():
    """Verifies Plotly Heatmap rendering, Egyptian Executive Palette styling, annotations, and hovertemplate."""
    print("\n[TEST 5/5] Testing Executive Plotly Heatmap Figure Generation...")

    base_price = 10_000_000.0
    gross_yield = 7.0
    opex = 15.0
    equity = 40.0
    horizon = 10
    height = 340

    fig = render_sensitivity_heatmap(
        base_price=base_price,
        gross_yield_pct=gross_yield,
        opex_pct=opex,
        equity_pct=equity,
        horizon_years=horizon,
        height=height
    )

    # 1. Verify Figure type
    assert isinstance(fig, go.Figure), f"Expected go.Figure, got {type(fig)}"
    print("  ✓ Successfully created plotly.graph_objects.Figure instance.")

    # 2. Verify Title and Axis Titles
    expected_title = "📊 Investment Return Sensitivity (Exit Cap vs. Rental Growth)"
    assert fig.layout.title.text == expected_title, f"Title mismatch: {fig.layout.title.text}"
    assert fig.layout.xaxis.title.text == "Rental Growth Rate (%)"
    assert fig.layout.yaxis.title.text == "Exit Cap Rate (%)"
    assert fig.layout.height == height, f"Expected height {height}, got {fig.layout.height}"
    print(f"  ✓ Title verified: '{fig.layout.title.text}'")
    print(f"  ✓ Axis titles verified: X='{fig.layout.xaxis.title.text}', Y='{fig.layout.yaxis.title.text}'")

    # 3. Verify Heatmap Trace & Dimensions
    assert len(fig.data) == 1, "Expected exactly 1 trace in figure"
    trace = fig.data[0]
    assert isinstance(trace, go.Heatmap), f"Expected go.Heatmap trace, got {type(trace)}"
    assert len(trace.y) == 5, f"Expected 5 y-ticks (Exit Caps), got {len(trace.y)}"
    assert len(trace.x) == 5, f"Expected 5 x-ticks (Rental Growths), got {len(trace.x)}"
    assert np.array(trace.z).shape == (5, 5), f"Expected z shape (5, 5), got {np.array(trace.z).shape}"
    print(f"  ✓ Trace confirmed as go.Heatmap with grid dimensions (5, 5).")

    # 4. Verify Cell Text Annotations:
    # <b>{irr:.1f}%</b><br><span style='font-size:9px;'>{moic:.2f}x</span>
    cell_sample = trace.text[0][0]
    assert "<b>" in cell_sample and "%</b>" in cell_sample, f"Cell text missing bold IRR %: {cell_sample}"
    assert "<span style='font-size:9px;'>" in cell_sample and "x</span>" in cell_sample, (
        f"Cell text missing subtext MOIC x: {cell_sample}"
    )
    print(f"  ✓ Cell annotation format verified: '{cell_sample}'")

    # 5. Verify Rich Custom Hovertemplate:
    # <b>Exit Cap Rate: %{y:.1f}% | Rental Growth: %{x:.1f}%</b><br>
    # Projected Leveraged IRR: <b>%{customdata[0]:.1f}%</b><br>
    # Equity Multiple (MOIC): <b>%{customdata[1]:.2f}x</b><br>
    # Projected Exit Valuation: <b>%{customdata[2]:,.1f}M EGP</b><extra></extra>
    expected_hover_parts = [
        "<b>Exit Cap Rate: %{y:.1f}% | Rental Growth: %{x:.1f}%</b>",
        "Projected Leveraged IRR: <b>%{customdata[0]:.1f}%</b>",
        "Equity Multiple (MOIC): <b>%{customdata[1]:.2f}x</b>",
        "Projected Exit Valuation: <b>%{customdata[2]:,.1f}M EGP</b>",
        "<extra></extra>"
    ]
    for part in expected_hover_parts:
        assert part in trace.hovertemplate, f"Hovertemplate missing segment: '{part}' in '{trace.hovertemplate}'"
    print("  ✓ Rich institutional hovertemplate fully verified.")

    # 6. Verify Customdata Array Dimensions: (5, 5, 3)
    customdata_arr = np.array(trace.customdata)
    assert customdata_arr.shape == (5, 5, 3), f"Expected customdata shape (5, 5, 3), got {customdata_arr.shape}"
    # Verify [0]=IRR, [1]=MOIC, [2]=Exit Valuation in Millions
    sample_cd = customdata_arr[2, 2]
    assert sample_cd[0] > 0.0  # IRR
    assert sample_cd[1] > 0.0  # MOIC
    assert sample_cd[2] > 0.0  # Exit Valuation in Millions EGP
    print(f"  ✓ Customdata payload verified: IRR={sample_cd[0]:.1f}%, MOIC={sample_cd[1]:.2f}x, ExitVal={sample_cd[2]:.1f}M EGP")

    # 7. Verify Palette & Theme Configuration
    assert fig.layout.paper_bgcolor == OFF_WHITE, f"Expected paper_bgcolor {OFF_WHITE}, got {fig.layout.paper_bgcolor}"
    assert fig.layout.title.font.color == DEEP_NAVY, f"Expected title color {DEEP_NAVY}, got {fig.layout.title.font.color}"
    assert trace.colorscale is not None
    print("  ✓ Egyptian Executive Palette and layout theme confirmed.")


def run_all_tests():
    print("=" * 75)
    print("  EGYPT REAL ESTATE INTELLIGENCE PLATFORM (v1)")
    print("  INSTITUTIONAL QUANTITATIVE FINANCIAL ANALYTICS TEST SUITE")
    print("=" * 75)

    test_cagr_vs_moic_discrepancy()
    test_compute_exact_irr_benchmarks()
    test_apply_negotiation_discount()
    test_generate_sensitivity_matrix()
    test_render_sensitivity_heatmap()

    print("\n" + "=" * 75)
    print("  >>> ALL INSTITUTIONAL FINANCIAL ANALYTICS TESTS PASSED (100% PASS) <<<")
    print("=" * 75)


if __name__ == "__main__":
    run_all_tests()

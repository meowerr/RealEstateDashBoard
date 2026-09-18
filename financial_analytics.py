# -*- coding: utf-8 -*-
"""
Egypt Real Estate Intelligence Platform (v1) - Institutional Financial Analytics
File: financial_analytics.py

Institutional Real Estate Quantitative Financial Modeling Engine:
1. CAGR vs. MOIC Calculation & Distinction:
   - Multiple on Invested Capital (MOIC / Equity Multiple) vs. Annualized CAGR.
   - Over a 10-year horizon: 1.84x MOIC corresponds to CAGR = (1.84)^(1/10) - 1 = 6.29%.
2. Exact DCF Internal Rate of Return (IRR):
   - High-precision Newton-Raphson solver with step damping and Brent method fallback.
3. Institutional 2-Way Underwriting Sensitivity Matrix:
   - Evaluates acquisitions across Exit Cap Rate (%) vs. Annual Rental Growth Rate (%).
   - Calculates dynamic NOI compounding, Exit Valuation, Leveraged Equity MOIC, and Leveraged IRR (%).
4. Executive Heatmap Visualization:
   - Interactive Plotly Heatmap styled in the Egyptian Executive Palette.
   - Dual-metric cell annotations: bold IRR (%) on top and muted MOIC (x) beneath.
   - Custom institutional hovertemplate reporting cap rate, rental growth, IRR, MOIC, and exit valuation.
5. Transaction Negotiation Haircut:
   - Calibrates portal asking prices to transacted execution clearing levels.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. EGYPTIAN EXECUTIVE PALETTE & CHART STYLING TOKENS
# -----------------------------------------------------------------------------
DEEP_NAVY = "#1B2A4A"       # Primary base, high-contrast dark headers
NILE_TEAL = "#2A9D8F"       # Primary brand / data accent, core returns
SANDSTONE_GOLD = "#E9B44C"  # Egyptian accent, premium return tier, highlights
CORAL_RED = "#E85D4C"       # Negative / downside bounds
SOFT_SAGE = "#8FB996"       # Moderate positive returns, sustainable yield
OFF_WHITE = "#F7F5F2"       # Clean page background / base canvas
LIGHT_GRAY = "#D9D9D6"      # Gridlines, subtle borders
CHARCOAL = "#333333"        # Crisp executive typography

# Continuous colorscale blending the Egyptian Executive Palette
# From Soft Sage -> Nile Teal -> Deep Navy -> Sandstone Gold
EGYPTIAN_SENSITIVITY_SCALE = [
    [0.0, OFF_WHITE],
    [0.25, SOFT_SAGE],
    [0.55, NILE_TEAL],
    [0.85, DEEP_NAVY],
    [1.0, SANDSTONE_GOLD],
]


# -----------------------------------------------------------------------------
# 2. CAGR VS MOIC CALCULATION
# -----------------------------------------------------------------------------
def calculate_cagr_and_moic(
    price_initial: float,
    price_terminal: float,
    horizon_years: int = 10
) -> Dict[str, float]:
    """
    Calculates Equity Multiple (MOIC) and Compound Annual Growth Rate (CAGR).

    In real estate finance:
    - 1.84x is an Equity Multiple (MOIC), NOT an annual growth rate.
    - Over 10 years: CAGR = (1.84)^(1/10) - 1 = 6.29%.

    Args:
        price_initial: Initial acquisition price or invested equity (must be > 0).
        price_terminal: Terminal value or total proceeds (must be >= 0).
        horizon_years: Investment holding duration in years (default 10).

    Returns:
        Dict[str, float]:
            "moic": Multiple on Invested Capital (e.g. 1.84)
            "cagr_pct": Compound annual growth rate percentage (e.g. 6.29)
            "nominal_gain": Absolute nominal gain in currency units (terminal - initial)

    Raises:
        ValueError: If price_initial <= 0 or horizon_years <= 0 or price_terminal < 0.
    """
    if price_initial <= 0:
        raise ValueError(f"price_initial must be strictly positive, got {price_initial}")
    if horizon_years <= 0:
        raise ValueError(f"horizon_years must be strictly positive, got {horizon_years}")
    if price_terminal < 0:
        raise ValueError(f"price_terminal cannot be negative, got {price_terminal}")

    moic = price_terminal / price_initial

    if price_terminal == 0:
        cagr_pct = -100.0
    else:
        cagr_raw = (moic ** (1.0 / horizon_years)) - 1.0
        cagr_pct = round(cagr_raw * 100.0, 2)

    nominal_gain = round(price_terminal - price_initial, 2)
    moic_rounded = round(moic, 4)

    return {
        "moic": moic_rounded,
        "cagr_pct": cagr_pct,
        "nominal_gain": nominal_gain,
    }


# Convenience alias
calculate_moic_and_cagr = calculate_cagr_and_moic


# -----------------------------------------------------------------------------
# 3. EXACT INTERNAL RATE OF RETURN (IRR) SOLVER
# -----------------------------------------------------------------------------
def compute_exact_irr(
    cash_flows: Union[List[float], np.ndarray],
    as_pct: bool = True,
    max_iter: int = 150,
    tol: float = 1e-7,
    guess: float = 0.10
) -> float:
    """
    Computes the exact Internal Rate of Return (IRR) for an irregular or regular
    annual cash flow stream using a robust Newton-Raphson solver.

    Includes step damping to prevent divergence into r <= -1.0, derivative zero-crossing
    protection, multi-guess restarts, and Brent's method root-finding fallback.

    Args:
        cash_flows: Sequence of cash flows where cash_flows[t] is net cash flow at time t.
                    Convention: cash_flows[0] is typically negative (initial equity outflow).
        as_pct: If True (default), returns the rate in percentage (e.g. 10.0 for 10.0%).
                If False, returns decimal fraction (e.g. 0.10).
        max_iter: Maximum Newton-Raphson iteration cycles.
        tol: Convergence tolerance on Net Present Value (NPV).
        guess: Initial trial discount rate (default 0.10 = 10%).

    Returns:
        float: Solved IRR (percentage or decimal). Returns 0.0 if cash flows have no sign change.
    """
    cfs = np.asarray(cash_flows, dtype=np.float64)

    # Edge cases: need at least 2 flows and at least one positive and one negative
    if len(cfs) < 2 or not ((cfs < 0).any() and (cfs > 0).any()):
        return 0.0

    n = len(cfs)
    t = np.arange(n, dtype=np.float64)

    # Multi-start initial guesses if first fails
    candidate_guesses = [guess, 0.05, 0.20, 0.0, -0.05, 0.50]

    for trial_guess in candidate_guesses:
        r = float(trial_guess)
        converged = False

        for _ in range(max_iter):
            # Guard domain: (1 + r) must stay positive
            if r <= -0.999:
                r = -0.90

            denom = (1.0 + r) ** t
            npv = np.sum(cfs / denom)

            if abs(npv) < tol:
                converged = True
                break

            denom_deriv = (1.0 + r) ** (t + 1.0)
            npv_prime = np.sum(-t * cfs / denom_deriv)

            # Derivative near zero protection
            if abs(npv_prime) < 1e-13:
                r += 0.02
                continue

            delta = -npv / npv_prime

            # Backtracking step damping to ensure r does not cross <= -0.99
            step = 1.0
            while (r + step * delta <= -0.99) and (step > 1e-6):
                step *= 0.5

            r += step * delta

            if abs(step * delta) < tol and abs(npv) < 1e-4:
                converged = True
                break

        if converged and not np.isnan(r) and not np.isinf(r):
            final_val = r * 100.0 if as_pct else r
            if abs(final_val) < 1e-6:
                final_val = 0.0
            return round(final_val, 4)

    # Fallback to Brent's method from scipy.optimize if available
    try:
        from scipy.optimize import brentq

        def npv_func(rate: float) -> float:
            return float(np.sum(cfs / ((1.0 + rate) ** t)))

        # Search brackets
        for bracket in [(-0.95, 2.0), (-0.95, 10.0), (-0.99, 100.0)]:
            try:
                f_a = npv_func(bracket[0])
                f_b = npv_func(bracket[1])
                if f_a * f_b <= 0:
                    r_brent = brentq(npv_func, bracket[0], bracket[1], xtol=tol)
                    final_val = r_brent * 100.0 if as_pct else r_brent
                    if abs(final_val) < 1e-6:
                        final_val = 0.0
                    return round(final_val, 4)
            except Exception:
                continue
    except ImportError:
        pass

    # Return best estimate
    final_val = r * 100.0 if as_pct else r
    if abs(final_val) < 1e-6:
        final_val = 0.0
    return round(final_val, 4)


# -----------------------------------------------------------------------------
# 4. NEGOTIATION DISCOUNT / HAIRCUT
# -----------------------------------------------------------------------------
def apply_negotiation_discount(
    asking_price: float,
    discount_pct: float = 7.5
) -> float:
    """
    Applies an institutional negotiation haircut discount to portal asking prices
    to derive estimated transacted execution clearing levels.

    Args:
        asking_price: Portal listed asking price in EGP (must be >= 0).
        discount_pct: Negotiation haircut percentage (default 7.5 for 7.5%).
                      Can be provided as 7.5 (percent) or 0.075 (decimal fraction).

    Returns:
        float: Estimated transacted execution price in EGP, rounded to 2 decimal places.

    Raises:
        ValueError: If asking_price < 0 or discount_pct < 0 or discount rate > 100%.
    """
    if asking_price < 0:
        raise ValueError(f"asking_price cannot be negative, got {asking_price}")
    if discount_pct < 0:
        raise ValueError(f"discount_pct cannot be negative, got {discount_pct}")

    # Handle percentage vs decimal fraction:
    # If passed as 7.5 -> 0.075
    # If passed as 0.075 -> 0.075
    if discount_pct > 1.0:
        rate = discount_pct / 100.0
    elif 0.0 < discount_pct <= 0.30:
        # Values like 0.075 or 0.10 are treated as decimal fractions
        rate = discount_pct
    else:
        rate = discount_pct / 100.0

    if rate > 1.0:
        raise ValueError(f"Discount rate cannot exceed 100%, got {rate * 100:.1f}%")

    execution_price = asking_price * (1.0 - rate)
    return round(execution_price, 2)


# -----------------------------------------------------------------------------
# 5. INSTITUTIONAL 2-WAY SENSITIVITY MATRIX GENERATOR
# -----------------------------------------------------------------------------
def generate_sensitivity_matrix(
    base_price: float,
    gross_yield_pct: float = 7.0,
    opex_pct: float = 15.0,
    equity_pct: float = 100.0,
    horizon_years: int = 10,
    exit_cap_rates: Optional[List[float]] = None,
    rental_growth_rates: Optional[List[float]] = None,
    remaining_debt: Optional[float] = None,
    debt_interest_rate_pct: float = 0.0,
) -> Dict[str, Any]:
    """
    Generates an institutional 2-way underwriting sensitivity matrix across:
      - Vertical Axis: Exit Cap Rate (%) (e.g. [5.5%, 6.5%, 7.5%, 8.5%, 9.5%])
      - Horizontal Axis: Annual Rental Growth Rate (%) (e.g. [3.0%, 5.0%, 7.0%, 9.0%, 11.0%])

    For each (Exit Cap, Rental Growth) pair:
      1. Year-by-year Net Operating Income (NOI) = Initial Rent * (1 + growth)^t
      2. Exit Valuation = NOI_T / Exit_Cap_Rate
      3. Total Net Cash Flows = Cumulative Net Rent + (Exit Valuation - Remaining Debt)
      4. Leveraged Equity Multiple (MOIC) = Total Returned Equity / Initial Equity Down Payment
      5. Leveraged IRR (%) solved via Newton-Raphson.

    Args:
        base_price: Initial property purchase price in EGP (must be > 0).
        gross_yield_pct: Initial gross rental yield percentage (e.g. 7.0%).
        opex_pct: Operating expense ratio (% of gross rent, e.g. 15.0%).
        equity_pct: Equity down payment percentage (e.g. 40.0% leveraged or 100.0% unleveraged).
        horizon_years: Investment holding horizon in years (default 10).
        exit_cap_rates: List of terminal exit capitalization rates (%)
                        Default: [5.5, 6.5, 7.5, 8.5, 9.5].
        rental_growth_rates: List of annual rental escalation rates (%)
                             Default: [3.0, 5.0, 7.0, 9.0, 11.0].
        remaining_debt: Explicit balloon debt balance at exit. If None, defaults to:
                        0 if equity_pct >= 100%, else initial debt balance.
        debt_interest_rate_pct: Annual borrowing interest rate on debt (default 0.0%).

    Returns:
        Dict[str, Any] containing:
            - "irr_matrix": 2D np.ndarray of Leveraged IRR (%) [M x N]
            - "moic_matrix": 2D np.ndarray of Leveraged Equity MOIC (x) [M x N]
            - "exit_val_matrix": 2D np.ndarray of Projected Exit Valuation (EGP) [M x N]
            - "exit_val_m_matrix": 2D np.ndarray of Exit Valuation in Millions EGP [M x N]
            - "exit_cap_rates": list of Exit Cap Rates (%)
            - "rental_growth_rates": list of Rental Growth Rates (%)
            - "df_irr": pd.DataFrame of IRRs
            - "df_moic": pd.DataFrame of MOICs
            - "df_exit_val": pd.DataFrame of Exit Valuations (Millions EGP)
            - "initial_equity": Initial equity invested (EGP)
            - "initial_debt": Initial debt borrowed (EGP)
            - "remaining_debt": Terminal debt payoff at exit (EGP)
            - "initial_net_rent": Year 0 stabilized net operating income (EGP)
            - "horizon_years": Holding period in years
    """
    if base_price <= 0:
        raise ValueError(f"base_price must be strictly positive, got {base_price}")
    if horizon_years <= 0:
        raise ValueError(f"horizon_years must be strictly positive, got {horizon_years}")

    # Standardize default grids
    if exit_cap_rates is None:
        exit_cap_rates = [5.5, 6.5, 7.5, 8.5, 9.5]
    if rental_growth_rates is None:
        rental_growth_rates = [3.0, 5.0, 7.0, 9.0, 11.0]

    # Normalize percentage parameters
    gy_ratio = gross_yield_pct / 100.0 if gross_yield_pct > 1.0 else gross_yield_pct
    opex_ratio = opex_pct / 100.0 if opex_pct > 1.0 else opex_pct
    eq_ratio = equity_pct / 100.0 if equity_pct > 1.0 else equity_pct
    eq_ratio = max(0.01, min(1.0, eq_ratio))

    # Capital Structure
    initial_equity = base_price * eq_ratio
    initial_debt = base_price * (1.0 - eq_ratio)

    if remaining_debt is not None:
        actual_rem_debt = float(remaining_debt)
    else:
        # If unleveraged (equity >= 100%), debt is 0. If leveraged, interest-only balloon debt is repaid at exit.
        actual_rem_debt = 0.0 if eq_ratio >= 0.999 else initial_debt

    # Annual Debt Service
    interest_rate = debt_interest_rate_pct / 100.0 if debt_interest_rate_pct > 1.0 else debt_interest_rate_pct
    annual_debt_service = initial_debt * interest_rate

    # Initial Net Operating Income (Year 0 stabilized net rent)
    initial_gross_rent = base_price * gy_ratio
    initial_net_rent = initial_gross_rent * (1.0 - opex_ratio)

    m = len(exit_cap_rates)
    n = len(rental_growth_rates)

    irr_mat = np.zeros((m, n), dtype=np.float64)
    moic_mat = np.zeros((m, n), dtype=np.float64)
    exit_val_mat = np.zeros((m, n), dtype=np.float64)
    exit_val_m_mat = np.zeros((m, n), dtype=np.float64)

    for i, cap_rate in enumerate(exit_cap_rates):
        cap_dec = cap_rate / 100.0 if cap_rate > 1.0 else cap_rate
        if cap_dec <= 0:
            raise ValueError(f"Exit cap rate must be positive, got {cap_rate}")

        for j, growth_rate in enumerate(rental_growth_rates):
            g_dec = growth_rate / 100.0 if growth_rate > 1.0 else growth_rate

            # 1. Year-by-year Net Operating Income (NOI) = Initial Rent * (1 + growth)^t
            # and net operating cash flows after debt service
            net_cash_flows_annual: List[float] = []
            for t in range(1, horizon_years + 1):
                noi_t = initial_net_rent * ((1.0 + g_dec) ** t)
                net_cf_t = noi_t - annual_debt_service
                net_cash_flows_annual.append(net_cf_t)

            # Terminal year NOI
            noi_t_exit = initial_net_rent * ((1.0 + g_dec) ** horizon_years)

            # 2. Exit Valuation = NOI_T / Exit_Cap_Rate
            exit_valuation = noi_t_exit / cap_dec

            # 3. Total Net Cash Flows = Cumulative Net Rent + (Exit Valuation - Remaining Debt)
            cumulative_net_rent = sum(net_cash_flows_annual)
            terminal_equity_proceeds = exit_valuation - actual_rem_debt
            total_returned_equity = cumulative_net_rent + terminal_equity_proceeds

            # 4. Leveraged Equity Multiple (MOIC) = Total Returned Equity / Initial Equity Down Payment
            moic = total_returned_equity / initial_equity if initial_equity > 0 else (total_returned_equity / base_price)

            # 5. Leveraged Cash Flows array for Newton-Raphson IRR
            irr_cfs = [-initial_equity] + net_cash_flows_annual[:-1] + [
                net_cash_flows_annual[-1] + terminal_equity_proceeds
            ]
            irr_val = compute_exact_irr(irr_cfs, as_pct=True)

            irr_mat[i, j] = round(irr_val, 2)
            moic_mat[i, j] = round(moic, 2)
            exit_val_mat[i, j] = round(exit_valuation, 2)
            exit_val_m_mat[i, j] = round(exit_valuation / 1_000_000.0, 2)

    # Formatted DataFrames for inspection / tabular reporting
    row_labels = [f"{c:.1f}%" for c in exit_cap_rates]
    col_labels = [f"{g:.1f}%" for g in rental_growth_rates]

    df_irr = pd.DataFrame(irr_mat, index=row_labels, columns=col_labels)
    df_moic = pd.DataFrame(moic_mat, index=row_labels, columns=col_labels)
    df_exit_val = pd.DataFrame(exit_val_m_mat, index=row_labels, columns=col_labels)

    return {
        "irr_matrix": irr_mat,
        "moic_matrix": moic_mat,
        "exit_val_matrix": exit_val_mat,
        "exit_val_m_matrix": exit_val_m_mat,
        "exit_cap_rates": [float(c) for c in exit_cap_rates],
        "rental_growth_rates": [float(g) for g in rental_growth_rates],
        "df_irr": df_irr,
        "df_moic": df_moic,
        "df_exit_val": df_exit_val,
        "initial_equity": round(initial_equity, 2),
        "initial_debt": round(initial_debt, 2),
        "remaining_debt": round(actual_rem_debt, 2),
        "initial_net_rent": round(initial_net_rent, 2),
        "horizon_years": int(horizon_years),
    }


# -----------------------------------------------------------------------------
# 6. EXECUTIVE SENSITIVITY HEATMAP VISUALIZATION
# -----------------------------------------------------------------------------
def render_sensitivity_heatmap(
    base_price: float,
    gross_yield_pct: float = 7.0,
    opex_pct: float = 15.0,
    equity_pct: float = 100.0,
    horizon_years: int = 10,
    height: int = 340,
    exit_cap_rates: Optional[List[float]] = None,
    rental_growth_rates: Optional[List[float]] = None,
    remaining_debt: Optional[float] = None,
    debt_interest_rate_pct: float = 0.0,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Renders an institutional 2-way underwriting sensitivity heatmap Plotly Figure
    styled in the Egyptian Executive Palette.

    Visual Specifications:
      - Vertical Axis: Exit Cap Rate (%) (e.g. [5.5%, 6.5%, 7.5%, 8.5%, 9.5%])
      - Horizontal Axis: Annual Rental Growth Rate (%) (e.g. [3.0%, 5.0%, 7.0%, 9.0%, 11.0%])
      - Colorscale: Egyptian Executive Palette (Nile Teal #2A9D8F, Deep Navy #1B2A4A,
                    Sandstone Gold #E9B44C, Soft Sage #8FB996, Off-White #F7F5F2, Charcoal #333333).
      - Cell Annotation:
          <b>{irr:.1f}%</b><br><span style='font-size:9px;'>{moic:.2f}x</span>
      - Hovertemplate:
          <b>Exit Cap Rate: %{y:.1f}% | Rental Growth: %{x:.1f}%</b><br>
          Projected Leveraged IRR: <b>%{customdata[0]:.1f}%</b><br>
          Equity Multiple (MOIC): <b>%{customdata[1]:.2f}x</b><br>
          Projected Exit Valuation: <b>%{customdata[2]:,.1f}M EGP</b><extra></extra>
      - Axes Titles: "Rental Growth Rate (%)" and "Exit Cap Rate (%)"
      - Title: "📊 Investment Return Sensitivity (Exit Cap vs. Rental Growth)"

    Args:
        base_price: Acquisition price or benchmark value in EGP.
        gross_yield_pct: Gross rental yield percentage.
        opex_pct: Operating expense ratio (% of gross rent).
        equity_pct: Equity down payment percentage (40% leveraged, 100% unleveraged).
        horizon_years: Investment holding horizon in years (default 10).
        height: Plotly figure height in pixels (default 340).
        exit_cap_rates: Optional custom list of Exit Cap Rates (%).
        rental_growth_rates: Optional custom list of Rental Growth Rates (%).
        remaining_debt: Optional balloon debt balance.
        debt_interest_rate_pct: Borrowing rate for debt service.
        title: Optional custom figure title.

    Returns:
        go.Figure: Ready-to-render Plotly Heatmap Figure.
    """
    # 1. Compute institutional sensitivity matrix
    matrix_data = generate_sensitivity_matrix(
        base_price=base_price,
        gross_yield_pct=gross_yield_pct,
        opex_pct=opex_pct,
        equity_pct=equity_pct,
        horizon_years=horizon_years,
        exit_cap_rates=exit_cap_rates,
        rental_growth_rates=rental_growth_rates,
        remaining_debt=remaining_debt,
        debt_interest_rate_pct=debt_interest_rate_pct,
    )

    caps = matrix_data["exit_cap_rates"]
    growths = matrix_data["rental_growth_rates"]
    irr_mat = matrix_data["irr_matrix"]
    moic_mat = matrix_data["moic_matrix"]
    exit_val_m_mat = matrix_data["exit_val_m_matrix"]

    m = len(caps)
    n = len(growths)

    # 2. Build cell text annotations and 3D customdata array
    text_matrix: List[List[str]] = []
    customdata: List[List[List[float]]] = []

    for i in range(m):
        row_text: List[str] = []
        row_cd: List[List[float]] = []
        for j in range(n):
            irr = float(irr_mat[i, j])
            moic = float(moic_mat[i, j])
            val_m = float(exit_val_m_mat[i, j])

            # Exact prompt requirement:
            # <b>{irr:.1f}%</b><br><span style='font-size:9px;'>{moic:.2f}x</span>
            cell_label = f"<b>{irr:.1f}%</b><br><span style='font-size:9px;'>{moic:.2f}x</span>"
            row_text.append(cell_label)
            row_cd.append([irr, moic, val_m])

        text_matrix.append(row_text)
        customdata.append(row_cd)

    # 3. Exact custom hovertemplate specified in requirements
    hovertemplate = (
        "<b>Exit Cap Rate: %{y:.1f}% | Rental Growth: %{x:.1f}%</b><br>"
        "Projected Leveraged IRR: <b>%{customdata[0]:.1f}%</b><br>"
        "Equity Multiple (MOIC): <b>%{customdata[1]:.2f}x</b><br>"
        "Projected Exit Valuation: <b>%{customdata[2]:,.1f}M EGP</b>"
        "<extra></extra>"
    )

    # 4. Construct Plotly Heatmap Trace
    heatmap_trace = go.Heatmap(
        x=growths,
        y=caps,
        z=irr_mat,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            size=10,
        ),
        customdata=customdata,
        hovertemplate=hovertemplate,
        colorscale=EGYPTIAN_SENSITIVITY_SCALE,
        colorbar=dict(
            title=dict(
                text="IRR %",
                font=dict(size=9, color=CHARCOAL, family="Inter, sans-serif"),
                side="top",
            ),
            tickfont=dict(size=8, color=CHARCOAL),
            thickness=12,
            len=0.9,
            ticksuffix="%",
            outlinewidth=0,
            x=1.02,
        ),
        showscale=True,
    )

    fig = go.Figure(data=[heatmap_trace])

    fig_title = title or "📊 Investment Return Sensitivity (Exit Cap vs. Rental Growth)"

    # 5. Apply Egyptian Executive Layout Defaults
    fig.update_layout(
        title=dict(
            text=fig_title,
            font=dict(
                color=DEEP_NAVY,
                size=11,
                family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                weight="bold",
            ),
            x=0.01,
            y=0.97,
        ),
        xaxis=dict(
            title=dict(
                text="Rental Growth Rate (%)",
                font=dict(size=9, color=CHARCOAL, family="Inter, sans-serif", weight="bold"),
            ),
            tickmode="array",
            tickvals=growths,
            ticktext=[f"{g:.1f}%" for g in growths],
            tickfont=dict(size=8, color=CHARCOAL),
            gridcolor=LIGHT_GRAY,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="Exit Cap Rate (%)",
                font=dict(size=9, color=CHARCOAL, family="Inter, sans-serif", weight="bold"),
            ),
            tickmode="array",
            tickvals=caps,
            ticktext=[f"{c:.1f}%" for c in caps],
            tickfont=dict(size=8, color=CHARCOAL),
            gridcolor=LIGHT_GRAY,
            autorange="reversed",  # Standard institutional matrix layout: low cap at top
            zeroline=False,
        ),
        height=height,
        paper_bgcolor=OFF_WHITE,
        plot_bgcolor="#FFFFFF",
        margin=dict(l=45, r=50, t=35, b=35),
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color=CHARCOAL,
            size=9,
        ),
        uirevision="sensitivity_matrix",
        transition=dict(duration=350, easing="cubic-in-out"),
    )

    return fig

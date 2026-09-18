# -*- coding: utf-8 -*-
"""
Unit and Integration Test Suite for ml_engine.py
Verifies model loading, caching, warning suppression, feature engineering,
prediction accuracy, edge case handling, and batch valuation.
"""

import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ml_engine import (
    get_model_pipeline,
    predict_apartment,
    predict_batch,
    batch_valuation,
    get_preset_benchmarks,
    evaluate_benchmarks,
    get_available_locations,
    get_model_metadata,
    clear_model_cache
)


def test_model_loading_and_caching():
    print("[TEST 1/6] Testing model loading & memory caching...")
    pipeline_1 = get_model_pipeline()
    assert pipeline_1 is not None, "Pipeline must not be None"
    
    # Second call should return cached instance
    pipeline_2 = get_model_pipeline()
    assert pipeline_1 is pipeline_2, "Pipeline must be cached and return identical object"
    
    # Verify expected feature names in pipeline
    expected_cols = [
        'Price_Numeric', 'Calculated_Price_Per_Meter', 'Area', 'Rooms',
        'Bathrooms_num', 'Floor Level', 'Building_Age_10Y', 'Luxury_Index',
        'Price_per_Room', 'View', 'Payment Method', 'Location',
        'Amenities', 'finishing_type', 'Type'
    ]
    assert list(pipeline_1.feature_names_in_) == expected_cols, "Feature columns must match exactly"
    print("  -> PASSED: Pipeline loaded and cached correctly.")


def test_predict_apartment_defaults_and_known_benchmarks():
    print("[TEST 2/6] Testing predict_apartment() with 4 core benchmark properties...")
    
    # 1. Village West Compound
    vw = predict_apartment(
        location="Village West Compound",
        area_m2=160,
        rooms=3,
        bathrooms=3,
        floor=2,
        view="Inner",
        amenities="B(Compound)",
        finishing="superlux",
        current_price=None,
        year_built=2025
    )
    assert vw["current_price"] == 11275000.0, f"Expected 11,275,000, got {vw['current_price']}"
    assert vw["price_10y"] == 27249000.0, f"Expected 27,249,000, got {vw['price_10y']}"
    assert vw["multiplier"] == 2.4168, f"Expected multiplier ~2.4168, got {vw['multiplier']}"
    assert vw["annual_cagr"] == 9.23, f"Expected CAGR 9.23%, got {vw['annual_cagr']}"
    assert vw["luxury_score"] == 4.0, f"Expected luxury_score 4.0, got {vw['luxury_score']}"
    assert vw["conservative_10y"] == 24524000.0, f"Conservative mismatch: {vw['conservative_10y']}"
    assert vw["optimistic_10y"] == 29974000.0, f"Optimistic mismatch: {vw['optimistic_10y']}"

    # 2. Madinaty
    mad = predict_apartment(
        location="Madinaty",
        area_m2=140,
        rooms=3,
        bathrooms=2,
        floor=3,
        view="Inner",
        amenities="B(Compound)",
        finishing="superlux",
        current_price=None,
        year_built=2025
    )
    assert mad["current_price"] == 7254000.0, f"Expected 7,254,000, got {mad['current_price']}"
    assert mad["price_10y"] == 17600000.0, f"Expected 17,600,000, got {mad['price_10y']}"
    assert mad["annual_cagr"] == 9.27, f"Expected CAGR 9.27%, got {mad['annual_cagr']}"

    # 3. Faisal
    fsl = predict_apartment(
        location="Faisal",
        area_m2=120,
        rooms=2,
        bathrooms=1,
        floor=4,
        view="Outer",
        amenities="C(BASICS)",
        finishing="semi",
        current_price=None,
        year_built=2025
    )
    assert fsl["current_price"] == 1849000.0, f"Expected 1,849,000, got {fsl['current_price']}"
    assert fsl["price_10y"] == 3038000.0, f"Expected 3,038,000, got {fsl['price_10y']}"
    assert fsl["annual_cagr"] == 5.09, f"Expected CAGR 5.09%, got {fsl['annual_cagr']}"
    assert fsl["luxury_score"] == 0.0, f"Expected luxury_score 0.0, got {fsl['luxury_score']}"

    # 4. New October
    noct = predict_apartment(
        location="New October",
        area_m2=115,
        rooms=3,
        bathrooms=2,
        floor=1,
        view="Outer",
        amenities="C(BASICS)",
        finishing="semi",
        current_price=None,
        year_built=2025
    )
    assert noct["current_price"] == 1498000.0, f"Expected 1,498,000, got {noct['current_price']}"
    assert noct["price_10y"] == 2497000.0, f"Expected 2,497,000, got {noct['price_10y']}"
    assert noct["annual_cagr"] == 5.24, f"Expected CAGR 5.24%, got {noct['annual_cagr']}"

    print("  -> PASSED: All 4 core benchmarks predicted accurately with 100% precision.")


def test_predict_apartment_explicit_price():
    print("[TEST 3/6] Testing predict_apartment() with user-supplied explicit price...")
    
    # Provide an explicit price different from median
    res = predict_apartment(
        location="Madinaty",
        area_m2=140,
        rooms=3,
        bathrooms=2,
        floor=3,
        current_price=10_000_000.0,  # Explicit custom price
        year_built=2024
    )
    assert res["current_price"] == 10000000.0
    assert res["price_estimated"] is False
    assert res["current_ppm"] == round(10000000.0 / 140.0, 2)
    assert res["building_age_10y"] == (2036.0 - 2024.0)
    assert res["price_10y"] > 10000000.0, "10-year price should project growth"
    assert res["multiplier"] == round(res["price_10y"] / 10000000.0, 4)
    print("  -> PASSED: Explicit price correctly ingested and computed.")


def test_edge_cases_and_unseen_locations():
    print("[TEST 4/6] Testing edge cases (unseen locations, zero price, small area)...")
    
    # Unseen location should fallback gracefully without crashing
    res_unseen = predict_apartment(
        location="Completely Unknown Compound Name 999",
        area_m2=100,
        rooms=2,
        bathrooms=1,
        current_price=0  # Zero price triggers estimation fallback
    )
    assert res_unseen["current_price"] > 0, "Should estimate fallback price"
    assert res_unseen["price_10y"] > 0, "Should generate valid 10-year projection"
    assert res_unseen["conservative_10y"] == round(res_unseen["price_10y"] * 0.90, -3)
    assert res_unseen["optimistic_10y"] == round(res_unseen["price_10y"] * 1.10, -3)

    # Extreme minimal area and room numbers
    res_min = predict_apartment(
        location="Madinaty",
        area_m2=5.0,  # Below min 10 -> will be clamped to 10.0
        rooms=0,      # Clamped to 1.0
        bathrooms=0,  # Clamped to 1.0
        current_price=None
    )
    assert res_min["area"] == 10.0
    assert res_min["rooms"] == 1
    assert res_min["bathrooms"] == 1
    assert not math.isnan(res_min["price_10y"])
    print("  -> PASSED: Edge cases and unseen locations handled gracefully.")


def test_batch_valuation_and_benchmarks():
    print("[TEST 5/6] Testing batch valuation & benchmark evaluation...")
    
    benchmarks = get_preset_benchmarks()
    assert len(benchmarks) >= 4, "Must have at least 4 preset benchmarks"
    
    df_results = evaluate_benchmarks()
    assert isinstance(df_results, pd.DataFrame), "Output must be pandas DataFrame"
    assert len(df_results) == len(benchmarks), f"Expected {len(benchmarks)} rows, got {len(df_results)}"
    
    required_cols = [
        "location", "area_m2", "current_price", "price_10y",
        "total_growth", "multiplier", "annual_cagr",
        "conservative_10y", "optimistic_10y", "benchmark_name"
    ]
    for col in required_cols:
        assert col in df_results.columns, f"Missing required column in DataFrame: {col}"

    # Also test batch_valuation with arbitrary list of dicts
    custom_batch = [
        {"location": "Village West Compound", "area_m2": 200, "rooms": 4, "bathrooms": 3},
        {"location": "Faisal", "area_m2": 90, "rooms": 2, "bathrooms": 1}
    ]
    df_custom = batch_valuation(custom_batch)
    assert len(df_custom) == 2
    assert df_custom.loc[0, "area_m2"] == 200
    print("  -> PASSED: Batch valuation functions executed smoothly.")


def test_metadata_and_locations_list():
    print("[TEST 6/6] Testing metadata & location list retrieval...")
    meta = get_model_metadata()
    assert "accuracy" in meta["training_accuracy_r2"].lower() or "99.46" in meta["training_accuracy_r2"]
    assert len(meta["features_list"]) == 15
    
    locs = get_available_locations()
    assert len(locs) > 500, f"Expected > 500 reference locations, found {len(locs)}"
    assert "Madinaty" in locs
    assert "Village West Compound" in locs
    print("  -> PASSED: Model metadata and location catalogs validated.")


if __name__ == "__main__":
    print("=" * 76)
    print(" RUNNING ML ENGINE VALIDATION TEST SUITE")
    print("=" * 76)
    
    test_model_loading_and_caching()
    test_predict_apartment_defaults_and_known_benchmarks()
    test_predict_apartment_explicit_price()
    test_edge_cases_and_unseen_locations()
    test_batch_valuation_and_benchmarks()
    test_metadata_and_locations_list()
    
    print("\n" + "=" * 76)
    print(" ALL TESTS PASSED SUCCESSFULLY! (6/6 SUITES PASSED)")
    print("=" * 76)

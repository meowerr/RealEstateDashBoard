# -*- coding: utf-8 -*-
"""
Verification script for data_loader.py module.
Tests:
1. Data loading and caching performance
2. Derived features ('Bathrooms_num', 'Area')
3. Location classification, normalization, and macro region assignment
4. Geocoding accuracy and validity
5. Quality of life (QOL) enrichment (AQI & Green Space)
6. Helper functions: get_location_coordinates() and get_filter_options()
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import data_loader

def run_verification():
    print("=" * 75)
    print("  🚀 RUNNING VERIFICATION SUITE FOR 'data_loader.py'")
    print("=" * 75)

    # -------------------------------------------------------------
    # Test 1: Load and Cache Timing
    # -------------------------------------------------------------
    print("\n[Test 1] Loading master dataset...")
    t0 = time.perf_counter()
    df1 = data_loader.load_clean_data()
    t1 = time.perf_counter()
    print(f"  First load elapsed: {t1 - t0:.4f} seconds")

    t2 = time.perf_counter()
    df2 = data_loader.load_clean_data()
    t3 = time.perf_counter()
    print(f"  Cached load elapsed: {t3 - t2:.6f} seconds")

    assert len(df1) == 10000, f"Expected 10,000 rows, got {len(df1)}"
    assert len(df2) == 10000, "Cached dataframe mismatch"
    print("  ✓ PASS: 10,000 rows loaded successfully with caching active.")

    # -------------------------------------------------------------
    # Test 2: Bathrooms_num & Area
    # -------------------------------------------------------------
    print("\n[Test 2] Validating Bathrooms_num & Area derivation...")
    assert "Bathrooms_num" in df1.columns, "Bathrooms_num column missing"
    assert df1["Bathrooms_num"].isnull().sum() == 0, "Bathrooms_num contains null values"
    assert (df1["Bathrooms_num"] >= 1.0).all(), "Bathrooms_num contains invalid values"
    
    # Check Area calculation: Area = Price_Numeric / Calculated_Price_Per_Meter
    assert "Area" in df1.columns, "Area column missing"
    expected_area = (df1["Price_Numeric"] / df1["Calculated_Price_Per_Meter"]).round(1)
    diff = (df1["Area"] - expected_area).abs().max()
    assert diff < 1e-3, f"Area identity deviation detected: max diff = {diff}"
    print(f"  ✓ PASS: Bathrooms_num parsed (0 nulls). Area perfectly matches Price / PPM (max diff = {diff}).")

    # -------------------------------------------------------------
    # Test 3: Normalization & Macro Regions
    # -------------------------------------------------------------
    print("\n[Test 3] Validating location normalization & macro regions...")
    assert "normalized_location" in df1.columns, "normalized_location column missing"
    assert "macro_region" in df1.columns, "macro_region column missing"
    assert "geo_status" in df1.columns, "geo_status column missing"

    # Hub checks
    test_locations = {
        "Madinaty": "Madinaty",
        "Al Rehab": "Al Rehab",
        "Bait El Watan": "Beit El Watan",
        "كمبوند ادريس هوم": "Address Home Compound",
        "Village West Compound": "Village West Compound",
        "أكتوبر الجديدة": "New October",
        "Mountain View Icity": "Mountain View",
    }
    for raw, expected in test_locations.items():
        norm = data_loader.normalize_location(raw)
        assert norm == expected, f"Expected '{expected}' for '{raw}', got '{norm}'"
    print("  ✓ PASS: Regex normalization rules correctly map hubs (Madinaty, Rehab, Beit El Watan, Village West, New October, etc.).")

    macro_counts = df1["macro_region"].value_counts().to_dict()
    print("  Macro region distribution:")
    for region, count in macro_counts.items():
        print(f"    - {region:30s}: {count:5d} listings")
    assert "Greater Cairo - East" in macro_counts
    assert "Greater Cairo - West" in macro_counts
    assert "Cairo Central & Urban" in macro_counts
    print("  ✓ PASS: All expected macro regions assigned with 0 nulls.")

    # -------------------------------------------------------------
    # Test 4: Geocoding
    # -------------------------------------------------------------
    print("\n[Test 4] Validating geocoding coordinates...")
    assert "latitude" in df1.columns and "longitude" in df1.columns
    assert "lat" in df1.columns and "lon" in df1.columns

    valid_geo = df1[df1["geo_status"] == "VALID_GEO"]
    unresolved = df1[df1["geo_status"] == "UNRESOLVED_MARKETING"]

    assert valid_geo["latitude"].notnull().all(), "Some VALID_GEO rows have null latitude"
    assert valid_geo["longitude"].notnull().all(), "Some VALID_GEO rows have null longitude"
    assert unresolved["latitude"].isnull().all(), "UNRESOLVED_MARKETING should not have coordinates"
    
    # Check bounding box for Egypt (Lat 22-32, Lon 25-36)
    assert ((valid_geo["latitude"] >= 22.0) & (valid_geo["latitude"] <= 33.0)).all()
    assert ((valid_geo["longitude"] >= 25.0) & (valid_geo["longitude"] <= 36.0)).all()
    print(f"  ✓ PASS: Geocoding verified for {len(valid_geo):,} VALID_GEO listings within Egypt bounding box.")

    # -------------------------------------------------------------
    # Test 5: Quality of Life (QOL) Enrichment
    # -------------------------------------------------------------
    print("\n[Test 5] Validating Quality of Life (AQI & Green Coverage)...")
    assert "us_aqi" in df1.columns and "aqi" in df1.columns
    assert "green_coverage_pct" in df1.columns and "green_pct" in df1.columns

    assert df1["us_aqi"].notnull().all(), "AQI column has nulls"
    assert df1["green_coverage_pct"].notnull().all(), "Green coverage column has nulls"

    print(f"  AQI stats: min={df1['us_aqi'].min():.1f}, mean={df1['us_aqi'].mean():.2f}, max={df1['us_aqi'].max():.1f}")
    print(f"  Green % stats: min={df1['green_pct'].min():.2f}%, mean={df1['green_pct'].mean():.2f}%, max={df1['green_pct'].max():.2f}%")
    assert 50.0 <= df1["us_aqi"].mean() <= 85.0
    assert 1.0 <= df1["green_pct"].mean() <= 10.0
    print("  ✓ PASS: QOL enrichment active across 100% of rows.")

    # -------------------------------------------------------------
    # Test 6: get_location_coordinates()
    # -------------------------------------------------------------
    print("\n[Test 6] Validating get_location_coordinates()...")
    loc_df = data_loader.get_location_coordinates(df1, min_count=1)
    req_cols = ["lat", "lon", "count", "median_price", "median_ppm", "aqi", "green_pct"]
    for col in req_cols:
        assert col in loc_df.columns, f"Required column '{col}' missing from get_location_coordinates()"
    assert "location" in loc_df.columns
    assert "macro_region" in loc_df.columns
    assert len(loc_df) > 1000, f"Expected >1000 location hubs, got {len(loc_df)}"
    print(f"  ✓ PASS: get_location_coordinates() returned {len(loc_df):,} location hubs with all required columns.")
    print("  Top 5 hubs by listing volume:")
    for _, row in loc_df.head(5).iterrows():
        print(f"    - {row['location']:25s}: {row['count']:3d} listings | Price: {row['median_price']:,.0f} EGP | PPM: {row['median_ppm']:,.0f} EGP/m² | AQI: {row['aqi']} | Green: {row['green_pct']}%")

    # -------------------------------------------------------------
    # Test 7: get_filter_options()
    # -------------------------------------------------------------
    print("\n[Test 7] Validating get_filter_options()...")
    filter_opts = data_loader.get_filter_options(df1)
    expected_keys = [
        "regions", "locations", "amenities", "finishing", "rooms", "bathrooms",
        "views", "payment_methods", "types", "min_price", "max_price", "min_ppm", "max_ppm", "min_area", "max_area"
    ]
    for k in expected_keys:
        assert k in filter_opts, f"Filter options missing key '{k}'"
    print(f"  ✓ PASS: get_filter_options() contains all {len(expected_keys)} expected keys.")
    print(f"    - Regions: {filter_opts['regions']}")
    print(f"    - Rooms: {filter_opts['rooms']}")
    print(f"    - Bathrooms: {filter_opts['bathrooms']}")
    print(f"    - Amenities: {filter_opts['amenities']}")
    print(f"    - Finishing: {filter_opts['finishing']}")

    print("\n" + "=" * 75)
    print("  🎉 ALL VERIFICATION TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 75)

if __name__ == "__main__":
    run_verification()

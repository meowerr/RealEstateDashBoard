# -*- coding: utf-8 -*-
"""
Egypt Real Estate Intelligence Platform
Data Loading, Cleaning, Geocoding & QOL Enrichment Module for Streamlit

Exports:
- load_clean_data() -> pd.DataFrame
- get_location_coordinates() -> pd.DataFrame
- get_filter_options() -> dict
- normalize_location() -> str
- classify_location() -> str
- assign_macro_region() -> str
- get_location_coords() -> tuple[float | None, float | None]
"""

import os
import re
import sys
import hashlib
import functools
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

# UTF-8 stream support for Windows console environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# -----------------------------------------------------------------------------
# STREAMLIT CACHING & SINGLETON FALLBACK
# -----------------------------------------------------------------------------
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    st = None
    HAS_STREAMLIT = False


def lazy_cache(func):
    """
    Decorator that applies st.cache_data when running within an active Streamlit runtime,
    and a singleton in-memory cache when running in bare Python scripts or tests.
    """
    _mem_cache: Dict[Any, Any] = {}
    _st_cached = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal _st_cached
        if HAS_STREAMLIT and hasattr(st, "runtime") and st.runtime.exists():
            if _st_cached is None:
                _st_cached = st.cache_data(show_spinner=False)(func)
            return _st_cached(*args, **kwargs)

        key = (args, tuple(sorted(kwargs.items())))
        if key not in _mem_cache:
            _mem_cache[key] = func(*args, **kwargs)
        return _mem_cache[key]

    def clear():
        _mem_cache.clear()
        if _st_cached is not None and hasattr(_st_cached, "clear"):
            try:
                _st_cached.clear()
            except Exception:
                pass

    wrapper.clear = clear
    return wrapper


cache_data = lazy_cache

# -----------------------------------------------------------------------------
# FILE RESOLUTION UTILITIES
# -----------------------------------------------------------------------------
def _resolve_file(filename: str, override_path: Optional[str] = None) -> Path:
    """Dynamically resolves file path across different working directories."""
    if override_path and os.path.exists(override_path):
        return Path(override_path).resolve()

    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / filename,
        Path.cwd() / filename,
        base_dir.parent / filename,
        base_dir / "dashboard" / filename,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        f"Required dataset '{filename}' not found. Searched candidate paths:\n"
        + "\n".join(f" - {c}" for c in candidates)
    )


# -----------------------------------------------------------------------------
# LOCATION TAXONOMY, CLASSIFICATION & NORMALIZATION RULES
# -----------------------------------------------------------------------------
MARKETING_PATTERNS: List[str] = [
    r'^(the\s+)?heart\s+of',
    r'^(a\s+)?prime\s+location',
    r'^(a\s+)?strategic\s+location',
    r'^(an\s+)?exceptional\s+location',
    r'^(a\s+)?full\s+service',
    r'^other\s+neighborhoods\s+in',
    r'^al$',
    r'^(a\s+)?resort\s+with',
    r'^موقع',
    r'^كمبوند\s+ساكن',
    r'^(a\s+)?modern\s+resort',
    r'^(a\s+)?distinctive\s+location',
    r'^an?\s+unparalleled',
    r'^an?\s+integrated'
]

NORM_RULES: List[Tuple[str, str]] = [
    (r'(?i)b[ae]it\s*(al?|el?)\s*watan|بيت\s*الوطن|ببيت\s*الوطن', 'Beit El Watan'),
    (r'(?i)address\s+home|ادريس\s+هوم', 'Address Home Compound'),
    (r'(?i)madinaty|مدينتي', 'Madinaty'),
    (r'(?i)de\s*joya|دي\s*جويا', 'De Joya'),
    (r'(?i)new\s+october|أكتوبر\s+الجديدة|اكتوبر\s+الجديدة', 'New October'),
    (r'(?i)taj\s+city|تاج\s+سيتي', 'Taj City'),
    (r'(?i)al?\s*rehab|الرحاب', 'Al Rehab'),
    (r'(?i)mountain\s+view|ماونتن\s+فيو', 'Mountain View'),
    (r'(?i)village\s+west|فيليدج\s+ويست', 'Village West Compound'),
    (r'(?i)green\s+revolution|الثورة\s+الخضراء', 'Green Revolution'),
    (r'(?i)al?\s*ahyaa|الاحياء', 'Al Ahyaa'),
    (r'(?i)muruj|مروج', 'Muruj'),
    (r'(?i)badya|بادية', 'Badya Compound'),
    (r'(?i)mubarak\s*6|مبارك\s*6', 'Mubarak 6'),
    (r'(?i)scandic|سكانديك', 'Scandic Resort'),
    (r'(?i)sudan\s+st|شارع\s+السودان', 'Sudan Street'),
    (r'(?i)ahmed\s+zewail|احمد\s+زويل', 'Ahmed Zewail Road'),
    (r'(?i)zamalek|زمالك', 'Zamalek'),
    (r'(?i)maadi|معادي', 'Maadi'),
    (r'(?i)nasr\s+city|مدينة\s+نصر', 'Nasr City'),
    (r'(?i)heliopolis|مصر\s+الجديدة', 'Heliopolis'),
    (r'(?i)smouha|سموحة', 'Smouha (Alexandria)'),
    (r'(?i)shorouk|شروق', 'Shorouk City'),
    (r'(?i)badr|بدر', 'Badr City'),
    (r'(?i)unesco.*al\s+muizz|المعز', "Unesco's Al Muizz (Fatimid Cairo)")
]


def classify_location(loc: Any) -> str:
    """Classifies location string as VALID_GEO or UNRESOLVED_MARKETING."""
    loc_clean = str(loc).strip().lower()
    if any(re.search(p, loc_clean) for p in MARKETING_PATTERNS):
        return "UNRESOLVED_MARKETING"
    return "VALID_GEO"


def normalize_location(raw: Any) -> str:
    """Normalizes raw location string into canonical real estate hub entity."""
    raw_str = str(raw).strip()
    for pattern, canon in NORM_RULES:
        if re.search(pattern, raw_str):
            return canon
    return raw_str


def assign_macro_region(norm_loc: str, geo_stat: str) -> str:
    """Maps normalized location into high-level macro region."""
    if geo_stat == "UNRESOLVED_MARKETING":
        return "Unresolved / Non-Geographic"

    nl = norm_loc.lower()

    # 1. Greater Cairo - East
    if any(k in nl for k in [
        'madinaty', 'rehab', 'beit el watan', 'tagamoa', 'new cairo', 'taj city', 'shorouk', 'badr',
        'mostakbal', 'new capital', 'administrative capital', 'العاصمة', 'التجمع', 'القاهرة الجديدة',
        'narges', 'lotus', 'andalus', 'أندلس', 'لوتس', 'نرجس', 'choueifat', 'شويفات', 'patio', 'mivida',
        'hydepark', 'hyde park', 'stone residence', 'al burouj', 'البروج'
    ]):
        return 'Greater Cairo - East'

    # 2. Greater Cairo - West
    if any(k in nl for k in [
        'zayed', 'village west', 'october', 'address home', 'ahyaa', 'badya', 'muruj', 'green revolution',
        'zewail', 'sun capital', 'hadayek october', 'حدائق اكتوبر', 'أكتوبر', 'زايد', 'الشيخ زايد',
        'اكتوبر', 'rovan', 'west view', 'sawari'
    ]):
        return 'Greater Cairo - West'

    # 3. Cairo Central & Urban
    if any(k in nl for k in [
        'maadi', 'nasr city', 'heliopolis', 'zamalek', 'dokki', 'muizz', 'sudan street', 'giza',
        'downtown', 'وسط البلد', 'المعادي', 'معادي', 'مدينة نصر', 'مصر الجديدة', 'الزمالك', 'الدقي',
        'المهندسين', 'فيصل', 'faisal', 'haram', 'هرم', 'manial', 'شبرا', 'shubra', 'abdin', 'عابدين',
        'mokattam', 'مقطم'
    ]):
        return 'Cairo Central & Urban'

    # 4. Alexandria & Coastal
    if any(k in nl for k in [
        'smouha', 'alexandria', 'miami', 'montaza', 'gleem', 'الإسكندرية', 'الاسكندرية', 'سموحة',
        'agami', 'العجمي'
    ]):
        return 'Alexandria & Coastal'

    # 5. Red Sea
    if any(k in nl for k in [
        'hurghada', 'mubarak 6', 'scandic', 'gouna', 'red sea', 'الغردقة', 'الجونة', 'البحر الاحمر',
        'البحر الأحمر', 'soma bay', 'makadi'
    ]):
        return 'Red Sea'

    # 6. North Coast
    if any(k in nl for k in [
        'alamein', 'north coast', 'marina', 'sidi abdel rahman', 'الساحل الشمالي', 'العلمين',
        'ras el hekma', 'راس الحكمة'
    ]):
        return 'North Coast'

    return 'Greater Cairo Metropolitan'


# -----------------------------------------------------------------------------
# GEOGRAPHIC COORDINATES RESOLUTION
# -----------------------------------------------------------------------------
KNOWN_GEO_CENTERS: List[Tuple[List[str], float, float]] = [
    (['madinaty', 'مدينتي'], 30.108, 31.625),
    (['al rehab', 'rehab', 'الرحاب'], 30.060, 31.493),
    (['beit el watan', 'tagamoa', 'new cairo', 'التجمع', 'القاهرة الجديدة', 'بيت الوطن'], 30.015, 31.485),
    (['taj city', 'mostakbal', 'تاج سيتي', 'المستقبل'], 30.082, 31.415),
    (['new capital', 'administrative capital', 'العاصمة الإدارية', 'العاصمة الادارية'], 30.016, 31.738),
    (['village west', 'green revolution', 'zayed', 'زايد', 'الشيخ زايد', 'الثورة الخضراء', 'فيليدج ويست'], 30.048, 30.985),
    (['new october', 'أكتوبر الجديدة', 'اكتوبر الجديدة'], 29.890, 30.850),
    (['october', 'address home', 'al ahyaa', 'muruj', 'badya', 'zewail', 'اكتوبر', 'أكتوبر', 'الأحياء', 'الاحياء', 'بادية', 'مروج', 'ادريس هوم', 'زويل'], 29.975, 30.935),
    (['maadi', 'معادي', 'المعادي'], 29.960, 31.265),
    (['nasr city', 'مدينة نصر'], 30.055, 31.345),
    (['heliopolis', 'مصر الجديدة'], 30.090, 31.330),
    (['zamalek', 'زمالك', 'الزمالك'], 30.061, 31.220),
    (['sudan street', 'dokki', 'giza', 'شارع السودان', 'الدقي', 'الجيزة', 'فيصل', 'faisal', 'هرم', 'haram', 'مهندسين', 'mohandessin'], 30.040, 31.210),
    (['muizz', 'unesco', 'المعز', 'وسط البلد', 'downtown', 'abdin', 'عابدين'], 30.045, 31.240),
    (['shorouk', 'شروق', 'الشروق'], 30.138, 31.615),
    (['badr', 'بدر', 'مدينة بدر'], 30.138, 31.720),
    (['smouha', 'alexandria', 'سموحة', 'الإسكندرية', 'الاسكندرية', 'ميامي', 'miami', 'منتزه', 'montaza', 'gleem', 'جليم'], 31.215, 29.955),
    (['mubarak 6', 'scandic', 'hurghada', 'الغردقة', 'مبارك 6', 'سكانديك', 'gouna', 'الجونة'], 27.257, 33.812),
    (['alamein', 'north coast', 'marina', 'sidi abdel rahman', 'الساحل الشمالي', 'العلمين', 'مارينا'], 30.835, 28.956)
]


def get_location_coords(norm_loc: str, geo_stat: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns approximate latitude and longitude for verified geographical entities.
    Returns (None, None) for non-geographic marketing listings.
    """
    if geo_stat != "VALID_GEO":
        return None, None

    nl = norm_loc.lower()
    for kw_list, lat, lon in KNOWN_GEO_CENTERS:
        if any(kw in nl for kw in kw_list):
            h = int(hashlib.md5(norm_loc.encode("utf-8")).hexdigest()[:6], 16)
            jitter_lat = ((h % 100) - 50) * 0.00015
            jitter_lon = (((h // 100) % 100) - 50) * 0.00015
            return round(lat + jitter_lat, 5), round(lon + jitter_lon, 5)

    # Valid Greater Cairo metropolitan baseline anchor with location-deterministic spread
    h = int(hashlib.md5(norm_loc.encode("utf-8")).hexdigest()[:6], 16)
    d_lat = 30.025 + ((h % 160) - 80) * 0.0007
    d_lon = 31.350 + (((h // 160) % 160) - 80) * 0.0007
    return round(d_lat, 5), round(d_lon, 5)


# -----------------------------------------------------------------------------
# QUALITY OF LIFE (QOL) ENRICHMENT ENGINE
# -----------------------------------------------------------------------------
MACRO_REGION_QOL_DEFAULTS: Dict[str, Tuple[float, float]] = {
    'Greater Cairo - East': (80.0, 3.5),
    'Greater Cairo - West': (79.5, 3.2),
    'Cairo Central & Urban': (79.5, 1.8),
    'Greater Cairo Metropolitan': (79.8, 2.7),
    'Alexandria & Coastal': (68.5, 2.5),
    'Red Sea': (52.0, 1.5),
    'North Coast': (55.0, 3.0),
    'Unresolved / Non-Geographic': (79.8, 2.7),
}

DEFAULT_AQI = 79.8
DEFAULT_GREEN_PCT = 2.7

HUB_QOL_MAP: Dict[str, Tuple[float, float]] = {
    'madinaty': (79.36, 15.1),
    'al rehab': (80.20, 5.25),
    'beit el watan': (80.20, 4.5),
    'taj city': (79.36, 6.0),
    'village west compound': (79.80, 5.0),
    'new october': (80.20, 2.5),
    'badya compound': (80.00, 4.2),
    'address home compound': (79.80, 3.5),
    'green revolution': (79.80, 8.5),
    'al ahyaa': (80.00, 3.0),
    'muruj': (79.80, 4.0),
    'mubarak 6': (52.0, 1.5),
    'scandic resort': (52.0, 2.0),
    'sudan street': (79.50, 1.2),
    'ahmed zewail road': (80.00, 3.0),
    'zamalek': (79.36, 7.5),
    'maadi': (79.36, 4.8),
    'nasr city': (79.36, 4.78),
    'heliopolis': (79.36, 1.52),
    'smouha (alexandria)': (68.5, 2.5),
    'shorouk city': (79.36, 4.2),
    'badr city': (80.20, 2.0),
    "unesco's al muizz (fatimid cairo)": (79.36, 1.0)
}


def _clean_str(s: Any) -> str:
    """Utility to normalize text for string matching."""
    if not isinstance(s, str):
        return ""
    cleaned = s.strip().lower()
    cleaned = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', cleaned)
    return " ".join(cleaned.split())


def _build_qol_lookup(qol_path: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
    """Builds a fast name -> (us_aqi, green_coverage_pct) dictionary from QOL CSV."""
    lookup: Dict[str, Tuple[float, float]] = {}
    try:
        resolved_path = _resolve_file("cairo_QOL_combined.csv", qol_path)
        qol_df = pd.read_csv(resolved_path)

        for _, row in qol_df.iterrows():
            en = _clean_str(row.get("name_en_x", ""))
            ar = _clean_str(row.get("name", ""))
            try:
                aqi_val = float(row["us_aqi"])
                green_val = float(row["green_coverage_pct"])
            except (ValueError, TypeError):
                continue

            if en and en not in lookup:
                lookup[en] = (aqi_val, green_val)
            if ar and ar not in lookup:
                lookup[ar] = (aqi_val, green_val)

    except Exception as exc:
        print(f"[WARN] Failed to fully parse QOL dataset: {exc}. Using robust fallbacks.")

    return lookup


# -----------------------------------------------------------------------------
# CORE DATA LOADING & PREPARATION FUNCTION
# -----------------------------------------------------------------------------
@cache_data
def load_clean_data(
    csv_path: Optional[str] = None,
    qol_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Loads, cleans, engineers features, geocodes, and enriches real estate listings.
    Cached via Streamlit @st.cache_data (or singleton decorator).

    Returns:
        pd.DataFrame: Master dataset with 10,000 canonical rows and enriched attributes.
    """
    data_file = _resolve_file("Apartments_Prices_Final_Version.csv", csv_path)
    df = pd.read_csv(data_file)

    # 1. Parse Bathrooms to numeric 'Bathrooms_num' (fill nulls with 2.0 mode)
    df['Bathrooms_num'] = (
        df['Bathrooms']
        .astype(str)
        .str.extract(r'(\d+)', expand=False)
        .astype(float)
        .fillna(2.0)
    )

    # Clean other numeric columns
    df['Floor Level'] = df['Floor Level'].fillna(2.0)
    df['Year Built'] = df['Year Built'].fillna(2025.0)
    if 'price after 10 years' in df.columns:
        df['price after 10 years'] = df['price after 10 years'].fillna(0.0)

    # 2. Derive Living Area = Price_Numeric / Calculated_Price_Per_Meter (Area Identity)
    df['Area'] = np.where(
        df['Calculated_Price_Per_Meter'] > 0,
        np.round(df['Price_Numeric'] / df['Calculated_Price_Per_Meter'], 1),
        120.0
    )

    # 3. Location classification & regex-based normalization
    df['geo_status'] = df['Location'].apply(classify_location)
    df['normalized_location'] = df['Location'].apply(normalize_location)

    # 4. Assign Macro Region
    df['macro_region'] = [
        assign_macro_region(nl, gs)
        for nl, gs in zip(df['normalized_location'], df['geo_status'])
    ]

    # 5. Geocoding Coordinates
    coords = [
        get_location_coords(nl, gs)
        for nl, gs in zip(df['normalized_location'], df['geo_status'])
    ]
    df['latitude'] = [c[0] for c in coords]
    df['longitude'] = [c[1] for c in coords]
    # Standard short aliases for geospatial libraries (e.g. st.map, pydeck, plotly)
    df['lat'] = df['latitude']
    df['lon'] = df['longitude']

    # 6. Quality of Life (Air Quality & Green Coverage) Enrichment
    qol_lookup = _build_qol_lookup(qol_path)

    aqi_values: List[float] = []
    green_values: List[float] = []

    for raw_loc, norm_loc, macro in zip(
        df['Location'], df['normalized_location'], df['macro_region']
    ):
        c_raw = _clean_str(raw_loc)
        c_norm = _clean_str(norm_loc)
        nl_lower = norm_loc.lower().strip()

        # Step 1: Direct exact match in QOL
        if c_raw in qol_lookup:
            aqi, grn = qol_lookup[c_raw]
        elif c_norm in qol_lookup:
            aqi, grn = qol_lookup[c_norm]
        # Step 2: Major hub calibrated lookup
        elif nl_lower in HUB_QOL_MAP:
            aqi, grn = HUB_QOL_MAP[nl_lower]
        # Step 3: Macro-region average fallback (~79.8 AQI, ~2.7% Green)
        else:
            aqi, grn = MACRO_REGION_QOL_DEFAULTS.get(macro, (DEFAULT_AQI, DEFAULT_GREEN_PCT))

        aqi_values.append(round(aqi, 2))
        green_values.append(round(grn, 2))

    df['us_aqi'] = aqi_values
    df['green_coverage_pct'] = green_values
    # Aliases
    df['aqi'] = df['us_aqi']
    df['green_pct'] = df['green_coverage_pct']

    # 7. Presentation & Friendly categorical labels
    df['amenities_presentation'] = df['Amenities'].apply(
        lambda x: 'Gated Compound' if str(x).strip() == 'B(Compound)' else 'Standalone (Non-Compound)'
    )

    return df


# -----------------------------------------------------------------------------
# HELPER EXPORTS
# -----------------------------------------------------------------------------
def get_location_coordinates(
    df: Optional[pd.DataFrame] = None,
    min_count: int = 1
) -> pd.DataFrame:
    """
    Aggregates verified geographical locations into a summary DataFrame
    with coordinates, sample counts, median prices, and environmental metrics.

    Args:
        df: Optional pre-loaded DataFrame. If None, load_clean_data() is used.
        min_count: Minimum listing volume threshold (default: 1).

    Returns:
        pd.DataFrame with columns:
        ['location', 'lat', 'lon', 'count', 'median_price', 'median_ppm', 'aqi', 'green_pct', 'macro_region']
    """
    if df is None:
        df = load_clean_data()

    # Filter to verified geographical listings with resolved coordinates
    valid_df = df[
        (df['geo_status'] == 'VALID_GEO') &
        (df['lat'].notnull()) &
        (df['lon'].notnull())
    ]

    grouped = valid_df.groupby('normalized_location').agg(
        lat=('lat', 'first'),
        lon=('lon', 'first'),
        count=('Price_Numeric', 'count'),
        median_price=('Price_Numeric', 'median'),
        median_ppm=('Calculated_Price_Per_Meter', 'median'),
        aqi=('us_aqi', 'median'),
        green_pct=('green_coverage_pct', 'median'),
        macro_region=('macro_region', lambda s: s.mode().iloc[0] if not s.empty else 'Greater Cairo Metropolitan')
    ).reset_index()

    grouped.rename(columns={'normalized_location': 'location'}, inplace=True)

    # Filter by minimum listing count threshold
    if min_count > 1:
        grouped = grouped[grouped['count'] >= min_count]

    grouped.sort_values(by='count', ascending=False, inplace=True)
    grouped.reset_index(drop=True, inplace=True)

    # Attach duplicate coordinate aliases for full library compatibility
    grouped['latitude'] = grouped['lat']
    grouped['longitude'] = grouped['lon']

    return grouped


def get_filter_options(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Extracts sorted unique categories and numeric bounding ranges for UI sidebar filters.

    Args:
        df: Optional pre-loaded DataFrame. If None, load_clean_data() is used.

    Returns:
        dict with unique values for regions, locations, amenities, finishing, rooms, etc.
    """
    if df is None:
        df = load_clean_data()

    rooms_list = sorted([
        int(r) if float(r).is_integer() else float(r)
        for r in df['Rooms'].dropna().unique()
    ])
    baths_list = sorted([
        int(b) if float(b).is_integer() else float(b)
        for b in df['Bathrooms_num'].dropna().unique()
    ])

    return {
        'regions': sorted(df['macro_region'].dropna().unique().tolist()),
        'macro_regions': sorted(df['macro_region'].dropna().unique().tolist()),
        'locations': sorted(df['normalized_location'].dropna().unique().tolist()),
        'amenities': sorted(df['amenities_presentation'].dropna().unique().tolist()),
        'amenities_raw': sorted(df['Amenities'].dropna().unique().tolist()),
        'finishing': sorted(df['finishing_type'].dropna().unique().tolist()),
        'finishing_types': sorted(df['finishing_type'].dropna().unique().tolist()),
        'rooms': rooms_list,
        'bathrooms': baths_list,
        'views': sorted(df['View'].dropna().unique().tolist()),
        'payment_methods': sorted(df['Payment Method'].dropna().unique().tolist()),
        'types': sorted(df['Type'].dropna().unique().tolist()),
        'geo_statuses': sorted(df['geo_status'].dropna().unique().tolist()),
        'min_price': float(df['Price_Numeric'].min()),
        'max_price': float(df['Price_Numeric'].max()),
        'min_ppm': float(df['Calculated_Price_Per_Meter'].min()),
        'max_ppm': float(df['Calculated_Price_Per_Meter'].max()),
        'min_area': float(df['Area'].min()),
        'max_area': float(df['Area'].max()),
        'min_year': int(df['Year Built'].min()),
        'max_year': int(df['Year Built'].max()),
    }


def clear_cache() -> None:
    """Clears dataset cache."""
    if hasattr(load_clean_data, "clear"):
        load_clean_data.clear()
    if HAS_STREAMLIT and hasattr(st, "cache_data") and hasattr(st.cache_data, "clear"):
        st.cache_data.clear()


# -----------------------------------------------------------------------------
# QUICK STANDALONE EXECUTION TEST
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Testing 'data_loader.py' module...")
    print("=" * 70)

    data = load_clean_data()
    print(f"[OK] Master data loaded: {len(data):,} rows, {len(data.columns)} columns.")

    loc_coords = get_location_coordinates(data, min_count=1)
    print(f"[OK] Location coordinates resolved: {len(loc_coords):,} unique hubs.")

    filters = get_filter_options(data)
    print(f"[OK] Filter options generated:")
    print(f"     - Regions ({len(filters['regions'])}): {filters['regions'][:4]}...")
    print(f"     - Rooms: {filters['rooms']}")
    print(f"     - Bathrooms: {filters['bathrooms']}")
    print(f"     - Price Range: {filters['min_price']:,.0f} -> {filters['max_price']:,.0f} EGP")
    print(f"     - Area Range: {filters['min_area']:.1f} -> {filters['max_area']:.1f} m²")
    print(f"     - AQI Range: {data['us_aqi'].min():.1f} -> {data['us_aqi'].max():.1f} (Mean: {data['us_aqi'].mean():.2f})")
    print(f"     - Green % Range: {data['green_pct'].min():.1f}% -> {data['green_pct'].max():.1f}% (Mean: {data['green_pct'].mean():.2f}%)")
    print("=" * 70)
    print("All tests completed successfully!")

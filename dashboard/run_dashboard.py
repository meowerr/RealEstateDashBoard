# -*- coding: utf-8 -*-
"""
Egypt Real Estate Intelligence Platform (v1)
Interactive Geospatial & Analytics Dashboard Launcher

Portably resolves the dashboard path without hardcoded directories.
Works directly across any drive or machine.
"""

import sys
import webbrowser
from pathlib import Path

# Ensure UTF-8 output across Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    base_dir = Path(__file__).resolve().parent

    # Check for index.html in dashboard subdirectory or current directory
    candidates = [
        base_dir / "dashboard" / "index.html",
        base_dir / "index.html"
    ]

    target = None
    for cand in candidates:
        if cand.exists():
            target = cand
            break

    if not target:
        print("[ERROR] Dashboard file 'index.html' not found.")
        print(f"Searched locations:")
        for cand in candidates:
            print(f" - {cand}")
        sys.exit(1)

    print("=" * 68)
    print("  🚀 EGYPT REAL ESTATE INTELLIGENCE PLATFORM (v1)")
    print("  Interactive Geospatial Heatmap & 10-Year Valuation Engine")
    print("=" * 68)
    print(f"  Target File : {target.name}")
    print(f"  Directory   : {target.parent}")
    print(f"  URI         : {target.as_uri()}")
    print("-" * 68)
    print("  Opening dashboard in your default web browser...")

    webbrowser.open(target.as_uri())
    print("  Dashboard loaded successfully!")
    print("=" * 68)

if __name__ == "__main__":
    main()

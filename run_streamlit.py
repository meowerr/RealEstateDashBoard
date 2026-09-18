# -*- coding: utf-8 -*-
"""
Egypt Real Estate Intelligence Platform (v1)
Streamlit Application Launcher
File: run_streamlit.py

Runs the Streamlit dashboard app.py with optimal executive settings:
- Zero-scroll viewports
- Left sidebar filters & financial controls
- Egyptian Executive Palette:
  * Deep Navy (#1B2A4A)
  * Nile Teal (#2A9D8F)
  * Sandstone Gold (#E9B44C)
  * Coral Red (#E85D4C)
  * Soft Sage (#8FB996)
  * Off-white (#F7F5F2)
"""

import sys
import subprocess
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent
    app_path = base_dir / "app.py"
    
    if not app_path.exists():
        print(f"[ERROR] Application file '{app_path}' not found.")
        sys.exit(1)
        
    print("=" * 68)
    print("  🚀 EGYPT REAL ESTATE INTELLIGENCE PLATFORM (v1)")
    print("  Launching Streamlit Interactive Dashboard...")
    print("  Mode: Zero-Page-Scroll Responsive Bento Cockpit")
    print("  Palette: Deep Navy | Nile Teal | Sandstone Gold | Soft Sage | Coral")
    print("=" * 68)
    print(f"  App Path : {app_path}")
    print("-" * 68)
    
    # Launch Streamlit with executive palette tokens
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",
        "--theme.primaryColor=#2A9D8F",
        "--theme.backgroundColor=#F7F5F2",
        "--theme.secondaryBackgroundColor=#1B2A4A",
        "--theme.textColor=#333333"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard closed successfully.")

if __name__ == "__main__":
    main()

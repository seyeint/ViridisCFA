# Backward-compatible wrapper for ViridisCFA
import sys
import os

# Ensure the package is importable if executed directly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from viridis_cfa.pipeline import (
    run_analysis,
    get_multi_year_trends,
    process_filing,
    process_transcript,
    create_final_analysis,
    analyze_ticker
)
from viridis_cfa.insider import get_insider_activity
from viridis_cfa.cli import main

if __name__ == "__main__":
    main()
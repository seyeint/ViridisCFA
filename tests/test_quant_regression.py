"""
Quant Engine Regression Tests
=============================
Compares computed financial scores against known-good snapshots to catch
coefficient, threshold, or parsing regressions.

Run: python -m pytest tests/test_quant_regression.py -v

These fixtures are pinned to FY 2025/2026 annual XBRL data from SEC EDGAR.
If a company files a restatement or SEC re-tags their XBRL, values may shift —
update the snapshots after manual verification.
"""

import pytest
import os
import sys

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from viridis_cfa.config import *  # noqa: F401,F403 — sets EDGAR identity
from edgar import Company
from quant_engine import (
    apply_balance_sheet_fallbacks,
    calculate_beneish_m,
    generate_quant_scorecard,
    QUANT_ENGINE_VERSION,
)


# ─── Known-Good Snapshots (quant_engine v1.2, captured 2026-05-26) ───

SNAPSHOTS = {
    "TSLA": {
        "altman_z": 1.8447,
        "piotroski_f": 5,
        "beneish_m": -1.955,
        "current_ratio": 2.1644,
        "debt_to_equity": 0.6689,
        "latest_year": "FY 2025",
        "beneish_imputed": ["DEPI (Depreciation)"],
    },
    "META": {
        "altman_z": 2.2796,
        "piotroski_f": 5,
        "beneish_m": -2.5203,
        "current_ratio": 2.5988,
        "debt_to_equity": 0.6848,
        "latest_year": "FY 2025",
        "beneish_imputed": ["DEPI (Depreciation)"],
    },
    "GOOGL": {
        "altman_z": 2.9032,
        "piotroski_f": 6,
        "beneish_m": -1.5568,
        "current_ratio": 2.0053,
        "debt_to_equity": 0.4335,
        "latest_year": "FY 2025",
        "beneish_imputed": ["DEPI (Depreciation)"],
    },
    "KSCP": {
        "altman_z": -5.7305,
        "piotroski_f": 3,
        "beneish_m": -2.3173,
        "current_ratio": 3.992,
        "debt_to_equity": 0.4859,
        "latest_year": "FY 2025",
        "beneish_imputed": [],
    },
    "AMBA": {
        "altman_z": 1.2537,
        "piotroski_f": 5,
        "beneish_m": -2.5446,
        "current_ratio": 2.3056,
        "debt_to_equity": 0.3427,
        "latest_year": "FY 2026",
        "beneish_imputed": [],
    },
}

# Tolerance: XBRL values are exact, but rounding at 4 decimal places
# allows for minor float arithmetic differences across platforms.
TOLERANCE = 0.005


@pytest.fixture(scope="module")
def computed_scores():
    """Compute all scorecard metadata once per test session."""
    results = {}
    for ticker in SNAPSHOTS:
        company = Company(ticker)
        _, metadata = generate_quant_scorecard(company)
        results[ticker] = metadata
    return results


class TestQuantEngineVersion:
    """Ensure we're testing against the expected engine version."""

    def test_version_matches(self):
        assert QUANT_ENGINE_VERSION == "1.2", (
            f"Snapshot fixtures are pinned to v1.2, but engine is v{QUANT_ENGINE_VERSION}. "
            f"Re-capture snapshots if version was intentionally bumped."
        )


class TestDeterministicFallbacks:
    """Pure unit tests for balance-sheet fallback contracts."""

    def test_derives_missing_total_liabilities(self):
        metrics = {
            "total_assets": {"FY 2025": 1000.0},
            "total_liabilities": {},
            "equity": {"FY 2025": 350.0},
            "redeemable_noncontrolling_interest": {"FY 2025": 25.0},
        }

        derived = apply_balance_sheet_fallbacks(metrics, ["FY 2025"])

        assert metrics["total_liabilities"]["FY 2025"] == 625.0
        assert derived == [{
            "metric": "total_liabilities",
            "year": "FY 2025",
            "formula": "Total Assets - Equity - Redeemable Noncontrolling Interest",
        }]

    def test_derives_missing_equity(self):
        metrics = {
            "total_assets": {"FY 2025": 1000.0},
            "total_liabilities": {"FY 2025": 625.0},
            "equity": {},
            "redeemable_noncontrolling_interest": {"FY 2025": 25.0},
        }

        derived = apply_balance_sheet_fallbacks(metrics, ["FY 2025"])

        assert metrics["equity"]["FY 2025"] == 350.0
        assert derived == [{
            "metric": "equity",
            "year": "FY 2025",
            "formula": "Total Assets - Total Liabilities - Redeemable Noncontrolling Interest",
        }]

    def test_beneish_invalid_year_shape(self):
        result = calculate_beneish_m({}, "FY")
        assert len(result) == 4


class TestAltmanZPrime:
    """Altman Z'-Score regression tests (book-value variant)."""

    @pytest.mark.parametrize("ticker", SNAPSHOTS.keys())
    def test_altman_z(self, ticker, computed_scores):
        expected = SNAPSHOTS[ticker]["altman_z"]
        actual = computed_scores[ticker].get("altman_z")
        if expected is None:
            assert actual is None, f"{ticker}: expected None, got {actual}"
        else:
            assert actual is not None, f"{ticker}: expected {expected}, got None"
            assert abs(actual - expected) < TOLERANCE, (
                f"{ticker}: Z'-Score drifted. Expected {expected}, got {round(actual, 4)}. "
                f"If intentional, update snapshot."
            )


class TestPiotroskiFScore:
    """Piotroski F-Score regression tests (integer 0-9)."""

    @pytest.mark.parametrize("ticker", SNAPSHOTS.keys())
    def test_piotroski_f(self, ticker, computed_scores):
        expected = SNAPSHOTS[ticker]["piotroski_f"]
        actual = computed_scores[ticker].get("piotroski_f")
        assert actual == expected, (
            f"{ticker}: F-Score changed. Expected {expected}, got {actual}."
        )


class TestBeneishMScore:
    """Beneish M-Score regression tests (8-variable model)."""

    @pytest.mark.parametrize("ticker", SNAPSHOTS.keys())
    def test_beneish_m(self, ticker, computed_scores):
        expected = SNAPSHOTS[ticker]["beneish_m"]
        actual = computed_scores[ticker].get("beneish_m")
        if expected is None:
            assert actual is None, f"{ticker}: expected None, got {actual}"
        else:
            assert actual is not None, f"{ticker}: expected {expected}, got None"
            assert abs(actual - expected) < TOLERANCE, (
                f"{ticker}: M-Score drifted. Expected {expected}, got {round(actual, 4)}."
            )

    @pytest.mark.parametrize("ticker", SNAPSHOTS.keys())
    def test_beneish_imputed(self, ticker, computed_scores):
        """Verify which Beneish components were imputed vs. computed from actual data."""
        expected = SNAPSHOTS[ticker]["beneish_imputed"]
        actual = computed_scores[ticker].get("beneish_imputed_components", [])
        assert actual == expected, (
            f"{ticker}: Imputed components changed. "
            f"Expected {expected}, got {actual}."
        )


class TestRatios:
    """Financial ratio regression tests."""

    @pytest.mark.parametrize("ticker", SNAPSHOTS.keys())
    def test_current_ratio(self, ticker, computed_scores):
        expected = SNAPSHOTS[ticker]["current_ratio"]
        actual = computed_scores[ticker].get("current_ratio")
        if expected is None:
            assert actual is None
        else:
            assert actual is not None and abs(actual - expected) < TOLERANCE, (
                f"{ticker}: Current Ratio drifted. Expected {expected}, got {round(actual, 4) if actual else None}."
            )

    @pytest.mark.parametrize("ticker", SNAPSHOTS.keys())
    def test_debt_to_equity(self, ticker, computed_scores):
        expected = SNAPSHOTS[ticker]["debt_to_equity"]
        actual = computed_scores[ticker].get("debt_to_equity")
        if expected is None:
            assert actual is None
        else:
            assert actual is not None and abs(actual - expected) < TOLERANCE, (
                f"{ticker}: Debt-to-Equity drifted. Expected {expected}, got {round(actual, 4) if actual else None}."
            )


class TestScorecard:
    """Scorecard format and metadata tests."""

    @pytest.mark.parametrize("ticker", SNAPSHOTS.keys())
    def test_latest_year(self, ticker, computed_scores):
        expected = SNAPSHOTS[ticker]["latest_year"]
        actual = computed_scores[ticker].get("latest_year")
        assert actual == expected, (
            f"{ticker}: Latest year changed from {expected} to {actual}. "
            f"Company may have filed a new annual report — re-capture snapshots."
        )

    @pytest.mark.parametrize("ticker", SNAPSHOTS.keys())
    def test_metadata_has_version(self, ticker, computed_scores):
        version = computed_scores[ticker].get("quant_engine_version")
        assert version == QUANT_ENGINE_VERSION, (
            f"{ticker}: Metadata version mismatch. Expected {QUANT_ENGINE_VERSION}, got {version}."
        )

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
    beneish_out_of_meaningful_range,
    calculate_beneish_m,
    cash_runway_years,
    classify_business_model,
    generate_quant_scorecard,
    QUANT_ENGINE_VERSION,
)


# ─── Known-Good Snapshots (quant_engine v1.4; Beneish re-captured at v1.3) ───
# Beneish values were re-captured after the coefficient fix (TATA 4.679, LVGI -0.327).
# v1.4 adds the Beneish out-of-range guard + cash-runway metric, both additive — these
# snapshot values are unchanged. The canonical Beneish formula is pinned independently
# in TestBeneishFormulaPinned, so these live snapshots guard data/extraction drift.

SNAPSHOTS = {
    "TSLA": {
        "altman_z": 1.8447,
        "piotroski_f": 5,
        "beneish_m": -2.3678,
        "current_ratio": 2.1644,
        "debt_to_equity": 0.6689,
        "latest_year": "FY 2025",
        "beneish_imputed": ["DEPI (Depreciation)"],
    },
    "META": {
        "altman_z": 2.2796,
        "piotroski_f": 5,
        "beneish_m": -3.0494,
        "current_ratio": 2.5988,
        "debt_to_equity": 0.6848,
        "latest_year": "FY 2025",
        "beneish_imputed": ["DEPI (Depreciation)"],
    },
    "GOOGL": {
        "altman_z": 2.9032,
        "piotroski_f": 6,
        "beneish_m": -1.9831,
        "current_ratio": 2.0053,
        "debt_to_equity": 0.4335,
        "latest_year": "FY 2025",
        "beneish_imputed": ["DEPI (Depreciation)"],
    },
    "KSCP": {
        "altman_z": -5.7305,
        "piotroski_f": 3,
        "beneish_m": -2.6386,
        "current_ratio": 3.992,
        "debt_to_equity": 0.4859,
        "latest_year": "FY 2025",
        "beneish_imputed": [],
    },
    "AMBA": {
        "altman_z": 1.2537,
        "piotroski_f": 5,
        "beneish_m": -3.1605,
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
        assert QUANT_ENGINE_VERSION == "1.4", (
            f"Snapshot fixtures are pinned to v1.4, but engine is v{QUANT_ENGINE_VERSION}. "
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


class TestBeneishFormulaPinned:
    """Pin the Beneish coefficients to canonical Beneish (1999) with a hand-computed
    reference, independent of any live ticker. This is the guard the live snapshots
    cannot provide: if a coefficient is mistyped again (as TATA=4.037 / LVGI=+0.0327
    once was), this fails against a fixed mathematical truth, not a self-referential
    captured value."""

    def test_beneish_canonical_baseline_is_minus_2_48(self):
        # Construct metrics so all eight indices == 1.0 and TATA == 0: every field is
        # equal across the year and the prior year, and Net Income == CFO.
        equal = {
            "revenue": 800, "receivables": 100, "gross_profit": 200,
            "total_assets": 1000, "current_assets": 400, "ppe_net": 300,
            "total_liabilities": 500, "operating_income": 70,
            "depreciation": 50, "sga": 80,
        }
        metrics = {k: {"FY 2025": v, "FY 2024": v} for k, v in equal.items()}
        metrics["net_income"] = {"FY 2025": 60}            # NI == CFO -> TATA = 0
        metrics["operating_cash_flow"] = {"FY 2025": 60}

        m_score, _status, _missing, imputed = calculate_beneish_m(metrics, "FY 2025")

        # -4.84 + 0.920 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 + 4.679*0 - 0.327 = -2.48
        assert imputed == [], f"unexpected imputation: {imputed}"
        assert m_score == pytest.approx(-2.48, abs=1e-9), (
            f"Beneish coefficients drifted from canonical Beneish (1999). Expected -2.48 at "
            f"all-indices=1.0 / TATA=0, got {m_score}. Check TATA (+4.679) and LVGI (-0.327) "
            f"in quant_engine.py."
        )

    def test_out_of_meaningful_range_flags_artifacts(self):
        # AISP-style micro-cap artifact and an implausibly-clean score are flagged;
        # normal values (incl. the high-risk cutoff and the canonical baseline) are not.
        assert beneish_out_of_meaningful_range(12.51) is True
        assert beneish_out_of_meaningful_range(-6.0) is True
        assert beneish_out_of_meaningful_range(-1.78) is False
        assert beneish_out_of_meaningful_range(-2.48) is False
        assert beneish_out_of_meaningful_range(1.0) is False   # inclusive bound
        assert beneish_out_of_meaningful_range(None) is False


class TestCashRunway:
    """Cash runway is only meaningful for cash-consuming issuers."""

    def test_runway_only_when_burning(self):
        assert cash_runway_years(1000.0, -250.0) == pytest.approx(4.0)   # 1000 / 250
        assert cash_runway_years(1000.0, 500.0) is None                  # cash-flow positive
        assert cash_runway_years(1000.0, 0.0) is None                    # not burning
        assert cash_runway_years(None, -100.0) is None                   # missing cash
        assert cash_runway_years(1000.0, None) is None                   # missing CFO


class _FakeCompany:
    def __init__(self, sic):
        self.sic = sic


class TestBusinessModelClassification:
    """Offline unit tests for the business-model applicability gate."""

    def test_financial_issuer_suppresses_altman_and_beneish(self):
        metrics = {"revenue": {"FY 2025": 100, "FY 2024": 95}, "total_assets": {"FY 2025": 5000}}
        bm = classify_business_model(_FakeCompany("6021"), metrics, "FY 2025")
        assert bm["archetype"] == "financial"
        assert bm["altman"] == "not_applicable"
        assert bm["beneish"] == "not_applicable"

    def test_pre_revenue_suppresses_beneish_and_softens_altman(self):
        metrics = {"revenue": {"FY 2025": 1, "FY 2024": 1}, "total_assets": {"FY 2025": 1000}}
        bm = classify_business_model(_FakeCompany("2834"), metrics, "FY 2025")
        assert bm["archetype"] == "pre-revenue"
        assert bm["beneish"] == "not_applicable"
        assert bm["altman"] == "advisory"

    def test_hypergrowth_flags_beneish_advisory(self):
        metrics = {"revenue": {"FY 2025": 200, "FY 2024": 100}, "total_assets": {"FY 2025": 400}}
        bm = classify_business_model(_FakeCompany("7372"), metrics, "FY 2025")
        assert bm["archetype"] == "hypergrowth"
        assert bm["beneish"] == "advisory"
        assert bm["altman"] == "applicable"

    def test_standard_issuer_all_applicable(self):
        metrics = {"revenue": {"FY 2025": 110, "FY 2024": 100}, "total_assets": {"FY 2025": 200}}
        bm = classify_business_model(_FakeCompany("3711"), metrics, "FY 2025")
        assert bm["archetype"] == "standard"
        assert bm["altman"] == "applicable"
        assert bm["beneish"] == "applicable"

    def test_missing_or_nonnumeric_sic_does_not_crash(self):
        metrics = {"revenue": {"FY 2025": 110, "FY 2024": 100}, "total_assets": {"FY 2025": 200}}
        for bad_sic in (None, "", "N/A"):
            bm = classify_business_model(_FakeCompany(bad_sic), metrics, "FY 2025")
            assert bm["sic"] is None
            assert bm["archetype"] == "standard"


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

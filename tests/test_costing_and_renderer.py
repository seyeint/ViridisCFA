from pathlib import Path

import pytest

from utils import calculate_actual_cost, estimate_cost
from viridis_cfa.report_renderer import render_report_html


def test_gpt54_flex_cost_is_half_standard_for_uncached_short_context():
    standard = calculate_actual_cost(
        prompt_tokens=100_000,
        completion_tokens=10_000,
        model="gpt-5.4",
        service_tier="standard",
    )
    flex = calculate_actual_cost(
        prompt_tokens=100_000,
        completion_tokens=10_000,
        model="gpt-5.4",
        service_tier="flex",
    )

    assert standard["total_cost"] == pytest.approx(0.40)
    assert flex["total_cost"] == pytest.approx(0.20)


def test_cached_input_uses_cached_token_rate():
    cost = calculate_actual_cost(
        prompt_tokens=100_000,
        completion_tokens=10_000,
        model="gpt-5.4",
        service_tier="flex",
        cached_input_tokens=40_000,
    )

    assert cost["uncached_input_cost"] == pytest.approx(0.075)
    assert cost["cached_input_cost"] == pytest.approx(0.005)
    assert cost["output_cost"] == pytest.approx(0.075)
    assert cost["total_cost"] == pytest.approx(0.155)


def test_gpt54_long_context_rate_applies_above_threshold():
    cost = calculate_actual_cost(
        prompt_tokens=300_000,
        completion_tokens=10_000,
        model="gpt-5.4",
        service_tier="flex",
    )

    assert cost["context_band"] == "long"
    assert cost["total_cost"] == pytest.approx(0.8625)


def test_estimate_cost_respects_requested_tier():
    standard = estimate_cost(100_000, "gpt-5.4", service_tier="standard")
    flex = estimate_cost(100_000, "gpt-5.4", service_tier="flex")

    assert flex["total_cost"] == pytest.approx(standard["total_cost"] / 2)


def test_report_renderer_includes_scorecard_and_artifact_links(tmp_path):
    scorecard = tmp_path / "scorecard.md"
    scorecard.write_text(
        "\n".join([
            "| **Altman Z'-Score (Z-Prime)** | `2.20` | Grey Zone | `Programmatically Verified` |",
            "| **Quick Ratio** | `UNABLE TO COMPUTE` | Missing: Receivables | `MISSING - Footnotes Search Required` |",
        ]),
        encoding="utf-8",
    )
    md_report = """# Executive Summary

## Stance
**Bullish, Medium conviction.**

---
##### Report Provenance
- **Quant Engine**: v1.2
- **Model**: gpt-5.4 (flex)
- **Filing**: 2026-05-11 | Accession: `abc`
- **Generated**: 2026-05-26 15:00 UTC
"""

    html = render_report_html(
        "TEST",
        md_report,
        artifact_paths={
            "scorecard": str(scorecard),
            "final_report_md": str(Path("data/intermediate/TEST_final_report.md")),
            "final_report_pdf": str(Path("data/TEST_final_report.pdf")),
        },
    )

    assert "ViridisCFA" in html
    assert "Quant Scorecard" in html
    assert "Altman Z&#x27;-Score" in html
    assert "UNABLE TO COMPUTE" in html
    assert "Bullish" in html
    assert "../TEST_final_report.pdf" in html

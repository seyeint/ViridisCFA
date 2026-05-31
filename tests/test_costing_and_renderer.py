from pathlib import Path
import json

import pytest

from utils import calculate_actual_cost, estimate_cost
from viridis_cfa.decision_brief import (
    extract_and_strip_brief_json,
    normalize_decision_brief,
    parse_decision_brief_json,
)
from viridis_cfa.report_renderer import render_markdown_document_html, render_report_html


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
    brief = tmp_path / "decision_brief.json"
    brief.write_text(
        json.dumps({
            "schema_version": "1.0",
            "ticker": "TEST",
            "stance": "Bullish",
            "conviction": "Medium",
            "thesis": "Structured thesis from the decision brief.",
            "upside_drivers": ["One structured upside driver."],
            "risk_drivers": ["One structured risk driver."],
            "key_catalysts": [],
            "data_quality_flags": [],
        }),
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
            "decision_brief": str(brief),
            "final_report_md": str(Path("data/intermediate/TEST_final_report.md")),
            "final_report_pdf": str(Path("data/TEST_final_report.pdf")),
        },
    )

    assert "ViridisCFA" in html
    assert "Quant Snapshot" in html
    assert "Altman Z&#x27;" in html
    assert "UNABLE TO COMPUTE" in html
    assert "Bullish" in html
    assert "../TEST_final_report.pdf" in html


def test_company_report_requires_decision_brief(tmp_path):
    scorecard = tmp_path / "scorecard.md"
    scorecard.write_text(
        "| **Altman Z'-Score (Z-Prime)** | `2.06` | Grey Zone | `Programmatically Verified` |",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing decision brief"):
        render_report_html("TEL", "# Any markdown shape", artifact_paths={"scorecard": str(scorecard)})


def test_report_renderer_prefers_decision_brief_contract(tmp_path):
    brief = tmp_path / "decision_brief.json"
    brief.write_text(
        json.dumps({
            "schema_version": "1.0",
            "ticker": "GEN",
            "stance": "Bearish",
            "conviction": "High",
            "thesis": "Structured thesis wins over whatever markdown headings happen to say.",
            "upside_drivers": ["One explicitly structured upside driver."],
            "risk_drivers": ["One explicitly structured risk driver."],
            "key_catalysts": ["One explicitly structured catalyst."],
            "data_quality_flags": ["One explicitly structured data-quality flag."],
        }),
        encoding="utf-8",
    )
    md_report = """# Sources

- Source text that should not become the thesis.

# Weird Arbitrary Report Shape

This markdown deliberately has no standard bull or bear headings.
"""

    html = render_report_html("GEN", md_report, artifact_paths={"decision_brief": str(brief)})

    assert "Structured thesis wins" in html
    assert "One explicitly structured upside driver." in html
    assert "One explicitly structured risk driver." in html
    assert "One explicitly structured data-quality flag." in html
    assert "Source text that should not become the thesis" not in html.split("class=\"thesis-text\"")[1].split("</p>")[0]


def test_generic_markdown_renderer_does_not_require_decision_brief():
    html = render_markdown_document_html("Batch Comparison", "# Ranking\n\n| Ticker | View |\n| --- | --- |\n| ABC | Bullish |")

    assert "Batch Comparison" in html
    assert "<table>" in html
    assert "ABC" in html


def test_decision_brief_normalization_is_total():
    brief = normalize_decision_brief({"stance": "nonsense", "upside_drivers": ["ok"]}, "ABC")

    assert brief["ticker"] == "ABC"
    assert brief["stance"] == "Neutral"
    assert brief["conviction"] == "N/A"
    assert brief["upside_drivers"] == ["ok"]


def test_decision_brief_parser_accepts_fenced_json():
    brief = parse_decision_brief_json(
        """```json
{"ticker":"ABC","stance":"Mixed","conviction":"Low","thesis":"Watch list name.","upside_drivers":[],"risk_drivers":[],"key_catalysts":[],"data_quality_flags":[]}
```""",
        "ABC",
    )

    assert brief["stance"] == "Mixed"
    assert brief["conviction"] == "Low"


def test_decision_brief_normalizer_splits_packed_json_fragments():
    brief = normalize_decision_brief(
        {
            "ticker": "BAD",
            "upside_drivers": [
                "Bookings increased 37% YoY\",\"AI exposure grew 20%\",\"Full-year guidance was raised\"],",
            ],
            "risk_drivers": [
                "Margins fell\",\"Tariffs rose\"],\"key_catalysts\":[\"Q2 results\",\"2028 settlement\"]} ```Oops, corrected below",
            ],
        },
        "BAD",
    )

    assert brief["upside_drivers"] == [
        "Bookings increased 37% YoY",
        "AI exposure grew 20%",
        "Full-year guidance was raised",
    ]
    assert brief["risk_drivers"] == ["Margins fell", "Tariffs rose"]


def test_extract_inline_brief_json_from_memo():
    memo = """## Investment Conclusion

**Stance:** Bearish, Medium conviction.

```json
{"stance":"Bearish","conviction":"Medium","thesis":"Weak setup.","upside_drivers":["x"],"risk_drivers":["y"],"key_catalysts":[],"data_quality_flags":[]}
```"""
    stripped, js = extract_and_strip_brief_json(memo)

    assert js is not None
    assert "```json" not in stripped
    assert stripped.endswith("Bearish, Medium conviction.")  # memo body preserved, block removed
    brief = parse_decision_brief_json(js, "ZZZ")
    assert brief["stance"] == "Bearish"
    assert brief["conviction"] == "Medium"
    assert brief["ticker"] == "ZZZ"


def test_extract_inline_brief_json_returns_none_when_absent():
    memo = "# Memo\n\nNo structured block here, just prose."
    stripped, js = extract_and_strip_brief_json(memo)

    assert js is None
    assert stripped == memo  # caller falls back to the separate structuring call

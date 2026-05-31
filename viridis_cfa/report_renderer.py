import base64
import html
import os
import re
from typing import Dict, List, Optional

import markdown
import nh3

from viridis_cfa.decision_brief import load_decision_brief

REPORT_RENDERER_VERSION = "2.1"

# Allowlist for sanitizing the LLM-generated report body before it is written to a
# file and opened in headless Chrome. Covers the tags python-markdown emits and the
# heading ids the in-page TOC links rely on; everything else (e.g. a <script> echoed
# verbatim from a filing or scraped transcript) is stripped.
_SANITIZE_TAGS = {
    "a", "abbr", "b", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4",
    "h5", "h6", "hr", "i", "ins", "li", "ol", "p", "pre", "s", "span", "strong", "sub",
    "sup", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
}
_SANITIZE_ATTRS = {
    "a": {"href", "title"},
    "h1": {"id"}, "h2": {"id"}, "h3": {"id"}, "h4": {"id"}, "h5": {"id"}, "h6": {"id"},
    "td": {"align"}, "th": {"align"},
}


def _sanitize_html(rendered_html: str) -> str:
    """Strip any raw HTML the LLM may have echoed from a filing/transcript (e.g. a
    <script> tag) before the report is written to disk and loaded in headless Chrome
    for PDF export. Preserves markdown's formatting tags and heading ids."""
    return nh3.clean(rendered_html, tags=_SANITIZE_TAGS, attributes=_SANITIZE_ATTRS)


def _read_optional(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _relative_href(target_path: Optional[str], from_dir: str, require_exists: bool = True) -> Optional[str]:
    if not target_path:
        return None
    if require_exists and not os.path.exists(target_path):
        return None
    return os.path.relpath(target_path, from_dir)


def _plain_inline(markdown_text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", markdown_text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


def _extract_bold_value(markdown_text: str, label: str) -> Optional[str]:
    pattern = rf"- \*\*{re.escape(label)}\*\*: ?(.+)"
    match = re.search(pattern, markdown_text)
    if not match:
        return None
    return _plain_inline(match.group(1))


def _stance_class(stance: str) -> str:
    return {
        "Bullish": "positive",
        "Bearish": "negative",
        "Neutral": "neutral",
        "Mixed": "watch",
    }.get(stance, "neutral")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def _build_short_toc(markdown_text: str, limit: int = 10) -> str:
    links = []
    for line in markdown_text.splitlines():
        match = re.match(r"^(#{1,2})\s+(.+)$", line)
        if not match:
            continue
        title = _plain_inline(match.group(2))
        if not title or title.lower() in {"sources"}:
            continue
        links.append((len(match.group(1)), title, _slugify(title)))
        if len(links) >= limit:
            break

    if not links:
        return ""

    items = []
    for level, title, slug in links:
        level_class = "toc-sub" if level > 1 else "toc-main"
        items.append(f'<a class="{level_class}" href="#{html.escape(slug)}">{html.escape(title)}</a>')
    return "".join(items)


def _parse_scorecard_rows(scorecard_md: Optional[str]) -> List[Dict[str, str]]:
    if not scorecard_md:
        return []

    rows = []
    for line in scorecard_md.splitlines():
        if not line.startswith("| **"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 4:
            continue
        metric = re.sub(r"^\*\*|\*\*$", "", parts[0]).strip()
        value = parts[1].strip("` ")
        interpretation = re.sub(r"`", "", parts[2]).strip()
        status = parts[3].strip("` ")
        rows.append({
            "metric": metric,
            "value": value,
            "interpretation": interpretation,
            "status": status,
        })
    return rows


def _metric_tone(row: Dict[str, str]) -> str:
    """Color a scorecard row by what the engine actually asserts — never by the
    provenance word 'verified', which previously turned every computed row green
    (so negative FCF and a sub-1 quick ratio rendered as 'healthy'). Genuine
    model-zone verdicts drive tone; a plain ratio earns a verdict only when the
    number itself is notable (negative). Everything else stays neutral rather than
    fabricating a green 'healthy' claim the engine never made."""
    value = (row.get("value") or "").strip()
    interp = (row.get("interpretation") or "").lower()
    status = (row.get("status") or "").lower()
    vlow = value.lower()

    # Not-applicable / uncomputable: a data gap or business-model mismatch is not a
    # risk judgment — keep it neutral instead of alarming or reassuring.
    if "not applicable" in vlow or "not applicable" in status or "unable" in vlow or "missing" in status:
        return "neutral"
    # Advisory: the model applies only loosely to this issuer's business model;
    # soften to "watch" regardless of the raw zone label.
    if "advisory" in status or "applicability" in status:
        return "watch"
    # Explicit model-zone verdicts the engine actually computed.
    if "distress" in interp or "high risk" in interp or "weak (0-2)" in interp or "critical" in interp:
        return "negative"
    if "safe" in interp or "low risk" in interp or "strong (8-9)" in interp or "ample" in interp:
        return "positive"
    if ("grey" in interp or "moderate" in interp or "imputed" in interp or "derived" in interp
            or "tightening" in interp):
        return "watch"
    # Plain ratios with no engine verdict (Current/Quick/D-E/FCF): only a negative
    # number is notable; never paint an un-judged positive ratio green.
    numeric = value.replace("%", "").replace(",", "").strip()
    try:
        if float(numeric) < 0:
            return "negative"
    except ValueError:
        pass
    return "neutral"


def _metric_short_name(metric: str) -> str:
    replacements = {
        "Altman Z'-Score (Z-Prime)": "Altman Z'",
        "Piotroski F-Score": "Piotroski",
        "Beneish M-Score": "Beneish",
        "Debt-to-Equity": "Debt / Equity",
        "FCF / Assets Yield": "FCF / Assets",
    }
    return replacements.get(metric, metric)


def _quant_flags(scorecard_rows: List[Dict[str, str]], limit: int = 3) -> List[str]:
    flags = []
    for row in scorecard_rows:
        tone = _metric_tone(row)
        if tone not in {"negative", "watch"}:
            continue
        flags.append(f"{_metric_short_name(row['metric'])}: {row['value']} - {row['interpretation']}")
        if len(flags) >= limit:
            break
    return flags


def _artifact_links(artifact_paths: Dict[str, Optional[str]], html_dir: str) -> List[Dict[str, str]]:
    labels = {
        "final_report_md": "Markdown",
        "decision_brief": "Decision Brief JSON",
        "final_report_pdf": "PDF Export",
        "scorecard": "Quant Scorecard",
        "expert_analysis": "Expert Analysis",
        "missing_analysis": "Missing Analysis",
        "transcript_analysis": "Transcript Notes",
        "insider_activity": "Insider Activity",
    }
    links = []
    for key, label in labels.items():
        href = _relative_href(
            artifact_paths.get(key),
            html_dir,
            require_exists=(key != "final_report_pdf"),
        )
        if href:
            links.append({"label": label, "href": href})
    return links


def _render_list(items: List[str]) -> str:
    if not items:
        return '<p class="empty-note">No structured items provided.</p>'
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def _render_quant_table(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return ""
    table_rows = []
    for row in rows:
        tone = _metric_tone(row)
        table_rows.append(
            f"""
            <tr class="quant-row quant-{tone}">
                <th><span class="tone-dot"></span>{html.escape(_metric_short_name(row['metric']))}</th>
                <td class="quant-value">{html.escape(row['value'])}</td>
                <td>{html.escape(row['interpretation'])}</td>
                <td>{html.escape(row['status'])}</td>
            </tr>
            """
        )
    return f"""
    <section class="quant-section" aria-labelledby="quant-title">
        <div class="section-heading">
            <p class="eyebrow">Programmatic Layer</p>
            <h2 id="quant-title">Quant Snapshot</h2>
        </div>
        <div class="quant-table-wrap">
            <table class="quant-table">
                <thead>
                    <tr><th>Metric</th><th>Value</th><th>Read</th><th>Status</th></tr>
                </thead>
                <tbody>{''.join(table_rows)}</tbody>
            </table>
        </div>
    </section>
    """


def render_report_html(
    ticker: str,
    report_markdown: str,
    artifact_paths: Optional[Dict[str, Optional[str]]] = None,
) -> str:
    artifact_paths = artifact_paths or {}
    html_dir = os.path.join("data", "intermediate")

    md = markdown.Markdown(extensions=["extra", "tables", "toc", "sane_lists"])
    body_html = _sanitize_html(md.convert(report_markdown))

    scorecard_md = _read_optional(artifact_paths.get("scorecard"))
    scorecard_rows = _parse_scorecard_rows(scorecard_md)
    decision_brief = load_decision_brief(artifact_paths.get("decision_brief"), ticker)
    if not decision_brief:
        raise ValueError(
            f"Missing decision brief for {ticker}; company report HTML requires "
            "data/intermediate/{TICKER}_decision_brief.json"
        )

    stance = {
        "stance": decision_brief["stance"],
        "conviction": decision_brief["conviction"],
        "class": _stance_class(decision_brief["stance"]),
    }
    thesis = decision_brief["thesis"]
    positives = decision_brief["upside_drivers"]
    risks = decision_brief["risk_drivers"]
    data_quality_flags = decision_brief["data_quality_flags"]
    quant_flags = _quant_flags(scorecard_rows)

    quant_engine = _extract_bold_value(report_markdown, "Quant Engine") or "N/A"
    model = _extract_bold_value(report_markdown, "Model") or "N/A"
    filing = _extract_bold_value(report_markdown, "Filing") or "N/A"
    generated = _extract_bold_value(report_markdown, "Generated") or "N/A"
    links = _artifact_links(artifact_paths, html_dir)
    toc_html = _build_short_toc(report_markdown)
    quant_html = _render_quant_table(scorecard_rows)

    link_html = "".join(
        f'<a href="{html.escape(link["href"])}">{html.escape(link["label"])}</a>'
        for link in links
    )
    # Lead with the computed quant reads so the column delivers on its name, then
    # fill remaining slots with data-quality flags. (Previously data_quality_flags
    # was prepended and the [:5] cap dropped every quant flag.)
    flag_items = (quant_flags[:3] + data_quality_flags)[:7]
    quant_flag_html = _render_list(flag_items)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(ticker)} Investment Analysis | ViridisCFA</title>
<style>
:root {{
    color-scheme: light;
    --paper: #f7f8f4;
    --surface: #ffffff;
    --surface-muted: #f1f3ed;
    --ink: #141a17;
    --muted: #626d66;
    --line: #dde2d8;
    --line-strong: #c6cdc1;
    --accent: #0f766e;
    --blue: #1d4ed8;
    --positive: #147447;
    --watch: #8a6200;
    --negative: #b42318;
    --neutral: #475569;
    --positive-soft: #e8f5ed;
    --watch-soft: #fff4d6;
    --negative-soft: #fce9e4;
    --neutral-soft: #e9eef3;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 15px;
    line-height: 1.58;
}}
a {{ color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 3px; }}
.skip-link {{
    position: absolute;
    left: -999px;
    top: 12px;
    background: var(--ink);
    color: white;
    padding: 8px 12px;
    z-index: 10;
}}
.skip-link:focus {{ left: 12px; }}
.shell {{
    max-width: 1180px;
    margin: 0 auto;
    padding: 30px 24px 58px;
}}
.masthead {{
    border-bottom: 1px solid var(--line-strong);
    padding-bottom: 22px;
}}
.brand-row {{
    display: flex;
    justify-content: space-between;
    gap: 18px;
    align-items: center;
    margin-bottom: 26px;
}}
.brand {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    color: var(--muted);
    font-size: 12px;
    font-weight: 750;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.brand-mark {{
    width: 22px;
    height: 22px;
    border: 2px solid var(--accent);
    border-radius: 50%;
    display: inline-block;
    position: relative;
}}
.brand-mark::after {{
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--watch);
    position: absolute;
    right: -3px;
    bottom: -3px;
}}
.generated {{
    color: var(--muted);
    font-size: 13px;
}}
.title-grid {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 28px;
    align-items: end;
}}
h1 {{
    margin: 0;
    font-size: clamp(44px, 7vw, 86px);
    line-height: 0.92;
    letter-spacing: 0;
}}
.subtitle {{
    margin: 10px 0 0;
    color: var(--muted);
    font-size: 17px;
}}
.stance-panel {{
    min-width: 240px;
    border-left: 4px solid var(--neutral);
    background: var(--surface);
    padding: 16px 18px;
}}
.stance-panel.positive {{ border-left-color: var(--positive); }}
.stance-panel.negative {{ border-left-color: var(--negative); }}
.stance-panel.watch {{ border-left-color: var(--watch); }}
.stance-label,
.eyebrow,
.meta-label {{
    color: var(--muted);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.stance-value {{
    margin-top: 3px;
    font-size: 28px;
    line-height: 1.1;
    font-weight: 850;
}}
.stance-value.positive {{ color: var(--positive); }}
.stance-value.negative {{ color: var(--negative); }}
.stance-value.watch {{ color: var(--watch); }}
.stance-value.neutral {{ color: var(--neutral); }}
.conviction {{ color: var(--muted); }}
.decision-brief {{
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
    gap: 24px;
    margin: 26px 0 24px;
}}
.thesis {{
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 22px 24px;
}}
.thesis h2,
.brief-column h2,
.quant-section h2 {{
    margin: 0;
    font-size: 20px;
    line-height: 1.2;
}}
.thesis-text {{
    margin: 11px 0 0;
    font-size: 18px;
    line-height: 1.52;
}}
.brief-meta {{
    display: grid;
    gap: 1px;
    border: 1px solid var(--line);
    background: var(--line);
}}
.meta-row {{
    display: grid;
    grid-template-columns: 92px minmax(0, 1fr);
    gap: 12px;
    background: var(--surface);
    padding: 11px 13px;
}}
.meta-value {{
    color: var(--ink);
    overflow-wrap: anywhere;
}}
.brief-columns {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 22px;
}}
.brief-column {{
    border-top: 3px solid var(--line-strong);
    background: var(--surface);
    border-right: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    border-left: 1px solid var(--line);
    padding: 17px 18px;
}}
.brief-column.upside {{ border-top-color: var(--positive); }}
.brief-column.risk {{ border-top-color: var(--negative); }}
.brief-column.flags {{ border-top-color: var(--watch); }}
.brief-column ul,
.artifact-links {{
    margin: 12px 0 0;
    padding: 0;
    list-style: none;
}}
.brief-column li {{
    padding: 8px 0;
    border-top: 1px solid var(--line);
}}
.brief-column li:first-child {{ border-top: 0; }}
.empty-note {{
    color: var(--muted);
    margin: 12px 0 0;
}}
.quant-section {{
    background: var(--surface);
    border: 1px solid var(--line);
    margin-bottom: 22px;
}}
.section-heading {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    padding: 18px 20px 14px;
    border-bottom: 1px solid var(--line);
}}
.quant-table-wrap {{
    overflow-x: auto;
}}
.quant-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}}
.quant-table th,
.quant-table td {{
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid var(--line);
    vertical-align: top;
}}
.quant-table thead th {{
    color: var(--muted);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: var(--surface-muted);
}}
.quant-table tbody th {{
    width: 20%;
    font-weight: 760;
    white-space: nowrap;
}}
.quant-value {{
    width: 14%;
    font-weight: 850;
    font-size: 17px;
}}
.tone-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    background: var(--neutral);
}}
.quant-positive .tone-dot {{ background: var(--positive); }}
.quant-watch .tone-dot {{ background: var(--watch); }}
.quant-negative .tone-dot {{ background: var(--negative); }}
.quant-negative td,
.quant-negative th {{ background: var(--negative-soft); }}
.quant-watch td,
.quant-watch th {{ background: var(--watch-soft); }}
.content-layout {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 250px;
    gap: 28px;
    align-items: start;
}}
.report-body {{
    min-width: 0;
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 34px 40px;
}}
.side-rail {{
    position: sticky;
    top: 18px;
    display: grid;
    gap: 14px;
}}
.rail-section {{
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 14px;
}}
.toc-links,
.artifact-links {{
    display: grid;
    gap: 3px;
}}
.toc-links a,
.artifact-links a {{
    color: var(--ink);
    text-decoration: none;
    padding: 6px 0;
    font-size: 13px;
    border-bottom: 1px solid transparent;
}}
.toc-links a:hover,
.artifact-links a:hover {{
    color: var(--accent);
    border-bottom-color: var(--line);
}}
.toc-sub {{
    color: var(--muted) !important;
    padding-left: 10px !important;
}}
.report-body h1,
.report-body h2,
.report-body h3,
.report-body h4,
.report-body h5 {{
    letter-spacing: 0;
    line-height: 1.22;
}}
.report-body h1 {{
    margin: 0 0 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--ink);
    font-size: 28px;
}}
.report-body h2 {{
    margin: 34px 0 12px;
    padding-bottom: 7px;
    border-bottom: 1px solid var(--line);
    font-size: 22px;
}}
.report-body h3 {{ margin: 24px 0 8px; font-size: 18px; }}
.report-body h4 {{ margin: 18px 0 6px; font-size: 16px; }}
.report-body h5 {{ margin: 18px 0 6px; color: var(--muted); font-size: 14px; }}
.report-body p {{ margin: 9px 0; }}
.report-body ul,
.report-body ol {{ padding-left: 22px; }}
.report-body li {{ margin: 5px 0; }}
.report-body strong {{ font-weight: 760; }}
.report-body table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 22px;
    font-size: 14px;
}}
.report-body th {{
    background: var(--surface-muted);
    text-align: left;
    color: var(--ink);
    font-weight: 760;
}}
.report-body th,
.report-body td {{
    border: 1px solid var(--line);
    padding: 9px 10px;
    vertical-align: top;
}}
.report-body tr:nth-child(even) td {{ background: #fcfcfa; }}
.report-body code {{
    background: var(--surface-muted);
    border: 1px solid var(--line);
    padding: 1px 5px;
    font-size: 0.92em;
}}
.report-body hr {{
    border: 0;
    border-top: 1px solid var(--line);
    margin: 30px 0;
}}
@media (max-width: 980px) {{
    .title-grid,
    .decision-brief,
    .brief-columns,
    .content-layout {{
        grid-template-columns: 1fr;
    }}
    .side-rail {{ position: static; }}
    .report-body {{ padding: 26px 22px; }}
}}
@media (max-width: 620px) {{
    .shell {{ padding: 22px 16px 44px; }}
    .brand-row {{ flex-direction: column; align-items: flex-start; }}
    .stance-panel {{ min-width: 0; }}
    .quant-table th,
    .quant-table td {{ padding: 9px 10px; }}
}}
@media print {{
    @page {{ size: A4; margin: 16mm 14mm; }}
    body {{ background: white; font-size: 10pt; }}
    .shell {{ max-width: none; padding: 0; }}
    .brief-columns,
    .decision-brief,
    .content-layout {{ display: block; }}
    .side-rail {{ display: none; }}
    .thesis,
    .brief-column,
    .quant-section,
    .report-body {{ border: 0; padding-left: 0; padding-right: 0; }}
    a {{ color: inherit; text-decoration: none; }}
}}
</style>
</head>
<body>
<a class="skip-link" href="#report">Skip to report</a>
<div class="shell">
    <header class="masthead">
        <div class="brand-row">
            <div class="brand"><span class="brand-mark" aria-hidden="true"></span> ViridisCFA Research Engine</div>
            <div class="generated">{html.escape(generated)}</div>
        </div>
        <div class="title-grid">
            <div>
                <h1>{html.escape(ticker)}</h1>
                <p class="subtitle">Agent-generated investment memo</p>
            </div>
            <aside class="stance-panel {stance['class']}">
                <div class="stance-label">Current View</div>
                <div class="stance-value {stance['class']}">{html.escape(stance['stance'])}</div>
                <div class="conviction">{html.escape(stance['conviction'])} conviction</div>
            </aside>
        </div>
    </header>

    <section class="decision-brief" aria-labelledby="decision-title">
        <div class="thesis">
            <p class="eyebrow">Decision Brief</p>
            <h2 id="decision-title">Thesis</h2>
            <p class="thesis-text">{html.escape(thesis)}</p>
        </div>
        <div class="brief-meta" aria-label="Report metadata">
            <div class="meta-row"><div class="meta-label">Filing</div><div class="meta-value">{html.escape(filing)}</div></div>
            <div class="meta-row"><div class="meta-label">Quant</div><div class="meta-value">{html.escape(quant_engine)}</div></div>
            <div class="meta-row"><div class="meta-label">Model</div><div class="meta-value">{html.escape(model)}</div></div>
        </div>
    </section>

    <section class="brief-columns" aria-label="Investment summary">
        <div class="brief-column upside">
            <p class="eyebrow">Upside Case</p>
            {_render_list(positives)}
        </div>
        <div class="brief-column risk">
            <p class="eyebrow">Risk Case</p>
            {_render_list(risks)}
        </div>
        <div class="brief-column flags">
            <p class="eyebrow">Quant &amp; Data Flags</p>
            {quant_flag_html}
        </div>
    </section>

    {quant_html}

    <div class="content-layout">
        <main class="report-body" id="report">
            {body_html}
        </main>
        <aside class="side-rail">
            <section class="rail-section">
                <p class="eyebrow">Contents</p>
                <nav class="toc-links">{toc_html}</nav>
            </section>
            <section class="rail-section">
                <p class="eyebrow">Artifacts</p>
                <div class="artifact-links">{link_html}</div>
            </section>
        </aside>
    </div>
</div>
</body>
</html>"""


def render_markdown_document_html(
    title: str,
    markdown_text: str,
    artifact_paths: Optional[Dict[str, Optional[str]]] = None,
) -> str:
    artifact_paths = artifact_paths or {}
    html_dir = os.path.join("data", "intermediate")
    md = markdown.Markdown(extensions=["extra", "tables", "toc", "sane_lists"])
    body_html = _sanitize_html(md.convert(markdown_text))
    links = _artifact_links(artifact_paths, html_dir)
    link_html = "".join(
        f'<a href="{html.escape(link["href"])}">{html.escape(link["label"])}</a>'
        for link in links
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} | ViridisCFA</title>
<style>
:root {{
    color-scheme: light;
    --paper: #f7f8f4;
    --surface: #ffffff;
    --ink: #141a17;
    --muted: #626d66;
    --line: #dde2d8;
    --accent: #0f766e;
}}
* {{ box-sizing: border-box; }}
body {{
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 15px;
    line-height: 1.6;
}}
a {{ color: var(--accent); text-underline-offset: 3px; }}
.shell {{
    max-width: 1040px;
    margin: 0 auto;
    padding: 34px 24px 58px;
}}
.masthead {{
    border-bottom: 1px solid var(--line);
    margin-bottom: 24px;
    padding-bottom: 20px;
}}
.brand {{
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
h1 {{
    margin: 14px 0 0;
    font-size: clamp(34px, 5vw, 58px);
    line-height: 1;
    letter-spacing: 0;
}}
.document-body {{
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 34px 40px;
}}
.document-body h1,
.document-body h2,
.document-body h3,
.document-body h4 {{
    letter-spacing: 0;
    line-height: 1.22;
}}
.document-body h1 {{ font-size: 28px; }}
.document-body h2 {{
    margin-top: 32px;
    padding-bottom: 7px;
    border-bottom: 1px solid var(--line);
    font-size: 22px;
}}
.document-body h3 {{ font-size: 18px; }}
.document-body table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 22px;
    font-size: 14px;
}}
.document-body th,
.document-body td {{
    border: 1px solid var(--line);
    padding: 9px 10px;
    vertical-align: top;
}}
.document-body th {{ background: #f1f3ed; text-align: left; }}
.artifact-links {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 16px;
}}
@media (max-width: 620px) {{
    .shell {{ padding: 24px 16px 44px; }}
    .document-body {{ padding: 24px 20px; }}
}}
</style>
</head>
<body>
<div class="shell">
    <header class="masthead">
        <div class="brand">ViridisCFA Research Engine</div>
        <h1>{html.escape(title)}</h1>
        <nav class="artifact-links">{link_html}</nav>
    </header>
    <main class="document-body">
        {body_html}
    </main>
</div>
</body>
</html>"""


def write_markdown_document_artifact(
    title: str,
    markdown_text: str,
    artifact_paths: Dict[str, Optional[str]],
    html_path: str,
) -> Dict[str, Optional[str]]:
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    html_text = render_markdown_document_html(title, markdown_text, artifact_paths)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return {"html": html_path, "pdf": None}


def write_report_artifacts(
    ticker: str,
    report_markdown: str,
    artifact_paths: Dict[str, Optional[str]],
    html_path: str,
    pdf_path: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    html_text = render_report_html(ticker, report_markdown, artifact_paths)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    written = {"html": html_path, "pdf": None}
    if not pdf_path:
        return written

    try:
        from selenium import webdriver

        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")

        driver = webdriver.Chrome(options=options)
        try:
            driver.get("file://" + os.path.abspath(html_path))
            print_params = {
                "printBackground": True,
                "preferCSSPageSize": True,
                "marginTop": 0.65,
                "marginBottom": 0.65,
                "marginLeft": 0.55,
                "marginRight": 0.55,
            }
            pdf_data = driver.execute_cdp_cmd("Page.printToPDF", print_params)
        finally:
            driver.quit()

        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(base64.b64decode(pdf_data["data"]))
        written["pdf"] = pdf_path
    except Exception as pdf_err:
        print(f"PDF generation failed (HTML still saved): {pdf_err}")

    return written

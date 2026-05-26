import base64
import html
import os
import re
from datetime import date
from typing import Dict, List, Optional

import markdown


REPORT_RENDERER_VERSION = "1.0"


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


def _extract_bold_value(markdown_text: str, label: str) -> Optional[str]:
    pattern = rf"- \*\*{re.escape(label)}\*\*: ?(.+)"
    match = re.search(pattern, markdown_text)
    if not match:
        return None
    return match.group(1).strip()


def _extract_stance(markdown_text: str) -> Dict[str, str]:
    tail = markdown_text[-2500:]
    stance_pattern = re.compile(
        r"\*\*(Bullish|Neutral|Bearish|Mixed)\s*(?:[,\u2014-]\s*)?(Low|Medium|High)?\s*(?:conviction)?\.?\*\*",
        re.IGNORECASE,
    )
    match = stance_pattern.search(tail)
    if not match:
        return {"stance": "Unstated", "conviction": "N/A", "class": "neutral"}

    stance = match.group(1).title()
    conviction = (match.group(2) or "N/A").title()
    css_class = {
        "Bullish": "positive",
        "Bearish": "negative",
        "Neutral": "neutral",
        "Mixed": "watch",
    }.get(stance, "neutral")
    return {"stance": stance, "conviction": conviction, "class": css_class}


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
    text = f"{row['value']} {row['interpretation']} {row['status']}".lower()
    if "distress" in text or "unable" in text or "missing" in text or "high risk" in text:
        return "negative"
    if "grey" in text or "imputed" in text or "derived" in text or "moderate" in text:
        return "watch"
    if "low risk" in text or "safe" in text or "verified" in text:
        return "positive"
    return "neutral"


def _artifact_links(artifact_paths: Dict[str, Optional[str]], html_dir: str) -> List[Dict[str, str]]:
    labels = {
        "final_report_md": "Markdown",
        "final_report_pdf": "PDF Export",
        "scorecard": "Quant Scorecard",
        "expert_analysis": "Expert Analysis",
        "missing_analysis": "Missing Analysis",
        "transcript_analysis": "Transcript Notes",
        "insider_activity": "Insider Activity",
    }
    links = []
    for key, label in labels.items():
        # PDF is written after the HTML file during fresh report generation, so
        # include the link optimistically when the target path is known.
        href = _relative_href(
            artifact_paths.get(key),
            html_dir,
            require_exists=(key != "final_report_pdf"),
        )
        if href:
            links.append({"label": label, "href": href})
    return links


def render_report_html(
    ticker: str,
    report_markdown: str,
    artifact_paths: Optional[Dict[str, Optional[str]]] = None,
) -> str:
    artifact_paths = artifact_paths or {}
    html_dir = os.path.join("data", "intermediate")

    md = markdown.Markdown(extensions=["extra", "tables", "toc", "sane_lists"])
    body_html = md.convert(report_markdown)
    toc_html = md.toc if md.toc else ""

    scorecard_md = _read_optional(artifact_paths.get("scorecard"))
    scorecard_rows = _parse_scorecard_rows(scorecard_md)
    stance = _extract_stance(report_markdown)

    quant_engine = _extract_bold_value(report_markdown, "Quant Engine") or "N/A"
    model = _extract_bold_value(report_markdown, "Model") or "N/A"
    filing = _extract_bold_value(report_markdown, "Filing") or "N/A"
    generated = _extract_bold_value(report_markdown, "Generated") or date.today().isoformat()
    links = _artifact_links(artifact_paths, html_dir)

    scorecard_html = ""
    if scorecard_rows:
        metric_cards = []
        for row in scorecard_rows:
            tone = _metric_tone(row)
            metric_cards.append(
                f"""
                <article class="metric metric-{tone}">
                    <div class="metric-label">{html.escape(row['metric'])}</div>
                    <div class="metric-value">{html.escape(row['value'])}</div>
                    <div class="metric-note">{html.escape(row['interpretation'])}</div>
                    <div class="metric-status">{html.escape(row['status'])}</div>
                </article>
                """
            )
        scorecard_html = f"""
        <section class="scorecard-band" aria-labelledby="scorecard-title">
            <div>
                <p class="eyebrow">Programmatic Layer</p>
                <h2 id="scorecard-title">Quant Scorecard</h2>
            </div>
            <div class="metric-grid">
                {''.join(metric_cards)}
            </div>
        </section>
        """

    link_html = "".join(
        f'<a href="{html.escape(link["href"])}">{html.escape(link["label"])}</a>'
        for link in links
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(ticker)} Investment Analysis | ViridisCFA</title>
<style>
:root {{
    color-scheme: light;
    --paper: #fbfbf8;
    --surface: #ffffff;
    --surface-muted: #f3f5f1;
    --ink: #17201b;
    --muted: #647067;
    --line: #dfe4dc;
    --accent: #0f766e;
    --accent-soft: #d9f2ee;
    --positive: #137547;
    --positive-soft: #dff4e7;
    --watch: #9a6700;
    --watch-soft: #fff2cc;
    --negative: #b42318;
    --negative-soft: #fce4df;
    --neutral: #475569;
    --neutral-soft: #e9eef3;
    --shadow: 0 18px 45px rgba(23, 32, 27, 0.08);
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 16px;
    line-height: 1.62;
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
.hero {{
    border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, #ffffff 0%, var(--paper) 100%);
}}
.hero-inner {{
    max-width: 1220px;
    margin: 0 auto;
    padding: 34px 24px 26px;
}}
.brand-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 22px;
}}
.brand {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    color: var(--muted);
    font-size: 13px;
    font-weight: 700;
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
    position: absolute;
    width: 8px;
    height: 8px;
    right: -3px;
    bottom: -3px;
    border-radius: 50%;
    background: var(--watch);
}}
.generated {{
    color: var(--muted);
    font-size: 13px;
}}
.title-row {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 28px;
    align-items: end;
}}
h1 {{
    margin: 0;
    font-size: clamp(38px, 6vw, 72px);
    line-height: 0.95;
    letter-spacing: 0;
}}
.subtitle {{
    margin: 12px 0 0;
    color: var(--muted);
    font-size: 18px;
}}
.stance-box {{
    min-width: 220px;
    padding: 18px;
    border: 1px solid var(--line);
    background: var(--surface);
    box-shadow: var(--shadow);
}}
.stance-label {{
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.stance-value {{
    margin-top: 5px;
    font-size: 28px;
    font-weight: 800;
}}
.stance-positive {{ color: var(--positive); }}
.stance-negative {{ color: var(--negative); }}
.stance-watch {{ color: var(--watch); }}
.stance-neutral {{ color: var(--neutral); }}
.meta-strip {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1px;
    margin-top: 26px;
    background: var(--line);
    border: 1px solid var(--line);
}}
.meta-item {{
    background: var(--surface);
    padding: 14px 16px;
    min-width: 0;
}}
.meta-k {{
    color: var(--muted);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.meta-v {{
    margin-top: 4px;
    font-size: 14px;
    overflow-wrap: anywhere;
}}
.layout {{
    max-width: 1220px;
    margin: 0 auto;
    padding: 28px 24px 60px;
    display: grid;
    grid-template-columns: 270px minmax(0, 1fr);
    gap: 34px;
}}
.sidebar {{
    position: sticky;
    top: 18px;
    align-self: start;
}}
.side-section {{
    border: 1px solid var(--line);
    background: var(--surface);
    padding: 16px;
    margin-bottom: 14px;
}}
.side-title {{
    margin: 0 0 10px;
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.toc ul {{
    list-style: none;
    margin: 0;
    padding: 0;
}}
.toc li {{ margin: 0; }}
.toc a {{
    display: block;
    padding: 6px 0;
    color: var(--ink);
    font-size: 14px;
    text-decoration: none;
    border-bottom: 1px solid transparent;
}}
.toc a:hover {{ color: var(--accent); border-bottom-color: var(--accent-soft); }}
.toc ul ul {{ padding-left: 12px; }}
.toc ul ul a {{ color: var(--muted); font-size: 13px; }}
.artifact-links {{
    display: grid;
    gap: 8px;
}}
.artifact-links a {{
    display: block;
    padding: 9px 10px;
    background: var(--surface-muted);
    color: var(--ink);
    text-decoration: none;
    font-size: 14px;
    border: 1px solid transparent;
}}
.artifact-links a:hover {{ border-color: var(--accent); color: var(--accent); }}
.report {{
    min-width: 0;
}}
.scorecard-band {{
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 22px;
    margin-bottom: 24px;
    box-shadow: var(--shadow);
}}
.eyebrow {{
    margin: 0 0 4px;
    color: var(--accent);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.scorecard-band h2 {{
    margin: 0 0 16px;
    font-size: 24px;
    border: 0;
    padding: 0;
}}
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
}}
.metric {{
    border: 1px solid var(--line);
    background: var(--surface-muted);
    padding: 13px;
    min-height: 138px;
}}
.metric-positive {{ border-top: 4px solid var(--positive); background: var(--positive-soft); }}
.metric-watch {{ border-top: 4px solid var(--watch); background: var(--watch-soft); }}
.metric-negative {{ border-top: 4px solid var(--negative); background: var(--negative-soft); }}
.metric-neutral {{ border-top: 4px solid var(--neutral); background: var(--neutral-soft); }}
.metric-label {{
    color: var(--muted);
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
}}
.metric-value {{
    margin-top: 7px;
    font-size: 25px;
    line-height: 1.05;
    font-weight: 850;
    overflow-wrap: anywhere;
}}
.metric-note {{
    margin-top: 8px;
    font-size: 13px;
    color: var(--ink);
}}
.metric-status {{
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
}}
.report-body {{
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 36px 42px;
    box-shadow: var(--shadow);
}}
.report-body h1,
.report-body h2,
.report-body h3,
.report-body h4,
.report-body h5 {{
    letter-spacing: 0;
    line-height: 1.2;
}}
.report-body h1 {{
    margin: 0 0 18px;
    padding-bottom: 12px;
    border-bottom: 2px solid var(--ink);
    font-size: 30px;
}}
.report-body h2 {{
    margin: 34px 0 12px;
    padding-bottom: 7px;
    border-bottom: 1px solid var(--line);
    font-size: 24px;
}}
.report-body h3 {{ margin: 24px 0 8px; font-size: 19px; }}
.report-body h4 {{ margin: 18px 0 6px; font-size: 16px; }}
.report-body h5 {{ margin: 18px 0 6px; font-size: 14px; color: var(--muted); }}
.report-body p {{ margin: 9px 0; }}
.report-body ul,
.report-body ol {{ padding-left: 22px; }}
.report-body li {{ margin: 5px 0; }}
.report-body strong {{ font-weight: 750; }}
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
    font-weight: 750;
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
    .title-row,
    .layout,
    .meta-strip {{
        grid-template-columns: 1fr;
    }}
    .sidebar {{ position: static; }}
    .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .report-body {{ padding: 26px 22px; }}
}}
@media (max-width: 620px) {{
    .hero-inner,
    .layout {{ padding-left: 16px; padding-right: 16px; }}
    .brand-row {{ align-items: flex-start; flex-direction: column; }}
    .metric-grid {{ grid-template-columns: 1fr; }}
    .stance-box {{ min-width: 0; }}
}}
@media print {{
    @page {{ size: A4; margin: 16mm 14mm; }}
    body {{ background: white; font-size: 10pt; }}
    .hero, .sidebar, .scorecard-band {{ box-shadow: none; }}
    .layout {{ display: block; padding: 0; }}
    .sidebar {{ display: none; }}
    .report-body {{ border: 0; box-shadow: none; padding: 0; }}
    .scorecard-band {{ page-break-inside: avoid; margin: 0 0 18px; }}
    a {{ color: inherit; text-decoration: none; }}
}}
</style>
</head>
<body>
<a class="skip-link" href="#report">Skip to report</a>
<header class="hero">
    <div class="hero-inner">
        <div class="brand-row">
            <div class="brand"><span class="brand-mark" aria-hidden="true"></span> ViridisCFA</div>
            <div class="generated">{html.escape(generated)}</div>
        </div>
        <div class="title-row">
            <div>
                <h1>{html.escape(ticker)}</h1>
                <p class="subtitle">Investment Analysis Report</p>
            </div>
            <aside class="stance-box">
                <div class="stance-label">Stance</div>
                <div class="stance-value stance-{stance['class']}">{html.escape(stance['stance'])}</div>
                <div>{html.escape(stance['conviction'])} conviction</div>
            </aside>
        </div>
        <div class="meta-strip" aria-label="Report metadata">
            <div class="meta-item"><div class="meta-k">Filing</div><div class="meta-v">{html.escape(filing)}</div></div>
            <div class="meta-item"><div class="meta-k">Quant Engine</div><div class="meta-v">{html.escape(quant_engine)}</div></div>
            <div class="meta-item"><div class="meta-k">Model</div><div class="meta-v">{html.escape(model)}</div></div>
        </div>
    </div>
</header>
<div class="layout">
    <aside class="sidebar">
        <section class="side-section toc">
            <p class="side-title">Contents</p>
            {toc_html}
        </section>
        <section class="side-section">
            <p class="side-title">Artifacts</p>
            <div class="artifact-links">{link_html}</div>
        </section>
    </aside>
    <main class="report" id="report">
        {scorecard_html}
        <article class="report-body">
            {body_html}
        </article>
    </main>
</div>
</body>
</html>"""


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

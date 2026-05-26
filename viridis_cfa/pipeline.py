import os
import time
import markdown
from datetime import datetime, date
from edgar import Company
from openai import RateLimitError, APITimeoutError

# Local package imports
from prompt_configs import (
    expert_analysis_prompt_template,
    missing_analysis_prompt_template,
    transcript_prompt_template,
    final_juice_prompt_template,
    batch_comparison_prompt_template
)
from utils import count_tokens, estimate_cost, calculate_actual_cost
from viridis_cfa.scrapers import get_transcript, html_to_markdown
from viridis_cfa.insider import get_insider_activity
from viridis_cfa.config import get_openai_client

def run_analysis(prompt, model="gpt-5.4", reasoning_effort="medium", service_tier="flex"):
    """Run analysis using OpenAI Responses API with configurable reasoning and pricing tier.
    
    service_tier='flex' gives batch API rates (~50% off) but slower + may get 429.
    Falls back to standard automatically on resource unavailable.
    """
    client = get_openai_client()
    if not client:
        print("OpenAI API client not initialized. Check OPENAI_API_KEY.")
        return None
    
    # Count tokens and estimate cost
    prompt_tokens = count_tokens(prompt)
    print(f"Prompt contains approximately {prompt_tokens:,} tokens")
    
    estimated_costs = estimate_cost(prompt_tokens, model)
    print(f"Estimated cost: ${estimated_costs['total_cost']:.4f} (before flex discount)")
    print(f"Reasoning: {reasoning_effort} | Tier: {service_tier}")
    
    def _call(tier):
        return client.with_options(timeout=900).responses.create(
            model=model,
            instructions="You are a high IQ expert financial engineer.",
            input=prompt,
            reasoning={"effort": reasoning_effort},
            service_tier=tier,
        )
    
    try:
        response = _call(service_tier)
        
    except RateLimitError as e:
        if service_tier == "flex":
            print(f"Flex unavailable, falling back to standard: {e}")
            try:
                response = _call("auto")
            except Exception as e2:
                print(f"Standard fallback also failed: {e2}")
                return None
        else:
            print(f"Rate limited: {e}")
            import time
            time.sleep(30)
            try:
                response = _call(service_tier)
            except Exception as e2:
                print(f"Retry failed: {e2}")
                return None
                
    except APITimeoutError as e:
        print(f"Request timed out: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    # Calculate actual cost from usage
    actual_prompt_tokens = response.usage.input_tokens
    actual_completion_tokens = response.usage.output_tokens
    
    # Log token breakdown (reasoning tokens are billed as output tokens)
    reasoning_tokens = getattr(response.usage.output_tokens_details, 'reasoning_tokens', 0) or 0
    visible_tokens = actual_completion_tokens - reasoning_tokens
    cached_input = getattr(response.usage.input_tokens_details, 'cached_tokens', 0) or 0
    used_tier = getattr(response, 'service_tier', service_tier)
    
    actual_costs = calculate_actual_cost(
        actual_prompt_tokens, 
        actual_completion_tokens, 
        model
    )
    
    # Flex is ~50% off standard rates
    discount = " (~50% off with flex)" if used_tier == "flex" else ""
    print(f"Tokens — input: {actual_prompt_tokens:,} (cached: {cached_input:,}) | output: {visible_tokens:,} | reasoning: {reasoning_tokens:,}")
    print(f"Actual cost: ${actual_costs['total_cost']:.4f}{discount} | Tier: {used_tier}")
    
    return response.output_text, actual_costs['total_cost']

def get_multi_year_trends(company):
    """Fetch multi-year financial trends from XBRL EntityFacts.
    Returns a formatted string with historical data, or empty string on failure."""
    try:
        facts = company.get_facts()
        if not facts:
            return ""
        
        sections = []
        
        # Multi-year income statement (4 years)
        try:
            hist_income = facts.income_statement(periods=4, period='annual')
            if hist_income:
                sections.append(f"MULTI-YEAR INCOME STATEMENT (from XBRL data across multiple filings):\n{str(hist_income)}")
        except Exception:
            pass
        
        # Multi-year balance sheet
        try:
            hist_balance = facts.balance_sheet(periods=4, period='annual')
            if hist_balance:
                sections.append(f"MULTI-YEAR BALANCE SHEET:\n{str(hist_balance)}")
        except Exception:
            pass
        
        if sections:
            return "\n\n--- HISTORICAL FINANCIAL CONTEXT (4-YEAR TRENDS) ---\n\n" + "\n\n".join(sections) + "\n\n--- END HISTORICAL CONTEXT ---\n"
        return ""
    except Exception as e:
        print(f"Could not fetch multi-year trends: {e}")
        return ""

def process_filing(ticker, filing, company=None, no_cache=False):
    """Process a single filing and return the expert and missing analyses"""
    print(f"Processing filing: {filing.accession_no} ({filing.form}, {filing.filing_date})")
    
    # Sanitize form name for filenames (e.g. 10-K/A -> 10-KA)
    safe_form = filing.form.replace("/", "")
    
    expert_filename = f"{ticker}-{safe_form}-{filing.filing_date}-expert-analysis.md"
    expert_path = os.path.join("data", "filings", expert_filename)
    
    missing_filename = f"{ticker}-{safe_form}-{filing.filing_date}-missing-analysis.md"
    missing_path = os.path.join("data", "filings", missing_filename)
    
    if not no_cache and os.path.exists(expert_path) and os.path.exists(missing_path):
        print(f"[CACHE HIT] Found existing expert and missing analyses for {ticker} {filing.filing_date}. Loading from cache...")
        try:
            with open(expert_path, "r", encoding="utf-8") as f:
                expert_analysis = f.read()
            with open(missing_path, "r", encoding="utf-8") as f:
                missing_analysis = f.read()
            return expert_analysis, missing_analysis, 0.0
        except Exception as cache_err:
            print(f"Failed to read cache: {cache_err}. Proceeding with fresh fetch/analysis...")
            
    try:
        # Get filing text as markdown (preserves headings, tables, structure)
        filing_text = filing.markdown()
        print(f"Filing text contains approximately {count_tokens(filing_text):,} tokens")
        
        # Save raw filing text
        os.makedirs(os.path.join("data", "filings"), exist_ok=True)
        raw_filename = f"{ticker}-{safe_form}-{filing.filing_date}-raw.md"
        with open(os.path.join("data", "filings", raw_filename), "w", encoding='utf-8') as f:
            f.write(filing_text)
        print(f"Raw filing saved to {raw_filename}")
        
        # Fetch multi-year trends and programmatic scorecard to augment the prompt
        trend_context = ""
        quant_scorecard_md = ""
        if company:
            print("Fetching multi-year financial trends from XBRL...")
            trend_context = get_multi_year_trends(company)
            if trend_context:
                print(f"Added {count_tokens(trend_context):,} tokens of historical context")
                # Save trend context for reference
                trend_filename = f"{ticker}-{safe_form}-{filing.filing_date}-xbrl-trends.md"
                with open(os.path.join("data", "filings", trend_filename), "w", encoding='utf-8') as f:
                    f.write(trend_context)
                print(f"XBRL trends saved to {trend_filename}")
            else:
                print("No historical trend data available")
                
            # Programmatic Scorecard Generation
            try:
                from quant_engine import generate_quant_scorecard
                print("Calculating deterministic quantitative scorecard...")
                quant_scorecard_md, _ = generate_quant_scorecard(company)
                
                # Save scorecard to filings folder
                scorecard_filename = f"{ticker}-{safe_form}-{filing.filing_date}-quant-scorecard.md"
                with open(os.path.join("data", "filings", scorecard_filename), "w", encoding='utf-8') as f:
                    f.write(quant_scorecard_md)
                print(f"Programmatic quant scorecard saved to {scorecard_filename}")
            except Exception as q_e:
                print(f"Could not calculate quant scorecard: {q_e}")
                quant_scorecard_md = "Unable to programmatically generate quantitative scorecard."
        
        # Step 1: Expert Analysis
        print("\n--- Running Expert Analysis ---")
        expert_prompt = expert_analysis_prompt_template.format(
            filing_text=filing_text + trend_context,
            quant_scorecard=quant_scorecard_md
        )
        expert_result = run_analysis(expert_prompt, reasoning_effort="high")
        expert_analysis, expert_cost = expert_result if expert_result else (None, 0)
        
        if not expert_analysis:
            return None, None, 0
            
        # Save expert analysis
        os.makedirs(os.path.join("data", "filings"), exist_ok=True)
        expert_filename = f"{ticker}-{safe_form}-{filing.filing_date}-expert-analysis.md"
        with open(os.path.join("data", "filings", expert_filename), "w", encoding='utf-8') as f:
            f.write(expert_analysis)
        print(f"Expert analysis saved to {expert_filename}")
        
        # Step 2: Missing Analysis
        print("\n--- Running Missing Analysis ---")
        missing_prompt = missing_analysis_prompt_template.format(
            expert_analysis=expert_analysis,
            filing_text=filing_text
        )
        missing_result = run_analysis(missing_prompt)
        missing_analysis, missing_cost = missing_result if missing_result else (None, 0)
        
        total_cost = expert_cost + missing_cost
        
        if missing_analysis:
            # Save missing analysis
            missing_filename = f"{ticker}-{safe_form}-{filing.filing_date}-missing-analysis.md"
            with open(os.path.join("data", "filings", missing_filename), "w", encoding='utf-8') as f:
                f.write(missing_analysis)
            print(f"Missing analysis saved to {missing_filename}")
            
        return expert_analysis, missing_analysis, total_cost
        
    except Exception as e:
        print(f"Could not process filing {filing.accession_no}: {e}")
        return None, None, 0

def process_transcript(ticker, no_cache=False):
    """Process transcript and return the analysis"""
    transcript_analysis_filename = f"{ticker}-transcript-analysis.md"
    transcript_analysis_path = os.path.join("data", "transcripts", transcript_analysis_filename)
    
    if not no_cache and os.path.exists(transcript_analysis_path):
        print(f"[CACHE HIT] Found existing transcript analysis for {ticker}. Loading from cache...")
        try:
            with open(transcript_analysis_path, "r", encoding="utf-8") as f:
                transcript_analysis = f.read()
            return transcript_analysis, 0.0
        except Exception as cache_err:
            print(f"Failed to read cache: {cache_err}. Proceeding with fresh fetch/analysis...")
            
    print("\n--- Fetching Transcript ---")
    
    html = get_transcript(ticker)
    
    if not html:
        print("No transcript found")
        return None, 0.0
        
    # Save HTML transcript
    os.makedirs(os.path.join("data", "transcripts"), exist_ok=True)
    html_path = os.path.join("data", "transcripts", f"{ticker}_transcript.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    transcript_text = html_to_markdown(html)
    md_path = os.path.join("data", "transcripts", f"{ticker}_transcript.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    print(f"Transcript saved to {md_path}")
    
    # Run transcript analysis
    print("\n--- Running Transcript Analysis ---")
    transcript_prompt = transcript_prompt_template.format(transcript_text=transcript_text)
    result = run_analysis(transcript_prompt)
    transcript_analysis, cost = result if result else (None, 0)
    
    if transcript_analysis:
        # Save transcript analysis
        transcript_analysis_filename = f"{ticker}-transcript-analysis.md"
        with open(os.path.join("data", "transcripts", transcript_analysis_filename), "w", encoding='utf-8') as f:
            f.write(transcript_analysis)
        print(f"Transcript analysis saved to {transcript_analysis_filename}")
        
    return transcript_analysis, cost

def create_final_analysis(ticker, expert_analysis, missing_analysis, transcript_analysis=None, filing_date=None, transcript_date=None, insider_activity=None, filing_accession_no=None):
    """Create the final analysis from all components. Transcript and insider data are optional."""
    if not (expert_analysis and missing_analysis):
        print("Missing required filing analyses for final report")
        return 0
        
    print("\n--- Running Final Analysis ---")
    
    # Build date context for the synthesis prompt
    date_parts = []
    if filing_date:
        date_parts.append(f"The SEC filing was filed on {filing_date}.")
    if transcript_date:
        date_parts.append(f"The earnings call transcript is from {transcript_date}.")
    if filing_date and transcript_date and str(filing_date) != str(transcript_date):
        date_parts.append("Note: The filing and transcript may cover different reporting periods. Flag any data that may be outdated or mismatched.")
    date_context = " ".join(date_parts) if date_parts else ""
    
    # Build transcript sections conditionally
    if transcript_analysis:
        transcript_intro = ", and finally a report on the earnings call transcript"
        transcript_section = f"\n\nThe third report analysis (earnings call transcript) is:\n\n{transcript_analysis}"
    else:
        transcript_intro = ""
        transcript_section = "\n\nNote: No earnings call transcript was available for this company. Base the final report on the filing analyses only."
    
    # Build insider activity section
    if insider_activity:
        transcript_section += f"\n\n--- INSIDER TRADING ACTIVITY (SEC Form 4) ---\n\n{insider_activity}\n\n--- END INSIDER ACTIVITY ---\n\nImportant: Cross-reference the insider trading dates and patterns with the filing date and any material events. Coordinated selling by multiple insiders around key dates, or insider buying during weakness, are particularly significant signals to highlight in the Investment Conclusion."
    
    # Format the final prompt
    final_prompt = final_juice_prompt_template.format(
        expert_analysis=expert_analysis,
        missing_analysis=missing_analysis,
        transcript_intro=transcript_intro,
        transcript_section=transcript_section,
        date_context=date_context
    )
    
    result = run_analysis(final_prompt)
    final_analysis, cost = result if result else (None, 0)
    
    if final_analysis:
        # Append provenance footer (deterministic — LLM cannot omit or rephrase)
        from quant_engine import QUANT_ENGINE_VERSION
        from datetime import datetime, timezone
        # Derive model/tier from run_analysis defaults rather than hardcoding
        import inspect
        _defaults = inspect.signature(run_analysis).parameters
        _model = _defaults['model'].default
        _tier = _defaults['service_tier'].default
        provenance_lines = [
            "",
            "---",
            "##### Report Provenance",
            f"- **Quant Engine**: v{QUANT_ENGINE_VERSION}",
            f"- **Model**: {_model} ({_tier})",
            f"- **Filing**: {filing_date or 'N/A'} | Accession: `{filing_accession_no or 'N/A'}`",
            f"- **Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            f"- **Pipeline**: ViridisCFA Research Engine",
        ]
        final_analysis += "\n".join(provenance_lines)
        
        # Save markdown and HTML intermediates to a subfolder
        intermediate_dir = os.path.join("data", "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        
        md_filename = f"{ticker}_final_report.md"
        md_path = os.path.join(intermediate_dir, md_filename)
        with open(md_path, "w", encoding='utf-8') as f:
            f.write(final_analysis)
        print(f"Final analysis saved to intermediate/{md_filename}")
        
        # Convert to styled HTML + PDF
        try:
            # Convert markdown to HTML body
            body_html = markdown.markdown(final_analysis, extensions=['extra', 'tables'])
            
            # Wrap in a professional styled template
            full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    @page {{
        size: A4;
        margin: 2.5cm 2cm;
    }}
    
    body {{
        font-family: 'Inter', sans-serif;
        font-size: 10pt;
        line-height: 1.6;
        color: #1e293b;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }}
    
    /* Title page header */
    .report-header {{
        text-align: center;
        padding: 40px 0 30px 0;
        border-bottom: 3px solid #0f172a;
        margin-bottom: 30px;
    }}
    .report-header h1 {{
        font-size: 22pt;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }}
    .report-header .subtitle {{
        font-size: 10pt;
        font-weight: 400;
        color: #64748b;
    }}
    
    /* Headings */
    h1 {{ font-size: 16pt; font-weight: 700; color: #0f172a; margin-top: 28px; padding-bottom: 6px; border-bottom: 2px solid #e2e8f0; }}
    h2 {{ font-size: 13pt; font-weight: 600; color: #1e40af; margin-top: 22px; }}
    h3 {{ font-size: 11pt; font-weight: 600; color: #334155; margin-top: 16px; }}
    
    /* Lists */
    ul, ol {{ margin: 8px 0; padding-left: 20px; }}
    li {{ margin-bottom: 4px; }}
    
    /* Bold emphasis */
    strong {{ font-weight: 600; color: #0f172a; }}
    
    /* Tables */
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 9pt; }}
    th {{ background: #f1f5f9; font-weight: 600; text-align: left; padding: 8px 10px; border: 1px solid #e2e8f0; color: #334155; }}
    td {{ padding: 6px 10px; border: 1px solid #e2e8f0; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
    
    /* Paragraphs */
    p {{ margin: 8px 0; }}
    
    /* Code / data blocks */
    code {{ font-family: 'Courier New', monospace; font-size: 9pt; background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }}
    pre {{ background: #f8fafc; padding: 12px; border-radius: 4px; border: 1px solid #e2e8f0; font-size: 9pt; overflow-x: auto; }}
</style>
</head>
<body>
    <div class="report-header">
        <h1>{ticker} — Investment Analysis Report</h1>
        <div class="subtitle">Generated {date.today().strftime('%B %d, %Y')} · ViridisCFA Research Pipeline</div>
    </div>
    {body_html}
</body>
</html>"""
            
            # Save HTML to intermediate folder
            html_filename = f"{ticker}_final_report.html"
            html_path = os.path.join(intermediate_dir, html_filename)
            with open(html_path, "w", encoding='utf-8') as f:
                f.write(full_html)
            print(f"HTML version saved to intermediate/{html_filename}")
            
            # Generate PDF via headless Chrome (reuses Selenium already installed for transcripts)
            try:
                from selenium import webdriver
                import base64
                
                options = webdriver.ChromeOptions()
                options.add_argument("--headless=new")
                options.add_argument("--disable-gpu")
                
                driver = webdriver.Chrome(options=options)
                driver.get("file://" + os.path.abspath(html_path))
                
                # Wait for fonts to load
                import time
                time.sleep(2)
                
                # Print to PDF via Chrome DevTools Protocol
                print_params = {
                    "printBackground": True,
                    "preferCSSPageSize": True,
                    "marginTop": 0.8,
                    "marginBottom": 0.8,
                    "marginLeft": 0.6,
                    "marginRight": 0.6,
                }
                pdf_data = driver.execute_cdp_cmd("Page.printToPDF", print_params)
                
                driver.quit()
                
                # PDF goes to data/ root — the main deliverable
                pdf_filename = f"{ticker}_final_report.pdf"
                pdf_path = os.path.join("data", pdf_filename)
                with open(pdf_path, "wb") as f:
                    f.write(base64.b64decode(pdf_data["data"]))
                print(f"PDF saved to {pdf_filename}")
                
            except Exception as pdf_err:
                print(f"PDF generation failed (HTML still saved): {pdf_err}")
            
        except Exception as e:
            print(f"Error creating report: {e}")
    
    return cost

def _get_latest_non_amended(company, form):
    """Get the latest filing of the given form type, skipping amendments (e.g. 10-K/A)."""
    filings = company.get_filings(form=form)
    for f in filings:
        if '/A' not in f.form:
            return f
    return None

def _read_file(path):
    """Read a text file and return its contents, or None if not found."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except (FileNotFoundError, IOError):
        return None


def analyze_ticker(ticker, no_cache=False):
    """Run the full analysis pipeline for a single ticker. Returns (cost, final_analysis_text).
    
    Uses ingredient-based caching: compares current data sources against a cached
    manifest to determine which pipeline steps need re-running. Only LLM steps whose
    inputs actually changed are re-executed. Filing freshness is checked via SEC
    accession number (lightweight). Insider data is always re-fetched (cheap) and
    hash-compared. Transcript is re-scraped only when the filing quarter changes or
    no transcript was previously available.
    """
    from concurrent.futures import ThreadPoolExecutor
    from viridis_cfa.cache import (
        load_manifest, save_manifest, check_artifacts_exist,
        compute_hash, now_iso
    )
    from quant_engine import QUANT_ENGINE_VERSION
    
    print(f"\n{'='*60}")
    print(f"  Analyzing {ticker}")
    print(f"{'='*60}")
    
    company = Company(ticker)
    
    # Get the most recent non-amended filing (whichever is newer: 10-Q or 10-K)
    latest_10q = _get_latest_non_amended(company, '10-Q')
    latest_10k = _get_latest_non_amended(company, '10-K')
    
    if latest_10q and latest_10k:
        filing = latest_10q if latest_10q.filing_date >= latest_10k.filing_date else latest_10k
    else:
        filing = latest_10q or latest_10k
    
    if not filing:
        print(f"No 10-Q or 10-K found for {ticker}")
        return 0, None
    
    print(f"Selected: {filing.form} filed {filing.filing_date}")
    print(filing)
 
    start_time = time.time()
    
    # ── Load cached manifest ──
    manifest = None
    cached = {}
    if not no_cache:
        manifest = load_manifest(ticker)
        if manifest:
            if not check_artifacts_exist(manifest):
                print("[CACHE] Manifest invalid — artifact files missing. Running fresh.")
                manifest = None
            else:
                cached = manifest.get('ingredients', {})
    
    # ── Determine staleness per ingredient ──
    safe_form = filing.form.replace("/", "")
    
    filing_stale = (not manifest or
                    cached.get('filing_accession_no') != filing.accession_no or
                    cached.get('quant_engine_version') != QUANT_ENGINE_VERSION)
    
    # Transcript: re-scrape if quarter changed or no transcript was cached previously
    transcript_stale = (filing_stale or not cached.get('transcript_hash'))
    
    # Log cache status
    if no_cache:
        print("[CACHE] Bypassed (--no-cache / --force)")
    elif filing_stale:
        reasons = []
        if not manifest:
            reasons.append("no manifest")
        elif cached.get('filing_accession_no') != filing.accession_no:
            reasons.append(f"new filing {filing.accession_no[:20]}...")
        elif cached.get('quant_engine_version') != QUANT_ENGINE_VERSION:
            reasons.append("quant engine updated")
        print(f"[CACHE] Filing stale ({', '.join(reasons)}) — full re-analysis required")
    elif transcript_stale:
        print("[CACHE] Filing cached ✓ | Transcript not yet cached — will attempt scrape")
    else:
        print("[CACHE] Filing cached ✓ | Transcript cached ✓ | Checking insider data...")
    
    # ── Run pipeline branches ──
    total_cost = 0
    steps_run = []
    cache_hits = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        
        # Branch 1: Filing analysis (only if stale)
        if filing_stale:
            futures['filing'] = executor.submit(
                process_filing, ticker, filing, company, no_cache=True
            )
        
        # Branch 2: Transcript (only if stale)
        if transcript_stale:
            futures['transcript'] = executor.submit(
                process_transcript, ticker, no_cache=True
            )
        
        # Branch 3: Insider data (always fetch — cheap, continuous updates)
        futures['insider'] = executor.submit(
            get_insider_activity, company, filing.filing_date
        )
        
        # ── Collect filing results ──
        if 'filing' in futures:
            expert_analysis, missing_analysis, filing_cost = futures['filing'].result()
            total_cost += filing_cost
            steps_run += ['expert', 'missing']
        else:
            expert_analysis = _read_file(manifest['artifacts']['expert_analysis'])
            missing_analysis = _read_file(manifest['artifacts']['missing_analysis'])
            cache_hits += ['expert', 'missing']
            print(f"[CACHE HIT] Reusing expert + missing analyses for {ticker}")
        
        # ── Collect transcript results ──
        if 'transcript' in futures:
            transcript_analysis, transcript_cost = futures['transcript'].result()
            total_cost += transcript_cost
            if transcript_analysis:
                steps_run.append('transcript')
        elif manifest and manifest['artifacts'].get('transcript_analysis'):
            transcript_analysis = _read_file(manifest['artifacts']['transcript_analysis'])
            if transcript_analysis:
                cache_hits.append('transcript')
                print(f"[CACHE HIT] Reusing transcript analysis for {ticker}")
            else:
                transcript_analysis = None
        else:
            transcript_analysis = None
        
        # ── Collect insider results (always fresh) ──
        insider_activity = futures['insider'].result()
    
    elapsed = time.time() - start_time
    print(f"\nData collection completed in {elapsed:.1f}s")
    
    # ── Save insider data (always, since we always fetch it) ──
    if insider_activity:
        insider_lines = insider_activity.count('\n')
        print(f"Insider activity: {insider_lines} transactions found")
        os.makedirs(os.path.join("data", "filings"), exist_ok=True)
        insider_filename = f"{ticker}-insider-activity.md"
        with open(os.path.join("data", "filings", insider_filename), "w", encoding='utf-8') as f:
            f.write(insider_activity)
        print(f"Insider activity saved to filings/{insider_filename}")
    else:
        print("No insider market transactions found")
    
    # ── Hash insider data and compare with cache ──
    insider_hash = compute_hash(insider_activity) if insider_activity else None
    insider_changed = insider_hash != cached.get('insider_hash')
    
    if insider_changed and not filing_stale:
        print("[CACHE] Insider activity changed — final synthesis will re-run")
    
    # ── Compute transcript hash for manifest ──
    if 'transcript' in futures and transcript_analysis:
        transcript_hash = compute_hash(transcript_analysis)
    elif manifest:
        transcript_hash = cached.get('transcript_hash')
    else:
        transcript_hash = None
    
    # ── Final synthesis ──
    # Re-run if any upstream ingredient changed
    need_final = filing_stale or transcript_stale or insider_changed
    final_text = None
    
    if need_final and expert_analysis and missing_analysis:
        # Extract transcript date from the transcript analysis if available
        transcript_date = None
        if transcript_analysis:
            for line in transcript_analysis.split('\n')[:20]:
                if 'call date' in line.lower() or 'date' in line.lower():
                    transcript_date = line.strip()
                    break
        
        final_cost = create_final_analysis(
            ticker, expert_analysis, missing_analysis,
            transcript_analysis=transcript_analysis,
            filing_date=filing.filing_date,
            transcript_date=transcript_date,
            insider_activity=insider_activity,
            filing_accession_no=filing.accession_no
        )
        total_cost += final_cost
        steps_run.append('final')
        
        # Read back the final report for batch comparison
        final_md_path = os.path.join("data", "intermediate", f"{ticker}_final_report.md")
        final_text = _read_file(final_md_path)
    
    elif not need_final and manifest and manifest['artifacts'].get('final_report_md'):
        # Full cache hit — serve the cached final report
        final_text = _read_file(manifest['artifacts']['final_report_md'])
        cache_hits.append('final')
        print(f"[FULL CACHE HIT] Report unchanged for {ticker} — $0.00 LLM cost")
    
    elif expert_analysis and missing_analysis:
        # No manifest but we have analyses (first run, or manifest was invalidated)
        transcript_date = None
        if transcript_analysis:
            for line in transcript_analysis.split('\n')[:20]:
                if 'call date' in line.lower() or 'date' in line.lower():
                    transcript_date = line.strip()
                    break
        
        final_cost = create_final_analysis(
            ticker, expert_analysis, missing_analysis,
            transcript_analysis=transcript_analysis,
            filing_date=filing.filing_date,
            transcript_date=transcript_date,
            insider_activity=insider_activity,
            filing_accession_no=filing.accession_no
        )
        total_cost += final_cost
        steps_run.append('final')
        
        final_md_path = os.path.join("data", "intermediate", f"{ticker}_final_report.md")
        final_text = _read_file(final_md_path)
    
    # ── Update manifest ──
    artifact_paths = {
        'expert_analysis': os.path.join("data", "filings",
            f"{ticker}-{safe_form}-{filing.filing_date}-expert-analysis.md"),
        'missing_analysis': os.path.join("data", "filings",
            f"{ticker}-{safe_form}-{filing.filing_date}-missing-analysis.md"),
        'transcript_analysis': os.path.join("data", "transcripts",
            f"{ticker}-transcript-analysis.md") if transcript_analysis else None,
        'insider_activity': os.path.join("data", "filings",
            f"{ticker}-insider-activity.md") if insider_activity else None,
        'final_report_md': os.path.join("data", "intermediate",
            f"{ticker}_final_report.md"),
        'final_report_pdf': os.path.join("data",
            f"{ticker}_final_report.pdf"),
    }
    
    previous_runs = manifest.get('runs', []) if manifest else []
    new_manifest = {
        'ticker': ticker,
        'ingredients': {
            'filing_accession_no': filing.accession_no,
            'filing_form': filing.form,
            'filing_date': str(filing.filing_date),
            'quant_engine_version': QUANT_ENGINE_VERSION,
            'transcript_hash': transcript_hash,
            'insider_hash': insider_hash,
        },
        'artifacts': artifact_paths,
        'runs': previous_runs + [{
            'timestamp': now_iso(),
            'cost': total_cost,
            'steps_run': steps_run,
            'cache_hits': cache_hits,
        }],
    }
    save_manifest(ticker, new_manifest)
 
    total_elapsed = time.time() - start_time
    print(f"\n--- {ticker} Complete ({total_elapsed:.1f}s) | Cost: ${total_cost:.4f} ---")
    if cache_hits:
        print(f"    Cache hits: {', '.join(cache_hits)}")
    if steps_run:
        print(f"    Steps run:  {', '.join(steps_run)}")
    return total_cost, final_text


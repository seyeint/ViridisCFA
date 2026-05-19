import os
from dotenv import load_dotenv
from edgar import *
from openai import OpenAI, RateLimitError, APITimeoutError
from prompt_configs import *
from utils import count_tokens, estimate_cost, calculate_actual_cost
from fetch_transcripts import get_transcript, html_to_markdown
import markdown
# Load .env file
load_dotenv()
edgar_identity = os.getenv("EDGAR_IDENTITY")
set_identity(edgar_identity)
print(f"Using EDGAR Identity: {edgar_identity}")

# Load OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client (sync — no need for async in a sequential pipeline)
client = OpenAI(api_key=openai_key) if openai_key else None

def run_analysis(prompt, model="gpt-5.4", reasoning_effort="medium", service_tier="flex"):
    """Run analysis using OpenAI Responses API with configurable reasoning and pricing tier.
    
    service_tier='flex' gives batch API rates (~50% off) but slower + may get 429.
    Falls back to standard automatically on resource unavailable.
    """
    if not client:
        print("OpenAI API key not found in .env file.")
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

def get_insider_activity(company, filing_date, max_filings=40):
    """Fetch recent insider transactions (Form 4) and return a formatted summary.
    Only includes market buys/sells — filters out tax withholding, gifts, awards.
    Window: 6 months before filing date to present."""
    from datetime import datetime, timedelta
    
    try:
        filings = company.get_filings(form='4')
        if not filings or len(filings) == 0:
            return ""
        
        # Parse filing_date for window calculation
        if isinstance(filing_date, str):
            ref_date = datetime.strptime(filing_date, '%Y-%m-%d').date()
        else:
            ref_date = filing_date
        
        window_start = ref_date - timedelta(days=180)
        
        transactions = []
        processed = 0
        
        for f in filings:
            if processed >= max_filings:
                break
            if '/A' in f.form:
                continue
            
            # Only look at filings within our window
            f_date = f.filing_date
            if hasattr(f_date, 'date'):
                f_date = f_date.date()
            elif isinstance(f_date, str):
                f_date = datetime.strptime(f_date, '%Y-%m-%d').date()
            
            if f_date < window_start:
                break  # Filings are reverse chronological, so we can stop
            
            processed += 1
            
            try:
                obj = f.obj()
                df = obj.to_dataframe()
                if df is None or len(df) == 0:
                    continue
                
                # Check for 10b5-1 footnotes in the current filing
                is_10b51_plan = False
                if hasattr(obj, 'footnotes') and obj.footnotes:
                    try:
                        footnote_texts = []
                        if isinstance(obj.footnotes, dict):
                            footnote_texts = list(obj.footnotes.values())
                        elif hasattr(obj.footnotes, 'values'):
                            footnote_texts = list(obj.footnotes.values())
                        else:
                            footnote_texts = [str(obj.footnotes)]
                            
                        for text in footnote_texts:
                            if "10b5-1" in str(text).lower() or "10b51" in str(text).lower():
                                is_10b51_plan = True
                                break
                    except Exception:
                        pass
                
                for _, row in df.iterrows():
                    code = row.get('Code', '')
                    # P = Purchase, S = Sale (market transactions)
                    if code in ('P', 'S'):
                        txn_type = 'BUY' if code == 'P' else 'SELL'
                        shares = row.get('Shares', 0)
                        price = row.get('Price', 0)
                        value = row.get('Value', 0)
                        remaining = row.get('Remaining Shares', 'N/A')
                        txn_date = row.get('Date', f.filing_date)
                        
                        shares_num = int(shares) if shares else 0
                        price_num = float(price) if price else 0.0
                        value_num = float(value) if value else (shares_num * price_num)
                        
                        # Relativize position size %-wise
                        pct_str = "N/A"
                        try:
                            if remaining is not None and str(remaining) != 'N/A':
                                rem_shares = float(str(remaining).replace(',', ''))
                                total_shares = shares_num + rem_shares
                                if total_shares > 0:
                                    sold_pct = (shares_num / total_shares) * 100
                                    pct_str = f"{sold_pct:.2f}%"
                        except Exception:
                            pass
                        
                        transactions.append({
                            'date': str(txn_date).split(' ')[0],  # Strip time component
                            'insider': obj.insider_name,
                            'position': obj.position,
                            'type': txn_type,
                            'shares': shares_num,
                            'price': price_num,
                            'value': value_num,
                            'remaining': remaining,
                            'is_10b51': is_10b51_plan,
                            'pct_position': pct_str
                        })
            except Exception:
                continue
        
        if not transactions:
            return ""
        
        # Sort by date descending
        transactions.sort(key=lambda x: x['date'], reverse=True)
        
        # Compute summary stats
        disc_buys_val = sum(t['value'] for t in transactions if t['type'] == 'BUY')
        disc_sells_val = sum(t['value'] for t in transactions if t['type'] == 'SELL' and not t['is_10b51'])
        sched_sells_val = sum(t['value'] for t in transactions if t['type'] == 'SELL' and t['is_10b51'])
        
        buy_count = sum(1 for t in transactions if t['type'] == 'BUY')
        disc_sell_count = sum(1 for t in transactions if t['type'] == 'SELL' and not t['is_10b51'])
        sched_sell_count = sum(1 for t in transactions if t['type'] == 'SELL' and t['is_10b51'])
        
        unique_insiders = set(t['insider'] for t in transactions)
        
        # Format output
        lines = []
        lines.append(f"### INSIDER TRADING ACTIVITY (Form 4 — last 6 months relative to {ref_date})")
        lines.append(f"**Discretionary Activity:** {buy_count} buy(s) totaling **${disc_buys_val:,.0f}** | {disc_sell_count} discretionary sell(s) totaling **${disc_sells_val:,.0f}**")
        lines.append(f"**Pre-Scheduled Activity:** {sched_sell_count} 10b5-1 sell(s) totaling **${sched_sells_val:,.0f}**")
        lines.append(f"Unique active insiders: **{len(unique_insiders)}**")
        lines.append("")
        lines.append("Date | Insider | Position | Action | Shares | Price | Value | % Position | Type")
        lines.append("--- | --- | --- | --- | --- | --- | --- | --- | ---")
        
        for t in transactions:
            type_str = "10b5-1 (Scheduled)" if t['is_10b51'] else "Discretionary (Open Market)"
            lines.append(f"{t['date']} | {t['insider']} | {t['position']} | {t['type']} | {t['shares']:,} | ${t['price']:.2f} | ${t['value']:,.0f} | {t['pct_position']} | {type_str}")
        
        return "\n".join(lines)
    
    except Exception as e:
        print(f"Could not fetch insider activity: {e}")
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

def create_final_analysis(ticker, expert_analysis, missing_analysis, transcript_analysis=None, filing_date=None, transcript_date=None, insider_activity=None):
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
            from datetime import date
            
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

def analyze_ticker(ticker, no_cache=False):
    """Run the full analysis pipeline for a single ticker. Returns (cost, final_analysis_text)."""
    from concurrent.futures import ThreadPoolExecutor
    import time
    
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
    total_cost = 0

    # Run filing, transcript, and insider data in parallel (they're independent)
    with ThreadPoolExecutor(max_workers=3) as executor:
        filing_future = executor.submit(process_filing, ticker, filing, company, no_cache=no_cache)
        transcript_future = executor.submit(process_transcript, ticker, no_cache=no_cache)
        insider_future = executor.submit(get_insider_activity, company, filing.filing_date)

        # Wait for all to complete
        expert_analysis, missing_analysis, filing_cost = filing_future.result()
        transcript_analysis, transcript_cost = transcript_future.result()
        insider_activity = insider_future.result()

    total_cost += filing_cost + transcript_cost
    elapsed = time.time() - start_time
    print(f"\nParallel processing completed in {elapsed:.1f}s")
    
    if insider_activity:
        insider_lines = insider_activity.count('\n')
        print(f"Insider activity: {insider_lines} transactions found")
        # Save raw insider data alongside other filing artifacts
        os.makedirs(os.path.join("data", "filings"), exist_ok=True)
        insider_filename = f"{ticker}-insider-activity.md"
        with open(os.path.join("data", "filings", insider_filename), "w", encoding='utf-8') as f:
            f.write(insider_activity)
        print(f"Insider activity saved to filings/{insider_filename}")
    else:
        print("No insider market transactions found")

    # Final synthesis (transcript is optional)
    final_text = None
    if expert_analysis and missing_analysis:
        # Extract transcript date from the transcript analysis if available
        transcript_date = None
        if transcript_analysis:
            # Try to extract date from first few lines of transcript analysis
            for line in transcript_analysis.split('\n')[:20]:
                if 'call date' in line.lower() or 'date' in line.lower():
                    transcript_date = line.strip()
                    break
        
        final_cost = create_final_analysis(
            ticker, expert_analysis, missing_analysis,
            transcript_analysis=transcript_analysis,
            filing_date=filing.filing_date,
            transcript_date=transcript_date,
            insider_activity=insider_activity
        )
        total_cost += final_cost
        
        # Read back the final report for batch comparison
        final_md_path = os.path.join("data", "intermediate", f"{ticker}_final_report.md")
        if os.path.exists(final_md_path):
            with open(final_md_path, "r", encoding='utf-8') as f:
                final_text = f.read()

    total_elapsed = time.time() - start_time
    print(f"\n--- {ticker} Complete ({total_elapsed:.1f}s) | Cost: ${total_cost:.4f} ---")
    return total_cost, final_text

def main():
    import time
    import sys
    
    no_cache = False
    args = sys.argv[1:]
    
    # Check for cache bypass flags in command-line arguments
    if "--no-cache" in args:
        no_cache = True
        args = [arg for arg in args if arg != "--no-cache"]
    if "--force" in args:
        no_cache = True
        args = [arg for arg in args if arg != "--force"]
        
    if len(args) > 0:
        tickers = [t.strip().upper() for arg in args for t in arg.split(",") if t.strip()]
    else:
        raw = input("Enter ticker(s) (comma-separated, optionally append --no-cache): ")
        # Parse interactive input for flags
        if "--no-cache" in raw:
            no_cache = True
            raw = raw.replace("--no-cache", "")
        if "--force" in raw:
            no_cache = True
            raw = raw.replace("--force", "")
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    
    if not tickers:
        print("No tickers provided.")
        return
    
    os.makedirs("data", exist_ok=True)
    
    start_time = time.time()
    cost_log = {}  # ticker -> cost
    final_reports = {}  # ticker -> final report text
    
    for ticker in tickers:
        try:
            cost, final_text = analyze_ticker(ticker, no_cache=no_cache)
            cost_log[ticker] = cost
            if final_text:
                final_reports[ticker] = final_text
        except Exception as e:
            print(f"\n--- {ticker} FAILED: {e} ---")
            cost_log[ticker] = 0
    
    # Cost summary table
    total_cost = sum(cost_log.values())
    total_elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"  Cost Summary")
    print(f"{'='*60}")
    for t, c in cost_log.items():
        status = "✓" if t in final_reports else "✗"
        print(f"  {status} {t:8s}  ${c:.4f}")
    print(f"  {'─'*24}")
    print(f"  Total:    ${total_cost:.4f}")
    print(f"  Time:     {total_elapsed:.1f}s")
    print(f"{'='*60}")
    
    # Batch comparison (only when multiple tickers with reports)
    if len(final_reports) > 1:
        print(f"\n--- Running Batch Comparison ({len(final_reports)} tickers) ---")
        all_reports_text = "\n\n".join(
            f"--- {t} ---\n{report}" for t, report in final_reports.items()
        )
        comparison_prompt = batch_comparison_prompt_template.format(
            ticker_count=len(final_reports),
            all_reports=all_reports_text
        )
        result = run_analysis(comparison_prompt)
        comparison, comp_cost = result if result else (None, 0)
        
        if comparison:
            intermediate_dir = os.path.join("data", "intermediate")
            os.makedirs(intermediate_dir, exist_ok=True)
            comp_filename = "batch_comparison.md"
            with open(os.path.join(intermediate_dir, comp_filename), "w", encoding='utf-8') as f:
                f.write(comparison)
            print(f"Batch comparison saved to intermediate/{comp_filename}")
            total_cost += comp_cost
            print(f"Comparison cost: ${comp_cost:.4f}")
    
    print(f"\n{'='*60}")
    print(f"  All done — {len(tickers)} ticker(s) | ${total_cost:.4f} | {total_elapsed:.1f}s")
    print(f"{'='*60}")

# Run the main function
if __name__ == "__main__":
    main()
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

def run_analysis(prompt, model="gpt-5.4", reasoning_effort="medium"):
    """Run analysis using OpenAI Responses API with configurable reasoning effort.
    
    GPT-5.4 defaults to 'none' (no reasoning). For financial analysis we use 'medium'
    which enables planning and synthesis reasoning. Options: none, low, medium, high, xhigh.
    """
    if not client:
        print("OpenAI API key not found in .env file.")
        return None
    
    # Count tokens and estimate cost
    prompt_tokens = count_tokens(prompt)
    print(f"Prompt contains approximately {prompt_tokens:,} tokens")
    
    estimated_costs = estimate_cost(prompt_tokens, model)
    print(f"Estimated cost: ${estimated_costs['total_cost']:.4f}")
    print(f"Reasoning effort: {reasoning_effort}")
    
    try:
        response = client.with_options(timeout=300).responses.create(
            model=model,
            instructions="You are a high IQ expert financial engineer.",
            input=prompt,
            reasoning={"effort": reasoning_effort},
        )
        
        # Calculate actual cost from usage
        actual_prompt_tokens = response.usage.input_tokens
        actual_completion_tokens = response.usage.output_tokens
        
        # Log token breakdown (reasoning tokens are billed as output tokens)
        reasoning_tokens = getattr(response.usage.output_tokens_details, 'reasoning_tokens', 0) or 0
        visible_tokens = actual_completion_tokens - reasoning_tokens
        cached_input = getattr(response.usage.input_tokens_details, 'cached_tokens', 0) or 0
        
        actual_costs = calculate_actual_cost(
            actual_prompt_tokens, 
            actual_completion_tokens, 
            model
        )
        
        print(f"Tokens — input: {actual_prompt_tokens:,} (cached: {cached_input:,}) | output: {visible_tokens:,} | reasoning: {reasoning_tokens:,}")
        print(f"Actual cost: ${actual_costs['total_cost']:.4f}")
        
        return response.output_text
        
    except RateLimitError as e:
        print(f"Rate limited: {e}")
        print("Waiting 30s and retrying...")
        import time
        time.sleep(30)
        try:
            response = client.with_options(timeout=300).responses.create(
                model=model,
                instructions="You are a high IQ expert financial engineer.",
                input=prompt,
                reasoning={"effort": reasoning_effort},
            )
            return response.output_text
        except Exception as e2:
            print(f"Retry failed: {e2}")
            return None
    except APITimeoutError as e:
        print(f"Request timed out: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

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

def process_filing(ticker, filing, company=None):
    """Process a single filing and return the expert and missing analyses"""
    print(f"Processing filing: {filing.accession_no} ({filing.form}, {filing.filing_date})")
    
    try:
        # Get filing text as markdown (preserves headings, tables, structure)
        filing_text = filing.markdown()
        print(f"Filing text contains approximately {count_tokens(filing_text):,} tokens")
        
        # Save raw filing text
        os.makedirs(os.path.join("data", "filings"), exist_ok=True)
        raw_filename = f"{ticker}-{filing.form}-{filing.filing_date}-raw.md"
        with open(os.path.join("data", "filings", raw_filename), "w", encoding='utf-8') as f:
            f.write(filing_text)
        print(f"Raw filing saved to {raw_filename}")
        
        # Fetch multi-year trends to augment the prompt
        trend_context = ""
        if company:
            print("Fetching multi-year financial trends from XBRL...")
            trend_context = get_multi_year_trends(company)
            if trend_context:
                print(f"Added {count_tokens(trend_context):,} tokens of historical context")
                # Save trend context for reference
                trend_filename = f"{ticker}-{filing.form}-{filing.filing_date}-xbrl-trends.md"
                with open(os.path.join("data", "filings", trend_filename), "w", encoding='utf-8') as f:
                    f.write(trend_context)
                print(f"XBRL trends saved to {trend_filename}")
            else:
                print("No historical trend data available")
        
        # Step 1: Expert Analysis
        print("\n--- Running Expert Analysis ---")
        expert_prompt = expert_analysis_prompt_template.format(
            filing_text=filing_text + trend_context
        )
        expert_analysis = run_analysis(expert_prompt)
        
        if not expert_analysis:
            return None, None
            
        # Save expert analysis
        os.makedirs(os.path.join("data", "filings"), exist_ok=True)
        expert_filename = f"{ticker}-{filing.form}-{filing.filing_date}-expert-analysis.md"
        with open(os.path.join("data", "filings", expert_filename), "w", encoding='utf-8') as f:
            f.write(expert_analysis)
        print(f"Expert analysis saved to {expert_filename}")
        
        # Step 2: Missing Analysis
        print("\n--- Running Missing Analysis ---")
        missing_prompt = missing_analysis_prompt_template.format(
            expert_analysis=expert_analysis,
            filing_text=filing_text
        )
        missing_analysis = run_analysis(missing_prompt)
        
        if missing_analysis:
            # Save missing analysis
            missing_filename = f"{ticker}-{filing.form}-{filing.filing_date}-missing-analysis.md"
            with open(os.path.join("data", "filings", missing_filename), "w", encoding='utf-8') as f:
                f.write(missing_analysis)
            print(f"Missing analysis saved to {missing_filename}")
            
        return expert_analysis, missing_analysis
        
    except Exception as e:
        print(f"Could not process filing {filing.accession_no}: {e}")
        return None, None

def process_transcript(ticker):
    """Process transcript and return the analysis"""
    print("\n--- Fetching Transcript ---")
    
    html = get_transcript(ticker)
    
    if not html:
        print("No transcript found")
        return None
        
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
    transcript_analysis = run_analysis(transcript_prompt)
    
    if transcript_analysis:
        # Save transcript analysis
        transcript_analysis_filename = f"{ticker}-transcript-analysis.md"
        with open(os.path.join("data", "transcripts", transcript_analysis_filename), "w", encoding='utf-8') as f:
            f.write(transcript_analysis)
        print(f"Transcript analysis saved to {transcript_analysis_filename}")
        
    return transcript_analysis

def create_final_analysis(ticker, expert_analysis, missing_analysis, transcript_analysis):
    """Create the final analysis from all components"""
    if not (expert_analysis and missing_analysis and transcript_analysis):
        print("Missing required analyses for final report")
        return
        
    print("\n--- Running Final Analysis ---")
    
    # Format the final prompt with the three analyses
    final_prompt = final_juice_prompt_template.format(
        expert_analysis=expert_analysis,
        missing_analysis=missing_analysis,
        transcript_analysis=transcript_analysis
    )
    
    final_analysis = run_analysis(final_prompt)
    
    if final_analysis:
        # Save markdown version
        md_filename = f"{ticker}_final_report.md"
        md_path = os.path.join("data", md_filename)
        with open(md_path, "w", encoding='utf-8') as f:
            f.write(final_analysis)
        print(f"Final analysis saved to {md_filename}")
        
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
            
            # Save HTML version (looks great in any browser)
            html_filename = f"{ticker}_final_report.html"
            html_path = os.path.join("data", html_filename)
            with open(html_path, "w", encoding='utf-8') as f:
                f.write(full_html)
            print(f"HTML version saved to {html_filename}")
            
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
                
                pdf_filename = f"{ticker}_final_report.pdf"
                pdf_path = os.path.join("data", pdf_filename)
                with open(pdf_path, "wb") as f:
                    f.write(base64.b64decode(pdf_data["data"]))
                print(f"PDF version saved to {pdf_filename}")
                
            except Exception as pdf_err:
                print(f"PDF generation failed (HTML still saved): {pdf_err}")
            
        except Exception as e:
            print(f"Error creating report: {e}")

def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    # Main execution
    ticker = input("Enter ticker symbol: ").upper()
    company = Company(ticker)
    filing = company.get_filings(form='10-Q').latest()  # Get latest 10-Q filing
    print(filing)
    # Create data directory
    os.makedirs("data", exist_ok=True)

    start_time = time.time()

    # Run filing and transcript branches in parallel (they're independent)
    with ThreadPoolExecutor(max_workers=2) as executor:
        filing_future = executor.submit(process_filing, ticker, filing, company)
        transcript_future = executor.submit(process_transcript, ticker)

        # Wait for both to complete
        expert_analysis, missing_analysis = filing_future.result()
        transcript_analysis = transcript_future.result()

    elapsed = time.time() - start_time
    print(f"\nParallel processing completed in {elapsed:.1f}s")

    # Step 3: Create final analysis after both complete
    if expert_analysis and missing_analysis and transcript_analysis:
        create_final_analysis(ticker, expert_analysis, missing_analysis, transcript_analysis)

    total_elapsed = time.time() - start_time
    print(f"\n--- Analysis Complete ({total_elapsed:.1f}s total) ---")

# Run the main function
if __name__ == "__main__":
    main()
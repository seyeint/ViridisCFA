import os
import asyncio
from dotenv import load_dotenv
from edgar import *
from openai import AsyncOpenAI, RateLimitError, APITimeoutError
from prompt_configs import *
from utils import count_tokens, estimate_cost, calculate_actual_cost
from fetch_transcripts import get_transcript, html_to_markdown
import markdown
from fpdf import FPDF
# Load .env file
load_dotenv()
edgar_identity = os.getenv("EDGAR_IDENTITY")
set_identity(edgar_identity)
print(f"Using EDGAR Identity: {edgar_identity}")

# Load OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize AsyncOpenAI client
client = AsyncOpenAI(api_key=openai_key) if openai_key else None

async def run_analysis(prompt, model="o3-mini", use_flex=True):
    """Run analysis using OpenAI API with optional flex processing"""
    if not client:
        print("OpenAI API key not found in .env file.")
        return None
    
    # Count tokens and estimate cost
    prompt_tokens = count_tokens(prompt)
    print(f"Prompt contains approximately {prompt_tokens:,} tokens")
    
    estimated_costs = estimate_cost(prompt_tokens, model)
    print(f"Estimated cost: ${estimated_costs['total_cost']:.4f}")
    
    while True:
        try:
            # Set up API call with or without flex based on current state
            if use_flex:
                print("Using flex processing (lower cost, may be slower)")
                response = await client.with_options(timeout=300).chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a high IQ expert financial engineer."},
                        {"role": "user", "content": prompt}
                    ],
                    service_tier="flex"
                )
            else:
                print("Using standard processing")
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a high IQ expert financial engineer."},
                        {"role": "user", "content": prompt}
                    ]
                )
            
            # Calculate actual cost
            actual_prompt_tokens = response.usage.prompt_tokens
            actual_completion_tokens = response.usage.completion_tokens
            
            actual_costs = calculate_actual_cost(
                actual_prompt_tokens, 
                actual_completion_tokens, 
                model
            )
            
            print(f"Actual cost: ${actual_costs['total_cost']:.4f}")
            
            return response.choices[0].message.content
            
        except (RateLimitError, APITimeoutError) as e:
            if use_flex:
                print("\nFlex processing unavailable or timed out.")
                choice = input("Switch to standard processing? (y/n): ").lower()
                if choice == 'y':
                    use_flex = False
                    continue
                else:
                    print("Aborting operation.")
                    return None
            else:
                print(f"API error: {e}")
                return None
        except Exception as e:
            if use_flex and "Invalid service_tier" in str(e):
                print(f"\nFlex processing not supported for {model}. Switching to standard processing.")
                use_flex = False
                continue
            else:
                print(f"Error: {e}")
                return None

async def process_filing(ticker, filing):
    """Process a single filing and return the expert and missing analyses"""
    print(f"Processing filing: {filing.accession_no} ({filing.form}, {filing.filing_date})")
    
    try:
        # Get filing text
        filing_text = filing.text()
        print(f"Filing text contains approximately {count_tokens(filing_text):,} tokens")
        
        # Step 1: Expert Analysis
        print("\n--- Running Expert Analysis ---")
        expert_prompt = expert_analysis_prompt_template.format(filing_text=filing_text)
        expert_analysis = await run_analysis(expert_prompt)
        
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
        missing_analysis = await run_analysis(missing_prompt)
        
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

async def process_transcript(ticker):
    """Process transcript and return the analysis"""
    print("\n--- Fetching Transcript ---")
    
    # Direct call without to_thread
    html = get_transcript(ticker)
    
    if not html:
        print("No transcript found")
        return None
        
    # Save HTML transcript
    os.makedirs(os.path.join("data", "transcripts"), exist_ok=True)
    html_path = os.path.join("data", "transcripts", f"{ticker}_transcript.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    # Direct call without to_thread
    transcript_text = html_to_markdown(html)
    md_path = os.path.join("data", "transcripts", f"{ticker}_transcript.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    print(f"Transcript saved to {md_path}")
    
    # Run transcript analysis
    print("\n--- Running Transcript Analysis ---")
    transcript_prompt = transcript_prompt_template.format(transcript_text=transcript_text)
    transcript_analysis = await run_analysis(transcript_prompt)
    
    if transcript_analysis:
        # Save transcript analysis
        transcript_analysis_filename = f"{ticker}-transcript-analysis.md"
        with open(os.path.join("data", "transcripts", transcript_analysis_filename), "w", encoding='utf-8') as f:
            f.write(transcript_analysis)
        print(f"Transcript analysis saved to {transcript_analysis_filename}")
        
    return transcript_analysis

async def create_final_analysis(ticker, expert_analysis, missing_analysis, transcript_analysis):
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
    
    final_analysis = await run_analysis(final_prompt)
    
    if final_analysis:
        # Save markdown version
        md_filename = f"{ticker}_final_report.md"
        md_path = os.path.join("data", md_filename)
        with open(md_path, "w", encoding='utf-8') as f:
            f.write(final_analysis)
        print(f"Final analysis saved to {md_filename}")
        
        # Convert to PDF
        try:
            # First convert markdown to HTML
            html_content = markdown.markdown(final_analysis, extensions=['extra'])
            
            # Create a nice title HTML
            title_html = f'<h1 style="font-size: 16pt; text-align: center;">{ticker} Investment Analysis Report</h1><br/>'
            full_html = title_html + html_content
            
            # Save HTML version
            html_filename = f"{ticker}_final_report.html" 
            html_path = os.path.join("data", html_filename)
            with open(html_path, "w", encoding='utf-8') as f:
                f.write(full_html)
            print(f"HTML version saved to {html_filename}")
            
            # Create PDF using fpdf2
            pdf_filename = f"{ticker}_final_report.pdf"
            pdf_path = os.path.join("data", pdf_filename)
            
            # Get font paths
            fonts_dir = "fonts"
            regular_font = os.path.join(fonts_dir, "Nunito-Light.ttf")
            bold_font = os.path.join(fonts_dir, "Nunito-ExtraBold.ttf")
            
            pdf = FPDF()
            pdf.add_page()
            
            # Add Nunito fonts
            pdf.add_font("Nunito", "", regular_font)
            pdf.add_font("Nunito", "B", bold_font)
            pdf.set_font("Nunito", size=11)
            
            # Write HTML content
            pdf.write_html(full_html)
            
            # Save the PDF
            pdf.output(pdf_path)
            print(f"PDF version saved to {pdf_filename}")
            
        except Exception as e:
            print(f"Error creating PDF: {e}")

async def main():
    # Main execution
    ticker = input("Enter ticker symbol: ").upper()
    company = Company(ticker)
    filing = company.get_filings(form=[ '10-Q']).head(1)[0]  # Get just one filing '10-K',
    print(filing)
    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Use gather to run both tasks truly in parallel
    (expert_analysis, missing_analysis), transcript_analysis = await asyncio.gather(
        process_filing(ticker, filing),
        process_transcript(ticker)
    )

    # Create final analysis after parallel tasks complete
    if expert_analysis and missing_analysis and transcript_analysis:
        await create_final_analysis(ticker, expert_analysis, missing_analysis, transcript_analysis)

    print("\n--- Analysis Complete ---")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
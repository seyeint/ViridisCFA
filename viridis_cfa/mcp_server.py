import sys
import os
import contextlib
from mcp.server.fastmcp import FastMCP

# Ensure the package directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from viridis_cfa.pipeline import analyze_ticker as pipeline_analyze_ticker, run_analysis
from viridis_cfa.cache import list_all_manifests
from prompt_configs import batch_comparison_prompt_template

# Initialize FastMCP Server
mcp = FastMCP("ViridisCFA")

@contextlib.contextmanager
def redirect_stdout_to_stderr():
    """Redirect stdout to stderr so print calls do not corrupt stdio JSON-RPC channel."""
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout

@mcp.tool()
def analyze_ticker(ticker: str, no_cache: bool = False) -> str:
    """Run the full ViridisCFA financial research pipeline for a single ticker.
    This includes quantitative scorecards, filing analysis, earnings call transcript
    scraping/analysis, and insider trading patterns. Returns a synthesized markdown report.
    
    Uses ingredient-based caching: only LLM steps whose data inputs have changed
    since the last run are re-executed. Pass no_cache=True to force a full refresh.
    """
    with redirect_stdout_to_stderr():
        ticker = ticker.strip().upper()
        print(f"MCP Tool call: analyze_ticker('{ticker}', no_cache={no_cache})", file=sys.stderr)
        try:
            cost, final_text = pipeline_analyze_ticker(ticker, no_cache=no_cache)
            if not final_text:
                return f"Error: Could not retrieve or analyze filing for {ticker}."
            return final_text
        except Exception as e:
            return f"Error running analysis for {ticker}: {str(e)}"

@mcp.tool()
def compare_tickers(tickers: list[str], no_cache: bool = False) -> str:
    """Compare multiple tickers using the ViridisCFA pipeline.
    Runs parallel analyses and returns a comparative batch ranking markdown report.
    """
    with redirect_stdout_to_stderr():
        # Sanitize ticker inputs
        clean_tickers = [t.strip().upper() for t in tickers if t.strip()]
        print(f"MCP Tool call: compare_tickers({clean_tickers}, no_cache={no_cache})", file=sys.stderr)
        
        if not clean_tickers:
            return "Error: No valid tickers provided."
        
        final_reports = {}
        for ticker in clean_tickers:
            try:
                cost, final_text = pipeline_analyze_ticker(ticker, no_cache=no_cache)
                if final_text:
                    final_reports[ticker] = final_text
            except Exception as e:
                print(f"Failed to analyze {ticker} during batch comparison: {e}", file=sys.stderr)
        
        if not final_reports:
            return "Error: Failed to analyze any of the provided tickers."
        
        if len(final_reports) == 1:
            ticker = list(final_reports.keys())[0]
            return f"# Comparison Summary\n\nOnly one ticker ({ticker}) was successfully analyzed.\n\n{final_reports[ticker]}"
            
        print(f"Running Batch Comparison ({len(final_reports)} tickers)...", file=sys.stderr)
        all_reports_text = "\n\n".join(
            f"--- {t} ---\n{report}" for t, report in final_reports.items()
        )
        comparison_prompt = batch_comparison_prompt_template.format(
            ticker_count=len(final_reports),
            all_reports=all_reports_text
        )
        
        result = run_analysis(comparison_prompt)
        comparison, comp_cost = result if result else (None, 0)
        
        if not comparison:
            return "Error: Failed to generate comparative analysis from the reports."
            
        # Save comparison intermediate
        intermediate_dir = os.path.join("data", "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        comp_filename = "batch_comparison.md"
        with open(os.path.join(intermediate_dir, comp_filename), "w", encoding='utf-8') as f:
            f.write(comparison)
        print(f"Batch comparison saved to intermediate/{comp_filename}", file=sys.stderr)
        
        return comparison

@mcp.tool()
def list_past_analyses() -> str:
    """List all previously completed ViridisCFA analyses with their dates,
    filing types, and cost history. Use this when the user asks about
    their past reports or wants to see what analyses they've already run."""
    with redirect_stdout_to_stderr():
        print("MCP Tool call: list_past_analyses()", file=sys.stderr)
        try:
            analyses = list_all_manifests()
            
            if not analyses:
                return "No past analyses found. Run `analyze_ticker` to create your first report."
            
            lines = []
            lines.append(f"# Past Analyses ({len(analyses)} reports)\n")
            lines.append("| # | Ticker | Filing | Filed | Last Analyzed | Last Cost | Status |")
            lines.append("|---|--------|--------|-------|---------------|-----------|--------|")
            
            for i, a in enumerate(analyses, 1):
                status = "Cached ✓" if a['has_manifest'] else "Pre-cache"
                last_cost = f"${a['last_cost']:.4f}" if a['last_cost'] else "—"
                analyzed = a['analyzed_at'][:10] if a['analyzed_at'] != 'N/A' else 'N/A'
                lines.append(
                    f"| {i} | **{a['ticker']}** | {a['filing_form']} | "
                    f"{a['filing_date']} | {analyzed} | {last_cost} | {status} |"
                )
            
            lines.append(f"\nUse `retrieve_past_analysis(ticker)` to get the full report for any ticker.")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing past analyses: {str(e)}"

@mcp.tool()
def retrieve_past_analysis(ticker: str) -> str:
    """Retrieve the full markdown report for a previously analyzed ticker.
    Returns the complete investment analysis instantly from cache.
    Use list_past_analyses() first to see what's available."""
    with redirect_stdout_to_stderr():
        ticker = ticker.strip().upper()
        print(f"MCP Tool call: retrieve_past_analysis('{ticker}')", file=sys.stderr)
        try:
            # Try the standard path
            report_path = os.path.join("data", "intermediate", f"{ticker}_final_report.md")
            
            if not os.path.exists(report_path):
                return (f"No report found for {ticker}. "
                        f"Use `analyze_ticker('{ticker}')` to generate one, "
                        f"or `list_past_analyses()` to see available reports.")
            
            with open(report_path, "r", encoding="utf-8") as f:
                report = f.read()
            
            return report
        except Exception as e:
            return f"Error retrieving report for {ticker}: {str(e)}"

if __name__ == "__main__":
    # Start the FastMCP stdio server
    mcp.run("stdio")

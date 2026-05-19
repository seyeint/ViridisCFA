import sys
import os
import contextlib
from mcp.server.fastmcp import FastMCP

# Ensure the package directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from viridis_cfa.pipeline import analyze_ticker, run_analysis
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
    """
    with redirect_stdout_to_stderr():
        ticker = ticker.strip().upper()
        print(f"MCP Tool call: analyze_ticker('{ticker}', no_cache={no_cache})", file=sys.stderr)
        try:
            cost, final_text = analyze_ticker(ticker, no_cache=no_cache)
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
                cost, final_text = analyze_ticker(ticker, no_cache=no_cache)
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

if __name__ == "__main__":
    # Start the FastMCP stdio server
    mcp.run("stdio")

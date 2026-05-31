import os
import sys
import time

# Local package imports
from prompt_configs import batch_comparison_prompt_template
from viridis_cfa.pipeline import analyze_ticker, run_analysis
from viridis_cfa.report_renderer import write_markdown_document_artifact

def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except AttributeError:
            pass

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
    cache_hits = 0
    for t, c in cost_log.items():
        status = "✓" if t in final_reports else "✗"
        cache_note = " (cached)" if c == 0 and t in final_reports else ""
        print(f"  {status} {t:8s}  ${c:.4f}{cache_note}")
        if c == 0 and t in final_reports:
            cache_hits += 1
    print(f"  {'─'*30}")
    print(f"  Total:    ${total_cost:.4f}")
    print(f"  Time:     {total_elapsed:.1f}s")
    if cache_hits > 0:
        print(f"  Cache:    {cache_hits}/{len(cost_log)} ticker(s) served from cache")
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
            comp_path = os.path.join(intermediate_dir, comp_filename)
            with open(comp_path, "w", encoding='utf-8') as f:
                f.write(comparison)
            print(f"Batch comparison saved to intermediate/{comp_filename}")
            comp_html_path = os.path.join(intermediate_dir, "batch_comparison.html")
            write_markdown_document_artifact(
                "Batch Comparison",
                comparison,
                {'final_report_md': comp_path, 'final_report_html': comp_html_path},
                comp_html_path,
            )
            print("Batch comparison HTML saved to intermediate/batch_comparison.html")
            total_cost += comp_cost
            print(f"Comparison cost: ${comp_cost:.4f}")
    
    print(f"\n{'='*60}")
    print(f"  All done — {len(tickers)} ticker(s) | ${total_cost:.4f} | {total_elapsed:.1f}s")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

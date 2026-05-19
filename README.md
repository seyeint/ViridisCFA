# ViridisCFA — AI-Powered Equity Research Pipeline

An automated equity research pipeline that pulls SEC filings and earnings call transcripts, runs a local deterministic quantitative financial scorecard, crawls insider Form 4 activity (discriminating pre-scheduled 10b5-1 plans), runs multi-stage AI analysis using GPT-5.4 with flex processing (~50% cheaper), and produces synthesized investment reports with bull/bear cases.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│               User enters ticker(s)                     │
│            (comma-separated, e.g. AAPL, SNAP)           │
│           (optionally appends --no-cache)               │
└─────────────────────┬───────────────────────────────────┘
                      │  (per ticker)
          ┌───────────┼───────────────┐
          ▼           ▼               ▼
   ┌──────────────┐ ┌───────────────┐ ┌───────────────┐
   │ Branch 1:    │ │ Branch 2:     │ │ Branch 3:     │
   │ SEC Filing   │ │ Transcript    │ │ Insider Trades│
   │              │ │               │ │               │
   │ 1. Latest    │ │ 1. Scrape     │ │ 1. Fetch      │
   │    10-K/10-Q │ │    roic.ai    │ │    Form 4s    │
   │              │ │    (optional) │ │    (6 months) │
   │ 2. Local     │ │               │ │               │
   │    Scorecard │ │ 2. GPT-5.4   │ │ 2. Footnotes  │
   │    (Altman Z'│ │    analyze    │ │    Audit      │
   │    Piotroski │ │    transcript │ │    (Tag       │
   │    Beneish M)│ │               │ │    10b5-1)    │
   │              │ └───────┬───────┘ └───────┬───────┘
   │ 3. Expert    │         │                 │
   │    Analysis  │         │                 │
   │    (reasoning│         │                 │
   │    high)     │         │                 │
   │              │         │                 │
   │ 4. Missing   │         │                 │
   │    Analysis  │         │                 │
   └──────┬───────┘         │                 │
          │                 │                 │
          └────────┬────────┴─────────────────┘
                   ▼
          ┌─────────────────────┐
          │ GPT-5.4 Final       │
          │ Synthesis + Bull/   │
          │ Bear Case           │
          └─────────┬───────────┘
                    ▼
          ┌─────────────────────┐
          │ PDF Report          │
          └─────────┬───────────┘
                    ▼  (if multi-ticker)
          ┌─────────────────────┐
          │ Batch Comparison    │
          │ & Ranking           │
          └─────────────────────┘
```

All three branches run **in parallel** using threads. The local quantitative scorecard values are injected directly into the LLM synthesis prompt with strict verification rules to prevent hallucinations.

## Setup

Requires **Python 3.10+** and **Google Chrome**.

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
EDGAR_IDENTITY=yourname@yourcompany.com
```

- **OPENAI_API_KEY**: Your OpenAI API key (needs access to `gpt-5.4`)
- **EDGAR_IDENTITY**: Required by [SEC fair access policy](https://www.sec.gov/os/webmaster-faq#developers). **Must be a professional/institutional email** (e.g. `name@company.com`). Consumer email providers (Gmail, Yahoo, Hotmail) will be rate-limited or blocked by SEC servers.

### 4. Chrome/ChromeDriver

Selenium requires Chrome/ChromeDriver for transcript scraping. On most systems, `selenium` auto-manages this. Just run it.

## Usage

```bash
source venv/bin/activate
# Standard run
python main.py

# CLI Arguments (comma-separated or multiple arguments)
python main.py AAPL
python main.py TSLA,TTD

# Bypass caching (forcing fresh calculation & LLM generation)
python main.py TSLA --no-cache
python main.py TSLA --force
```

When run interactively (with no CLI arguments), you can also append `--no-cache` or `--force` directly to the prompt:
```
Enter ticker(s) (comma-separated, optionally append --no-cache): AAPL --no-cache
```

Single ticker → one PDF report with investment conclusion (bull/bear case).

Multiple tickers → one PDF per ticker + a batch comparison ranking saved to `data/intermediate/batch_comparison.md`.

### Output

Reports are saved to `data/`:

```
data/
├── {TICKER}_final_report.pdf                      # ← Final PDF reports (top level)
├── filings/
│   ├── {TICKER}-10-K-{DATE}-raw.md                # Raw SEC filing
│   ├── {TICKER}-10-K-{DATE}-xbrl-trends.md        # 4-year financial trends
│   ├── {TICKER}-10-K-{DATE}-expert-analysis.md    # Expert analysis
│   ├── {TICKER}-10-K-{DATE}-missing-analysis.md   # Missing info check
│   └── {TICKER}-insider-activity.md               # Form 4 insider buys/sells
├── transcripts/
│   ├── {TICKER}_transcript.md                     # Transcript markdown
│   └── {TICKER}-transcript-analysis.md            # Transcript analysis
└── intermediate/
    ├── {TICKER}_final_report.md                   # Final report (markdown source)
    ├── {TICKER}_final_report.html                 # Styled HTML (Chrome input)
    └── batch_comparison.md                        # Multi-ticker ranking
```

## Architecture

### Data Sources

| Source | Library | Data Retrieved |
|--------|---------|---------------|
| SEC EDGAR | `edgartools` v5.30.0 | Latest 10-K or 10-Q (auto, skips amendments) |
| SEC XBRL | `edgartools` EntityFacts | 4-year income statement + balance sheet |
| SEC Form 4 | `edgartools` | Insider buys/sells (6-month window around filing) |
| roic.ai | `selenium` + stealth | Latest earnings call transcript (optional) |

### Analysis Pipeline

| Stage | Model | Reasoning | Purpose |
|-------|-------|-----------|---------|
| Expert Analysis | GPT-5.4 | **High** | Deep analysis of filing + historical trends |
| Missing Analysis | GPT-5.4 | Medium | Identify gaps in expert analysis vs raw filing |
| Transcript Analysis | GPT-5.4 | Medium | Extract insights from earnings call |
| Final Synthesis | GPT-5.4 | Medium | Merge all analyses + insider signals, produce bull/bear case |
| Batch Comparison | GPT-5.4 | Medium | Rank and compare tickers (multi-ticker only) |

All calls use **flex processing** by default (~50% off standard pricing). If flex capacity is unavailable, requests automatically fall back to standard.

### Key Design Decisions

- **`filing.markdown()` over `filing.text()`**: Preserves table structure, headings, and section hierarchy for better LLM comprehension
- **Local Quantitative Scorecard**: Programmatically calculates Altman Z'-Score (Z-Prime), Piotroski F-Score, Beneish M-Score, and critical liquidity/leverage ratios. These ground-truth values are injected into the final synthesis prompt with verification rules to prevent LLM mathematical hallucinations.
- **Insider 10b5-1 Discrimination**: Form 4 trades are audited using footnote scans to distinguish discretionary trades from pre-scheduled 10b5-1 executions, isolating higher-conviction signals.
- **XBRL multi-year trends**: Injects 4-year financial history the LLM couldn't see from a single filing
- **Insider trading signals**: Form 4 buys/sells are injected as raw data into the final synthesis — no extra LLM call, the model cross-references insider behavior with filing timeline
- **Flex processing**: Batch API rates (~50% off) with automatic fallback to standard on 429
- **`reasoning: high` for expert**: The main filing analysis gets deeper reasoning; other stages use medium
- **Optional transcripts**: Reports are generated even when no earnings call is available
- **Date-aware synthesis**: Filing and transcript dates are passed to the final prompt; mismatches are flagged
- **Sync client + ThreadPoolExecutor**: Simpler than async, works with Selenium, achieves real parallelism for I/O-bound work

## Cost

Typical cost per ticker (GPT-5.4, flex pricing):

| Filing Size | Estimated Total |
|-------------|----------------|
| Small (~20K tokens) | ~$0.20 |
| Medium (~85K tokens) | ~$0.50 |
| Large (~230K tokens) | ~$1.00 |

Cost summary is printed at the end of each run.

## Project Structure

```
├── main.py               # Pipeline orchestrator
├── quant_engine.py       # Deterministic quantitative financial scorecard
├── prompt_configs.py     # LLM prompt templates
├── utils.py              # Token counting + cost estimation
├── fetch_transcripts.py  # Selenium-based transcript scraper
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not committed)
└── .gitignore
```

## License

Private research tool.

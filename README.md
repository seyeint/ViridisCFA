# ViridisCFA — AI-Powered Equity Research Pipeline

An automated equity research pipeline that pulls SEC filings and earnings call transcripts, runs multi-stage AI analysis using GPT-5.4, and produces a synthesized investment report.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    User enters ticker                    │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   ┌──────────────┐      ┌───────────────┐
   │ Branch 1:    │      │ Branch 2:     │
   │ SEC Filing   │      │ Transcript    │
   │              │      │               │
   │ 1. Fetch     │      │ 1. Scrape     │
   │    latest    │      │    roic.ai    │
   │    10-Q      │      │               │
   │              │      │ 2. GPT-5.4    │
   │ 2. XBRL      │      │    analyze    │
   │    4-year    │      │    transcript │
   │    trends    │      │               │
   │              │      └───────┬───────┘
   │ 3. GPT-5.4   │              │
   │    Expert    │              │
   │    Analysis  │              │
   │              │              │
   │ 4. GPT-5.4   │              │
   │    Missing   │              │
   │    Analysis  │              │
   └──────┬───────┘              │
          │                      │
          └──────────┬───────────┘
                     ▼
          ┌─────────────────────┐
          │ GPT-5.4 Final       │
          │ Synthesis Report    │
          └─────────┬───────────┘
                    ▼
          ┌─────────────────────┐
          │ Output:             │
          │ • Markdown report   │
          │ • Styled HTML       │
          │ • PDF               │
          └─────────────────────┘
```

Both branches run **in parallel** using threads — the transcript analysis completes while the filing analysis is still processing.

## Setup

### 1. Clone and create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
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

Selenium requires Chrome and ChromeDriver for transcript scraping. On most systems, `selenium` auto-manages this via `selenium-manager`.

## Usage

```bash
source venv/bin/activate
python main.py
```

Enter a ticker symbol when prompted. The pipeline will:

1. Fetch the latest 10-Q filing from SEC EDGAR (as structured markdown)
2. Pull 4-year historical financials from XBRL
3. Scrape the latest earnings call transcript from roic.ai
4. Run 4 GPT-5.4 analysis stages with `medium` reasoning effort
5. Generate a final report in Markdown, HTML, and PDF

### Output

Reports are saved to `data/`:

```
data/
├── filings/
│   ├── {TICKER}-10-Q-{DATE}-raw.md              # Raw SEC filing
│   ├── {TICKER}-10-Q-{DATE}-xbrl-trends.md      # 4-year financial trends
│   ├── {TICKER}-10-Q-{DATE}-expert-analysis.md   # Expert analysis
│   └── {TICKER}-10-Q-{DATE}-missing-analysis.md  # Missing info check
├── transcripts/
│   ├── {TICKER}_transcript.html                  # Raw transcript HTML
│   ├── {TICKER}_transcript.md                    # Transcript markdown
│   └── {TICKER}-transcript-analysis.md           # Transcript analysis
├── {TICKER}_final_report.md                      # Final synthesized report
├── {TICKER}_final_report.html                    # Styled HTML report
└── {TICKER}_final_report.pdf                     # PDF report
```

## Architecture

### Data Sources

| Source | Library | Data Retrieved |
|--------|---------|---------------|
| SEC EDGAR | `edgartools` v5.30.0 | Latest 10-Q filing as markdown |
| SEC XBRL | `edgartools` EntityFacts | 4-year income statement + balance sheet |
| roic.ai | `selenium` + stealth | Latest earnings call transcript |

### Analysis Pipeline

| Stage | Model | Reasoning | Purpose |
|-------|-------|-----------|---------|
| Expert Analysis | GPT-5.4 | Medium | Deep analysis of filing + historical trends |
| Missing Analysis | GPT-5.4 | Medium | Identify gaps in expert analysis vs raw filing |
| Transcript Analysis | GPT-5.4 | Medium | Extract insights from earnings call |
| Final Synthesis | GPT-5.4 | Medium | Merge all analyses, filter neutral information |

### Key Design Decisions

- **`filing.markdown()` over `filing.text()`**: Preserves table structure, headings, and section hierarchy for better LLM comprehension (+10% tokens, significantly better parsing)
- **XBRL multi-year trends**: Injects 4-year financial history the LLM couldn't see from a single quarterly filing
- **OpenAI Responses API**: Better prompt caching and reasoning support vs Chat Completions
- **`reasoning: medium`**: GPT-5.4 defaults to `none` (no chain-of-thought). Medium enables synthesis/planning reasoning, critical for financial analysis
- **Sync client + ThreadPoolExecutor**: Simpler than async, works with Selenium (which blocks the event loop in async), achieves real parallelism for I/O-bound work

## Cost

Typical cost per ticker (GPT-5.4, medium reasoning):

| Filing Size | Estimated Total |
|-------------|----------------|
| Small (~20K tokens) | ~$0.35 |
| Large (~85K tokens) | ~$0.75 |

Prompt caching can reduce repeat-run costs significantly (90% discount on cached input tokens).

## Project Structure

```
├── main.py               # Pipeline orchestrator
├── prompt_configs.py     # LLM prompt templates
├── utils.py              # Token counting + cost estimation
├── fetch_transcripts.py  # Selenium-based transcript scraper
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not committed)
└── .gitignore
```

## License

Private research tool.

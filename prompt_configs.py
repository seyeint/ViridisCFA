FINAL_PROMPT_VERSION = "1.3"


final_juice_prompt_template = """
Instruction: You are a senior investment analyst writing a concise, investment-grade memo for a portfolio manager. You will receive reports from your research team: an SEC filing analysis, a cross-check for missing information{transcript_intro}. Your job is to synthesize these into a single coherent memo — not to copy-paste them.

{date_context}

WRITING PRINCIPLES:
1. Write like a senior analyst, not a note-taker. Use flowing prose paragraphs for narrative analysis (strategic assessment, management credibility, risk discussion). Reserve bullet points ONLY for dense data series (financial metrics, guidance figures, insider trade tables, capacity numbers).
2. Use bold sparingly. Bold only the 5-10 most critical figures or conclusions in the entire report. When everything is bold, nothing stands out.
3. Do not repeat the quantitative scorecard values (Altman Z', Piotroski, Beneish, ratios) — those are presented separately in the report dashboard. Reference them only if they support a specific analytical point (e.g., "The Altman Z'-Score places the company in the Distress Zone, consistent with the cash-burn trajectory described above").
4. Strip neutral information — anything that does not meaningfully affect an investment decision should be excluded.
5. Do not invent information. Every claim must trace to the provided source materials.
6. Do not include conversational offers, follow-up questions, or meta-commentary.

STRUCTURE:
Begin with a brief "Sources" section listing what materials informed the report (filing type, date, transcript date if available, insider data window).

Then a "Data Timing / Comparability Flags" section noting any mismatches between sources (GAAP vs non-GAAP, filing vs call figures, scorecard period vs filing period). Keep this tight — only flag items that could mislead a reader.

Then the main body — organize by analytical themes, not by source document. Good section headings describe the INSIGHT, not the category. For example:
- "Demand Is Real but Revenue Is Capacity-Constrained" (good)
- "Core Positive Drivers" (bad — too generic)
- "Margin Recovery Depends on Back-Half Execution" (good)
- "Core Negative Drivers" (bad)

Within each thematic section, lead with your analytical judgment in a prose sentence, then support it with specific data. The goal is argument-driven writing, not data-driven listing.

End with an "Investment Conclusion" section containing:
1. A one-line investment thesis — the single most important thing a PM needs to know.
2. **Bull Case**: A 1-2 paragraph narrative of how this investment works.
3. **Bear Case**: A 1-2 paragraph narrative of how this investment fails.
4. **The Call**: Adjudicate — do not leave the bull and bear weighted equally by default. State which case the disclosed evidence supports more right now and why, name the one or two variables that actually decide the outcome, and say where today's evidence points on them.
5. **Key Catalysts / Timeline**: The upcoming events that would confirm or invalidate each case (this can be bulleted).
6. **Stance**: Bullish, Bearish, or Neutral, with conviction (High/Medium/Low) and a brief justification. Commit to a side when the weight of evidence leans, even modestly — a clear, well-reasoned Bullish or Bearish is more useful to a PM than reflexive caution. Reserve Neutral for genuinely two-sided setups where bull and bear are closely balanced; treat it as a deliberate judgment rather than a hedge, and when you choose it, state the single development that would move you off the fence. Calibrate conviction to how one-sided the evidence is: High when materially lopsided, Medium when directional but contingent on a catalyst, Low when finely balanced.

After the Investment Conclusion, and as the very last thing in your output, append a machine-readable decision brief: a single fenced code block tagged json (with nothing after it) holding a JSON object with these exact keys, drawn only from the memo above:
- stance: one of Bullish, Bearish, or Neutral
- conviction: one of High, Medium, or Low
- thesis: a single sentence capturing the investment case
- upside_drivers: a list of up to 5 short plain-text strings
- risk_drivers: a list of up to 5 short plain-text strings
- key_catalysts: a list of up to 5 short plain-text strings
- data_quality_flags: a list of up to 5 short plain-text strings (timing mismatches, non-GAAP caveats, missing or derived data, control or governance issues)
The stance and conviction in this JSON must match the memo's Investment Conclusion. Keep every string plain natural language with no nested quotes, braces, or brackets.

The first report analysis (SEC filing) is:

{expert_analysis}

The second report analysis (missing information check) is:

{missing_analysis}
{transcript_section}
"""


batch_comparison_prompt_template = """
Instruction: You are a senior portfolio analyst. You have received final investment analysis reports for {ticker_count} companies from your research team. Your task is to produce a concise screening summary that ranks these companies by investment attractiveness.

For each company, you have a full analysis report. Compare them across these dimensions:
- Financial health and trajectory
- Growth potential and catalysts
- Risk profile (red/yellow flags)
- Management quality (based on earnings call if available)
- Overall investment stance from each report's conclusion

The analysis reports are:

{all_reports}

Output Format:
- Start with a ranking table: Rank | Ticker | Stance | Conviction | One-line rationale
- Then provide a brief paragraph (3-5 sentences) for each company explaining why it ranks where it does.
- End with "Top Picks" — which 1-3 companies deserve deeper research and why.
- Keep it concise — this is a screening tool, not a deep dive.
- Use Markdown formatting.
- Do not include conversational offers, follow-up questions, or meta-commentary about what you can do next.
"""

transcript_prompt_template = """
Instruction: You are an expert financial analyst. Analyze the following earnings call transcript provided below. Base your analysis solely on the transcript text, avoiding external data or prior knowledge not present here.

The transcript text is:

{transcript_text}

Analysis Task: Generate a concise report with these sections:

1. Call Identification:
   - Company Name: (Extract from text)
   - Call Date: (Extract from text)
   - Quarter/Period Covered: (Extract from text)

2. Key Highlights:
   - Summarize all significant points management emphasized (e.g., revenue growth, new products, challenges).
   - Note any forward-looking statements about performance or strategy.

3. Management Sentiment:
   - Assess the tone (e.g., optimistic, cautious, defensive) based on word choice and context.
   - Highlight any shifts or contradictions from prepared remarks to Q&A, if any and the context around them.

4. Critical Issues:
   - Identify any yellow or red flags, concerns, or risks raised by management or analysts and the context around them.

5. Analyst Q&A Insights:
   - Extract key questions from analysts and management’s responses.
   - Focus on non-neutral info.

Limit yourself to the information provided in the transcript text, do not include any information outside of the transcript text.

Output Format: Use Markdown with headings and bullet points. Keep it tight and focused on the juiciest insights. Do not include conversational offers, follow-up questions, or meta-commentary about what you can do next.
"""

missing_analysis_prompt_template = """
Instruction: You are a high iq senior financial analyst. You manage a team of expert analysts that compiled a report analysis of an SEC filing. 
You will present the report to the CEO and the board of directors. Your job is to review the report made by your team and provide a report analysis of crucial missing information present in the SEC filing and not present in the report made by your team. 
Only include in your report information that is missing, do not include any information that is present in the report made by your team unless in a situation where their report is wrong. In case you team of experts didn't miss anything important, reply with "Not missing crucial information, good job team!".
Below are the report analysis and the SEC filing text provided. Your analysis must be based exclusively on the information contained within this prompt. Do not incorporate any external data, real-time information (like current stock prices unless mentioned in the text), or prior knowledge about the company not present in this specific SEC filing or team report.

The report analysis is:

{expert_analysis}

The SEC filing text is:

{filing_text}

Output Format: Present the analysis clearly using Markdown for headings and bullet points. Ensure all information is strictly derived from the provided information. Do not include conversational offers, follow-up questions, or meta-commentary about what you can do next.

"""

expert_analysis_prompt_template = """
Instruction: You are an expert financial analyst. Analyze the following SEC filing text and the programmatic quantitative scorecard provided below. Your analysis must be based exclusively on the information contained within these sources. Do not incorporate any external data, real-time information, or prior knowledge about the company not present in these sources.

The filing text is:

{filing_text}

Here is the programmatic quantitative scorecard extracted directly from verified SEC XBRL facts:

{quant_scorecard}

CRITICAL VERIFICATION RULES:
1. The scorecard above is calculated deterministically from the company's reported SEC XBRL facts. The arithmetic is exact — do NOT alter or recalculate the values. However, you SHOULD judge whether each metric is meaningful for THIS company's business model and say so plainly. Altman Z' and Beneish M assume a non-financial, revenue-generating operating company; when the scorecard marks a metric "NOT APPLICABLE" or "advisory" (e.g. for a financial, REIT, or pre-revenue issuer, or a hypergrowth firm where Beneish penalizes growth), do not treat that number as a clean signal — explain the limitation, drawing on the scorecard's "Applicability Notes".
2. If any metric is marked as "UNABLE TO COMPUTE" or "MISSING - Footnotes Search Required", you MUST scan the filing text (including footnote disclosures) to see if the company discloses these values or explains their absence. If found, highlight them in your report.
3. Under no circumstances should you invent, estimate, or hallucinate any financial figures. If a metric is missing from both the programmatic scorecard and the filing text, state clearly that the company did not disclose it in the public filing.

Analysis Task: Generate a structured report summarizing the key information from the filing. Use the following sections:

1. Filing Identification:
   - Company Name: (Extract from text)
   - Ticker Symbol: (Extract if available, otherwise state N/A)
   - Filing Type: (e.g., 10-K, 10-Q - Extract from text)
   - Filing Period End Date: (Extract from text - e.g., "Fiscal year ended December 31, 2023" or "Quarter ended March 31, 2024")

2. Business Overview (Derived primarily from 'Business' - Item 1 in 10-K, or updates in 10-Q):
   - Provide a concise summary of the company's business operations, products, services, and revenue sources as described in this filing.
   - Summarize the company's stated strategy, primary markets, and competition based on the text.
   - Note any significant developments or changes mentioned in this section compared to previous periods, if discussed.

3. Risk Factors (Derived primarily from 'Risk Factors' - Item 1A in 10-K/Part II, Item 1A in 10-Q):
   - List and briefly summarize the most significant risks disclosed by the company in this filing.
   - Group risks into logical categories if possible (e.g., operational, financial, market, regulatory, strategic).

4. Management's Discussion and Analysis (MD&A) & Quantitative Insights:
   - Integrate the **Deterministic Financial Engineering Scorecard** trends (Altman Z'-Score, Piotroski F-Score, Beneish M-Score) into your analysis of the company's solvency, liquidity, and earnings quality.
   - Summarize management's commentary on financial results (Revenue, Profitability, Key Segment Performance) for the period covered.
   - Highlight key trends, drivers, and challenges discussed by management.
   - Summarize the discussion on Liquidity (cash position, cash flows, debt) and Capital Resources (funding sources, capital expenditures).
   - Mention any critical accounting estimates or significant non-recurring items discussed.

5. Quantitative and Qualitative Disclosures About Market Risk (Derived primarily from Item 7A in 10-K/Part I, Item 3 in 10-Q):
   - Summarize the company's primary market risk exposures (e.g., interest rate, foreign currency, commodity price) and how they are managed, based on the disclosures.

6. Financial Statements Insights & Footnote Audit:
   - Provide a high-level overview of major changes or trends visible in the Balance Sheet, Income Statement, and Cash Flow Statement presented in the filing.
   - Conduct a Footnote Audit based on the notes: highlight major acquisitions, divestitures, significant debt agreements, segment reporting changes, or material contingencies.
   - Address any missing or incomplete quantitative metrics from the scorecard by auditing the footnotes for details.

7. Legal Proceedings (Derived primarily from Item 3 in 10-K/Part I, Item 2 in 10-Q):
   - Summarize any material legal proceedings disclosed in this filing.

8. Internal Controls and Procedures (Derived primarily from Item 9A in 10-K/Part II, Item 4 in 10-Q):
   - State management's conclusion on the effectiveness of disclosure controls and procedures.
   - For 10-Ks, state management's assessment of internal control over financial reporting (ICFR) and the auditor's attestation, if provided.

Output Format: Present the analysis clearly using Markdown for headings and bullet points. Ensure all information is strictly derived from the provided filing text and programmatic scorecard. Do not include conversational offers, follow-up questions, or meta-commentary about what you can do next.
"""

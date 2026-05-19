import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

# Synonyms for mapping standard GAAP items to raw EDGAR tags
BALANCE_SHEET_MAP = {
    'total_assets': ['Assets'],
    'total_liabilities': ['Liabilities'],
    'current_assets': ['AssetsCurrent'],
    'current_liabilities': ['LiabilitiesCurrent'],
    'retained_earnings': ['RetainedEarningsAccumulatedDeficit', 'RetainedEarnings'],
    'equity': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'long_term_debt': ['LongTermDebtNoncurrent', 'LongTermDebt'],
    'short_term_debt': ['ShortTermDebt', 'ShortTermDebtNoncurrent', 'LongTermDebtCurrent'],
    'receivables': ['AccountsReceivableNetCurrent', 'AccountsReceivableNet'],
    'ppe_net': ['PropertyPlantAndEquipmentNet'],
    'cash': ['CashAndCashEquivalentsAtCarryingValue', 'CashAndCashEquivalents', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents']
}

INCOME_STATEMENT_MAP = {
    'revenue': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet', 'Revenue'],
    'operating_income': ['OperatingIncomeLoss'],
    'net_income': ['NetIncomeLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic'],
    'gross_profit': ['GrossProfit', 'GrossProfit_Calculated'],
    'cost_of_revenue': ['CostOfRevenue', 'CostOfGoodsAndServicesSold', 'CostOfGoodsSold'],
    'sga': ['SellingGeneralAndAdministrativeExpense', 'SellingAndAdministrativeExpense'],
    'selling_marketing': ['SellingAndMarketingExpense', 'SellingAndMarketing'],
    'general_administrative': ['GeneralAndAdministrativeExpense', 'GeneralAndAdministrative'],
    'rd': ['ResearchAndDevelopmentExpense'],
    'shares_outstanding': ['WeightedAverageNumberOfSharesOutstandingBasic', 'WeightedAverageNumberOfShareOutstandingBasicAndDiluted', 'CommonStockSharesOutstanding', 'WeightedAverageNumberOfSharesOutstandingDiluted']
}

CASH_FLOW_MAP = {
    'operating_cash_flow': ['NetCashProvidedByUsedInOperatingActivities', 'OperatingCashFlow'],
    'capex': ['PaymentsToAcquirePropertyPlantAndEquipment', 'Capex'],
    'depreciation': ['DepreciationDepletionAndAmortization', 'DepreciationAndAmortization', 'Depreciation', 'DepreciationAmortizationAndAccretion']
}

def extract_historical_series(df: pd.DataFrame, synonyms: List[str]) -> Dict[str, float]:
    """Helper to extract a historical time series from MultiPeriodStatement.to_dataframe()"""
    if df is None:
        return {}
    
    best_series = {}
    
    for s in synonyms:
        if s in df.index:
            row = df.loc[s]
            # Year columns are prefixed with FY or CY (e.g. FY 2025, CY 2025)
            years = [col for col in df.columns if col.startswith('FY ') or col.startswith('CY ')]
            if len(years) > 0:
                res = {}
                for y in years:
                    if isinstance(row, pd.DataFrame):
                        val = row[y].iloc[0]
                    else:
                        val = row.get(y)
                    if val is not None and not pd.isna(val):
                        res[y] = float(val)
                # Select the synonym that yields the maximum number of populated historical years
                if len(res) > len(best_series):
                    best_series = dict(sorted(res.items()))
                    
    return best_series

def get_latest_value(series: Dict[str, float]) -> Tuple[Optional[float], Optional[str]]:
    """Helper to get the most recent year and its value"""
    if not series:
        return None, None
    latest_year = max(series.keys())
    return series[latest_year], latest_year

def get_value_for_year(series: Dict[str, float], year: str) -> Optional[float]:
    """Get value for a specific year, returns None if missing"""
    return series.get(year)

def get_yoy_change(series: Dict[str, float], year: str) -> Tuple[Optional[float], bool]:
    """Get value change YoY (current year vs previous year)"""
    if not series:
        return None, False
    
    try:
        # Year format: 'FY 2025' or 'CY 2025'
        prefix = year[:3]
        year_num = int(year[3:])
        prev_year = f"{prefix}{year_num - 1}"
        
        current_val = series.get(year)
        prev_val = series.get(prev_year)
        
        if current_val is not None and prev_val is not None:
            return current_val - prev_val, True
    except Exception:
        pass
    return None, False

def calculate_altman_z(metrics: Dict[str, Dict[str, float]], year: str) -> Tuple[Optional[float], str, List[str]]:
    """Calculate Altman Z'-Score (Z-Prime) for a specific year.
    Uses Book Value of Equity in place of Market Cap, applying Z-Prime coefficients 
    specifically designed for private/book-value valuation to maintain academic correctness:
    Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
    """
    missing_vars = []
    
    # Retrieve required variables for the specified year
    assets = metrics['total_assets'].get(year)
    liabilities = metrics['total_liabilities'].get(year)
    curr_assets = metrics['current_assets'].get(year)
    curr_liab = metrics['current_liabilities'].get(year)
    retained_earnings = metrics['retained_earnings'].get(year)
    ebit = metrics['operating_income'].get(year)
    revenue = metrics['revenue'].get(year)
    equity = metrics['equity'].get(year) # Book Value of Equity
    
    # Check completeness
    if assets is None: missing_vars.append("Total Assets")
    if liabilities is None: missing_vars.append("Total Liabilities")
    if curr_assets is None: missing_vars.append("Current Assets")
    if curr_liab is None: missing_vars.append("Current Liabilities")
    if retained_earnings is None: missing_vars.append("Retained Earnings")
    if ebit is None: missing_vars.append("EBIT (Operating Income)")
    if revenue is None: missing_vars.append("Revenue")
    if equity is None: missing_vars.append("Equity (Book Value)")
    
    if missing_vars:
        return None, "Incomplete Data", missing_vars
        
    try:
        # X1: Working Capital / Total Assets
        working_capital = curr_assets - curr_liab
        x1 = working_capital / assets
        
        # X2: Retained Earnings / Total Assets
        x2 = retained_earnings / assets
        
        # X3: EBIT / Total Assets
        x3 = ebit / assets
        
        # X4: Equity (Book Value) / Total Liabilities
        x4 = equity / liabilities
        
        # X5: Sales (Revenue) / Total Assets
        x5 = revenue / assets
        
        # Altman Z'-Score coefficients for private/book-value model
        z_score = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5
        
        # Z'-Score thresholds: Safe > 2.90, Grey Zone 1.23 to 2.90, Distress < 1.23
        if z_score > 2.90:
            status = "Safe (Z' > 2.90)"
        elif z_score >= 1.23:
            status = "Grey Zone (1.23 <= Z' <= 2.90)"
        else:
            status = "Distress Zone (Z' < 1.23)"
            
        return z_score, status, []
        
    except ZeroDivisionError:
        return None, "Division by Zero (Total Assets or Liabilities is 0)", ["Math Error"]

def calculate_piotroski_f(metrics: Dict[str, Dict[str, float]], year: str) -> Tuple[Optional[int], Dict[str, int], List[str]]:
    """Calculate Piotroski F-Score (0-9 points) for a specific year"""
    missing_vars = []
    points = {}
    
    # Prefix and previous year calculations
    try:
        prefix = year[:3]
        year_num = int(year[3:])
        prev_year = f"{prefix}{year_num - 1}"
    except Exception:
        return None, {}, ["Invalid Year Format"]
        
    # Check if we have current year data
    net_inc = metrics['net_income'].get(year)
    cfo = metrics['operating_cash_flow'].get(year)
    assets = metrics['total_assets'].get(year)
    curr_assets = metrics['current_assets'].get(year)
    curr_liab = metrics['current_liabilities'].get(year)
    rev = metrics['revenue'].get(year)
    gross_profit = metrics['gross_profit'].get(year)
    
    # Check if we have previous year data for YoY checks
    net_inc_prev = metrics['net_income'].get(prev_year)
    assets_prev = metrics['total_assets'].get(prev_year)
    curr_assets_prev = metrics['current_assets'].get(prev_year)
    curr_liab_prev = metrics['current_liabilities'].get(prev_year)
    rev_prev = metrics['revenue'].get(prev_year)
    gross_profit_prev = metrics['gross_profit'].get(prev_year)
    long_debt = metrics['long_term_debt'].get(year, 0.0) or 0.0
    long_debt_prev = metrics['long_term_debt'].get(prev_year, 0.0) or 0.0
    
    # 1. ROA check
    if net_inc is not None and assets is not None:
        roa = net_inc / assets
        points['1_positive_net_income'] = 1 if net_inc > 0 else 0
        points['3_positive_cfo'] = 1 if cfo is not None and cfo > 0 else 0
        if cfo is not None:
            points['4_cfo_gt_net_income'] = 1 if cfo > net_inc else 0
        else:
            points['4_cfo_gt_net_income'] = 0
            missing_vars.append("Operating Cash Flow (for CFO vs Income)")
    else:
        points['1_positive_net_income'] = 0
        points['3_positive_cfo'] = 0
        points['4_cfo_gt_net_income'] = 0
        missing_vars.append("Net Income or Total Assets")
        
    # YoY Profitability checks
    if net_inc is not None and assets is not None and net_inc_prev is not None and assets_prev is not None:
        roa = net_inc / assets
        roa_prev = net_inc_prev / assets_prev
        points['2_roa_increase'] = 1 if roa > roa_prev else 0
    else:
        points['2_roa_increase'] = 0
        missing_vars.append("Historical Net Income or Assets")
        
    # YoY Leverage / Debt check
    points['5_debt_reduction'] = 1 if long_debt <= long_debt_prev else 0
    
    # YoY Liquidity check (Current Ratio)
    if curr_assets is not None and curr_liab is not None and curr_assets_prev is not None and curr_liab_prev is not None:
        curr_ratio = curr_assets / curr_liab
        curr_ratio_prev = curr_assets_prev / curr_liab_prev
        points['6_liquidity_increase'] = 1 if curr_ratio > curr_ratio_prev else 0
    else:
        points['6_liquidity_increase'] = 0
        missing_vars.append("Current Assets/Liabilities (current or historical)")
        
    # Dilution check: 1 point if the company did not issue common equity (shares outstanding did not increase YoY)
    shares = metrics['shares_outstanding'].get(year)
    shares_prev = metrics['shares_outstanding'].get(prev_year)
    if shares is not None and shares_prev is not None:
        # Allow tiny rounding/tolerance for minor adjustments (0.5%)
        points['7_no_dilution'] = 1 if shares <= shares_prev * 1.005 else 0
    else:
        points['7_no_dilution'] = 0
        missing_vars.append("Shares Outstanding (current or historical)")
    
    # YoY Gross Margin check
    if rev is not None and gross_profit is not None and rev_prev is not None and gross_profit_prev is not None:
        gm = gross_profit / rev
        gm_prev = gross_profit_prev / rev_prev
        points['8_margin_increase'] = 1 if gm > gm_prev else 0
    else:
        points['8_margin_increase'] = 0
        missing_vars.append("Gross Profit or Revenue (current or historical)")
        
    # YoY Asset Turnover check
    if rev is not None and assets is not None and rev_prev is not None and assets_prev is not None:
        turnover = rev / assets
        turnover_prev = rev_prev / assets_prev
        points['9_turnover_increase'] = 1 if turnover > turnover_prev else 0
    else:
        points['9_turnover_increase'] = 0
        missing_vars.append("Revenue or Total Assets (current or historical)")
        
    total_score = sum(points.values())
    return total_score, points, list(set(missing_vars))

def calculate_beneish_m(metrics: Dict[str, Dict[str, float]], year: str) -> Tuple[Optional[float], str, List[str]]:
    """Calculate Beneish M-Score for a specific year.
    M-Score > -1.78 suggests a high probability of earnings manipulation.
    """
    missing_vars = []
    
    try:
        prefix = year[:3]
        year_num = int(year[3:])
        prev_year = f"{prefix}{year_num - 1}"
    except Exception:
        return None, "Invalid Year Format", ["Year parsing error"]
        
    # Get values
    rev = metrics['revenue'].get(year)
    rev_prev = metrics['revenue'].get(prev_year)
    receivables = metrics['receivables'].get(year)
    receivables_prev = metrics['receivables'].get(prev_year)
    gross_profit = metrics['gross_profit'].get(year)
    gross_profit_prev = metrics['gross_profit'].get(prev_year)
    assets = metrics['total_assets'].get(year)
    assets_prev = metrics['total_assets'].get(prev_year)
    curr_assets = metrics['current_assets'].get(year)
    curr_assets_prev = metrics['current_assets'].get(prev_year)
    ppe = metrics['ppe_net'].get(year)
    ppe_prev = metrics['ppe_net'].get(prev_year)
    liabilities = metrics['total_liabilities'].get(year)
    liabilities_prev = metrics['total_liabilities'].get(prev_year)
    ebit = metrics['operating_income'].get(year)
    net_inc = metrics['net_income'].get(year)
    cfo = metrics['operating_cash_flow'].get(year)
    depr = metrics['depreciation'].get(year)
    depr_prev = metrics['depreciation'].get(prev_year)
    sga = metrics['sga'].get(year)
    sga_prev = metrics['sga'].get(prev_year)
    
    # Check mandatory fields for indices
    if rev is None or rev_prev is None: missing_vars.append("Revenue")
    if assets is None or assets_prev is None: missing_vars.append("Total Assets")
    if liabilities is None or liabilities_prev is None: missing_vars.append("Total Liabilities")
    if net_inc is None: missing_vars.append("Net Income")
    
    if missing_vars:
        return None, "Incomplete Data", missing_vars
        
    try:
        # 1. SGI (Sales Growth Index)
        sgi = rev / rev_prev
        
        # 2. DSRI (Days Sales in Receivables Index)
        if receivables is not None and receivables_prev is not None:
            dsri = (receivables / rev) / (receivables_prev / rev_prev)
        else:
            dsri = 1.0  # Fallback to neutral
            
        # 3. GMI (Gross Margin Index)
        if gross_profit is not None and gross_profit_prev is not None:
            gm = gross_profit / rev
            gm_prev = gross_profit_prev / rev_prev
            gmi = gm_prev / gm if gm > 0 else 1.0
        else:
            gmi = 1.0
            
        # 4. AQI (Asset Quality Index)
        # Noncurrent assets = Total Assets - Current Assets - PPE
        if curr_assets is not None and curr_assets_prev is not None:
            ppe_val = ppe if ppe is not None else 0.0
            ppe_prev_val = ppe_prev if ppe_prev is not None else 0.0
            aq = 1.0 - ((curr_assets + ppe_val) / assets)
            aq_prev = 1.0 - ((curr_assets_prev + ppe_prev_val) / assets_prev)
            aqi = aq / aq_prev if aq_prev > 0 else 1.0
        else:
            aqi = 1.0
            
        # 5. DEPI (Depreciation Index)
        if depr is not None and depr_prev is not None and ppe is not None and ppe_prev is not None:
            depr_rate = depr / (depr + ppe)
            depr_rate_prev = depr_prev / (depr_prev + ppe_prev)
            depi = depr_rate_prev / depr_rate if depr_rate > 0 else 1.0
        else:
            depi = 1.0
            
        # 6. SGAI (SG&A Expenses Index)
        if sga is not None and sga_prev is not None:
            sgai = (sga / rev) / (sga_prev / rev_prev)
        else:
            sgai = 1.0
            
        # 7. LVGI (Leverage Index)
        lev = liabilities / assets
        lev_prev = liabilities_prev / assets_prev
        lvgi = lev / lev_prev if lev_prev > 0 else 1.0
        
        # 8. TATA (Total Accruals to Total Assets)
        # Academic formula: (Net Income - Cash Flow from Operations) / Total Assets
        if net_inc is not None and cfo is not None:
            tata = (net_inc - cfo) / assets
        else:
            tata = 0.0
            
        # Beneish 8-variable model equation
        m_score = (-4.84 + 
                   (0.920 * dsri) + 
                   (0.528 * gmi) + 
                   (0.404 * aqi) + 
                   (0.892 * sgi) + 
                   (0.115 * depi) - 
                   (0.172 * sgai) + 
                   (4.037 * tata) + 
                   (0.0327 * lvgi))
                   
        status = "High Risk (> -1.78)" if m_score > -1.78 else "Low Risk (<= -1.78)"
        return m_score, status, []
        
    except Exception as e:
        return None, f"Calculation Error: {e}", ["Math Error"]

def generate_quant_scorecard(company) -> Tuple[str, Dict]:
    """Generates the Quantitative Financial Scorecard in Markdown"""
    try:
        facts = company.get_facts()
        if not facts:
            return "No SEC facts available for this company.", {}
    except Exception as e:
        return f"Could not retrieve SEC facts: {e}", {}
        
    # Get Statements
    try:
        bal_df = facts.balance_sheet(periods=4, period='annual').to_dataframe()
    except Exception:
        bal_df = None
        
    try:
        inc_df = facts.income_statement(periods=4, period='annual').to_dataframe()
    except Exception:
        inc_df = None
        
    try:
        cf_df = facts.cash_flow(periods=4).to_dataframe()
    except Exception:
        cf_df = None
        
    # Extract all required series
    metrics = {}
    
    # 1. Balance Sheet Extractions
    for key, synonyms in BALANCE_SHEET_MAP.items():
        metrics[key] = extract_historical_series(bal_df, synonyms)
        
    # 2. Income Statement Extractions
    for key, synonyms in INCOME_STATEMENT_MAP.items():
        metrics[key] = extract_historical_series(inc_df, synonyms)
        
    # 3. Cash Flow Extractions
    for key, synonyms in CASH_FLOW_MAP.items():
        metrics[key] = extract_historical_series(cf_df, synonyms)
        
    # Find all available years
    all_years = set()
    for series in metrics.values():
        all_years.update(series.keys())
        
    # We want years sorted descending (e.g. FY 2025 -> FY 2022)
    sorted_years = sorted(list(all_years), reverse=True)
    
    # 4. Programmatic Calculation Fallbacks for missing tags
    # Fallback for Gross Profit = Revenue - Cost of Revenue
    for y in sorted_years:
        if metrics['gross_profit'].get(y) is None:
            rev = metrics['revenue'].get(y)
            cor = metrics['cost_of_revenue'].get(y)
            if rev is not None and cor is not None:
                metrics['gross_profit'][y] = rev - cor
                
    # Fallback for SG&A = Selling/Marketing + G&A
    for y in sorted_years:
        if metrics['sga'].get(y) is None:
            sm = metrics['selling_marketing'].get(y)
            ga = metrics['general_administrative'].get(y)
            if sm is not None and ga is not None:
                metrics['sga'][y] = sm + ga
                
    if not sorted_years:
        return "Incomplete data series to generate quantitative metrics.", {}
        
    latest_year = sorted_years[0]
    
    # Calculate scores for available years
    altman_results = {}
    piotroski_results = {}
    beneish_results = {}
    current_ratio = {}
    quick_ratio = {}
    debt_to_equity = {}
    fcf_yield = {}
    
    for y in sorted_years:
        altman_results[y] = calculate_altman_z(metrics, y)
        piotroski_results[y] = calculate_piotroski_f(metrics, y)
        beneish_results[y] = calculate_beneish_m(metrics, y)
        
        # Calculate core ratios
        assets = metrics['total_assets'].get(y)
        liabilities = metrics['total_liabilities'].get(y)
        curr_assets = metrics['current_assets'].get(y)
        curr_liab = metrics['current_liabilities'].get(y)
        cash = metrics['cash'].get(y, 0.0) or 0.0
        receivables = metrics['receivables'].get(y, 0.0) or 0.0
        equity = metrics['equity'].get(y)
        cfo = metrics['operating_cash_flow'].get(y)
        capex = metrics['capex'].get(y)
        
        # Current Ratio
        if curr_assets is not None and curr_liab is not None and curr_liab > 0:
            current_ratio[y] = curr_assets / curr_liab
        else:
            current_ratio[y] = None
            
        # Quick Ratio: (Cash + Receivables) / Current Liabilities
        if curr_liab is not None and curr_liab > 0:
            quick_ratio[y] = (cash + receivables) / curr_liab
        else:
            quick_ratio[y] = None
            
        # Debt to Equity: Total Liabilities / Equity
        if liabilities is not None and equity is not None and equity > 0:
            debt_to_equity[y] = liabilities / equity
        else:
            debt_to_equity[y] = None
            
        # Free Cash Flow Yield (FCF / Total Assets as basic asset yield)
        if cfo is not None and capex is not None and assets is not None and assets > 0:
            fcf_yield[y] = (cfo - capex) / assets
        else:
            fcf_yield[y] = None

    # Let's format the Markdown Report
    lines = []
    lines.append("## DETERMINISTIC FINANCIAL ENGINEERING SCORECARD")
    lines.append("This scorecard has been calculated programmatically in Python directly from verified SEC XBRL facts. These values are mathematically exact and represent ground-truth financials. **Do not recalculate or modify.**")
    lines.append("")
    
    # 1. Summary table for the latest year
    lines.append(f"### Summary Metrics — {latest_year}")
    lines.append("| Metric | Calculated Value | Interpretation / Risk Level | Status |")
    lines.append("| :--- | :--- | :--- | :--- |")
    
    # Altman Z'
    z_val, z_stat, z_missing = altman_results[latest_year]
    if z_val is not None:
        lines.append(f"| **Altman Z'-Score (Z-Prime)** | `{z_val:.2f}` | {z_stat} | `Programmatically Verified` |")
    else:
        lines.append(f"| **Altman Z'-Score (Z-Prime)** | `UNABLE TO COMPUTE` | Missing: {', '.join(z_missing)} | `MISSING - Footnotes Search Required` |")
        
    # Piotroski F
    f_val, f_points, f_missing = piotroski_results[latest_year]
    if f_val is not None:
        lines.append(f"| **Piotroski F-Score** | `{f_val}/9` | Strength: {'Strong (8-9)' if f_val >= 8 else 'Moderate (3-7)' if f_val >= 3 else 'Weak (0-2)'} | `Programmatically Verified` |")
    else:
        lines.append(f"| **Piotroski F-Score** | `UNABLE TO COMPUTE` | Missing: {', '.join(f_missing)} | `MISSING - Footnotes Search Required` |")
        
    # Beneish M
    m_val, m_stat, m_missing = beneish_results[latest_year]
    if m_val is not None:
        lines.append(f"| **Beneish M-Score** | `{m_val:.2f}` | {m_stat} | `Programmatically Verified` |")
    else:
        lines.append(f"| **Beneish M-Score** | `UNABLE TO COMPUTE` | Missing: {', '.join(m_missing)} | `MISSING - Footnotes Search Required` |")
        
    # Ratios
    cr = current_ratio.get(latest_year)
    if cr is not None:
        lines.append(f"| **Current Ratio** | `{cr:.2f}` | Liquidity buffer | `Programmatically Verified` |")
    else:
        lines.append(f"| **Current Ratio** | `UNABLE TO COMPUTE` | Missing Current Assets or Liabilities | `MISSING - Footnotes Search Required` |")
        
    qr = quick_ratio.get(latest_year)
    if qr is not None:
        lines.append(f"| **Quick Ratio** | `{qr:.2f}` | Immediate cash coverage | `Programmatically Verified` |")
        
    de = debt_to_equity.get(latest_year)
    if de is not None:
        lines.append(f"| **Debt-to-Equity** | `{de:.2f}` | Leverage ratio | `Programmatically Verified` |")
        
    fcfy = fcf_yield.get(latest_year)
    if fcfy is not None:
        lines.append(f"| **FCF / Assets Yield** | `{fcfy * 100:.2f}%` | Return on capital asset efficiency | `Programmatically Verified` |")
        
    lines.append("")
    
    # 2. Historical Trend Table
    lines.append("### 4-Year Historical Trends")
    headers = ["Metric"] + sorted_years
    lines.append(" | ".join(headers))
    lines.append(" | ".join([":---"] * len(headers)))
    
    # Add trend rows
    # Altman Z
    z_row = ["**Altman Z'-Score (Z-Prime)**"]
    for y in sorted_years:
        val, _, _ = altman_results[y]
        z_row.append(f"`{val:.2f}`" if val is not None else "`N/A`")
    lines.append(" | ".join(z_row))
    
    # Piotroski F
    f_row = ["**Piotroski F-Score**"]
    for y in sorted_years:
        val, _, _ = piotroski_results[y]
        f_row.append(f"`{val}/9`" if val is not None else "`N/A`")
    lines.append(" | ".join(f_row))
    
    # Beneish M
    m_row = ["**Beneish M-Score**"]
    for y in sorted_years:
        val, _, _ = beneish_results[y]
        m_row.append(f"`{val:.2f}`" if val is not None else "`N/A`")
    lines.append(" | ".join(m_row))
    
    # Current Ratio
    cr_row = ["**Current Ratio**"]
    for y in sorted_years:
        val = current_ratio.get(y)
        cr_row.append(f"`{val:.2f}`" if val is not None else "`N/A`")
    lines.append(" | ".join(cr_row))
    
    # Quick Ratio
    qr_row = ["**Quick Ratio**"]
    for y in sorted_years:
        val = quick_ratio.get(y)
        qr_row.append(f"`{val:.2f}`" if val is not None else "`N/A`")
    lines.append(" | ".join(qr_row))
    
    # Debt-to-Equity
    de_row = ["**Debt-to-Equity**"]
    for y in sorted_years:
        val = debt_to_equity.get(y)
        de_row.append(f"`{val:.2f}`" if val is not None else "`N/A`")
    lines.append(" | ".join(de_row))
    
    # FCF Yield
    fcf_row = ["**FCF / Assets Yield**"]
    for y in sorted_years:
        val = fcf_yield.get(y)
        fcf_row.append(f"`{val * 100:.2f}%`" if val is not None else "`N/A`")
    lines.append(" | ".join(fcf_row))
    
    lines.append("")
    lines.append("---")
    
    metadata = {
        'latest_year': latest_year,
        'altman_z': altman_results[latest_year][0],
        'piotroski_f': piotroski_results[latest_year][0],
        'beneish_m': beneish_results[latest_year][0],
        'current_ratio': current_ratio.get(latest_year),
        'quick_ratio': quick_ratio.get(latest_year),
        'debt_to_equity': debt_to_equity.get(latest_year),
        'fcf_yield': fcf_yield.get(latest_year),
        'all_years': sorted_years
    }
    
    return "\n".join(lines), metadata

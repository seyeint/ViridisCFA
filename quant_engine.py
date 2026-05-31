import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

# Bump this version when calculation logic changes (scores, coefficients, etc.)
# The cache system uses this to automatically invalidate stale expert analyses.
# v1.3: corrected Beneish TATA/LVGI coefficients; added business-model applicability gate.
# v1.4: Beneish out-of-range guard; cash-runway metric.
QUANT_ENGINE_VERSION = "1.4"

# Synonyms for mapping standard GAAP items to raw EDGAR tags
BALANCE_SHEET_MAP = {
    'total_assets': ['Assets'],
    'total_liabilities': ['Liabilities'],
    'current_assets': ['AssetsCurrent'],
    'current_liabilities': ['LiabilitiesCurrent'],
    'retained_earnings': ['RetainedEarningsAccumulatedDeficit', 'RetainedEarnings'],
    'equity': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'redeemable_noncontrolling_interest': ['RedeemableNoncontrollingInterestEquityCarryingAmount'],
    'long_term_debt': ['LongTermDebtNoncurrent', 'LongTermDebt'],
    'short_term_debt': ['ShortTermDebt', 'ShortTermDebtNoncurrent', 'LongTermDebtCurrent'],
    'receivables': [
        'AccountsReceivableNetCurrent',
        'AccountsReceivableNet',
        'AccountsNotesAndLoansReceivableNetCurrent',
        'TradeAccountsReceivableNetCurrent',
        'ReceivablesNetCurrent',
        'ReceivablesNet',
    ],
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

def apply_balance_sheet_fallbacks(metrics: Dict[str, Dict[str, float]], years: List[str]) -> List[Dict[str, str]]:
    """Derive missing balance-sheet totals only from accounting identities.

    Some filers omit a plain XBRL `Liabilities` or `StockholdersEquity` row from
    the standardized statement dataframe while still reporting the components
    needed to derive it. These fallbacks are deterministic and auditable.
    """
    derived = []
    redeemable_series = metrics.get('redeemable_noncontrolling_interest', {})

    for year in years:
        assets = metrics['total_assets'].get(year)
        liabilities = metrics['total_liabilities'].get(year)
        equity = metrics['equity'].get(year)
        redeemable_nci = redeemable_series.get(year) or 0.0

        if liabilities is None and assets is not None and equity is not None:
            derived_liabilities = assets - equity - redeemable_nci
            if derived_liabilities >= 0:
                metrics['total_liabilities'][year] = derived_liabilities
                liabilities = derived_liabilities
                derived.append({
                    'metric': 'total_liabilities',
                    'year': year,
                    'formula': 'Total Assets - Equity - Redeemable Noncontrolling Interest',
                })

        if equity is None and assets is not None and liabilities is not None:
            derived_equity = assets - liabilities - redeemable_nci
            if derived_equity >= 0:
                metrics['equity'][year] = derived_equity
                derived.append({
                    'metric': 'equity',
                    'year': year,
                    'formula': 'Total Assets - Total Liabilities - Redeemable Noncontrolling Interest',
                })

    return derived

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

def calculate_beneish_m(metrics: Dict[str, Dict[str, float]], year: str) -> Tuple[Optional[float], str, List[str], List[str]]:
    """Calculate Beneish M-Score for a specific year.
    M-Score > -1.78 suggests a high probability of earnings manipulation.
    """
    missing_vars = []
    
    try:
        prefix = year[:3]
        year_num = int(year[3:])
        prev_year = f"{prefix}{year_num - 1}"
    except Exception:
        return None, "Invalid Year Format", ["Year parsing error"], []
        
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
        return None, "Incomplete Data", missing_vars, []
        
    try:
        imputed = []  # Track components defaulted to neutral due to missing data
        
        # 1. SGI (Sales Growth Index)
        sgi = rev / rev_prev
        
        # 2. DSRI (Days Sales in Receivables Index)
        if receivables is not None and receivables_prev is not None:
            dsri = (receivables / rev) / (receivables_prev / rev_prev)
        else:
            dsri = 1.0
            imputed.append("DSRI (Receivables)")
            
        # 3. GMI (Gross Margin Index)
        if gross_profit is not None and gross_profit_prev is not None:
            gm = gross_profit / rev
            gm_prev = gross_profit_prev / rev_prev
            gmi = gm_prev / gm if gm > 0 else 1.0
        else:
            gmi = 1.0
            imputed.append("GMI (Gross Margin)")
            
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
            imputed.append("AQI (Asset Quality)")
            
        # 5. DEPI (Depreciation Index)
        if depr is not None and depr_prev is not None and ppe is not None and ppe_prev is not None:
            depr_rate = depr / (depr + ppe)
            depr_rate_prev = depr_prev / (depr_prev + ppe_prev)
            depi = depr_rate_prev / depr_rate if depr_rate > 0 else 1.0
        else:
            depi = 1.0
            imputed.append("DEPI (Depreciation)")
            
        # 6. SGAI (SG&A Expenses Index)
        if sga is not None and sga_prev is not None:
            sgai = (sga / rev) / (sga_prev / rev_prev)
        else:
            sgai = 1.0
            imputed.append("SGAI (SG&A)")
            
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
            imputed.append("TATA (Accruals)")
            
        # Beneish 8-variable model equation (Beneish 1999, Financial Analysts Journal).
        # Canonical coefficients — TATA = +4.679 and LVGI = -0.327. Do not "simplify"
        # these: an earlier transcription used 4.037 and +0.0327, which biased every
        # M-Score upward by ~+0.36 and flipped the leverage term's sign.
        m_score = (-4.84 +
                   (0.920 * dsri) +
                   (0.528 * gmi) +
                   (0.404 * aqi) +
                   (0.892 * sgi) +
                   (0.115 * depi) -
                   (0.172 * sgai) +
                   (4.679 * tata) -
                   (0.327 * lvgi))
                   
        status = "High Risk (> -1.78)" if m_score > -1.78 else "Grey Zone (-2.22 to -1.78)" if m_score > -2.22 else "Low Risk (<= -2.22)"
        return m_score, status, [], imputed
        
    except Exception as e:
        return None, f"Calculation Error: {e}", ["Math Error"], []

# SEC Division H (SIC 6000–6799): Finance, Insurance, Real Estate. For these issuers
# Altman Z' (working-capital and sales-to-assets terms) and Beneish M (sales/accrual
# structure) are category errors — banks/insurers/REITs don't run a classified
# balance sheet or convert "sales" the way the models assume.
FINANCIAL_SIC_RANGE = (6000, 6799)

# Beneish M is a probit score calibrated on established firms; real companies fall well
# inside this band. A value outside it is a small/early-stage numerical artifact (tiny
# or volatile denominators blow up the component indices), not a real manipulation
# signal — flag it advisory regardless of sector.
BENEISH_MEANINGFUL_RANGE = (-5.0, 1.0)


def beneish_out_of_meaningful_range(m_val) -> bool:
    """True when a Beneish M-Score is outside its empirically meaningful band and
    should not be read as a clean manipulation signal."""
    if m_val is None:
        return False
    return not (BENEISH_MEANINGFUL_RANGE[0] <= m_val <= BENEISH_MEANINGFUL_RANGE[1])


def cash_runway_years(cash, operating_cash_flow):
    """Years of cash left at the current operating burn rate. Returns None when the
    company is not burning operating cash (CFO >= 0) or inputs are missing — runway is
    only a meaningful signal for cash-consuming issuers (where it matters far more than
    a static liquidity ratio)."""
    if cash is None or operating_cash_flow is None or operating_cash_flow >= 0:
        return None
    return cash / abs(operating_cash_flow)


def classify_business_model(company, metrics, latest_year) -> Dict:
    """Deterministically classify the issuer so the scorecard never presents model
    output where the model does not apply. Uses only the SIC code and the
    already-extracted financials — no extra network calls. Returns the archetype and
    a per-model applicability verdict ('applicable' | 'advisory' | 'not_applicable')
    plus plain-language caveats for the narrative layer.
    """
    sic = None
    try:
        raw_sic = getattr(company, "sic", None)
        if raw_sic is not None and str(raw_sic).strip().isdigit():
            sic = int(str(raw_sic).strip())
    except Exception:
        sic = None

    revenue = metrics.get("revenue", {}).get(latest_year)
    assets = metrics.get("total_assets", {}).get(latest_year)
    rev_to_assets = (revenue / assets) if (revenue is not None and assets and assets > 0) else None

    sgi = None
    try:
        prefix, year_num = latest_year[:3], int(latest_year[3:])
        rev_prev = metrics.get("revenue", {}).get(f"{prefix}{year_num - 1}")
        if revenue is not None and rev_prev not in (None, 0):
            sgi = revenue / rev_prev
    except Exception:
        sgi = None

    is_financial = sic is not None and FINANCIAL_SIC_RANGE[0] <= sic <= FINANCIAL_SIC_RANGE[1]
    is_pre_revenue = (revenue is None) or (rev_to_assets is not None and rev_to_assets < 0.02)
    is_hypergrowth = sgi is not None and sgi > 1.5

    result = {
        "sic": sic,
        "archetype": "standard",
        "altman": "applicable",
        "beneish": "applicable",
        "piotroski": "applicable",
        "caveats": [],
    }

    if is_financial:
        result.update({"archetype": "financial", "altman": "not_applicable",
                       "beneish": "not_applicable", "piotroski": "advisory"})
        result["caveats"].append(
            f"Business model — financial / real-estate issuer (SIC {sic}). Altman Z' and Beneish M "
            "assume a non-financial operating company with a classified balance sheet; they are not "
            "meaningful here and are suppressed. Read Piotroski signals (ROA, turnover, margin) with "
            "caution and prefer sector-appropriate measures (e.g. NIM, FFO, combined ratio)."
        )
    elif is_pre_revenue:
        result.update({"archetype": "pre-revenue", "beneish": "not_applicable", "altman": "advisory"})
        result["caveats"].append(
            "Business model — pre-revenue / non-operating issuer (sales are negligible relative to "
            "assets). Beneish M (built on sales-growth and gross-margin indices) is not meaningful and "
            "is suppressed; a deeply negative Altman Z' reflects the absence of operating revenue, not "
            "necessarily imminent insolvency — assess cash runway and burn rate instead."
        )
    elif is_hypergrowth:
        result.update({"archetype": "hypergrowth", "beneish": "advisory"})
        result["caveats"].append(
            f"Business model — high revenue growth (sales up ~{(sgi - 1) * 100:.0f}% YoY). Beneish M "
            "penalizes rapid sales and asset growth and routinely flags clean high-growth companies as "
            "'High Risk'; treat any elevated M-Score as growth-driven rather than dispositive of "
            "manipulation."
        )

    return result


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
        cf_df = facts.cashflow_statement(periods=4, period='annual').to_dataframe()
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

    derived_fields = apply_balance_sheet_fallbacks(metrics, sorted_years)
                
    if not sorted_years:
        return "Incomplete data series to generate quantitative metrics.", {}
        
    latest_year = sorted_years[0]
    business_model = classify_business_model(company, metrics, latest_year)

    # Calculate scores for available years
    altman_results = {}
    piotroski_results = {}
    beneish_results = {}
    current_ratio = {}
    quick_ratio = {}
    quick_ratio_missing = {}
    debt_to_equity = {}
    fcf_yield = {}
    runway_years = {}

    for y in sorted_years:
        altman_results[y] = calculate_altman_z(metrics, y)
        piotroski_results[y] = calculate_piotroski_f(metrics, y)
        beneish_results[y] = calculate_beneish_m(metrics, y)
        
        # Calculate core ratios
        assets = metrics['total_assets'].get(y)
        liabilities = metrics['total_liabilities'].get(y)
        curr_assets = metrics['current_assets'].get(y)
        curr_liab = metrics['current_liabilities'].get(y)
        cash = metrics['cash'].get(y)
        receivables = metrics['receivables'].get(y)
        equity = metrics['equity'].get(y)
        cfo = metrics['operating_cash_flow'].get(y)
        capex = metrics['capex'].get(y)
        
        # Current Ratio
        if curr_assets is not None and curr_liab is not None and curr_liab > 0:
            current_ratio[y] = curr_assets / curr_liab
        else:
            current_ratio[y] = None
            
        # Quick Ratio: (Cash + Receivables) / Current Liabilities.
        # Do not silently default missing receivables to zero; that can make
        # working-capital-heavy companies look artificially illiquid.
        missing_quick_inputs = []
        if cash is None:
            missing_quick_inputs.append("Cash")
        if receivables is None:
            missing_quick_inputs.append("Receivables")
        if curr_liab is None:
            missing_quick_inputs.append("Current Liabilities")

        if curr_liab is not None and curr_liab > 0 and cash is not None and receivables is not None:
            quick_ratio[y] = (cash + receivables) / curr_liab
            quick_ratio_missing[y] = []
        else:
            quick_ratio[y] = None
            quick_ratio_missing[y] = missing_quick_inputs or ["Current Liabilities is zero"]
            
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

        # Cash runway (years) at the current operating burn — meaningful only when burning.
        runway_years[y] = cash_runway_years(cash, cfo)

    # Let's format the Markdown Report
    lines = []
    lines.append("## DETERMINISTIC FINANCIAL ENGINEERING SCORECARD")
    lines.append("These metrics are computed deterministically from the company's reported SEC XBRL facts. The arithmetic is exact — **do not recalculate or modify the values** — but each metric is only as meaningful as its fit to the company's business model. **Altman Z' and Beneish M assume a non-financial, revenue-generating operating company; see Applicability Notes below.**")
    lines.append("")
    
    # 1. Summary table for the latest year
    lines.append(f"### Summary Metrics — {latest_year}")
    lines.append("| Metric | Calculated Value | Interpretation / Risk Level | Status |")
    lines.append("| :--- | :--- | :--- | :--- |")
    
    # Altman Z'
    z_val, z_stat, z_missing = altman_results[latest_year]
    if business_model['altman'] == 'not_applicable':
        lines.append("| **Altman Z'-Score (Z-Prime)** | `NOT APPLICABLE` | Not meaningful for this business model — see Applicability Notes | `Not applicable for this business model` |")
    elif z_val is not None:
        z_status = "`Programmatically Computed`"
        if any(d['year'] == latest_year and d['metric'] in ('total_liabilities', 'equity') for d in derived_fields):
            z_status = "`Computed (derived balance-sheet inputs)`"
        z_interp = z_stat
        if business_model['altman'] == 'advisory':
            z_interp = f"{z_stat} — advisory; see Applicability Notes"
            z_status = "`Computed (applicability caveat)`"
        lines.append(f"| **Altman Z'-Score (Z-Prime)** | `{z_val:.2f}` | {z_interp} | {z_status} |")
    else:
        lines.append(f"| **Altman Z'-Score (Z-Prime)** | `UNABLE TO COMPUTE` | Missing: {', '.join(z_missing)} | `MISSING - Footnotes Search Required` |")
        
    # Piotroski F
    f_val, f_points, f_missing = piotroski_results[latest_year]
    if f_val is not None:
        strength = 'Strong (8-9)' if f_val >= 8 else 'Moderate (3-7)' if f_val >= 3 else 'Weak (0-2)'
        f_status = "`Programmatically Computed`"
        if business_model['piotroski'] == 'advisory':
            strength = f"{strength} — advisory; see Applicability Notes"
            f_status = "`Computed (applicability caveat)`"
        lines.append(f"| **Piotroski F-Score** | `{f_val}/9` | Strength: {strength} | {f_status} |")
    else:
        lines.append(f"| **Piotroski F-Score** | `UNABLE TO COMPUTE` | Missing: {', '.join(f_missing)} | `MISSING - Footnotes Search Required` |")
        
    # Beneish M
    m_val, m_stat, m_missing, m_imputed = beneish_results[latest_year]
    if business_model['beneish'] == 'not_applicable':
        lines.append("| **Beneish M-Score** | `NOT APPLICABLE` | Not meaningful for this business model — see Applicability Notes | `Not applicable for this business model` |")
    elif m_val is not None:
        imputed_note = ""
        verification = "`Programmatically Computed`"
        if m_imputed:
            imputed_note = f" ⚠️ {len(m_imputed)}/8 components defaulted to neutral: {', '.join(m_imputed)}. Score may understate risk."
            verification = "`Computed (imputed components)`"
        m_interp = f"{m_stat}{imputed_note}"
        if beneish_out_of_meaningful_range(m_val):
            m_interp = f"{m_stat} — advisory: outside the model's meaningful range, so the indices are numerically unstable (typical of very small or early-stage issuers); not a clean manipulation signal{imputed_note}"
            verification = "`Computed (advisory — out of range)`"
            business_model['caveats'].append(
                f"Beneish M-Score ({m_val:.2f}) is outside the model's empirically meaningful range; for very "
                "small or early-stage issuers the component indices are numerically unstable, so it is not a "
                "reliable earnings-manipulation signal."
            )
        elif business_model['beneish'] == 'advisory':
            m_interp = f"{m_stat} — advisory (high-growth false positives likely); see Applicability Notes{imputed_note}"
            verification = "`Computed (applicability caveat)`"
        lines.append(f"| **Beneish M-Score** | `{m_val:.2f}` | {m_interp} | {verification} |")
    else:
        lines.append(f"| **Beneish M-Score** | `UNABLE TO COMPUTE` | Missing: {', '.join(m_missing)} | `MISSING - Footnotes Search Required` |")
        
    # Ratios
    cr = current_ratio.get(latest_year)
    if cr is not None:
        lines.append(f"| **Current Ratio** | `{cr:.2f}` | Liquidity buffer | `Programmatically Computed` |")
    else:
        lines.append(f"| **Current Ratio** | `UNABLE TO COMPUTE` | Missing Current Assets or Liabilities | `MISSING - Footnotes Search Required` |")
        
    qr = quick_ratio.get(latest_year)
    if qr is not None:
        lines.append(f"| **Quick Ratio** | `{qr:.2f}` | Immediate cash coverage | `Programmatically Computed` |")
    else:
        missing_qr = ", ".join(quick_ratio_missing.get(latest_year, ["Cash or Receivables"]))
        lines.append(f"| **Quick Ratio** | `UNABLE TO COMPUTE` | Missing: {missing_qr} | `MISSING - Footnotes Search Required` |")
        
    de = debt_to_equity.get(latest_year)
    if de is not None:
        de_status = "`Programmatically Computed`"
        if any(d['year'] == latest_year and d['metric'] in ('total_liabilities', 'equity') for d in derived_fields):
            de_status = "`Computed (derived balance-sheet inputs)`"
        lines.append(f"| **Debt-to-Equity** | `{de:.2f}` | Leverage ratio | {de_status} |")
        
    fcfy = fcf_yield.get(latest_year)
    if fcfy is not None:
        lines.append(f"| **FCF / Assets Yield** | `{fcfy * 100:.2f}%` | Return on capital asset efficiency | `Programmatically Computed` |")

    rwy = runway_years.get(latest_year)
    ocf_latest = metrics['operating_cash_flow'].get(latest_year)
    if rwy is not None:
        if rwy >= 3:
            rwy_verdict = "ample"
        elif rwy >= 1.5:
            rwy_verdict = "adequate"
        elif rwy >= 0.75:
            rwy_verdict = "tightening"
        else:
            rwy_verdict = "critical (under ~9 months of cash)"
        lines.append(f"| **Cash Runway** | `{rwy:.1f} yrs` | Cash / current operating burn — {rwy_verdict} | `Programmatically Computed` |")
    elif ocf_latest is not None and ocf_latest >= 0:
        lines.append("| **Cash Runway** | `N/A` | Operating cash flow positive — self-funding, not burning | `Programmatically Computed` |")

    lines.append("")
    
    # Applicability notes — surfaced when a model does not fit the issuer's business model.
    if business_model['caveats']:
        lines.append("### Applicability Notes")
        for caveat in business_model['caveats']:
            lines.append(f"- {caveat}")
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
        val, _, _, _ = beneish_results[y]
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

    # Cash Runway
    rwy_row = ["**Cash Runway**"]
    for y in sorted_years:
        val = runway_years.get(y)
        rwy_row.append(f"`{val:.1f} yrs`" if val is not None else "`N/A`")
    lines.append(" | ".join(rwy_row))

    lines.append("")
    
    # Methodology definitions — self-documenting for audit/reproducibility
    lines.append("### Methodology")
    lines.append(f"- **Engine Version**: `{QUANT_ENGINE_VERSION}`")
    lines.append("- **Altman Z'-Score (Z-Prime)**: Book-value variant (private firm model). Coefficients: 0.717×X1 + 0.847×X2 + 3.107×X3 + 0.420×X4 + 0.998×X5. Thresholds: Safe > 2.90, Grey 1.23–2.90, Distress < 1.23.")
    lines.append("- **Piotroski F-Score**: 9-point binary scoring across profitability (4), leverage/liquidity (3), and operating efficiency (2). Higher is stronger.")
    lines.append("- **Beneish M-Score**: 8-variable model (Beneish 1999): −4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI + 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI. Thresholds: M > −1.78 = High Risk, −2.22 to −1.78 = Grey Zone, M ≤ −2.22 = Low Risk. Components defaulted to neutral (1.0) when XBRL data is unavailable are flagged.")
    lines.append("- **Quick Ratio**: (Cash + Receivables) / Current Liabilities. Not computed if cash, receivables, or current liabilities are missing from the XBRL statement extraction.")
    lines.append("- **Cash Runway**: Cash & equivalents / annual operating cash burn, shown only when operating cash flow is negative. A business-model-agnostic solvency read for cash-consuming issuers; `N/A` means the company is operating-cash-flow positive.")
    lines.append(f"- **Business-Model Applicability**: archetype = `{business_model['archetype']}` (SIC `{business_model['sic']}`). Altman Z' and Beneish M assume a non-financial, revenue-generating operating company; they are suppressed for financial/real-estate issuers (SIC 6000–6799) and pre-revenue issuers, and flagged advisory for hypergrowth names, where the models read structurally misleading.")

    latest_derived = [d for d in derived_fields if d['year'] == latest_year]
    if latest_derived:
        derived_desc = "; ".join(f"{d['metric']} via {d['formula']}" for d in latest_derived)
        lines.append(f"- **Derived Balance-Sheet Inputs ({latest_year})**: {derived_desc}.")
    
    # Flag any imputed Beneish components for the latest year
    _, _, _, latest_imputed = beneish_results[latest_year]
    if latest_imputed:
        lines.append(f"- **⚠️ Beneish Data Gaps ({latest_year})**: {', '.join(latest_imputed)} — defaulted to neutral. M-Score may understate manipulation risk.")
    
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
        'cash_runway_years': runway_years.get(latest_year),
        'all_years': sorted_years,
        'quant_engine_version': QUANT_ENGINE_VERSION,
        'beneish_imputed_components': latest_imputed,
        'quick_ratio_missing_components': quick_ratio_missing.get(latest_year, []),
        'derived_balance_sheet_fields': latest_derived,
        'business_model': business_model,
        'business_model_archetype': business_model['archetype'],
    }
    
    return "\n".join(lines), metadata

import os
from datetime import datetime, timedelta
from edgar import Company

def get_insider_activity(company, filing_date, max_filings=40):
    """Fetch recent insider transactions (Form 4) and return a formatted summary.
    Only includes market buys/sells — filters out tax withholding, gifts, awards.
    Window: 6 months before filing date to present."""
    try:
        filings = company.get_filings(form='4')
        if not filings or len(filings) == 0:
            return ""
        
        # Parse filing_date for window calculation
        if isinstance(filing_date, str):
            ref_date = datetime.strptime(filing_date, '%Y-%m-%d').date()
        else:
            ref_date = filing_date
        
        window_start = ref_date - timedelta(days=180)
        
        transactions = []
        seen_transactions = set()
        processed = 0
        
        for f in filings:
            if processed >= max_filings:
                break
            if '/A' in f.form:
                continue
            
            # Only look at filings within our window
            f_date = f.filing_date
            if hasattr(f_date, 'date'):
                f_date = f_date.date()
            elif isinstance(f_date, str):
                f_date = datetime.strptime(f_date, '%Y-%m-%d').date()
            
            if f_date < window_start:
                break  # Filings are reverse chronological, so we can stop
            
            processed += 1
            
            try:
                obj = f.obj()
                df = obj.to_dataframe()
                if df is None or len(df) == 0:
                    continue
                
                # Check for 10b5-1 footnotes in the current filing
                is_10b51_plan = False
                if hasattr(obj, 'footnotes') and obj.footnotes:
                    try:
                        footnote_texts = []
                        if isinstance(obj.footnotes, dict):
                            footnote_texts = list(obj.footnotes.values())
                        elif hasattr(obj.footnotes, 'values'):
                            footnote_texts = list(obj.footnotes.values())
                        else:
                            footnote_texts = [str(obj.footnotes)]
                            
                        for text in footnote_texts:
                            if "10b5-1" in str(text).lower() or "10b51" in str(text).lower():
                                is_10b51_plan = True
                                break
                    except Exception:
                        pass
                
                for _, row in df.iterrows():
                    code = row.get('Code', '')
                    # P = Purchase, S = Sale (market transactions)
                    if code in ('P', 'S'):
                        txn_type = 'BUY' if code == 'P' else 'SELL'
                        shares = row.get('Shares', 0)
                        price = row.get('Price', 0)
                        value = row.get('Value', 0)
                        remaining = row.get('Remaining Shares', 'N/A')
                        txn_date = row.get('Date', f.filing_date)
                        
                        shares_num = int(shares) if shares else 0
                        price_num = float(price) if price else 0.0
                        value_num = float(value) if value else (shares_num * price_num)
                        
                        # Relativize position size %-wise
                        pct_str = "N/A"
                        try:
                            if remaining is not None and str(remaining) != 'N/A':
                                rem_shares = float(str(remaining).replace(',', ''))
                                total_shares = shares_num + rem_shares
                                if total_shares > 0:
                                    sold_pct = (shares_num / total_shares) * 100
                                    pct_str = f"{sold_pct:.2f}%"
                        except Exception:
                            pass

                        transaction_key = (
                            f.accession_no,
                            str(txn_date).split(' ')[0],
                            obj.insider_name,
                            obj.position,
                            txn_type,
                            shares_num,
                            round(price_num, 4),
                            round(value_num, 2),
                            str(remaining),
                            is_10b51_plan,
                        )
                        if transaction_key in seen_transactions:
                            continue
                        seen_transactions.add(transaction_key)
                        
                        transactions.append({
                            'date': str(txn_date).split(' ')[0],  # Strip time component
                            'insider': obj.insider_name,
                            'position': obj.position,
                            'type': txn_type,
                            'shares': shares_num,
                            'price': price_num,
                            'value': value_num,
                            'remaining': remaining,
                            'is_10b51': is_10b51_plan,
                            'pct_position': pct_str
                        })
            except Exception:
                continue
        
        if not transactions:
            return ""
        
        # Sort by date descending
        transactions.sort(key=lambda x: x['date'], reverse=True)
        
        # Compute summary stats
        disc_buys_val = sum(t['value'] for t in transactions if t['type'] == 'BUY')
        disc_sells_val = sum(t['value'] for t in transactions if t['type'] == 'SELL' and not t['is_10b51'])
        sched_sells_val = sum(t['value'] for t in transactions if t['type'] == 'SELL' and t['is_10b51'])
        
        buy_count = sum(1 for t in transactions if t['type'] == 'BUY')
        disc_sell_count = sum(1 for t in transactions if t['type'] == 'SELL' and not t['is_10b51'])
        sched_sell_count = sum(1 for t in transactions if t['type'] == 'SELL' and t['is_10b51'])
        
        unique_insiders = set(t['insider'] for t in transactions)
        
        # Format output
        lines = []
        lines.append(f"### INSIDER TRADING ACTIVITY (Form 4 — last 6 months relative to {ref_date})")
        lines.append(f"**Discretionary Activity:** {buy_count} buy(s) totaling **${disc_buys_val:,.0f}** | {disc_sell_count} discretionary sell(s) totaling **${disc_sells_val:,.0f}**")
        lines.append(f"**Pre-Scheduled Activity:** {sched_sell_count} 10b5-1 sell(s) totaling **${sched_sells_val:,.0f}**")
        lines.append(f"Unique active insiders: **{len(unique_insiders)}**")
        lines.append("")
        lines.append("Date | Insider | Position | Action | Shares | Price | Value | % Position | Type")
        lines.append("--- | --- | --- | --- | --- | --- | --- | --- | ---")
        
        for t in transactions:
            type_str = "10b5-1 (Scheduled)" if t['is_10b51'] else "Discretionary (Open Market)"
            lines.append(f"{t['date']} | {t['insider']} | {t['position']} | {t['type']} | {t['shares']:,} | ${t['price']:.2f} | ${t['value']:,.0f} | {t['pct_position']} | {type_str}")
        
        return "\n".join(lines)
    
    except Exception as e:
        print(f"Could not fetch insider activity: {e}")
        return ""

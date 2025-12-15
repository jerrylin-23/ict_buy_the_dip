#!/usr/bin/env python3
"""
AAPL Daily Backtest - Trailing Stop Strategy
~700 trades over 2 years with chart snapshots for key trades.
"""

import sys
import os
import warnings
import gc
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/jerry/Projects/ICT/src')

from datetime import datetime, timedelta
from ict_analyzer import analyze_stock

# Get symbol from CLI or default to NVDA
if len(sys.argv) > 1:
    SYMBOL = sys.argv[1].upper()
else:
    SYMBOL = "NVDA"

DAYS_AFTER = 30
INTERVAL_DAYS = 1  # DAILY
OUTPUT_DIR = f"/Users/jerry/Projects/ICT/src/samples/{SYMBOL.lower()}_daily_{datetime.now().strftime('%m%d_%H%M')}"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/charts", exist_ok=True)

# Generate dates - DAILY for 2 years
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

dates = []
current = start_date
while current < end_date - timedelta(days=DAYS_AFTER):
    dates.append(current.strftime('%Y-%m-%d'))
    current += timedelta(days=INTERVAL_DAYS)

print(f"📊 AAPL Daily Backtest: {len(dates)} trades")
print(f"📅 Range: {dates[0]} → {dates[-1]}")
print(f"📁 Output: {OUTPUT_DIR}\n")

results = []
charts_saved = 0
MAX_CHARTS = 500  # High limit to capture all trades
errors = 0
last_entry_price = None  # For deduplication

for i, backtest_date in enumerate(dates):
    try:
        outcome_date = (datetime.strptime(backtest_date, '%Y-%m-%d') + timedelta(days=DAYS_AFTER)).strftime('%Y-%m-%d')
        
        # Suppress output
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            analyzer = analyze_stock(SYMBOL, timeframe="4h", end_date=backtest_date)
            analyzer_after = analyze_stock(SYMBOL, timeframe="4h", end_date=outcome_date)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        
        if not analyzer or analyzer.df.empty or not analyzer_after or analyzer_after.df.empty:
            continue
        
        plan = analyzer.calculate_trade_plan()
        entry, sl = plan['entry'], plan['sl']
        tp1, tp2, tp3 = plan['tp1'], plan['tp2'], plan['tp3']
        
        # DEDUPLICATION: Skip if entry is within 0.5% of last trade
        if last_entry_price is not None:
            if abs(entry - last_entry_price) / last_entry_price < 0.005:
                continue  # Same setup, skip
        
        entry_time = analyzer.df.index[-1]
        outcome_df = analyzer_after.df[analyzer_after.df.index > entry_time]
        if outcome_df.empty:
            continue
        
        # Simple logic: Either hit SL or see how far TPs go
        sl_hit = tp1_hit = tp2_hit = tp3_hit = False
        exit_price = float(outcome_df['Close'].iloc[-1])
        exit_reason = "Timeout"
        
        for _, row in outcome_df.iterrows():
            low, high = row['Low'], row['High']
            
            # Track TPs reached
            if high >= tp1 and not tp1_hit:
                tp1_hit = True
            if high >= tp2 and not tp2_hit:
                tp2_hit = True
            if high >= tp3 and not tp3_hit:
                tp3_hit = True
                exit_price = tp3
                exit_reason = "TP3"
                break  # Full target hit, exit
            
            # Check SL (no trailing stop)
            if low <= sl and not sl_hit:
                sl_hit = True
                exit_price = sl
                exit_reason = "SL"
                break  # Stopped out
        
        # Determine final exit status
        if tp3_hit:
            exit_reason = "TP3"
            exit_price = tp3
        elif tp2_hit:
            exit_reason = "TP2"
            exit_price = tp2
        elif tp1_hit:
            exit_reason = "TP1"
            exit_price = tp1
        else:
            # No TP hit
            if sl_hit:
                exit_reason = "SL"
                exit_price = sl
            # else Timeout/Hold (already set)
        
        # Calculate P&L based on the determined exit price
        pnl = ((exit_price - entry) / entry) * 100
        win = tp1_hit or tp2_hit or tp3_hit
        
        result = {
            'date': backtest_date, 'entry': entry, 'exit': exit_price,
            'pnl': pnl, 'win': win, 'reason': exit_reason,
            'sl_hit': sl_hit,
            'tp1': tp1_hit, 'tp2': tp2_hit, 'tp3': tp3_hit
        }
        results.append(result)
        last_entry_price = entry  # Update for deduplication
        
        # Save charts for SL hits and TP3 wins
        save_chart = charts_saved < MAX_CHARTS and (tp3_hit or sl_hit)
        if save_chart:
            charts_saved += 1
            sys.stdout = open(os.devnull, 'w')
            try:
                before_html = analyzer.generate_chart_html()
                after_html = analyzer_after.generate_chart_html()
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout
            
            # Determine outcome label
            if win:
                if tp3_hit:
                    outcome_label = "✅ WIN - TP3 HIT"
                    outcome_color = "#4caf50"
                elif tp2_hit:
                    outcome_label = "✅ WIN - TP2"
                    outcome_color = "#4caf50"
                else:
                    outcome_label = "✅ WIN"
                    outcome_color = "#4caf50"
            else:
                outcome_label = "❌ LOSS - SL HIT"
                outcome_color = "#ef5350"
            
            # Combined HTML with stacked layout
            combined_html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{SYMBOL} {backtest_date} - {exit_reason}</title>
    <style>
        body {{ 
            background: #0a0a0a; 
            color: #d1d4dc; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; 
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid {outcome_color};
            margin-bottom: 20px;
        }}
        .header h1 {{
            color: {outcome_color};
            margin: 0 0 10px 0;
            font-size: 28px;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            font-size: 16px;
        }}
        .stat {{ padding: 5px 15px; background: #1a1a2e; border-radius: 5px; }}
        .panel {{
            margin-bottom: 30px;
            border: 1px solid #363c4e;
            border-radius: 8px;
            overflow: hidden;
        }}
        .panel-header {{
            background: #1a1a2e;
            padding: 10px 20px;
            font-size: 18px;
            font-weight: bold;
        }}
        .panel iframe {{
            width: 100%;
            height: 500px;
            border: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{outcome_label}</h1>
        <div class="stats">
            <div class="stat">📅 {backtest_date}</div>
            <div class="stat">💰 Entry: ${entry:.2f}</div>
            <div class="stat">🎯 Exit: ${exit_price:.2f}</div>
            <div class="stat" style="color: {"#4caf50" if pnl >= 0 else "#ef5350"}">P&L: {pnl:+.2f}%</div>
        </div>
    </div>
    
    <div class="panel">
        <div class="panel-header">📊 SETUP (Before Entry)</div>
        <iframe srcdoc="{before_html.replace('"', '&quot;')}"></iframe>
    </div>
    
    <div class="panel">
        <div class="panel-header">📈 OUTCOME (+{DAYS_AFTER} days)</div>
        <iframe srcdoc="{after_html.replace('"', '&quot;')}"></iframe>
    </div>
</body>
</html>'''
            
            # Single combined file with win/loss in filename
            status = "WIN" if win else "LOSS"
            filename = f"{charts_saved:02d}_{backtest_date}_{status}_{exit_reason}.html"
            with open(f"{OUTPUT_DIR}/charts/{filename}", 'w') as f:
                f.write(combined_html)
        
        # Progress every 50 trades
        if (i + 1) % 50 == 0:
            wins = sum(r['win'] for r in results)
            print(f"  [{i+1}/{len(dates)}] {len(results)} valid | Win: {wins}/{len(results)} | Charts: {charts_saved}")
            gc.collect()  # Free memory
        
        # Clean up to save memory
        del analyzer, analyzer_after, outcome_df
        gc.collect()
        
    except Exception as e:
        errors += 1
        if errors <= 5:
            print(f"⚠️ {backtest_date}: {str(e)[:40]}")

# Final Summary
print("\n" + "="*60)
total = len(results)
if total > 0:
    wins = sum(r['win'] for r in results)
    sl = sum(r['sl_hit'] for r in results)
    t1 = sum(r['tp1'] for r in results)
    t2 = sum(r['tp2'] for r in results)
    t3 = sum(r['tp3'] for r in results)
    avg = sum(r['pnl'] for r in results) / total
    
    print(f"📈 {SYMBOL} DAILY BACKTEST COMPLETE")
    print(f"   Trades: {total} | Win Rate: {wins/total*100:.1f}% | Avg P&L: {avg:+.2f}%")
    print(f"   SL: {sl} ({sl/total*100:.0f}%)")
    print(f"   TP1: {t1} ({t1/total*100:.0f}%) | TP2: {t2} ({t2/total*100:.0f}%) | TP3: {t3} ({t3/total*100:.0f}%)")
    print(f"   📊 Charts saved: {charts_saved}")
    print(f"   ⚠️ Errors: {errors}")

    # Save summary
    with open(f"{OUTPUT_DIR}/summary.txt", 'w') as f:
        f.write(f"{SYMBOL} Daily Backtest - No Trailing Stop\n{'='*50}\n")
        f.write(f"Trades: {total} | Win Rate: {wins/total*100:.1f}% | Avg P&L: {avg:+.2f}%\n")
        f.write(f"SL: {sl} | TP1: {t1} | TP2: {t2} | TP3: {t3}\n\n")
        f.write("TRADES:\n" + "-"*50 + "\n")
        for r in results:
            i = "✓" if r['win'] else "✗"
            f.write(f"{i} {r['date']}: ${r['entry']:.2f}→${r['exit']:.2f} ({r['pnl']:+.1f}%) {r['reason']}\n")
    
    print(f"\n📄 {OUTPUT_DIR}/summary.txt")
else:
    print("❌ No valid trades completed")

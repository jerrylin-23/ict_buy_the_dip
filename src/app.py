from flask import Flask, render_template, request, Response, jsonify
from ict_analyzer import analyze_stock
import pandas as pd
import threading
import time
from datetime import datetime

app = Flask(__name__)

# ============ Background Scanner Cache ============
SCAN_CACHE = {}  # {symbol: {price, entry, dist, setup, timestamp}}
SCAN_LOCK = threading.Lock()
SCAN_STATUS = {
    'last_scan': None,
    'next_scan': None,
    'is_scanning': False,
    'scan_count': 0
}
SCAN_INTERVAL = 300  # 5 minutes

# Default Watchlist (Fallback)
DEFAULT_WATCHLIST = [
    "SPY", "IWM", "QQQ", "DIA",
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL"
]

def load_watchlist():
    """Load tickers from tickers.txt or return default."""
    try:
        import os
        # Look in current directory or src/
        paths = ['tickers.txt', 'src/tickers.txt', '/Users/jerry/Projects/ICT/src/tickers.txt']
        for p in paths:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
                print(f"📋 Loaded {len(tickers)} tickers from {p}")
                return list(set(tickers)) # Deduplicate
    except Exception as e:
        print(f"⚠️ Error loading watchlist: {e}")
    
    print("⚠️ Using default watchlist")
    return DEFAULT_WATCHLIST

WATCHLIST = load_watchlist()


def run_scanner():
    """Background scanner function - runs silently."""
    global SCAN_CACHE, SCAN_STATUS, WATCHLIST
    
    # 1. Reload Watchlist (to pick up changes in tickers.txt)
    WATCHLIST = load_watchlist()
    
    with SCAN_LOCK:
        SCAN_STATUS['is_scanning'] = True
        
        # 2. Clear Stale Cache (Remove tickers not in new watchlist)
        current_tickers = set(WATCHLIST)
        cached_tickers = list(SCAN_CACHE.keys())
        for t in cached_tickers:
            if t not in current_tickers:
                del SCAN_CACHE[t]
                print(f"🧹 [Background] Removed stale ticker from cache: {t}")
    
    print(f"🔄 [Background] Starting scan on {len(WATCHLIST)} tickers...")
    new_results = {}
    
    for ticker in WATCHLIST:
        try:
            analyzer = analyze_stock(ticker, timeframe="4h")
            
            if analyzer and not analyzer.df.empty:
                plan = analyzer.calculate_trade_plan()
                current_price = analyzer.df['Close'].iloc[-1]
                entry_price = plan['entry']
                dist_pct = (current_price - entry_price) / current_price
                
                # Store all scanned tickers with their data
                result = {
                    'symbol': ticker,
                    'price': float(current_price),
                    'entry': float(entry_price),
                    'dist': float(dist_pct * 100),
                    'setup': plan['type'],
                    'timestamp': datetime.now().isoformat(),
                    'near_entry': bool(0 <= dist_pct <= 0.015)  # Flag if near entry (1.5%)
                }
                
                # Update cache immediately for this ticker
                with SCAN_LOCK:
                    SCAN_CACHE[ticker] = result
                    
        except Exception as e:
            print(f"  ⚠️ [Background] Error scanning {ticker}: {e}")
            
    # Final Status Update
    with SCAN_LOCK:
        SCAN_STATUS['last_scan'] = datetime.now().isoformat()
        SCAN_STATUS['is_scanning'] = False
        SCAN_STATUS['scan_count'] += 1
    
    print(f"✅ [Background] Scan complete. Updated cache.")


def scanner_thread():
    """Background thread that runs scanner periodically."""
    global SCAN_STATUS
    
    # Initial scan on startup (with small delay to let Flask start)
    time.sleep(2)
    run_scanner()
    
    while True:
        with SCAN_LOCK:
            SCAN_STATUS['next_scan'] = datetime.fromtimestamp(
                time.time() + SCAN_INTERVAL
            ).isoformat()
        
        time.sleep(SCAN_INTERVAL)
        run_scanner()


# Start background scanner thread
scanner_bg_thread = threading.Thread(target=scanner_thread, daemon=True)
scanner_bg_thread.start()
print("🚀 Background scanner thread started")


# ============ Routes ============

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    symbol = data.get('symbol', 'NVDA')
    
    try:
        print(f"Web Request: Analyzing {symbol}...")
        analyzer = analyze_stock(symbol, timeframe="4h")
        
        if analyzer and not analyzer.df.empty:
            html_content = analyzer.generate_chart_html()
            return Response(html_content, mimetype='text/html')
        else:
            return "Error: No data found or analysis failed", 400
            
    except Exception as e:
        return f"Server Error: {str(e)}", 500


@app.route('/scan', methods=['GET'])
def scan():
    """Return cached scan results instantly."""
    with SCAN_LOCK:
        # Filter to only show tickers near entry
        near_entry = [v for v in SCAN_CACHE.values() if v.get('near_entry', False)]
        return jsonify({
            'results': near_entry,
            'total_scanned': len(SCAN_CACHE),
            'last_scan': SCAN_STATUS['last_scan']
        })


@app.route('/scan/all', methods=['GET'])
def scan_all():
    """Return ALL cached results (not just near entry)."""
    with SCAN_LOCK:
        return jsonify({
            'results': list(SCAN_CACHE.values()),
            'last_scan': SCAN_STATUS['last_scan']
        })


@app.route('/scan/status', methods=['GET'])
def scan_status():
    """Return scanner status."""
    with SCAN_LOCK:
        return jsonify(SCAN_STATUS)


@app.route('/scan/force', methods=['POST'])
def scan_force():
    """Force immediate rescan in background."""
    # Run in new thread to not block response
    threading.Thread(target=run_scanner, daemon=True).start()
    return jsonify({'message': 'Rescan triggered', 'status': 'running'})


@app.route('/scan/clear', methods=['POST'])
def scan_clear():
    """Clear the scanner cache."""
    with SCAN_LOCK:
        SCAN_CACHE.clear()
    return jsonify({'message': 'Scanner cache cleared', 'status': 'success'})


@app.route('/backtest', methods=['POST'])
def backtest():
    """
    Generate before/after backtest visualization.
    Request: {symbol, date, days_after}
    Returns: HTML with side-by-side charts
    """
    data = request.json
    symbol = data.get('symbol', 'NVDA')
    backtest_date = data.get('date')  # e.g., "2024-10-15"
    days_after = data.get('days_after', 20)
    
    if not backtest_date:
        return jsonify({'error': 'date is required'}), 400
    
    try:
        from datetime import datetime, timedelta
        
        # Parse dates
        setup_date = datetime.strptime(backtest_date, '%Y-%m-%d')
        outcome_date = setup_date + timedelta(days=days_after)
        outcome_date_str = outcome_date.strftime('%Y-%m-%d')
        
        print(f"📊 Backtesting {symbol}: Setup={backtest_date}, Outcome={outcome_date_str}")
        
        # Generate "Before" chart (as of setup_date)
        analyzer_before = analyze_stock(symbol, timeframe="4h", end_date=backtest_date)
        if not analyzer_before or analyzer_before.df.empty:
            return jsonify({'error': 'No data for setup date'}), 400
        chart_before = analyzer_before.generate_chart_html()
        
        # Generate "After" chart (extended by days_after)
        analyzer_after = analyze_stock(symbol, timeframe="4h", end_date=outcome_date_str)
        if not analyzer_after or analyzer_after.df.empty:
            return jsonify({'error': 'No data for outcome date'}), 400
        chart_after = analyzer_after.generate_chart_html()
        
        # Get trade plan from "before" for comparison
        plan = analyzer_before.calculate_trade_plan()
        outcome_price = analyzer_after.df['Close'].iloc[-1]
        entry_price = plan['entry']
        
        # Calculate P&L
        if outcome_price > entry_price:
            pnl_pct = ((outcome_price - entry_price) / entry_price) * 100
            pnl_class = "profit"
        else:
            pnl_pct = ((outcome_price - entry_price) / entry_price) * 100
            pnl_class = "loss"
        
        # Generate combined HTML with side-by-side iframes
        combined_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Backtest: {symbol} - {backtest_date}</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .summary {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 15px 0;
            font-size: 14px;
        }}
        .summary .profit {{ color: #26a69a; }}
        .summary .loss {{ color: #ef5350; }}
        .container {{
            display: flex;
            gap: 20px;
            height: calc(100vh - 150px);
        }}
        .panel {{
            flex: 1;
            display: flex;
            flex-direction: column;
        }}
        .panel h3 {{
            margin: 0 0 10px 0;
            padding: 10px;
            background: #16213e;
            border-radius: 8px 8px 0 0;
            text-align: center;
        }}
        .panel iframe {{
            flex: 1;
            border: none;
            border-radius: 0 0 8px 8px;
        }}
        .before h3 {{ border-left: 4px solid #ff9800; }}
        .after h3 {{ border-left: 4px solid #26a69a; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Backtest: {symbol}</h1>
        <div class="summary">
            <span>Setup: {backtest_date}</span>
            <span>Entry: ${entry_price:.2f}</span>
            <span>Outcome (+{days_after}d): ${outcome_price:.2f}</span>
            <span class="{pnl_class}">P&L: {pnl_pct:+.2f}%</span>
        </div>
    </div>
    <div class="container">
        <div class="panel before">
            <h3>📅 Setup: {backtest_date}</h3>
            <iframe srcdoc="{chart_before.replace('"', '&quot;')}"></iframe>
        </div>
        <div class="panel after">
            <h3>📈 Outcome: +{days_after} days</h3>
            <iframe srcdoc="{chart_after.replace('"', '&quot;')}"></iframe>
        </div>
    </div>
</body>
</html>
'''
        return Response(combined_html, mimetype='text/html')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)  # Disable reloader to avoid double threads

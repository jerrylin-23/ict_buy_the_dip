from flask import Flask, render_template, request, Response
from ict_analyzer import analyze_stock
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    symbol = data.get('symbol', 'NVDA')
    
    try:
        # Run Analysis using existing function
        print(f"Web Request: Analyzing {symbol}...")
        analyzer = analyze_stock(symbol, timeframe="4h")
        
        if analyzer and not analyzer.df.empty:
            # Generate Chart HTML
            html_content = analyzer.generate_chart_html()
            return Response(html_content, mimetype='text/html')
        else:
            return "Error: No data found or analysis failed", 400
            
    except Exception as e:
        return f"Server Error: {str(e)}", 500

@app.route('/scan', methods=['GET'])
def scan():
    """Scan a list of tickers for setups near entry."""
    # Default Watchlist (IWM/SPY focus as requested)
    tickers = [
        "SPY", "IWM", "QQQ", "DIA", # Indices
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AMD", # Mag 7 + Tech
        "JPM", "BAC", "WFC", # Banks
        "XOM", "CVX", # Energy
        "LLY", "UNH" # Healthcare
    ]
    
    results = []
    print(f"🔄 Starting Scan on {len(tickers)} tickers...")
    
    for ticker in tickers:
        try:
            print(f"  Scanning {ticker}...")
            analyzer = analyze_stock(ticker, timeframe="4h")
            
            if analyzer and not analyzer.df.empty:
                plan = analyzer.calculate_trade_plan()
                current_price = analyzer.df['Close'].iloc[-1]
                entry_price = plan['entry']
                
                # Calculate distance to entry
                # We want price to be ABOVE entry but close (Pullback mode)
                # or just generally close.
                
                dist_pct = (current_price - entry_price) / current_price
                
                # logic: 1-3% away
                # If dist_pct is negative, price is BELOW entry (Triggered/Invalid)
                # If dist_pct is 0.01 to 0.03, it's approaching.
                
                # Let's be slightly flexible: 0% to 3%
                if 0 <= dist_pct <= 0.035:
                    results.append({
                        'symbol': ticker,
                        'price': current_price,
                        'entry': entry_price,
                        'dist': dist_pct * 100,
                        'setup': plan['type']
                    })
                    
        except Exception as e:
            print(f"  ❌ Error scanning {ticker}: {e}")
            
    return {'results': results}

if __name__ == '__main__':
    app.run(debug=True, port=5000)

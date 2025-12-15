# ICT Buy the Dip Analyzer

A Python-based algorithmic trading tool that detects **ICT concepts** like Fair Value Gaps (FVGs), Equal Highs/Lows, and Swing Points to generate actionable trade plans.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20UI-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

### 📊 Core Analysis
- **Fair Value Gap (FVG) Detection**: Identifies bullish and bearish imbalances.
- **Equal Highs/Lows (EQH/EQL)**: Detects liquidity zones and key support/resistance.
- **Swing Point Detection**: Finds market structure highs and lows.
- **Automatic Trade Plans**: Generates Entry, Stop Loss, and 3 Take Profit levels.

### 🌐 Web Interface
- **Real-time Market Scanner**: Monitors 500+ tickers from S&P 500 and Nasdaq 100.
- **Interactive Charts**: TradingView Lightweight Charts with 2 years of history.
- **Live Results**: Streaming updates as each ticker is scanned.
- **Proximity Filter**: Only shows tickers within 1.5% of calculated entry.

### 📈 Backtesting
- **Historical Signal Scanner**: Tests strategy across 700+ trading days.
- **Performance Metrics**: Win rate, P&L, TP1/TP2/TP3 hit rates.
- **Chart Snapshots**: Generates HTML charts for every trade found.

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/jerrylin-23/ict_buy_the_dip.git
cd ict_buy_the_dip

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Web Scanner
```bash
source venv/bin/activate
python src/app.py
```
Open **http://127.0.0.1:5001** in your browser.

### 3. Run Backtest
```bash
source venv/bin/activate
python src/run_backtest.py NVDA   # or AAPL, GOOGL, SPY, etc.
```
Results saved to `src/samples/<symbol>_daily_<date>/`

---

## 📁 Project Structure

```
ICT/
├── src/
│   ├── app.py              # Flask web application
│   ├── ict_analyzer.py     # Core ICT analysis logic
│   ├── run_backtest.py     # Backtesting script
│   ├── tickers.txt         # Watchlist (S&P 500 + Nasdaq 100)
│   ├── templates/          # HTML templates
│   └── samples/            # Backtest output (charts & summaries)
├── requirements.txt        # Python dependencies
├── generate_tickers.py     # Script to regenerate watchlist
└── README.md
```

---

## 📊 Backtest Results (Dec 2024)

| Symbol | Trades | Win Rate | Avg P&L | TP1 Hit | TP2 Hit | TP3 Hit |
|--------|--------|----------|---------|---------|---------|---------|
| NVDA   | 286    | **74.5%** | +4.63%  | 74%     | 60%     | 49%     |
| GOOGL  | 248    | **77.0%** | +2.68%  | 77%     | 54%     | 42%     |
| AAPL   | 221    | **74.7%** | +2.38%  | 75%     | 60%     | 43%     |

### 📸 Example Charts

View sample trade snapshots from the backtest (click to view live):

| Trade | Outcome | Chart |
|-------|---------|-------|
| NVDA 2023-12-20 | ✅ WIN (TP3) | [View Chart](https://jerrylin-23.github.io/ict_buy_the_dip/src/samples/nvda_daily_1214_2105/charts/04_2023-12-20_WIN_TP3.html) |
| NVDA 2024-03-08 | ✅ WIN (TP1) | [View Chart](https://jerrylin-23.github.io/ict_buy_the_dip/src/samples/nvda_daily_1214_2105/charts/35_2024-03-08_WIN_TP1.html) |
| NVDA 2024-04-11 | ❌ LOSS (SL) | [View Chart](https://jerrylin-23.github.io/ict_buy_the_dip/src/samples/nvda_daily_1214_2105/charts/44_2024-04-11_LOSS_SL.html) |

> 📁 **Browse all 700+ charts**: [src/samples/](src/samples/) - includes NVDA, GOOGL, and AAPL backtests

---

## 🛠️ Configuration

### Watchlist
Edit `src/tickers.txt` to customize the scanner watchlist:
```
AAPL
NVDA
TSLA
# Add your tickers here
```

### Scanner Sensitivity
In `src/app.py`, adjust the `near_entry` threshold (default: 1.5%):
```python
'near_entry': bool(0 <= dist_pct <= 0.015)
```

---

## 📋 Requirements

- Python 3.10+
- Flask
- yfinance
- pandas
- numpy

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**. It is not financial advice. Always do your own research and consult with a licensed financial advisor before making investment decisions.

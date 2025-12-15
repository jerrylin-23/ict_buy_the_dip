"""
ICT Buy-the-Dip Analyzer
Detects order blocks, fair value gaps, and key levels for high-probability entries.
Timeframes: 4H and Daily
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime, timedelta


class ICTAnalyzer:
    """Implements ICT concepts for buy-the-dip detection."""
    
    def __init__(self, symbol: str, timeframe: str = "4h"):
        """
        Initialize the analyzer.
        
        Args:
            symbol: Stock ticker (e.g., "AAPL", "SPY")
            timeframe: "4h" or "1d"
        """
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.df = None
        
    def fetch_data(self, period: str = "2y") -> pd.DataFrame:
        """Fetch OHLCV data from yfinance."""
        interval = "1h" if self.timeframe == "4h" else "1d"
        
        self.df = yf.download(
            self.symbol, 
            period=period, 
            interval=interval,
            progress=False,
            threads=False  # Disable threading to avoid Flask/scanner conflicts
        )
        
        # Flatten column names if MultiIndex (yfinance sometimes returns MultiIndex)
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)
        
        # Resample to 4H if needed
        if self.timeframe == "4h" and interval == "1h":
            self.df = self.df.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
        print(f"✅ Fetched {len(self.df)} candles for {self.symbol} ({self.timeframe})")
        return self.df
    
    def detect_swing_points(self, lookback: int = 3) -> pd.DataFrame:
        """
        Detect swing highs and swing lows.
        
        A swing high: High is the highest of surrounding candles
        A swing low: Low is the lowest of surrounding candles
        """
        df = self.df.copy()
        
        df['swing_high'] = False
        df['swing_low'] = False
        
        for i in range(lookback, len(df) - lookback):
            # Swing High
            if df['High'].iloc[i] == df['High'].iloc[i-lookback:i+lookback+1].max():
                df.iloc[i, df.columns.get_loc('swing_high')] = True
            # Swing Low
            if df['Low'].iloc[i] == df['Low'].iloc[i-lookback:i+lookback+1].min():
                df.iloc[i, df.columns.get_loc('swing_low')] = True
        
        self.df = df
        swing_highs = df[df['swing_high']].shape[0]
        swing_lows = df[df['swing_low']].shape[0]
        print(f"📍 Found {swing_highs} swing highs, {swing_lows} swing lows")
        return df
    
    def detect_order_blocks(self, min_move_pct: float = 1.0) -> list:
        """
        Detect bullish and bearish order blocks.
        
        Bullish OB: Last down candle before a strong up move
        Bearish OB: Last up candle before a strong down move
        """
        order_blocks = []
        df = self.df
        
        for i in range(2, len(df) - 1):
            # Current candle
            curr_open = df['Open'].iloc[i]
            curr_close = df['Close'].iloc[i]
            curr_high = df['High'].iloc[i]
            curr_low = df['Low'].iloc[i]
            
            # Previous candle
            prev_open = df['Open'].iloc[i-1]
            prev_close = df['Close'].iloc[i-1]
            prev_high = df['High'].iloc[i-1]
            prev_low = df['Low'].iloc[i-1]
            
            # Next candle (to confirm move)
            next_close = df['Close'].iloc[i+1]
            
            # Calculate move percentage
            move_pct = abs(next_close - curr_close) / curr_close * 100
            
            # Bullish Order Block
            # Previous candle is bearish, followed by strong bullish move
            if prev_close < prev_open:  # Bearish candle
                if next_close > curr_high and move_pct >= min_move_pct:
                    order_blocks.append({
                        'type': 'bullish',
                        'date': df.index[i-1],
                        'high': prev_high,
                        'low': prev_low,
                        'strength': move_pct
                    })
            
            # Bearish Order Block  
            # Previous candle is bullish, followed by strong bearish move
            if prev_close > prev_open:  # Bullish candle
                if next_close < curr_low and move_pct >= min_move_pct:
                    order_blocks.append({
                        'type': 'bearish',
                        'date': df.index[i-1],
                        'high': prev_high,
                        'low': prev_low,
                        'strength': move_pct
                    })
        
        self.order_blocks = order_blocks
        bullish = len([ob for ob in order_blocks if ob['type'] == 'bullish'])
        bearish = len([ob for ob in order_blocks if ob['type'] == 'bearish'])
        print(f"🟩 Found {bullish} bullish OBs, 🟥 {bearish} bearish OBs")
        return order_blocks
    
    def detect_fair_value_gaps(self, min_gap_pct: float = 0.5) -> list:
        """
        Detect Fair Value Gaps (FVGs) - Basic Structural Logic.
        
        Bullish FVG: Gap between candle 1's high and candle 3's low
        Bearish FVG: Gap between candle 1's low and candle 3's high
        """
        fvgs = []
        df = self.df
        
        for i in range(2, len(df)):
            # Three-candle pattern
            # i = Candle 3 (Current)
            # i-2 = Candle 1 (Left)
            
            candle1_high = df['High'].iloc[i-2]
            candle1_low = df['Low'].iloc[i-2]
            
            candle3_high = df['High'].iloc[i]
            candle3_low = df['Low'].iloc[i]
            
            # --- Bullish FVG ---
            # Condition: High[i-2] < Low[i] (Gap Up)
            if candle1_high < candle3_low:
                gap_size = (candle3_low - candle1_high) / candle1_high * 100
                if gap_size >= min_gap_pct:
                    # Check filled
                    filled = False
                    for j in range(i+1, len(df)):
                        if df['Low'].iloc[j] <= candle1_high:
                            filled = True
                            break
                            
                    fvgs.append({
                        'type': 'bullish',
                        'date': df.index[i-1], # Use middle candle date
                        'top': candle3_low,
                        'bottom': candle1_high,
                        'gap_pct': gap_size,
                        'filled': filled
                    })

            # --- Bearish FVG ---
            # Condition: Low[i-2] > High[i] (Gap Down)
            if candle1_low > candle3_high:
                gap_size = (candle1_low - candle3_high) / candle1_low * 100
                if gap_size >= min_gap_pct:
                    # Check filled
                    filled = False
                    for j in range(i+1, len(df)):
                        if df['High'].iloc[j] >= candle1_low:
                            filled = True
                            break
                            
                    fvgs.append({
                        'type': 'bearish',
                        'date': df.index[i-1],
                        'top': candle1_low,
                        'bottom': candle3_high,
                        'gap_pct': gap_size,
                        'filled': filled
                    })
        
        self.fvgs = fvgs
        bullish = len([f for f in fvgs if f['type'] == 'bullish'])
        bearish = len([f for f in fvgs if f['type'] == 'bearish'])
        unfilled = len([f for f in fvgs if not f.get('filled', False)])
        print(f"📊 Found {bullish} bullish FVGs, {bearish} bearish FVGs ({unfilled} unfilled)")
        return fvgs
    
    def detect_equal_highs_lows(self, threshold_pct: float = 1.0) -> dict:
        """
        Detect Relative Equal Highs (Resistance/Liquidity) and Relative Equal Lows (Support).
        """
        df = self.df
        swing_highs = df[df['swing_high']]
        swing_lows = df[df['swing_low']]
        
        eqh = []
        eql = []
        
        # Check highs
        # Simple clustering: if multiple swing points are within threshold, group them
        sorted_highs = swing_highs.sort_values('High', ascending=False)
        processed_highs = set()
        
        for idx, row in sorted_highs.iterrows():
            if idx in processed_highs:
                continue
            
            # Find close highs
            price = row['High']
            lower_bound = price * (1 - threshold_pct/100)
            upper_bound = price * (1 + threshold_pct/100)
            
            matches = sorted_highs[
                (sorted_highs['High'] >= lower_bound) & 
                (sorted_highs['High'] <= upper_bound) &
                (~sorted_highs.index.isin(processed_highs))
            ]
            
            if len(matches) > 1: # Found relative equal highs
                avg_price = matches['High'].mean()
                eqh.append({
                    'price': avg_price,
                    'count': len(matches),
                    'indices': matches.index.tolist()
                })
                processed_highs.update(matches.index)
        
        # Check lows
        sorted_lows = swing_lows.sort_values('Low')
        processed_lows = set()
        
        for idx, row in sorted_lows.iterrows():
            if idx in processed_lows:
                continue
            
            price = row['Low']
            lower_bound = price * (1 - threshold_pct/100)
            upper_bound = price * (1 + threshold_pct/100)
            
            matches = sorted_lows[
                (sorted_lows['Low'] >= lower_bound) & 
                (sorted_lows['Low'] <= upper_bound) &
                (~sorted_lows.index.isin(processed_lows))
            ]
            
            if len(matches) > 1: # Found relative equal lows
                avg_price = matches['Low'].mean()
                eql.append({
                    'price': avg_price,
                    'count': len(matches),
                    'indices': matches.index.tolist()
                })
                processed_lows.update(matches.index)
                
        self.eq_levels = {'highs': eqh, 'lows': eql}
        print(f"⚖️ Found {len(eqh)} EQH zones, {len(eql)} EQL zones")
        return self.eq_levels

    def get_last_earnings_date(self):
        """
        Get the most recent past earnings ANNOUNCEMENT date for a stock.
        Returns None if not available or if stock is an ETF.
        """
        try:
            ticker = yf.Ticker(self.symbol)
            today = pd.Timestamp.now(tz='America/New_York').normalize()
            
            # Primary: Use earnings_dates (actual announcement dates with timezone)
            try:
                ed = ticker.earnings_dates
                if ed is not None and hasattr(ed, 'index') and len(ed.index) > 0:
                    # Index contains actual earnings announcement dates (e.g., 2025-11-19)
                    announcement_dates = pd.to_datetime(ed.index)
                    # Make timezone-aware comparison work
                    past_dates = []
                    for d in announcement_dates:
                        if d.tzinfo is not None:
                            d_compare = d.tz_convert('America/New_York')
                        else:
                            d_compare = d.tz_localize('America/New_York')
                        if d_compare < today:
                            past_dates.append(d)
                    
                    if past_dates:
                        last_earnings = max(past_dates)
                        # Convert to date string without timezone for display
                        print(f"📅 Last earnings: {last_earnings.strftime('%Y-%m-%d')}")
                        return last_earnings
            except Exception as e:
                print(f"⚠️ earnings_dates error: {e}")
                pass
            
            # Secondary: Use earnings_history (fiscal quarter end dates - less accurate)
            try:
                eh = ticker.earnings_history
                if eh is not None and hasattr(eh, 'index') and len(eh.index) > 0:
                    quarter_dates = pd.to_datetime(eh.index)
                    today_naive = pd.Timestamp.now().normalize()
                    past_quarters = [d for d in quarter_dates if d < today_naive]
                    if past_quarters:
                        last_earnings = max(past_quarters)
                        print(f"📅 Last earnings (quarter): {last_earnings.strftime('%Y-%m-%d')}")
                        return last_earnings
            except Exception:
                pass
            
            # Tertiary: Infer from calendar future dates
            try:
                calendar = ticker.calendar
                if isinstance(calendar, dict) and 'Earnings Date' in calendar:
                    ed_val = calendar['Earnings Date']
                    if isinstance(ed_val, (list, tuple)):
                        next_date = pd.to_datetime(ed_val[0])
                    else:
                        next_date = pd.to_datetime(ed_val)
                    # Subtract ~90 days
                    last_earnings = next_date - pd.Timedelta(days=90)
                    print(f"📅 Last earnings (inferred): {last_earnings.strftime('%Y-%m-%d')}")
                    return last_earnings
            except Exception:
                pass
            
            print(f"⚠️ No earnings data for {self.symbol} (likely an ETF)")
            return None
            
        except Exception as e:
            print(f"⚠️ Error fetching earnings date: {e}")
            return None

    def calculate_earnings_vwap(self):
        """
        Calculate VWAP anchored from the most recent earnings date.
        Returns list of {time, value} for chart line series.
        """
        earnings_date = self.get_last_earnings_date()
        
        if earnings_date is None:
            self.earnings_vwap = None
            self.earnings_vwap_date = None
            return None
        
        # Filter data from earnings date onward
        df = self.df.copy()
        
        # Handle timezone: match earnings_date to df.index timezone
        if df.index.tz is not None:
            if earnings_date.tzinfo is None:
                earnings_date = earnings_date.tz_localize(df.index.tz)
            else:
                earnings_date = earnings_date.tz_convert(df.index.tz)
        
        df_since_earnings = df[df.index >= earnings_date]
        
        if len(df_since_earnings) < 2:
            print(f"⚠️ Not enough data since earnings date")
            self.earnings_vwap = None
            self.earnings_vwap_date = None
            return None
        
        # Calculate VWAP: cumulative(typical_price * volume) / cumulative(volume)
        typical_price = (df_since_earnings['High'] + df_since_earnings['Low'] + df_since_earnings['Close']) / 3
        cumulative_tp_vol = (typical_price * df_since_earnings['Volume']).cumsum()
        cumulative_vol = df_since_earnings['Volume'].cumsum()
        
        vwap = cumulative_tp_vol / cumulative_vol
        
        # Prepare line data for chart
        vwap_data = []
        for idx, val in vwap.items():
            if pd.notna(val):
                vwap_data.append({
                    'time': int(idx.timestamp()),
                    'value': round(val, 2)
                })
        
        self.earnings_vwap = vwap_data
        self.earnings_vwap_date = earnings_date
        self.earnings_vwap_current = vwap.iloc[-1] if len(vwap) > 0 else None
        
        print(f"📈 ER VWAP calculated: ${self.earnings_vwap_current:.2f} (from {earnings_date.strftime('%Y-%m-%d')})")
        return vwap_data

    def generate_analysis_summary(self) -> str:
        """Generate a text summary of key levels relative to current price."""
        current_price = self.df['Close'].iloc[-1]
        summary = []
        
        # 1. Resistance (Above)
        # Find closest EQH above
        resistances = [l for l in self.eq_levels.get('highs', []) if l['price'] > current_price]
        if resistances:
            closest_r = min(resistances, key=lambda x: x['price'])
            summary.append(f"🔴 RESISTANCE (EQH) @ ${closest_r['price']:.2f}")
        else:
            # Fallback to recent swing high
            recent_highs = self.df[self.df['swing_high']].tail(5)
            highs_above = recent_highs[recent_highs['High'] > current_price]
            if not highs_above.empty:
                r = highs_above.iloc[-1]['High']
                summary.append(f"🔴 RESISTANCE (Swing) @ ${r:.2f}")

        # 2. Bearish Gaps (Above)
        bear_fvgs = [f for f in self.fvgs if f['type'] == 'bearish' and not f.get('filled', False) and f['bottom'] > current_price]
        if bear_fvgs:
            closest_gap = min(bear_fvgs, key=lambda x: x['bottom'])
            mid = (closest_gap['top'] + closest_gap['bottom']) / 2
            summary.append(f"⚠️ BEAR GAP above @ ${mid:.2f}")

        # 3. Bullish Gaps (Below)
        bull_fvgs = [f for f in self.fvgs if f['type'] == 'bullish' and not f.get('filled', False) and f['top'] < current_price]
        if bull_fvgs:
            closest_gap = max(bull_fvgs, key=lambda x: x['top'])
            mid = (closest_gap['top'] + closest_gap['bottom']) / 2
            summary.append(f"✅ BULL GAP below @ ${mid:.2f}")

        # 4. Support (Below)
        # Find closest EQL below
        supports = [l for l in self.eq_levels.get('lows', []) if l['price'] < current_price]
        if supports:
            closest_s = max(supports, key=lambda x: x['price'])
            summary.append(f"🟢 SUPPORT (EQL) @ ${closest_s['price']:.2f}")
        else:
             # Fallback to recent swing low
            recent_lows = self.df[self.df['swing_low']].tail(5)
            lows_below = recent_lows[recent_lows['Low'] < current_price]
            if not lows_below.empty:
                s = lows_below.iloc[-1]['Low']
                summary.append(f"🟢 SUPPORT (Swing) @ ${s:.2f}")
        
        # 5. Earnings VWAP
        if hasattr(self, 'earnings_vwap_current') and self.earnings_vwap_current is not None:
            er_date = self.earnings_vwap_date.strftime('%m/%d') if self.earnings_vwap_date else ''
            summary.append(f"📊 ER VWAP @ ${self.earnings_vwap_current:.2f} ({er_date})")
                
        return "<br>".join(summary)

    def calculate_trade_plan(self):
        """Calculate Bullish Trade Setup (Entry, SL, TP1-3)."""
        current_price = self.df['Close'].iloc[-1]
        
        # 1. Determine Entry
        # Criteria: Closest Support below price (FVG Top or EQL)
        entry_candidates = []
        
        # Candidate A: Bullish FVGs (Top)
        bull_fvgs = [f for f in getattr(self, 'fvgs', []) 
                     if f['type'] == 'bullish' and not f.get('filled', False) 
                     and f['top'] < current_price]
        for f in bull_fvgs:
            entry_candidates.append({
                'price': f['top'], 
                'type': 'FVG',
                'bottom': f['bottom'] # Relevant for SL
            })
            
        # Candidate B: EQLs (Price)
        eqls = [l for l in self.eq_levels.get('lows', []) if l['price'] < current_price]
        for l in eqls:
            entry_candidates.append({
                'price': l['price'], 
                'type': 'EQL',
                'bottom': l['price'] # SL relative to line
            })
            
        # select best
        entry_price = current_price 
        setup_type = "Aggressive (Market)"
        entry_obj = None
        
        if entry_candidates:
            # Closest to current price (max price < current)
            entry_obj = max(entry_candidates, key=lambda x: x['price'])
            entry_price = entry_obj['price']
            setup_type = f"Limit Buy ({entry_obj['type']})"
            
        # 2. Determine Stop Loss
        sl_price = entry_price * 0.98 # Default 2%
        
        if entry_obj:
            if entry_obj['type'] == 'FVG':
                sl_price = entry_obj['bottom'] * 0.99 # 1% Buffer for Liq Sweeps
            else:
                sl_price = entry_obj['price'] * 0.99 # 1% Buffer for Liq Sweeps
        
        # Ensure Min Risk (0.5%)
        if (entry_price - sl_price) / entry_price < 0.005:
            sl_price = entry_price * 0.99
            
        # 3. Determine Take Profits
        eqh_levels = sorted([l['price'] for l in self.eq_levels.get('highs', []) if l['price'] > entry_price])
        
        tp1 = eqh_levels[0] if len(eqh_levels) > 0 else entry_price * 1.02
        tp2 = eqh_levels[1] if len(eqh_levels) > 1 else (entry_price + (entry_price - sl_price) * 2)
        
        # TP3: Moonbag / 1:3 AR
        risk = entry_price - sl_price
        tp3 = entry_price + (risk * 3)
        
        # Verify TPs are ascending
        tps = sorted(list(set([tp1, tp2, tp3])))
        if len(tps) >= 3:
            tp1, tp2, tp3 = tps[0], tps[1], tps[2]
            
        return {
            'type': setup_type,
            'entry': entry_price,
            'sl': sl_price,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3
        }

    def generate_chart_html(self, show_fvgs: bool = True, max_fvgs: int = 2, max_levels: int = 2) -> str:
        """
        Generate interactive TradingView Lightweight Chart (HTML).
        Includes side panel for analysis summary and Trade Plan.
        """
        df = self.df.tail(1500).copy() # Last 1500 candles (~2y data loaded)
        
        # Calculate Trade Plan
        plan = self.calculate_trade_plan()
        
        # Prepare Candle Data
        candle_data = []
        
        for idx, row in df.iterrows():
            # Convert timestamp to unix seconds
            t = int(idx.timestamp())
            
            candle_data.append({
                'time': t,
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close']
            })
            
        # Prepare FVG Lines (with start time from formation candle)
        fvg_lines = []
        current_price = df['Close'].iloc[-1]
        last_candle_time = int(df.index[-1].timestamp())
        threshold_pct = 0.03 # Only show levels within 3% of current price
        
        if show_fvgs and hasattr(self, 'fvgs'):
            # Only show unfilled and visible
            recent_fvgs = [f for f in self.fvgs if not f.get('filled', False)]
            
            # Filter by relevance (distance < 3%)
            relevant_fvgs = [
                f for f in recent_fvgs 
                if abs(((f['top'] + f['bottom'])/2) - current_price) / current_price <= threshold_pct
            ]
            
            # Sort by distance
            relevant_fvgs.sort(key=lambda x: abs(((x['top'] + x['bottom'])/2) - current_price))
            display_fvgs = relevant_fvgs[:max_fvgs] # Use argument limit
            
            for f in display_fvgs:
                midpoint = (f['top'] + f['bottom']) / 2
                color = '#ffeb3b' if f['type'] == 'bullish' else '#ff9800'
                title = "BULL GAP" if f['type'] == 'bullish' else "BEAR GAP"
                # Get start time from FVG date
                start_time = int(f['date'].timestamp())
                
                fvg_lines.append({
                    'price': midpoint,
                    'color': color,
                    'title': title,
                    'start_time': start_time,
                    'end_time': last_candle_time
                })
                
        # Prepare EQH/EQL Levels (Filtered by distance but ensure closest are shown)
        level_lines = []
        all_levels = []
        if hasattr(self, 'eq_levels'):
            for eqh in self.eq_levels.get('highs', []):
                all_levels.append({'price': eqh['price'], 'type': 'EQH', 'color': '#ef5350'})
            for eql in self.eq_levels.get('lows', []):
                all_levels.append({'price': eql['price'], 'type': 'EQL', 'color': '#26a69a'})
        
        # 1. Filter: EQH must be ABOVE, EQL must be BELOW current price
        valid_eqh = [l for l in all_levels if l['type'] == 'EQH' and l['price'] > current_price]
        valid_eql = [l for l in all_levels if l['type'] == 'EQL' and l['price'] < current_price]
        
        # Sort by distance
        valid_eqh.sort(key=lambda x: abs(x['price'] - current_price))
        valid_eql.sort(key=lambda x: abs(x['price'] - current_price))
        
        forced_levels = []
        
        # Ensure at least one Resistance (EQH or Swing High)
        if valid_eqh:
            forced_levels.append(valid_eqh[0])
        else:
            # Fallback to nearest Swing High above price
            swing_highs = self.df[self.df['swing_high']]
            highs_above = swing_highs[swing_highs['High'] > current_price]
            if not highs_above.empty:
                nearest_high = highs_above.iloc[(highs_above['High'] - current_price).abs().argsort()[:1]]
                forced_levels.append({
                    'price': nearest_high['High'].iloc[0], 
                    'type': 'Swing High', 
                    'color': '#ef5350'
                })
            else:
                # Secondary Fallback: Any recent high in last 300 candles
                recent_window = self.df.iloc[-300:]
                recent_highs = recent_window[recent_window['High'] > current_price]
                if not recent_highs.empty:
                     highest_recent = recent_highs.iloc[(recent_highs['High'] - current_price).abs().argsort()[:1]]
                     forced_levels.append({
                        'price': highest_recent['High'].iloc[0],
                        'type': 'Recent High',
                        'color': '#ef5350'
                     })

        # Ensure at least one Support (EQL or Swing Low)
        if valid_eql:
            forced_levels.append(valid_eql[0])
        else:
            # Fallback to nearest Swing Low below price
            swing_lows = self.df[self.df['swing_low']]
            lows_below = swing_lows[swing_lows['Low'] < current_price]
            if not lows_below.empty:
                nearest_low = lows_below.iloc[(current_price - lows_below['Low']).abs().argsort()[:1]]
                forced_levels.append({
                    'price': nearest_low['Low'].iloc[0], 
                    'type': 'Swing Low', 
                    'color': '#26a69a'
                })
            else:
                # Secondary Fallback: Any recent low in last 300 candles
                recent_window = self.df.iloc[-300:]
                recent_lows = recent_window[recent_window['Low'] < current_price]
                if not recent_lows.empty:
                     lowest_recent = recent_lows.iloc[(current_price - recent_lows['Low']).abs().argsort()[:1]]
                     forced_levels.append({
                        'price': lowest_recent['Low'].iloc[0],
                        'type': 'Recent Low',
                        'color': '#26a69a'
                     })
        
        final_levels_pool = []
        seen_prices = set()
        
        # Add forced first
        for l in forced_levels:
            if l['price'] not in seen_prices:
                final_levels_pool.append(l)
                seen_prices.add(l['price'])
                
        # Add relevant
        relevant_levels = valid_eqh + valid_eql
        for l in relevant_levels:
            if l['price'] not in seen_prices:
                final_levels_pool.append(l)
                seen_prices.add(l['price'])
        
        # Sort by distance and take top N
        final_levels_pool.sort(key=lambda x: abs(x['price'] - current_price))
        display_levels = final_levels_pool[:max_levels] 

        for lvl in display_levels:
             # Determine role based on current price location
             is_resistance = lvl['price'] > current_price
             role_label = "Res" if is_resistance else "Sup"
             
             level_lines.append({
                'price': lvl['price'],
                'color': lvl['color'],
                'title': f"{lvl['type']} ({role_label})",
                'lineWidth': 1,
                'lineStyle': 2, # Dashed
                'axisLabelVisible': True
            })
                
        # Generate Summary Text (HTML formatted)
        summary_html = self.generate_analysis_summary()
        
        # Append Trade Plan to Sidebar Summary
        tp_html = f"""
        <div style="margin-top: 15px; border-top: 1px solid #363c4e; padding-top: 10px;">
            <h3 style="color: #4caf50; margin-bottom: 5px;">🚀 BULLISH SETUP</h3>
            <div style="font-size: 13px; color: #d1d4dc;">
                <div><strong>Entry:</strong> <span style="color: #4caf50;">${plan['entry']:.2f}</span> ({plan['type']})</div>
                <div><strong>Stop Loss:</strong> <span style="color: #ef5350;">${plan['sl']:.2f}</span></div>
                <div style="margin-top: 5px;"><strong>Targets:</strong></div>
                <div>🎯 TP1: <span style="color: #2196f3;">${plan['tp1']:.2f}</span></div>
                <div>🎯 TP2: <span style="color: #2196f3;">${plan['tp2']:.2f}</span></div>
                <div>🚀 TP3: <span style="color: #2196f3;">${plan['tp3']:.2f}</span></div>
            </div>
        </div>
        """
        summary_html += tp_html
        
        # Prepare Trade Plan Lines for Chart
        trade_lines = [
            {'price': plan['entry'], 'color': '#4caf50', 'title': 'ENTRY', 'style': 2}, # Green Dashed
            {'price': plan['sl'], 'color': '#ef5350', 'title': 'STOP', 'style': 0},     # Red Solid
            {'price': plan['tp1'], 'color': '#2196f3', 'title': 'TP1', 'style': 2},    # Blue Dashed
            {'price': plan['tp2'], 'color': '#2196f3', 'title': 'TP2', 'style': 2},
            {'price': plan['tp3'], 'color': '#2196f3', 'title': 'TP3', 'style': 2},
        ]

        # Prepare Earnings VWAP data
        earnings_vwap_data = getattr(self, 'earnings_vwap', None) or []
        earnings_vwap_date = getattr(self, 'earnings_vwap_date', None)
        earnings_vwap_current = getattr(self, 'earnings_vwap_current', None)
        
        # Serialize to JSON
        chart_data = json.dumps({
            'candles': candle_data,
            'fvg_lines': fvg_lines,
            'level_lines': level_lines,
            'trade_lines': trade_lines,
            'earnings_vwap': earnings_vwap_data,
            'symbol': self.symbol,
            'width': 800,
            'height': 600
        }, default=str)
        
        # HTML Template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.symbol} ICT Analysis</title>
            <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
            <style>
                body {{ background: #0a0a0a; color: #d1d4dc; font-family: -apple-system, sans-serif; margin: 0; display: flex; height: 100vh; }}
                #chart-container {{ flex: 1; position: relative; }}
                #sidebar {{ width: 280px; background: #1a1a2e; padding: 20px; border-left: 1px solid #363c4e; overflow-y: auto; box-shadow: -2px 0 10px rgba(0,0,0,0.3); }}
                h2 {{ color: #2196f3; margin-top: 0; font-size: 20px; }}
                h3 {{ color: #a0a0a0; font-size: 14px; margin-top: 20px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #363c4e; padding-bottom: 5px; }}
                .stat-item {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px; }}
                .bullish {{ color: #4caf50; }}
                .bearish {{ color: #ef5350; }}
                .legend {{ position: absolute; top: 12px; left: 12px; z-index: 10; font-size: 24px; font-weight: bold; color: rgba(255, 255, 255, 0.7); }}
            </style>
        </head>
        <body>
            <div id="chart-container">
                <div class="legend">{self.symbol} 4H</div>
            </div>
            
            <div id="sidebar">
                <h2>📊 Analysis Summary</h2>
                <div style="font-size: 13px; line-height: 1.6;">
                    {summary_html}
                </div>
                
                <h3>Key Levels Found</h3>
                <div class="stat-item">
                    <span>Fair Value Gaps:</span>
                    <span>{len(getattr(self, 'fvgs', []))}</span>
                </div>
                <div class="stat-item">
                    <span>EQH/EQL Zones:</span>
                    <span>{len(self.eq_levels.get('highs', [])) + len(self.eq_levels.get('lows', []))}</span>
                </div>
                
            </div>

            <script>
                try {{
                    const data = {chart_data};
                    
                    if (!data.candles || data.candles.length === 0) {{
                        document.getElementById('chart-container').innerHTML = '<div style="color:white;text-align:center;padding:20px;">No candle data available</div>';
                        throw new Error("No data");
                    }}

                    const chartContainer = document.getElementById('chart-container');
                    const chart = LightweightCharts.createChart(chartContainer, {{
                        layout: {{ background: {{ type: 'solid', color: '#0a0a0a' }}, textColor: '#d1d4dc' }},
                        grid: {{ vertLines: {{ color: '#1f2937' }}, horzLines: {{ color: '#1f2937' }} }},
                        rightPriceScale: {{ borderColor: '#363c4e' }},
                        timeScale: {{ borderColor: '#363c4e', timeVisible: true }},
                        crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                    }});
                    
                    // Candle Series
                    const candlestickSeries = chart.addCandlestickSeries({{
                        upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350'
                    }});
                    candlestickSeries.setData(data.candles);

                    // Markers (Text for Lines)
                    const markers = [];
                    
                    // FVG Areas
                    if (data.fvg_lines) {{
                        data.fvg_lines.forEach(line => {{
                            const fvgSeries = chart.addLineSeries({{
                                color: line.color,
                                lineWidth: 2,
                                lineStyle: LightweightCharts.LineStyle.Solid,
                                priceLineVisible: false,
                                lastValueVisible: true,
                                title: line.title,
                                crosshairMarkerVisible: true
                            }});
                            
                            const lineData = [];
                            data.candles.forEach(c => {{
                                if (c.time >= line.start_time) {{
                                    lineData.push({{ time: c.time, value: line.price }});
                                }}
                            }});
                            if (lineData.length > 0) fvgSeries.setData(lineData);
                        }});
                    }}
                    
                    // Set Markers
                    candlestickSeries.setMarkers(markers);

                    // Key Levels
                    if (data.level_lines) {{
                        data.level_lines.forEach(line => {{
                            const levelSeries = chart.addLineSeries({{
                                color: line.color,
                                lineWidth: line.lineWidth || 1,
                                lineStyle: line.lineStyle || LightweightCharts.LineStyle.Dashed,
                                priceLineVisible: true,
                                lastValueVisible: true,
                                title: line.title
                            }});
                             
                            const lineData = data.candles.map(c => ({{ time: c.time, value: line.price }}));
                            levelSeries.setData(lineData);
                        }});
                    }}
                    
                    // Trade Plan Lines
                    if (data.trade_lines) {{
                        data.trade_lines.forEach(line => {{
                            const tradeSeries = chart.addLineSeries({{
                                color: line.color,
                                lineWidth: 2,
                                lineStyle: line.style === 2 ? LightweightCharts.LineStyle.Dashed : LightweightCharts.LineStyle.Solid,
                                priceLineVisible: true,
                                lastValueVisible: true,
                                title: line.title
                            }});
                            
                            const recentCandles = data.candles.slice(-100);
                            const lineData = recentCandles.map(c => ({{ time: c.time, value: line.price }}));
                            tradeSeries.setData(lineData);
                        }});
                    }}

                    // Earnings VWAP
                    if (data.earnings_vwap && data.earnings_vwap.length > 0) {{
                        const vwapSeries = chart.addLineSeries({{
                            color: '#b39ddb',
                            lineWidth: 2,
                            lineStyle: LightweightCharts.LineStyle.Solid,
                            title: 'Earnings VWAP',
                            priceLineVisible: false
                        }});
                        vwapSeries.setData(data.earnings_vwap);
                    }}
                    
                    // Set visible range safely
                    const lastTime = data.candles[data.candles.length - 1].time;
                    const startTime = data.candles[Math.max(0, data.candles.length - 100)].time;
                    chart.timeScale().setVisibleRange({{ from: startTime, to: lastTime }});
                    
                    // Resizing
                    window.addEventListener('resize', () => {{
                        chart.resize(chartContainer.clientWidth, chartContainer.clientHeight);
                    }});
                    
                }} catch (e) {{
                    console.error("Chart Error:", e);
                    document.body.innerHTML += '<div style="position:absolute;top:10px;left:10px;color:red;background:rgba(0,0,0,0.8);padding:10px;">JS Error: ' + e.message + '</div>';
                }}
            </script>
        </body>
        </html>
        """
        return html

def analyze_stock(symbol: str, timeframe: str = "4h", end_date: str = None):
    """
    Main entry point for analysis.
    """
    print(f"\n🔎 ANALYZING {symbol} ({timeframe})...")
    print('='*50)
    
    analyzer = ICTAnalyzer(symbol, timeframe)
    analyzer.fetch_data()
    
    # Backtesting: Slice data if end_date provided
    if end_date:
        try:
            cutoff = pd.Timestamp(end_date) + pd.Timedelta(days=1) # Include the full day
            
            # Handle Timezone match (yfinance data is usually UTC)
            if analyzer.df.index.tz is not None:
                if cutoff.tz is None:
                    cutoff = cutoff.tz_localize(analyzer.df.index.tz)
            
            analyzer.df = analyzer.df[analyzer.df.index < cutoff]
            if analyzer.df.empty:
                print("❌ No data available for this date range.")
                return None
            print(f"✂️  Data sliced to {end_date} (Last candle: {analyzer.df.index[-1]})")
        except Exception as e:
            print(f"⚠️ Error slicing data: {e}")

    analyzer.detect_swing_points()
    analyzer.detect_equal_highs_lows()
    analyzer.detect_fair_value_gaps()
    analyzer.calculate_earnings_vwap()  # Earnings VWAP
    
    # Check for valid structure (at least one FVG or Level to trade against)
    has_structure = (hasattr(analyzer, 'fvgs') and analyzer.fvgs) or \
                    (hasattr(analyzer, 'eq_levels') and (analyzer.eq_levels['highs'] or analyzer.eq_levels['lows']))
    
    if has_structure:
        plan = analyzer.calculate_trade_plan()
        print(f"\n📊 TRADE PLAN: {plan['type']}")
        print(f"   Entry: ${plan['entry']:.2f}")
        print(f"   sl:    ${plan['sl']:.2f}")
        print(f"   tp1:   ${plan['tp1']:.2f}")
        print(f"   tp3:   ${plan['tp3']:.2f}")
    else:
        print("⚠️ No sufficient structure found for trade plan.")
    
    return analyzer


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description="ICT Buy the Dip Analyzer")
    parser.add_argument("--date", type=str, help="Backtest date (YYYY-MM-DD)", default=None)
    parser.add_argument("--symbol", type=str, help="Specific symbol (optional)", default=None)
    
    args = parser.parse_args()
    
    symbol = args.symbol if args.symbol else "NVDA"
    
    # Single Analysis
    symbols = [symbol] if args.symbol else ["AAPL", "NVDA", "SPY"]
    for s in symbols:
        analyzer = analyze_stock(s, timeframe="4h", end_date=args.date)
        
        if analyzer and not analyzer.df.empty:
            html_content = analyzer.generate_chart_html()
            
            # Filename based on date
            date_str = args.date if args.date else "latest"
            filename = f"{s}_{date_str}_ict_analysis.html"
            
            # Save to specific folder
            import os
            output_dir = "samples"
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except OSError:
                    # Fallback to current dir if permission issue (e.g. cloud env)
                    output_dir = "."
                    
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w") as f:
                f.write(html_content)
            print(f"📊 Chart saved to {filepath}")

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
            progress=False
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
    
    def detect_swing_points(self, lookback: int = 5) -> pd.DataFrame:
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
    
    def detect_equal_highs_lows(self, threshold_pct: float = 0.2) -> dict:
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
                
        return "<br>".join(summary)

    def calculate_fib_levels(self) -> dict:
        """
        Calculate Fibonacci retracement levels for premium/discount zones.
        Uses the most recent swing high to swing low.
        """
        df = self.df
        
        # Find recent swing points
        swing_highs = df[df.get('swing_high', False) == True]
        swing_lows = df[df.get('swing_low', False) == True]
        
        if len(swing_highs) == 0 or len(swing_lows) == 0:
            print("⚠️ Not enough swing points for Fib levels")
            return {}
        
        # Get most recent swing high and low
        recent_high = swing_highs['High'].iloc[-1]
        recent_low = swing_lows['Low'].iloc[-1]
        
        # Ensure we have a valid range
        if recent_high <= recent_low:
            recent_high = df['High'].max()
            recent_low = df['Low'].min()
        
        range_size = recent_high - recent_low
        
        fib_levels = {
            '0.0 (Low)': recent_low,
            '0.236': recent_low + range_size * 0.236,
            '0.382': recent_low + range_size * 0.382,
            '0.5 (Equilibrium)': recent_low + range_size * 0.5,
            '0.618': recent_low + range_size * 0.618,
            '0.786': recent_low + range_size * 0.786,
            '1.0 (High)': recent_high,
        }
        
        self.fib_levels = fib_levels
        print(f"📐 Fib levels: Discount zone below ${fib_levels['0.5 (Equilibrium)']:.2f}")
        return fib_levels
    
    def find_buy_signals(self) -> list:
        """
        Find "buy the dip" signals based on ICT concepts.
        
        Signal criteria:
        1. Price enters discount zone (below 0.5 fib)
        2. Near a bullish order block
        3. Potential FVG fill
        """
        signals = []
        df = self.df
        current_price = df['Close'].iloc[-1]
        
        # Get zones
        fib_levels = getattr(self, 'fib_levels', {})
        order_blocks = getattr(self, 'order_blocks', [])
        fvgs = getattr(self, 'fvgs', [])
        
        equilibrium = fib_levels.get('0.5 (Equilibrium)', 0)
        
        # Check if in discount zone
        in_discount = current_price < equilibrium
        
        # Find nearby bullish order blocks (within 3% of current price)
        nearby_obs = [
            ob for ob in order_blocks 
            if ob['type'] == 'bullish' 
            and abs(ob['low'] - current_price) / current_price < 0.03
        ]
        
        # Find unfilled bullish FVGs below current price
        unfilled_fvgs = [
            fvg for fvg in fvgs
            if fvg['type'] == 'bullish'
            and fvg['top'] > current_price * 0.97
        ]
        
        # Generate signal
        signal_strength = 0
        reasons = []
        
        if in_discount:
            signal_strength += 1
            reasons.append("Price in discount zone (below 0.5 fib)")
        
        if nearby_obs:
            signal_strength += 2
            reasons.append(f"Near {len(nearby_obs)} bullish order block(s)")
        
        if unfilled_fvgs:
            signal_strength += 1
            reasons.append(f"{len(unfilled_fvgs)} bullish FVG(s) nearby")
        
        if signal_strength >= 2:
            signals.append({
                'symbol': self.symbol,
                'price': current_price,
                'strength': signal_strength,
                'rating': '🟢 STRONG' if signal_strength >= 3 else '🟡 MODERATE',
                'reasons': reasons
            })
        
        self.signals = signals
        return signals # End of find_buy_signals

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

    def generate_chart_html(self, show_obs: bool = True, show_fvgs: bool = True, 
                           max_obs: int = 3, max_fvgs: int = 2, max_levels: int = 2) -> str:
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
            
        # Prepare FVG Lines (Midpoints)
        fvg_lines = []
        current_price = df['Close'].iloc[-1]
        threshold_pct = 0.03 # Only show levels within 3% of current price (Reduced from 5%)
        
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
                
                fvg_lines.append({
                    'price': midpoint,
                    'color': color,
                    'title': title,
                    'lineWidth': 2,
                    'lineStyle': 0, # Solid
                    'axisLabelVisible': True
                })
                
        # Prepare EQH/EQL Levels (Filtered by distance but ensure closest are shown)
        level_lines = []
        all_levels = []
        if hasattr(self, 'eq_levels'):
            for eqh in self.eq_levels.get('highs', []):
                all_levels.append({'price': eqh['price'], 'type': 'EQH', 'color': '#ef5350'})
            for eql in self.eq_levels.get('lows', []):
                all_levels.append({'price': eql['price'], 'type': 'EQL', 'color': '#26a69a'})
        
        # 1. Select levels within threshold
        relevant_levels = [
            l for l in all_levels 
            if abs(l['price'] - current_price) / current_price <= threshold_pct
        ]
        
        # 2. Ensure closest EQH and EQL are always included
        all_eqh = [l for l in all_levels if l['type'] == 'EQH']
        all_eql = [l for l in all_levels if l['type'] == 'EQL']
        
        all_eqh.sort(key=lambda x: abs(x['price'] - current_price))
        all_eql.sort(key=lambda x: abs(x['price'] - current_price))
        
        forced_levels = []
        if all_eqh: forced_levels.append(all_eqh[0]) # Closest EQH
        if all_eql: forced_levels.append(all_eql[0]) # Closest EQL
        
        # Combine and remove duplicates
        final_levels_pool = []
        seen_prices = set()
        
        # Add forced first
        for l in forced_levels:
            if l['price'] not in seen_prices:
                final_levels_pool.append(l)
                seen_prices.add(l['price'])
                
        # Add relevant
        for l in relevant_levels:
            if l['price'] not in seen_prices:
                final_levels_pool.append(l)
                seen_prices.add(l['price'])
        
        # Sort by distance and take top N
        final_levels_pool.sort(key=lambda x: abs(x['price'] - current_price))
        display_levels = final_levels_pool[:max_levels] 

        for lvl in display_levels:
             level_lines.append({
                'price': lvl['price'],
                'color': lvl['color'],
                'title': f"{lvl['type']} ({'Res' if lvl['type']=='EQH' else 'Sup'})",
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

        # Serialize to JSON
        chart_data = json.dumps({
            'candles': candle_data,
            'fvg_lines': fvg_lines,
            'level_lines': level_lines,
            'trade_lines': trade_lines,
            'symbol': self.symbol,
            'timeframe': self.timeframe
        })
        
        # HTML Template
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.symbol} - ICT Analysis</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background-color: #131722; color: #d1d4dc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif; overflow: hidden; }}
        .container {{ display: flex; height: 100vh; width: 100vw; }}
        #chart-container {{ flex: 1; height: 100%; position: relative; }}
        #sidebar {{ width: 300px; background-color: #1e222d; border-left: 1px solid #363c4e; padding: 20px; overflow-y: auto; box-sizing: border-box; }}
        h2 {{ margin-top: 0; color: #d1d4dc; font-size: 18px; }}
        .summary-item {{ margin-bottom: 12px; padding: 10px; background-color: #2a2e39; border-radius: 4px; font-size: 14px; line-height: 1.4; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
        .symbol {{ font-size: 24px; font-weight: bold; color: #ffffff; }}
        .timeframe {{ color: #787b86; }}
        #loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: white; font-size: 24px; pointer-events: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div id="chart-container">
            <div id="loading">Loading Chart...</div>
        </div>
        <div id="sidebar">
            <div class="header">
                <span class="symbol">{self.symbol}</span>
                <span class="timeframe">{self.timeframe}</span>
            </div>
            <h2>ICT Analysis</h2>
            <div class="summary-item">
                {summary_html}
            </div>
            <div style="margin-top: 20px; color: #787b86; font-size: 12px;">
                <p><strong>Legend:</strong></p>
                <p><span style="color: #ffeb3b">■</span> Bullish FVG (Midpoint)</p>
                <p><span style="color: #ff9800">■</span> Bearish FVG (Midpoint)</p>
                <p><span style="color: #ef5350">--</span> Resistance (EQH)</p>
                <p><span style="color: #26a69a">--</span> Support (EQL)</p>
                <p><span style="color: #4caf50">--</span> <strong>Entry</strong></p>
                <p><span style="color: #2196f3">--</span> <strong>Take Profit (TP)</strong></p>
                <p><span style="color: #ef5350">__</span> <strong>Stop Loss (SL)</strong></p>
            </div>
        </div>
    </div>

    <script>
        try {{
            const data = {chart_data};
            
            const chartContainer = document.getElementById('chart-container');
            const chart = LightweightCharts.createChart(chartContainer, {{
                layout: {{
                    background: {{ type: 'solid', color: '#131722' }},
                    textColor: '#d1d4dc',
                }},
                grid: {{
                    vertLines: {{ color: '#363c4e' }},
                    horzLines: {{ color: '#363c4e' }},
                }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Normal,
                }},
                rightPriceScale: {{
                    borderColor: '#363c4e',
                }},
                timeScale: {{
                    borderColor: '#363c4e',
                    timeVisible: true,
                }},
            }});

            // Candlestick Series
            const candlestickSeries = chart.addCandlestickSeries({{
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderVisible: false,
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350',
            }});
            candlestickSeries.setData(data.candles);

            // Add FVG Lines
            data.fvg_lines.forEach(line => {{
                candlestickSeries.createPriceLine({{
                    price: line.price, color: line.color, lineWidth: line.lineWidth,
                    lineStyle: line.lineStyle, axisLabelVisible: line.axisLabelVisible, title: line.title,
                }});
            }});

            // Add Level Lines (EQH/EQL)
            data.level_lines.forEach(line => {{
                candlestickSeries.createPriceLine({{
                    price: line.price, color: line.color, lineWidth: line.lineWidth,
                    lineStyle: line.lineStyle, axisLabelVisible: line.axisLabelVisible, title: line.title,
                }});
            }});

            // Add Trade Plan Lines
            if (data.trade_lines) {{
                data.trade_lines.forEach(line => {{
                    candlestickSeries.createPriceLine({{
                        price: line.price, color: line.color, lineWidth: 2,
                        lineStyle: line.style, axisLabelVisible: true, title: line.title,
                    }});
                }});
            }}

            // Adjust Visible Range (Compact View - Last ~3 months / 150 candles)
            const totalCandles = data.candles.length;
            const zoomCandles = 150; // Approx 3 trading months (4h timeframe)
            
            if (totalCandles > zoomCandles) {{
                chart.timeScale().setVisibleLogicalRange({{
                    from: totalCandles - zoomCandles,
                    to: totalCandles - 1
                }});
            }} else {{
                chart.timeScale().fitContent();
            }}

            // Responsive Resize
            window.addEventListener('resize', () => {{
                chart.applyOptions({{ 
                    width: chartContainer.clientWidth, 
                    height: chartContainer.clientHeight 
                }});
            }});
            
            // Remove loading text
            document.getElementById('loading').style.display = 'none';
            console.log("Chart loaded successfully");
            
        }} catch (e) {{
            console.error("Error loading chart:", e);
            document.getElementById('loading').innerText = "Error: " + e.message;
        }}
    </script>
</body>
</html>
        """
        return html_template


def analyze_stock(symbol: str, timeframe: str = "4h", end_date: str = None):
    """Run full ICT analysis on a stock."""
    print(f"\n{'='*50}")
    print(f"🔍 ICT Analysis: {symbol} ({timeframe})")
    if end_date:
        print(f"🕒 Backtesting Date: {end_date}")
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
    analyzer.detect_order_blocks()
    analyzer.detect_fair_value_gaps()
    analyzer.calculate_fib_levels()
    signals = analyzer.find_buy_signals()
    
    print(f"\n{'='*50}")
    print("📈 BUY THE DIP SIGNALS")
    print('='*50)
    
    if signals:
        for signal in signals:
            print(f"\n{signal['rating']} - {signal['symbol']} @ ${signal['price']:.2f}")
            for reason in signal['reasons']:
                print(f"  • {reason}")
    else:
        print("❌ No strong buy signals detected")
    
    return analyzer


def scan_history(symbol: str, days_back: int = 365):
    """
    Scan history candle-by-candle to find past signals and simulate outcomes.
    """
    print(f"\n🔄 Scanning {symbol} history ({days_back} days)...")
    
    # 1. Fetch full data
    analyzer = ICTAnalyzer(symbol, "4h")
    full_df = analyzer.fetch_data(period="2y")
    
    # 2. Iterate backwards from today
    # We need enough history for indicators, so start from end
    start_idx = len(full_df) - (days_back * 6) # Approx 6 candles per day (4h)
    if start_idx < 200: start_idx = 200 # Minimum warm-up
    
    trades = []
    
    # Iterate through history
    # Step size: 1 candle
    for i in range(start_idx, len(full_df) - 5): # Leave room for outcome check
        # Slice data to simulate "live" at time i
        current_date = full_df.index[i]
        
        # Create temp analyzer for this point in time
        sim = ICTAnalyzer(symbol, "4h")
        # Optimization: Don't re-download, just slice existing DF
        sim.df = full_df.iloc[:i+1].copy() 
        
        # Run Detection
        sim.detect_swing_points()
        sim.detect_equal_highs_lows()
        sim.detect_order_blocks()
        sim.detect_fair_value_gaps()
        sim.calculate_fib_levels()
        signals = sim.find_buy_signals()
        
        # Check if Signal Found
        strong_signal = next((s for s in signals if "STRONG" in s['rating']), None)
        
        if strong_signal:
            # Generate Trade Plan
            plan = sim.calculate_trade_plan()
            
            # Simulate Outcome
            entry = plan['entry']
            tp1 = plan['tp1']
            sl = plan['sl']
            
            outcome = "PENDING"
            pnl_r = 0
            
            # Check future candles
            future_df = full_df.iloc[i+1:]
            for _, row in future_df.iterrows():
                # Check hit SL first (Worst case assumption)
                if row['Low'] <= sl:
                    outcome = "LOSS ❌"
                    pnl_r = -1.0
                    break
                # Check hit TP
                if row['High'] >= tp1:
                    outcome = "WIN ✅"
                    pnl_r = (tp1 - entry) / (entry - sl)
                    break
            
            # Record Trade
            print(f"  Found Signal @ {current_date}: {outcome} (R: {pnl_r:.2f})")
            
            trades.append({
                'date': current_date,
                'entry': entry,
                'outcome': outcome,
                'pnl': pnl_r,
                'plan': plan
            })
            
            # Generate Snapshot ID
            import os
            output_dir = os.path.join(os.getcwd(), 'samples')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            snapshot_name = f"{symbol}_{current_date.strftime('%Y-%m-%d_%H%M')}_{'WIN' if 'WIN' in outcome else 'LOSS'}.html"
            filepath = os.path.join(output_dir, snapshot_name)
            
            html = sim.generate_chart_html()
            with open(filepath, "w") as f:
                f.write(html)
                
            # Skip forward to avoid duplicate signals for same move
            # Simple skip: 10 candles
            # Note: The loop variable 'i' cannot be modified directly in for-loop
            # but we can't easily skip in a standard for-range. 
            # Ideally use while loop, but for now we just accept clusters or filter externally.
            
    print(f"\n📊 Scan Complete. Found {len(trades)} trades.")
    
    if trades:
        wins = len([t for t in trades if "WIN" in t['outcome']])
        win_rate = (wins / len(trades)) * 100
        print(f"🏆 Win Rate: {win_rate:.1f}%")


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description="ICT Buy the Dip Analyzer")
    parser.add_argument("--date", type=str, help="Backtest date (YYYY-MM-DD)", default=None)
    parser.add_argument("--symbol", type=str, help="Specific symbol (optional)", default=None)
    parser.add_argument("--scan", action="store_true", help="Scan last year for historical signals")
    
    args = parser.parse_args()
    
    symbol = args.symbol if args.symbol else "NVDA"
    
    if args.scan:
        scan_history(symbol)
    else:
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

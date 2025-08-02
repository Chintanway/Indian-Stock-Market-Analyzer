"""
Volume Tracker - Track today's most bought and sold stocks based on volume
Shows top volume stocks for the current trading day
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import concurrent.futures
from threading import Lock
import time
import logging

class VolumeTracker:
    def __init__(self):
        self.cache = {}
        self.cache_lock = Lock()
        self.cache_expiry = 300  # 5 minutes cache
        self.last_update = None
        
        # Popular NSE stocks to track
        self.tracked_stocks = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'ICICIBANK', 'INFY', 'HINDUNILVR',
            'ITC', 'SBIN', 'BHARTIARTL', 'ASIANPAINT', 'MARUTI', 'AXISBANK',
            'KOTAKBANK', 'LT', 'ULTRACEMCO', 'TITAN', 'SUNPHARMA', 'WIPRO',
            'ONGC', 'NTPC', 'POWERGRID', 'COALINDIA', 'NESTLEIND', 'DRREDDY',
            'JSWSTEEL', 'TATASTEEL', 'INDUSINDBK', 'BAJFINANCE', 'HCLTECH',
            'TECHM', 'CIPLA', 'GRASIM', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO',
            'HEROMOTOCO', 'EICHERMOT', 'DIVISLAB', 'BRITANNIA', 'DABUR',
            'MARICO', 'GODREJCP', 'COLPAL', 'PIDILITIND', 'BERGEPAINT',
            'ADANIGREEN', 'ADANIPORTS', 'ADANIENT', 'ADANITRANS', 'SHREECEM',
            'ACC', 'AMBUJCEM', 'SAIL', 'HINDALCO', 'VEDL', 'JINDALSTEL',
            'NMDC', 'NATIONALUM', 'HINDZINC', 'RATNAMANI', 'APOLLOHOSP',
            'FORTIS', 'MAXHEALTH', 'LALPATHLAB', 'BIOCON', 'AUROPHARMA',
            'LUPIN', 'GLENMARK', 'TORNTPHARM', 'ALKEM', 'CADILAHC'
        ]
    
    def get_stock_volume_data(self, symbol: str) -> Optional[Dict]:
        """Get volume and price data for a single stock"""
        try:
            if not symbol or not isinstance(symbol, str):
                print(f"❌ Invalid symbol: {symbol}")
                return None
                
            # Convert to Yahoo Finance format
            yahoo_symbol = f"{symbol}.NS"
            print(f"🔍 Fetching data for {symbol} ({yahoo_symbol})...")
            
            # Get 2 days of data to compare with previous day
            try:
                stock = yf.Ticker(yahoo_symbol)
                hist = stock.history(period="2d", interval="1d")
                print(f"📊 History data for {symbol}:", hist.to_string() if not hist.empty else "No data")
            except Exception as e:
                print(f"❌ Error fetching history for {symbol}: {str(e)}")
                return None
            
            if hist.empty or len(hist) < 1:
                return None
            
            # Get today's data (latest available)
            latest_data = hist.iloc[-1]
            
            # Get previous day's data if available
            prev_data = hist.iloc[-2] if len(hist) > 1 else latest_data
            
            # Calculate volume change
            volume_change = 0
            volume_change_pct = 0
            
            if prev_data['Volume'] > 0:
                volume_change = latest_data['Volume'] - prev_data['Volume']
                volume_change_pct = (volume_change / prev_data['Volume']) * 100
            
            # Calculate price change
            price_change = latest_data['Close'] - prev_data['Close']
            price_change_pct = (price_change / prev_data['Close']) * 100 if prev_data['Close'] > 0 else 0
            
            # Calculate turnover (Volume * Price)
            turnover = latest_data['Volume'] * latest_data['Close']
            
            # Get company name from indian_stock_names
            from indian_stock_names import get_company_info
            company_info = get_company_info(symbol)
            company_name = company_info.get('name', symbol) if company_info else symbol
            
            return {
                'symbol': symbol,
                'company_name': company_name,
                'current_price': float(latest_data['Close']),
                'volume': int(latest_data['Volume']),
                'prev_volume': int(prev_data['Volume']),
                'volume_change': int(volume_change),
                'volume_change_pct': float(volume_change_pct),
                'price_change': float(price_change),
                'price_change_pct': float(price_change_pct),
                'turnover': float(turnover),
                'high': float(latest_data['High']),
                'low': float(latest_data['Low']),
                'open': float(latest_data['Open']),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_all_volume_data(self) -> List[Dict]:
        """Fetch volume data for all tracked stocks using parallel processing"""
        volume_data = []
        
        # Use ThreadPoolExecutor for parallel API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.get_stock_volume_data, symbol): symbol 
                for symbol in self.tracked_stocks
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result(timeout=10)  # 10 second timeout per stock
                    if data:
                        volume_data.append(data)
                except Exception as e:
                    logging.error(f"Error processing {symbol}: {e}")
        
        return volume_data
    
    def get_top_volume_stocks(self, force_refresh: bool = False) -> Dict:
        """Get top volume stocks with caching"""
        with self.cache_lock:
            current_time = time.time()
            
            # Return cached data if available and not expired
            if not force_refresh and 'top_volume' in self.cache:
                cache_time, cached_data = self.cache['top_volume']
                if current_time - cache_time < self.cache_expiry:
                    logging.info("Using cached volume data")
                    return cached_data
                    
            logging.info("Fetching fresh volume data...")
        
        try:
            # Fetch fresh data
            logging.info(f"Fetching volume data for {len(self.tracked_stocks)} stocks...")
            volume_data = self.fetch_all_volume_data()
            
            if not volume_data:
                error_msg = "No volume data returned from fetch_all_volume_data"
                logging.error(error_msg)
                return {"success": False, "error": error_msg}
                
            logging.info(f"Fetched data for {len(volume_data)} stocks")
            
            # Sort by different criteria
            by_volume = sorted(volume_data, key=lambda x: x['volume'], reverse=True)
            by_turnover = sorted(volume_data, key=lambda x: x['turnover'], reverse=True)
            by_volume_increase = sorted(
                [x for x in volume_data if x['volume_change_pct'] > 0], 
                key=lambda x: x['volume_change_pct'], 
                reverse=True
            )
            by_volume_decrease = sorted(
                [x for x in volume_data if x['volume_change_pct'] < 0], 
                key=lambda x: x['volume_change_pct']
            )
            
            # Get top gainers and losers by price
            gainers = sorted(
                [x for x in volume_data if x['price_change_pct'] > 0], 
                key=lambda x: x['price_change_pct'], 
                reverse=True
            )
            losers = sorted(
                [x for x in volume_data if x['price_change_pct'] < 0], 
                key=lambda x: x['price_change_pct']
            )
            
            result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'total_stocks': len(volume_data),
                'top_by_volume': by_volume[:10],
                'top_by_turnover': by_turnover[:10],
                'most_bought': by_volume_increase[:10],  # Highest volume increase
                'most_sold': by_volume_decrease[:10],    # Highest volume decrease
                'top_gainers': gainers[:10],
                'top_losers': losers[:10],
                'market_summary': {
                    'total_volume': sum(x['volume'] for x in volume_data),
                    'total_turnover': sum(x['turnover'] for x in volume_data),
                    'avg_price_change': sum(x['price_change_pct'] for x in volume_data) / len(volume_data),
                    'stocks_up': len([x for x in volume_data if x['price_change_pct'] > 0]),
                    'stocks_down': len([x for x in volume_data if x['price_change_pct'] < 0]),
                    'stocks_unchanged': len([x for x in volume_data if x['price_change_pct'] == 0])
                }
            }
            
            # Update cache
            with self.cache_lock:
                self.cache = result
                self.last_update = current_time
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error fetching volume data: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_stock_volume_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed volume information for a specific stock"""
        try:
            data = self.get_stock_volume_data(symbol.upper())
            if data:
                # Add additional analysis
                data['volume_category'] = self.categorize_volume(data['volume_change_pct'])
                data['price_category'] = self.categorize_price_change(data['price_change_pct'])
                data['trading_intensity'] = self.calculate_trading_intensity(data)
                
            return data
        except Exception as e:
            print(f"Error getting volume details for {symbol}: {e}")
            return None
    
    def categorize_volume(self, volume_change_pct: float) -> str:
        """Categorize volume change"""
        if volume_change_pct > 50:
            return "Very High Increase"
        elif volume_change_pct > 20:
            return "High Increase"
        elif volume_change_pct > 5:
            return "Moderate Increase"
        elif volume_change_pct > -5:
            return "Normal"
        elif volume_change_pct > -20:
            return "Moderate Decrease"
        elif volume_change_pct > -50:
            return "High Decrease"
        else:
            return "Very High Decrease"
    
    def categorize_price_change(self, price_change_pct: float) -> str:
        """Categorize price change"""
        if price_change_pct > 5:
            return "Strong Gain"
        elif price_change_pct > 2:
            return "Moderate Gain"
        elif price_change_pct > 0:
            return "Slight Gain"
        elif price_change_pct == 0:
            return "Unchanged"
        elif price_change_pct > -2:
            return "Slight Loss"
        elif price_change_pct > -5:
            return "Moderate Loss"
        else:
            return "Strong Loss"
    
    def calculate_trading_intensity(self, data: Dict) -> str:
        """Calculate trading intensity based on volume and price movement"""
        volume_pct = abs(data['volume_change_pct'])
        price_pct = abs(data['price_change_pct'])
        
        intensity_score = (volume_pct * 0.6) + (price_pct * 0.4)
        
        if intensity_score > 30:
            return "Very High"
        elif intensity_score > 15:
            return "High"
        elif intensity_score > 5:
            return "Moderate"
        else:
            return "Low"

# Test the volume tracker
if __name__ == "__main__":
    tracker = VolumeTracker()
    result = tracker.get_top_volume_stocks()
    
    if result['success']:
        print("=== TOP VOLUME STOCKS ===")
        print(f"Total stocks tracked: {result['total_stocks']}")
        print(f"Last updated: {result['timestamp']}")
        
        print("\n=== MOST BOUGHT (Volume Increase) ===")
        for i, stock in enumerate(result['most_bought'][:5], 1):
            print(f"{i}. {stock['symbol']}: +{stock['volume_change_pct']:.1f}% volume")
        
        print("\n=== MOST SOLD (Volume Decrease) ===")
        for i, stock in enumerate(result['most_sold'][:5], 1):
            print(f"{i}. {stock['symbol']}: {stock['volume_change_pct']:.1f}% volume")
    else:
        print(f"Error: {result['error']}")

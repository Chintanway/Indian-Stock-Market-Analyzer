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

# Import NSE tools
from nsetools import Nse

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
        """
        Get volume and price data for a single stock with trade count and size metrics
        
        Args:
            symbol: Stock symbol (without .NS suffix)
            
        Returns:
            Dictionary containing volume and price data, or None if data is unavailable
        """
        def log_error(msg):
            print(f"❌ {msg}")
            return None
            
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str) or len(symbol.strip()) < 1:
                return log_error(f"Invalid symbol: {symbol}")
                
            # Clean and format symbol
            symbol = symbol.strip().upper()
            yahoo_symbol = f"{symbol}.NS"
            
            print(f"\n{'='*80}")
            print(f"🔍 FETCHING DATA FOR {symbol} ({yahoo_symbol})")
            print(f"{'='*80}")
            
            # Get company info for better logging
            from indian_stock_names import get_company_info
            company_info = get_company_info(symbol)
            company_name = company_info.get('name', 'Unknown') if company_info else 'Unknown'
            print(f"🏢 Company: {company_name} ({symbol})")
            
            try:
                print(f"📡 Connecting to Yahoo Finance for {symbol}...")
                stock = yf.Ticker(yahoo_symbol)
                
                # Try different periods to get data
                periods_to_try = ["2d", "5d", "1mo"]
                hist = None
                
                for period in periods_to_try:
                    print(f"📅 Fetching {period} of historical data for {symbol}...")
                    try:
                        hist = stock.history(period=period, interval="1d")
                        if not hist.empty and len(hist) >= 2:
                            print(f"✅ Got {len(hist)} days of data")
                            break
                        print(f"⚠️ Not enough data points with period={period}")
                    except Exception as e:
                        print(f"⚠️ Error with period={period}: {str(e)}")
                
                if hist is None or hist.empty:
                    return log_error(f"No historical data available for {symbol}")
                
                print(f"📊 Data columns: {', '.join(hist.columns)}")
                print(f"📅 Date range: {hist.index[0].date()} to {hist.index[-1].date()}")
                
                # Ensure we have the required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in hist.columns]
                if missing_columns:
                    return log_error(f"Missing required columns: {', '.join(missing_columns)}")
                    
                print(f"📈 Data columns: {hist.columns.tolist()}")
                print(f"📅 Date range: {hist.index[0]} to {hist.index[-1]}")
                    
                # Get today's and previous day's data
                today = hist.iloc[-1]
                prev_day = hist.iloc[-2]
                
                # Calculate volume metrics
                total_volume = today['Volume']
                avg_volume_5d = hist['Volume'].mean()
                volume_change_pct = ((total_volume - avg_volume_5d) / avg_volume_5d) * 100 if avg_volume_5d > 0 else 0
                
                # Calculate turnover (volume * average price)
                avg_price = (today['High'] + today['Low'] + today['Close']) / 3
                turnover = total_volume * avg_price
                
                # Estimate trade count (this is a simplified estimation)
                # In a real-world scenario, you would get this from a market data API
                avg_trade_size = 1000  # Default average trade size
                trade_count = int(total_volume / avg_trade_size) if avg_trade_size > 0 else 0
                
                return {
                    'symbol': symbol,
                    'price': float(today['Close']),
                    'open': float(today['Open']),
                    'high': float(today['High']),
                    'low': float(today['Low']),
                    'price_change': float(today['Close'] - today['Open']),
                    'price_change_pct': ((today['Close'] - today['Open']) / today['Open'] * 100) if today['Open'] > 0 else 0,
                    'volume': int(total_volume),
                    'volume_change': int(total_volume - prev_day['Volume']),
                    'volume_change_pct': ((total_volume - prev_day['Volume']) / prev_day['Volume'] * 100) if prev_day['Volume'] > 0 else 0,
                    'trade_count': trade_count,
                    'total_volume': int(total_volume),
                    'avg_trade_size': int(avg_trade_size),
                    'volume_change_pct_vs_avg': volume_change_pct,
                    'turnover': float(turnover),
                    'timestamp': datetime.now().isoformat()
                }
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
            print(f"Error fetching data for {symbol} ({yahoo_symbol}): {e}")
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
    
    def _get_trade_activity_level(self, trade_count: int, avg_trade_size: float) -> str:
        """
        Determine the trade activity level based on count and size
        
        Args:
            trade_count: Number of trades
            avg_trade_size: Average size per trade
            
        Returns:
            Activity level (Very High, High, Medium, Low)
        """
        if trade_count > 1000 and avg_trade_size > 5000:
            return "Very High"
        elif trade_count > 500 or (trade_count > 200 and avg_trade_size > 2000):
            return "High"
        elif trade_count > 100 or (trade_count > 50 and avg_trade_size > 1000):
            return "Medium"
        return "Low"

    def get_top_volume_stocks(self, limit: int = 10, min_volume: int = 100) -> Dict:
        """
        Get today's top volume stocks with trade count and size metrics
        
        Args:
            limit: Maximum number of stocks to return per category
            min_volume: Minimum volume threshold for filtering stocks
            
        Returns:
            Dictionary containing top volume stocks and trade metrics
        """
        print("\n" + "="*80)
        print("📊 GET_TOP_VOLUME_STOCKS CALLED")
        print("="*80)
        
        # List of actively traded NSE stocks (verified)
        symbols = [
            'RELIANCE',    # Reliance Industries
            'TCS',         # Tata Consultancy Services
            'HDFCBANK',    # HDFC Bank
            'ICICIBANK',   # ICICI Bank
            'INFY',        # Infosys
            'ITC',         # ITC Limited
            'BHARTIARTL',  # Bharti Airtel
            'LT',          # Larsen & Toubro
            'KOTAKBANK',   # Kotak Mahindra Bank
            'AXISBANK',    # Axis Bank
            'HCLTECH',     # HCL Technologies
            'ASIANPAINT',  # Asian Paints
            'TITAN',       # Titan Company
            'NTPC',        # NTPC Limited
            'TATAMOTORS',  # Tata Motors
            'ULTRACEMCO',  # UltraTech Cement
            'SUNPHARMA',   # Sun Pharmaceutical
            'TATASTEEL',   # Tata Steel
            'BAJFINANCE',  # Bajaj Finance
            'MARUTI'       # Maruti Suzuki
        ]
        data_source = 'nse_active_stocks'
        
        print(f"📊 Processing {len(symbols)} major stocks...")
        
        volume_data = []
        failed_symbols = []
        
        for idx, symbol in enumerate(symbols, 1):
            try:
                symbol = str(symbol).strip().upper()
                print(f"\n{'='*50}")
                print(f"📊 Processing {idx}/{len(symbols)}: {symbol}")
                print(f"{'='*50}")
                
                # Get stock data with detailed logging
                yahoo_symbol = f"{symbol}.NS"
                print(f"📡 Fetching data for {symbol}...")
                
                # Get stock data
                stock = yf.Ticker(yahoo_symbol)
                hist = stock.history(period="2d")
                
                if hist.empty:
                    print(f"⚠️ No data for {symbol}")
                    continue
                    
                print(f"📅 Data for {symbol}:")
                print(hist[['Open', 'High', 'Low', 'Close', 'Volume']])
                
                # Calculate price change
                if len(hist) >= 2:
                    today = hist.iloc[-1]
                    prev_day = hist.iloc[-2]
                    
                    price_change = today['Close'] - prev_day['Close']
                    price_change_pct = (price_change / prev_day['Close']) * 100 if prev_day['Close'] > 0 else 0
                    
                    # Create stock data
                    stock_data = {
                        'symbol': symbol,
                        'company_name': symbol,  # Simplified for now
                        'current_price': round(today['Close'], 2),
                        'price_change': round(price_change, 2),
                        'price_change_pct': round(price_change_pct, 2),
                        'volume': int(today['Volume']),
                        'open': round(today['Open'], 2),
                        'high': round(today['High'], 2),
                        'low': round(today['Low'], 2),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    print(f"✅ Successfully processed {symbol}")
                    volume_data.append(stock_data)
                else:
                    print(f"⚠️ Not enough data points for {symbol}")
                    
            except Exception as e:
                print(f"❌ Error processing {symbol}: {str(e)}")
                failed_symbols.append(symbol)
        
        # Sort by volume and limit results
        top_by_volume = sorted(volume_data, 
                             key=lambda x: x.get('volume', 0), 
                             reverse=True)[:limit]
        
        # Prepare market summary
        market_summary = {
            'total_stocks_analyzed': len(volume_data),
            'total_volume': sum(stock.get('volume', 0) for stock in volume_data),
            'avg_volume': sum(stock.get('volume', 0) for stock in volume_data) / len(volume_data) if volume_data else 0,
            'top_stocks': [{
                'symbol': s['symbol'],
                'volume': s['volume'],
                'price': s['current_price']
            } for s in top_by_volume[:3]]  # Top 3 by volume for summary
        }
        
        print("\n✅ DATA READY:")
        print(f"- Stocks processed: {len(volume_data)}")
        print(f"- Top volume stocks: {[s['symbol'] for s in top_by_volume[:3]]}")
        
        return {
            'success': True,
            'top_by_volume': top_by_volume,
            'top_by_trade_count': top_by_volume,  # Using same data for now
            'top_by_avg_trade_size': top_by_volume,  # Using same data for now
            'most_active': top_by_volume,  # Using same data for now
            'market_summary': market_summary,
            'metadata': {
                'data_source': data_source,
                'processed_at': datetime.now().isoformat(),
                'stocks_processed': len(volume_data),
                'stocks_failed': len(failed_symbols)
            }
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

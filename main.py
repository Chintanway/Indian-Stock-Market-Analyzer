from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import yfinance as yf
import pandas as pd
# Fix for numpy compatibility with pandas-ta
import numpy as np
np.NaN = np.nan
import pandas_ta as ta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
import plotly.utils
from database import (
    create_tables, save_stock_analysis, update_user_search, 
    get_popular_stocks, get_analysis_history, save_model_performance,
    get_model_accuracy_stats
)
from investment_advisor import InvestmentAdvisor
from market_dashboard import MarketDashboard
from nse_stocks import get_yahoo_symbol, validate_nse_symbol, get_stock_info, suggest_similar_symbols
from indian_stock_names import find_stock_by_name, suggest_stock_names, get_company_info
from ipo_tracker import IPOTracker
from volume_tracker import VolumeTracker

# Import advanced AI models
try:
    from advanced_models import AdvancedStockPredictor
    ADVANCED_MODELS_AVAILABLE = True
    print("Advanced AI models (Prophet, LSTM, ARIMA) loaded successfully")
except ImportError as e:
    print(f"Advanced models not available: {e}")
    ADVANCED_MODELS_AVAILABLE = False
    AdvancedStockPredictor = None



# Initialize database on startup
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_tables()
    yield
    # Shutdown
    pass

app = FastAPI(
    title="Indian Stock Market Analyzer", 
    description="AI-Powered Technical Analysis for NSE Stocks",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class StockAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = "stock_model.joblib"
        self.scaler_path = "scaler.joblib"
        
        # Initialize advanced predictor if available
        if ADVANCED_MODELS_AVAILABLE:
            self.advanced_predictor = AdvancedStockPredictor()
        else:
            self.advanced_predictor = None
            
        # Initialize investment advisor
        self.investment_advisor = InvestmentAdvisor()
        
    def fetch_stock_data(self, symbol, period="60d"):
        """Fetch stock data from Yahoo Finance for NSE stocks"""
        try:
            # Add .NS suffix for NSE stocks
            ticker = f"{symbol}.NS"
            stock = yf.Ticker(ticker)
            
            # Get historical data with more recent timeframe for better accuracy
            data = stock.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Ensure we have enough data for indicators
            if len(data) < 50:
                # Try getting more data if insufficient
                data = stock.history(period="3mo")
                
            if len(data) < 20:
                raise ValueError(f"Insufficient data for technical analysis: {len(data)} days")
                
            # Try to get the most current price from multiple sources
            try:
                # Method 1: Try to get current price from info
                info = stock.info
                current_price = None
                
                # Try different price fields from info
                price_fields = ['currentPrice', 'regularMarketPrice', 'previousClose', 'open']
                for field in price_fields:
                    if field in info and info[field] and info[field] > 0:
                        current_price = float(info[field])
                        break
                
                # Method 2: Try 1-day history if info fails
                if not current_price:
                    recent = stock.history(period="2d")
                    if not recent.empty:
                        current_price = float(recent['Close'].iloc[-1])
                
                # Method 3: Try 1-minute data for most recent price
                if not current_price:
                    latest = stock.history(period="1d", interval="1m")
                    if not latest.empty:
                        current_price = float(latest['Close'].iloc[-1])
                
                # Update the data with most current price if found
                if current_price and current_price > 0:
                    data.iloc[-1, data.columns.get_loc('Close')] = current_price
                    # Update high/low appropriately
                    current_high = data.iloc[-1]['High']
                    current_low = data.iloc[-1]['Low']
                    data.iloc[-1, data.columns.get_loc('High')] = max(current_high, current_price)
                    data.iloc[-1, data.columns.get_loc('Low')] = min(current_low, current_price)
                    # Update index to current date for real-time feel
                    from datetime import datetime
                    current_time = datetime.now()
                    data.index = pd.DatetimeIndex([data.index[i] if i < len(data)-1 else current_time for i in range(len(data))])
                    
            except Exception as price_error:
                print(f"Price update error: {price_error}")
                # Use historical data as is if all methods fail
                
            return data
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Error fetching data for {symbol}: {str(e)}")
    
    def calculate_indicators(self, data):
        """Calculate technical indicators with improved accuracy"""
        df = data.copy()
        
        # Ensure we have numeric data
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['High'] = pd.to_numeric(df['High'], errors='coerce')
        df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        
        # RSI (14-period) - more accurate calculation
        try:
            df['RSI'] = ta.rsi(df['Close'], length=14)
            # Fill NaN values with neutral RSI
            df['RSI'] = df['RSI'].fillna(50)
        except:
            df['RSI'] = 50
        
        # EMA (5-period)
        try:
            df['EMA_5'] = ta.ema(df['Close'], length=5)
            df['EMA_5'] = df['EMA_5'].fillna(df['Close'])
        except:
            df['EMA_5'] = df['Close']
        
        # MACD with proper error handling
        try:
            macd_result = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            if macd_result is not None and not macd_result.empty:
                df['MACD'] = macd_result['MACD_12_26_9'].fillna(0)
                df['MACD_Signal'] = macd_result['MACDs_12_26_9'].fillna(0)
                df['MACD_Histogram'] = macd_result['MACDh_12_26_9'].fillna(0)
            else:
                df['MACD'] = 0
                df['MACD_Signal'] = 0
                df['MACD_Histogram'] = 0
        except:
            df['MACD'] = 0
            df['MACD_Signal'] = 0
            df['MACD_Histogram'] = 0
        
        # Bollinger Bands
        try:
            bb_result = ta.bbands(df['Close'], length=20, std=2)
            if bb_result is not None and not bb_result.empty:
                df['BB_Upper'] = bb_result['BBU_20_2.0'].fillna(df['Close'])
                df['BB_Lower'] = bb_result['BBL_20_2.0'].fillna(df['Close'])
                df['BB_Middle'] = bb_result['BBM_20_2.0'].fillna(df['Close'])
            else:
                df['BB_Upper'] = df['Close'] * 1.02
                df['BB_Lower'] = df['Close'] * 0.98
                df['BB_Middle'] = df['Close']
        except:
            df['BB_Upper'] = df['Close'] * 1.02
            df['BB_Lower'] = df['Close'] * 0.98
            df['BB_Middle'] = df['Close']
        
        # Moving averages with better error handling
        try:
            df['SMA_20'] = ta.sma(df['Close'], length=20).fillna(df['Close'])
        except:
            df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            
        try:
            df['SMA_50'] = ta.sma(df['Close'], length=50).fillna(df['Close'])
        except:
            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        
        # Volume indicators
        try:
            df['Volume_SMA'] = ta.sma(df['Volume'], length=10).fillna(df['Volume'])
        except:
            df['Volume_SMA'] = df['Volume'].rolling(window=10, min_periods=1).mean()
        
        # Price change percentage
        df['Price_Change'] = df['Close'].pct_change().fillna(0) * 100
        
        # Ensure all values are numeric and not NaN
        numeric_columns = ['RSI', 'EMA_5', 'MACD', 'MACD_Signal', 'MACD_Histogram', 
                          'BB_Upper', 'BB_Lower', 'BB_Middle', 'SMA_20', 'SMA_50', 
                          'Volume_SMA', 'Price_Change']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        # Select features for prediction
        feature_columns = ['RSI', 'EMA_5', 'MACD', 'MACD_Histogram', 'Price_Change']
        
        # Create target (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        # Remove rows with NaN values
        clean_df = df.dropna()
        
        if len(clean_df) < 10:
            raise ValueError("Insufficient data for prediction")
        
        X = clean_df[feature_columns]
        y = clean_df['Target']
        
        return X, y, clean_df
    
    def train_model(self, X, y):
        """Train the AI prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Save model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        return {
            'train_accuracy': round(float(train_score) * 100, 2),
            'test_accuracy': round(float(test_score) * 100, 2)
        }
    
    def load_model(self):
        """Load existing model"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False
    
    def predict_price(self, latest_features, feature_names=None):
        """Predict next day's price"""
        if self.model is None:
            if not self.load_model():
                raise ValueError("No trained model available")
        
        if self.model is None:
            raise ValueError("Model is not available for prediction")
        
        # Create DataFrame with feature names to avoid sklearn warning
        if feature_names:
            import pandas as pd
            features_df = pd.DataFrame([latest_features], columns=feature_names)
            features_scaled = self.scaler.transform(features_df)
        else:
            features_scaled = self.scaler.transform([latest_features])
        
        predicted_price = float(self.model.predict(features_scaled)[0])
        
        return predicted_price
    
    def generate_signal(self, current_price, predicted_price, rsi, macd_histogram):
        """Generate trading signal based on technical indicators and prediction"""
        try:
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Ensure values are numeric
            rsi = float(rsi) if not pd.isna(rsi) else 50
            macd_histogram = float(macd_histogram) if not pd.isna(macd_histogram) else 0
            price_change_pct = float(price_change_pct) if not pd.isna(price_change_pct) else 0
            
            # Enhanced signal logic with multiple conditions
            buy_signals = 0
            sell_signals = 0
            
            # RSI signals (more refined)
            if rsi < 30:
                buy_signals += 3  # Strong oversold
            elif rsi < 40:
                buy_signals += 1  # Mild oversold
            elif rsi > 70:
                sell_signals += 3  # Strong overbought
            elif rsi > 60:
                sell_signals += 1  # Mild overbought
            elif 45 <= rsi <= 55:
                buy_signals += 0.5  # Neutral zone slightly bullish
            
            # MACD signals (stronger weight)
            if macd_histogram > 0.1:
                buy_signals += 2  # Strong positive momentum
            elif macd_histogram > 0:
                buy_signals += 1  # Positive momentum
            elif macd_histogram < -0.1:
                sell_signals += 2  # Strong negative momentum
            elif macd_histogram < 0:
                sell_signals += 1  # Negative momentum
                
            # Price prediction signals (refined thresholds)
            if price_change_pct > 5:
                buy_signals += 3  # Very bullish prediction
            elif price_change_pct > 2:
                buy_signals += 2  # Bullish prediction
            elif price_change_pct > 0.5:
                buy_signals += 1  # Mildly bullish
            elif price_change_pct < -5:
                sell_signals += 3  # Very bearish prediction
            elif price_change_pct < -2:
                sell_signals += 2  # Bearish prediction
            elif price_change_pct < -0.5:
                sell_signals += 1  # Mildly bearish
            
            # Determine final signal with better thresholds
            total_signals = buy_signals + sell_signals
            
            if buy_signals >= 4 and buy_signals > sell_signals * 1.5:
                signal = "BUY"
                confidence = "High" if buy_signals >= 6 else "Medium"
            elif sell_signals >= 4 and sell_signals > buy_signals * 1.5:
                signal = "SELL"
                confidence = "High" if sell_signals >= 6 else "Medium"
            elif abs(price_change_pct) < 1 and 40 <= rsi <= 60 and abs(macd_histogram) < 0.05:
                signal = "NEUTRAL"
                confidence = "Medium"
            else:
                signal = "CAUTION"
                confidence = "Medium" if total_signals >= 3 else "Low"
            
            return signal, confidence
            
        except Exception as e:
            # Fallback signal generation
            print(f"Error in signal generation: {e}")
            return "NEUTRAL", "Low"
    
    def calculate_targets(self, current_price, predicted_price, signal, bb_lower, bb_upper, sma_20):
        """Calculate target price and stop loss based on technical levels"""
        try:
            current_price = float(current_price)
            predicted_price = float(predicted_price)
            bb_lower = float(bb_lower)
            bb_upper = float(bb_upper)
            sma_20 = float(sma_20)
            
            if signal == "BUY":
                # Target: Higher of predicted price + 3% or Bollinger Upper Band
                target_price = max(predicted_price * 1.03, bb_upper)
                # Stop Loss: Lower of 3% below current or Bollinger Lower Band
                stop_loss = min(current_price * 0.97, bb_lower)
                
            elif signal == "SELL":
                # Target: Lower of predicted price - 3% or Bollinger Lower Band  
                target_price = min(predicted_price * 0.97, bb_lower)
                # Stop Loss: Higher of 3% above current or Bollinger Upper Band
                stop_loss = max(current_price * 1.03, bb_upper)
                
            elif signal == "CAUTION":
                # Conservative targets for caution signals
                if predicted_price > current_price:
                    target_price = current_price * 1.02  # 2% gain target
                    stop_loss = current_price * 0.98    # 2% stop loss
                else:
                    target_price = current_price * 0.98  # 2% decline target
                    stop_loss = current_price * 1.02    # 2% stop loss above
                    
            else:  # NEUTRAL
                # Neutral: Target near SMA20, tight stop loss
                target_price = sma_20
                stop_loss = current_price * 0.975  # 2.5% stop loss
            
            # Ensure stop loss makes sense (never set stop loss = current price)
            if signal in ["BUY", "NEUTRAL"] and stop_loss >= current_price:
                stop_loss = current_price * 0.95  # 5% below current
            elif signal == "SELL" and stop_loss <= current_price:
                stop_loss = current_price * 1.05  # 5% above current
                
            return round(target_price, 2), round(stop_loss, 2)
            
        except Exception as e:
            print(f"Error calculating targets: {e}")
            # Fallback calculation
            target_price = predicted_price
            stop_loss = current_price * 0.95 if signal != "SELL" else current_price * 1.05
            return round(target_price, 2), round(stop_loss, 2)

# Initialize analyzer
analyzer = StockAnalyzer()
dashboard = MarketDashboard()
ipo_tracker = IPOTracker()
volume_tracker = VolumeTracker()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_stock(symbol: str = Form(...)):
    try:
        symbol = symbol.upper().strip()
        
        # Validate and clean symbol
        if len(symbol) > 20:
            return JSONResponse({
                'success': False,
                'error': f'Symbol "{symbol}" is too long. Use standard NSE symbols like RELIANCE, TCS, INFY, ADANIGREEN, HDFCBANK, etc.'
            }, status_code=400)
        
        # Common symbol mappings for user convenience
        symbol_mappings = {
            'ADANIGREEN': 'ADANIGREEN',
            'ADANIGREENENERGYLTD': 'ADANIGREEN',
            'ADANIGREENENERGY': 'ADANIGREEN',
            'ADANIGREENENERGYL': 'ADANIGREEN',
            'RELIANCEIND': 'RELIANCE',
            'RELIANCEINDUSTRIES': 'RELIANCE',
            'TATACONSULTANCY': 'TCS',
            'TATACONS': 'TCS',
            'TATAPOWERRENEWABLE': 'TATAPOWER',
            'TATA': 'TATAMOTORS',
            'TATAMOTOTRS': 'TATAMOTORS',
            'INFOSYS': 'INFY',
            'INFOSYSLTD': 'INFY',
            'HDFCBANKLIMITED': 'HDFCBANK',
            'BHARTIAIRTEL': 'BHARTIARTL',
            'BHARTI': 'BHARTIARTL',
            'ICICIBANKLIMITED': 'ICICIBANK',
            'SBIN': 'SBIN',
            'STATEBANKOFINDIA': 'SBIN',
            'ITCLIMITED': 'ITC',
            'HINDUNILVR': 'HINDUNILVR',
            'HINDUNILEVERLIMITED': 'HINDUNILVR',
            'JSW': 'JSWSTEEL',
            'JSWSTEEL': 'JSWSTEEL',
            'MARUTI': 'MARUTI',
            'MARUTISUZUKI': 'MARUTI',
            'MARUTISUZUKIINDIA': 'MARUTI',
            'LT': 'LT',
            'LARSENTOUBRO': 'LT',
            'LARSEN': 'LT',
            'COALINDIA': 'COALINDIA',
            'CIL': 'COALINDIA',
            'NTPC': 'NTPC',
            'ONGC': 'ONGC',
            'POWERGRID': 'POWERGRID',
            'ULTRATECH': 'ULTRACEMCO',
            'ULTRACEMCO': 'ULTRACEMCO',
            'SUNPHARMA': 'SUNPHARMA',
            'DRREDDY': 'DRREDDY',
            'DRREDDYS': 'DRREDDY',
            'ASIANPAINTS': 'ASIANPAINT',
            'ASIANPAINT': 'ASIANPAINT'
        }
        
        # Map common variations to correct symbols
        if symbol in symbol_mappings:
            symbol = symbol_mappings[symbol]
        
        # Clean user input
        original_input = symbol.strip()
        
        # First, try to find by company name using comprehensive database
        company_match = find_stock_by_name(original_input)
        
        if company_match and company_match['confidence'] >= 0.7:
            # Found a good match by company name
            clean_symbol = company_match['symbol']
            company_name = company_match['company_name']
            yahoo_symbol = get_yahoo_symbol(clean_symbol)
            print(f"Found company: {original_input} -> {clean_symbol} ({company_name})")
        else:
            # Fall back to symbol-based validation
            original_symbol = original_input.upper()
            stock_info = get_stock_info(original_symbol)
            
            # Check if symbol is valid
            if not stock_info['is_valid']:
                # Try comprehensive name suggestions first
                name_suggestions = suggest_stock_names(original_input, 5)
                symbol_suggestions = suggest_similar_symbols(original_symbol)
                
                # Update user search tracking even for invalid symbols (for analytics)
                try:
                    update_user_search(original_input)
                except Exception as search_error:
                    print(f"Search tracking error: {search_error}")
                
                # Combine suggestions from both systems
                all_suggestions = []
                
                # Add name-based suggestions
                for suggestion in name_suggestions:
                    all_suggestions.append(f"{suggestion['symbol']} ({suggestion['company_name']})")
                
                # Add symbol-based suggestions if not already included
                for suggestion in symbol_suggestions:
                    if suggestion not in [s['symbol'] for s in name_suggestions]:
                        all_suggestions.append(suggestion)
                
                # Create helpful error message
                error_message = f'Stock "{original_input}" not found.'
                if all_suggestions:
                    error_message += f' Did you mean: {", ".join(all_suggestions[:5])}?'
                else:
                    error_message += ' Please try entering the full company name (e.g., "Tata Consultancy Services", "Reliance Industries") or standard NSE symbols like RELIANCE, TCS, INFY, TATAMOTORS, HDFCBANK, etc.'
                
                return JSONResponse({
                    'success': False,
                    'error': error_message,
                    'suggestions': all_suggestions[:5] if all_suggestions else [],
                    'symbol_searched': original_input,
                    'search_type': 'comprehensive'
                }, status_code=400)
            
            clean_symbol = stock_info['symbol']
            company_name = get_company_info(clean_symbol) or clean_symbol
            yahoo_symbol = stock_info['yahoo_symbol']
        
        # Update user search tracking (with error handling)
        try:
            update_user_search(clean_symbol)
        except Exception as search_error:
            print(f"Search tracking error: {search_error}")
        
        # Fetch and analyze stock data using correct symbol
        data = analyzer.fetch_stock_data(clean_symbol)
        df_with_indicators = analyzer.calculate_indicators(data)
        
        # Prepare data for ML
        X, y, clean_df = analyzer.prepare_features(df_with_indicators)
        
        # Train model with current data
        model_stats = analyzer.train_model(X, y)
        
        # Get latest data for prediction
        latest_data = clean_df.iloc[-1]
        current_price = float(latest_data['Close'])
        
        # Get fresh current price from Yahoo Finance
        try:
            stock = yf.Ticker(yahoo_symbol)
            info = stock.info
            
            # Try to get the most current price
            fresh_price = None
            price_fields = ['currentPrice', 'regularMarketPrice', 'previousClose']
            for field in price_fields:
                if field in info and info[field] and info[field] > 0:
                    fresh_price = float(info[field])
                    break
            
            # Use fresh price if reasonable (within 15% of historical)
            if fresh_price and abs(fresh_price - current_price) / current_price < 0.15:
                current_price = fresh_price
                print(f"Updated {clean_symbol} price: ₹{current_price:.2f} (from Yahoo Finance)")
            else:
                print(f"Using historical price for {clean_symbol}: ₹{current_price:.2f}")
                
        except Exception as price_error:
            print(f"Could not get fresh price for {symbol}: {price_error}")
            print(f"Using historical price: ₹{current_price:.2f}")
        
        # Prepare features for prediction (with proper feature names for sklearn)
        feature_names = ['RSI', 'EMA_5', 'MACD', 'MACD_Histogram', 'Price_Change']
        latest_features = [
            latest_data['RSI'],
            latest_data['EMA_5'],
            latest_data['MACD'],
            latest_data['MACD_Histogram'],
            latest_data['Price_Change']
        ]
        
        # Make prediction with multiple models
        predicted_price = analyzer.predict_price(latest_features, feature_names)
        
        # Try advanced AI models if available
        advanced_predictions = None
        model_confidence = "Medium"
        
        if analyzer.advanced_predictor:
            try:
                # Get ensemble prediction from Prophet, LSTM, and ARIMA
                ensemble_result = analyzer.advanced_predictor.ensemble_prediction(data)
                
                if ensemble_result:
                    # Use ensemble prediction if confidence is high
                    if ensemble_result['confidence'] > 0.6:
                        predicted_price = ensemble_result['ensemble_prediction']
                        model_confidence = "High" if ensemble_result['confidence'] > 0.8 else "Medium"
                        advanced_predictions = ensemble_result
                        print(f"Using advanced AI ensemble prediction: ₹{predicted_price:.2f}")
                    else:
                        print(f"Using traditional model (low ensemble confidence: {ensemble_result['confidence']:.2f})")
                        advanced_predictions = ensemble_result
                
            except Exception as advanced_error:
                print(f"Advanced models error: {advanced_error}")
                # Continue with traditional prediction
        
        # Get current timestamp for real-time feel
        from datetime import datetime
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Generate trading signal
        signal, confidence = analyzer.generate_signal(
            current_price, predicted_price, latest_data['RSI'], latest_data['MACD_Histogram']
        )
        
        # Calculate targets using technical levels
        target_price, stop_loss = analyzer.calculate_targets(
            current_price, predicted_price, signal,
            latest_data['BB_Lower'], latest_data['BB_Upper'], latest_data['SMA_20']
        )
        
        # Generate comprehensive investment recommendation
        sentiment_analysis = analyzer.investment_advisor.analyze_market_sentiment(clean_symbol, data)
        fundamental_analysis = analyzer.investment_advisor.fundamental_analysis(clean_symbol, current_price)
        risk_analysis = analyzer.investment_advisor.risk_analysis(data, current_price, predicted_price)
        
        investment_recommendation = analyzer.investment_advisor.generate_investment_recommendation(
            clean_symbol, signal, confidence, sentiment_analysis, fundamental_analysis, risk_analysis,
            current_price, predicted_price, target_price, stop_loss
        )
        
        # Prepare chart data
        chart_data = clean_df.tail(30).copy()
        
        # Create Plotly chart
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=chart_data.index,
            open=chart_data['Open'],
            high=chart_data['High'],
            low=chart_data['Low'],
            close=chart_data['Close'],
            name='Price'
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['EMA_5'],
            mode='lines',
            name='EMA 5',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='blue')
        ))
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=chart_data.index,
            y=chart_data['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=f'{clean_symbol.upper()} - Technical Analysis',
            yaxis_title='Price (₹)',
            xaxis_title='Date',
            template='plotly_white',
            height=400
        )
        
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Prepare data for database storage (convert numpy types to Python types)
        analysis_data = {
            'symbol': clean_symbol,
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'target_price': float(target_price),
            'stop_loss': float(stop_loss),
            'signal': signal,
            'confidence': confidence,
            'rsi': float(latest_data['RSI']),
            'ema_5': float(latest_data['EMA_5']),
            'macd': float(latest_data['MACD']),
            'macd_signal': float(latest_data['MACD_Signal']),
            'macd_histogram': float(latest_data['MACD_Histogram']),
            'sma_20': float(latest_data['SMA_20']),
            'bb_upper': float(latest_data['BB_Upper']),
            'bb_lower': float(latest_data['BB_Lower']),
            'train_accuracy': float(model_stats['train_accuracy']),
            'test_accuracy': float(model_stats['test_accuracy']),
            'prediction_change': float(((predicted_price - current_price) / current_price) * 100),
            'data_timestamp': current_timestamp
        }
        
        # Save analysis to database
        try:
            save_stock_analysis(analysis_data)
        except Exception as db_error:
            print(f"Database save error: {db_error}")
        
        # Save model performance for tracking
        try:
            save_model_performance(clean_symbol, float(predicted_price))
        except Exception as perf_error:
            print(f"Performance tracking error: {perf_error}")
        
        return JSONResponse({
            'success': True,
            'data': {
                'symbol': clean_symbol,
                'current_price': round(float(current_price), 2),
                'predicted_price': round(float(predicted_price), 2),
                'target_price': round(float(target_price), 2),
                'stop_loss': round(float(stop_loss), 2),
                'signal': signal,
                'confidence': confidence,
                'data_timestamp': current_timestamp,
                'indicators': {
                    'rsi': round(float(latest_data['RSI']), 2),
                    'ema_5': round(float(latest_data['EMA_5']), 2),
                    'macd': round(float(latest_data['MACD']), 4),
                    'macd_signal': round(float(latest_data['MACD_Signal']), 4),
                    'macd_histogram': round(float(latest_data['MACD_Histogram']), 4),
                    'sma_20': round(float(latest_data['SMA_20']), 2),
                    'bb_upper': round(float(latest_data['BB_Upper']), 2),
                    'bb_lower': round(float(latest_data['BB_Lower']), 2)
                },
                'model_stats': model_stats,
                'advanced_predictions': advanced_predictions,
                'model_confidence': model_confidence,
                'chart': chart_json,
                'prediction_change': round(float(((predicted_price - current_price) / current_price) * 100), 2),
                'investment_recommendation': investment_recommendation,
                'sentiment_analysis': sentiment_analysis,
                'fundamental_analysis': fundamental_analysis,
                'risk_analysis': risk_analysis
            }
        })
        
    except Exception as e:
        error_msg = str(e)
        if "No data found" in error_msg or "possibly delisted" in error_msg:
            # Provide intelligent suggestions for similar symbols
            suggestions = suggest_similar_symbols(original_symbol)
            suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            
            return JSONResponse({
                'success': False,
                'error': f'Stock symbol "{original_symbol}" not found or may be delisted.{suggestion_text} Common NSE symbols: RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, JSWSTEEL, MARUTI, SUNPHARMA, ULTRACEMCO, BHARTIARTL'
            }, status_code=400)
        else:
            return JSONResponse({
                'success': False,
                'error': f'Analysis error for {original_symbol}: {error_msg}'
            }, status_code=400)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Indian Stock Market Analyzer is running", "advanced_models": ADVANCED_MODELS_AVAILABLE}

@app.get("/api/ai-models")
async def get_ai_models_info():
    """Get information about available AI models"""
    return JSONResponse({
        "advanced_models_available": ADVANCED_MODELS_AVAILABLE,
        "models": {
            "traditional": "Linear Regression with Technical Indicators",
            "prophet": "Facebook Prophet Time Series" if ADVANCED_MODELS_AVAILABLE else "Not Available",
            "lstm": "LSTM Neural Network" if ADVANCED_MODELS_AVAILABLE else "Not Available", 
            "arima": "ARIMA Classical Model" if ADVANCED_MODELS_AVAILABLE else "Not Available",
            "ensemble": "AI Ensemble Prediction" if ADVANCED_MODELS_AVAILABLE else "Not Available"
        }
    })

@app.get("/api/popular-stocks")
async def get_popular_stocks_endpoint():
    """Get most searched stocks"""
    try:
        popular = get_popular_stocks(limit=10)
        return JSONResponse({"success": True, "data": popular})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/analysis-history/{symbol}")
async def get_analysis_history_endpoint(symbol: str):
    """Get analysis history for a symbol"""
    try:
        history = get_analysis_history(symbol.upper(), limit=10)
        return JSONResponse({"success": True, "data": history})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/model-stats")
async def get_model_stats_endpoint():
    """Get model accuracy statistics"""
    try:
        stats = get_model_accuracy_stats()
        return JSONResponse({"success": True, "data": stats})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/market-overview")
async def get_market_overview():
    """Get investment recommendations for popular stocks"""
    try:
        recommendations = dashboard.get_market_overview()
        return JSONResponse(recommendations)
    except Exception as e:
        return JSONResponse({
            'success': False,
            'error': f'Error getting market overview: {str(e)}'
        }, status_code=500)

@app.get("/upcoming-ipos")
async def get_upcoming_ipos():
    """Get upcoming IPOs for the next week"""
    try:
        ipo_data = ipo_tracker.get_upcoming_ipos()
        return JSONResponse(ipo_data)
    except Exception as e:
        return JSONResponse({
            'success': False,
            'error': f'Error fetching IPO data: {str(e)}',
            'ipos': [],
            'total_count': 0
        }, status_code=500)

@app.get("/ipo-details/{company_name}")
async def get_ipo_details(company_name: str):
    """Get detailed information about a specific IPO"""
    try:
        ipo_details = ipo_tracker.get_ipo_details(company_name)
        if ipo_details:
            return JSONResponse({
                'success': True,
                'ipo': ipo_details
            })
        else:
            return JSONResponse({
                'success': False,
                'error': f'IPO details not found for {company_name}'
            }, status_code=404)
    except Exception as e:
        return JSONResponse({
            'success': False,
            'error': f'Error fetching IPO details: {str(e)}'
        }, status_code=500)

@app.get("/ipo-calendar")
async def get_ipo_calendar(days: int = 30):
    """Get IPO calendar for specified number of days"""
    try:
        if days > 90:  # Limit to 90 days
            days = 90
        
        calendar_data = ipo_tracker.get_ipo_calendar(days)
        return JSONResponse(calendar_data)
    except Exception as e:
        return JSONResponse({
            'success': False,
            'error': f'Error fetching IPO calendar: {str(e)}',
            'ipos': [],
            'total_count': 0
        }, status_code=500)

@app.get("/api/volume/top")
async def get_top_volume_stocks():
    """
    Get today's top volume stocks (most bought and most sold)
    """
    print("\n" + "="*50)
    print("📡 /api/volume/top endpoint called")
    
    try:
        # Get the volume data
        print("🔄 Fetching top volume data...")
        result = volume_tracker.get_top_volume_stocks()
        print(f"✅ Got result from volume_tracker: {bool(result)}")
        
        # If there was an error in the volume tracker, return it
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown error in volume tracker')
            print(f"❌ Error in volume tracker: {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'debug': 'Error in volume_tracker.get_top_volume_stocks()',
                'data': {
                    'most_bought': [],
                    'most_sold': [],
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        # Structure the response to match what the frontend expects
        most_bought = result.get('most_bought', [])
        most_sold = result.get('most_sold', [])
        
        print(f"📊 Data stats - Most Bought: {len(most_bought)}, Most Sold: {len(most_sold)}")
        
        if most_bought:
            print("📈 Sample most bought:", {k: v for k, v in most_bought[0].items() if k != 'timestamp'})
        if most_sold:
            print("📉 Sample most sold:", {k: v for k, v in most_sold[0].items() if k != 'timestamp'})
            
        response_data = {
            'most_bought': most_bought,
            'most_sold': most_sold,
            'timestamp': result.get('timestamp', datetime.now().isoformat())
        }
        
        print("✅ Sending response with volume data")
        print("="*50 + "\n")
        
        return {
            'success': True,
            'data': response_data
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Unexpected error in /api/volume/top: {str(e)}")
        print(error_trace)
        
        return {
            'success': False, 
            'error': f'Failed to fetch volume data: {str(e)}',
            'debug': error_trace,
            'data': {
                'most_bought': [],
                'most_sold': [],
                'timestamp': datetime.now().isoformat()
            }
        }

@app.get("/api/volume/test")
async def test_volume_data():
    """Test endpoint to check volume tracker functionality"""
    try:
        # Test with a known stock
        test_symbol = 'RELIANCE'
        print(f"Testing with symbol: {test_symbol}")
        
        # Get data directly
        stock_data = volume_tracker.get_stock_volume_data(test_symbol)
        print(f"Stock data for {test_symbol}:", stock_data)
        
        # Get top volume data
        top_data = volume_tracker.get_top_volume_stocks()
        print("Top volume data:", top_data)
        
        return {
            "success": True,
            "stock_data": stock_data,
            "top_data": top_data
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in test_volume_data: {error_trace}")
        return {"success": False, "error": str(e), "trace": error_trace}

@app.get("/api/volume/stock/{symbol}")
async def get_stock_volume_data(symbol: str):
    """
    Get volume data for a specific stock
    """
    try:
        data = volume_tracker.get_stock_volume_data(symbol.upper())
        if data:
            return {"success": True, "data": data}
        else:
            return {"success": False, "error": f"No data found for {symbol}"}
    except Exception as e:
        import traceback
        print(f"Error in get_stock_volume_data: {traceback.format_exc()}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
"""
Market Dashboard - Batch Stock Analysis
Shows investment recommendations for multiple stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from investment_advisor import InvestmentAdvisor
import pandas_ta as ta

class MarketDashboard:
    def __init__(self):
        self.advisor = InvestmentAdvisor()
        self.popular_stocks = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK',
            'MARUTI', 'JSWSTEEL', 'ADANIGREEN', 'BAJFINANCE', 'WIPRO'
        ]
    
    def get_stock_recommendation(self, symbol):
        """Get quick investment recommendation for a stock"""
        try:
            # Fetch stock data
            ticker = yf.Ticker(f"{symbol}.NS")
            data = ticker.history(period="60d")
            
            if data.empty:
                return None
            
            current_price = float(data['Close'].iloc[-1])
            
            # Calculate basic technical indicators
            data['RSI'] = ta.rsi(data['Close'], length=14)
            data['EMA_5'] = ta.ema(data['Close'], length=5)
            data['MACD'] = ta.macd(data['Close'])['MACD_12_26_9']
            data['SMA_20'] = ta.sma(data['Close'], length=20)
            
            # Quick technical signal
            rsi = data['RSI'].iloc[-1]
            ema_5 = data['EMA_5'].iloc[-1]
            macd = data['MACD'].iloc[-1]
            sma_20 = data['SMA_20'].iloc[-1]
            
            # Generate signal
            if rsi < 30 and current_price > ema_5 and macd > 0:
                signal = "BUY"
                confidence = "High"
            elif rsi > 70 and current_price < ema_5 and macd < 0:
                signal = "SELL"
                confidence = "High"
            elif rsi < 40 and current_price > sma_20:
                signal = "BUY"
                confidence = "Medium"
            elif rsi > 60 and current_price < sma_20:
                signal = "SELL"
                confidence = "Medium"
            else:
                signal = "HOLD"
                confidence = "Medium"
            
            # Simple price prediction (moving average)
            predicted_price = (current_price + ema_5 + sma_20) / 3
            
            # Get comprehensive analysis
            sentiment_analysis = self.advisor.analyze_market_sentiment(symbol, data)
            fundamental_analysis = self.advisor.fundamental_analysis(symbol, current_price)
            risk_analysis = self.advisor.risk_analysis(data, current_price, predicted_price)
            
            # Calculate targets
            if signal == "BUY":
                target_price = current_price * 1.08  # 8% upside
                stop_loss = current_price * 0.95     # 5% downside
            elif signal == "SELL":
                target_price = current_price * 0.95  # 5% downside
                stop_loss = current_price * 1.03     # 3% upside
            else:
                target_price = current_price * 1.02  # 2% upside
                stop_loss = current_price * 0.98     # 2% downside
            
            # Generate investment recommendation
            investment_rec = self.advisor.generate_investment_recommendation(
                symbol, signal, confidence, sentiment_analysis, fundamental_analysis, 
                risk_analysis, current_price, predicted_price, target_price, stop_loss
            )
            
            return {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'signal': signal,
                'confidence': confidence,
                'recommendation': investment_rec['recommendation'],
                'action_color': investment_rec['action_color'],
                'risk_level': investment_rec['risk_level'],
                'position_size': investment_rec['position_size'],
                'potential_return': investment_rec['potential_return'],
                'sector': sentiment_analysis['sector'],
                'momentum_1w': round(sentiment_analysis['momentum_1w'], 1),
                'volatility': round(risk_analysis['volatility'], 1),
                'key_reason': investment_rec['key_reasons'][0] if investment_rec['key_reasons'] else 'Technical analysis',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def get_market_overview(self):
        """Get investment recommendations for all popular stocks"""
        recommendations = []
        
        for symbol in self.popular_stocks:
            rec = self.get_stock_recommendation(symbol)
            if rec:
                recommendations.append(rec)
        
        # Sort by recommendation strength
        recommendation_order = {'STRONG BUY': 5, 'BUY': 4, 'HOLD': 3, 'SELL': 2, 'STRONG SELL': 1}
        recommendations.sort(key=lambda x: recommendation_order.get(x['recommendation'], 3), reverse=True)
        
        return recommendations
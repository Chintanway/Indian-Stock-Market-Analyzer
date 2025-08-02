"""
Advanced Investment Advisory System
Provides comprehensive buy/sell recommendations based on multiple analysis factors
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

class InvestmentAdvisor:
    def __init__(self):
        self.risk_levels = {
            'Conservative': {'max_risk': 5, 'min_return': 8},
            'Moderate': {'max_risk': 10, 'min_return': 12},
            'Aggressive': {'max_risk': 20, 'min_return': 18}
        }
    
    def analyze_market_sentiment(self, symbol, data):
        """Analyze overall market sentiment and sector performance"""
        try:
            # Get sector data for comparison
            sector_symbols = {
                'RELIANCE': 'Energy',
                'TCS': 'IT',
                'INFY': 'IT', 
                'HDFCBANK': 'Banking',
                'ICICIBANK': 'Banking',
                'MARUTI': 'Auto',
                'JSWSTEEL': 'Steel',
                'ADANIGREEN': 'Renewable Energy'
            }
            
            current_sector = sector_symbols.get(symbol, 'General')
            
            # Calculate price momentum over different periods
            price_1w = data['Close'].iloc[-5:].mean() if len(data) >= 5 else data['Close'].iloc[-1]
            price_1m = data['Close'].iloc[-20:].mean() if len(data) >= 20 else data['Close'].iloc[-1]
            price_3m = data['Close'].iloc[-60:].mean() if len(data) >= 60 else data['Close'].iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            momentum_1w = ((current_price - price_1w) / price_1w) * 100
            momentum_1m = ((current_price - price_1m) / price_1m) * 100
            momentum_3m = ((current_price - price_3m) / price_3m) * 100
            
            # Volume analysis
            avg_volume = data['Volume'].tail(20).mean()
            recent_volume = data['Volume'].tail(5).mean()
            volume_surge = (recent_volume / avg_volume) * 100 if avg_volume > 0 else 100
            
            sentiment_score = 0
            factors = []
            
            # Momentum scoring
            if momentum_1w > 2:
                sentiment_score += 2
                factors.append(f"Strong 1-week momentum: +{momentum_1w:.1f}%")
            elif momentum_1w < -2:
                sentiment_score -= 2
                factors.append(f"Weak 1-week momentum: {momentum_1w:.1f}%")
            
            if momentum_1m > 5:
                sentiment_score += 3
                factors.append(f"Strong 1-month momentum: +{momentum_1m:.1f}%")
            elif momentum_1m < -5:
                sentiment_score -= 3
                factors.append(f"Weak 1-month momentum: {momentum_1m:.1f}%")
            
            # Volume analysis
            if volume_surge > 150:
                sentiment_score += 2
                factors.append(f"High volume activity: {volume_surge:.0f}% of average")
            elif volume_surge < 70:
                sentiment_score -= 1
                factors.append(f"Low volume activity: {volume_surge:.0f}% of average")
            
            return {
                'sentiment_score': sentiment_score,
                'sector': current_sector,
                'momentum_1w': momentum_1w,
                'momentum_1m': momentum_1m,
                'momentum_3m': momentum_3m,
                'volume_surge': volume_surge,
                'factors': factors
            }
            
        except Exception as e:
            print(f"Market sentiment analysis error: {e}")
            return {
                'sentiment_score': 0,
                'sector': 'Unknown',
                'momentum_1w': 0,
                'momentum_1m': 0,
                'momentum_3m': 0,
                'volume_surge': 100,
                'factors': []
            }
    
    def fundamental_analysis(self, symbol, current_price):
        """Basic fundamental analysis using available data"""
        try:
            ticker = yf.Ticker(f"{symbol}.NS")
            info = ticker.info
            
            analysis = {
                'pe_ratio': info.get('trailingPE', 0),
                'market_cap': info.get('marketCap', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margin': info.get('profitMargins', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
            
            # Scoring based on fundamentals
            fundamental_score = 0
            fundamental_factors = []
            
            # P/E Ratio analysis
            pe = analysis['pe_ratio']
            if 0 < pe < 15:
                fundamental_score += 3
                fundamental_factors.append(f"Attractive P/E ratio: {pe:.1f}")
            elif 15 <= pe <= 25:
                fundamental_score += 1
                fundamental_factors.append(f"Reasonable P/E ratio: {pe:.1f}")
            elif pe > 35:
                fundamental_score -= 2
                fundamental_factors.append(f"High P/E ratio: {pe:.1f}")
            
            # Profit margin
            if analysis['profit_margin'] > 0.15:
                fundamental_score += 2
                fundamental_factors.append(f"Strong profit margin: {analysis['profit_margin']*100:.1f}%")
            elif analysis['profit_margin'] < 0.05:
                fundamental_score -= 1
                fundamental_factors.append(f"Low profit margin: {analysis['profit_margin']*100:.1f}%")
            
            # Dividend yield
            if analysis['dividend_yield'] > 0.02:
                fundamental_score += 1
                fundamental_factors.append(f"Good dividend yield: {analysis['dividend_yield']*100:.1f}%")
            
            analysis['fundamental_score'] = fundamental_score
            analysis['fundamental_factors'] = fundamental_factors
            
            return analysis
            
        except Exception as e:
            print(f"Fundamental analysis error: {e}")
            return {
                'fundamental_score': 0,
                'fundamental_factors': [],
                'pe_ratio': 0,
                'market_cap': 0,
                'profit_margin': 0
            }
    
    def risk_analysis(self, data, current_price, predicted_price):
        """Calculate risk metrics and volatility"""
        try:
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            
            # Volatility (standard deviation of returns)
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            # Maximum drawdown
            rolling_max = data['Close'].expanding().max()
            drawdown = (data['Close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # Beta calculation (approximate using market proxy)
            # Using simple volatility as risk proxy since we don't have market index
            
            # Price risk based on prediction
            prediction_risk = abs((predicted_price - current_price) / current_price) * 100
            
            # Risk scoring
            risk_score = 0
            risk_factors = []
            
            if volatility < 15:
                risk_score += 2
                risk_factors.append(f"Low volatility: {volatility:.1f}%")
            elif volatility > 30:
                risk_score -= 3
                risk_factors.append(f"High volatility: {volatility:.1f}%")
            else:
                risk_factors.append(f"Moderate volatility: {volatility:.1f}%")
            
            if max_drawdown > -20:
                risk_score += 1
                risk_factors.append(f"Controlled drawdown: {max_drawdown:.1f}%")
            else:
                risk_score -= 2
                risk_factors.append(f"High drawdown risk: {max_drawdown:.1f}%")
            
            return {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'prediction_risk': prediction_risk,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'risk_level': 'Low' if risk_score > 0 else 'High' if risk_score < -1 else 'Medium'
            }
            
        except Exception as e:
            print(f"Risk analysis error: {e}")
            return {
                'volatility': 20,
                'max_drawdown': -15,
                'prediction_risk': 5,
                'risk_score': 0,
                'risk_factors': [],
                'risk_level': 'Medium'
            }
    
    def generate_investment_recommendation(self, symbol, technical_signal, technical_confidence, 
                                         sentiment_analysis, fundamental_analysis, risk_analysis,
                                         current_price, predicted_price, target_price, stop_loss):
        """Generate comprehensive investment recommendation"""
        
        # Calculate total score
        technical_score = 3 if technical_signal == "BUY" else -3 if technical_signal == "SELL" else 0
        if technical_confidence == "High":
            technical_score *= 1.5
        
        total_score = (
            technical_score +
            sentiment_analysis['sentiment_score'] +
            fundamental_analysis['fundamental_score'] +
            risk_analysis['risk_score']
        )
        
        # Generate recommendation
        if total_score >= 6:
            recommendation = "STRONG BUY"
            action_color = "success"
        elif total_score >= 3:
            recommendation = "BUY"
            action_color = "primary"
        elif total_score >= -2:
            recommendation = "HOLD"
            action_color = "warning"
        elif total_score >= -5:
            recommendation = "SELL"
            action_color = "danger"
        else:
            recommendation = "STRONG SELL"
            action_color = "dark"
        
        # Calculate investment horizon
        risk_level = risk_analysis['risk_level']
        if risk_level == 'Low':
            investment_horizon = "Long-term (1-2 years)"
        elif risk_level == 'Medium':
            investment_horizon = "Medium-term (6-12 months)"
        else:
            investment_horizon = "Short-term (1-3 months)"
        
        # Calculate position size suggestion
        if recommendation in ["STRONG BUY", "BUY"]:
            if risk_level == 'Low':
                position_size = "5-10% of portfolio"
            elif risk_level == 'Medium':
                position_size = "3-7% of portfolio"
            else:
                position_size = "1-3% of portfolio"
        else:
            position_size = "Avoid new positions"
        
        # Key reasons
        key_reasons = []
        
        # Technical reasons
        if technical_signal == "BUY":
            key_reasons.append(f"Technical analysis shows {technical_signal} signal with {technical_confidence} confidence")
        elif technical_signal == "SELL":
            key_reasons.append(f"Technical indicators suggest {technical_signal} with caution")
        
        # Add top sentiment factors
        key_reasons.extend(sentiment_analysis['factors'][:2])
        
        # Add top fundamental factors
        key_reasons.extend(fundamental_analysis['fundamental_factors'][:2])
        
        # Add risk factors
        key_reasons.extend(risk_analysis['risk_factors'][:1])
        
        return {
            'recommendation': recommendation,
            'action_color': action_color,
            'total_score': round(total_score, 1),
            'confidence_level': "High" if abs(total_score) >= 5 else "Medium" if abs(total_score) >= 2 else "Low",
            'investment_horizon': investment_horizon,
            'position_size': position_size,
            'risk_level': risk_level,
            'key_reasons': key_reasons[:5],  # Top 5 reasons
            'entry_price': current_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'potential_return': round(((target_price - current_price) / current_price) * 100, 1),
            'risk_reward_ratio': round(abs(target_price - current_price) / abs(current_price - stop_loss), 2) if stop_loss != current_price else 1.0
        }
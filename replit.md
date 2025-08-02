# Indian Stock Market Analyzer

## Overview

A comprehensive FastAPI-based Indian stock market analyzer with AI-powered predictions and real technical analysis. The application provides advanced technical indicators, machine learning predictions, and trading signals for NSE stocks.

**Built:** July 25, 2025  
**Status:** Production Ready  

## User Preferences

- Preferred communication style: Simple, everyday language
- Focus on real technical analysis with authentic data
- Use NSE stock symbols (RELIANCE, TCS, INFY, etc.)
- No mock or placeholder data - only real market data

## System Architecture

**Full-Stack Web Application:**
- FastAPI backend with Python 3.11
- PostgreSQL database for data persistence
- HTML/CSS/JavaScript frontend with Bootstrap 5
- Real-time stock data from Yahoo Finance
- Local AI model using scikit-learn
- Interactive charts with Plotly

## Key Components

**Backend (FastAPI)**
- `/` - Main application interface
- `/analyze` - Stock analysis API endpoint
- `/api/popular-stocks` - Most searched stocks
- `/api/analysis-history/{symbol}` - Historical analysis data
- `/api/model-stats` - AI model performance statistics
- `/health` - Health check endpoint
- Real-time NSE stock data fetching
- Technical indicator calculations
- AI price prediction model
- Trading signal generation

**Frontend (HTML/CSS/JS)**
- Responsive Bootstrap 5 interface
- Real-time stock analysis form
- Interactive Plotly charts
- Technical indicators dashboard
- AI prediction display
- Popular stock quick-select buttons

**AI/ML Components**
- Linear Regression model (scikit-learn)
- Feature engineering from technical indicators
- Model training on historical data
- Real-time price predictions
- Performance tracking and accuracy metrics

**Database (PostgreSQL)**
- Stock analysis history storage
- User search pattern tracking
- Model performance monitoring
- Popular stocks based on user searches
- Historical prediction accuracy tracking

## Technical Indicators

**Implemented Indicators:**
- RSI (Relative Strength Index) - 14 period
- EMA (Exponential Moving Average) - 5 period
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands (20 period)
- SMA (Simple Moving Average) - 20, 50 periods
- Volume analysis

**Trading Signals:**
- BUY: RSI < 30, MACD positive, predicted increase > 2%
- SELL: RSI > 70, MACD negative, predicted decrease > 2%
- CAUTION: Mixed signals
- NEUTRAL: Stable conditions

## Data Flow

1. User enters NSE stock symbol (e.g., RELIANCE, TCS)
2. Backend fetches 60-day historical data from Yahoo Finance
3. Calculate technical indicators using pandas-ta
4. Train AI model on indicator features
5. Generate next-day price prediction
6. Determine trading signal and targets
7. Create interactive charts
8. Display comprehensive analysis

## External Dependencies

**Core Libraries:**
- `yfinance` - Real NSE stock data
- `pandas-ta` - Technical analysis indicators
- `scikit-learn` - Machine learning models
- `plotly` - Interactive charts
- `fastapi` - Web framework
- `uvicorn` - ASGI server

**Data Sources:**
- Yahoo Finance API (NSE stocks with .NS suffix)
- Real-time stock prices and volumes
- Historical price data for training

## Deployment Strategy

- **Platform:** Replit Deployments
- **Port:** 5000 (configured for Replit)
- **Server:** Uvicorn ASGI server
- **Static Files:** CSS/JS served via FastAPI
- **Templates:** Jinja2 templating
- **Health Check:** `/health` endpoint available

## Recent Changes

✓ Created complete FastAPI backend with StockAnalyzer class  
✓ Implemented all technical indicators (RSI, EMA, MACD, Bollinger Bands)  
✓ Built AI prediction model with Linear Regression  
✓ Designed responsive frontend with Bootstrap 5  
✓ Added interactive Plotly charts  
✓ Fixed numpy compatibility issues with pandas-ta  
✓ Configured Replit workflow for port 5000  
✓ Added popular NSE stock quick-select buttons  
✓ Integrated PostgreSQL database for data persistence  
✓ Added analysis history and popular stocks tracking  
✓ Implemented model performance monitoring  
✓ Enhanced with advanced AI models: Facebook Prophet, ARIMA  
✓ Fixed real-time price accuracy with multiple data sources  
✓ Improved stop-loss calculations using technical analysis  
✓ Added AI ensemble prediction system  
✓ Fixed real-time timestamp display issue  
✓ **Built comprehensive Investment Advisory System**  
✓ Added market sentiment analysis and sector performance  
✓ Integrated fundamental analysis with P/E ratios and financials  
✓ Implemented risk analysis with volatility and drawdown metrics  
✓ Created STRONG BUY/BUY/HOLD/SELL/STRONG SELL recommendations  
✓ Added position sizing and investment horizon suggestions
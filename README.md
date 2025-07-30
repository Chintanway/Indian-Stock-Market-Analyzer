# StockInsightAI

StockInsightAI is a comprehensive Indian stock market analysis platform that provides AI-powered technical analysis, investment recommendations, and market insights for NSE (National Stock Exchange) listed stocks.

## Features

- **Stock Analysis**: Technical indicators (RSI, MACD, Moving Averages)
- **AI-Powered Predictions**: Price prediction using machine learning models
- **Investment Recommendations**: Buy/Sell signals with confidence levels
- **Market Dashboard**: Overview of popular stocks and sector performance
- **User-Friendly Interface**: Interactive charts and visualizations

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd StockInsightAI
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. The application uses SQLite by default, which will be automatically created at `StockInsightAI/stockinsight.db` when you first run the application.

2. (Optional) To use a different database, set the `DATABASE_URL` environment variable:
   ```bash
   # For SQLite (default)
   set DATABASE_URL=sqlite:///stockinsight.db
   
   # For PostgreSQL (example)
   # set DATABASE_URL=postgresql://username:password@localhost:5432/stockinsight
   ```

## Running the Application

1. Start the FastAPI development server:
   ```bash
   uvicorn StockInsightAI.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Access the application:
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Interactive API Docs: http://localhost:8000/redoc

## Project Structure

```
StockInsightAI/
├── advanced_models.py    # Advanced prediction models
├── database.py          # Database models and operations
├── investment_advisor.py # Investment recommendation engine
├── main.py              # Main FastAPI application
├── market_dashboard.py  # Market overview and batch analysis
├── nse_stocks.py        # NSE stock symbol mapping
├── requirements.txt     # Project dependencies
└── static/              # Static files (CSS, JS, images)
    └── ...
└── templates/           # HTML templates
    └── index.html       # Main dashboard template
```

## Usage

1. **Search for a Stock**: Enter the NSE stock symbol (e.g., RELIANCE, TCS, INFY)
2. **View Analysis**: See technical indicators, price predictions, and recommendations
3. **Explore Market**: Check the market dashboard for popular stocks and sector performance

## API Endpoints

- `GET /`: Main web interface
- `POST /analyze`: Analyze a stock
- `GET /api/popular-stocks`: Get most searched stocks
- `GET /api/analysis-history/{symbol}`: Get analysis history for a symbol
- `GET /api/health`: Health check endpoint

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with FastAPI, SQLAlchemy, and yfinance
- Uses pandas-ta for technical analysis
- Inspired by modern fintech applications

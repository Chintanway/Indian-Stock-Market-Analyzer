"""
Stock Prediction Test Script
--------------------------
This script demonstrates how to use the AdvancedStockPredictor class
to get stock predictions for Indian stocks.
"""

from advanced_models import AdvancedStockPredictor
from datetime import datetime
import pandas as pd

def print_prediction(result):
    """Helper function to print prediction results"""
    if 'error' in result:
        print(f"\n❌ Error for {result.get('symbol', 'unknown')}:")
        print(f"   {result['details']}")
        return
    
    print(f"\n📈 {result['name']} ({result['symbol']})")
    print("-" * 50)
    print(f"💵 Current Price:  ₹{result['last_price']:,.2f}")
    print(f"🎯 Predicted:      ₹{result['predicted_price']:,.2f}")
    print(f"📊 Signal:         {result['signal']} (Potential: {result['potential_return']:+.2f}%)")
    if 'confidence_interval' in result and result['confidence_interval']:
        print(f"📏 Confidence:     ₹{result['confidence_interval'][0]:,.2f} - ₹{result['confidence_interval'][1]:,.2f}")
    print(f"📈 Volatility:     {result.get('volatility', 'N/A')}%")
    print(f"🏛️  Exchange:       {result.get('exchange', 'N/A')}")
    print(f"🏭 Sector:         {result.get('sector', 'N/A')}")
    print(f"🔄 Last Updated:   {result.get('last_updated', 'N/A')}")

def main():
    print("🚀 StockInsightAI - Advanced Stock Predictor")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = AdvancedStockPredictor()
    
    # List of popular Indian stocks with their Yahoo Finance symbols
    stocks = [
        'RELIANCE.NS',     # Reliance Industries (NSE)
        'TCS.NS',          # Tata Consultancy Services (NSE)
        'HDFCBANK.NS',     # HDFC Bank (NSE)
        'INFY.NS',         # Infosys (NSE)
        'ICICIBANK.NS',    # ICICI Bank (NSE)
        'HINDUNILVR.NS',   # Hindustan Unilever (NSE)
        'ITC.NS',          # ITC Limited (NSE)
        'BHARTIARTL.NS',   # Bharti Airtel (NSE)
        'LT.NS',           # Larsen & Toubro (NSE)
        'KOTAKBANK.NS',    # Kotak Mahindra Bank (NSE)
        'AXISBANK.NS',     # Axis Bank (NSE)
        'ASIANPAINT.NS',   # Asian Paints (NSE)
        'HCLTECH.NS',      # HCL Technologies (NSE)
        'MARUTI.NS',       # Maruti Suzuki (NSE)
        'SUNPHARMA.NS',    # Sun Pharma (NSE)
        'TATAMOTORS.NS',   # Tata Motors (NSE)
        'BAJFINANCE.NS',   # Bajaj Finance (NSE)
        'WIPRO.NS',        # Wipro (NSE)
        'ONGC.NS',         # Oil & Natural Gas Corp (NSE)
        'NESTLEIND.NS'     # Nestle India (NSE)
    ]
    
    # Create or clear the CSV file
    try:
        with open('stock_predictions.csv', 'w') as f:
            f.write('')  # Clear the file
    except:
        pass
    
    # Process each stock
    for symbol in stocks:
        print(f"\n🔍 Analyzing {symbol}...")
        try:
            # Get prediction with a longer period for more data
            result = predictor.predict_next_day(symbol, period='2y')
            print_prediction(result)
            
            # Save results to CSV
            if 'error' not in result:
                df = pd.DataFrame([result])
                df.to_csv('stock_predictions.csv', 
                         mode='a', 
                         header=not pd.io.common.file_exists('stock_predictions.csv'),
                         index=False,
                         columns=['symbol', 'name', 'last_price', 'predicted_price', 
                                 'signal', 'potential_return', 'volatility', 'exchange', 'sector'])
                
        except Exception as e:
            print(f"❌ Error processing {symbol}: {str(e)}")
    
    print("\n✅ Analysis complete! Check 'stock_predictions.csv' for detailed results.")

if __name__ == "__main__":
    main()

"""
Simple test script to check yfinance functionality
"""
import yfinance as yf

def test_yfinance():
    print("Testing yfinance with RELIANCE.NS...")
    
    try:
        # Try to get data for RELIANCE
        stock = yf.Ticker("RELIANCE.NS")
        print("\nStock info:", stock.info.get('longName', 'No name found'))
        
        # Get historical data
        hist = stock.history(period="2d")
        print("\nHistorical data:")
        print(hist)
        
        if not hist.empty:
            print("\nLatest data:")
            latest = hist.iloc[-1]
            print(f"Date: {latest.name}")
            print(f"Open: {latest['Open']}")
            print(f"High: {latest['High']}")
            print(f"Low: {latest['Low']}")
            print(f"Close: {latest['Close']}")
            print(f"Volume: {latest['Volume']}")
        else:
            print("\nNo historical data available")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_yfinance()

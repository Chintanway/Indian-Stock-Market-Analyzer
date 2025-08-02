"""
Direct test of the volume tracker functionality
"""
import sys
import os
import json
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from volume_tracker import VolumeTracker

def test_volume_tracker():
    print("🚀 Starting Volume Tracker Test...")
    
    # Initialize the tracker
    tracker = VolumeTracker()
    
    print("\n🔍 Testing with RELIANCE stock...")
    
    # Test with a known stock
    symbol = 'RELIANCE'
    stock_data = tracker.get_stock_volume_data(symbol)
    
    if stock_data:
        print("✅ Successfully fetched stock data:")
        print(json.dumps(stock_data, indent=2, default=str))
    else:
        print("❌ Failed to fetch stock data")
        return
    
    print("\n📊 Fetching top volume data...")
    
    # Get top volume data
    result = tracker.get_top_volume_stocks()
    
    if result.get('success', False):
        data = result.get('data', {})
        print("✅ Successfully fetched top volume data")
        
        # Print most bought
        print("\n🔼 Most Bought Stocks:")
        for i, stock in enumerate(data.get('most_bought', [])[:5], 1):
            print(f"{i}. {stock.get('symbol')} - {stock.get('company_name')} "
                  f"(Volume: {stock.get('volume'):,}, Change: {stock.get('volume_change_pct', 0):.2f}%)")
        
        # Print most sold
        print("\n🔽 Most Sold Stocks:")
        for i, stock in enumerate(data.get('most_sold', [])[:5], 1):
            print(f"{i}. {stock.get('symbol')} - {stock.get('company_name')} "
                  f"(Volume: {stock.get('volume'):,}, Change: {stock.get('volume_change_pct', 0):.2f}%)")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_volume_tracker()

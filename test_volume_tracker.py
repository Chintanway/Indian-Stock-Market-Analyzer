"""
Test script for Volume Tracker functionality
"""
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from volume_tracker import VolumeTracker

def test_volume_tracker():
    print("Testing Volume Tracker...")
    
    # Initialize the tracker
    tracker = VolumeTracker()
    
    # Test getting data for a single stock
    print("\nTesting get_stock_volume_data for RELIANCE:")
    reliance_data = tracker.get_stock_volume_data('RELIANCE')
    if reliance_data:
        print(f"Successfully fetched data for RELIANCE:")
        for key, value in reliance_data.items():
            print(f"  {key}: {value}")
    else:
        print("Failed to fetch data for RELIANCE")
    
    # Test getting top volume stocks
    print("\nTesting get_top_volume_stocks:")
    result = tracker.get_top_volume_stocks()
    
    if result.get('success', False):
        print("Successfully fetched top volume data")
        
        # Print most bought stocks
        print("\nMost Bought Stocks:")
        for i, stock in enumerate(result.get('most_bought', [])[:5], 1):
            print(f"{i}. {stock.get('symbol')} - {stock.get('company_name')} "
                  f"(Volume: {stock.get('volume'):,}, Change: {stock.get('volume_change_pct'):.2f}%)")
        
        # Print most sold stocks
        print("\nMost Sold Stocks:")
        for i, stock in enumerate(result.get('most_sold', [])[:5], 1):
            print(f"{i}. {stock.get('symbol')} - {stock.get('company_name')} "
                  f"(Volume: {stock.get('volume'):,}, Change: {stock.get('volume_change_pct'):.2f}%)")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_volume_tracker()

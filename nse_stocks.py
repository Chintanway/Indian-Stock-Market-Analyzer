"""
NSE Stock Symbol Mapping and Validation
Comprehensive database of NSE stock symbols with their correct Yahoo Finance mappings
"""

# Comprehensive NSE stock symbol mapping
NSE_STOCK_MAPPING = {
    # Major IT Companies
    'TCS': 'TCS.NS',
    'INFY': 'INFY.NS',
    'WIPRO': 'WIPRO.NS',
    'HCLTECH': 'HCLTECH.NS',
    'TECHM': 'TECHM.NS',
    'LTIM': 'LTIM.NS',
    'MPHASIS': 'MPHASIS.NS',
    'COFORGE': 'COFORGE.NS',
    'MINDTREE': 'MINDTREE.NS',
    
    # Banking & Finance
    'HDFCBANK': 'HDFCBANK.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'SBIN': 'SBIN.NS',
    'AXISBANK': 'AXISBANK.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'INDUSINDBK': 'INDUSINDBK.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'BAJAJFINSV': 'BAJAJFINSV.NS',
    'HDFCLIFE': 'HDFCLIFE.NS',
    'SBILIFE': 'SBILIFE.NS',
    'ICICIGI': 'ICICIGI.NS',
    'HDFCAMC': 'HDFCAMC.NS',
    
    # Energy & Oil
    'RELIANCE': 'RELIANCE.NS',
    'ONGC': 'ONGC.NS',
    'IOC': 'IOC.NS',
    'BPCL': 'BPCL.NS',
    'HINDPETRO': 'HINDPETRO.NS',
    'GAIL': 'GAIL.NS',
    'NTPC': 'NTPC.NS',
    'POWERGRID': 'POWERGRID.NS',
    'COALINDIA': 'COALINDIA.NS',
    'ADANIGREEN': 'ADANIGREEN.NS',
    'ADANIPOWER': 'ADANIPOWER.NS',
    'TATAPOWER': 'TATAPOWER.NS',
    
    # Automotive
    'MARUTI': 'MARUTI.NS',
    'TATAMOTORS': 'TATAMOTORS.NS',
    'M&M': 'M&M.NS',
    'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
    'HEROMOTOCO': 'HEROMOTOCO.NS',
    'TVSMOTOR': 'TVSMOTOR.NS',
    'EICHERMOT': 'EICHERMOT.NS',
    'ASHOKLEY': 'ASHOKLEY.NS',
    'BHARATFORG': 'BHARATFORG.NS',
    'MOTHERSON': 'MOTHERSON.NS',
    
    # Metals & Mining
    'JSWSTEEL': 'JSWSTEEL.NS',
    'TATASTEEL': 'TATASTEEL.NS',
    'HINDALCO': 'HINDALCO.NS',
    'SAIL': 'SAIL.NS',
    'JINDALSTEL': 'JINDALSTEL.NS',
    'NMDC': 'NMDC.NS',
    'VEDL': 'VEDL.NS',
    'NATIONALUM': 'NATIONALUM.NS',
    'HINDZINC': 'HINDZINC.NS',
    'RATNAMANI': 'RATNAMANI.NS',
    
    # Pharmaceuticals
    'SUNPHARMA': 'SUNPHARMA.NS',
    'DRREDDY': 'DRREDDY.NS',
    'CIPLA': 'CIPLA.NS',
    'DIVISLAB': 'DIVISLAB.NS',
    'BIOCON': 'BIOCON.NS',
    'AUROPHARMA': 'AUROPHARMA.NS',
    'LUPIN': 'LUPIN.NS',
    'GLENMARK': 'GLENMARK.NS',
    'TORNTPHARM': 'TORNTPHARM.NS',
    'ALKEM': 'ALKEM.NS',
    'VIMTALABS': 'VIMTALABS.NS',
    'LAURUSLABS': 'LAURUSLABS.NS',
    'NATCOPHARM': 'NATCOPHARM.NS',
    'CADILAHC': 'CADILAHC.NS',
    
    # Consumer Goods
    'HINDUNILVR': 'HINDUNILVR.NS',
    'ITC': 'ITC.NS',
    'NESTLEIND': 'NESTLEIND.NS',
    'BRITANNIA': 'BRITANNIA.NS',
    'DABUR': 'DABUR.NS',
    'MARICO': 'MARICO.NS',
    'GODREJCP': 'GODREJCP.NS',
    'COLPAL': 'COLPAL.NS',
    'TATACONSUM': 'TATACONSUM.NS',
    'UBL': 'UBL.NS',
    
    # Cement
    'ULTRACEMCO': 'ULTRACEMCO.NS',
    'SHREECEM': 'SHREECEM.NS',
    'GRASIM': 'GRASIM.NS',
    'ACC': 'ACC.NS',
    'AMBUJCEM': 'AMBUJCEM.NS',
    'RAMCOCEM': 'RAMCOCEM.NS',
    'JKCEMENT': 'JKCEMENT.NS',
    'HEIDELBERG': 'HEIDELBERG.NS',
    
    # Telecom
    'BHARTIARTL': 'BHARTIARTL.NS',
    'IDEA': 'IDEA.NS',
    'MTNL': 'MTNL.NS',
    'GTPL': 'GTPL.NS',
    
    # Real Estate
    'DLF': 'DLF.NS',
    'GODREJPROP': 'GODREJPROP.NS',
    'OBEROI': 'OBEROI.NS',
    'BRIGADE': 'BRIGADE.NS',
    'SOBHA': 'SOBHA.NS',
    'PRESTIGE': 'PRESTIGE.NS',
    
    # Technology Hardware
    'DIXON': 'DIXON.NS',
    'AMBER': 'AMBER.NS',
    'ROUTE': 'ROUTE.NS',
    
    # Airlines & Travel  
    'INDIGO': 'INDIGO.NS',
    'SPICEJET': 'SPICEJET.NS',
    'IRCTC': 'IRCTC.NS',
    'EASEMYTRIP': 'EASEMYTRIP.NS',
    
    # Textiles
    'ARVIND': 'ARVIND.NS',
    'WELSPUNIND': 'WELSPUNIND.NS',
    'PAGEIND': 'PAGEIND.NS',
    'RTNPOWER': 'RTNPOWER.NS',
    
    # Food & Beverages
    'JUBLFOOD': 'JUBLFOOD.NS',
    'VBL': 'VBL.NS',
    'DEVYANI': 'DEVYANI.NS',
    'WESTLIFE': 'WESTLIFE.NS',
    'ZOMATO': 'ZOMATO.NS',
    
    # E-commerce & New Age
    'NYKAA': 'NYKAA.NS',
    'PAYTM': 'PAYTM.NS',
    'POLICYBZR': 'POLICYBZR.NS',
    'CARTRADE': 'CARTRADE.NS',
    
    # Engineering & Capital Goods
    'LT': 'LT.NS',
    'SIEMENS': 'SIEMENS.NS',
    'ABB': 'ABB.NS',
    'BHEL': 'BHEL.NS',
    'THERMAX': 'THERMAX.NS',
    'CUMMINSIND': 'CUMMINSIND.NS',
    'VOLTAS': 'VOLTAS.NS',
    'CROMPTON': 'CROMPTON.NS',
    'HAVELLS': 'HAVELLS.NS',
    'POLYCAB': 'POLYCAB.NS',
    
    # Chemicals
    'PIDILITIND': 'PIDILITIND.NS',
    'AARTI': 'AARTI.NS',
    'BALRAMCHIN': 'BALRAMCHIN.NS',
    'GHCL': 'GHCL.NS',
    'TATACHEM': 'TATACHEM.NS',
    'DEEPAKNI': 'DEEPAKNI.NS',
    'VINATIORG': 'VINATIORG.NS',
    'CHEMCON': 'CHEMCON.NS',
    
    # Specialty Cases and Common Variations
    'BOSCH': 'BOSCHLTD.NS',  # Correct mapping for Bosch
    'SUZLON': 'SUZLON.NS',
    'YESBANK': 'YESBANK.NS',
    'PNB': 'PNB.NS',
    'CANBK': 'CANBK.NS',
    'BANKBARODA': 'BANKBARODA.NS',
    'FEDERALBNK': 'FEDERALBNK.NS',
    'IDFCFIRSTB': 'IDFCFIRSTB.NS',
    'BANDHANBNK': 'BANDHANBNK.NS',
    'RBLBANK': 'RBLBANK.NS',
    'AUBANK': 'AUBANK.NS',
}

# Alternative symbol mappings for common user inputs
SYMBOL_ALIASES = {
    'BOSCH': 'BOSCHLTD',
    'MAHINDRA': 'M&M',
    'BAJAJ': 'BAJAJ-AUTO',
    'HERO': 'HEROMOTOCO',
    'TVS': 'TVSMOTOR',
    'EICHER': 'EICHERMOT',
    'ASHOK': 'ASHOKLEY',
    'TATA MOTORS': 'TATAMOTORS',
    'TATA STEEL': 'TATASTEEL',
    'TATA CONSUMER': 'TATACONSUM',
    'TATA CHEMICALS': 'TATACHEM',
    'TATA POWER': 'TATAPOWER',
    'JSW': 'JSWSTEEL',
    'BHARTI': 'BHARTIARTL',
    'HINDUSTAN UNILEVER': 'HINDUNILVR',
    'HINDALCO': 'HINDALCO',
    'ULTRACEMCO': 'ULTRACEMCO',
    'SHREE CEMENT': 'SHREECEM',
    'LARSEN': 'LT',
    'L&T': 'LT',
}

def get_yahoo_symbol(user_input):
    """
    Convert user input to correct Yahoo Finance symbol
    """
    # Clean and normalize input
    symbol = user_input.upper().strip()
    
    # Remove common suffixes
    symbol = symbol.replace('.NS', '').replace('.BO', '')
    
    # Check direct mapping first
    if symbol in NSE_STOCK_MAPPING:
        return NSE_STOCK_MAPPING[symbol]
    
    # Check aliases
    if symbol in SYMBOL_ALIASES:
        mapped_symbol = SYMBOL_ALIASES[symbol]
        if mapped_symbol in NSE_STOCK_MAPPING:
            return NSE_STOCK_MAPPING[mapped_symbol]
    
    # If not found in mapping, try with .NS suffix
    return f"{symbol}.NS"

def validate_nse_symbol(symbol):
    """
    Check if symbol is a known NSE stock
    """
    clean_symbol = symbol.upper().strip().replace('.NS', '').replace('.BO', '')
    return clean_symbol in NSE_STOCK_MAPPING or clean_symbol in SYMBOL_ALIASES

def get_stock_info(symbol):
    """
    Get basic information about a stock symbol
    """
    clean_symbol = symbol.upper().strip().replace('.NS', '').replace('.BO', '')
    
    # Stock categories for context
    categories = {
        'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'MPHASIS', 'COFORGE'],
        'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK', 'INDUSINDBK'],
        'Energy': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'NTPC', 'ADANIGREEN'],
        'Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO'],
        'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON'],
        'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR'],
        'Metals': ['JSWSTEEL', 'TATASTEEL', 'HINDALCO', 'SAIL', 'VEDL', 'RATNAMANI'],
        'Cement': ['ULTRACEMCO', 'SHREECEM', 'GRASIM', 'ACC', 'AMBUJCEM']
    }
    
    # Check if symbol is in any category
    for category, stocks in categories.items():
        if clean_symbol in stocks:
            return {
                'symbol': clean_symbol,
                'yahoo_symbol': get_yahoo_symbol(clean_symbol),
                'category': category,
                'is_valid': True
            }
    
    # If not in categories but in NSE_STOCK_MAPPING, it's still valid
    if clean_symbol in NSE_STOCK_MAPPING:
        return {
            'symbol': clean_symbol,
            'yahoo_symbol': get_yahoo_symbol(clean_symbol),
            'category': 'Other',
            'is_valid': True
        }
    
    # Not found in either categories or NSE mapping
    return {
        'symbol': clean_symbol,
        'yahoo_symbol': get_yahoo_symbol(clean_symbol),
        'category': 'Other',
        'is_valid': False
    }

def suggest_similar_symbols(user_input):
    """
    Suggest similar stock symbols if exact match not found
    """
    user_input = user_input.upper().strip()
    suggestions = []
    
    # Look for partial matches
    for symbol in NSE_STOCK_MAPPING.keys():
        if user_input in symbol or symbol.startswith(user_input[:3]):
            suggestions.append(symbol)
    
    # Look for alias matches
    for alias, mapped in SYMBOL_ALIASES.items():
        if user_input in alias:
            suggestions.append(mapped)
    
    return suggestions[:5]  # Return top 5 suggestions
"""
Comprehensive Indian Stock Name Database
Allows users to search by company names, not just symbols
"""

import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional

# Comprehensive mapping of company names to NSE symbols
COMPANY_NAME_TO_SYMBOL = {
    # IT Companies
    'TATA CONSULTANCY SERVICES': 'TCS',
    'TCS': 'TCS',
    'INFOSYS': 'INFY',
    'INFOSYS LIMITED': 'INFY',
    'WIPRO': 'WIPRO',
    'WIPRO LIMITED': 'WIPRO',
    'HCL TECHNOLOGIES': 'HCLTECH',
    'HCL TECH': 'HCLTECH',
    'TECH MAHINDRA': 'TECHM',
    'LTI MINDTREE': 'LTIM',
    'MPHASIS': 'MPHASIS',
    'COFORGE': 'COFORGE',
    
    # Banking & Finance
    'HDFC BANK': 'HDFCBANK',
    'HOUSING DEVELOPMENT FINANCE CORPORATION BANK': 'HDFCBANK',
    'ICICI BANK': 'ICICIBANK',
    'INDUSTRIAL CREDIT AND INVESTMENT CORPORATION OF INDIA BANK': 'ICICIBANK',
    'STATE BANK OF INDIA': 'SBIN',
    'SBI': 'SBIN',
    'AXIS BANK': 'AXISBANK',
    'KOTAK MAHINDRA BANK': 'KOTAKBANK',
    'KOTAK BANK': 'KOTAKBANK',
    'INDUSIND BANK': 'INDUSINDBK',
    'BAJAJ FINANCE': 'BAJFINANCE',
    'BAJAJ FINSERV': 'BAJAJFINSV',
    'HDFC LIFE': 'HDFCLIFE',
    'SBI LIFE': 'SBILIFE',
    'ICICI GENERAL INSURANCE': 'ICICIGI',
    'HDFC ASSET MANAGEMENT': 'HDFCAMC',
    
    # Energy & Oil
    'RELIANCE INDUSTRIES': 'RELIANCE',
    'RELIANCE': 'RELIANCE',
    'OIL AND NATURAL GAS CORPORATION': 'ONGC',
    'ONGC': 'ONGC',
    'INDIAN OIL CORPORATION': 'IOC',
    'IOC': 'IOC',
    'BHARAT PETROLEUM': 'BPCL',
    'BPCL': 'BPCL',
    'HINDUSTAN PETROLEUM': 'HINDPETRO',
    'GAS AUTHORITY OF INDIA': 'GAIL',
    'GAIL': 'GAIL',
    'NATIONAL THERMAL POWER CORPORATION': 'NTPC',
    'NTPC': 'NTPC',
    'POWER GRID CORPORATION': 'POWERGRID',
    'POWERGRID': 'POWERGRID',
    'COAL INDIA': 'COALINDIA',
    'ADANI GREEN ENERGY': 'ADANIGREEN',
    'ADANI POWER': 'ADANIPOWER',
    'TATA POWER': 'TATAPOWER',
    
    # Automotive
    'MARUTI SUZUKI': 'MARUTI',
    'MARUTI SUZUKI INDIA': 'MARUTI',
    'MARUTI': 'MARUTI',
    'TATA MOTORS': 'TATAMOTORS',
    'MAHINDRA AND MAHINDRA': 'M&M',
    'MAHINDRA': 'M&M',
    'M&M': 'M&M',
    'BAJAJ AUTO': 'BAJAJ-AUTO',
    'HERO MOTOCORP': 'HEROMOTOCO',
    'HERO HONDA': 'HEROMOTOCO',
    'TVS MOTOR': 'TVSMOTOR',
    'EICHER MOTORS': 'EICHERMOT',
    'ASHOK LEYLAND': 'ASHOKLEY',
    'BHARAT FORGE': 'BHARATFORG',
    'MOTHERSON SUMI': 'MOTHERSON',
    
    # Metals & Mining
    'JSW STEEL': 'JSWSTEEL',
    'TATA STEEL': 'TATASTEEL',
    'HINDALCO INDUSTRIES': 'HINDALCO',
    'HINDALCO': 'HINDALCO',
    'STEEL AUTHORITY OF INDIA': 'SAIL',
    'SAIL': 'SAIL',
    'JINDAL STEEL': 'JINDALSTEL',
    'NATIONAL MINERAL DEVELOPMENT CORPORATION': 'NMDC',
    'NMDC': 'NMDC',
    'VEDANTA': 'VEDL',
    'NATIONAL ALUMINIUM': 'NATIONALUM',
    'HINDUSTAN ZINC': 'HINDZINC',
    
    # Pharmaceuticals
    'SUN PHARMACEUTICAL': 'SUNPHARMA',
    'SUN PHARMA': 'SUNPHARMA',
    'DR REDDYS LABORATORIES': 'DRREDDY',
    'DR REDDY': 'DRREDDY',
    'CIPLA': 'CIPLA',
    'DIVIS LABORATORIES': 'DIVISLAB',
    'BIOCON': 'BIOCON',
    'AUROBINDO PHARMA': 'AUROPHARMA',
    'LUPIN': 'LUPIN',
    'GLENMARK PHARMACEUTICALS': 'GLENMARK',
    'TORRENT PHARMACEUTICALS': 'TORNTPHARM',
    'ALKEM LABORATORIES': 'ALKEM',
    'VIMTA LABS': 'VIMTALABS',
    'VIMTA LABORATORIES': 'VIMTALABS',
    'VIMTALABS': 'VIMTALABS',
    'LAURUS LABS': 'LAURUSLABS',
    'LAURUS LABORATORIES': 'LAURUSLABS',
    'NATCO PHARMA': 'NATCOPHARM',
    'STRIDES PHARMA': 'STAR',
    'CADILA HEALTHCARE': 'CADILAHC',
    'ZYDUS CADILA': 'CADILAHC',
    
    # Consumer Goods
    'HINDUSTAN UNILEVER': 'HINDUNILVR',
    'HUL': 'HINDUNILVR',
    'UNILEVER': 'HINDUNILVR',
    'INDIAN TOBACCO COMPANY': 'ITC',
    'ITC': 'ITC',
    'NESTLE INDIA': 'NESTLEIND',
    'NESTLE': 'NESTLEIND',
    'BRITANNIA INDUSTRIES': 'BRITANNIA',
    'BRITANNIA': 'BRITANNIA',
    'DABUR INDIA': 'DABUR',
    'DABUR': 'DABUR',
    'MARICO': 'MARICO',
    'GODREJ CONSUMER PRODUCTS': 'GODREJCP',
    'COLGATE PALMOLIVE': 'COLPAL',
    'COLGATE': 'COLPAL',
    'TATA CONSUMER PRODUCTS': 'TATACONSUM',
    'UNITED BREWERIES': 'UBL',
    
    # Cement
    'ULTRATECH CEMENT': 'ULTRACEMCO',
    'ULTRATECH': 'ULTRACEMCO',
    'SHREE CEMENT': 'SHREECEM',
    'GRASIM INDUSTRIES': 'GRASIM',
    'GRASIM': 'GRASIM',
    'ACC': 'ACC',
    'AMBUJA CEMENTS': 'AMBUJCEM',
    'AMBUJA': 'AMBUJCEM',
    
    # Telecom
    'BHARTI AIRTEL': 'BHARTIARTL',
    'AIRTEL': 'BHARTIARTL',
    'VODAFONE IDEA': 'IDEA',
    'IDEA': 'IDEA',
    'RELIANCE JIO': 'RELIANCE',
    
    # Retail
    'AVENUE SUPERMARTS': 'DMART',
    'DMART': 'DMART',
    'FUTURE RETAIL': 'FRETAIL',
    'TRENT': 'TRENT',
    
    # Airlines
    'INDIGO': 'INDIGO',
    'INTERGLOBE AVIATION': 'INDIGO',
    'SPICEJET': 'SPICEJET',
    
    # Hotels
    'INDIAN HOTELS': 'INDHOTEL',
    'TAJ HOTELS': 'INDHOTEL',
    'EIH': 'EIH',
    
    # Media
    'ZEE ENTERTAINMENT': 'ZEEL',
    'ZEE': 'ZEEL',
    'STAR INDIA': 'STAR',
    'SUN TV': 'SUNTV',
    
    # Real Estate
    'DLF': 'DLF',
    'GODREJ PROPERTIES': 'GODREJPROP',
    'OBEROI REALTY': 'OBEROIRLTY',
    'PRESTIGE ESTATES': 'PRESTIGE',
    
    # Textiles
    'WELSPUN INDIA': 'WELSPUNIND',
    'RAYMOND': 'RAYMOND',
    'ARVIND': 'ARVIND',
    'VARDHMAN TEXTILES': 'VTL',
    
    # Agriculture & Food
    'UNITED PHOSPHORUS': 'UPL',
    'UPL': 'UPL',
    'RALLIS INDIA': 'RALLIS',
    'KRBL': 'KRBL',
    
    # Chemicals
    'ASIAN PAINTS': 'ASIANPAINT',
    'BERGER PAINTS': 'BERGEPAINT',
    'KANSAI NEROLAC': 'KANSAINER',
    'PIDILITE INDUSTRIES': 'PIDILITIND',
    'AARTI INDUSTRIES': 'AARTIIND',
    
    # Infrastructure
    'LARSEN AND TOUBRO': 'LT',
    'L&T': 'LT',
    'LARSEN TOUBRO': 'LT',
    'IRB INFRASTRUCTURE': 'IRB',
    'GMR INFRASTRUCTURE': 'GMRINFRA',
    
    # Logistics
    'BLUE DART': 'BLUEDART',
    'VRL LOGISTICS': 'VRL',
    'MAHINDRA LOGISTICS': 'MAHLOG',
    
    # Insurance
    'LIFE INSURANCE CORPORATION': 'LICI',
    'LIC': 'LICI',
    'GENERAL INSURANCE CORPORATION': 'GICRE',
    'NEW INDIA ASSURANCE': 'NIACL',
    
    # Jewellery
    'TITAN COMPANY': 'TITAN',
    'TITAN': 'TITAN',
    'KALYAN JEWELLERS': 'KALYANKJIL',
    'PC JEWELLER': 'PCJEWELLER',
    
    # Education
    'CAREER POINT': 'CAREERP',
    'APTECH': 'APTECH',
    'NIIT': 'NIIT',
}

# Alternative names and common variations
ALTERNATIVE_NAMES = {
    'TATA CONSULTANCY': 'TCS',
    'TATA CONSULTING': 'TCS',
    'INFOSYS TECH': 'INFY',
    'WIPRO TECH': 'WIPRO',
    'HCL': 'HCLTECH',
    'TECH MAHINDRA': 'TECHM',
    'HDFC': 'HDFCBANK',
    'ICICI': 'ICICIBANK',
    'STATE BANK': 'SBIN',
    'AXIS': 'AXISBANK',
    'KOTAK': 'KOTAKBANK',
    'INDUSIND': 'INDUSINDBK',
    'BAJAJ FIN': 'BAJFINANCE',
    'RELIANCE IND': 'RELIANCE',
    'OIL AND GAS': 'ONGC',
    'INDIAN OIL': 'IOC',
    'BHARAT PETRO': 'BPCL',
    'HINDUSTAN PETRO': 'HINDPETRO',
    'POWER GRID': 'POWERGRID',
    'COAL INDIA': 'COALINDIA',
    'ADANI GREEN': 'ADANIGREEN',
    'MARUTI': 'MARUTI',
    'TATA MOTOR': 'TATAMOTORS',
    'MAHINDRA': 'M&M',
    'BAJAJ': 'BAJAJ-AUTO',
    'HERO MOTOR': 'HEROMOTOCO',
    'JSW': 'JSWSTEEL',
    'TATA STEEL': 'TATASTEEL',
    'HINDUSTAN UNILEVER': 'HINDUNILVR',
    'UNILEVER': 'HINDUNILVR',
    'NESTLE': 'NESTLEIND',
    'BRITANNIA': 'BRITANNIA',
    'ULTRATECH': 'ULTRACEMCO',
    'SHREE CEMENT': 'SHREECEM',
    'BHARTI': 'BHARTIARTL',
    'AIRTEL': 'BHARTIARTL',
    'ASIAN PAINT': 'ASIANPAINT',
    'L&T': 'LT',
    'LARSEN': 'LT',
    'TITAN': 'TITAN',
}

def normalize_text(text: str) -> str:
    """Normalize text for better matching"""
    # Convert to uppercase and remove extra spaces
    text = text.upper().strip()
    # Remove common suffixes
    text = re.sub(r'\s+(LIMITED|LTD|CORPORATION|CORP|COMPANY|CO|INDUSTRIES|IND|ENTERPRISES|ENT)$', '', text)
    # Remove special characters except &
    text = re.sub(r'[^\w\s&]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def similarity_score(a: str, b: str) -> float:
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a, b).ratio()

def find_stock_by_name(user_input: str, threshold: float = 0.6) -> Optional[Dict]:
    """
    Find stock symbol by company name with fuzzy matching
    
    Args:
        user_input: User's input (company name or partial name)
        threshold: Minimum similarity threshold for fuzzy matching
        
    Returns:
        Dictionary with symbol, company name, and confidence score
    """
    normalized_input = normalize_text(user_input)
    
    # First, try exact matches
    if normalized_input in COMPANY_NAME_TO_SYMBOL:
        return {
            'symbol': COMPANY_NAME_TO_SYMBOL[normalized_input],
            'company_name': normalized_input,
            'confidence': 1.0,
            'match_type': 'exact'
        }
    
    # Try alternative names
    if normalized_input in ALTERNATIVE_NAMES:
        symbol = ALTERNATIVE_NAMES[normalized_input]
        return {
            'symbol': symbol,
            'company_name': normalized_input,
            'confidence': 0.95,
            'match_type': 'alternative'
        }
    
    # Try partial matches and fuzzy matching
    best_matches = []
    
    for company_name, symbol in COMPANY_NAME_TO_SYMBOL.items():
        normalized_company = normalize_text(company_name)
        
        # Check if input is contained in company name
        if normalized_input in normalized_company:
            confidence = len(normalized_input) / len(normalized_company)
            best_matches.append({
                'symbol': symbol,
                'company_name': company_name,
                'confidence': confidence,
                'match_type': 'partial'
            })
        
        # Check fuzzy similarity
        similarity = similarity_score(normalized_input, normalized_company)
        if similarity >= threshold:
            best_matches.append({
                'symbol': symbol,
                'company_name': company_name,
                'confidence': similarity,
                'match_type': 'fuzzy'
            })
    
    # Also check alternative names for partial matches
    for alt_name, symbol in ALTERNATIVE_NAMES.items():
        normalized_alt = normalize_text(alt_name)
        
        if normalized_input in normalized_alt:
            confidence = len(normalized_input) / len(normalized_alt)
            best_matches.append({
                'symbol': symbol,
                'company_name': alt_name,
                'confidence': confidence,
                'match_type': 'partial_alternative'
            })
        
        similarity = similarity_score(normalized_input, normalized_alt)
        if similarity >= threshold:
            best_matches.append({
                'symbol': symbol,
                'company_name': alt_name,
                'confidence': similarity,
                'match_type': 'fuzzy_alternative'
            })
    
    # Remove duplicates and sort by confidence
    unique_matches = {}
    for match in best_matches:
        key = match['symbol']
        if key not in unique_matches or match['confidence'] > unique_matches[key]['confidence']:
            unique_matches[key] = match
    
    if unique_matches:
        # Return the best match
        best_match = max(unique_matches.values(), key=lambda x: x['confidence'])
        return best_match
    
    return None

def suggest_stock_names(user_input: str, max_suggestions: int = 5) -> List[Dict]:
    """
    Suggest multiple stock names based on user input
    
    Args:
        user_input: User's input
        max_suggestions: Maximum number of suggestions to return
        
    Returns:
        List of dictionaries with symbol, company name, and confidence
    """
    normalized_input = normalize_text(user_input)
    suggestions = []
    
    # Collect all possible matches
    for company_name, symbol in COMPANY_NAME_TO_SYMBOL.items():
        normalized_company = normalize_text(company_name)
        
        # Partial match
        if normalized_input in normalized_company or normalized_company in normalized_input:
            confidence = min(len(normalized_input), len(normalized_company)) / max(len(normalized_input), len(normalized_company))
            suggestions.append({
                'symbol': symbol,
                'company_name': company_name,
                'confidence': confidence,
                'match_type': 'partial'
            })
        
        # Fuzzy match
        similarity = similarity_score(normalized_input, normalized_company)
        if similarity >= 0.3:  # Lower threshold for suggestions
            suggestions.append({
                'symbol': symbol,
                'company_name': company_name,
                'confidence': similarity,
                'match_type': 'fuzzy'
            })
    
    # Check alternative names too
    for alt_name, symbol in ALTERNATIVE_NAMES.items():
        normalized_alt = normalize_text(alt_name)
        
        if normalized_input in normalized_alt or normalized_alt in normalized_input:
            confidence = min(len(normalized_input), len(normalized_alt)) / max(len(normalized_input), len(normalized_alt))
            suggestions.append({
                'symbol': symbol,
                'company_name': alt_name,
                'confidence': confidence,
                'match_type': 'partial_alternative'
            })
        
        similarity = similarity_score(normalized_input, normalized_alt)
        if similarity >= 0.3:
            suggestions.append({
                'symbol': symbol,
                'company_name': alt_name,
                'confidence': similarity,
                'match_type': 'fuzzy_alternative'
            })
    
    # Remove duplicates and sort by confidence
    unique_suggestions = {}
    for suggestion in suggestions:
        key = suggestion['symbol']
        if key not in unique_suggestions or suggestion['confidence'] > unique_suggestions[key]['confidence']:
            unique_suggestions[key] = suggestion
    
    # Sort by confidence and return top suggestions
    sorted_suggestions = sorted(unique_suggestions.values(), key=lambda x: x['confidence'], reverse=True)
    return sorted_suggestions[:max_suggestions]

def get_company_info(symbol: str) -> Optional[str]:
    """Get company name from symbol"""
    symbol = symbol.upper()
    for company_name, stock_symbol in COMPANY_NAME_TO_SYMBOL.items():
        if stock_symbol == symbol:
            return company_name
    return None

# Test function
if __name__ == "__main__":
    # Test cases
    test_inputs = [
        "Tata Consultancy",
        "Reliance",
        "HDFC Bank",
        "Infosys",
        "Maruti Suzuki",
        "Asian Paints",
        "Hindustan Unilever",
        "tata motors",
        "ICICI",
        "wipro",
        "bharti airtel",
        "ultratech cement"
    ]
    
    print("Testing stock name search:")
    for test_input in test_inputs:
        result = find_stock_by_name(test_input)
        if result:
            print(f"'{test_input}' -> {result['symbol']} ({result['company_name']}) - Confidence: {result['confidence']:.2f}")
        else:
            print(f"'{test_input}' -> No match found")
            suggestions = suggest_stock_names(test_input, 3)
            if suggestions:
                print(f"  Suggestions: {[s['symbol'] for s in suggestions]}")

"""
IPO Tracker - Fetch and display upcoming IPOs in India
Tracks IPOs coming in the next week with closing dates
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import re
from bs4 import BeautifulSoup

class IPOTracker:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.current_date = datetime.now()
        self.next_week = self.current_date + timedelta(days=7)
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats commonly used in IPO listings"""
        if not date_str or date_str.lower() in ['tba', 'to be announced', '-', 'n/a']:
            return None
        
        # Clean the date string
        date_str = date_str.strip().replace(',', '')
        
        # Common date formats
        date_formats = [
            '%d %b %Y',      # 15 Jan 2025
            '%d-%b-%Y',      # 15-Jan-2025
            '%d/%m/%Y',      # 15/01/2025
            '%d-%m-%Y',      # 15-01-2025
            '%Y-%m-%d',      # 2025-01-15
            '%b %d, %Y',     # Jan 15, 2025
            '%B %d, %Y',     # January 15, 2025
            '%d %B %Y',      # 15 January 2025
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try to extract date from ranges like "Jan 15 - Jan 20, 2025"
        if ' - ' in date_str or ' to ' in date_str:
            parts = re.split(r' - | to ', date_str)
            if len(parts) == 2:
                # Try to parse the end date
                return self.parse_date(parts[1].strip())
        
        return None
    
    def get_upcoming_ipos_from_multiple_sources(self) -> List[Dict]:
        """Fetch IPO data from multiple sources and combine results"""
        all_ipos = []
        
        # Source 1: Mock data (since real APIs require authentication)
        # In production, you would integrate with actual IPO data providers
        mock_ipos = self.get_mock_ipo_data()
        all_ipos.extend(mock_ipos)
        
        # Source 2: Try to scrape from public sources (if available)
        try:
            scraped_ipos = self.scrape_ipo_data()
            all_ipos.extend(scraped_ipos)
        except Exception as e:
            print(f"Could not scrape IPO data: {e}")
        
        # Remove duplicates and filter for next week
        unique_ipos = self.filter_and_deduplicate_ipos(all_ipos)
        
        return unique_ipos
    
    def get_mock_ipo_data(self) -> List[Dict]:
        """
        Mock IPO data for demonstration
        In production, replace with actual API calls to IPO data providers
        """
        base_date = self.current_date
        
        mock_data = [
            {
                'company_name': 'TechCorp India Limited',
                'issue_size': '₹500 Cr',
                'price_band': '₹120-130',
                'opening_date': base_date + timedelta(days=1),
                'closing_date': base_date + timedelta(days=3),
                'listing_date': base_date + timedelta(days=7),
                'lot_size': 100,
                'category': 'Mainboard',
                'lead_managers': 'ICICI Securities, Kotak Mahindra Capital',
                'registrar': 'Link Intime India',
                'status': 'Upcoming'
            },
            {
                'company_name': 'Green Energy Solutions Ltd',
                'issue_size': '₹750 Cr',
                'price_band': '₹85-95',
                'opening_date': base_date + timedelta(days=2),
                'closing_date': base_date + timedelta(days=4),
                'listing_date': base_date + timedelta(days=8),
                'lot_size': 150,
                'category': 'Mainboard',
                'lead_managers': 'Axis Capital, HDFC Bank',
                'registrar': 'KFin Technologies',
                'status': 'Upcoming'
            },
            {
                'company_name': 'FinTech Innovations Private Limited',
                'issue_size': '₹300 Cr',
                'price_band': '₹200-220',
                'opening_date': base_date + timedelta(days=3),
                'closing_date': base_date + timedelta(days=5),
                'listing_date': base_date + timedelta(days=9),
                'lot_size': 50,
                'category': 'SME',
                'lead_managers': 'Edelweiss Financial Services',
                'registrar': 'Bigshare Services',
                'status': 'Upcoming'
            },
            {
                'company_name': 'Healthcare Plus Ltd',
                'issue_size': '₹1200 Cr',
                'price_band': '₹450-500',
                'opening_date': base_date + timedelta(days=4),
                'closing_date': base_date + timedelta(days=6),
                'listing_date': base_date + timedelta(days=10),
                'lot_size': 25,
                'category': 'Mainboard',
                'lead_managers': 'Morgan Stanley, Goldman Sachs',
                'registrar': 'Computer Age Management Services',
                'status': 'Upcoming'
            },
            {
                'company_name': 'Logistics Express India',
                'issue_size': '₹400 Cr',
                'price_band': '₹110-125',
                'opening_date': base_date - timedelta(days=1),
                'closing_date': base_date + timedelta(days=1),
                'listing_date': base_date + timedelta(days=5),
                'lot_size': 120,
                'category': 'Mainboard',
                'lead_managers': 'SBI Capital Markets',
                'registrar': 'Karvy Fintech',
                'status': 'Open'
            }
        ]
        
        return mock_data
    
    def scrape_ipo_data(self) -> List[Dict]:
        """
        Attempt to scrape IPO data from public sources
        Note: This is a basic implementation. In production, you'd need robust scraping
        """
        scraped_ipos = []
        
        # This is a placeholder for actual scraping logic
        # You would implement specific scrapers for IPO websites here
        
        return scraped_ipos
    
    def filter_and_deduplicate_ipos(self, ipos: List[Dict]) -> List[Dict]:
        """Filter IPOs for next week and remove duplicates"""
        filtered_ipos = []
        seen_companies = set()
        
        for ipo in ipos:
            # Skip duplicates
            if ipo['company_name'] in seen_companies:
                continue
            
            # Check if IPO is relevant for next week
            closing_date = ipo.get('closing_date')
            opening_date = ipo.get('opening_date')
            
            if closing_date and isinstance(closing_date, datetime):
                # Include if closing date is within next week
                if self.current_date <= closing_date <= self.next_week:
                    filtered_ipos.append(ipo)
                    seen_companies.add(ipo['company_name'])
            elif opening_date and isinstance(opening_date, datetime):
                # Include if opening date is within next week
                if self.current_date <= opening_date <= self.next_week:
                    filtered_ipos.append(ipo)
                    seen_companies.add(ipo['company_name'])
        
        # Sort by closing date
        filtered_ipos.sort(key=lambda x: x.get('closing_date', datetime.max))
        
        return filtered_ipos
    
    def format_ipo_for_display(self, ipo: Dict) -> Dict:
        """Format IPO data for user-friendly display"""
        def format_date(date_obj):
            if isinstance(date_obj, datetime):
                return date_obj.strftime('%d %b %Y')
            return str(date_obj) if date_obj else 'TBA'
        
        def calculate_days_remaining(date_obj):
            if isinstance(date_obj, datetime):
                diff = (date_obj - self.current_date).days
                if diff == 0:
                    return 'Today'
                elif diff == 1:
                    return 'Tomorrow'
                elif diff > 0:
                    return f'{diff} days'
                else:
                    return 'Closed'
            return 'TBA'
        
        return {
            'company_name': ipo.get('company_name', 'Unknown'),
            'issue_size': ipo.get('issue_size', 'TBA'),
            'price_band': ipo.get('price_band', 'TBA'),
            'opening_date': format_date(ipo.get('opening_date')),
            'closing_date': format_date(ipo.get('closing_date')),
            'listing_date': format_date(ipo.get('listing_date')),
            'days_to_close': calculate_days_remaining(ipo.get('closing_date')),
            'lot_size': ipo.get('lot_size', 'TBA'),
            'category': ipo.get('category', 'TBA'),
            'lead_managers': ipo.get('lead_managers', 'TBA'),
            'registrar': ipo.get('registrar', 'TBA'),
            'status': ipo.get('status', 'Unknown')
        }
    
    def get_upcoming_ipos(self) -> Dict:
        """Main method to get formatted upcoming IPOs"""
        try:
            raw_ipos = self.get_upcoming_ipos_from_multiple_sources()
            
            if not raw_ipos:
                return {
                    'success': True,
                    'message': 'No upcoming IPOs found for the next week',
                    'ipos': [],
                    'total_count': 0,
                    'last_updated': self.current_date.strftime('%d %b %Y, %I:%M %p')
                }
            
            formatted_ipos = [self.format_ipo_for_display(ipo) for ipo in raw_ipos]
            
            # Categorize IPOs
            open_ipos = [ipo for ipo in formatted_ipos if ipo['status'] == 'Open']
            upcoming_ipos = [ipo for ipo in formatted_ipos if ipo['status'] == 'Upcoming']
            
            return {
                'success': True,
                'ipos': formatted_ipos,
                'open_ipos': open_ipos,
                'upcoming_ipos': upcoming_ipos,
                'total_count': len(formatted_ipos),
                'open_count': len(open_ipos),
                'upcoming_count': len(upcoming_ipos),
                'last_updated': self.current_date.strftime('%d %b %Y, %I:%M %p'),
                'search_period': f"{self.current_date.strftime('%d %b')} - {self.next_week.strftime('%d %b %Y')}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error fetching IPO data: {str(e)}',
                'ipos': [],
                'total_count': 0
            }
    
    def get_ipo_details(self, company_name: str) -> Optional[Dict]:
        """Get detailed information about a specific IPO"""
        ipos = self.get_upcoming_ipos_from_multiple_sources()
        
        for ipo in ipos:
            if ipo['company_name'].lower() == company_name.lower():
                return self.format_ipo_for_display(ipo)
        
        return None
    
    def get_ipo_calendar(self, days: int = 30) -> Dict:
        """Get IPO calendar for specified number of days"""
        end_date = self.current_date + timedelta(days=days)
        
        # This would fetch IPOs for the extended period
        # For now, return the weekly data
        return self.get_upcoming_ipos()

# Test the IPO tracker
if __name__ == "__main__":
    tracker = IPOTracker()
    result = tracker.get_upcoming_ipos()
    
    print("=== UPCOMING IPOs (Next 7 Days) ===")
    print(json.dumps(result, indent=2, default=str))

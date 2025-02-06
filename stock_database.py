import json
import os
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List
import logging
from phi.tools.yfinance import YFinanceTools
import requests
from decimal import Decimal

logger = logging.getLogger(__name__)

class StockDatabase:
    def __init__(self, cache_file: str = "stock_cache.json"):
        self.cache_file = cache_file
        self.cache_duration = timedelta(hours=24)  # Cache data for 24 hours
        self.yfinance_tools = YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        )
        self.inr_to_usd_rate = self._get_exchange_rate()
        self.load_cache()

    def _get_exchange_rate(self) -> float:
        """Get INR to USD exchange rate"""
        try:
            usdinr = yf.Ticker("INR=X")
            rate = usdinr.info.get('regularMarketPrice', 83.0)
            return 1/rate if rate else 1/83.0
        except:
            return 1/83.0  # Fallback exchange rate

    def _get_stock_price(self, symbol: str) -> tuple:
        """Get stock price with reliable fallbacks"""
        try:
            stock = yf.Ticker(symbol)
            # Try different price attributes
            price = None
            for attr in ['regularMarketPrice', 'currentPrice', 'previousClose']:
                price = stock.info.get(attr)
                if price:
                    break
            
            currency = stock.info.get('currency', 'USD')
            
            if not price:
                # Fallback to history
                hist = stock.history(period='1d')
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
            
            return float(price) if price else 0, currency
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0, 'USD'

    def load_cache(self):
        """Load cached stock data from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.cache = data.get('stocks', {})
                    self.last_updated = datetime.fromisoformat(data.get('last_updated', '2000-01-01'))
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                self.cache = {}
                self.last_updated = datetime.min
        else:
            self.cache = {}
            self.last_updated = datetime.min

    def save_cache(self):
        """Save current stock data to cache file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'stocks': self.cache,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def update_stock_data(self, symbols: List[str]):
        """Update stock data for given symbols using multiple data sources"""
        self.inr_to_usd_rate = self._get_exchange_rate()  # Refresh exchange rate
        
        for symbol in symbols:
            try:
                # Get price and currency
                price, currency = self._get_stock_price(symbol)
                
                # Convert price to USD if needed
                if currency == 'INR':
                    price_usd = price * self.inr_to_usd_rate
                else:
                    price_usd = price

                # Get stock info
                stock = yf.Ticker(symbol)
                info = stock.info
                
                # Convert market cap to USD if in INR
                market_cap = info.get('marketCap', 0)
                if currency == 'INR':
                    market_cap = market_cap * self.inr_to_usd_rate
                
                # Format market cap in billions
                market_cap_billions = market_cap / 1_000_000_000 if market_cap else 0
                
                logger.info(f"Fetched {symbol}: {currency} {price} -> USD {price_usd:.2f}")
                
                self.cache[symbol] = {
                    'symbol': symbol,
                    'price': price_usd,
                    'original_price': price,
                    'original_currency': currency,
                    'name': info.get('longName', symbol),
                    'recommendation': info.get('recommendationKey', 'N/A'),
                    'forward_pe': info.get('forwardPE', 'N/A'),
                    'dividend_yield': info.get('dividendYield', 0),
                    'market_cap': round(market_cap_billions, 2),  # Store in billions USD
                    'currency': currency,
                    'last_updated': datetime.now().isoformat(),
                    'volume': info.get('volume', 0),
                    'avg_volume': info.get('averageVolume', 0),
                    'fifty_day_average': info.get('fiftyDayAverage', 0)
                }
                
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
                continue
        
        self.last_updated = datetime.now()
        self.save_cache()
        
    def get_stock_data(self, symbol: str) -> Dict:
        """Get stock data from cache or fetch if needed"""
        if symbol not in self.cache or \
           datetime.now() - datetime.fromisoformat(self.cache[symbol]['last_updated']) > self.cache_duration:
            self.update_stock_data([symbol])
        return self.cache.get(symbol)

    def get_stocks_in_budget(self, budget_usd: float) -> List[Dict]:
        """Get all stocks within budget from cache with additional filtering"""
        valid_stocks = []
        for stock in self.cache.values():
            try:
                price = float(stock['price'])
                if price > 0 and price <= budget_usd:
                    valid_stocks.append(stock)
            except (ValueError, TypeError):
                continue
                
        return sorted(valid_stocks, key=lambda x: x['market_cap'] or 0, reverse=True)

    def refresh_all_stocks(self):
        """Refresh all cached stock data"""
        if datetime.now() - self.last_updated > self.cache_duration:
            all_symbols = list(self.cache.keys())
            self.update_stock_data(all_symbols)

    def get_stock_price_in_currency(self, symbol: str, target_currency: str = 'USD') -> tuple:
        """Get stock price in specified currency"""
        stock_data = self.get_stock_data(symbol)
        if not stock_data:
            return 0, 'USD'
            
        price = stock_data['original_price']
        currency = stock_data['original_currency']
        
        if target_currency == currency:
            return price, currency
        elif target_currency == 'USD' and currency == 'INR':
            return price * self.inr_to_usd_rate, 'USD'
        elif target_currency == 'INR' and currency == 'USD':
            return price / self.inr_to_usd_rate, 'INR'
        
        return price, currency

    def check_price_only(self, symbol: str) -> float:
        """Quick check of stock price without fetching full data"""
        try:
            stock = yf.Ticker(symbol)
            price = stock.info.get('regularMarketPrice', 0)
            currency = stock.info.get('currency', 'USD')
            
            if currency == 'INR':
                return price * self.inr_to_usd_rate
            return price
            
        except Exception as e:
            logger.error(f"Error checking price for {symbol}: {e}")
            return float('inf')  # Return infinity to exclude this stock

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import yfinance as yf
from typing import List, Dict
import pandas as pd
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from stock_database import StockDatabase
from stock_analysis import get_top_stock_recommendations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq model with better model
groq_model = Groq(
    api_key=groq_api_key,
    id="llama3-8b-8192"
)

# Initialize stock database
stock_db = StockDatabase()

def inr_to_usd(amount_inr: float) -> float:
    """Convert INR to USD using current exchange rate"""
    try:
        inr = yf.Ticker("INR=X")
        exchange_rate = 1 / inr.info['regularMarketPrice']
        return round(amount_inr * exchange_rate, 2)
    except:
        # Fallback to approximate conversion if API fails
        return round(amount_inr / 83, 2)  # Using approximate exchange rate

def get_nifty_stocks():
    """Get list of NIFTY stocks"""
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'WIPRO.NS',
        'BAJFINANCE.NS', 'HCLTECH.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 'ULTRACEMCO.NS',
        'TITAN.NS', 'ADANIENT.NS', 'BAJAJFINSV.NS', 'NTPC.NS', 'POWERGRID.NS',
        'TATASTEEL.NS', 'M&M.NS', 'ONGC.NS', 'GRASIM.NS', 'HINDALCO.NS',
        'JSWSTEEL.NS', 'TECHM.NS', 'ADANIPORTS.NS', 'DRREDDY.NS', 'COALINDIA.NS'
    ]

def get_sp500_stocks():
    """Get list of popular S&P 500 stocks"""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
        'META', 'BRK-B', 'JPM', 'V', 'PG',
        'TSLA', 'UNH', 'MA', 'HD', 'BAC',
        'XOM', 'JNJ', 'CVX', 'WMT', 'LLY',
        'AVGO', 'KO', 'PEP', 'ABBV', 'MRK',
        'COST', 'AMD', 'DIS', 'CSCO', 'ADBE',
        'NFLX', 'CRM', 'INTC', 'VZ', 'CMCSA',
        'PM', 'ORCL', 'NKE', 'TMO', 'MS',
        'IBM', 'GS', 'QCOM', 'BA', 'CAT',
        'GE', 'MCD', 'MMM', 'HON', 'PYPL'
    ]

def fetch_stock_data(symbol: str, budget_usd: float) -> dict:
    """Fetch individual stock data with caching"""
    try:
        stock_data = stock_db.get_stock_data(symbol)
        if stock_data and stock_data['price'] <= budget_usd:
            return stock_data
    except Exception as e:
        logger.warning(f"Error fetching data for {symbol}: {str(e)}")
    return None

def analyze_stocks_in_budget(budget_inr: float) -> Dict:
    """Analyze stocks within budget using cached data"""
    budget_usd = inr_to_usd(budget_inr)
    logger.info(f"Analyzing stocks for budget: ₹{budget_inr:,.2f} (${budget_usd:,.2f})")
    
    # Quick budget validation
    if budget_usd < 1:  # If budget is less than $1
        return {
            'budget_inr': budget_inr,
            'budget_usd': budget_usd,
            'stocks': []
        }
    
    # Get all stock symbols
    all_stocks = get_nifty_stocks() + get_sp500_stocks()
    
    # First check cached data without updating
    potential_stocks = []
    need_to_fetch = []
    
    for symbol in all_stocks:
        cached_data = stock_db.cache.get(symbol)
        if cached_data:
            if cached_data['price'] <= budget_usd:
                potential_stocks.append(cached_data)
        else:
            need_to_fetch.append(symbol)
    
    # Only fetch new data if we have symbols not in cache
    if need_to_fetch:
        stock_db.update_stock_data(need_to_fetch)
        # Add any new stocks within budget
        for symbol in need_to_fetch:
            stock_data = stock_db.cache.get(symbol)
            if stock_data and stock_data['price'] <= budget_usd:
                potential_stocks.append(stock_data)
    
    # Sort stocks by market cap
    sorted_stocks = sorted(
        potential_stocks,
        key=lambda x: (x['market_cap'] if isinstance(x['market_cap'], (int, float)) else 0),
        reverse=True
    )
    
    return {
        'budget_inr': budget_inr,
        'budget_usd': budget_usd,
        'stocks': sorted_stocks
    }

## web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=[
        "Use the DuckDuckGo search tool to find relevant information",
        "Always include sources in your response",
        "Format responses in a clear, structured way",
        "If search fails, provide a graceful error message"
    ],
    show_tools_calls=True,
    markdown=True,
)

## Financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=groq_model,
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=[
        "Use tables to display the data",
        "For budget analysis, show prices in both USD and INR",
        "Include analyst recommendations when available"
    ],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=groq_model,
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

def get_stock_recommendations_by_budget(budget_inr: float) -> str:
    """Get formatted stock recommendations with comparison"""
    analysis = analyze_stocks_in_budget(budget_inr)
    if not analysis['stocks']:
        return f"No stocks found within budget: ₹{analysis['budget_inr']:,.2f} (${analysis['budget_usd']:,.2f})"
    
    response = [
        f"Analysis for Budget: ₹{analysis['budget_inr']:,.2f} (${analysis['budget_usd']:,.2f})\n",
        "| Stock | Name | Price (USD) | Recommendation | Forward P/E | Dividend Yield |",
        "|-------|------|-------------|----------------|-------------|----------------|"
    ]
    
    for stock in analysis['stocks']:
        div_yield = f"{stock['dividend_yield']*100:.2f}%" if isinstance(stock['dividend_yield'], (int, float)) else 'N/A'
        response.append(
            f"| {stock['symbol']} | {stock['name'][:30]} | ${stock['price']:,.2f} | "
            f"{stock['recommendation']} | {stock['forward_pe']} | {div_yield} |"
        )
    
    # Add top recommendations analysis
    response.append("\n" + get_top_stock_recommendations(analysis['stocks']))
    
    return '\n'.join(response)

# Example query
if __name__ == "__main__":
    multi_ai_agent.print_response(
        "Summarize analyst recommendation and share the latest news for NVDA",
        stream=True
    )


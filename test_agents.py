from financial_agent import web_search_agent, finance_agent, multi_ai_agent, get_stock_recommendations_by_budget
import time
from stock_database import StockDatabase
from stock_analysis import calculate_stock_scores

def safe_run_query(agent, query):
    try:
        print(f"\nQuery: {query}")
        response = agent.run(query)
        print("Response:", response.content if hasattr(response, 'content') else response)
    except Exception as e:
        print(f"Error running query: {str(e)}")
        return None
    time.sleep(2)  # Add delay between requests
    return response

# def get_test_stocks():
    """Get comprehensive list of stocks to test"""
    return [
        # US Tech Stocks
        'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA',
        # Indian IT Stocks
        'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS',
        # Indian Banks
        'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
        # Indian Conglomerates
        'RELIANCE.NS', 'TATAMOTORS.NS', 'ITC.NS', 'LT.NS'
    ]

# def test_stock_prices():
    """Test stock price fetching specifically"""
    print("\n=== Testing Stock Prices ===")
    stock_db = StockDatabase()
    
    test_stocks = [
        ('RELIANCE.NS', 'INR'),
        ('TCS.NS', 'INR'),
        ('AAPL', 'USD'),
        ('MSFT', 'USD')
    ]
    
    print("\n| Symbol | Original Price | Currency | USD Price |")
    print("|--------|----------------|----------|------------|")
    
    for symbol, expected_currency in test_stocks:
        stock_db.update_stock_data([symbol])
        data = stock_db.get_stock_data(symbol)
        if data:
            orig_price = data['original_price']
            usd_price = data['price']
            print(f"| {symbol} | {orig_price:,.2f} | {expected_currency} | ${usd_price:,.2f} |")

# def test_finance_agent():
#     print("\n=== Testing Finance Agent ===")
#     queries = [
#         "What is the current stock price of AAPL?",
#         "Show me the latest analyst recommendations for TSLA",
#         "What are the key fundamentals of MSFT?"
#     ]
    
#     for query in queries:
#         safe_run_query(finance_agent, query)

# def test_web_search_agent():
#     print("\n=== Testing Web Search Agent ===")
#     queries = [
#         "Latest developments in AI technology",
#         "Top tech companies performance 2024"
#     ]
    
#     for query in queries:
#         safe_run_query(web_search_agent, query)

# def test_multi_agent():
    # print("\n=== Testing Multi-Agent ===")
    # queries = [
    #     "Compare tech giants AAPL, MSFT, and GOOGL performance",
    #     "Analyze NVDA's market position and recent AI developments"
    # ]
    
    # for query in queries:
    #     safe_run_query(multi_ai_agent, query)

# def test_stock_cache():
    # print("\n=== Testing Stock Cache ===")
    # stock_db = StockDatabase()
    
    # print("Populating cache with initial stock data...")
    # initial_stocks = get_test_stocks()
    # stock_db.update_stock_data(initial_stocks)
    
    # print("\nTesting cache retrieval:")
    # print("\n| Symbol | Price (USD) | Currency | Name | Recommendation |")
    # print("|--------|-------------|----------|------|----------------|")
    
    # for symbol in initial_stocks:
    #     data = stock_db.get_stock_data(symbol)
    #     if data:
    #         print(
    #             f"| {symbol} | ${data['price']:,.2f} | {data['currency']} | "
    #             f"{data['name'][:20]} | {data['recommendation']} |"
    #         )
    #     else:
    #         print(f"| {symbol} | N/A | N/A | Data not available | N/A |")

def test_budget_analysis():
    """Test budget analysis with cached data"""
    print("\n=== Testing Budget Analysis ===")
    test_budgets = [
        21000   # ₹50k
         # ₹2 lakhs
            # ₹10 lakhs
    ]
    
    for budget in test_budgets:
        print(f"\nAnalyzing stocks within budget: ₹{budget:,}")
        result = get_stock_recommendations_by_budget(budget)
        print(result)
        time.sleep(2)

def test_stock_comparison():
    """Test stock comparison functionality"""
    print("\n=== Testing Stock Comparison ===")
    stock_db = StockDatabase()
    
    # Get some test stocks
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'INFY.NS', 'TCS.NS']
    stock_db.update_stock_data(test_stocks)
    
    stocks_data = [stock_db.get_stock_data(symbol) for symbol in test_stocks]
    stocks_data = [s for s in stocks_data if s]  # Remove None values
    
    scores = calculate_stock_scores(stocks_data)
    
    print("\nStock Rankings:")
    print("| Stock | Total Score | Price | Recommendation | P/E | Dividend |")
    print("|-------|-------------|--------|----------------|-----|-----------|")
    
    for score in scores:
        print(f"| {score.symbol} | {score.total_score:.2f} | {score.price_score:.2f} | "
              f"{score.recommendation_score:.2f} | {score.pe_score:.2f} | "
              f"{score.dividend_score:.2f} |")

if __name__ == "__main__":
    try:
        test_stock_comparison()  # Add this before other tests
        test_budget_analysis()
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"Critical error occurred: {str(e)}")

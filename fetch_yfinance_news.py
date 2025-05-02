import yfinance as yf
import json

ticker_symbol = "AAPL"
print(f"Fetching news for {ticker_symbol} using yfinance...")

try:
    # Create a Ticker object
    ticker = yf.Ticker(ticker_symbol)

    # Fetch news
    news = ticker.news

    if news:
        print(f"\nFound {len(news)} news items.")
        # Print the structure of the first news item for inspection
        if len(news) > 0:
            print("\nStructure of the first news item:")
            # Use default=str to handle potential non-serializable types like timestamps
            print(json.dumps(news[0], indent=2, default=str))

        # Save all news to a file for detailed review
        output_filename = f"{ticker_symbol}_news_output.json"
        with open(output_filename, 'w') as f:
            json.dump(news, f, indent=2, default=str)
        print(f"\nFull news list saved to {output_filename}")

    else:
        print("No news found for this ticker.")

except Exception as e:
    print(f"An error occurred: {e}")

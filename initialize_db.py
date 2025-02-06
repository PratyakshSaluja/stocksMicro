from financial_agent import get_nifty_stocks, get_sp500_stocks
from stock_database import StockDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_stock_database():
    """Initialize and update the stock database with all configured stocks"""
    logger.info("Initializing stock database...")
    
    # Get all stock symbols
    all_stocks = get_nifty_stocks() + get_sp500_stocks()
    logger.info(f"Total stocks to process: {len(all_stocks)}")
    
    # Initialize database and update data
    db = StockDatabase()
    db.update_stock_data(all_stocks)
    
    total_cached = len(db.cache)
    logger.info(f"Stock database initialization complete! Cached {total_cached} stocks")
    return {
        "status": "success",
        "total_stocks": total_cached,
        "stocks": list(db.cache.keys())
    }

if __name__ == "__main__":
    initialize_stock_database()

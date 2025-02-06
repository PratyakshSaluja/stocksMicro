from typing import List, Dict
import logging
from dataclasses import dataclass
from stock_database import StockDatabase

logger = logging.getLogger(__name__)

@dataclass
class StockScore:
    symbol: str
    total_score: float
    price_score: float
    recommendation_score: float
    pe_score: float
    dividend_score: float
    volume_score: float

def calculate_stock_scores(stocks: List[Dict]) -> List[StockScore]:
    """Calculate composite scores for stocks based on multiple factors"""
    if not stocks:
        return []
    
    # Score mappings
    recommendation_weights = {
        'strong_buy': 1.0,
        'buy': 0.8,
        'hold': 0.6,
        'sell': 0.3,
        'strong_sell': 0.1,
        'none': 0.5,
        'N/A': 0.5
    }
    
    # Calculate ranges for normalization
    prices = [s['price'] for s in stocks]
    min_price, max_price = min(prices), max(prices)
    
    pe_ratios = [float(s['forward_pe']) for s in stocks if isinstance(s['forward_pe'], (int, float))]
    if pe_ratios:
        min_pe, max_pe = min(pe_ratios), max(pe_ratios)
    else:
        min_pe, max_pe = 0, 1
    
    scores = []
    for stock in stocks:
        try:
            # Price score (lower is better)
            price_score = 1 - ((stock['price'] - min_price) / (max_price - min_price)) if max_price != min_price else 0.5
            
            # Recommendation score
            rec_score = recommendation_weights.get(stock['recommendation'].lower(), 0.5)
            
            # P/E score (lower is better, but not too low)
            pe = float(stock['forward_pe']) if isinstance(stock['forward_pe'], (int, float)) else max_pe
            pe_score = 1 - ((pe - min_pe) / (max_pe - min_pe)) if max_pe != min_pe else 0.5
            if pe < 0:  # Negative P/E is usually bad
                pe_score = 0.1
            
            # Dividend score
            div_yield = float(stock['dividend_yield']) if stock['dividend_yield'] else 0
            div_score = min(div_yield * 10, 1)  # Cap at 100%
            
            # Volume score (higher volume is better)
            avg_volume = stock['avg_volume']
            volume_score = min(avg_volume / 1000000, 1) if avg_volume else 0.5
            
            # Calculate total score with weights
            total_score = (
                price_score * 0.3 +       # 30% weight to price
                rec_score * 0.25 +        # 25% weight to analyst recommendations
                pe_score * 0.20 +         # 20% weight to P/E ratio
                div_score * 0.15 +        # 15% weight to dividend yield
                volume_score * 0.10       # 10% weight to trading volume
            )
            
            scores.append(StockScore(
                symbol=stock['symbol'],
                total_score=total_score,
                price_score=price_score,
                recommendation_score=rec_score,
                pe_score=pe_score,
                dividend_score=div_score,
                volume_score=volume_score
            ))
            
        except Exception as e:
            logger.error(f"Error calculating score for {stock['symbol']}: {e}")
            continue
    
    return sorted(scores, key=lambda x: x.total_score, reverse=True)

def get_top_stock_recommendations(stocks: List[Dict], top_n: int = 5) -> str:
    """Get formatted recommendations for top N stocks"""
    scores = calculate_stock_scores(stocks)
    if not scores:
        return "No stocks available for comparison"
    
    response = [
        f"\nTop {top_n} Recommended Stocks:\n",
        "| Stock | Score | Price Score | Analyst Rating | P/E Score | Dividend Score | Volume Score |",
        "|-------|--------|-------------|----------------|-----------|----------------|--------------|"
    ]
    
    for score in scores[:top_n]:
        response.append(
            f"| {score.symbol} | {score.total_score:.2f} | {score.price_score:.2f} | "
            f"{score.recommendation_score:.2f} | {score.pe_score:.2f} | "
            f"{score.dividend_score:.2f} | {score.volume_score:.2f} |"
        )
    
    return '\n'.join(response)

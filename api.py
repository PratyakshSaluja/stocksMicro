from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from financial_agent import inr_to_usd
from stock_analysis import get_top_stock_recommendations
from stock_database import StockDatabase
from perplexity_stock import StockAnalyzer
from perplexity_mutual_fund import MutualFundAnalyzer
import uvicorn
import requests
from typing import List, Optional
import json
import sys
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mutual_fund_analysis import get_fund_metrics
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Dict
import os

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the current directory is in the Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import with better error handling
try:
    from .test import fetch_latest_data, main as fetch_mutual_funds
except ImportError:
    try:
        from test import fetch_latest_data, main as fetch_mutual_funds
    except ImportError as e:
        logging.error(f"Error importing from test.py: {e}")
        logging.error(f"Current directory: {current_dir}")
        logging.error(f"Python path: {sys.path}")
        raise

app = FastAPI(
    title="Stock Analysis API",
    description="API for analyzing stocks within a given budget",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

class BudgetRequest(BaseModel):
    budget_inr: float = Field(..., description="Budget in INR", gt=0)

class BudgetResponse(BaseModel):
    budget_inr: float
    budget_usd: float
    stocks: list
    recommendations: str

class StockSentiment(BaseModel):
    label: Optional[str] = None
    score: Optional[float] = None
    news_title: Optional[str] = None
    error: Optional[str] = None

class StockDataWithSentiment(BaseModel):
    symbol: str
    name: Optional[str] = None
    price: Optional[float] = None
    sentiment: Optional[StockSentiment] = None

class StockRecommendation(BaseModel):
    symbol: str
    score: float
    price_score: float
    analyst_rating: float
    pe_score: float
    dividend_score: float
    volume_score: float
    sentiment: Optional[StockSentiment] = None

class TopRecommendationsResponse(BaseModel):
    top_stocks: List[StockRecommendation]

class StockInfo(BaseModel):
    current_price: Optional[float] = None
    previous_close: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    volume: Optional[int] = None
    average_volume: Optional[int] = None

class PredictionData(BaseModel):
    predicted_next_day_close: Optional[float] = None

class PredictionResponse(BaseModel):
    ticker: str
    info: Optional[StockInfo] = None
    prediction: Optional[PredictionData] = None
    error: Optional[str] = None

class MutualFundRecommendationsResponse(BaseModel):
    funds: list[dict]

class StockAnalysisRequest(BaseModel):
    query: str
    context: Optional[str] = None

class StockAnalysisResponse(BaseModel):
    content: str
    citations: Optional[List[str]] = None
    usage: Optional[Dict] = None
    error: Optional[str] = None

class MutualFundAnalysisRequest(BaseModel):
    query: str
    context: Optional[str] = None

class MutualFundAnalysisResponse(BaseModel):
    content: str
    citations: Optional[List[str]] = None
    usage: Optional[Dict] = None
    error: Optional[str] = None

# Initialize analyzers
stock_db = StockDatabase()
stock_analyzer = StockAnalyzer()
mutual_fund_analyzer = MutualFundAnalyzer()

# Load FinBERT model
device = 0 if torch.cuda.is_available() else -1
logging.info(f"Attempting to load FinBERT on device: {'GPU' if device == 0 else 'CPU'}")
try:
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    sentiment_pipeline = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer, device=device)
    logging.info("FinBERT sentiment analysis pipeline loaded successfully.")
except Exception as e:
    logging.error(f"Error loading FinBERT model: {e}. Sentiment analysis will be unavailable.")
    sentiment_pipeline = None

@app.post("/analyze-stock-query", response_model=StockAnalysisResponse)
async def analyze_stock_query(request: StockAnalysisRequest):
    """
    Analyze a stock or answer stock-related questions using Perplexity API
    with real-time data and citations
    """
    try:
        result = stock_analyzer.analyze_stock(request.query, request.context)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
            
        return StockAnalysisResponse(
            content=result["content"],
            citations=result.get("citations"),
            usage=result.get("usage"),
            error=None
        )
    except Exception as e:
        logging.error(f"Error in stock analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-mutual-fund", response_model=MutualFundAnalysisResponse)
async def analyze_mutual_fund(request: MutualFundAnalysisRequest):
    """
    Analyze mutual funds or answer mutual fund related questions using Perplexity API
    with real-time data and citations
    """
    try:
        result = mutual_fund_analyzer.analyze_mutual_fund(request.query, request.context)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
            
        return MutualFundAnalysisResponse(
            content=result["content"],
            citations=result.get("citations"),
            usage=result.get("usage"),
            error=None
        )
    except Exception as e:
        logging.error(f"Error in mutual fund analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quick-market-check/{symbol}")
async def quick_market_check(symbol: str):
    """
    Get a quick analysis of current market conditions for a stock
    """
    try:
        result = stock_analyzer.quick_market_check(symbol)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        logging.error(f"Error in quick market check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quick-fund-check/{scheme_code}")
async def quick_fund_check(scheme_code: str):
    """
    Get a quick analysis of current mutual fund metrics
    """
    try:
        result = mutual_fund_analyzer.quick_fund_check(scheme_code)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        logging.error(f"Error in quick fund check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-stocks", response_model=BudgetResponse)
async def analyze_stocks(request: BudgetRequest):
    try:
        stock_db.refresh_all_stocks()
        budget_usd = inr_to_usd(request.budget_inr)

        if budget_usd < 1:
            return {
                "budget_inr": request.budget_inr,
                "budget_usd": budget_usd,
                "stocks": [],
                "recommendations": f"No stocks found within budget: ₹{request.budget_inr:,.2f} (${budget_usd:,.2f})"
            }

        stocks_in_budget = stock_db.get_stocks_in_budget(budget_usd)

        analysis_result = {
            "budget_inr": request.budget_inr,
            "budget_usd": budget_usd,
            "stocks": stocks_in_budget
        }

        if stocks_in_budget:
            recommendations = get_top_stock_recommendations(stocks_in_budget)
        else:
            recommendations = f"No stocks found within budget: ₹{request.budget_inr:,.2f} (${budget_usd:,.2f})"

        return {
            **analysis_result,
            "recommendations": recommendations
        }

    except Exception as e:
        logging.error(f"Error in /analyze-stocks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/analyze-stocks")
async def analyze_stocks_options():
    return {}

@app.post("/top-recommendations", response_model=TopRecommendationsResponse)
async def get_top_recommendations(request: BudgetRequest):
    try:
        budget_usd = inr_to_usd(request.budget_inr)

        if budget_usd < 1:
            return TopRecommendationsResponse(top_stocks=[])

        stocks_in_budget = stock_db.get_stocks_in_budget(budget_usd)

        if not stocks_in_budget:
            return TopRecommendationsResponse(top_stocks=[])

        top_stocks_data = stocks_in_budget[:5]
        recommendations_text = get_top_stock_recommendations(stocks_in_budget, top_n=5)
        
        top_stocks = []
        for line in recommendations_text.split('\n'):
            if '|' in line and not line.startswith('|--') and not 'Stock |' in line:
                parts = line.split('|')
                if len(parts) >= 7:
                    stock_rec = StockRecommendation(
                        symbol=parts[1].strip(),
                        score=float(parts[2].strip()),
                        price_score=float(parts[3].strip()),
                        analyst_rating=float(parts[4].strip()),
                        pe_score=float(parts[5].strip()),
                        dividend_score=float(parts[6].strip()),
                        volume_score=float(parts[7].strip())
                    )
                    top_stocks.append(stock_rec)

        if sentiment_pipeline:
            for stock_rec in top_stocks:
                sentiment_result = StockSentiment()
                news_title_analyzed = None

                try:
                    logging.info(f"--- Processing Ticker: {stock_rec.symbol} ---")
                    logging.info(f"Fetching news for {stock_rec.symbol}...")
                    stock_obj = yf.Ticker(stock_rec.symbol)
                    news = stock_obj.news
                    if news and len(news) > 0:
                        first_news = news[0]
                        news_title_analyzed = first_news.get('content', {}).get('title')
                        if news_title_analyzed:
                            logging.info(f"Analyzing title: '{news_title_analyzed}'")
                            analysis = sentiment_pipeline(news_title_analyzed)
                            if analysis and len(analysis) > 0:
                                sentiment_result.label = analysis[0].get('label')
                                sentiment_result.score = analysis[0].get('score')
                                logging.info(f"Sentiment for {stock_rec.symbol}: Label={sentiment_result.label}, Score={sentiment_result.score:.4f}")
                            else:
                                sentiment_result.error = "Sentiment analysis returned empty result."
                                logging.warning(f"Sentiment analysis returned empty result for {stock_rec.symbol}")
                        else:
                            sentiment_result.error = "First news item has no title."
                            logging.warning(f"First news item has no title for {stock_rec.symbol}")
                    else:
                        sentiment_result.error = "No news found for ticker."
                        logging.info(f"No news found for ticker {stock_rec.symbol}")
                except Exception as news_err:
                    logging.error(f"Error fetching or analyzing news for {stock_rec.symbol}: {news_err}", exc_info=True)
                    sentiment_result.error = f"Error processing news: {str(news_err)}"

                sentiment_result.news_title = news_title_analyzed
                stock_rec.sentiment = sentiment_result
        else:
            logging.warning("Sentiment pipeline not available. Skipping sentiment analysis.")
            for stock_rec in top_stocks:
                stock_rec.sentiment = StockSentiment(error="Sentiment model not loaded")

        return TopRecommendationsResponse(top_stocks=top_stocks)

    except Exception as e:
        logging.error(f"Error in /top-recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/top-recommendations")
async def top_recommendations_options():
    return {}

@app.get("/top-mutual-funds", response_model=MutualFundRecommendationsResponse)
async def get_top_mutual_funds():
    """Get top 5 recommended mutual funds with key metrics"""
    try:
        # Fetch mutual fund data
        results = fetch_mutual_funds()
        if not results:
            with open("filtered_schemes_2025.json", "r") as f:
                results = json.load(f)
        
        if not results:
            raise HTTPException(status_code=404, detail="No mutual fund data available")
        
        # Get metrics for all funds
        funds_with_metrics = []
        for fund in results:
            metrics = get_fund_metrics(fund)
            if metrics:
                funds_with_metrics.append(metrics)
        
        # Sort by 1-year returns (or 6-month if 1-year not available)
        def get_sort_key(fund):
            return (
                fund.get('1_year_return', 0) or 
                fund.get('6_month_return', 0) or 
                fund.get('3_month_return', 0) or 
                fund.get('1_month_return', 0)
            )
        
        top_funds = sorted(funds_with_metrics, key=get_sort_key, reverse=True)[:5]
        return {"funds": top_funds}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/top-mutual-funds")
async def top_mutual_funds_options():
    return {}

@app.post("/initialize-stocks")
async def initialize_stocks():
    """Initialize or refresh stock data and mutual funds data"""
    try:
        stock_db.refresh_all_stocks()
        results = fetch_mutual_funds()

        return {
            "message": "Stock data and mutual funds initialized successfully",
            "stocks_updated": True,
            "mutual_funds_updated": bool(results)
        }
    except Exception as e:
        logging.error(f"Error initializing data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error initializing data: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Minimal startup event that just prints ready message"""
    logging.info("API is ready to serve requests")

@app.post("/refresh-stocks")
async def refresh_stocks():
    """Endpoint to manually refresh all cached stock data"""
    try:
        stock_db.refresh_all_stocks()
        return {"message": "Stock cache refreshed successfully"}
    except Exception as e:
        logging.error(f"Error refreshing stock cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mutual-funds")
async def get_mutual_funds():
    try:
        results = fetch_mutual_funds()
        if not results:
            try:
                with open("filtered_schemes_2025.json", "r") as f:
                    results = json.load(f)
            except FileNotFoundError:
                logging.warning("Mutual funds file not found, returning empty list.")
                results = []
        return results
    except Exception as e:
        logging.error(f"Error fetching mutual funds: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mutual-fund/{scheme_code}")
async def get_mutual_fund(scheme_code: str):
    try:
        data = fetch_latest_data(scheme_code)
        if not data:
            raise HTTPException(status_code=404, detail="No data found for scheme")
        return data
    except Exception as e:
        logging.error(f"Error fetching mutual fund {scheme_code}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Stock Analysis API!"}

@app.head("/")
async def head_root():
    """Handle HEAD requests (for uptime monitoring)"""
    return {}

@app.get("/predict-stock/{ticker}", response_model=PredictionResponse)
async def predict_stock(ticker: str):
    """Predict the next day's closing price for a given stock ticker using Linear Regression."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for ticker: {ticker}")

        df = hist[['Close']].copy()
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        if len(df) < 10:
            raise HTTPException(status_code=400, detail=f"Not enough historical data to train model for ticker: {ticker}")

        X = df[['Close']]
        y = df['Target']

        model = LinearRegression()
        model.fit(X, y)

        last_close_price = hist['Close'].iloc[-1]
        prediction_input = np.array([[last_close_price]])
        predicted_price = model.predict(prediction_input)[0]

        stock_info_data = stock.info
        stock_info = StockInfo(
            current_price=stock_info_data.get('regularMarketPrice'),
            previous_close=stock_info_data.get('previousClose'),
            day_high=stock_info_data.get('dayHigh'),
            day_low=stock_info_data.get('dayLow'),
            fifty_two_week_high=stock_info_data.get('fiftyTwoWeekHigh'),
            fifty_two_week_low=stock_info_data.get('fiftyTwoWeekLow'),
            volume=stock_info_data.get('volume'),
            average_volume=stock_info_data.get('averageVolume')
        )

        prediction_data = PredictionData(predicted_next_day_close=round(predicted_price, 2))

        return PredictionResponse(
            ticker=ticker,
            info=stock_info,
            prediction=prediction_data
        )

    except requests.exceptions.RequestException as e:
        logging.error(f"Network error fetching data for {ticker}: {e}", exc_info=True)
        return PredictionResponse(ticker=ticker, error=f"Network error fetching data for {ticker}: {e}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"An error occurred processing {ticker}: {str(e)}", exc_info=True)
        return PredictionResponse(ticker=ticker, error=f"An error occurred processing {ticker}: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=os.getenv("PORT", 8000))
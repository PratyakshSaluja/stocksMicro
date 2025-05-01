from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel, Field
from financial_agent import analyze_stocks_in_budget, get_stock_recommendations_by_budget, inr_to_usd
from stock_analysis import get_top_stock_recommendations
from stock_database import StockDatabase
import uvicorn
import requests
from typing import List, Optional, Dict
import json
import sys
import os
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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
        print(f"Error importing from test.py: {e}")
        print(f"Current directory: {current_dir}")
        print(f"Python path: {sys.path}")
        raise

app = FastAPI(
    title="Stock Analysis API",
    description="API for analyzing stocks within a given budget",
    version="1.0.0"
)

# Configure CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

class BudgetRequest(BaseModel):
    budget_inr: float = Field(..., description="Budget in INR", gt=0)

class BudgetResponse(BaseModel):
    budget_inr: float
    budget_usd: float
    stocks: list
    recommendations: str

class TopRecommendationsResponse(BaseModel):
    recommendations: str
    stocks: list

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

stock_db = StockDatabase()

@app.post("/analyze-stocks", response_model=BudgetResponse)
async def analyze_stocks(request: BudgetRequest):
    try:
        # Refresh stock data first to ensure fresh stock prices
        stock_db.refresh_all_stocks()
        
        # Convert budget to USD first
        budget_usd = inr_to_usd(request.budget_inr)
        
        # Quick validation
        if budget_usd < 1:
            return {
                "budget_inr": request.budget_inr,
                "budget_usd": budget_usd,
                "stocks": [],
                "recommendations": f"No stocks found within budget: ₹{request.budget_inr:,.2f} (${budget_usd:,.2f})"
            }
        
        # Get freshly updated stocks within budget
        stocks_in_budget = stock_db.get_stocks_in_budget(budget_usd)
        
        # Format results
        analysis_result = {
            "budget_inr": request.budget_inr,
            "budget_usd": budget_usd,
            "stocks": stocks_in_budget
        }
        
        # Get detailed recommendations only if we have stocks
        if stocks_in_budget:
            recommendations = get_top_stock_recommendations(stocks_in_budget)
        else:
            recommendations = f"No stocks found within budget: ₹{request.budget_inr:,.2f} (${budget_usd:,.2f})"
        
        return {
            **analysis_result,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/analyze-stocks")
async def analyze_stocks_options():
    return {}

@app.post("/top-recommendations", response_model=TopRecommendationsResponse)
async def get_top_recommendations(request: BudgetRequest):
    try:
        # Don't refresh stocks here since analyze-stocks already does it
        # and both endpoints are called together on the same page
        
        # Convert budget to USD first
        budget_usd = inr_to_usd(request.budget_inr)
        
        # Quick validation
        if budget_usd < 1:
            return {
                "recommendations": f"No stocks found within budget: ₹{request.budget_inr:,.2f} (${budget_usd:,.2f})",
                "stocks": []
            }
        
        # Get stocks within budget using the current cache
        stocks_in_budget = stock_db.get_stocks_in_budget(budget_usd)
        
        if not stocks_in_budget:
            return {
                "recommendations": f"No stocks found within budget: ₹{request.budget_inr:,.2f} (${budget_usd:,.2f})",
                "stocks": []
            }
        
        # Get top 5 recommendations based on the same data as analyze-stocks
        recommendations = get_top_stock_recommendations(stocks_in_budget, top_n=5)
        
        return {
            "recommendations": recommendations,
            "stocks": stocks_in_budget[:5]  # Include full stock data for top 5
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/top-recommendations")
async def top_recommendations_options():
    return {}

@app.post("/initialize-stocks")
async def initialize_stocks():
    """Initialize or refresh stock data and mutual funds data"""
    try:
        # Initialize stock cache
        stock_db.refresh_all_stocks()
        
        # Fetch mutual funds data
        results = fetch_mutual_funds()
        
        return {
            "message": "Stock data and mutual funds initialized successfully",
            "stocks_updated": True,
            "mutual_funds_updated": bool(results)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error initializing data: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Minimal startup event that just prints ready message"""
    print("API is ready to serve requests")

@app.post("/refresh-stocks")
async def refresh_stocks():
    """Endpoint to manually refresh all cached stock data"""
    try:
        stock_db.refresh_all_stocks()
        return {"message": "Stock cache refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mutual-funds")
async def get_mutual_funds():
    try:
        # Fetch fresh mutual fund data instead of reading from file
        results = fetch_mutual_funds()
        if not results:
            # If no fresh data, try reading from file as fallback
            with open("filtered_schemes_2025.json", "r") as f:
                results = json.load(f)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mutual-fund/{scheme_code}")
async def get_mutual_fund(scheme_code: str):
    try:
        # Get fresh data for the specific scheme
        data = fetch_latest_data(scheme_code)
        if not data:
            raise HTTPException(status_code=404, detail="No data found for scheme")
        return data
    except Exception as e:
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
        hist = stock.history(period="1y") # Fetch 1 year of historical data

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for ticker: {ticker}")

        # Feature Engineering: Use lagged close price to predict next day's close
        df = hist[['Close']].copy()
        df['Target'] = df['Close'].shift(-1) # Target is the next day's close price
        df.dropna(inplace=True) # Remove last row with NaN target

        if len(df) < 10: # Need sufficient data for training
             raise HTTPException(status_code=400, detail=f"Not enough historical data to train model for ticker: {ticker}")

        X = df[['Close']] # Feature: Today's Close Price
        y = df['Target']  # Target: Tomorrow's Close Price

        # Train a simple Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict the next day's price based on the most recent closing price
        last_close_price = hist['Close'].iloc[-1]
        prediction_input = np.array([[last_close_price]])
        predicted_price = model.predict(prediction_input)[0]

        # Fetch additional info
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
         # Handle potential yfinance network errors
        return PredictionResponse(ticker=ticker, error=f"Network error fetching data for {ticker}: {e}")
    except HTTPException as e:
        # Re-raise HTTP exceptions from checks (like 404 or 400)
        raise e
    except Exception as e:
        # Catch other potential errors during processing
        return PredictionResponse(ticker=ticker, error=f"An error occurred processing {ticker}: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

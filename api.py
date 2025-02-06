from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel, Field
from financial_agent import analyze_stocks_in_budget, get_stock_recommendations_by_budget, inr_to_usd
from stock_analysis import get_top_stock_recommendations
from stock_database import StockDatabase
import uvicorn

app = FastAPI(
    title="Stock Analysis API",
    description="API for analyzing stocks within a given budget",
    version="1.0.0"
)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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

stock_db = StockDatabase()

@app.post("/analyze-stocks", response_model=BudgetResponse)
async def analyze_stocks(request: BudgetRequest):
    try:
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
        
        # Get cached stocks within budget
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

@app.post("/top-recommendations", response_model=TopRecommendationsResponse)
async def get_top_recommendations(request: BudgetRequest):
    try:       # Convert budget to USD first
        budget_usd = inr_to_usd(request.budget_inr)
        
        # Quick validation
        if budget_usd < 1:
            return {
                "recommendations": f"No stocks found within budget: ₹{request.budget_inr:,.2f} (${budget_usd:,.2f})",
                "stocks": []
            }
        
        # Get cached stocks within budget
        stocks_in_budget = stock_db.get_stocks_in_budget(budget_usd)
        
        if not stocks_in_budget:
            return {
                "recommendations": f"No stocks found within budget: ₹{request.budget_inr:,.2f} (${budget_usd:,.2f})",
                "stocks": []
            }
        
        # Get top 5 recommendations
        recommendations = get_top_stock_recommendations(stocks_in_budget, top_n=5)
        
        return {
            "recommendations": recommendations,
            "stocks": stocks_in_budget[:5]  # Include full stock data for top 5
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Stock Analysis API!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

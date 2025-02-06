from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from financial_agent import analyze_stocks_in_budget, get_stock_recommendations_by_budget
from stock_analysis import get_top_stock_recommendations
import uvicorn

app = FastAPI(
    title="Stock Analysis API",
    description="API for analyzing stocks within a given budget",
    version="1.0.0"
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

@app.post("/analyze-stocks", response_model=BudgetResponse)
async def analyze_stocks(request: BudgetRequest):
    try:
        # Get basic analysis
        basic_analysis = analyze_stocks_in_budget(request.budget_inr)
        
        # Get detailed recommendations
        detailed_recommendations = get_stock_recommendations_by_budget(request.budget_inr)
        
        # Combine results
        return {
            **basic_analysis,
            "recommendations": detailed_recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/top-recommendations", response_model=TopRecommendationsResponse)
async def get_top_recommendations(request: BudgetRequest):
    try:
        # Get basic analysis first to get the list of stocks within budget
        analysis = analyze_stocks_in_budget(request.budget_inr)
        
        # Get top 5 recommendations
        recommendations = get_top_stock_recommendations(analysis['stocks'], top_n=5)
        
        return {
            "recommendations": recommendations,
            "stocks": analysis['stocks'][:5]  # Include full stock data for top 5
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

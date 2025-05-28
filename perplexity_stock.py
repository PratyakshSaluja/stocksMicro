import json
from typing import Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perp import PerplexityAPI

class StockAnalyzer:
    def __init__(self):
        API_KEY = "***REMOVED***"
        self.api = PerplexityAPI(API_KEY)

    def analyze_stock(self, query: str, context: Optional[str] = None) -> Dict:
        system_message = """You are a stock market analyst.
Rules:
1. Provide only the final answer. Do not include explanations of your steps.
2. Do not show intermediate steps or your internal "thinking" process.
3. Structure your response using Markdown.
4. Ensure all factual claims, data, and specific numbers are supported by citations from your search results. The Perplexity API will automatically handle numbering and list sources.

For stock analysis queries, structure your response in this format:

**Market Summary**
[Brief overview of current stock performance and key metrics. Cite sources for data.]

**Recent Performance**
* [Key point 1. Cite sources.]
* [Key point 2. Cite sources.]
* [Key point 3. Cite sources.]

**Technical Analysis**
* [Key technical indicator 1. Cite sources.]
* [Key technical indicator 2. Cite sources.]
* [Key technical indicator 3. Cite sources.]

**Fundamental Analysis**
* [Key fundamental 1. Cite sources.]
* [Key fundamental 2. Cite sources.]
* [Key fundamental 3. Cite sources.]

**Risk Assessment**
* [Risk factor 1. Cite sources.]
* [Risk factor 2. Cite sources.]
* [Risk factor 3. Cite sources.]

**Expert Recommendations**
* [Recommendation 1. Cite sources.]
* [Recommendation 2. Cite sources.]
* [Recommendation 3. Cite sources.]

For non-analysis questions, provide a clear, structured answer with relevant citations (from search results) organized in bullet points. Each point should be a complete sentence.
"""
        # The 'context' variable here is the user's specific context for the query.
        # It will be appended to the system_message by the PerplexityAPI class.
        response = self.api.get_detailed_analysis(query, context=context, system_message_override=system_message)
        formatted = self.api.format_response(response)
        return formatted

    def quick_market_check(self, symbol: str) -> Dict:
        query = f"Provide a quick market update for {symbol} stock. Include current price, recent performance, and key metrics. Present as bullet points. Cite your sources."
        
        system_message_quick = """You are a stock market analyst.
Rules:
1. Provide only the final answer.
2. Do not show intermediate steps or your internal "thinking" process.
3. Structure your response using Markdown bullet points.
4. Ensure all factual claims are supported by citations from your search results.
"""
        response = self.api.get_quick_response(query, system_message_override=system_message_quick)
        formatted = self.api.format_response(response)
        return formatted

def test_analyzer():
    analyzer = StockAnalyzer()
    
    print("\nTesting detailed stock analysis for NVDA:")
    response = analyzer.analyze_stock(
        "Analyze NVIDIA (NVDA) stock. What is the latest news?",
        "Consider recent market trends, company fundamentals, and overall economic conditions for Q1 2025."
    )
    print(json.dumps(response, indent=2))
    
    print("\nTesting quick market check for AAPL:")
    response = analyzer.quick_market_check("AAPL")
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    import json
    test_analyzer()

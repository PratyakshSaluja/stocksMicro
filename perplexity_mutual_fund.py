import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perp import PerplexityAPI
import json 
from typing import Dict, Optional

class MutualFundAnalyzer:
    def __init__(self):
        API_KEY = os.environ.get("PPLX_API_KEY", "")
        self.api = PerplexityAPI(API_KEY)

    def analyze_mutual_fund(self, query: str, context: Optional[str] = None) -> Dict:
        system_message = """You are a mutual fund expert.
Rules:
1. Provide only the final answer. Do not include explanations of your steps.
2. Do not show intermediate steps or your internal "thinking" process.
3. Structure your response using Markdown.
4. Ensure all factual claims, data, and specific numbers are supported by citations from your search results. The Perplexity API will automatically handle numbering and list sources.

For mutual fund analysis queries, structure your response in this format:

**Fund Overview**
[Fund details. Cite sources for data.]

**Performance Analysis**
* [Historical performance metric 1. Cite sources.]
* [Historical performance metric 2. Cite sources.]

**Risk Assessment**
* [Risk metric 1. Cite sources.]
* [Risk metric 2. Cite sources.]

**Asset Allocation**
[Breakdown of fund holdings. Cite sources.]

**Expense Analysis**
[Expense ratio and fee structure. Cite sources.]

**Expert Recommendations**
* [Expert opinion 1. Cite sources.]
* [Expert opinion 2. Cite sources.]

**Investment Suitability**
* [Suitable investor profile. Cite sources.]
* [Investment horizon. Cite sources.]
* [Risk tolerance level. Cite sources.]

For non-analysis questions, provide a clear, structured answer with relevant citations (from search results) organized in bullet points. Each point should be a complete sentence.
"""
        response = self.api.get_detailed_analysis(query, context=context, system_message_override=system_message)
        formatted = self.api.format_response(response)
        return formatted

    def quick_fund_check(self, scheme_code: str) -> Dict:
        query = f"Provide a quick update for mutual fund scheme {scheme_code}. Include key metrics and current performance indicators. Present as bullet points. Cite your sources."
        
        system_message_quick = """You are a mutual fund expert.
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
    analyzer = MutualFundAnalyzer()
    
    print("\nTesting detailed mutual fund analysis:")
    response = analyzer.analyze_mutual_fund(
        "Analyze HDFC Top 100 Fund Direct Plan Growth mutual fund performance and suitability",
        "Consider recent market conditions and comparable large-cap funds"
    )
    print(json.dumps(response, indent=2))
    
    print("\nTesting quick fund check:")
    response = analyzer.quick_fund_check("INF174K01LS2") 
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    test_analyzer()

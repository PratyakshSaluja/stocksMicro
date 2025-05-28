import requests
import json
from typing import Dict, List, Optional
import logging 
import os 
import sys 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

MODEL_QUICK_RESPONSE = "sonar-pro"
MODEL_DETAILED_ANALYSIS = "sonar-reasoning-pro"

class PerplexityAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"

    def _make_request(self, endpoint: str, payload: Dict) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        response_obj = None
        try:
            logging.info(f"Making request to Perplexity API: {endpoint} with model {payload.get('model')}")
            response_obj = requests.post(f"{self.base_url}/{endpoint}", headers=headers, json=payload)
            response_obj.raise_for_status()
            return response_obj.json()
        except requests.exceptions.RequestException as e:
            error_detail = str(e)
            if response_obj is not None:
                logging.error(f"Request failed: {e}, Status Code: {response_obj.status_code}, Response Text: {response_obj.text}")
                try:
                    error_json = response_obj.json()
                    if "error" in error_json and "message" in error_json["error"]:
                        error_detail = f"Perplexity API Error: {error_json['error']['message']} (Type: {error_json['error'].get('type', 'N/A')})"
                    else:
                        error_detail = response_obj.text
                except json.JSONDecodeError:
                    error_detail = response_obj.text
            else:
                logging.error(f"Request failed before response object was assigned: {e}")
            return {"error": error_detail}
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON response: {response_obj.text if response_obj else 'No response object'}")
            return {"error": "Failed to decode JSON response", "raw_response": response_obj.text if response_obj else 'No response object'}

    def get_quick_response(self, query: str, system_message_override: Optional[str] = None) -> Dict:
        payload = {
            "model": MODEL_QUICK_RESPONSE,
            "messages": [
                {"role": "system", "content": system_message_override or "Be precise and concise."},
                {"role": "user", "content": query}
            ]
        }
        return self._make_request("chat/completions", payload)

    def get_detailed_analysis(self, query: str, context: Optional[str] = None, system_message_override: Optional[str] = None) -> Dict:
        system_content = system_message_override or "Provide a detailed and analytical response."
        if context:
            system_content += f"\n\nAdditional Context for this specific query: {context}"
            
        payload = {
            "model": MODEL_DETAILED_ANALYSIS,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ]
        }
        return self._make_request("chat/completions", payload)

    def format_response(self, response_data: Dict) -> Dict:
        if "error" in response_data:
            logging.error(f"Perplexity API returned an error: {response_data['error']}")
            return response_data
        try:
            choice = response_data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            content = self._remove_think_tags(content)

            # Corrected: Get citations from the top-level "citations" field in the response_data
            raw_citations_list = response_data.get("citations") 
            logging.info(f"Raw citations list from API (top-level): {raw_citations_list}")

            extracted_citation_urls = []
            if raw_citations_list and isinstance(raw_citations_list, list):
                for item in raw_citations_list:
                    if isinstance(item, str): # If it's a list of URL strings
                        extracted_citation_urls.append(item)
                    elif isinstance(item, dict) and "url" in item: # If it's a list of dicts with a "url" key
                         extracted_citation_urls.append(item["url"])
            
            logging.info(f"Extracted citation URLs: {extracted_citation_urls}")
            
            usage = response_data.get("usage", {})
            
            if extracted_citation_urls:
                content += "\n\n**Sources:**\n"
                for i, citation_url in enumerate(extracted_citation_urls, 1):
                    content += f"[{i}] {citation_url}\n" 
            else:
                logging.warning("No citation URLs found or extracted from top-level 'citations' field.")

            return {
                "content": content.strip(),
                "citations": extracted_citation_urls, # Return the extracted URLs
                "usage": usage
            }
        except (IndexError, KeyError, TypeError) as e:
            logging.error(f"Error formatting response: {e}, Response Data: {json.dumps(response_data, indent=2)}")
            return {"error": "Failed to parse API response structure", "raw_response": response_data}

    def _remove_think_tags(self, text: str) -> str:
        import re
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

if __name__ == "__main__":
    API_KEY = os.environ.get("PPLX_API_KEY", "***REMOVED***") 
    if API_KEY == "***REMOVED***" and not os.environ.get("PPLX_API_KEY"):
        logging.warning("Using a hardcoded API key for testing. Set PPLX_API_KEY environment variable for production.")
        
    perplexity_api = PerplexityAPI(API_KEY)
    logging.info("Starting Perplexity API tests with 'pro' model names and corrected citation parsing...")
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) 
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from stocksMicro.perplexity_stock import StockAnalyzer
        stock_analyzer = StockAnalyzer() 
        logging.info(f"\nTesting StockAnalyzer's analyze_stock method for NVDA (using {MODEL_DETAILED_ANALYSIS}):")
        stock_response = stock_analyzer.analyze_stock(
            "Analyze NVIDIA (NVDA) stock. What is the latest news?",
            "Consider recent market trends, company fundamentals, and overall economic conditions for Q1 2025."
        )
        print(json.dumps(stock_response, indent=2))
        logging.info("-" * 30)
    except ImportError as e:
        logging.error(f"Could not import StockAnalyzer for testing: {e}. Ensure PYTHONPATH is correct.")
    except Exception as e:
        logging.error(f"Error during StockAnalyzer test: {e}")

    logging.info(f"Testing direct call with {MODEL_DETAILED_ANALYSIS} for citations:")
    citation_query = "What is the latest news on NVIDIA (NVDA) stock? Include sources."
    # Make a direct call to get the raw response structure again
    raw_response = perplexity_api.get_detailed_analysis(citation_query, context="Consider Q1 2025 earnings.")
    print("--- Raw Response for Direct Call (for verification) ---")
    print(json.dumps(raw_response, indent=2))
    print("--- Formatted Response for Direct Call ---")
    formatted_citation_res = perplexity_api.format_response(raw_response)
    print(json.dumps(formatted_citation_res, indent=2))
    logging.info("-" * 30)

    logging.info(f"Testing direct call with {MODEL_QUICK_RESPONSE}:")
    quick_query = "What is Apple's current stock price?"
    quick_res = perplexity_api.get_quick_response(quick_query)
    formatted_quick_res = perplexity_api.format_response(quick_res)
    print(json.dumps(formatted_quick_res, indent=2))
    logging.info("-" * 30)

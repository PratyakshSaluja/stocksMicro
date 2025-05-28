import json
import os
import sys

# Ensure the parent directory (stocksMicro) and its parent (project root) are in the Python path
# to allow imports like `from perp import PerplexityAPI` and `from stocksMicro.perplexity_stock import StockAnalyzer`
current_dir = os.path.dirname(os.path.abspath(__file__)) # stocksMicro directory
project_root_dir = os.path.dirname(current_dir) # Parent of stocksMicro

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
if current_dir not in sys.path: # If perp.py is in stocksMicro itself
    sys.path.insert(0, current_dir)


# Attempt to import PerplexityAPI. If perp.py is in the project root, this will fail.
# If perp.py is in stocksMicro, this should work if current_dir is in sys.path.
# If perp.py is in project_root, then `from perp import PerplexityAPI` is correct.
# Given the project structure, perp.py is in the project root.
try:
    from perp import PerplexityAPI
except ImportError:
    print("Failed to import PerplexityAPI directly. Ensure perp.py is in the project root or PYTHONPATH.")
    sys.exit(1)

# Perplexity API Key - In a real application, use environment variables
# For this test, ensure your PPLX_API_KEY environment variable is set, or hardcode temporarily for isolated testing.
API_KEY = os.environ.get("PPLX_API_KEY") 
if not API_KEY:
    # Fallback if environment variable is not set, use the one from the codebase for consistency in this test
    API_KEY = "***REMOVED***" 
    print("Warning: PPLX_API_KEY environment variable not found. Using hardcoded key for testing.")


def get_raw_perplexity_response(query: str, context: str = None, model: str = "sonar-reasoning-pro"):
    """
    Makes a direct call to the Perplexity API and returns the raw JSON response.
    """
    perplexity_api = PerplexityAPI(API_KEY)

    system_prompt_template = """You are a financial analyst.
Rules:
1. Provide only the final answer.
2. Do not show intermediate steps or your internal "thinking" process.
3. Structure your response using Markdown.
4. Ensure all factual claims are supported by citations from your search results.
   The Perplexity API will automatically handle numbering and list sources.

Please analyze the following based on the user's query and additional context.
"""
    
    final_system_prompt = system_prompt_template
    if model == "sonar-pro": # Simpler prompt for quick model
        final_system_prompt = "Be precise and concise. Cite your sources."

    # The `get_detailed_analysis` or `get_quick_response` methods in PerplexityAPI
    # already construct the messages payload. We are testing that behavior.
    # The `context` here is the user-provided specific context for the query.
    
    print(f"\n--- Testing Raw Response for Query: '{query}' ---")
    print(f"--- Model: {model} ---")
    print(f"--- System Prompt (template part): ---\n{final_system_prompt[:200]}...") # Print start of system prompt
    print(f"--- User Specific Context: {context} ---")

    if model == "sonar-large-online":
        raw_response_json = perplexity_api.get_detailed_analysis(
            query=query,
            context=context, # This is the user's specific context
            system_message_override=final_system_prompt # This is the general instruction set
        )
    elif model == "sonar-small-online":
        raw_response_json = perplexity_api.get_quick_response(
            query=query,
            system_message_override=final_system_prompt
        )
    else:
        print(f"Unsupported model: {model}")
        return

    print("\n--- Raw API Response (JSON): ---")
    print(json.dumps(raw_response_json, indent=2))

    if "error" not in raw_response_json:
        # Also print the formatted response to see what our current logic extracts
        print("\n--- Formatted Response (using PerplexityAPI.format_response): ---")
        formatted_response = perplexity_api.format_response(raw_response_json)
        print(json.dumps(formatted_response, indent=2))
        
        # Specifically check the 'references' part of the raw response
        try:
            references = raw_response_json.get("choices", [{}])[0].get("message", {}).get("references")
            if references:
                print("\n--- Extracted 'references' field from raw response: ---")
                print(json.dumps(references, indent=2))
            else:
                print("\n--- 'references' field is missing or empty in raw response. ---")
        except (IndexError, KeyError, TypeError) as e:
            print(f"\n--- Could not extract 'references' field: {e} ---")
    else:
        print(f"\n--- API call resulted in an error: {raw_response_json.get('error')} ---")


if __name__ == "__main__":
    # Test Case 1: Detailed analysis query for NVDA (sonar-large-online)
    nvda_query = "What is the latest news on NVIDIA (NVDA) stock performance and outlook?"
    nvda_context = "Focus on Q1 2025 results, Blackwell GPU demand, and overall market sentiment. Include data from reliable financial news sources."
    get_raw_perplexity_response(query=nvda_query, context=nvda_context, model="sonar-large-online")

    print("\n" + "="*50 + "\n")

    # Test Case 2: Quick query (sonar-small-online)
    # quick_query = "What is Apple's current stock price and P/E ratio?"
    # get_raw_perplexity_response(query=quick_query, model="sonar-small-online")
    
    # print("\n" + "="*50 + "\n")

    # Test Case 3: A query that might not have many citable facts easily
    # general_query = "What are general investment strategies for beginners?"
    # get_raw_perplexity_response(query=general_query, context="Keep it simple and actionable.", model="sonar-large-online")

    print("Raw response test script finished.")
    print("Check the console output for raw JSON and extracted references.")
    print("Ensure your PPLX_API_KEY is correctly set as an environment variable or in the script if testing locally.")

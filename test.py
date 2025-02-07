import requests
import json
import concurrent.futures

__all__ = ['fetch_scheme_codes', 'fetch_latest_data', 'main']

BASE_URL = "https://api.mfapi.in/mf"
LATEST_URL = "https://api.mfapi.in/mf/{}/latest"
MAX_THREADS = 10
LIMIT = 200  # Limit the number of scheme codes processed

def fetch_scheme_codes():
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    }
    print(f"Fetching scheme codes from {BASE_URL}")
    response = requests.get(BASE_URL, headers=headers)
    if response.status_code == 200:
        schemes = response.json()
        return [scheme["schemeCode"] for scheme in schemes][-LIMIT:]  # Take the last 200 scheme codes
    else:
        print("Failed to fetch scheme codes")
        return []

def fetch_latest_data(scheme_code):
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    }
    url = LATEST_URL.format(scheme_code)
    print(f"Fetching latest data from {url}")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        filtered_data = {
            "meta": data["meta"],
            "data": [entry for entry in data.get("data", []) if entry["date"].endswith("2025")],
            "status": data["status"]
        }
        return filtered_data if filtered_data["data"] else None
    else:
        print(f"Failed to fetch latest data for scheme {scheme_code}")
        return None

def main():
    scheme_codes = fetch_scheme_codes()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_scheme = {executor.submit(fetch_latest_data, code): code for code in scheme_codes}
        for future in concurrent.futures.as_completed(future_to_scheme):
            data = future.result()
            if data:
                results.append(data)
    
    # Save and return results
    with open("filtered_schemes_2025.json", "w") as f:
        json.dump(results, f, indent=4)
    return results

# Only run if called directly
if __name__ == "__main__":
    main()

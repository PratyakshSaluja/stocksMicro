from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import random

# --- Configuration ---
MODEL_CRYPTO = "kk08/CryptoBERT"
MODEL_FINBERT = "ProsusAI/finbert"

# --- Device Setup ---
# Check if GPU is available and set the device preference
# Note: Loading two models might exceed VRAM. If you encounter CUDA out-of-memory errors,
# you might need to force one or both models to CPU by setting device=-1 manually for one pipeline.
device_preference = 0 if torch.cuda.is_available() else -1
print(f"Preferred device: {'GPU' if device_preference == 0 else 'CPU'} (Device ID: {device_preference})")
print("Note: Actual device usage depends on pipeline implementation and available memory.")

# --- Load Models and Tokenizers ---
print(f"\nLoading {MODEL_CRYPTO}...")
tokenizer_crypto = BertTokenizer.from_pretrained(MODEL_CRYPTO)
model_crypto = BertForSequenceClassification.from_pretrained(MODEL_CRYPTO)
print(f"{MODEL_CRYPTO} loaded.")

print(f"\nLoading {MODEL_FINBERT}...")
tokenizer_finbert = AutoTokenizer.from_pretrained(MODEL_FINBERT)
model_finbert = AutoModelForSequenceClassification.from_pretrained(MODEL_FINBERT)
print(f"{MODEL_FINBERT} loaded.")

# --- Create Pipelines ---
# Pass the device parameter: 0 for GPU, -1 for CPU
print("\nCreating pipelines...")
classifier_crypto = pipeline("sentiment-analysis", model=model_crypto, tokenizer=tokenizer_crypto, device=device_preference)
# FinBERT uses different labels
classifier_finbert = pipeline("sentiment-analysis", model=model_finbert, tokenizer=tokenizer_finbert, device=device_preference)
print("Pipelines created.")

# --- Test Messages ---
test_messages = [
    "Bitcoin (BTC) touches $29k, Ethereum (ETH) Set To Explode, RenQ Finance (RENQ) Crosses Massive Milestone",
    "Crypto market sees slight downturn as regulatory concerns linger.",
    "Dogecoin surges after a tweet from a prominent influencer.",
    "Investing in blockchain technology shows long-term potential despite volatility.",
    "Warning: Many new altcoins are highly speculative and carry significant risk.",
    "The company reported strong earnings, beating analyst expectations.",
    "Stock prices fell sharply following the central bank's interest rate hike.",
    "Analysts remain neutral on the stock's outlook for the next quarter."
]

# --- Random Analysis ---
print("\n--- Starting Random Batch Analysis ---")

models_info = [
    {"name": "CryptoBERT", "classifier": classifier_crypto},
    {"name": "FinBERT", "classifier": classifier_finbert}
]

# Analyze each message with a randomly chosen model
for i, text in enumerate(test_messages):
    chosen_model_info = random.choice(models_info)
    model_name = chosen_model_info["name"]
    classifier = chosen_model_info["classifier"]

    print(f"\nAnalyzing message {i+1} with {model_name}: '{text}'")
    try:
        # Perform sentiment analysis
        result = classifier(text)
        # Print the result
        print(f"Analysis Result ({model_name}):")
        print(result)
    except Exception as e:
        print(f"Error analyzing with {model_name}: {e}")
        # If one model fails (e.g., OOM), you might want to try the other or skip.
        # For simplicity, we just report the error here.

print("\n--- Batch Analysis Complete ---")

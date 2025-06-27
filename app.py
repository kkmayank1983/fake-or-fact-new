import os
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/mrm8488/bert-tiny-finetuned-fake-news-detection"

API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

if not API_KEY:
    # This will now print a much more helpful message in the terminal if the key is missing.
    print("FATAL ERROR: HUGGINGFACE_API_KEY environment variable not found!")
    # We will keep the original error for the app to function, but the terminal will tell us the real story.
    # raise ValueError("HUGGINGFACE_API_KEY environment variable not set!")

headers = {"Authorization": f"Bearer {API_KEY}"}

def query_model(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status() 
    return response.json()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error_message = None

    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        if text:
            print("\n--- FORM SUBMITTED ---")
            print(f"Text to analyze: '{text}'")
            try:
                api_payload = {"inputs": text}
                
                # --- DEBUG: Let's see what we are about to send ---
                print(f"Attempting to send request to: {API_URL}")
                print(f"Headers being sent: {headers}")
                print(f"Payload being sent: {api_payload}")

                scores = query_model(api_payload)
                
                print("--- SUCCESS: Received response from API ---")
                print(f"API Response: {scores}")

                if scores and scores[0]:
                    label_map = {"LABEL_1": "FAKE", "LABEL_0": "REAL"}
                    best_prediction = max(scores[0], key=lambda x: x["score"])
                    label = label_map.get(best_prediction["label"], "Unknown")
                    score = best_prediction["score"] * 100
                    result = f"{label} ({score:.2f}%)"
                else:
                    error_message = "The model did not return a valid prediction. Please try again."

            except requests.exceptions.RequestException as e:
                # --- DEBUG: This is the MOST IMPORTANT part. We print the REAL error ---
                print("\n--- ERROR: requests.exceptions.RequestException was caught! ---")
                print(f"THE SPECIFIC ERROR IS: {e}")
                print("--- END OF ERROR ---")
                
                error_message = "Could not connect to the AI model. The service might be temporarily down or loading. Please try again in a moment."
            except Exception as e:
                print(f"\n--- An unexpected error occurred: {e} ---")
                error_message = "An unexpected error occurred. Please check the input and try again."

    return render_template("index.html", result=result, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
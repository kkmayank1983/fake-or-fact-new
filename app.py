import os
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# --- USING THE RELIABLE AND FAST BERT-TINY MODEL ---
API_URL = "https://api-inference.huggingface.co/models/mrm8488/bert-tiny-finetuned-fake-news-detection"

API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

if not API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY environment variable not set!")

headers = {"Authorization": f"Bearer {API_KEY}"}

def query_model(payload):
    """Sends a request to the Hugging Face Inference API."""
    # We keep the timeout as a best practice, even for a fast model.
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status() 
    return response.json()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error_message = None

    if request.method == "POST":
        text = request.form.get("news_text", "").strip()
        if text:
            try:
                api_payload = {"inputs": text}
                scores = query_model(api_payload)
                
                if scores and scores[0]:
                    # This model uses LABEL_1 for FAKE and LABEL_0 for REAL.
                    label_map = {"LABEL_1": "FAKE", "LABEL_0": "REAL"}
                    
                    best_prediction = max(scores[0], key=lambda x: x["score"])
                    
                    label = label_map.get(best_prediction["label"], best_prediction["label"].upper())
                    score = best_prediction["score"] * 100
                    
                    result = f"{label} ({score:.2f}% confidence)"

                else:
                    error_message = "The model did not return a valid prediction. Please try again."

            except requests.exceptions.Timeout:
                error_message = "The AI model is taking too long to respond. Please try again in a moment."
            except requests.exceptions.RequestException:
                error_message = "Could not connect to the AI model. Check your network or the service status."
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                error_message = "An unexpected error occurred. Please check the input and try again."

    return render_template("index.html", result=result, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
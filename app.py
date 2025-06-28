import os
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# --- UPGRADING TO A SUPERIOR, FACT-CHECKING MODEL ---
API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Get your Hugging Face API key from environment variables
API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

if not API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY environment variable not set!")

headers = {"Authorization": f"Bearer {API_KEY}"}

def query_model(payload):
    """Sends a request to the Hugging Face Inference API."""
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
            try:
                # This model expects a "candidate_labels" parameter, but we can often
                # get away with sending it as a standard text-classification request.
                api_payload = {"inputs": text}
                scores = query_model(api_payload)
                
                if scores and scores[0]:
                    # This model uses different labels: entailment, neutral, contradiction
                    # We will map them to our desired output.
                    label_map = {
                        "contradiction": "Likely FAKE",
                        "entailment": "Likely REAL",
                        "neutral": "UNCERTAIN"
                    }
                    
                    best_prediction = max(scores[0], key=lambda x: x["score"])
                    
                    # Get our mapped label, or use the original if not in our map
                    label = label_map.get(best_prediction["label"], best_prediction["label"].upper())
                    score = best_prediction["score"] * 100
                    
                    # Create a more descriptive result
                    if label == "UNCERTAIN":
                        result = f"{label} (Not enough information to make a determination)"
                    else:
                        result = f"{label} ({score:.2f}% confidence)"

                else:
                    error_message = "The model did not return a valid prediction. Please try again."

            except requests.exceptions.RequestException:
                error_message = "Could not connect to the AI model. The service might be temporarily down or loading. Please try again in a moment."
            except Exception as e:
                # For debugging, let's see what other errors might occur
                print(f"An unexpected error occurred: {e}")
                error_message = "An unexpected error occurred. Please check the input and try again."

    return render_template("index.html", result=result, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
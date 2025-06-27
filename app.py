import os
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# The API URL for the model you are using
API_URL = "https://api-inference.huggingface.co/models/hamzab/roberta-fake-news-classification"

# Get your Hugging Face API key from environment variables
# We will set this in Render's dashboard later
API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

# Check if the API key is available
if not API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY environment variable not set!")

headers = {"Authorization": f"Bearer {API_KEY}"}

def query_model(payload):
    """Sends a request to the Hugging Face Inference API."""
    response = requests.post(API_URL, headers=headers, json=payload)
    # Raise an exception if the request was not successful
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
                # The API expects data in a specific format
                api_payload = {"inputs": text}
                
                # Query the API
                scores = query_model(api_payload)
                
                # The API response is a list containing a list of dictionaries, e.g.,
                # [[{'label': 'FAKE', 'score': 0.99...}, {'label': 'REAL', 'score': 0.00...}]]
                if scores and scores[0]:
                    # The model you chose seems to have labels 'LABEL_0' and 'LABEL_1'.
                    # We need to map them correctly. The model card says LABEL_1 = FAKE, LABEL_0 = REAL.
                    label_map = {"LABEL_1": "FAKE", "LABEL_0": "REAL"}
                    
                    # Find the prediction with the highest score
                    best_prediction = max(scores[0], key=lambda x: x["score"])
                    
                    label = label_map.get(best_prediction["label"], "Unknown")
                    score = best_prediction["score"] * 100
                    result = f"{label} ({score:.2f}%)"
                else:
                    error_message = "The model did not return a valid prediction. Please try again."

            except requests.exceptions.RequestException as e:
                # Handle network errors, API errors, etc.
                print(f"API request failed: {e}")
                error_message = "Could not connect to the AI model. The service might be temporarily down. Please try again later."
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                error_message = "An unexpected error occurred. Please check the input and try again."

    # Pass the result OR the error message to the template
    return render_template("index.html", result=result, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
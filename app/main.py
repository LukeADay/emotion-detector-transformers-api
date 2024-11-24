from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained Hugging Face model
emotion_model = pipeline("text-classification", model="./app/model")

@app.route("/")
def home():
    return jsonify({"message": "Emotion Detection API is live!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    result = emotion_model(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

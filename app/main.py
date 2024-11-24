from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./app/emotion_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Emotion labels from the dataset
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

@app.route("/")
def home():
    """Health check endpoint."""
    return jsonify({"message": "Emotion Detection API is live!"})

@app.route("/predict", methods=["POST"])
def predict():
    """Predict emotion from input text."""
    data = request.json

    # Validate input
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]

    # Tokenize the input and make a prediction
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Return the predicted emotion
    return jsonify({"emotion": labels[predicted_class]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

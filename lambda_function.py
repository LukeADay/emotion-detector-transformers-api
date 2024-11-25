import os
import json
import boto3
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# S3 Bucket and Key (update to match your bucket name and folder path)
S3_BUCKET = "emotion-detection-model"
MODEL_KEY_PREFIX = "emotion_model/"
LOCAL_MODEL_DIR = "/tmp/emotion_model"

# Download model files from S3
def download_model():
    if not os.path.exists(LOCAL_MODEL_DIR):
        os.makedirs(LOCAL_MODEL_DIR)
    
    s3 = boto3.client("s3")
    files = ["model.safetensors", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt"]
    for file in files:
        s3.download_file(S3_BUCKET, f"{MODEL_KEY_PREFIX}{file}", os.path.join(LOCAL_MODEL_DIR, file))

# Load the model and tokenizer
download_model()
model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)

def lambda_handler(event, context):
    # Parse the input text
    body = json.loads(event.get("body", "{}"))
    input_text = body.get("text", "")

    if not input_text:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No text provided"}),
        }

    # Perform inference
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Map label IDs to emotion names
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    emotion = labels[predicted_class]

    # Return the prediction
    return {
        "statusCode": 200,
        "body": json.dumps({"emotion": emotion}),
        "headers": {"Content-Type": "application/json"},
    }
